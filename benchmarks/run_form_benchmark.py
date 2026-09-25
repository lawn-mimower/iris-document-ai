#!/usr/bin/env python3
"""
Runs the form-filling benchmark (benchmarks/data/forms) and writes benchmarks/results/forms_<run>.json.

Methods
  agent             the repo's pipeline: phase1/phase1.py -> prototype1/gestalt_processor.py on the empty
                    form and the golden sample, then prototype1/agent.py (its own prompt and JSON mode)
                    with --model; the LLM call goes through the benchmark's cache
  agent_vector      the same agent and prompt, but both forms' elements come from the PDF's text spans and
                    vector drawings (geometric_filler.vector_elements) + gestalt_processor instead of phase1
  geometric_phase1  nearest-label geometry (geometric_filler.py) on the same phase1 + gestalt elements
                    of the empty form; no LLM, golden sample not used
  geometric_vector  nearest-label geometry on elements read from the PDF's text spans and vector
                    drawings (geometric_filler.vector_elements); no LLM, golden sample not used

A value counts as placed correctly when a plan entry with that text has its centre inside the target
box; a checkbox group when the chosen option's box, and no other option's, holds a tick.

Examples
  python benchmarks/run_form_benchmark.py --run offline --methods geometric_phase1,geometric_vector
  python benchmarks/run_form_benchmark.py --run gemini-2.5-flash --methods agent,geometric_phase1,geometric_vector \\
      --model gemini-2.5-flash --min-interval 65 --daily-cap 60 --resume
  python benchmarks/run_form_benchmark.py --run gemini-2.5-flash --methods agent --dry-run   # prompt sizes only

The agent's prompt is large (the full element lists of both forms), so free-tier runs should pace
calls (--min-interval) and may need --forms to spread the forms over several days.
"""

import argparse
import importlib.util
import json
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

import geometric_filler  # noqa: E402
from llm_client import LLMClient, LLMStop  # noqa: E402
from scoring import iou, score_form  # noqa: E402

ALL_METHODS = ["agent", "agent_vector", "geometric_phase1", "geometric_vector"]
DESCRIPTIONS = {
    "agent": "repo pipeline: phase1 + gestalt_processor + prototype1/agent.py (LLM, golden sample in the prompt)",
    "agent_vector": "prototype1/agent.py unchanged, but fed gestalt-processed vector elements instead of phase1 output",
    "geometric_phase1": "nearest-label geometry on the phase1 + gestalt elements, no LLM, no golden sample",
    "geometric_vector": "nearest-label geometry on PDF text spans + vector rectangles, no LLM, no golden sample",
}


def _load(relpath, name):
    spec = importlib.util.spec_from_file_location(name, REPO / relpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class RepoPipeline:
    def __init__(self):
        self.phase1 = _load("phase1/phase1.py", "bench_phase1")
        self.gestalt = _load("prototype1/gestalt_processor.py", "bench_gestalt")
        self.agent = _load("prototype1/agent.py", "bench_agent")

    def structure(self, pdf: Path, out_dir: Path, name: str, perception: str = "phase1") -> Path:
        """phase1 (or vector) elements -> gestalt, as in the README; returns the processed JSON path."""
        if perception == "phase1":
            elements = self.phase1.process_pdf(str(pdf))
        else:
            elements = geometric_filler.vector_elements(pdf)
            for e in elements:
                b = e["bbox"]
                e.update({"center": [round((b[0] + b[2]) / 2, 2), round((b[1] + b[3]) / 2, 2)], "page_num": 0})
            name = f"{name}_vector"
        for i, e in enumerate(elements):
            e["id"] = f"element_{i}"
        raw = out_dir / f"{name}_raw.json"
        raw.write_text(json.dumps(elements))
        processed = out_dir / f"{name}.json"
        self.gestalt.process_elements(str(raw), str(processed))
        return processed

    def run_agent(self, empty_json, golden_json, new_data, out_plan, llm_call):
        """Run prototype1/agent.py's main() with its LLM call replaced by llm_call(prompt) -> text."""
        self.agent.call_llm_api = llm_call
        argv = sys.argv
        sys.argv = ["agent.py", "--empty-form-structure", str(empty_json), "--golden-sample-structure",
                    str(golden_json), "--new-data", str(new_data), "--output-plan", str(out_plan)]
        try:
            self.agent.main()
        finally:
            sys.argv = argv
        return json.loads(Path(out_plan).read_text()) if Path(out_plan).exists() else None


def _ms(seconds):
    """Durations are stored as whole milliseconds."""
    return None if seconds is None else int(round(seconds * 1000))


def _round_entry(e):
    if isinstance(e, dict) and isinstance(e.get("bbox"), list):
        return {**e, "bbox": [round(v, 1) if isinstance(v, float) else v for v in e["bbox"]]}
    return e


def perception_recall(elements, gold_fields) -> dict:
    """How many target boxes the perception step reports (IoU > 0.5 with any box or checkbox)."""
    boxes = [e["bbox"] for e in elements if e.get("type") in ("INPUT_FIELD", "CHECKBOX")]
    found = total = 0
    for f in gold_fields:
        targets = [o["bbox"] for o in f["options"]] if f["kind"] == "checkbox" else [f["bbox"]]
        for t in targets:
            total += 1
            found += any(iou(b, t) > 0.5 for b in boxes)
    return {"targets": total, "found": found, "n_elements": len(elements)}


def main():
    ap = argparse.ArgumentParser(description="Form-filling benchmark", epilog=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="results go to benchmarks/results/forms_<run>.json")
    ap.add_argument("--data", default=str(HERE / "data" / "forms"))
    ap.add_argument("--methods", default=",".join(ALL_METHODS))
    ap.add_argument("--forms", default="", help="comma-separated form ids (default: all)")
    ap.add_argument("--provider", choices=["gemini", "ollama"], default="gemini")
    ap.add_argument("--model", default=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
    ap.add_argument("--think", default=None, help="Ollama think level for reasoning models (low, medium, high)")
    ap.add_argument("--min-interval", type=float, default=0.0)
    ap.add_argument("--timeout", type=float, default=600.0, help="client deadline per LLM call in seconds")
    ap.add_argument("--max-live-calls", type=int, default=None)
    ap.add_argument("--daily-cap", type=int, default=None)
    ap.add_argument("--cache-dir", default=str(HERE / ".cache"))
    ap.add_argument("--results-dir", default=str(HERE / "results"))
    ap.add_argument("--work-dir", default=str(HERE / "work" / "forms"))
    ap.add_argument("--resume", action="store_true", help="keep finished forms/methods from the last run")
    ap.add_argument("--redo", action="store_true",
                    help="with --resume: recompute the selected --methods/--forms (cached LLM responses are reused)")
    ap.add_argument("--offline", action="store_true", help="cached LLM responses only")
    ap.add_argument("--dry-run", action="store_true", help="build the agent prompts and report their size, no LLM")
    args = ap.parse_args()

    data = Path(args.data)
    truth = json.loads((data / "ground_truth.json").read_text())
    methods = [m for m in args.methods.split(",") if m]
    forms = [f for f in truth["forms"] if not args.forms or f["id"] in args.forms.split(",")]
    out_path = Path(args.results_dir) / f"forms_{args.run}.json"
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)

    prev = json.loads(out_path.read_text()) if args.resume and out_path.exists() else None
    results = {}
    if prev:
        redo = {f"{f['id']}|{m}" for f in forms for m in methods} if args.redo else set()
        for key, r in prev["results"].items():
            if r["status"] == "ok" and key not in redo:
                results[key] = r
    perception = prev.get("perception", {}) if prev else {}

    pipe = RepoPipeline()
    llm = None
    if any(m.startswith("agent") for m in methods) and not args.dry_run:
        # temperature None: keep the model's default sampling, as prototype1/agent.py does
        llm = LLMClient(args.provider, args.model, cache_dir=args.cache_dir, min_interval_s=args.min_interval, think=args.think,
                        max_live_calls=args.max_live_calls, daily_cap=args.daily_cap, temperature=None,
                        offline=args.offline, timeout_s=args.timeout)
    stop_reason = None

    for form in forms:
        fid, d = form["id"], data / form["id"]
        new_data = json.loads((d / "new_data.json").read_text())
        fdir = work / fid
        fdir.mkdir(exist_ok=True)
        todo = [m for m in methods if f"{fid}|{m}" not in results]
        if not todo and not args.dry_run:
            continue
        t = time.time()
        empty_json = pipe.structure(d / "empty.pdf", fdir, "empty")
        phase1_s = time.time() - t
        empty_elements = json.loads(empty_json.read_text())
        t = time.time()
        vector = geometric_filler.vector_elements(d / "empty.pdf")
        vector_s = time.time() - t
        perception[fid] = {"phase1_gestalt": {**perception_recall(empty_elements, form["fields"]),
                                              "ms": _ms(phase1_s)},
                           "vector": {**perception_recall(vector, form["fields"]), "ms": _ms(vector_s)}}

        for m in todo:
            rec = {"form": fid, "method": m, "status": "ok", "llm": None}
            t = time.time()
            if m == "geometric_phase1":
                plan = geometric_filler.fill(empty_elements, new_data)
            elif m == "geometric_vector":
                plan = geometric_filler.fill(vector, new_data)
            else:
                perc = "vector" if m == "agent_vector" else "phase1"
                golden_json = pipe.structure(d / "golden.pdf", fdir, "golden", perc)
                agent_empty = empty_json if perc == "phase1" else pipe.structure(d / "empty.pdf", fdir, "empty", perc)
                if args.dry_run:
                    seen = {}

                    def measure(prompt):
                        seen["chars"] = len(prompt)
                        return "[]"
                    pipe.run_agent(agent_empty, golden_json, d / "new_data.json", fdir / "plan_dry.json", measure)
                    print(f"{fid}: {m} prompt {seen['chars']:,} characters (~{seen['chars'] // 4:,} tokens)")
                    continue
                if stop_reason:
                    results[f"{fid}|{m}"] = {**rec, "status": "pending"}
                    continue
                call = {}

                def llm_call(prompt):
                    try:
                        r = llm.complete(prompt, json_mode=True)
                    except LLMStop as e:
                        call["quota"] = str(e)
                        return None
                    call["rec"] = r
                    call["prompt_chars"] = len(prompt)
                    return r["text"]
                plan_path = fdir / f"{m}_plan.json"
                if plan_path.exists():
                    plan_path.unlink()
                plan = pipe.run_agent(agent_empty, golden_json, d / "new_data.json", plan_path, llm_call)
                if "quota" in call:
                    stop_reason = call["quota"]
                    results[f"{fid}|{m}"] = {**rec, "status": "pending"}
                    continue
                r = call["rec"]
                rec["llm"] = {"model": f"{args.provider}:{args.model}", "prompt_chars": call["prompt_chars"],
                              "prompt_tokens": r.get("prompt_tokens"), "completion_tokens": r.get("completion_tokens"),
                              "thinking_tokens": r.get("thinking_tokens", 0), "latency_ms": _ms(r.get("latency_s")),
                              "cached": r.get("cached")}
                if plan is None:
                    rec["status_detail"] = "agent reply was not valid JSON"
                    plan = []
            rec["ms"] = _ms(time.time() - t)
            if not isinstance(plan, list):
                rec["status_detail"] = "plan is not a JSON array"
                plan = []
            rec["score"] = score_form(plan, form["fields"])
            rec["plan"] = [_round_entry(e) for e in plan]   # coordinates to 0.1 pt
            results[f"{fid}|{m}"] = rec
            s = rec["score"]
            print(f"{fid:18s} {m:17s} {s['n_correct']}/{s['n_fields']}")

    if args.dry_run:
        return
    summary = {}
    methods = [m for m in ALL_METHODS if m in methods or any(r["method"] == m for r in results.values())]
    for m in methods:
        rs = [r for r in results.values() if r["method"] == m and r["status"] == "ok"]
        n = sum(r["score"]["n_fields"] for r in rs)
        c = sum(r["score"]["n_correct"] for r in rs)
        by_kind = {}
        for r in rs:
            for f in r["score"]["fields"]:
                k = by_kind.setdefault(f["kind"], {"n": 0, "correct": 0})
                k["n"] += 1
                k["correct"] += f["correct"]
        llm_recs = [r["llm"] for r in rs if r.get("llm")]
        summary[m] = {"forms_scored": len(rs), "fields": n, "correct": c,
                      "by_kind": by_kind,
                      "per_form": {r["form"]: f"{r['score']['n_correct']}/{r['score']['n_fields']}" for r in rs},
                      "llm_calls": len(llm_recs),
                      "prompt_tokens": sum(x["prompt_tokens"] or 0 for x in llm_recs),
                      "completion_tokens": sum(x["completion_tokens"] or 0 for x in llm_recs),
                      "thinking_tokens": sum(x.get("thinking_tokens") or 0 for x in llm_recs),
                      "llm_time_ms": sum(x["latency_ms"] or 0 for x in llm_recs)}
    status = "partial" if any(r["status"] == "pending" for r in results.values()) else "complete"
    out = {"benchmark": "form_filling", "run": args.run, "status": status,
           "updated": datetime.now().isoformat(timespec="seconds"),
           "conditions": {"agent_llm": f"{args.provider}:{args.model}" if any(m.startswith("agent") for m in methods) else None,
                          "agent_temperature": "model default (as prototype1/agent.py)",
                          "client_timeout_s": args.timeout,
                          "n_forms": len({r["form"] for r in results.values()}), "python": platform.python_version(),
                          "platform": platform.platform(terse=True), "stop_reason": stop_reason},
           "methods": {m: DESCRIPTIONS[m] for m in methods},
           "summary": summary, "perception": perception,
           "results": dict(sorted(results.items()))}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(f"\nWrote {out_path} ({status})")
    for m, s in summary.items():
        print(f"  {m:17s} {s['correct']}/{s['fields']} fields on {s['forms_scored']} forms")
    if stop_reason:
        print(f"\nStopped early: {stop_reason}\nResume with the same command plus --resume.")


if __name__ == "__main__":
    main()
