#!/usr/bin/env python3
"""
Runs the DAM field-extraction benchmark (benchmarks/data/dam) and writes benchmarks/results/dam_<run>.json.

Methods (every LLM-based method in one run uses the same --provider / --model):

  dam           the repo's pipeline: DAM/oneshot_ingestion.py (Docling -> HierarchicalChunker ->
                BAAI/bge-large-en-v1.5 -> Milvus Lite), DAM/experiment_harness.retrieve_from_milvus
                top-k with the form label as query, one LLM call per field; the answer cites a passage,
                whose file and page are the returned source
  bm25          baseline A, basic RAG: pypdf / python-docx / openpyxl text, 1,000-character chunks with
                200 overlap, BM25 top-k, the same prompt, one LLM call per field
  dam_batched   the dam retrieval, but all fields of a company answered in one LLM call
  bm25_batched  the bm25 retrieval, all fields in one LLM call
  single_call   baseline B: all of a company's extracted text in one prompt, all fields as JSON
  rules         baseline C: regular expressions and keyword rules, no model

Examples
  # fully local, Ollama llama3.2 for every LLM-based method
  OLLAMA_HOST=http://127.0.0.1:11434 python benchmarks/run_dam_benchmark.py --run local-llama3.2 \\
      --provider ollama --model llama3.2
  # Gemini free tier: one call per company per method
  python benchmarks/run_dam_benchmark.py --run gemini-2.5-flash-lite --provider gemini \\
      --model gemini-2.5-flash-lite --methods rules,single_call,dam_batched,bm25_batched \\
      --min-interval 10 --daily-cap 60 --resume

Responses are cached in benchmarks/.cache/ (git-ignored). With --resume, companies and methods that
finished in an earlier run of the same --run name are kept and only the rest are computed; a run
interrupted by a quota error saves what it has and prints the command to finish it.
"""

import argparse
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

import dam_fields  # noqa: E402
import plain_text  # noqa: E402
import rules_baseline  # noqa: E402
from llm_client import LLMClient, LLMStop, parse_json  # noqa: E402
from scoring import score_field, source_correct  # noqa: E402

ALL_METHODS = ["rules", "single_call", "bm25", "dam", "bm25_batched", "dam_batched"]
LLM_METHODS = {"single_call", "bm25", "dam", "bm25_batched", "dam_batched"}
DESCRIPTIONS = {
    "dam": "repo pipeline: Docling + HierarchicalChunker + BGE-large + Milvus Lite, top-k, one LLM call per field",
    "bm25": "baseline A: plain text + 1,000/200-char chunks + BM25 top-k, one LLM call per field",
    "dam_batched": "repo retrieval (as dam), all fields of a company in one LLM call",
    "bm25_batched": "baseline A retrieval (as bm25), all fields of a company in one LLM call",
    "single_call": "baseline B: all extracted text of a company in one prompt, all fields as JSON",
    "rules": "baseline C: regex / keyword rules, no LLM",
}

ANSWER_RULES = """If the passages do not contain the answer, use "UNANSWERABLE" as the answer.
Do not guess or infer beyond what is explicitly stated."""


# --------------------------------------------------------------------------- prompts

def _passage_block(passages) -> str:
    return "\n\n".join(
        f"[{i}] (source: {p['file']}, page {p['page'] if p.get('page') else '-'})\n{p['text'].strip()}"
        for i, p in enumerate(passages, start=1))


def field_prompt(field, passages) -> str:
    return f"""You are a financial document analyst. Answer the question using ONLY the numbered passages below.
{ANSWER_RULES}
Reply with JSON only: {{"answer": "<the answer, no explanation>", "passage": <number of the passage that contains the answer, or null>}}

Passages:
{_passage_block(passages)}

Question: {dam_fields.question(field)}"""


def batched_prompt(fields, passages, per_field) -> str:
    lines = []
    for f in fields:
        refs = ", ".join(str(n) for n in per_field[f["id"]])
        lines.append(f"- {f['id']}: {f['label']} (answer format: {dam_fields.FORMAT_HINTS[f['type']]}; "
                     f"retrieved passages: {refs})")
    return f"""You are a financial document analyst. Fill in every field below using ONLY the numbered passages.
{ANSWER_RULES}
Reply with JSON only, one entry per field id:
{{"<field id>": {{"answer": "<the answer, no explanation>", "passage": <number of the passage that contains it, or null>}}, ...}}

Passages:
{_passage_block(passages)}

Fields:
""" + "\n".join(lines)


def single_call_prompt(fields, units) -> str:
    docs = []
    for u in units:
        where = f"page {u['page']}" if u.get("page") else (f"sheet {u['sheet']}" if u.get("sheet") else "")
        docs.append(f"===== {u['file']}{', ' + where if where else ''} =====\n{u['text'].strip()}")
    lines = [f"- {f['id']}: {f['label']} (answer format: {dam_fields.FORMAT_HINTS[f['type']]})" for f in fields]
    return f"""You are a financial document analyst. Below is the text of a company's documents. Fill in every field using ONLY these documents.
If the documents do not contain a field's answer, use "UNANSWERABLE" for it.
Do not guess or infer beyond what is explicitly stated.
Reply with JSON only: {{"<field id>": "<answer>", ...}}

Documents:
""" + "\n\n".join(docs) + "\n\nFields:\n" + "\n".join(lines)


# --------------------------------------------------------------------------- retrieval back ends

class DamIndex:
    """The repo's ingestion and retrieval code, one Milvus Lite database per company."""

    def __init__(self, work_dir: Path):
        os.environ.setdefault("DAM_WORK_DIR", str(work_dir))
        sys.path.insert(0, str(REPO / "DAM"))
        import experiment_harness
        import oneshot_ingestion

        import logging

        oneshot_ingestion.logger.setLevel(logging.WARNING)
        self.harness, self.oneshot = experiment_harness, oneshot_ingestion
        self.work_dir = work_dir
        self.model = None
        self.model_load_ms = None

    def ingest(self, company_dir: Path, name: str) -> dict:
        pipe = self.oneshot.OneShotIngestion(input_dir=str(company_dir), db_name=str(self.work_dir / f"{name}.db"),
                                             collection_name="oneshot")
        if self.model is None:
            t = time.time()
            pipe.load_embedding_model()
            self.model_load_ms = _ms(time.time() - t)
            self.model = pipe.embedding_model
            self.harness._embedding_model = self.model   # share one model with the retrieval code
        pipe.embedding_model, pipe.embedding_dim = self.model, 1024
        t0 = time.time()
        files = pipe.discover_files()
        pipe.initialize_converter_and_chunker()
        chunks = pipe.process_documents(files)
        t1 = time.time()
        emb = pipe.embed_chunks(chunks)
        t2 = time.time()
        pipe.setup_milvus()
        pipe.insert_data(chunks, emb)
        pipe.milvus_client.close()
        t3 = time.time()
        return {"convert_chunk_ms": _ms(t1 - t0), "embed_ms": _ms(t2 - t1), "index_ms": _ms(t3 - t2),
                "total_ms": _ms(t3 - t0), "n_chunks": len(chunks)}

    def retrieve(self, name: str, query: str, k: int) -> list:
        hits = self.harness.retrieve_from_milvus(query, str(self.work_dir / f"{name}.db"), "oneshot", top_k=k)
        return [{"file": h["source_file"], "page": h.get("page_number"), "text": h["text"]} for h in hits]


class Bm25Index:
    def __init__(self, units, size=1000, overlap=200):
        self.chunks = plain_text.chunk_units(units, size, overlap)
        self.bm25 = plain_text.BM25([c["text"] for c in self.chunks])

    def retrieve(self, query: str, k: int) -> list:
        return [self.chunks[i] for i in self.bm25.top_k(query, k)]


# --------------------------------------------------------------------------- helpers

def _ms(seconds):
    """Durations are stored as whole milliseconds."""
    return None if seconds is None else int(round(seconds * 1000))


def _norm_ws(s: str) -> str:
    return " ".join(str(s).split())


def evidence_retrieved(passages, sources) -> bool:
    texts = [_norm_ws(p["text"]) for p in passages]
    return any(_norm_ws(ev) in t for s in sources for ev in s.get("evidence", []) for t in texts)


def _as_answer(v):
    if isinstance(v, dict):
        v = v.get("answer")
    if v is None:
        return None
    if isinstance(v, bool):
        return "Yes" if v else "No"
    return str(v)


def _as_passage(v, n):
    if isinstance(v, dict):
        v = v.get("passage")
    if isinstance(v, list):
        v = v[0] if v else None
    try:
        i = int(str(v).strip().strip("[]"))
    except (TypeError, ValueError):
        return None
    return i if 1 <= i <= n else None


def _source_of(passages, idx, fallback_first=True):
    if idx:
        p = passages[idx - 1]
        return {"file": p["file"], "page": p.get("page"), "via": "citation"}
    if fallback_first and passages:
        return {"file": passages[0]["file"], "page": passages[0].get("page"), "via": "top1"}
    return None


def _call_record(method, company, field, rec):
    return {"method": method, "company": company, "field": field,
            "prompt_tokens": rec.get("prompt_tokens"), "completion_tokens": rec.get("completion_tokens"),
            "thinking_tokens": rec.get("thinking_tokens", 0), "latency_ms": _ms(rec.get("latency_s")),
            "cached": rec.get("cached"), "context_limit_hit": rec.get("context_limit_hit", False)}


def make_record(method, company, fdef, gold_entry, answer, source, status, evidence=None, raw=None):
    gold = gold_entry["gold"]
    sc = score_field(fdef["type"], gold, answer) if status == "ok" else {"correct": False, "strict": False}
    return {"method": method, "company": company, "field": fdef["id"], "type": fdef["type"], "gold": gold,
            "answer": answer, "correct": sc["correct"], "strict": sc["strict"], "status": status,
            "source": source,
            "source_correct": (source_correct(source, gold_entry["sources"]) if source and gold is not None else None),
            "evidence_in_context": evidence, "raw": raw}


# --------------------------------------------------------------------------- summary

def companies_of(recs) -> set:
    return {r["company"] for r in recs}


def summarise(records, calls, ingestion, methods) -> dict:
    out = {}
    for m in methods:
        recs = [r for r in records if r["method"] == m]
        done = [r for r in recs if r["status"] != "pending"]
        answerable = [r for r in done if r["gold"] is not None]
        with_src = [r for r in answerable if r["source_correct"] is not None]
        with_ev = [r for r in answerable if r["evidence_in_context"] is not None]
        cs = [c for c in calls if c["method"] == m]
        per_field = {}
        for f in dam_fields.FIELDS:
            fr = [r for r in done if r["field"] == f["id"]]
            per_field[f["id"]] = {"n": len(fr), "correct": sum(r["correct"] for r in fr)}
        out[m] = {
            "n_records": len(recs), "n_scored": len(done), "n_pending": len(recs) - len(done),
            "correct": sum(r["correct"] for r in done), "strict": sum(r["strict"] for r in done),
            "answerable_n": len(answerable), "answerable_correct": sum(r["correct"] for r in answerable),
            "unanswerable_n": len(done) - len(answerable),
            "unanswerable_correct": sum(r["correct"] for r in done if r["gold"] is None),
            "parse_errors": sum(r["status"] == "parse_error" for r in recs),
            "source_n": len(with_src), "source_correct": sum(r["source_correct"] for r in with_src),
            "evidence_n": len(with_ev), "evidence_in_context": sum(r["evidence_in_context"] for r in with_ev),
            "llm_calls": len(cs), "live_llm_calls": sum(not c["cached"] for c in cs),
            "prompt_tokens": sum(c["prompt_tokens"] or 0 for c in cs),
            "completion_tokens": sum(c["completion_tokens"] or 0 for c in cs),
            "thinking_tokens": sum(c.get("thinking_tokens") or 0 for c in cs),
            "llm_time_ms": sum(c["latency_ms"] or 0 for c in cs),
            "context_limit_hits": sum(bool(c.get("context_limit_hit")) for c in cs),
            "ingestion_ms": sum(v["total_ms"] for c, v in ingestion.get(
                "dam" if m.startswith("dam") else "plain", {}).items() if c in companies_of(recs)),
            "per_field": per_field,
        }
    return out


# --------------------------------------------------------------------------- main loop

def main():
    ap = argparse.ArgumentParser(description="DAM field-extraction benchmark",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--run", required=True, help="run name; results go to benchmarks/results/dam_<run>.json")
    ap.add_argument("--data", default=str(HERE / "data" / "dam"))
    ap.add_argument("--methods", default=",".join(ALL_METHODS))
    ap.add_argument("--companies", default="", help="comma-separated company ids (default: all)")
    ap.add_argument("--provider", choices=["ollama", "gemini"], default="ollama")
    ap.add_argument("--model", default="llama3.2")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--num-ctx", type=int, default=16384, help="Ollama context window")
    ap.add_argument("--think", default=None, help="Ollama think level for reasoning models (low, medium, high)")
    ap.add_argument("--min-interval", type=float, default=0.0, help="seconds between live LLM calls")
    ap.add_argument("--max-live-calls", type=int, default=None, help="stop after this many live calls")
    ap.add_argument("--daily-cap", type=int, default=None, help="refuse live calls once today's ledger has this many")
    ap.add_argument("--cache-dir", default=str(HERE / ".cache"))
    ap.add_argument("--results-dir", default=str(HERE / "results"))
    ap.add_argument("--work-dir", default=str(HERE / "work"), help="Milvus databases (git-ignored)")
    ap.add_argument("--resume", action="store_true", help="keep finished method/company results from the last run")
    ap.add_argument("--redo", action="store_true",
                    help="with --resume: recompute the selected --methods/--companies (e.g. after regenerating data); "
                         "cached LLM responses are reused")
    ap.add_argument("--offline", action="store_true", help="use cached LLM responses only")
    args = ap.parse_args()

    data = Path(args.data)
    truth = json.loads((data / "ground_truth.json").read_text())
    methods = [m for m in args.methods.split(",") if m]
    for m in methods:
        if m not in ALL_METHODS:
            ap.error(f"unknown method {m}")
    companies = [c for c in truth["companies"] if not args.companies or c["id"] in args.companies.split(",")]
    fields = truth["fields"]
    out_path = Path(args.results_dir) / f"dam_{args.run}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)

    prev = json.loads(out_path.read_text()) if args.resume and out_path.exists() else None
    done_keys = set()
    records, calls, ingestion = [], [], {"dam": {}, "plain": {}}
    if prev:
        complete = {(m, c) for m, c in {(r["method"], r["company"]) for r in prev["records"]}
                    if all(r["status"] != "pending" for r in prev["records"] if r["method"] == m and r["company"] == c)}
        if args.redo:   # recompute the selected methods and companies, keep everything else
            complete -= {(m, c["id"]) for m in methods for c in companies}
        done_keys = complete
        records = [r for r in prev["records"] if (r["method"], r["company"]) in done_keys]
        calls = [c for c in prev["calls"] if (c["method"], c["company"]) in done_keys]
        ingestion = prev.get("ingestion", ingestion)

    llm = None
    if any(m in LLM_METHODS for m in methods):
        llm = LLMClient(args.provider, args.model, cache_dir=args.cache_dir, min_interval_s=args.min_interval, think=args.think,
                        max_live_calls=args.max_live_calls, daily_cap=args.daily_cap, num_ctx=args.num_ctx,
                        offline=args.offline)
    dam_index = None
    stop_reason = None
    t_run = time.time()

    def save(final: bool) -> dict:
        """Write the results file; called after every company so a crash keeps finished work."""
        order = {m: i for i, m in enumerate(ALL_METHODS)}
        comp_order = {c["id"]: i for i, c in enumerate(truth["companies"])}
        records.sort(key=lambda r: (order[r["method"]], comp_order[r["company"]],
                                    [f["id"] for f in fields].index(r["field"])))
        unfinished = any(r["status"] == "pending" for r in records) or not final
        all_methods = [m for m in ALL_METHODS if m in methods or any(r["method"] == m for r in records)]
        result = {
            "benchmark": "dam_field_extraction", "run": args.run, "status": "partial" if unfinished else "complete",
            "updated": datetime.now().isoformat(timespec="seconds"),
            "conditions": {
                "llm": f"{args.provider}:{args.model}" if llm else None, "temperature": 0.0, "top_k": args.top_k,
                "ollama_num_ctx": args.num_ctx if args.provider == "ollama" else None,
                "embedding_model": "BAAI/bge-large-en-v1.5", "bm25_chunking": "1000 characters, 200 overlap",
                "n_companies": len({r["company"] for r in records}), "n_fields": len(fields),
                "python": platform.python_version(), "platform": platform.platform(terse=True),
                "cpu_count": os.cpu_count(), "gpu": _gpu(),
                "stop_reason": stop_reason,
            },
            "methods": {m: DESCRIPTIONS[m] for m in all_methods},
            "summary": summarise(records, calls, ingestion, all_methods),
            "ingestion": ingestion,
            "calls": calls,
            "records": records,
        }
        out_path.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n")
        return result


    for comp in companies:
        cid, folder = comp["id"], data / comp["folder"]
        todo = [m for m in methods if (m, cid) not in done_keys]
        if not todo:
            continue
        print(f"\n=== {cid}: {', '.join(todo)}")
        t = time.time()
        units = plain_text.extract_units(folder)
        extract_s = time.time() - t
        bm25 = Bm25Index(units)
        ingestion["plain"][cid] = {"total_ms": _ms(time.time() - t), "extract_ms": _ms(extract_s),
                                   "n_chunks": len(bm25.chunks)}
        if any(m.startswith("dam") for m in todo):
            if dam_index is None:
                dam_index = DamIndex(work)
            ingestion["dam"][cid] = dam_index.ingest(folder, cid)
            ingestion["dam_model_load_ms"] = dam_index.model_load_ms

        def retrieve(kind, query):
            return dam_index.retrieve(cid, query, args.top_k) if kind == "dam" else bm25.retrieve(query, args.top_k)

        for m in todo:
            gf = comp["fields"]
            if m == "rules":
                res = rules_baseline.extract(units)
                for f in fields:
                    r = res[f["id"]]
                    records.append(make_record(m, cid, f, gf[f["id"]], r["answer"], r["source"], "ok"))
                continue

            if stop_reason:
                records.extend(make_record(m, cid, f, gf[f["id"]], None, None, "pending") for f in fields)
                continue

            if m in ("dam", "bm25"):
                for f in fields:
                    passages = retrieve(m, f["label"])
                    ev = evidence_retrieved(passages, gf[f["id"]]["sources"]) if gf[f["id"]]["gold"] else None
                    if stop_reason:
                        records.append(make_record(m, cid, f, gf[f["id"]], None, None, "pending", ev))
                        continue
                    try:
                        rec = llm.complete(field_prompt(f, passages))
                    except LLMStop as e:
                        stop_reason = str(e)
                        records.append(make_record(m, cid, f, gf[f["id"]], None, None, "pending", ev))
                        continue
                    calls.append(_call_record(m, cid, f["id"], rec))
                    js = parse_json(rec["text"])
                    if not isinstance(js, dict):
                        records.append(make_record(m, cid, f, gf[f["id"]], None, None, "parse_error", ev, rec["text"]))
                        continue
                    answer = _as_answer(js.get("answer", js))
                    src = _source_of(passages, _as_passage(js, len(passages)))
                    records.append(make_record(m, cid, f, gf[f["id"]], answer, src, "ok", ev))
                    print(f"  {m:13s} {f['id']:24s} {str(answer)[:50]!r:52s} {'OK' if records[-1]['correct'] else '--'}")
                continue

            # one call per company
            if m == "single_call":
                prompt, passages, per_field = single_call_prompt(fields, units), None, None
            else:
                kind = "dam" if m == "dam_batched" else "bm25"
                passages, per_field, seen = [], {}, {}
                for f in fields:
                    refs = []
                    for p in retrieve(kind, f["label"]):
                        key = (p["file"], p.get("page"), p["text"])
                        if key not in seen:
                            passages.append(p)
                            seen[key] = len(passages)
                        refs.append(seen[key])
                    per_field[f["id"]] = refs
                prompt = batched_prompt(fields, passages, per_field)
            try:
                rec = llm.complete(prompt)
            except LLMStop as e:
                stop_reason = str(e)
                records.extend(make_record(m, cid, f, gf[f["id"]], None, None, "pending") for f in fields)
                continue
            calls.append(_call_record(m, cid, None, rec))
            js = parse_json(rec["text"])
            for f in fields:
                ev = None
                if passages is not None and gf[f["id"]]["gold"] is not None:
                    ev = evidence_retrieved([passages[i - 1] for i in per_field[f["id"]]], gf[f["id"]]["sources"])
                if not isinstance(js, dict):
                    records.append(make_record(m, cid, f, gf[f["id"]], None, None, "parse_error", ev,
                                               rec["text"] if f is fields[0] else None))
                    continue
                v = js.get(f["id"])
                answer = _as_answer(v)
                src = None
                if passages is not None:
                    idx = _as_passage(v, len(passages))
                    src = _source_of(passages, idx, fallback_first=False)
                    if src is None and per_field[f["id"]]:
                        top = passages[per_field[f["id"]][0] - 1]
                        src = {"file": top["file"], "page": top.get("page"), "via": "top1"}
                records.append(make_record(m, cid, f, gf[f["id"]], answer, src, "ok", ev))
            acc = sum(r["correct"] for r in records if r["method"] == m and r["company"] == cid)
            print(f"  {m:13s} {acc}/{len(fields)} correct")
        save(final=False)

    result = save(final=True)
    status = result["status"]
    print(f"\nWrote {out_path} ({status}) in {time.time() - t_run:.0f}s")
    for m, s in result["summary"].items():
        print(f"  {m:13s} {s['correct']}/{s['n_scored']} correct, {s['llm_calls']} LLM calls "
              f"({s['live_llm_calls']} live), pending {s['n_pending']}")
    if stop_reason:
        print(f"\nStopped early: {stop_reason}\nResume with the same command plus --resume.")


def _gpu():
    try:
        import torch

        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()
