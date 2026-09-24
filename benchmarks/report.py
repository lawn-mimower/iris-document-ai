#!/usr/bin/env python3
"""Print the Markdown tables used in benchmarks/RESULTS.md from the JSON files in benchmarks/results/.

    python benchmarks/report.py                 # every results file
    python benchmarks/report.py dam_local-llama3.2.json forms_gemini-2.5-flash.json
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from dam_fields import FIELDS  # noqa: E402

DAM_ORDER = ["dam", "bm25", "dam_batched", "bm25_batched", "single_call", "rules"]
FORM_ORDER = ["agent", "agent_vector", "geometric_phase1", "geometric_vector"]


def _s(ms):
    """Milliseconds as seconds with one decimal."""
    return f"{ms / 1000:,.1f}"


def pct(a, b):
    return f"{a}/{b} ({100 * a / b:.0f}%)" if b else "–"


def dam_tables(r: dict) -> str:
    s = r["summary"]
    methods = [m for m in DAM_ORDER if m in s]
    out = [f"### {r['run']} (status: {r['status']}; LLM: {r['conditions']['llm'] or 'none'})", ""]
    out.append("| Method | Correct (all fields) | Strict | Answerable correct | Declined when not stated "
               "| Source file/page correct | Gold evidence in retrieved context | LLM calls | Prompt tokens "
               "| Output tokens | LLM time (s) | Ingestion (s) |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for m in methods:
        x = s[m]
        out.append(
            f"| {m} | {pct(x['correct'], x['n_scored'])} | {pct(x['strict'], x['n_scored'])} "
            f"| {pct(x['answerable_correct'], x['answerable_n'])} | {x['unanswerable_correct']}/{x['unanswerable_n']} "
            f"| {pct(x['source_correct'], x['source_n']) if x['source_n'] else '–'} "
            f"| {pct(x['evidence_in_context'], x['evidence_n']) if x['evidence_n'] else '–'} "
            f"| {x['llm_calls']} | {x['prompt_tokens']:,} | {x['completion_tokens']:,}"
            f"{' (+' + format(x['thinking_tokens'], ',') + ' thinking)' if x.get('thinking_tokens') else ''} "
            f"| {_s(x['llm_time_ms'])} | {_s(x['ingestion_ms'])} |")
        if x["n_pending"]:
            out[-1] += f" {x['n_pending']} pending"
    out += ["", "Per field (correct out of companies scored):", ""]
    out.append("| Field | " + " | ".join(methods) + " |")
    out.append("|---|" + "---|" * len(methods))
    for f in FIELDS:
        cells = [f"{s[m]['per_field'][f['id']]['correct']}/{s[m]['per_field'][f['id']]['n']}" for m in methods]
        out.append(f"| {f['id']} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def form_tables(r: dict) -> str:
    s = r["summary"]
    methods = [m for m in FORM_ORDER if m in s]
    out = [f"### {r['run']} (status: {r['status']}; agent LLM: {r['conditions']['agent_llm'] or 'none'})", ""]
    out.append("| Form | " + " | ".join(methods) + " |")
    out.append("|---|" + "---|" * len(methods))
    forms = sorted({k.split("|")[0] for k in r["results"]})
    status = {k: v["status"] for k, v in r["results"].items()}
    for f in forms:
        out.append(f"| {f} | " + " | ".join(
            s[m]["per_form"].get(f, "pending" if status.get(f"{f}|{m}") == "pending" else "–") for m in methods) + " |")
    out.append("| **all** | " + " | ".join(f"**{pct(s[m]['correct'], s[m]['fields'])}**" for m in methods) + " |")
    kinds = sorted({k for m in methods for k in s[m]["by_kind"]})
    for k in kinds:
        out.append(f"| {k} fields | " + " | ".join(
            pct(s[m]["by_kind"].get(k, {}).get("correct", 0), s[m]["by_kind"].get(k, {}).get("n", 0)) for m in methods) + " |")
    llm = [m for m in methods if s[m]["llm_calls"]]
    if llm:
        out += ["", "| Method | LLM calls | Prompt tokens | Output tokens | Thinking tokens | LLM time (s) |",
                "|---|---|---|---|---|---|"]
        for m in llm:
            x = s[m]
            out.append(f"| {m} | {x['llm_calls']} | {x['prompt_tokens']:,} | {x['completion_tokens']:,} "
                       f"| {x['thinking_tokens']:,} | {_s(x['llm_time_ms'])} |")
    per = r.get("perception", {})
    if per:
        out += ["", "Target boxes reported by the perception step (IoU > 0.5):", "",
                "| Form | phase1 + gestalt | vector drawings |", "|---|---|---|"]
        for f, p in sorted(per.items()):
            a, b = p["phase1_gestalt"], p["vector"]
            out.append(f"| {f} | {a['found']}/{a['targets']} ({a['n_elements']} elements) "
                       f"| {b['found']}/{b['targets']} ({b['n_elements']} elements) |")
    return "\n".join(out)


def main():
    names = sys.argv[1:] or sorted(p.name for p in (HERE / "results").glob("*.json"))
    for name in names:
        r = json.loads((HERE / "results" / name).read_text())
        print(dam_tables(r) if r["benchmark"] == "dam_field_extraction" else form_tables(r))
        print()


if __name__ == "__main__":
    main()
