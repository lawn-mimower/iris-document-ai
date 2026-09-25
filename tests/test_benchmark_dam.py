"""DAM benchmark: generator, plain-text baselines, rules, cached LLM client and the runner (LLM faked)."""

import json
import sys

import pytest

from conftest import REPO_ROOT, load_module

sys.path.insert(0, str(REPO_ROOT / "benchmarks"))
import llm_client  # noqa: E402
import plain_text  # noqa: E402
import rules_baseline  # noqa: E402
from scoring import score_field  # noqa: E402

DATA = REPO_ROOT / "benchmarks" / "data" / "dam"
gen = load_module("benchmarks/generate_dam_benchmark.py", "generate_dam_benchmark")
runner = load_module("benchmarks/run_dam_benchmark.py", "run_dam_benchmark")


@pytest.fixture(scope="module")
def truth():
    return json.loads((DATA / "ground_truth.json").read_text())


def test_generator_is_deterministic_and_matches_committed_data(tmp_path):
    gen.build(tmp_path)
    assert json.loads((tmp_path / "ground_truth.json").read_text()) == json.loads((DATA / "ground_truth.json").read_text())
    for company in gen.COMPANIES:
        for name in (gen.PDF_NAME, gen.DOCX_NAME, gen.XLSX_NAME):
            assert (tmp_path / company["id"] / name).read_bytes() == (DATA / company["id"] / name).read_bytes(), name


def test_ground_truth_shape(truth):
    assert len(truth["companies"]) == 6 and len(truth["fields"]) == 16
    nulls = [(c["id"], f) for c in truth["companies"] for f, v in c["fields"].items() if v["gold"] is None]
    assert nulls == [("c4_quorrin", "email"), ("c6_lumeqa", "agm_date")]
    for c in truth["companies"]:
        for f, v in c["fields"].items():
            assert v["gold"] is None or v["sources"], (c["id"], f)
            # every CIN is fictional: state code ZZ does not exist
            if f == "cin":
                assert v["gold"][6:8] == "ZZ"


def test_inr_grouping_and_units():
    assert gen.inr(1846235000) == "1,84,62,35,000"
    assert gen.in_unit(4231864000, "lakhs") == "42,318.64"
    assert gen.in_unit(84123500000, "crore") == "8,412.35"
    assert gen.in_unit(973160000, "millions") == "973.16"
    with pytest.raises(ValueError):
        gen.in_unit(123, "lakhs")


def test_extract_units_and_chunking():
    units = plain_text.extract_units(DATA / "c1_velmora")
    assert [(u["file"], u["page"]) for u in units[:4]] == [("financial_statements.pdf", p) for p in (1, 2, 3, 4)]
    assert any(u.get("sheet") == "Company master" for u in units)
    chunks = plain_text.chunk_units([{"file": "a", "page": 1, "text": "x" * 2500}], size=1000, overlap=200)
    assert [len(c["text"]) for c in chunks] == [1000, 1000, 900]
    with pytest.raises(ValueError):
        plain_text.chunk_units([], size=100, overlap=100)


def test_bm25_ranks_the_matching_chunk_first():
    bm = plain_text.BM25(["the board met four times", "authorised share capital is Rs. 10 crore",
                          "revenue from operations increased"])
    assert bm.top_k("Authorised capital of the company", k=1) == [1]


def test_rules_baseline_on_one_company(truth):
    comp = next(c for c in truth["companies"] if c["id"] == "c2_tarnwick")
    res = rules_baseline.extract(plain_text.extract_units(DATA / "c2_tarnwick"))
    ok = {f: score_field(fd["type"], comp["fields"][f]["gold"], res[f]["answer"])["correct"]
          for f, fd in ((f["id"], f) for f in truth["fields"])}
    assert ok["cin"] and ok["authorised_capital"] and ok["revenue"] and ok["board_meeting_date"]
    # the distractors do their job: the registrar's e-mail, a subsidiary's secretarial-audit sentence and
    # the dividend note that mentions the "ensuing Annual General Meeting" next to the board date
    assert not ok["email"] and not ok["secretarial_audit"] and not ok["agm_date"]
    assert res["cin"]["source"] == {"file": "financial_statements.pdf", "page": 1}


def test_llm_client_caches_and_keeps_a_ledger(tmp_path, monkeypatch):
    calls = []

    def fake(self, prompt, json_mode):
        calls.append(prompt)
        return {"text": '{"answer": "42"}', "prompt_tokens": 10, "completion_tokens": 3}

    monkeypatch.setattr(llm_client.LLMClient, "_call_ollama", fake)
    c = llm_client.LLMClient("ollama", "tiny", cache_dir=tmp_path)
    first = c.complete("q")
    second = c.complete("q")
    assert calls == ["q"] and not first["cached"] and second["cached"] and second["text"] == '{"answer": "42"}'
    assert c.calls_today() == 1
    capped = llm_client.LLMClient("ollama", "tiny", cache_dir=tmp_path, daily_cap=1)
    with pytest.raises(llm_client.QuotaExhausted):
        capped.complete("another prompt")
    assert capped.complete("q")["cached"]            # cached answers are still served
    offline = llm_client.LLMClient("ollama", "tiny", cache_dir=tmp_path, offline=True)
    with pytest.raises(llm_client.QuotaExhausted):
        offline.complete("never seen")
    thinking = llm_client.LLMClient("ollama", "tiny", cache_dir=tmp_path, think="low")
    assert thinking._key("q", True) != c._key("q", True)   # the think level is part of the cache key
    assert not thinking.complete("q")["cached"]


def test_parse_json_tolerates_fences():
    assert llm_client.parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert llm_client.parse_json('Here: {"a": 2} done') == {"a": 2}
    assert llm_client.parse_json("not json") is None


def test_answer_and_passage_helpers():
    passages = [{"file": "a.pdf", "page": 2, "text": "CIN: L29120ZZ1994PLC011872"},
                {"file": "b.docx", "page": 0, "text": "other"}]
    assert runner._as_passage({"passage": "[2]"}, 2) == 2
    assert runner._as_passage({"passage": 7}, 2) is None
    assert runner._as_answer({"answer": True}) == "Yes"
    assert runner._source_of(passages, 2)["file"] == "b.docx"
    assert runner._source_of(passages, None) == {"file": "a.pdf", "page": 2, "via": "top1"}
    sources = [{"file": "a.pdf", "page": 2, "evidence": ["L29120ZZ1994PLC011872"]}]
    assert runner.evidence_retrieved(passages, sources)
    assert not runner.evidence_retrieved(passages[1:], sources)
    prompt = runner.field_prompt(runner.dam_fields.FIELDS[0], passages)
    assert "[1] (source: a.pdf, page 2)" in prompt and "UNANSWERABLE" in prompt


class FakeLLM:
    """Answers every field from the ground truth of the company whose name is in the prompt."""

    fail_after = None
    n = 0

    def __init__(self, *a, **k):
        pass

    def complete(self, prompt, json_mode=True):
        FakeLLM.n += 1
        if FakeLLM.fail_after is not None and FakeLLM.n > FakeLLM.fail_after:
            raise llm_client.QuotaExhausted("429 quota")
        truth = json.loads((DATA / "ground_truth.json").read_text())
        comp = next((c for c in truth["companies"] if c["name"] in prompt), truth["companies"][0])
        gold = {f: (v["gold"] or "UNANSWERABLE") for f, v in comp["fields"].items()}
        if "Question:" in prompt:
            label = prompt.split("Question: ", 1)[1].split("\n")[0]
            fid = next(f["id"] for f in runner.dam_fields.FIELDS if f["label"] == label)
            text = json.dumps({"answer": gold[fid], "passage": 1})
        elif "retrieved passages" in prompt:
            text = json.dumps({f: {"answer": v, "passage": None} for f, v in gold.items()})
        else:
            text = json.dumps(gold)
        return {"text": text, "prompt_tokens": len(prompt) // 4, "completion_tokens": 20, "latency_s": 0.1,
                "cached": False}


def _run(monkeypatch, tmp_path, *extra):
    monkeypatch.setattr(runner, "LLMClient", FakeLLM)
    monkeypatch.setattr(sys, "argv", ["run_dam_benchmark.py", "--run", "t", "--results-dir", str(tmp_path),
                                      "--work-dir", str(tmp_path / "work"), "--companies", "c2_tarnwick,c6_lumeqa",
                                      "--methods", "rules,single_call,bm25,bm25_batched", *extra])
    runner.main()
    return json.loads((tmp_path / "dam_t.json").read_text())


def test_runner_scores_methods_with_a_fake_llm(tmp_path, monkeypatch):
    FakeLLM.fail_after, FakeLLM.n = None, 0
    out = _run(monkeypatch, tmp_path)
    s = out["summary"]
    assert out["status"] == "complete" and set(s) == {"rules", "single_call", "bm25", "bm25_batched"}
    for m in ("single_call", "bm25", "bm25_batched"):
        assert s[m]["correct"] == s[m]["n_scored"] == 32, m      # the fake answers with the gold values
    assert s["bm25"]["llm_calls"] == 32 and s["single_call"]["llm_calls"] == 2 and s["bm25_batched"]["llm_calls"] == 2
    assert s["rules"]["llm_calls"] == 0 and s["rules"]["correct"] < 32
    assert s["bm25"]["evidence_n"] == 31                          # one gold value is null
    assert all(r["source"] is None for r in out["records"] if r["method"] == "single_call")


def test_runner_saves_partial_results_and_resumes(tmp_path, monkeypatch):
    FakeLLM.fail_after, FakeLLM.n = 5, 0
    out = _run(monkeypatch, tmp_path)
    assert out["status"] == "partial" and "429" in out["conditions"]["stop_reason"]
    assert out["summary"]["bm25"]["n_pending"] > 0 and out["summary"]["rules"]["n_pending"] == 0
    FakeLLM.fail_after, FakeLLM.n = None, 0
    out = _run(monkeypatch, tmp_path, "--resume")
    assert out["status"] == "complete"
    # rules and c2's single call finished in the first run and were kept; the rest was redone:
    # bm25 (16) + bm25_batched (1) for c2, single_call (1) + bm25 (16) + bm25_batched (1) for c6
    assert FakeLLM.n == 35


def test_redo_recomputes_only_the_selected_companies(tmp_path, monkeypatch):
    FakeLLM.fail_after, FakeLLM.n = None, 0
    _run(monkeypatch, tmp_path)
    FakeLLM.n = 0
    monkeypatch.setattr(sys, "argv", ["run_dam_benchmark.py", "--run", "t", "--results-dir", str(tmp_path),
                                      "--work-dir", str(tmp_path / "work"), "--companies", "c6_lumeqa",
                                      "--methods", "single_call", "--resume", "--redo"])
    runner.main()
    out = json.loads((tmp_path / "dam_t.json").read_text())
    assert FakeLLM.n == 1
    assert {(r["method"], r["company"]) for r in out["records"]} == {
        (m, c) for m in ("rules", "single_call", "bm25", "bm25_batched") for c in ("c2_tarnwick", "c6_lumeqa")}
    assert out["summary"]["bm25"]["n_scored"] == 32
