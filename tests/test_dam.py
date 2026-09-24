"""DAM/ scripts: ingestion, retrieval, experiment harness and helpers (offline, heavy parts mocked)."""

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import FIXTURES, load_module

DAM_DOCS = FIXTURES / "dam_docs"


@pytest.fixture(scope="module")
def dam_env(tmp_path_factory):
    """Point the DAM scripts' import-time folders at a temporary directory."""
    work = tmp_path_factory.mktemp("dam_work")
    old = {k: os.environ.get(k) for k in ("DAM_WORK_DIR", "DAM_GROUND_TRUTH", "DAM_REINDEX_OUTPUT")}
    os.environ["DAM_WORK_DIR"] = str(work)
    os.environ["DAM_GROUND_TRUTH"] = str(FIXTURES / "dam_ground_truth.json")
    os.environ["DAM_REINDEX_OUTPUT"] = str(work / "reindex_output")
    yield work
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


@pytest.fixture(scope="module")
def harness(dam_env):
    return load_module("DAM/experiment_harness.py", "experiment_harness")


@pytest.fixture(scope="module")
def reindex(dam_env):
    return load_module("DAM/reindex.py", "reindex")


@pytest.fixture(scope="module")
def oneshot():
    return load_module("DAM/oneshot_ingestion.py", "oneshot_ingestion")


class FakeEmbedder:
    """Deterministic bag-of-words embedder standing in for BGE."""

    VOCAB = ["cin", "identity", "capital", "authorised", "board", "meeting", "revenue", "audit", "company", "name"]

    def encode(self, texts, **kwargs):
        rows = []
        for t in texts:
            words = t.lower()
            v = np.array([words.count(w) for w in self.VOCAB], dtype=float) + 1e-3
            rows.append(v / np.linalg.norm(v))
        return np.array(rows)


# --------------------------------------------------------------------------- oneshot_ingestion


def _fake_chunk(text, page=1, bbox=(10.0, 700.0, 200.0, 690.0), label="text"):
    prov = SimpleNamespace(page_no=page, bbox=SimpleNamespace(l=bbox[0], t=bbox[1], r=bbox[2], b=bbox[3]))
    item = SimpleNamespace(prov=[prov], label=label)
    return SimpleNamespace(text=text, meta=SimpleNamespace(doc_items=[item]))


def test_extract_metadata_from_chunk(oneshot):
    pipe = oneshot.OneShotIngestion()
    meta = pipe.extract_metadata_from_chunk(_fake_chunk("x", page=2), None, "a.pdf")
    assert meta["page_number"] == 2 and meta["item_type"] == "text"
    assert (meta["bbox_l"], meta["bbox_t"], meta["bbox_r"], meta["bbox_b"]) == (10.0, 700.0, 200.0, 690.0)
    assert meta["position_x"] == 105.0 and meta["position_y"] == 695.0
    empty = pipe.extract_metadata_from_chunk(SimpleNamespace(text="x", meta=None), None, "b.xlsx")
    assert empty["page_number"] is None and empty["item_type"] == "unknown"


def test_discover_files(oneshot):
    files = oneshot.OneShotIngestion(input_dir=str(DAM_DOCS)).discover_files()
    assert sorted(f.suffix for f in files) == [".docx", ".pdf", ".xlsx"]
    with pytest.raises(FileNotFoundError):
        oneshot.OneShotIngestion(input_dir="/nonexistent/dir").discover_files()


def test_process_documents_counts_failures(oneshot):
    pipe = oneshot.OneShotIngestion(input_dir=str(DAM_DOCS))

    class FakeConverter:
        def convert(self, path):
            if path.endswith(".xlsx"):
                raise RuntimeError("boom")
            return SimpleNamespace(document=path)

    class FakeChunker:
        def chunk(self, doc):
            return [_fake_chunk(f"{Path(doc).name} part {i}") for i in range(2)]

    pipe.converter, pipe.chunker = FakeConverter(), FakeChunker()
    chunks = pipe.process_documents(pipe.discover_files())
    assert len(chunks) == 4
    assert pipe.stats["failed_files"] == 1 and pipe.stats["processed_files"] == 2
    assert {c["metadata"]["chunk_id"] for c in chunks} == {0, 1}


def test_milvus_insert_and_search_roundtrip(oneshot, tmp_path):
    pipe = oneshot.OneShotIngestion(db_name=str(tmp_path / "t.db"), collection_name="t")
    pipe.embedding_dim = len(FakeEmbedder.VOCAB)
    chunks = [
        {"text": "Corporate Identity Number (CIN): U00000XX0000PTC000000",
         "metadata": pipe.extract_metadata_from_chunk(_fake_chunk("a"), None, "fs.pdf") | {"chunk_id": 0, "total_chunks": 2}},
        {"text": "The authorised capital is Rs. 10,00,000",
         "metadata": pipe.extract_metadata_from_chunk(SimpleNamespace(meta=None), None, "fs.pdf") | {"chunk_id": 1, "total_chunks": 2}},
    ]
    emb = FakeEmbedder()
    pipe.setup_milvus()
    pipe.insert_data(chunks, emb.encode([c["text"] for c in chunks]).tolist())
    hits = pipe.milvus_client.search(
        collection_name="t", data=[emb.encode(["authorised capital"])[0].tolist()], limit=1,
        output_fields=["text", "filename", "page_number"],
    )
    assert "authorised capital" in hits[0][0]["entity"]["text"]
    assert hits[0][0]["entity"]["page_number"] == 0  # missing metadata is stored as 0
    pipe.milvus_client.close()


def test_oneshot_cli_arguments(oneshot, monkeypatch):
    seen = {}
    monkeypatch.setattr(oneshot.OneShotIngestion, "run", lambda self: seen.update(vars(self)))
    monkeypatch.setattr(sys, "argv", ["oneshot_ingestion.py", "--input-dir", "docs", "--db", "x.db",
                                      "--collection", "c", "--batch-size", "4"])
    oneshot.main()
    assert (str(seen["input_dir"]), seen["db_name"], seen["collection_name"], seen["batch_size"]) == ("docs", "x.db", "c", 4)


# --------------------------------------------------------------------------- reindex


def test_reindex_uses_configured_folders(reindex, dam_env):
    assert reindex.OUTPUT_DIR == dam_env / "reindex_output"
    assert reindex.MILVUS_DB.startswith(str(dam_env))


def test_recursive_split(reindex):
    text = "\n\n".join(f"Paragraph {i} " + "word " * 60 for i in range(6))
    chunks = reindex.recursive_split(text, chunk_size=400, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 400 + 50 for c in chunks)
    assert "Paragraph 5" in chunks[-1]
    assert reindex.recursive_split("short", chunk_size=100) == ["short"]


# --------------------------------------------------------------------------- experiment_harness


def test_harness_configuration_from_env(harness, dam_env):
    assert harness.WORK_DIR == dam_env
    assert harness.RESULTS_DIR == dam_env / "experiment_results"
    assert harness.GROUND_TRUTH_PATH == FIXTURES / "dam_ground_truth.json"
    assert harness.load_ground_truth()["questions"][0]["id"] == "Q1"
    fn_src = harness.EXPERIMENTS["baseline"]["fn"].__code__.co_names
    assert "WORK_DIR" in fn_src


def test_retrieve_from_milvus_reads_filename_and_filters_excel(harness, monkeypatch):
    hits = [
        {"distance": 0.9, "entity": {"text": "CIN U999", "filename": "fs.pdf", "page_number": 1}},
        {"distance": 0.8, "entity": {"text": "trial balance", "filename": "tb.xlsx", "page_number": 1}},
        {"distance": 0.7, "entity": {"text": "old schema", "source_file": "legacy", "doc_type": "xlsx"}},
    ]

    class FakeClient:
        def __init__(self, path):
            self.path = path

        def search(self, **kwargs):
            return [hits]

        def close(self):
            pass

    import pymilvus

    monkeypatch.setattr(pymilvus, "MilvusClient", FakeClient)
    monkeypatch.setattr(harness, "_embedding_model", FakeEmbedder())
    chunks = harness.retrieve_from_milvus("cin", "x.db", "oneshot")
    assert [c["source_file"] for c in chunks] == ["fs.pdf", "tb.xlsx", "legacy"]
    filtered = harness.retrieve_from_milvus("cin", "x.db", "oneshot", exclude_excel=True)
    assert [c["source_file"] for c in filtered] == ["fs.pdf"]


def test_judge_answer_parses_fenced_json(harness, monkeypatch):
    fake = SimpleNamespace(generate_content=lambda p: SimpleNamespace(
        text='```json\n{"verdict": "correct", "reasoning": "same value"}\n```'))
    monkeypatch.setattr(harness, "get_gemini", lambda: fake)
    assert harness.judge_answer("q", "a", "a")["verdict"] == "correct"
    broken = SimpleNamespace(generate_content=lambda p: SimpleNamespace(text="nope"))
    monkeypatch.setattr(harness, "get_gemini", lambda: broken)
    assert harness.judge_answer("q", "a", "a")["verdict"] == "error"


def test_run_rag_experiment_summary(harness, monkeypatch, tmp_path):
    monkeypatch.setattr(harness, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(harness.time, "sleep", lambda s: None)
    monkeypatch.setattr(harness, "retrieve_from_milvus", lambda *a, **k: [
        {"text": "ctx", "source_file": "fs.pdf", "distance": 0.5, "page_number": 1}])
    monkeypatch.setattr(harness, "generate_answer", lambda q, c: "answer")
    verdicts = iter(["correct", "correct", "wrong", "unanswerable", "correct"])
    monkeypatch.setattr(harness, "judge_answer", lambda q, g, e: {"verdict": next(verdicts), "reasoning": ""})

    out = harness.run_rag_experiment("unit", "x.db", "oneshot", top_k=3)
    assert out["summary"] == {"correct": 3, "unanswerable": 1, "wrong": 1, "error": 0, "total": 5,
                              "answerability_rate": 80.0, "accuracy_rate": 60.0}
    saved = list(tmp_path.glob("unit_*.json"))
    assert len(saved) == 1 and json.loads(saved[0].read_text())["config"]["top_k"] == 3


def test_harness_list_cli(harness, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["experiment_harness.py", "--list"])
    harness.main()
    assert "baseline" in capsys.readouterr().out


# --------------------------------------------------------------------------- analyze_chunk_stats


def test_analyze_chunks_reads_all_batches(monkeypatch, tmp_path):
    stats_mod = load_module("DAM/analyze_chunk_stats.py", "analyze_chunk_stats")
    batches = [[{"text": "one two three", "doc_type": "pdf", "source_file": "a"}] * 3,
               [{"text": "four five", "doc_type": "xlsx", "source_file": "b"}] * 2, []]

    class FakeIterator:
        def next(self):
            return batches.pop(0)

        def close(self):
            pass

    class FakeClient:
        def __init__(self, path):
            pass

        def query_iterator(self, **kwargs):
            assert kwargs["filter"] == ""
            return FakeIterator()

    monkeypatch.setattr(stats_mod, "MilvusClient", FakeClient)
    monkeypatch.setattr(stats_mod, "count_tokens", lambda text: len(text.split()))
    monkeypatch.setattr(stats_mod, "create_visualizations", lambda *a: None)
    monkeypatch.chdir(tmp_path)
    stats, token_counts, by_type = stats_mod.analyze_chunks("x.db", "c")
    assert stats["total_chunks"] == 5
    assert sorted(by_type) == ["pdf", "xlsx"]
    assert (tmp_path / "chunk_statistics_report.json").exists()


# --------------------------------------------------------------------------- visual grounding


def test_visual_grounding_queries_and_cli(monkeypatch, capsys):
    vg = load_module("DAM/visual_grounding.py", "visual_grounding")
    assert len(vg.queries) == 17  # every form field is a separate query

    class FakeClient:
        def __init__(self, path):
            assert path == "x.db"

        def search(self, **kwargs):
            assert kwargs["collection_name"] == "c" and kwargs["limit"] == 1
            return [[{"entity": {"filename": "fs.pdf", "page_number": 1, "bbox_l": 1, "bbox_t": 2,
                                 "bbox_r": 3, "bbox_b": 4, "text": "CIN: U999"}}]]

    monkeypatch.setattr(vg, "SentenceTransformer", lambda name: FakeEmbedder())
    monkeypatch.setattr(vg, "MilvusClient", FakeClient)
    monkeypatch.setattr(sys, "argv", ["visual_grounding.py", "--db", "x.db", "--collection", "c",
                                      "-q", "CIN of company", "--limit", "1"])
    vg.main()
    out = capsys.readouterr().out
    assert "Query: CIN of company" in out and "File: fs.pdf" in out


def test_visual_grounding_enhanced_highlights_oneshot_pdf_hit(tmp_path):
    vge = load_module("DAM/visual_grounding_enhanced.py", "visual_grounding_enhanced")
    grounder = object.__new__(vge.VisualGrounder)  # skip model/database loading
    grounder.data_dir = DAM_DOCS
    entity = {  # oneshot schema: filename with extension, 1-based page, bottom-left bbox
        "filename": "examplecorp_financial_statements.pdf", "page_number": 1,
        "bbox_l": 78.0, "bbox_t": 673.0, "bbox_r": 348.0, "bbox_b": 663.8, "text": "CIN",
    }
    grounder.visualize_chunk(entity, 1, "CIN", tmp_path)
    out = tmp_path / "result_1_examplecorp_financial_statements_page1.png"
    assert out.exists() and out.stat().st_size > 0
