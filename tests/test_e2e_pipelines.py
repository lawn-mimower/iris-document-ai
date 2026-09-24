"""Live end-to-end runs on the synthetic fixtures. Run with: pytest -m e2e

These need the docling layout models and BAAI/bge-large-en-v1.5 (downloaded or already in the
Hugging Face cache); the harness test also needs GEMINI_API_KEY / GOOGLE_API_KEY.
"""

import os
import subprocess
import sys

import pytest

from conftest import FIXTURES, REPO_ROOT, gemini_key_available, load_module

pytestmark = pytest.mark.e2e


def _require(module_name):
    try:
        __import__(module_name)
    except Exception as exc:  # broken installs raise more than ImportError
        pytest.skip(f"{module_name} not importable: {exc}")


@pytest.fixture(scope="module")
def ingested_db(tmp_path_factory):
    _require("docling")
    _require("sentence_transformers")
    oneshot = load_module("DAM/oneshot_ingestion.py", "oneshot_ingestion_e2e")
    work = tmp_path_factory.mktemp("oneshot")
    db = work / "oneshot.db"
    pipe = oneshot.OneShotIngestion(input_dir=str(FIXTURES / "dam_docs"), db_name=str(db), collection_name="oneshot")
    cwd = os.getcwd()
    os.chdir(work)  # the ingestion report is written to the working directory
    try:
        pipe.run()
    finally:
        os.chdir(cwd)
    assert pipe.stats["processed_files"] == 3 and pipe.stats["total_chunks"] > 0
    pipe.milvus_client.close()
    return db


def test_oneshot_ingestion_and_retrieval(ingested_db):
    from pymilvus import MilvusClient
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("BAAI/bge-large-en-v1.5")
    client = MilvusClient(str(ingested_db))
    hits = client.search(
        collection_name="oneshot",
        data=[model.encode(["Corporate Identity Number (CIN) of company"])[0].tolist()],
        limit=3,
        output_fields=["text", "filename", "page_number"],
    )[0]
    client.close()
    top = hits[0]["entity"]
    assert "U00000XX0000PTC000000" in top["text"]
    assert top["filename"] == "examplecorp_financial_statements.pdf" and top["page_number"] == 1


@pytest.mark.skipif(not gemini_key_available(), reason="GEMINI_API_KEY / GOOGLE_API_KEY not set")
def test_harness_baseline_on_synthetic_ground_truth(ingested_db, monkeypatch, tmp_path):
    monkeypatch.setenv("DAM_WORK_DIR", str(tmp_path))
    monkeypatch.setenv("DAM_GROUND_TRUTH", str(FIXTURES / "dam_ground_truth.json"))
    harness = load_module("DAM/experiment_harness.py", "experiment_harness_e2e")
    out = harness.run_rag_experiment("e2e_baseline", str(ingested_db), "oneshot", top_k=5)
    errors = [r["reasoning"] for r in out["results"] if r["verdict"] == "error"]
    if out["summary"]["correct"] < 3 and errors and all("429" in e for e in errors):
        pytest.skip("Gemini quota exhausted (HTTP 429)")
    # the synthetic documents answer every question; allow one miss or transient API error
    assert out["summary"]["correct"] >= 3


def test_docling_convert_cli(tmp_path):
    _require("docling")
    out = tmp_path / "fs.md"
    subprocess.run([sys.executable, str(REPO_ROOT / "KG/doclingConvert.py"),
                    str(FIXTURES / "dam_docs/examplecorp_financial_statements.pdf"), "-o", str(out)],
                   check=True, capture_output=True)
    md = out.read_text()
    assert "U00000XX0000PTC000000" in md and "Share Capital" in md


def test_table_extraction_cli(tmp_path):
    _require("docling")
    subprocess.run([sys.executable, str(REPO_ROOT / "KG/tables.py"),
                    str(FIXTURES / "dam_docs/examplecorp_financial_statements.pdf"), "-o", str(tmp_path / "t")],
                   check=True, capture_output=True, cwd=tmp_path)
    csv = (tmp_path / "t" / "table_1.csv").read_text()
    assert "Revenue from operations" in csv and "1,23,45,000" in csv
