"""Notebooks that select their behaviour with a setting in the first code cell (tagged "parameters").

The cells are executed offline with small stand-ins for Pinecone, Milvus, OCR and the embedding model.
"""

import ast
import io
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
from typing import List

import numpy as np
import pandas as pd
import pytest
from PIL import Image, ImageDraw

from conftest import REPO_ROOT

TRUERAG = REPO_ROOT / "ComparisonScriptsOctober/trueRAG.ipynb"
CHUNKING_NB = REPO_ROOT / "DAM/phase2_chunking_embedding.ipynb"
BGE_NB = REPO_ROOT / "DAM/beg_rag-excel.ipynb"
MERGED = [TRUERAG, CHUNKING_NB, BGE_NB]


def code_cells(path):
    nb = json.loads(Path(path).read_text(encoding="utf-8"))
    return [c for c in nb["cells"] if c["cell_type"] == "code"]


def source(cell):
    return "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]


def cell_starting_with(path, prefix):
    matches = [source(c) for c in code_cells(path) if source(c).startswith(prefix)]
    assert len(matches) == 1, (prefix, len(matches))
    return matches[0]


def run_settings(path, name, value):
    """Execute the parameters cell (with `name` overridden) and the settings cell after it."""
    cells = code_cells(path)
    assert "parameters" in cells[0]["metadata"].get("tags", [])
    ns = {}
    exec(source(cells[0]), ns)
    assert name in ns
    ns[name] = value
    exec(source(cells[1]), ns)
    return ns


@pytest.mark.parametrize("path", MERGED, ids=[str(p.relative_to(REPO_ROOT)) for p in MERGED])
def test_merged_notebook_has_parameters_cell_and_valid_code(path):
    cells = code_cells(path)
    assert cells[0]["metadata"].get("tags") == ["parameters"]
    for cell in cells:
        ast.parse(source(cell))


# --- ComparisonScriptsOctober/trueRAG.ipynb ---------------------------------------------------------------------

TRUERAG_EXPECTED = {
    "full": dict(RUN_INGESTION=True, EXCEL_MODE="text", namespace=None, CLEAR_INDEX=True, batch_size=1024),
    "no_excel": dict(RUN_INGESTION=True, EXCEL_MODE="none", namespace="no-excel", CLEAR_INDEX=False, batch_size=256),
    "excel_ocr": dict(RUN_INGESTION=True, EXCEL_MODE="ocr", namespace="excel-ocr", CLEAR_INDEX=False, batch_size=256),
    "query_only": dict(RUN_INGESTION=False, EXCEL_MODE="text", namespace=None, CLEAR_INDEX=False, batch_size=1024),
}


@pytest.mark.parametrize("variant", list(TRUERAG_EXPECTED))
def test_truerag_variant_settings(variant):
    ns = run_settings(TRUERAG, "VARIANT", variant)
    expected = dict(TRUERAG_EXPECTED[variant])
    assert ns["settings"]["batch_size"] == expected.pop("batch_size")
    for key, value in expected.items():
        assert ns[key] == value, key


def test_truerag_unknown_variant_rejected():
    with pytest.raises(ValueError, match="Unknown VARIANT"):
        run_settings(TRUERAG, "VARIANT", "everything")


class _Splitter:
    def __init__(self, **kwargs):
        pass

    def split_text(self, text):
        return [text]


def truerag_helpers(excel_mode, **extra):
    ns = dict(Path=Path, List=List, io=io, ET=ET, pd=pd, Image=Image, ImageDraw=ImageDraw,
              tqdm=lambda it, **kw: it, RecursiveCharacterTextSplitter=_Splitter,
              fitz=None, docx=None, pytesseract=None, index_name="test-index", EXCEL_MODE=excel_mode)
    ns.update(extra)
    exec(cell_starting_with(TRUERAG, "def find_all_documents"), ns)
    return ns


@pytest.fixture
def docs_folder(tmp_path):
    for name in ["report.pdf", "sheet.xlsx", "notes.txt", "._hidden.pdf", "table.csv"]:
        (tmp_path / name).write_bytes(b"x")
    (tmp_path / "__MACOSX").mkdir()
    (tmp_path / "__MACOSX" / "meta.pdf").write_bytes(b"x")
    return tmp_path


@pytest.mark.parametrize("excel_mode,has_excel", [("text", True), ("ocr", True), ("none", False)])
def test_truerag_find_all_documents_respects_excel_mode(docs_folder, excel_mode, has_excel):
    found = [Path(p).name for p in truerag_helpers(excel_mode)["find_all_documents"](str(docs_folder))]
    assert found == sorted(["notes.txt", "report.pdf"] + (["sheet.xlsx"] if has_excel else []))


@pytest.fixture
def workbook(tmp_path):
    path = tmp_path / "book.xlsx"
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame([["Revenue", 100], ["Costs", 40]]).to_excel(writer, sheet_name="PnL", header=False, index=False)
    return path


def test_truerag_reads_excel_as_text(workbook):
    pages = truerag_helpers("text")["extract_text_from_document"](str(workbook))
    assert len(pages) == 1
    assert pages[0]["content"].startswith("--- Content from Sheet: PnL ---")
    assert "Revenue" in pages[0]["content"]


def test_truerag_reads_excel_with_ocr(workbook):
    seen = {}

    def image_to_string(image, config=None):
        seen["size"], seen["config"] = image.size, config
        return "OCR TEXT"

    ns = truerag_helpers("ocr", pytesseract=SimpleNamespace(image_to_string=image_to_string))
    pages = ns["extract_text_from_document"](str(workbook))
    assert pages == [{"page_number": 1, "content": "=== Sheet: PnL ===\n\nOCR TEXT"}]
    assert seen["config"] == "--psm 6" and seen["size"][0] > 0


def test_truerag_skips_excel_without_excel_mode(workbook):
    assert truerag_helpers("none")["extract_text_from_document"](str(workbook)) == []


class _Index:
    def __init__(self):
        self.calls = []

    def upsert(self, **kwargs):
        self.calls.append(kwargs)


class _Model:
    def encode(self, texts, **kwargs):
        return np.ones((len(texts), 3))


@pytest.mark.parametrize("namespace", [None, "no-excel"])
def test_truerag_upsert_uses_variant_namespace(tmp_path, namespace):
    doc = tmp_path / "notes.txt"
    doc.write_text("Board meeting held on 1 April.", encoding="utf-8")
    index = _Index()
    truerag_helpers("text")["process_and_upsert_document"](str(doc), index, _Model(), 8, namespace)
    assert len(index.calls) == 1
    call = index.calls[0]
    assert call.get("namespace") == namespace and ("namespace" in call) == (namespace is not None)
    (vector_id, vector, metadata), = call["vectors"]
    assert vector_id == "notes.txt_p1_c0" and metadata["source_file"] == "notes.txt"


# --- DAM/phase2_chunking_embedding.ipynb ------------------------------------------------------------------------

CHUNKING_EXPECTED = {
    "hierarchical": ("milvus_all_docs.db", "rag_all_docs", "gemini-2.5-flash-lite", None, False),
    "hybrid": ("milvus_all_docs_hybrid.db", "rag_all_docs_hybrid", "gemini-2.5-flash-lite", 1000, False),
    "mixed": ("milvus_all_docs_mixed.db", "rag_all_docs_mixed", "gemini-2.5-flash", None, True),
}


def chunking_config(strategy):
    ns = run_settings(CHUNKING_NB, "CHUNKING", strategy)
    ns["Path"] = Path
    exec(cell_starting_with(CHUNKING_NB, "# Configuration"), ns)
    return ns


@pytest.mark.parametrize("strategy", list(CHUNKING_EXPECTED))
def test_chunking_strategy_settings(strategy):
    db, collection, model, insert_batch, has_chunker_field = CHUNKING_EXPECTED[strategy]
    ns = chunking_config(strategy)
    assert (ns["MILVUS_DB"], ns["COLLECTION_NAME"], ns["GEMINI_MODEL"]) == (db, collection, model)
    assert ns["INSERT_BATCH_SIZE"] == insert_batch
    assert ("chunker_used" in ns["METADATA_FIELDS"]) == has_chunker_field
    assert ns["USE_FINANCIAL_PROMPT"] == (strategy != "hierarchical")


class _Chunker:
    def __init__(self, **kwargs):
        self.max_tokens = 512

    def chunk(self, doc):
        return [SimpleNamespace(text=f"{type(self).__name__}:{doc['name']}:{i}") for i in range(2)]


class _Hierarchical(_Chunker):
    pass


class _Hybrid(_Chunker):
    pass


@pytest.mark.parametrize("strategy,xlsx_chunker,pdf_chunker", [
    ("hierarchical", "_Hierarchical", "_Hierarchical"),
    ("hybrid", "_Hybrid", "_Hybrid"),
    ("mixed", "_Hierarchical", "_Hybrid"),
])
def test_chunking_picks_chunker_per_document_type(strategy, xlsx_chunker, pdf_chunker):
    ns = chunking_config(strategy)
    ns.update(HierarchicalChunker=_Hierarchical, HybridChunker=_Hybrid, tqdm=lambda it, **kw: it,
              DoclingDocument=SimpleNamespace(model_validate=lambda data: data),
              documents={"tb.json": {"name": "tb", "origin": {"filename": "trial_balance.xlsx"}},
                         "fs.json": {"name": "fs", "origin": {"filename": "statements.pdf"}}})
    exec(cell_starting_with(CHUNKING_NB, 'if CHUNKING == "hierarchical":'), ns)
    by_file = {}
    for chunk in ns["all_chunks"]:
        by_file.setdefault(chunk["source_file"], []).append(chunk)
    assert [c["text"].split(":")[0] for c in by_file["tb.json"]] == [xlsx_chunker] * 2
    assert [c["text"].split(":")[0] for c in by_file["fs.json"]] == [pdf_chunker] * 2
    assert {c["doc_type"] for c in by_file["tb.json"]} == {"xlsx"}
    if strategy == "mixed":
        assert {c["chunker_used"] for c in by_file["tb.json"]} == {"HierarchicalChunker"}
        assert {c["chunker_used"] for c in by_file["fs.json"]} == {"HybridChunker"}
    else:
        assert all("chunker_used" not in c for c in ns["all_chunks"])


class _Milvus:
    def __init__(self):
        self.batches = []

    def insert(self, collection_name, data):
        self.batches.append(len(data))
        return {"insert_count": len(data)}


@pytest.mark.parametrize("strategy,expected_batches", [
    ("hierarchical", [2500]), ("hybrid", [1000, 1000, 500]), ("mixed", [2500]),
])
def test_chunking_insert_batches(strategy, expected_batches):
    ns = chunking_config(strategy)
    chunks = [{"text": f"t{i}", "source_file": "a.json", "doc_type": "pdf", "chunk_id": i, "total_chunks": 2500,
               "chunker_used": "HybridChunker"} for i in range(2500)]
    client = _Milvus()
    ns.update(all_chunks=chunks, embeddings_list=[[0.0]] * 2500, client=client, time=time)
    exec(cell_starting_with(CHUNKING_NB, "# Prepare data for insertion"), ns)
    assert client.batches == expected_batches
    assert ("chunker_used" in ns["data_to_insert"][0]) == (strategy == "mixed")


# --- DAM/beg_rag-excel.ipynb ------------------------------------------------------------------------------------

@pytest.mark.parametrize("source_name,suffix,form_questions", [("excel", ".xlsx", True), ("pdf", ".pdf", False)])
def test_bge_rag_source_settings(source_name, suffix, form_questions):
    ns = run_settings(BGE_NB, "SOURCE", source_name)
    assert ns["SOURCE_PATH"].endswith(suffix)
    assert ns["RUN_FORM_QUESTIONS"] is form_questions
    assert ns["QUESTION"]


@pytest.mark.parametrize("source_name,creates_collection", [("excel", True), ("pdf", False)])
def test_bge_rag_form_questions_only_for_excel(source_name, creates_collection):
    ns = run_settings(BGE_NB, "SOURCE", source_name)
    opened = []

    class Client:
        def __init__(self, uri):
            opened.append(uri)

        def has_collection(self, name):
            return False

        def create_collection(self, **kwargs):
            pass

        def insert(self, **kwargs):
            pass

    ns.update(MilvusClient=Client, texts=["a", "b"], tqdm=lambda it, **kw: it, emb_text=lambda t: [0.0],
              embedding_dim=1)
    exec(cell_starting_with(BGE_NB, "if RUN_FORM_QUESTIONS:"), ns)
    assert bool(opened) == creates_collection
