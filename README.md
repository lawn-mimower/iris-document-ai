# iris-document-ai

Business documents such as filled-in forms, financial statements and spreadsheets rarely come with a schema. This repo collects experiments on two ways to get structured answers out of them. The first reads a form's layout from its geometry, so that an LLM can learn from one filled-in example where each value goes. The second runs retrieval over a folder of mixed documents to answer the fields of a company-law filing, pointing back to the file, page and box of the passages behind each answer. It is a research repo: the main steps are scripts with command-line options and tests, and the rest are notebooks kept as a lab record.

## Strand 1: form understanding

```
form PDF ─► phase1/phase1.py
              text spans from PyMuPDF + shapes from a 300-dpi render (OpenCV contours)
              shapes overlapping text → INPUT_FIELD / CHECKBOX; leftover boxes → empty fields
         ─► prototype1/gestalt_processor.py
              merges adjacent text fragments and adjacent box pieces (checkboxes stay separate),
              adds each element's nearest neighbours (distance, direction, alignment)
         ─► prototype1/agent.py   (Gemini, JSON output)
              prompt = processed empty form + processed filled "golden sample" + new data (JSON)
         ─► fill plan: [{"text": "...", "bbox": [x0, y0, x1, y1]}, ...]   (PDF points, top-left origin)
```

`checkbox_trials/visualise_elements.py` draws the detected elements on the page so you can inspect them. The agent only produces the plan: writing the values into the PDF is not implemented. Input must be a PDF, because labels come from its text layer. The command line rejects other file types.

To try it on the fictional demo form in `tests/fixtures/`:

```bash
mkdir -p out
python phase1/phase1.py tests/fixtures/form_empty.pdf out/empty_raw.json
python phase1/phase1.py tests/fixtures/form_filled.pdf out/filled_raw.json
python prototype1/gestalt_processor.py out/empty_raw.json out/empty.json
python prototype1/gestalt_processor.py out/filled_raw.json out/filled.json
python checkbox_trials/visualise_elements.py tests/fixtures/form_empty.pdf out/empty_raw.json out/empty_raw.png
python prototype1/agent.py --empty-form-structure out/empty.json \
  --golden-sample-structure out/filled.json \
  --new-data tests/fixtures/form_new_data.json --output-plan out/fill_plan.json   # needs a Gemini key
```

## Strand 2: document Q&A with source grounding (DAM)

The target is the set of fields in an Indian company's annual financial-statement filing (MCA form AOC-4): CIN, company name, registered office, financial year, board meeting date, AGM details, audit questions and so on. The fields are answered from a folder of the company's documents.

```
folder of documents (anything Docling reads; tried here on PDF, DOCX and XLSX)
  ─► Docling conversion (layout, tables, OCR)
  ─► chunking: HierarchicalChunker (DAM/oneshot_ingestion.py); DAM/reindex.py also builds
     HybridChunker and a 1,000/200-character recursive split side by side for comparison
  ─► BAAI/bge-large-en-v1.5 embeddings (1024-d) ─► Milvus Lite (.db file)
     each chunk keeps its file name, page, bounding box and item type (text, table, heading, ...)
  ─► form-field label as the query ─► top-k chunks ─► Gemini answer ("UNANSWERABLE" if not in context)
  ─► DAM/visual_grounding.py            prints file / page / box for each hit
     DAM/visual_grounding_enhanced.py   draws the retrieved chunk on the PDF page (PNG)
  ─► DAM/experiment_harness.py          runs a question set through a retrieval setup; Gemini judges
                                        each answer against the ground truth (correct / unanswerable / wrong)
```

The harness experiments (`--list`) cover top-k sweeps, excluding Excel files, the different chunkings, BGE cross-encoder reranking, Gemini query reformulation, multi-query retrieval with reciprocal rank fusion, and a no-RAG "whole corpus in the prompt" baseline. The notebooks in `DAM/` hold the earlier steps: batch Docling conversion, phase-2 chunking and embedding with `CHUNKING = "hierarchical" | "hybrid" | "mixed"`, and query-only RAG over the resulting database. `ComparisonScriptsOctober/trueRAG.ipynb` is a Pinecone-based variant of the same idea with `VARIANT = "full" | "no_excel" | "excel_ocr" | "query_only"`, and `DAM/beg_rag-excel.ipynb` is a single-document BGE + Milvus RAG with `SOURCE = "excel" | "pdf"`.

To try it on the fictional financial statements in `tests/fixtures/dam_docs/`:

```bash
mkdir -p dam_work
python DAM/oneshot_ingestion.py --input-dir tests/fixtures/dam_docs --db dam_work/oneshot.db --collection oneshot
python DAM/visual_grounding.py --db dam_work/oneshot.db -q "Corporate Identity Number (CIN) of company"
python DAM/visual_grounding_enhanced.py --db dam_work/oneshot.db --data-dir tests/fixtures/dam_docs \
  --output-dir dam_work/grounding -q "Authorised capital of the company"

# scoring needs a Gemini key; ground truth is {"questions": [{"id", "question", "answer"}, ...]}
DAM_WORK_DIR=dam_work DAM_GROUND_TRUTH=tests/fixtures/dam_ground_truth.json \
  python DAM/experiment_harness.py --experiment baseline
DAM_WORK_DIR=dam_work python DAM/experiment_harness.py --results
```

`DAM/reindex.py` rebuilds all three chunkings into one database (`DAM_INPUT_DIR` → `DAM_REINDEX_OUTPUT`), and `DAM/analyze_chunk_stats.py --db … --collection …` reports token statistics and plots for a collection.

## Also here: knowledge-graph experiments (`KG/`)

- `doclingConvert.py SOURCE -o out.md` converts a document to Markdown with Docling.
- `tables.py SOURCE -o DIR` saves every table in a document as CSV and HTML.
- `inventory_management.py` parses a built-in fictional bill of sale and loads companies, the invoice and products into Neo4j with `MERGE`, so re-runs do not duplicate anything.
- `REBEL_kggen.ipynb` extracts relation triples from Wikipedia text with the REBEL model.
- `vlm.py` converts a PDF with SmolDocling through MLX, so it only runs on Apple Silicon. Its input file name `policy.pdf` is hard-coded.

## Setup and configuration

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # requirements-dev.txt adds pytest and the fixture builders' packages
cp .env.example .env
```

The first Docling and embedding runs download Docling's layout models and `BAAI/bge-large-en-v1.5` from Hugging Face. A CUDA GPU is used when available.

| Variable | Used by | Notes |
|---|---|---|
| `GEMINI_API_KEY` or `GOOGLE_API_KEY` | `agent.py`, `experiment_harness.py`, notebooks | either name works for the scripts |
| `GEMINI_MODEL` | `agent.py`, `experiment_harness.py` | defaults: `gemini-2.5-flash` (agent), `gemini-2.5-flash-lite` (harness) |
| `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD` | `KG/inventory_management.py` | default URI `bolt://localhost:7687` |
| `DAM_WORK_DIR` | `experiment_harness.py` | Milvus `.db` files, `docling_outputs/`, `experiment_results/`; default `DAM/` |
| `DAM_GROUND_TRUTH` | `experiment_harness.py` | default `$DAM_WORK_DIR/ground_truth.json` |
| `DAM_INPUT_DIR`, `DAM_REINDEX_OUTPUT` | `reindex.py` | defaults `DAM/all_data`, `DAM/reindex_output` |
| `PINECONE_API_KEY`, `PINECONE_INDEX_NAME` | `trueRAG.ipynb` | Pinecone variant only |

Extra requirements for some notebooks:

- The OCR paths in `trueRAG.ipynb` (scanned pages, images and the `excel_ocr` variant) and `pdf_table_extraction_test.ipynb` need the Tesseract binary.
- `trueRAG.ipynb` also needs `pinecone` and `langchain`.
- `pdf_table_extraction_test.ipynb` needs `tabula-py`, which in turn needs Java.

## Tests

```bash
pip install -r requirements-dev.txt
pytest          # 84 offline tests; Gemini, Milvus, Pinecone, OCR and the embedding model are stubbed
pytest -m e2e   # 6 live tests; each is skipped when what it needs is missing
```

The offline suite covers:

- form detection on the demo form
- the gestalt processor
- the agent, with Gemini mocked
- the DAM scripts and the notebook `VARIANT` / `CHUNKING` / `SOURCE` settings, with their heavy parts stubbed
- the inventory parser

The e2e tests run the real Docling + BGE ingestion and retrieval on the fixture documents, the Markdown and table command-line tools, and a double load into Neo4j (needs `NEO4J_URI` and `NEO4J_PASSWORD`). Two of them call Gemini: a fill plan for the demo form and a harness run against `tests/fixtures/dam_ground_truth.json`.

All fixtures are fictional and can be rebuilt with `python tests/fixtures/make_fixtures.py`. No real documents ship with the repo, so bring your own to go beyond the fixtures.

## Status and limitations

These are experiments, not a product. The following has been checked on the fictional fixtures only:

- Form detection finds all four input boxes and all six checkboxes on the demo form and keeps filled-in values as text.
- Ingesting the three fixture documents gives 14 chunks. The CIN query retrieves the right PDF passage on page 1 with its box, and the grounding script highlights it on the page.

Two things are not measured here. For the fill plan, the live test only checks where one value (the name) goes. For RAG answers, the harness exists to score them against your own ground truth, but no results ship with the repo.

Known limitations:

- **Form detection over-reports.** Individual letters become boxes and checkboxes: on the one-page demo form the raw output has 189 elements, including 93 input fields and 80 checkboxes, for 4 real boxes and 6 checkboxes. Underline-style fields, such as the Signature and Date lines, are not reported.
- **Very large agent prompts.** Because the agent sends both forms' full element lists in one prompt, the request for the demo form is about 105k tokens, which can run into free-tier rate limits. The reply is only checked to be valid JSON, not checked against the form.
- **Source boxes only mean something for PDFs.** Word chunks come back with page 0 and an all-zero box, and Excel chunks get a cell range instead of page coordinates.
- **The harness expects databases from other steps.** Only `baseline` and `no_excel` run on the `oneshot.db` from `oneshot_ingestion.py`. The other experiments expect the phase-2 notebook's `milvus_all_docs_mixed.db` / `milvus_all_docs_hybrid.db` in `DAM_WORK_DIR`, and `monolithic` expects `docling_outputs/md/`. The chunk counts in the `--list` descriptions come from an earlier document set, and verdicts come from an LLM judge, not exact match.
- **`reindex.py`'s built-in check strings are specific to an earlier document set.** On the fixtures most of them report MISSING.
- **The notebooks are a lab record.** They point at local folders (`all_data/`, `docs/`) that are not in the repo. Two of them still name Gemini models that are no longer served: `REBEL_kggen.ipynb` uses `gemini-1.5-flash-latest` and `pdf_table_extraction_test.ipynb` uses `gemini-2.0-flash-lite`.
- **Deprecated SDK.** Everything uses Google's deprecated `google-generativeai` package.

## Repository layout

```
phase1/                     form element detection (phase1.py)
prototype1/                 gestalt_processor.py (layout enrichment), agent.py (Gemini fill plan)
checkbox_trials/            visualise_elements.py and checkbox / font-size detection notebooks
DAM/                        Docling → BGE → Milvus Lite RAG: ingestion, reindexing, grounding,
                            experiment harness, chunk statistics, phase-1/phase-2 notebooks
ComparisonScriptsOctober/   trueRAG.ipynb (Pinecone), Excel extraction and PDF table extraction notebooks
KG/                         Docling conversion, table extraction, Neo4j loader, REBEL notebook, vlm.py
tests/                      offline and e2e tests; fixtures/ holds the fictional form and financial documents
```

Licence: not yet specified.
