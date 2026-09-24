# Benchmark results

Two small synthetic benchmarks compare the repo's methods with conventional baselines:

1. **Field extraction (DAM).** Answer 16 AOC-4 style fields from a folder of a company's documents.
2. **Form filling.** Place new values on a blank form, given one filled-in example.

Everything here is synthetic and small. There are 6 fictional companies (96 field instances) and 6 fictional forms (52 fields). Each system ran once. Read the numbers as a description of these documents, not as general accuracy. Every number below comes from a JSON file in `benchmarks/results/`. `python benchmarks/report.py` prints these tables again from those files.

## Summary

- **Field extraction, `gemini-3.5-flash-lite`, one call per company.** The whole-folder single call gets 93/96, BM25 RAG 91/96, the repo's Docling + BGE + Milvus pipeline 80/96 and regex rules 86/96. The repo's pipeline loses at retrieval: its top-5 passages held the gold evidence for 72 of 94 answerable fields, against 86 for BM25.
- **Form filling.** The repo's `phase1.py` + gestalt step report 39 of 68 target boxes. With that perception, a no-LLM nearest-label rule places 28/52 values. With boxes read from the PDF's vector drawings, the unchanged agent (`gemini-3.5-flash`) and the rule both place 52/52. The agent on phase1 output is pending (free quota).

## Task 1: AOC-4 field extraction

### Data (`generate_dam_benchmark.py` → `data/dam/`)

Six fictional listed companies. Each has three documents written from a ground-truth record in the script:

- `financial_statements.pdf`, 6 pages: cover, balance sheet, profit and loss, notes (corporate information, share capital, approval), accounting policies, and related-party and other notes
- `directors_report.docx`: board meetings, subsidiaries, secretarial audit, CAG comments where relevant, the AGM, investor contact, and the usual boilerplate sections
- `trial_balance.xlsx`: a company master sheet plus current-year and prior-year trial balances

Each company has about 12,000 characters of text (3–4k tokens).

The 16 fields are CIN, name, registered office, e-mail, financial year from and to, the board meeting that approved the statements, AGM date, authorised capital, industry, consolidated statements (Y/N), CAG comments (Y/N), CAG supplementary audit (Y/N), secretarial audit (Y/N), revenue and profit after tax.

Facts are spread across the files. For example, the AGM date is only in the report, and one company's e-mail is only in the workbook. The data varies on purpose:

- Amounts are in rupees, lakhs, crores or millions.
- Date formats differ between companies.

The documents also contain distractors:

- prior-year figures
- an increase in authorised capital during the year
- earlier board meeting and AGM dates
- a holding company's CIN
- a share registrar's e-mail
- a subsidiary for which secretarial audit "is not applicable"
- a dividend note that mentions the "ensuing Annual General Meeting" next to the board date

Two field instances are stated nowhere (gold `null`):

- c4's e-mail: only the registrar's e-mail is present
- c6's AGM date: "will be communicated", with the previous AGM date given

The generator checks every recorded (file, page, text) location against text extracted from the files. Rebuilding gives byte-identical files. CINs use the non-existent state code `ZZ`.

### Systems

| Method | Ingestion and retrieval | LLM calls |
|---|---|---|
| `dam` | The repo's code: `DAM/oneshot_ingestion.py` (Docling → HierarchicalChunker → BAAI/bge-large-en-v1.5 → Milvus Lite), then `DAM/experiment_harness.retrieve_from_milvus`, top 5, with the form label as the query | one per field |
| `bm25` | Baseline A, basic RAG: pypdf / python-docx / openpyxl text, 1,000-character chunks with 200 overlap, Okapi BM25, top 5 | one per field |
| `dam_batched`, `bm25_batched` | Same retrieval, but all 16 fields and the union of their passages go into one prompt | one per company |
| `single_call` | Baseline B: all extracted text of the company in one prompt, all fields as JSON | one per company |
| `rules` | Baseline C: regex and keyword rules (CIN pattern, e-mail, date near keywords, amount after "authorised", …) | none |

`dam` and `bm25` share one prompt: the harness's answer instructions ("use only the context", "UNANSWERABLE" if absent), plus a request to cite the passage number, with JSON output. Every prompt states the answer format, including "convert lakhs or crores to rupees". Temperature is 0.

### Scoring (`scoring.py`)

- **Correct.** The answer matches after normalising for the field type: CIN, e-mail, date (any common format, day first), amount (Indian or international grouping, lakh / crore / million), or Yes/No. For text fields (name, address, industry), a token F1 of at least 0.8 also counts. **Strict** means an exact normalised match. When the gold value is `null`, only a refusal counts as correct. No LLM judge is used.
- **Source correct.** The cited passage's file (and page, for PDFs) is one of the places that state the gold value. If nothing is cited, the top-ranked passage is used. Word and Excel files are checked at file level only.
- **Gold evidence in context.** At least one passage retrieved for the field contains the text that states the gold value. This measures retrieval alone. In the batched methods the model sees the passages retrieved for every field, so it can answer from another field's passage.

### Run L: `llama3.2` on local Ollama, all six methods

Run L is still in progress. Its results will be added here, and until then it can be finished with the Run L command under "Reproduce, resume" plus `--resume`.

### Run G: `gemini-3.5-flash-lite` (Google AI Studio free tier), one call per company per method

The per-field methods were not run on Gemini: 6 companies × 16 fields × 2 methods would take 192 requests, far above the free tier's 20 requests per model per day.

| Method | Correct (all fields) | Strict | Answerable correct | Declined when not stated | Source file/page correct | Gold evidence in retrieved context | LLM calls | Prompt tokens | Output tokens | LLM time (s) | Ingestion (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dam_batched | 80/96 (83%) | 80/96 (83%) | 78/94 (83%) | 2/2 | 85/94 (90%) | 72/94 (77%) | 6 | 29,222 | 2,838 | 61.2 | 1,043.0 |
| bm25_batched | 91/96 (95%) | 91/96 (95%) | 89/94 (95%) | 2/2 | 88/94 (94%) | 86/94 (91%) | 6 | 26,731 | 2,822 | 67.7 | 0.9 |
| single_call | 93/96 (97%) | 93/96 (97%) | 92/94 (98%) | 1/2 | – | – | 6 | 26,307 | 1,722 | 65.2 | 0.9 |
| rules | 86/96 (90%) | 86/96 (90%) | 86/94 (91%) | 0/2 | 78/84 (93%) | – | 0 | 0 | 0 | 0.0 | 0.9 |

Per field (correct out of companies scored):

| Field | dam_batched | bm25_batched | single_call | rules |
|---|---|---|---|---|
| cin | 6/6 | 6/6 | 6/6 | 6/6 |
| company_name | 6/6 | 6/6 | 6/6 | 6/6 |
| registered_office | 6/6 | 6/6 | 6/6 | 6/6 |
| email | 6/6 | 6/6 | 5/6 | 4/6 |
| fy_from | 6/6 | 6/6 | 6/6 | 6/6 |
| fy_to | 6/6 | 6/6 | 6/6 | 6/6 |
| board_meeting_date | 4/6 | 6/6 | 6/6 | 6/6 |
| agm_date | 6/6 | 6/6 | 6/6 | 0/6 |
| authorised_capital | 4/6 | 5/6 | 6/6 | 5/6 |
| industry | 4/6 | 6/6 | 6/6 | 6/6 |
| consolidated_fs | 4/6 | 6/6 | 6/6 | 6/6 |
| cag_comments | 3/6 | 5/6 | 6/6 | 6/6 |
| cag_supplementary_audit | 3/6 | 3/6 | 6/6 | 6/6 |
| secretarial_audit | 6/6 | 6/6 | 6/6 | 5/6 |
| revenue | 6/6 | 6/6 | 5/6 | 6/6 |
| profit_after_tax | 4/6 | 6/6 | 5/6 | 6/6 |

LLM time is the sum of request latencies measured from this machine. Ingestion is summed over the six companies. For `dam_batched` it is Docling conversion (15–26 s per company) plus BGE-large embedding on a CPU shared with other jobs (109–202 s per company for 43–46 chunks). For the other methods it is plain text extraction (under 0.4 s per company).

What the run shows:

- The single call and BM25 RAG beat the repo's pipeline on these documents: 93 and 91 of 96, against 80 for `dam_batched`. They also beat the regex rules (86).
- **Retrieval is where the repo's pipeline loses.** For 22 of 94 answerable fields, the passages retrieved for that field did not contain the gold evidence; for BM25 it was 8. Looking at the Docling chunks shows why:
  - The cover-page title and the section headings are labelled title / section header, so they are not part of any chunk's text.
  - The numbered notes on page 4 (corporate information, share capital, approval of the statements) come out as one long `list_item` chunk.
  - That long chunk ranks below short, board-related paragraphs for queries such as the board meeting date.
  - The one-line note "not a Government company", which settles the two CAG fields, is part of that long chunk and is rarely retrieved for the CAG query.
- The other errors fall into three groups:
  - **Retrieval.** 11 of the 16 errors of `dam_batched` are "UNANSWERABLE" answers where the passage was never retrieved, and 2 more are a prior-year board date.
  - **CAG refusals.** 4 of the 5 errors of `bm25_batched` are "UNANSWERABLE" on the CAG fields, sometimes with the "not a Government company" note in the context.
  - **Amounts misread.** Lakhs or Indian digit grouping are converted with errors of a factor of 10 or 100: 2 of the 3 errors of the single call, 3 of `dam_batched` and 1 of `bm25_batched`.
- The single call accepted the registrar's e-mail as the company's e-mail (c4, gold `null`). The two retrieval methods declined.
- The rules miss every AGM date (the dividend note puts the board date next to the words "Annual General Meeting"). They are also caught by the registrar's e-mail (2), the subsidiary's "not applicable" sentence (1) and the prior-year figure in "increased from … to …" (1). They get every CIN, financial-year date, board date, revenue, profit and CAG answer right.
- On documents this short (about 4.4k tokens per company), the batched RAG prompts are as long as the whole-folder prompt (4.2–5.1k tokens per call for all three methods). Retrieval saves no tokens here.


## Task 2: form filling

### Data (`generate_form_benchmark.py` → `data/forms/`)

There are six one-page fictional forms, each with an empty version, a filled "golden sample" and a `new_data.json`. The 52 fields are 37 text fields, 12 checkbox groups and 3 character-cell ("comb") fields. Target boxes are recorded in PDF points with a top-left origin, as PyMuPDF and `phase1.py` report them.

| Form | Layout |
|---|---|
| `f1_labels_left` | label left of each box; checkbox before its option text |
| `f2_labels_above` | label above each 400 pt wide box |
| `f3_two_column` | two columns of label/box pairs |
| `f4_yes_no_rows` | questionnaire rows `Yes [ ]  No [ ]` (option text before its box) |
| `f5_ruled_grid` | ruled table: label cells and value cells share grid lines |
| `f6_comb_kyc` | labels above, comb fields for date of birth / PIN / tax ID, 9 pt checkboxes |

### Systems

| Method | Perception | Placement |
|---|---|---|
| `agent` | the repo's `phase1.py` (PyMuPDF text + OpenCV shapes) → `gestalt_processor.py` on the empty and golden forms | `prototype1/agent.py`, unchanged: its prompt, JSON mode, model default temperature |
| `agent_vector` | PDF text spans + vector drawings (rectangles, ruled-table cells, merged comb cells) → `gestalt_processor.py` | `prototype1/agent.py`, unchanged |
| `geometric_phase1` | as `agent` (empty form only) | nearest-label geometry, no LLM, golden sample not used |
| `geometric_vector` | as `agent_vector` (empty form only) | nearest-label geometry, no LLM, golden sample not used |

Nearest-label geometry (`geometric_filler.py`) works in three steps:

1. Match each data key to the most similar label by word overlap.
2. For a checkbox group, find the option text equal to the value on the label's row (or just below it) and tick the closest checkbox on that row.
3. Otherwise take the nearest unused box to the right of the label on the same row, or below it.

It drops shapes that sit inside a text span (letters the shape detector reported as boxes) and shapes that contain text (label cells).

**Scoring.** A text value counts when a plan entry with that text has its centre inside the target box (2 pt tolerance). A checkbox group counts when the chosen option's box, and no other option's box, holds a tick. A comb field counts when the whole value, or one character per cell, lies inside it.

### Results: agents on `gemini-3.5-flash`, geometry offline

| Form | agent | agent_vector | geometric_phase1 | geometric_vector |
|---|---|---|---|---|
| f1_labels_left | pending | 7/7 | 4/7 | 7/7 |
| f2_labels_above | pending | 6/6 | 1/6 | 6/6 |
| f3_two_column | pending | 10/10 | 10/10 | 10/10 |
| f4_yes_no_rows | pending | 8/8 | 7/8 | 8/8 |
| f5_ruled_grid | pending | 9/9 | 0/9 | 9/9 |
| f6_comb_kyc | pending | 12/12 | 6/12 | 12/12 |
| **all** | **–** | **52/52 (100%)** | **28/52 (54%)** | **52/52 (100%)** |
| checkbox fields | – | 12/12 (100%) | 9/12 (75%) | 12/12 (100%) |
| comb fields | – | 3/3 (100%) | 3/3 (100%) | 3/3 (100%) |
| text fields | – | 37/37 (100%) | 16/37 (43%) | 37/37 (100%) |

| Method | LLM calls | Prompt tokens | Output tokens | Thinking tokens | LLM time (s) |
|---|---|---|---|---|---|
| agent_vector | 6 | 138,932 | 4,240 | 0 | 872.7 |

Target boxes reported by the perception step (IoU > 0.5):

| Form | phase1 + gestalt | vector drawings |
|---|---|---|
| f1_labels_left | 8/11 (160 elements) | 11/11 (27 elements) |
| f2_labels_above | 3/8 (141 elements) | 8/8 (20 elements) |
| f3_two_column | 11/12 (152 elements) | 12/12 (29 elements) |
| f4_yes_no_rows | 11/12 (205 elements) | 12/12 (31 elements) |
| f5_ruled_grid | 0/11 (73 elements) | 11/11 (36 elements) |
| f6_comb_kyc | 6/14 (198 elements) | 14/14 (33 elements) |

`agent` (the repo's own configuration) has **no results yet**. The first call, on f1 (305k characters of prompt), hit the SDK's default 600 s deadline. The client deadline was then raised to 1,800 s. By then the free-tier limit of 20 requests per day for `gemini-3.5-flash` had been used up; this benchmark made 7 of those requests, and the key is shared with other work. `gemini-2.5-flash` (the agent's default), `gemini-3.7-flash` and `gemini-3.8-flash` also answered 429 on their first request. The resume command is below. Until it runs, the benchmark says nothing about how well the agent does on phase1 output.

What the run shows:

- **Perception is the bottleneck on these forms, not placement.** `phase1.py` + `gestalt_processor.py` report only 39 of the 68 target boxes. It misses three kinds:
  - boxes more than 10 times as wide as they are tall (the shape filter requires an aspect ratio below 10), so 250–400 pt wide name and address boxes are missed
  - everything inside a ruled table (only outermost contours are kept, so f5 yields one box for the whole table)
  - 9 pt checkboxes: they are reported as input fields, and the gestalt step then merges them with letter fragments of the option text
- Nearest-label geometry on the same phase1 elements places 28 of 52 values. Every miss is a field whose box was not reported. Where the box is present, the simple rule places the value correctly.
- With perception from the PDF's vector drawings, both the unchanged agent and the geometric rule place **52 of 52**. The agent needed 6 LLM calls and about 139k prompt tokens; the rule needed no model and under 10 ms per form, perception included.
- The vector perception, the geometric rule and the six layouts were all written by the same person, knowing the layouts. The rule's 52/52 shows that these forms are easy once the boxes are known; it does not show that the rule generalises. Scanned forms have no vector drawings at all.


## Run conditions

The runs were made on 24–25 September 2026.

**Machine.** Intel Core i7-13700KF (24 threads), 31 GB RAM, no GPU (Docling and BGE ran on the CPU). Other benchmark jobs, including other Ollama servers, shared the machine throughout, with a load average of 20–37. Read every time in this file as indicative only. Token counts are the better measure of cost.

**Software.**

- Python 3.12.12
- docling 2.70.0 (docling-core 2.61.0)
- sentence-transformers 6.1.0, torch 2.5.1
- pymilvus 2.6.11, milvus-lite 2.5.1
- PyMuPDF 1.26.7, opencv 4.12.0
- pypdf 6.9.2, python-docx 1.2.0, openpyxl 3.1.5, reportlab 4.4.10
- google-generativeai 0.8.6
- Ollama 0.12.5 serving `llama3.2` (3.2B, Q4_K_M) with a context of 8,192 tokens, temperature 0 and seed 0 (the largest prompt is under 6k tokens, and no prompt reached the limit)

**Retrieval settings.** Top-k is 5. BM25 uses k1 = 1.5 and b = 0.75. BM25 chunks are 1,000 characters with 200 overlap. BGE queries are the form labels, without an instruction prefix, as in the repo.

**Gemini models (free tier).** Availability forced the choice:

- `gemini-2.5-flash-lite` (the harness default) answered 404: "no longer available to new users".
- `gemini-2.5-flash` (the agent default) answered 429 (quota exceeded) on its first request.
- Task 1 therefore uses `gemini-3.5-flash-lite` and Task 2 uses `gemini-3.5-flash`.

In total the runs made 30 Gemini requests. 24 succeeded: 18 on `gemini-3.5-flash-lite` and 6 on `gemini-3.5-flash`. 6 failed: one 404, four 429s and one client deadline. The Task 1 Gemini run used temperature 0. The agent keeps the model's default temperature, as `prototype1/agent.py` does. Each configuration ran once.

**Caching.** Every LLM response is stored in `benchmarks/.cache/` (git-ignored), keyed by provider, model, settings and prompt. `--offline` re-scores from the cache without calling a model. After the Gemini pass, one date in the generator (c4's AGM) was changed. c4's documents were rebuilt before either run read them, and c4's Gemini scores were recomputed from the cached responses (`--resume --redo --offline`); no request was repeated. Run L was first started with a 16,384-token context. With other jobs holding most of the memory, its Ollama runner exited in the middle of a request and then could not be reloaded (HTTP 500), so the whole run was repeated from scratch with an 8,192-token context. The committed file holds only the 8k run.

## Caveats

- **Small and synthetic.** There are 96 field instances and 52 form fields, and every configuration ran once, so there are no confidence intervals. A gap of a few fields could come from rewording a single template.
- **Short, born-digital documents.** Each company has about 12,000 characters of clean text with a proper text layer and no scans. pypdf extracts it perfectly, so Docling's layout and OCR strengths cannot show. Long-document effects are not tested: with real annual reports of 100+ pages, retrieval matters more and a single call costs far more tokens, or does not fit a local context window at all.
- **Baselines written by the benchmark's author.** The same person wrote the generators, the rules, the vector perception and the geometric filler, knowing the templates and layouts. The DAM rules were written against a first version of the documents. The boilerplate sections (accounting policies, committees, the dividend note, …) were added afterwards and the rules were not changed; that change cost them the AGM dates. The rules and the geometric filler are still an optimistic picture of what hand-written rules achieve.
- **The answer prompt is not the harness's.** It adds a passage citation and JSON output. The harness's LLM judge is not used: answers are matched after normalisation, and text fields accept a token F1 of at least 0.8.
- **Defaults only for the repo's pipeline.** It ran with top-5 retrieval and no reranking, query reformulation or chunking changes, all of which the harness offers. BM25 was not tuned either.
- **Models.** `llama3.2` is a 3B model, which pulls down every LLM-based method in run L. On Gemini only the one-call-per-company methods ran. The Gemini models differ from the repo's defaults because of availability on the day.
- **Coarse source check.** For Word and Excel files the source is checked at file level only. For Yes/No fields the source is the file that holds the deciding sentence.
- **Form agent.** The agent ran with the model's default temperature and one sample per form. The arm with the repo's own perception has not run yet.

## Reproduce, resume

Rebuild the data. The DAM files come out byte-identical.

```bash
pip install -r requirements-dev.txt
python benchmarks/generate_dam_benchmark.py
python benchmarks/generate_form_benchmark.py
```

Run L, with a local Ollama serving `llama3.2`:

```bash
OLLAMA_HOST=http://127.0.0.1:11434 python benchmarks/run_dam_benchmark.py --run local-llama3.2 \
  --provider ollama --model llama3.2 --num-ctx 8192
```

Run G, with `GEMINI_API_KEY` or `GOOGLE_API_KEY` in the environment:

```bash
python benchmarks/run_dam_benchmark.py --run gemini-3.5-flash-lite --provider gemini \
  --model gemini-3.5-flash-lite --methods rules,single_call,dam_batched,bm25_batched \
  --min-interval 10 --daily-cap 60 --work-dir benchmarks/work/gemini
```

Task 2, then print the tables:

```bash
python benchmarks/run_form_benchmark.py --run gemini-3.5-flash --model gemini-3.5-flash \
  --min-interval 65 --timeout 1800
python benchmarks/report.py
```

**Still to run.** Run the `agent` arm of Task 2 once the `gemini-3.5-flash` free quota has reset (6 requests with prompts of 150–400k characters). It adds its results to `forms_gemini-3.5-flash.json`:

```bash
python benchmarks/run_form_benchmark.py --run gemini-3.5-flash --methods agent --model gemini-3.5-flash \
  --min-interval 65 --timeout 1800 --max-live-calls 6 --resume
```

Both runners save partial results when a quota or provider error stops them. Re-running with `--resume` finishes the rest without repeating any finished request. The cache is not committed, so a fresh run elsewhere makes new LLM calls. Its numbers can differ: Gemini models change over time, and the agent samples at the default temperature.
