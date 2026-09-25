# Benchmark results

Two small synthetic benchmarks compare the repo's methods with conventional baselines:

1. **Field extraction (DAM).** Answer 16 AOC-4 style fields from a folder of a company's documents.
2. **Form filling.** Place new values on a blank form, given one filled-in example.

Everything here is synthetic and small. There are 6 fictional companies (96 field instances) and 6 fictional forms (52 fields). Each system ran once. The complete runs use free models: a local gpt-oss:20b for every field-extraction method, and Gemini 3.5 Flash-Lite on the API free tier for the one-call-per-company methods and the form agents. Read the numbers as a description of these documents, not as general accuracy. Every number below comes from a JSON file in `benchmarks/results/`. `python benchmarks/report.py` prints these tables again from those files.

## Summary

- **Field extraction.** With every method on one local model (gpt-oss:20b): BM25 RAG with all fields in one call 89/96, regex rules 86/96, the whole folder in one call 81/96, the repo's Docling + BGE + Milvus pipeline 75/96 batched and 67/96 one call per field, BM25 one call per field 77/96. On `gemini-3.5-flash-lite` (one call per company only): single call 93/96, BM25 91/96, the repo's pipeline 80/96. The repo's pipeline loses at retrieval, whichever model answers: its top-5 passages held the gold evidence for 72 of 94 answerable fields, against 86 for BM25.
- **Form filling.** The repo's `phase1.py` + gestalt step report 39 of 68 target boxes. On that perception the repo's agent (`gemini-3.5-flash-lite`) places 12/52 values and a no-LLM nearest-label rule 28/52. With boxes read from the PDF's vector drawings, the same agent places 44/52 on `gemini-3.5-flash-lite` and 52/52 on `gemini-3.5-flash`, and the rule 52/52 with no model.

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

### Run O: gpt-oss:20b on local Ollama, all six methods

Every method with the same local model, so the per-field methods (the repo's own configuration: one call per field) are compared with the batched and single-call forms like for like.

| Method | Correct (all fields) | Strict | Answerable correct | Declined when not stated | Source file/page correct | Gold evidence in retrieved context | LLM calls | Prompt tokens | Output tokens | LLM time (s) | Ingestion (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dam | 67/96 (70%) | 66/96 (69%) | 66/94 (70%) | 1/2 | 79/94 (84%) | 72/94 (77%) | 96 | 69,084 | 7,022 | 250.1 | 455.4 |
| bm25 | 77/96 (80%) | 77/96 (80%) | 75/94 (80%) | 2/2 | 80/94 (85%) | 86/94 (91%) | 96 | 126,933 | 6,556 | 282.7 | 0.2 |
| dam_batched | 75/96 (78%) | 75/96 (78%) | 74/94 (79%) | 1/2 | 85/94 (90%) | 72/94 (77%) | 6 | 25,031 | 5,469 | 177.8 | 455.4 |
| bm25_batched | 89/96 (93%) | 89/96 (93%) | 89/94 (95%) | 0/2 | 94/94 (100%) | 86/94 (91%) | 6 | 24,023 | 4,404 | 146.2 | 0.2 |
| single_call | 81/96 (84%) | 81/96 (84%) | 80/94 (85%) | 1/2 | – | – | 6 | 23,013 | 1,568 | 63.9 | 0.2 |
| rules | 86/96 (90%) | 86/96 (90%) | 86/94 (91%) | 0/2 | 78/84 (93%) | – | 0 | 0 | 0 | 0.0 | 0.2 |

Per field (correct out of companies scored):

| Field | dam | bm25 | dam_batched | bm25_batched | single_call | rules |
|---|---|---|---|---|---|---|
| cin | 6/6 | 6/6 | 6/6 | 6/6 | 6/6 | 6/6 |
| company_name | 6/6 | 6/6 | 6/6 | 6/6 | 6/6 | 6/6 |
| registered_office | 6/6 | 6/6 | 6/6 | 6/6 | 6/6 | 6/6 |
| email | 6/6 | 5/6 | 6/6 | 5/6 | 5/6 | 4/6 |
| fy_from | 4/6 | 2/6 | 6/6 | 6/6 | 6/6 | 6/6 |
| fy_to | 6/6 | 6/6 | 6/6 | 6/6 | 6/6 | 6/6 |
| board_meeting_date | 0/6 | 6/6 | 4/6 | 6/6 | 6/6 | 6/6 |
| agm_date | 5/6 | 6/6 | 2/6 | 5/6 | 6/6 | 0/6 |
| authorised_capital | 3/6 | 4/6 | 2/6 | 3/6 | 1/6 | 5/6 |
| industry | 4/6 | 5/6 | 4/6 | 6/6 | 6/6 | 6/6 |
| consolidated_fs | 4/6 | 6/6 | 5/6 | 6/6 | 6/6 | 6/6 |
| cag_comments | 3/6 | 2/6 | 4/6 | 6/6 | 5/6 | 6/6 |
| cag_supplementary_audit | 2/6 | 3/6 | 4/6 | 6/6 | 5/6 | 6/6 |
| secretarial_audit | 6/6 | 5/6 | 6/6 | 6/6 | 6/6 | 5/6 |
| revenue | 4/6 | 4/6 | 4/6 | 6/6 | 4/6 | 6/6 |
| profit_after_tax | 2/6 | 5/6 | 4/6 | 4/6 | 1/6 | 6/6 |

What the run shows:

- **Retrieval is the repo pipeline's problem, and it does not depend on the model.** Its top-5 passages hold the gold evidence for 72 of 94 answerable fields, BM25's for 86, the same counts as in run G. With the passages it has, the model answers most of what is answerable: the batched form answers 74 of the 94 answerable fields, more than the 72 whose evidence was in their own top-5, because a field can be answered from a passage retrieved for another one. The misses follow the retrieval: `board_meeting_date` 0/6 per field (the sentence sits in a directors' report chunk that the BGE query never ranks in the top 5), `cag_supplementary_audit` 2/6, `profit_after_tax` 2/6.
- **One call per field is worse than one call per company for both retrievers** (67 → 75 for the repo pipeline, 77 → 89 for BM25), at sixteen times the calls and three to five times the prompt tokens. The union of the passages retrieved for all sixteen fields covers what a single field's top-5 misses, and the model can answer a field from a passage retrieved for another one.
- **Batched BM25 beats the whole folder in one prompt on this model** (89 against 81), the reverse of run G (91 against 93). The single call loses on the two amount fields (`authorised_capital` 1/6, `profit_after_tax` 1/6): asked to convert "Rs. 25 crore" or a lakhs figure to rupees inside a 4,000-token prompt, gpt-oss:20b slips where Gemini did not. Given the shorter retrieved context it converts correctly. The rules, which parse the amount after "authorised" directly, get 5/6 and 6/6.
- **The rules are competitive** (86/96) because the documents were generated from templates the rules were written against; see the caveats. They miss `agm_date` on every company (the date is stated in a sentence the regex does not cover) and never decline the two unanswerable fields.
- **Cost.** Ingestion for the repo pipeline (Docling, BGE-large and Milvus Lite on the CPU) took 455 s for six companies against 0.2 s for the plain-text BM25 index. Per query, the batched methods make 6 calls and use 24,000–25,000 prompt tokens in total; the per-field methods make 96 calls and use 69,000–127,000. Every LLM figure is from a 12 GB GPU that held about 80% of the model, so the times are indicative only.

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

### Results: agents on `gemini-3.5-flash-lite` (complete) and `gemini-3.5-flash` (vector arm only), geometry offline

`gemini-3.5-flash-lite`, all four methods:

| Form | agent | agent_vector | geometric_phase1 | geometric_vector |
|---|---|---|---|---|
| f1_labels_left | 2/7 | 7/7 | 4/7 | 7/7 |
| f2_labels_above | 0/6 | 6/6 | 1/6 | 6/6 |
| f3_two_column | 9/10 | 10/10 | 10/10 | 10/10 |
| f4_yes_no_rows | 0/8 | 8/8 | 7/8 | 8/8 |
| f5_ruled_grid | 0/9 | 2/9 | 0/9 | 9/9 |
| f6_comb_kyc | 1/12 | 11/12 | 6/12 | 12/12 |
| **all** | **12/52 (23%)** | **44/52 (85%)** | **28/52 (54%)** | **52/52 (100%)** |
| checkbox fields | 1/12 (8%) | 12/12 (100%) | 9/12 (75%) | 12/12 (100%) |
| comb fields | 1/3 (33%) | 2/3 (67%) | 3/3 (100%) | 3/3 (100%) |
| text fields | 10/37 (27%) | 30/37 (81%) | 16/37 (43%) | 37/37 (100%) |

| Method | LLM calls | Prompt tokens | Output tokens | Thinking tokens | LLM time (s) |
|---|---|---|---|---|---|
| agent | 6 | 676,413 | 4,278 | 0 | 813.4 |
| agent_vector | 6 | 138,932 | 4,200 | 0 | 20.0 |

`gemini-3.5-flash`, the agent with vector perception only (its free daily quota was exhausted before the phase1 arm could run, on two keys and on two days):

| Form | agent_vector |
|---|---|
| f1_labels_left | 7/7 |
| f2_labels_above | 6/6 |
| f3_two_column | 10/10 |
| f4_yes_no_rows | 8/8 |
| f5_ruled_grid | 9/9 |
| f6_comb_kyc | 12/12 |
| **all** | **52/52 (100%)** |

| Method | LLM calls | Prompt tokens | Output tokens | Thinking tokens | LLM time (s) |
|---|---|---|---|---|---|
| agent_vector | 6 | 138,932 | 4,240 | 0 | 872.7 |

Target boxes reported by the perception step (IoU > 0.5), the same for both runs:

| Form | phase1 + gestalt | vector drawings |
|---|---|---|
| f1_labels_left | 8/11 (160 elements) | 11/11 (27 elements) |
| f2_labels_above | 3/8 (141 elements) | 8/8 (20 elements) |
| f3_two_column | 11/12 (152 elements) | 12/12 (29 elements) |
| f4_yes_no_rows | 11/12 (205 elements) | 12/12 (31 elements) |
| f5_ruled_grid | 0/11 (73 elements) | 11/11 (36 elements) |
| f6_comb_kyc | 6/14 (198 elements) | 14/14 (33 elements) |

What the runs show:

- **Perception is the bottleneck on these forms, not placement.** `phase1.py` + `gestalt_processor.py` report only 39 of the 68 target boxes. It misses three kinds:
  - boxes more than 10 times as wide as they are tall (the shape filter requires an aspect ratio below 10), so 250–400 pt wide name and address boxes are missed
  - everything inside a ruled table (only outermost contours are kept, so f5 yields one box for the whole table)
  - 9 pt checkboxes: they are reported as input fields, and the gestalt step then merges them with letter fragments of the option text
- **The repo's own configuration places 12 of 52 values.** Given the phase1 elements, the agent does worse than the nearest-label rule on the same elements (28/52). The prompt holds 140–205 elements per form, most of them letter fragments and merged boxes, and runs to 113,000 tokens on average; the model then places text in the wrong box (10/37 text fields) and ticks the wrong or no checkbox (1/12). Where the box for a field was not reported, no placement method can succeed, and every one of the rule's 24 misses is such a field.
- **With perception from the PDF's vector drawings** the unchanged agent places 44 of 52 on `gemini-3.5-flash-lite` and 52 of 52 on `gemini-3.5-flash`; the geometric rule places 52 of 52 with no model and under 10 ms per form, perception included. The smaller model's misses are the ruled grid (2/9: it puts values in the header row) and one comb field. The agent needed 6 calls and about 139,000 prompt tokens in either run.
- The vector perception, the geometric rule and the six layouts were all written by the same person, knowing the layouts. The rule's 52/52 shows that these forms are easy once the boxes are known; it does not show that the rule generalises. Scanned forms have no vector drawings at all.

## Run conditions

The runs were made on 24–26 September 2026.

**Machine.** Intel Core i7-13700KF (24 threads), 31 GB RAM, an RTX 4070 SUPER with 12 GB. Docling and BGE ran on the CPU in every run. The Gemini runs of 24–25 September shared the machine with other benchmark jobs (load average 20–37); the gpt-oss run and the form runs of 26 September had it to themselves. Read every time in this file as indicative only. Token counts are the better measure of cost.

**Software.**

- Python 3.12.12
- docling 2.70.0 (docling-core 2.61.0)
- sentence-transformers 6.1.0, torch 2.5.1
- pymilvus 2.6.11, milvus-lite 2.5.1
- PyMuPDF 1.26.7, opencv 4.12.0
- pypdf 6.9.2, python-docx 1.2.0, openpyxl 3.1.5, reportlab 4.4.10
- google-generativeai 0.8.6
- Ollama 0.12.5 serving `gpt-oss:20b` (21B parameters, 3.6B active, MXFP4, digest 17052f91) with a context of 16,384 tokens, temperature 0, seed 0 and `think: low`, about 80% of the weights on the GPU. The largest prompt is under 6k tokens, and no prompt reached the limit. Ollama's JSON grammar mode is not used for this model: the reasoning leaked into the answer and the JSON came out mangled, so the model gets the plain prompt and its fenced JSON is parsed (`llm_client.py`).

**Retrieval settings.** Top-k is 5. BM25 uses k1 = 1.5 and b = 0.75. BM25 chunks are 1,000 characters with 200 overlap. BGE queries are the form labels, without an instruction prefix, as in the repo.

**Gemini models (free tier).** Availability forced the choice:

- `gemini-2.5-flash-lite` (the harness default) answered 404: "no longer available to new users".
- `gemini-2.5-flash` (the agent default) answered 429 (quota exceeded) on its first request.
- Task 1 therefore uses `gemini-3.5-flash-lite`. Task 2 used `gemini-3.5-flash` on 24 September; on 26 September that model answered 429 on two keys before any request, so the complete Task 2 run is on `gemini-3.5-flash-lite`.

In total the runs made 44 Gemini requests. 36 succeeded: 30 on `gemini-3.5-flash-lite` (18 for Task 1, 12 for Task 2) and 6 on `gemini-3.5-flash`. 8 failed: one 404, six 429s and one client deadline. The Task 1 Gemini run used temperature 0. The agent keeps the model's default temperature, as `prototype1/agent.py` does. Each configuration ran once.

**Caching.** Every LLM response is stored in `benchmarks/.cache/` (git-ignored), keyed by provider, model, settings and prompt. `--offline` re-scores from the cache without calling a model. After the Gemini pass, one date in the generator (c4's AGM) was changed. c4's documents were rebuilt before either run read them, and c4's Gemini scores were recomputed from the cached responses (`--resume --redo --offline`); no request was repeated. Run O was first started with Ollama's JSON grammar mode on; every answer came back mangled, so those responses were discarded, the mode was switched off for reasoning models and the run was repeated from scratch. The committed file holds only the repeated run.

## Caveats

- **Small and synthetic.** There are 96 field instances and 52 form fields, and every configuration ran once, so there are no confidence intervals. A gap of a few fields could come from rewording a single template.
- **Short, born-digital documents.** Each company has about 12,000 characters of clean text with a proper text layer and no scans. pypdf extracts it perfectly, so Docling's layout and OCR strengths cannot show. Long-document effects are not tested: with real annual reports of 100+ pages, retrieval matters more and a single call costs far more tokens, or does not fit a local context window at all.
- **Baselines written by the benchmark's author.** The same person wrote the generators, the rules, the vector perception and the geometric filler, knowing the templates and layouts. The DAM rules were written against a first version of the documents. The boilerplate sections (accounting policies, committees, the dividend note, …) were added afterwards and the rules were not changed; that change cost them the AGM dates. The rules and the geometric filler are still an optimistic picture of what hand-written rules achieve.
- **The answer prompt is not the harness's.** It adds a passage citation and JSON output. The harness's LLM judge is not used: answers are matched after normalisation, and text fields accept a token F1 of at least 0.8.
- **Defaults only for the repo's pipeline.** It ran with top-5 retrieval and no reranking, query reformulation or chunking changes, all of which the harness offers. BM25 was not tuned either.
- **Models.** The local run uses gpt-oss:20b, a 21B mixture-of-experts model; a 3B model tried first (llama3.2) could not follow the answer format and its run was abandoned. On Gemini only the one-call-per-company methods ran, so run G and run O compare a model as much as a method. The Gemini models differ from the repo's defaults because of availability on the day, and Task 2's two agent runs are on two different Gemini models.
- **Coarse source check.** For Word and Excel files the source is checked at file level only. For Yes/No fields the source is the file that holds the deciding sentence.
- **Form agent.** The agent ran with the model's default temperature and one sample per form. The arm with the repo's own perception ran on `gemini-3.5-flash-lite` only; on `gemini-3.5-flash` it is still pending.

## Reproduce, resume

Rebuild the data. The DAM files come out byte-identical.

```bash
pip install -r requirements-dev.txt
python benchmarks/generate_dam_benchmark.py
python benchmarks/generate_form_benchmark.py
```

Run O, with a local Ollama serving `gpt-oss:20b`:

```bash
OLLAMA_HOST=http://127.0.0.1:11434 CUDA_VISIBLE_DEVICES="" python benchmarks/run_dam_benchmark.py \
  --run local-gpt-oss-20b --provider ollama --model gpt-oss:20b --num-ctx 16384 --think low
```

Run G, with `GEMINI_API_KEY` or `GOOGLE_API_KEY` in the environment:

```bash
python benchmarks/run_dam_benchmark.py --run gemini-3.5-flash-lite --provider gemini \
  --model gemini-3.5-flash-lite --methods rules,single_call,dam_batched,bm25_batched \
  --min-interval 10 --daily-cap 60 --work-dir benchmarks/work/gemini
```

Task 2, then print the tables:

```bash
python benchmarks/run_form_benchmark.py --run gemini-3.5-flash-lite --model gemini-3.5-flash-lite \
  --min-interval 65 --timeout 1800
python benchmarks/run_form_benchmark.py --run gemini-3.5-flash --model gemini-3.5-flash \
  --methods agent_vector,geometric_phase1,geometric_vector --min-interval 65 --timeout 1800
python benchmarks/report.py
```

**Still to run.** The `agent` arm of Task 2 on `gemini-3.5-flash` (6 requests with prompts of 150–400k characters), once that model's free quota allows it. It adds its results to `forms_gemini-3.5-flash.json`:

```bash
python benchmarks/run_form_benchmark.py --run gemini-3.5-flash --methods agent --model gemini-3.5-flash \
  --min-interval 65 --timeout 1800 --max-live-calls 6 --resume
```

Both runners save partial results when a quota or provider error stops them. Re-running with `--resume` finishes the rest without repeating any finished request. The cache is not committed, so a fresh run elsewhere makes new LLM calls. Its numbers can differ: Gemini models change over time, and the agent samples at the default temperature.
