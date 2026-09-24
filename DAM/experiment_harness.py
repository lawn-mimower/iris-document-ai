#!/usr/bin/env python3
"""
RAG Experiment Harness for Blog Part 1
=======================================
Runs profiling questions against different RAG configurations,
logs everything, and produces a comparison table.

Usage:
    # Activate environment first:
    # source ~/anaconda3/bin/activate ml-env

    # Run a single experiment:
    python experiment_harness.py --experiment baseline

    # List available experiments:
    python experiment_harness.py --list

    # Show results so far:
    python experiment_harness.py --results
"""

import json
import os
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, List

# ── Config ──────────────────────────────────────────────────────

# Folder holding the Milvus databases, docling markdown exports and results
# (defaults to this directory; override with DAM_WORK_DIR)
WORK_DIR = Path(os.getenv("DAM_WORK_DIR", Path(__file__).parent))
GROUND_TRUTH_PATH = Path(os.getenv("DAM_GROUND_TRUTH", WORK_DIR / "ground_truth.json"))
RESULTS_DIR = WORK_DIR / "experiment_results"
RESULTS_DIR.mkdir(exist_ok=True)

# Milvus databases (already built)
MILVUS_DBS = {
    "baseline":     ("oneshot.db",               "oneshot",             5086),
    "mixed":        ("milvus_all_docs_mixed.db",  "rag_all_docs_mixed", 3813),
    "hybrid":       ("milvus_all_docs_hybrid.db", "rag_all_docs_hybrid", 28554),
    "oneshot":      ("oneshot.db",                "oneshot",             5086),
}

EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
EMBEDDING_DIM = 1024

# Gemini config
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

# ── Lazy globals ────────────────────────────────────────────────

_embedding_model = None
_gemini_client = None


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading {EMBEDDING_MODEL} on {device}...")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL, device=device)
        print(f"  Loaded. Dim={_embedding_model.get_sentence_embedding_dimension()}")
    return _embedding_model


def get_gemini():
    global _gemini_client
    if _gemini_client is None:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).parent / ".env")
        load_dotenv()
        import google.generativeai as genai
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("No GEMINI_API_KEY or GOOGLE_API_KEY found in environment")
        genai.configure(api_key=api_key)
        _gemini_client = genai.GenerativeModel(GEMINI_MODEL)
        print(f"Gemini model: {GEMINI_MODEL}")
    return _gemini_client


def load_ground_truth():
    if not GROUND_TRUTH_PATH.exists():
        print(f"ERROR: Ground truth file not found at {GROUND_TRUTH_PATH}")
        print("Create it first with the ground truth extraction.")
        sys.exit(1)
    with open(GROUND_TRUTH_PATH) as f:
        return json.load(f)


# ── Retrieval Functions ─────────────────────────────────────────

def retrieve_from_milvus(query: str, db_path: str, collection: str,
                         top_k: int = 5, exclude_excel: bool = False) -> list:
    """Retrieve chunks from Milvus. Returns list of {text, source_file, distance, ...}."""
    from pymilvus import MilvusClient

    model = get_embedding_model()
    query_embedding = model.encode([query])[0].tolist()

    client = MilvusClient(db_path)
    # "source_file"/"doc_type" come from the phase-2 notebook schema, "filename" from oneshot_ingestion.py
    output_fields = ["text", "source_file", "filename", "doc_type", "page_number"]

    # Try to include extra fields if they exist
    try:
        results = client.search(
            collection_name=collection,
            data=[query_embedding],
            limit=top_k,
            output_fields=output_fields + ["item_type", "bbox_l", "bbox_t"],
        )
    except Exception:
        results = client.search(
            collection_name=collection,
            data=[query_embedding],
            limit=top_k,
            output_fields=output_fields,
        )

    client.close()

    chunks = []
    for hit in results[0]:
        entity = hit.get("entity", hit)
        source = entity.get("source_file") or entity.get("filename", "")
        doc_type = entity.get("doc_type", "")

        # Filter Excel if requested
        if exclude_excel and doc_type in ("xlsx", "xls"):
            continue
        if exclude_excel and any(ext in source.lower() for ext in [".xlsx", ".xls", "lakhs", "consolidation", "mt_2017"]):
            continue

        chunks.append({
            "text": entity.get("text", ""),
            "source_file": source,
            "doc_type": doc_type,
            "page_number": entity.get("page_number"),
            "distance": hit.get("distance", 0),
        })

    return chunks


def generate_answer(query: str, context_chunks: list) -> str:
    """Send context + query to Gemini, get answer."""
    model = get_gemini()

    context = "\n\n---\n\n".join(
        f"[Source: {c['source_file']}, Page: {c.get('page_number', '?')}]\n{c['text']}"
        for c in context_chunks
    )

    prompt = f"""You are a financial document analyst. Answer the following question using ONLY the provided context.
If the context does not contain the answer, respond with exactly: "UNANSWERABLE"
Do not guess or infer beyond what is explicitly stated.
Be concise — give only the answer, no explanation.

Context:
{context}

Question: {query}

Answer:"""

    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"ERROR: {e}"


def generate_answer_monolithic(query: str, all_docs_text: str) -> str:
    """Send full document text + query to Gemini."""
    model = get_gemini()

    prompt = f"""You are a financial document analyst. Answer the following question using ONLY the provided documents.
If the documents do not contain the answer, respond with exactly: "UNANSWERABLE"
Be concise — give only the answer, no explanation.

Documents:
{all_docs_text}

Question: {query}

Answer:"""

    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"ERROR: {e}"


def judge_answer(question: str, generated: str, ground_truth: str) -> dict:
    """Use Gemini as judge to evaluate answer correctness."""
    model = get_gemini()

    prompt = f"""You are a strict evaluator comparing a generated answer against the expected ground truth for a financial form-filling task.

Question: {question}
Expected Answer: {ground_truth}
Generated Answer: {generated}

STRICT RULES:
- If the generated answer is "UNANSWERABLE" or any refusal to answer (e.g., "I cannot answer", "Information not found", "not available"), and the expected answer is an actual value (not "Not found"), then the verdict MUST be "unanswerable". A refusal is NEVER "correct" when a real answer exists.
- "correct" means the generated answer contains the same factual content as the expected answer, even if phrased differently (e.g., "Rs. 5,000,000" and "50,00,000" and "50 Lakhs" are all correct for the same amount).
- "wrong" means the generated answer provides a specific value that contradicts the expected answer.
- ONLY mark "correct" if the expected answer is "Not found in documents" AND the generated answer is also a refusal/UNANSWERABLE.

Respond in JSON only:
{{"verdict": "correct" | "unanswerable" | "wrong", "reasoning": "one sentence explanation"}}"""

    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        # Parse JSON from response
        if "```" in text:
            text = text.split("```")[1].replace("json", "").strip()
        return json.loads(text)
    except Exception as e:
        return {"verdict": "error", "reasoning": str(e)}


# ── Experiment Runners ──────────────────────────────────────────

def run_rag_experiment(name: str, db_path: str, collection: str,
                       top_k: int = 5, exclude_excel: bool = False) -> dict:
    """Run all profiling questions against a Milvus RAG config."""
    gt = load_ground_truth()
    questions = gt["questions"]

    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {name}")
    print(f"  DB: {db_path}")
    print(f"  Collection: {collection}")
    print(f"  Top-K: {top_k}")
    print(f"  Exclude Excel: {exclude_excel}")
    print(f"  Questions: {len(questions)}")
    print(f"{'='*70}\n")

    results = []
    for i, q in enumerate(questions):
        qid = q["id"]
        question = q["question"]
        expected = q["answer"]

        print(f"[{i+1}/{len(questions)}] {qid}: {question[:60]}...")

        # Retrieve
        t0 = time.time()
        chunks = retrieve_from_milvus(question, db_path, collection, top_k, exclude_excel)
        retrieval_time = time.time() - t0

        # Generate
        t0 = time.time()
        answer = generate_answer(question, chunks)
        generation_time = time.time() - t0

        # Judge
        judgment = judge_answer(question, answer, expected)

        result = {
            "id": qid,
            "question": question,
            "expected_answer": expected,
            "generated_answer": answer,
            "verdict": judgment.get("verdict", "error"),
            "reasoning": judgment.get("reasoning", ""),
            "num_chunks_retrieved": len(chunks),
            "top_chunk_distance": chunks[0]["distance"] if chunks else None,
            "top_chunk_source": chunks[0]["source_file"] if chunks else None,
            "top_chunk_preview": chunks[0]["text"][:150] if chunks else None,
            "retrieval_time_s": round(retrieval_time, 3),
            "generation_time_s": round(generation_time, 3),
        }
        results.append(result)

        v = result["verdict"]
        symbol = {"correct": "✓", "unanswerable": "✗", "wrong": "✗✗", "error": "⚠"}.get(v, "?")
        print(f"  {symbol} {v}: {answer[:80]}")
        print(f"    Retrieved {len(chunks)} chunks in {retrieval_time:.2f}s, generated in {generation_time:.2f}s")

        # Rate limit
        time.sleep(0.5)

    # Summary
    verdicts = [r["verdict"] for r in results]
    summary = {
        "correct": verdicts.count("correct"),
        "unanswerable": verdicts.count("unanswerable"),
        "wrong": verdicts.count("wrong"),
        "error": verdicts.count("error"),
        "total": len(verdicts),
        "answerability_rate": round(
            (verdicts.count("correct") + verdicts.count("wrong")) / len(verdicts) * 100, 1
        ),
        "accuracy_rate": round(verdicts.count("correct") / len(verdicts) * 100, 1),
    }

    output = {
        "experiment": name,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "db_path": db_path,
            "collection": collection,
            "top_k": top_k,
            "exclude_excel": exclude_excel,
            "embedding_model": EMBEDDING_MODEL,
            "generation_model": GEMINI_MODEL,
            "num_questions": len(questions),
        },
        "summary": summary,
        "results": results,
    }

    # Save
    out_path = RESULTS_DIR / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"RESULTS: {name}")
    print(f"  Correct:       {summary['correct']}/{summary['total']}")
    print(f"  Unanswerable:  {summary['unanswerable']}/{summary['total']}")
    print(f"  Wrong:         {summary['wrong']}/{summary['total']}")
    print(f"  Accuracy:      {summary['accuracy_rate']}%")
    print(f"  Answerability: {summary['answerability_rate']}%")
    print(f"  Saved to: {out_path}")
    print(f"{'='*70}")

    return output


def run_monolithic_experiment() -> dict:
    """Run all questions with full document context (no RAG)."""
    gt = load_ground_truth()
    questions = gt["questions"]

    # Load all markdown documents
    md_dir = WORK_DIR / "docling_outputs" / "md"
    all_text = ""
    doc_count = 0
    for md_file in sorted(md_dir.glob("*.md")):
        text = md_file.read_text()
        all_text += f"\n\n===== DOCUMENT: {md_file.stem} =====\n\n{text}"
        doc_count += 1

    token_estimate = len(all_text.split()) * 1.3  # rough estimate

    print(f"\n{'='*70}")
    print(f"EXPERIMENT: monolithic (no RAG)")
    print(f"  Documents: {doc_count} markdown files")
    print(f"  Total chars: {len(all_text):,}")
    print(f"  Estimated tokens: ~{int(token_estimate):,}")
    print(f"  Questions: {len(questions)}")
    print(f"{'='*70}\n")

    results = []
    for i, q in enumerate(questions):
        qid = q["id"]
        question = q["question"]
        expected = q["answer"]

        print(f"[{i+1}/{len(questions)}] {qid}: {question[:60]}...")

        t0 = time.time()
        answer = generate_answer_monolithic(question, all_text)
        gen_time = time.time() - t0

        judgment = judge_answer(question, answer, expected)

        result = {
            "id": qid,
            "question": question,
            "expected_answer": expected,
            "generated_answer": answer,
            "verdict": judgment.get("verdict", "error"),
            "reasoning": judgment.get("reasoning", ""),
            "num_chunks_retrieved": "N/A (full context)",
            "generation_time_s": round(gen_time, 3),
        }
        results.append(result)

        v = result["verdict"]
        symbol = {"correct": "✓", "unanswerable": "✗", "wrong": "✗✗", "error": "⚠"}.get(v, "?")
        print(f"  {symbol} {v}: {answer[:80]}")
        print(f"    Generated in {gen_time:.2f}s")

        time.sleep(1)  # Rate limit

    verdicts = [r["verdict"] for r in results]
    summary = {
        "correct": verdicts.count("correct"),
        "unanswerable": verdicts.count("unanswerable"),
        "wrong": verdicts.count("wrong"),
        "error": verdicts.count("error"),
        "total": len(verdicts),
        "answerability_rate": round(
            (verdicts.count("correct") + verdicts.count("wrong")) / len(verdicts) * 100, 1
        ),
        "accuracy_rate": round(verdicts.count("correct") / len(verdicts) * 100, 1),
    }

    output = {
        "experiment": "monolithic",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "method": "full_document_context",
            "generation_model": GEMINI_MODEL,
            "num_documents": doc_count,
            "total_chars": len(all_text),
            "estimated_tokens": int(token_estimate),
            "num_questions": len(questions),
        },
        "summary": summary,
        "results": results,
    }

    out_path = RESULTS_DIR / f"monolithic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"RESULTS: monolithic")
    print(f"  Correct:       {summary['correct']}/{summary['total']}")
    print(f"  Unanswerable:  {summary['unanswerable']}/{summary['total']}")
    print(f"  Wrong:         {summary['wrong']}/{summary['total']}")
    print(f"  Accuracy:      {summary['accuracy_rate']}%")
    print(f"  Saved to: {out_path}")
    print(f"{'='*70}")

    return output


def show_results():
    """Show comparison table of all completed experiments."""
    result_files = sorted(RESULTS_DIR.glob("*.json"))
    if not result_files:
        print("No experiment results found.")
        return

    print(f"\n{'='*90}")
    print(f"{'Experiment':<25} {'Correct':>8} {'Unans.':>8} {'Wrong':>8} {'Accuracy':>10} {'Answerability':>14}")
    print(f"{'-'*90}")

    for f in result_files:
        with open(f) as fh:
            data = json.load(fh)
        s = data["summary"]
        name = data["experiment"]
        print(f"{name:<25} {s['correct']:>5}/{s['total']:<3} {s['unanswerable']:>5}/{s['total']:<3} "
              f"{s['wrong']:>5}/{s['total']:<3} {s['accuracy_rate']:>8}%  {s['answerability_rate']:>12}%")

    print(f"{'='*90}")


# ── Advanced Retrieval Functions ─────────────────────────────────

_reranker_model = None

def get_reranker():
    """Load BGE cross-encoder reranker."""
    global _reranker_model
    if _reranker_model is None:
        from sentence_transformers import CrossEncoder
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading BAAI/bge-reranker-v2-m3 on {device}...")
        _reranker_model = CrossEncoder("BAAI/bge-reranker-v2-m3", device=device)
        print("  Reranker loaded.")
    return _reranker_model


def retrieve_with_reranking(query: str, db_path: str, collection: str,
                            initial_top_k: int = 50, final_top_k: int = 5) -> list:
    """Retrieve top-N chunks, rerank with cross-encoder, return top-K."""
    # Step 1: broad retrieval
    chunks = retrieve_from_milvus(query, db_path, collection, top_k=initial_top_k)

    if not chunks:
        return []

    # Step 2: rerank with cross-encoder
    reranker = get_reranker()
    pairs = [(query, c["text"]) for c in chunks]
    scores = reranker.predict(pairs)

    # Attach scores and sort
    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = float(score)

    chunks.sort(key=lambda c: c["rerank_score"], reverse=True)
    return chunks[:final_top_k]


def reformulate_query(raw_query: str) -> str:
    """Use Gemini to reformulate a form field label into a natural language question."""
    model = get_gemini()
    prompt = f"""Convert this form field label into a clear, natural language search query for retrieving information from Indian financial documents.

Form field: {raw_query}

Rules:
- Write a clear question a human would ask
- Include relevant context (e.g., "of the company", "for the financial year")
- Keep it concise (one sentence)
- Do not add information not implied by the field

Natural language query:"""

    try:
        response = model.generate_content(prompt)
        reformulated = response.text.strip().strip('"').strip("'")
        return reformulated
    except Exception as e:
        return raw_query  # fallback to original


def multi_query_retrieve(query: str, db_path: str, collection: str,
                         num_variants: int = 3, top_k_per: int = 5) -> list:
    """Generate query variants, retrieve for each, merge via reciprocal rank fusion."""
    model = get_gemini()

    # Step 1: Generate query variants
    prompt = f"""Generate {num_variants} different ways to phrase the following search query for a financial document database.
Each variant should approach the information from a different angle.
Return only the queries, one per line, no numbering.

Original query: {query}

Variants:"""

    try:
        response = model.generate_content(prompt)
        variants = [v.strip().strip("-").strip("•").strip() for v in response.text.strip().split("\n") if v.strip()]
        variants = variants[:num_variants]
    except Exception:
        variants = [query]

    # Always include original
    all_queries = [query] + variants

    # Step 2: Retrieve for each variant
    all_chunks = {}  # text -> {chunk, ranks}
    for qi, q in enumerate(all_queries):
        chunks = retrieve_from_milvus(q, db_path, collection, top_k=top_k_per)
        for rank, c in enumerate(chunks):
            key = c["text"][:200]  # dedup key
            if key not in all_chunks:
                all_chunks[key] = {"chunk": c, "rrf_score": 0}
            # Reciprocal rank fusion: score = sum(1 / (k + rank)) across queries
            all_chunks[key]["rrf_score"] += 1.0 / (60 + rank)  # k=60 is standard RRF constant

    # Step 3: Sort by RRF score, return top results
    ranked = sorted(all_chunks.values(), key=lambda x: x["rrf_score"], reverse=True)
    results = [item["chunk"] for item in ranked[:top_k_per]]
    for item, ranked_item in zip(results, ranked[:top_k_per]):
        item["rrf_score"] = ranked_item["rrf_score"]

    return results


def run_reformulated_experiment(name: str, db_path: str, collection: str,
                                top_k: int = 5) -> dict:
    """Run experiment with query reformulation — transform form labels to natural queries."""
    gt = load_ground_truth()
    questions = gt["questions"]

    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {name} (with query reformulation)")
    print(f"  DB: {db_path}, Top-K: {top_k}")
    print(f"{'='*70}\n")

    results = []
    for i, q in enumerate(questions):
        qid = q["id"]
        question = q["question"]
        expected = q["answer"]

        # Reformulate
        reformed = reformulate_query(question)
        print(f"[{i+1}/{len(questions)}] {qid}")
        print(f"  Original:     {question[:60]}...")
        print(f"  Reformulated: {reformed[:60]}...")

        # Retrieve with reformulated query
        t0 = time.time()
        chunks = retrieve_from_milvus(reformed, db_path, collection, top_k)
        retrieval_time = time.time() - t0

        # Generate answer
        t0 = time.time()
        answer = generate_answer(reformed, chunks)
        generation_time = time.time() - t0

        # Judge
        judgment = judge_answer(question, answer, expected)

        result = {
            "id": qid,
            "question": question,
            "reformulated_query": reformed,
            "expected_answer": expected,
            "generated_answer": answer,
            "verdict": judgment.get("verdict", "error"),
            "reasoning": judgment.get("reasoning", ""),
            "num_chunks_retrieved": len(chunks),
            "top_chunk_distance": chunks[0]["distance"] if chunks else None,
            "top_chunk_source": chunks[0]["source_file"] if chunks else None,
            "top_chunk_preview": chunks[0]["text"][:150] if chunks else None,
            "retrieval_time_s": round(retrieval_time, 3),
            "generation_time_s": round(generation_time, 3),
        }
        results.append(result)

        v = result["verdict"]
        symbol = {"correct": "✓", "unanswerable": "✗", "wrong": "✗✗", "error": "⚠"}.get(v, "?")
        print(f"  {symbol} {v}: {answer[:80]}")
        time.sleep(0.5)

    return _save_experiment(name, results, questions, {
        "db_path": db_path, "collection": collection, "top_k": top_k,
        "method": "query_reformulation",
    })


def run_reranking_experiment(name: str, db_path: str, collection: str,
                             initial_top_k: int = 50, final_top_k: int = 5) -> dict:
    """Run experiment with cross-encoder reranking."""
    gt = load_ground_truth()
    questions = gt["questions"]

    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {name} (with BGE reranker)")
    print(f"  DB: {db_path}, Retrieve {initial_top_k} → Rerank → Top {final_top_k}")
    print(f"{'='*70}\n")

    results = []
    for i, q in enumerate(questions):
        qid = q["id"]
        question = q["question"]
        expected = q["answer"]

        print(f"[{i+1}/{len(questions)}] {qid}: {question[:60]}...")

        t0 = time.time()
        chunks = retrieve_with_reranking(question, db_path, collection, initial_top_k, final_top_k)
        retrieval_time = time.time() - t0

        t0 = time.time()
        answer = generate_answer(question, chunks)
        generation_time = time.time() - t0

        judgment = judge_answer(question, answer, expected)

        result = {
            "id": qid,
            "question": question,
            "expected_answer": expected,
            "generated_answer": answer,
            "verdict": judgment.get("verdict", "error"),
            "reasoning": judgment.get("reasoning", ""),
            "num_chunks_retrieved": len(chunks),
            "top_chunk_rerank_score": chunks[0].get("rerank_score") if chunks else None,
            "top_chunk_source": chunks[0]["source_file"] if chunks else None,
            "top_chunk_preview": chunks[0]["text"][:150] if chunks else None,
            "retrieval_time_s": round(retrieval_time, 3),
            "generation_time_s": round(generation_time, 3),
        }
        results.append(result)

        v = result["verdict"]
        symbol = {"correct": "✓", "unanswerable": "✗", "wrong": "✗✗", "error": "⚠"}.get(v, "?")
        print(f"  {symbol} {v}: {answer[:80]}")
        if chunks:
            print(f"    Top rerank score: {chunks[0].get('rerank_score', '?'):.3f}")
        time.sleep(0.5)

    return _save_experiment(name, results, questions, {
        "db_path": db_path, "collection": collection,
        "initial_top_k": initial_top_k, "final_top_k": final_top_k,
        "method": "cross_encoder_reranking", "reranker": "BAAI/bge-reranker-v2-m3",
    })


def run_multi_query_experiment(name: str, db_path: str, collection: str,
                               num_variants: int = 3, top_k: int = 5) -> dict:
    """Run experiment with multi-query RAG fusion."""
    gt = load_ground_truth()
    questions = gt["questions"]

    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {name} (multi-query RAG fusion)")
    print(f"  DB: {db_path}, {num_variants} variants, RRF merge → Top {top_k}")
    print(f"{'='*70}\n")

    results = []
    for i, q in enumerate(questions):
        qid = q["id"]
        question = q["question"]
        expected = q["answer"]

        print(f"[{i+1}/{len(questions)}] {qid}: {question[:60]}...")

        t0 = time.time()
        chunks = multi_query_retrieve(question, db_path, collection, num_variants, top_k)
        retrieval_time = time.time() - t0

        t0 = time.time()
        answer = generate_answer(question, chunks)
        generation_time = time.time() - t0

        judgment = judge_answer(question, answer, expected)

        result = {
            "id": qid,
            "question": question,
            "expected_answer": expected,
            "generated_answer": answer,
            "verdict": judgment.get("verdict", "error"),
            "reasoning": judgment.get("reasoning", ""),
            "num_chunks_retrieved": len(chunks),
            "top_chunk_rrf_score": chunks[0].get("rrf_score") if chunks else None,
            "top_chunk_source": chunks[0]["source_file"] if chunks else None,
            "retrieval_time_s": round(retrieval_time, 3),
            "generation_time_s": round(generation_time, 3),
        }
        results.append(result)

        v = result["verdict"]
        symbol = {"correct": "✓", "unanswerable": "✗", "wrong": "✗✗", "error": "⚠"}.get(v, "?")
        print(f"  {symbol} {v}: {answer[:80]}")
        time.sleep(1)  # more API calls per question

    return _save_experiment(name, results, questions, {
        "db_path": db_path, "collection": collection,
        "num_variants": num_variants, "top_k": top_k,
        "method": "multi_query_rag_fusion",
    })


def run_topk_sweep(db_path: str, collection: str, top_ks: list = [5, 10, 20, 50]) -> dict:
    """Run same questions at different top-K values to see the retrieval curve."""
    gt = load_ground_truth()
    questions = gt["questions"]

    all_results = {}
    for top_k in top_ks:
        name = f"topk_{top_k}"
        print(f"\n--- Top-K = {top_k} ---")
        result = run_rag_experiment(name, db_path, collection, top_k=top_k)
        all_results[top_k] = result["summary"]

    print(f"\n{'='*70}")
    print("TOP-K SWEEP RESULTS")
    print(f"{'Top-K':<8} {'Correct':>8} {'Unans.':>8} {'Wrong':>8} {'Accuracy':>10}")
    print(f"{'-'*50}")
    for k, s in all_results.items():
        print(f"{k:<8} {s['correct']:>5}/{s['total']:<3} {s['unanswerable']:>5}/{s['total']:<3} "
              f"{s['wrong']:>5}/{s['total']:<3} {s['accuracy_rate']:>8}%")
    print(f"{'='*70}")
    return all_results


def _save_experiment(name: str, results: list, questions: list, config: dict) -> dict:
    """Common save logic for experiment results."""
    verdicts = [r["verdict"] for r in results]
    summary = {
        "correct": verdicts.count("correct"),
        "unanswerable": verdicts.count("unanswerable"),
        "wrong": verdicts.count("wrong"),
        "error": verdicts.count("error"),
        "total": len(verdicts),
        "answerability_rate": round(
            (verdicts.count("correct") + verdicts.count("wrong")) / len(verdicts) * 100, 1
        ),
        "accuracy_rate": round(verdicts.count("correct") / len(verdicts) * 100, 1),
    }

    output = {
        "experiment": name,
        "timestamp": datetime.now().isoformat(),
        "config": {
            **config,
            "embedding_model": EMBEDDING_MODEL,
            "generation_model": GEMINI_MODEL,
            "num_questions": len(questions),
        },
        "summary": summary,
        "results": results,
    }

    out_path = RESULTS_DIR / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"RESULTS: {name}")
    print(f"  Correct:       {summary['correct']}/{summary['total']}")
    print(f"  Unanswerable:  {summary['unanswerable']}/{summary['total']}")
    print(f"  Wrong:         {summary['wrong']}/{summary['total']}")
    print(f"  Accuracy:      {summary['accuracy_rate']}%")
    print(f"  Answerability: {summary['answerability_rate']}%")
    print(f"  Saved to: {out_path}")
    print(f"{'='*70}")

    return output


# ── Experiment Definitions ──────────────────────────────────────

EXPERIMENTS = {
    "baseline": {
        "desc": "All docs, hierarchical chunking, top-5 (5,086 chunks)",
        "fn": lambda: run_rag_experiment(
            "baseline",
            str(WORK_DIR / "oneshot.db"),
            "oneshot",
            top_k=5,
        ),
    },
    "no_excel": {
        "desc": "Exclude Excel files, top-5",
        "fn": lambda: run_rag_experiment(
            "no_excel",
            str(WORK_DIR / "oneshot.db"),
            "oneshot",
            top_k=5,
            exclude_excel=True,
        ),
    },
    "mixed_chunking": {
        "desc": "Mixed strategy (hierarchical for xlsx, hybrid for rest), top-5 (3,813 chunks)",
        "fn": lambda: run_rag_experiment(
            "mixed_chunking",
            str(WORK_DIR / "milvus_all_docs_mixed.db"),
            "rag_all_docs_mixed",
            top_k=5,
        ),
    },
    "hybrid_chunking": {
        "desc": "Hybrid token-aware chunking, top-5 (28,554 chunks)",
        "fn": lambda: run_rag_experiment(
            "hybrid_chunking",
            str(WORK_DIR / "milvus_all_docs_hybrid.db"),
            "rag_all_docs_hybrid",
            top_k=5,
        ),
    },
    "mixed_top20": {
        "desc": "Mixed strategy, top-20 (more context)",
        "fn": lambda: run_rag_experiment(
            "mixed_top20",
            str(WORK_DIR / "milvus_all_docs_mixed.db"),
            "rag_all_docs_mixed",
            top_k=20,
        ),
    },
    "brute_force": {
        "desc": "Mixed strategy, top-350 (brute force, ~70K tokens)",
        "fn": lambda: run_rag_experiment(
            "brute_force",
            str(WORK_DIR / "milvus_all_docs_mixed.db"),
            "rag_all_docs_mixed",
            top_k=350,
        ),
    },
    "monolithic": {
        "desc": "Full document text to Gemini, no RAG",
        "fn": run_monolithic_experiment,
    },
    # ── Standard RAG improvements (PhD-demanded) ──
    "query_reformulation": {
        "desc": "Gemini reformulates form labels → natural language queries, then top-5",
        "fn": lambda: run_reformulated_experiment(
            "query_reformulation",
            str(WORK_DIR / "milvus_all_docs_mixed.db"),
            "rag_all_docs_mixed",
            top_k=5,
        ),
    },
    "reranking": {
        "desc": "Retrieve top-50, rerank with BGE-reranker-v2-m3, take top-5",
        "fn": lambda: run_reranking_experiment(
            "reranking",
            str(WORK_DIR / "milvus_all_docs_mixed.db"),
            "rag_all_docs_mixed",
            initial_top_k=50,
            final_top_k=5,
        ),
    },
    "multi_query": {
        "desc": "Generate 3 query variants, retrieve each, merge via reciprocal rank fusion",
        "fn": lambda: run_multi_query_experiment(
            "multi_query",
            str(WORK_DIR / "milvus_all_docs_mixed.db"),
            "rag_all_docs_mixed",
            num_variants=3,
            top_k=5,
        ),
    },
    "rerank_plus_reform": {
        "desc": "Query reformulation + cross-encoder reranking (the full SOTA stack)",
        "fn": lambda: _run_combined_experiment(),
    },
    "topk_sweep": {
        "desc": "Systematic top-K sweep: 5, 10, 20, 50 on mixed chunking",
        "fn": lambda: run_topk_sweep(
            str(WORK_DIR / "milvus_all_docs_mixed.db"),
            "rag_all_docs_mixed",
            top_ks=[5, 10, 20, 50],
        ),
    },
}


def _run_combined_experiment():
    """Combined: reformulate query + rerank."""
    gt = load_ground_truth()
    questions = gt["questions"]
    db_path = str(WORK_DIR / "milvus_all_docs_mixed.db")
    collection = "rag_all_docs_mixed"

    print(f"\n{'='*70}")
    print(f"EXPERIMENT: rerank_plus_reform (reformulation + reranking)")
    print(f"{'='*70}\n")

    results = []
    for i, q in enumerate(questions):
        qid = q["id"]
        question = q["question"]
        expected = q["answer"]

        # Step 1: Reformulate
        reformed = reformulate_query(question)
        print(f"[{i+1}/{len(questions)}] {qid}")
        print(f"  Original:     {question[:50]}...")
        print(f"  Reformulated: {reformed[:50]}...")

        # Step 2: Retrieve + Rerank with reformulated query
        t0 = time.time()
        chunks = retrieve_with_reranking(reformed, db_path, collection, initial_top_k=50, final_top_k=5)
        retrieval_time = time.time() - t0

        # Step 3: Generate
        t0 = time.time()
        answer = generate_answer(reformed, chunks)
        generation_time = time.time() - t0

        judgment = judge_answer(question, answer, expected)

        result = {
            "id": qid,
            "question": question,
            "reformulated_query": reformed,
            "expected_answer": expected,
            "generated_answer": answer,
            "verdict": judgment.get("verdict", "error"),
            "reasoning": judgment.get("reasoning", ""),
            "num_chunks_retrieved": len(chunks),
            "top_chunk_rerank_score": chunks[0].get("rerank_score") if chunks else None,
            "top_chunk_source": chunks[0]["source_file"] if chunks else None,
            "retrieval_time_s": round(retrieval_time, 3),
            "generation_time_s": round(generation_time, 3),
        }
        results.append(result)

        v = result["verdict"]
        symbol = {"correct": "✓", "unanswerable": "✗", "wrong": "✗✗", "error": "⚠"}.get(v, "?")
        print(f"  {symbol} {v}: {answer[:80]}")
        time.sleep(0.5)

    return _save_experiment("rerank_plus_reform", results, questions, {
        "db_path": db_path, "collection": collection,
        "method": "query_reformulation + cross_encoder_reranking",
        "reranker": "BAAI/bge-reranker-v2-m3",
        "initial_top_k": 50, "final_top_k": 5,
    })


# ── CLI ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RAG Experiment Harness")
    parser.add_argument("--experiment", "-e", help="Experiment name to run")
    parser.add_argument("--list", "-l", action="store_true", help="List available experiments")
    parser.add_argument("--results", "-r", action="store_true", help="Show results comparison")
    args = parser.parse_args()

    if args.list:
        print("\nAvailable experiments:")
        for name, exp in EXPERIMENTS.items():
            print(f"  {name:<20} {exp['desc']}")
        return

    if args.results:
        show_results()
        return

    if args.experiment:
        if args.experiment not in EXPERIMENTS:
            print(f"Unknown experiment: {args.experiment}")
            print(f"Available: {', '.join(EXPERIMENTS.keys())}")
            sys.exit(1)
        EXPERIMENTS[args.experiment]["fn"]()
        print("\nAll results so far:")
        show_results()
        return

    parser.print_help()


if __name__ == "__main__":
    main()
