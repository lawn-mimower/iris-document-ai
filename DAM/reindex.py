#!/usr/bin/env python3
"""
Clean Reindexing Pipeline
==========================
Parses all documents, chunks three ways, embeds, stores in Milvus.
Verifies every index contains the expected data.

Usage:
    python reindex.py

    Input/output folders can be changed with the DAM_INPUT_DIR and
    DAM_REINDEX_OUTPUT environment variables.
"""

import json
import os
import time
import hashlib
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import torch
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient, DataType
from docling.document_converter import DocumentConverter
from docling.chunking import HierarchicalChunker, HybridChunker
from tqdm import tqdm

# ── Config ──────────────────────────────────────────────────────

INPUT_DIR = Path(os.getenv("DAM_INPUT_DIR", Path(__file__).parent / "all_data"))
OUTPUT_DIR = Path(os.getenv("DAM_REINDEX_OUTPUT", Path(__file__).parent / "reindex_output"))
OUTPUT_DIR.mkdir(exist_ok=True)

MILVUS_DB = str(OUTPUT_DIR / "experiment.db")
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
EMBEDDING_DIM = 1024
BATCH_SIZE = 32

# Three chunking strategies
COLLECTIONS = {
    "hierarchical": "exp_hierarchical",
    "hybrid": "exp_hybrid",
    "recursive": "exp_recursive",
}

# Verification strings (known content that MUST be in the index)
VERIFY_STRINGS = {
    "CIN": "U00000XX0000PTC000000",
    "Company": "EXAMPLECO Engineering",
    "Address": "EXAMPLE",
    "Capital": "50,000",
    "FY Start": "01/04/2023",
    "Board Date": "06/09/2024",
    "Industry": "Commercial and Industrial",
    "Secretarial": "secretarial audit",
}


# ── Step 1: Parse all documents with Docling ────────────────────

def parse_all_documents():
    """Convert all documents to Docling format. Returns list of (DoclingDocument, metadata)."""
    converter = DocumentConverter()
    docs = []
    report = {"parsed": [], "failed": [], "skipped": []}

    files = sorted(INPUT_DIR.iterdir())
    print(f"\n{'='*70}")
    print(f"STEP 1: PARSING {len(files)} files from {INPUT_DIR}")
    print(f"{'='*70}\n")

    for f in files:
        if f.name.startswith("."):
            continue

        ext = f.suffix.lower()
        print(f"  Parsing: {f.name} ({ext})...", end=" ", flush=True)

        try:
            result = converter.convert(str(f))
            doc = result.document

            # Extract text to verify it's non-empty
            md_text = doc.export_to_markdown()
            if len(md_text.strip()) < 10:
                print(f"EMPTY ({len(md_text)} chars)")
                report["failed"].append({"file": f.name, "reason": "empty output"})
                continue

            metadata = {
                "filename": f.name,
                "stem": f.stem,
                "ext": ext,
                "doc_type": ext.lstrip("."),
                "chars": len(md_text),
            }

            docs.append((doc, metadata, md_text))
            print(f"OK ({len(md_text):,} chars)")
            report["parsed"].append(metadata)

        except Exception as e:
            print(f"FAILED: {e}")
            report["failed"].append({"file": f.name, "reason": str(e)[:200]})

    print(f"\nParsed: {len(report['parsed'])}, Failed: {len(report['failed'])}")

    # Save markdown exports for monolithic experiment
    md_dir = OUTPUT_DIR / "markdown"
    md_dir.mkdir(exist_ok=True)
    for doc, meta, md_text in docs:
        (md_dir / f"{meta['stem']}.md").write_text(md_text)
    print(f"Markdown exports saved to {md_dir}")

    # Save report
    with open(OUTPUT_DIR / "parsing_report.json", "w") as f:
        json.dump(report, f, indent=2)

    return docs, report


# ── Step 2: Chunk documents three ways ──────────────────────────

def chunk_all_documents(docs):
    """Chunk each document three ways. Returns dict of strategy -> chunks list."""
    print(f"\n{'='*70}")
    print(f"STEP 2: CHUNKING {len(docs)} documents × 3 strategies")
    print(f"{'='*70}\n")

    hierarchical_chunker = HierarchicalChunker()
    hybrid_chunker = HybridChunker(
        tokenizer=EMBEDDING_MODEL,
        max_tokens=512,
    )

    all_chunks = {"hierarchical": [], "hybrid": [], "recursive": []}

    for doc, meta, md_text in docs:
        fname = meta["filename"]

        # Strategy 1: Hierarchical
        try:
            h_chunks = list(hierarchical_chunker.chunk(doc))
            for i, chunk in enumerate(h_chunks):
                text = chunk.text if hasattr(chunk, 'text') else str(chunk)
                if len(text.strip()) < 5:
                    continue
                all_chunks["hierarchical"].append({
                    "text": text,
                    "filename": fname,
                    "doc_type": meta["doc_type"],
                    "chunk_index": i,
                    "chunk_strategy": "hierarchical",
                    "chunk_id": hashlib.md5(f"{fname}:h:{i}:{text[:50]}".encode()).hexdigest()[:12],
                })
        except Exception as e:
            print(f"  Hierarchical failed on {fname}: {e}")

        # Strategy 2: Hybrid
        try:
            hb_chunks = list(hybrid_chunker.chunk(doc))
            for i, chunk in enumerate(hb_chunks):
                text = chunk.text if hasattr(chunk, 'text') else str(chunk)
                if len(text.strip()) < 5:
                    continue
                all_chunks["hybrid"].append({
                    "text": text,
                    "filename": fname,
                    "doc_type": meta["doc_type"],
                    "chunk_index": i,
                    "chunk_strategy": "hybrid",
                    "chunk_id": hashlib.md5(f"{fname}:hb:{i}:{text[:50]}".encode()).hexdigest()[:12],
                })
        except Exception as e:
            print(f"  Hybrid failed on {fname}: {e}")

        # Strategy 3: Recursive character splitting (on markdown text)
        rec_chunks = recursive_split(md_text, chunk_size=1000, overlap=200)
        for i, text in enumerate(rec_chunks):
            if len(text.strip()) < 5:
                continue
            all_chunks["recursive"].append({
                "text": text,
                "filename": fname,
                "doc_type": meta["doc_type"],
                "chunk_index": i,
                "chunk_strategy": "recursive",
                "chunk_id": hashlib.md5(f"{fname}:r:{i}:{text[:50]}".encode()).hexdigest()[:12],
            })

        print(f"  {fname}: H={len([c for c in all_chunks['hierarchical'] if c['filename']==fname])}, "
              f"Hb={len([c for c in all_chunks['hybrid'] if c['filename']==fname])}, "
              f"R={len([c for c in all_chunks['recursive'] if c['filename']==fname])}")

    for strategy, chunks in all_chunks.items():
        by_type = defaultdict(int)
        for c in chunks:
            by_type[c["doc_type"]] += 1
        print(f"\n  {strategy}: {len(chunks)} total chunks")
        for dt, count in sorted(by_type.items()):
            print(f"    {dt}: {count}")

    # Save chunk reports
    for strategy, chunks in all_chunks.items():
        with open(OUTPUT_DIR / f"chunks_{strategy}.json", "w") as f:
            json.dump({"total": len(chunks), "chunks": [
                {k: v for k, v in c.items() if k != "text"} | {"text_preview": c["text"][:100]}
                for c in chunks
            ]}, f, indent=2)

    return all_chunks


def recursive_split(text: str, chunk_size: int = 1000, overlap: int = 200) -> list:
    """Simple recursive character text splitter."""
    separators = ["\n\n", "\n", ". ", " ", ""]
    chunks = []

    def _split(text, seps):
        if len(text) <= chunk_size:
            return [text]

        sep = seps[0] if seps else ""
        parts = text.split(sep) if sep else [text[i:i+chunk_size] for i in range(0, len(text), chunk_size - overlap)]

        result = []
        current = ""
        for part in parts:
            candidate = current + sep + part if current else part
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    result.append(current)
                if len(part) > chunk_size and len(seps) > 1:
                    result.extend(_split(part, seps[1:]))
                else:
                    current = part
        if current:
            result.append(current)

        # Add overlap
        if overlap > 0 and len(result) > 1:
            overlapped = [result[0]]
            for i in range(1, len(result)):
                prev_tail = result[i-1][-overlap:] if len(result[i-1]) > overlap else result[i-1]
                overlapped.append(prev_tail + result[i])
            return overlapped

        return result

    return _split(text, separators)


# ── Step 3: Embed and store in Milvus ───────────────────────────

def embed_and_store(all_chunks):
    """Embed all chunks and store in Milvus collections."""
    print(f"\n{'='*70}")
    print(f"STEP 3: EMBEDDING AND STORING")
    print(f"{'='*70}\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {EMBEDDING_MODEL} on {device}...")
    model = SentenceTransformer(EMBEDDING_MODEL, device=device)
    print(f"  Loaded. Dim={model.get_sentence_embedding_dimension()}")

    client = MilvusClient(MILVUS_DB)

    for strategy, collection_name in COLLECTIONS.items():
        chunks = all_chunks[strategy]
        print(f"\n--- {strategy}: {len(chunks)} chunks → {collection_name} ---")

        # Drop and recreate
        if client.has_collection(collection_name):
            client.drop_collection(collection_name)

        client.create_collection(
            collection_name=collection_name,
            dimension=EMBEDDING_DIM,
        )

        # Embed in batches
        texts = [c["text"] for c in chunks]
        all_embeddings = []

        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc=f"  Embedding {strategy}"):
            batch = texts[i:i+BATCH_SIZE]
            embeddings = model.encode(batch, normalize_embeddings=True)
            all_embeddings.extend(embeddings.tolist())

        # Prepare data for insert
        data = []
        for i, (chunk, embedding) in enumerate(zip(chunks, all_embeddings)):
            data.append({
                "id": i,
                "vector": embedding,
                "text": chunk["text"],
                "filename": chunk["filename"],
                "doc_type": chunk["doc_type"],
                "chunk_index": chunk["chunk_index"],
                "chunk_id": chunk["chunk_id"],
            })

        # Insert in batches
        for i in range(0, len(data), 1000):
            batch = data[i:i+1000]
            client.insert(collection_name=collection_name, data=batch)

        # Verify
        stats = client.get_collection_stats(collection_name)
        print(f"  Stored: {stats['row_count']} vectors")

    client.close()
    print(f"\nAll collections stored in {MILVUS_DB}")


# ── Step 4: Verify indexes ──────────────────────────────────────

def verify_indexes():
    """Search each index for known strings to verify completeness."""
    print(f"\n{'='*70}")
    print(f"STEP 4: VERIFICATION")
    print(f"{'='*70}\n")

    client = MilvusClient(MILVUS_DB)

    for strategy, collection_name in COLLECTIONS.items():
        print(f"\n--- {strategy} ({collection_name}) ---")

        stats = client.get_collection_stats(collection_name)
        print(f"  Total vectors: {stats['row_count']}")

        # Count by doc_type
        for doc_type in ["pdf", "docx", "xlsx", "jpg", "xml"]:
            results = client.query(
                collection_name, filter=f'doc_type == "{doc_type}"',
                output_fields=["filename"], limit=10000
            )
            if results:
                files = set(r["filename"] for r in results)
                print(f"  {doc_type}: {len(results)} chunks from {len(files)} files")

        # Verify known strings exist in the chunks
        print(f"\n  Content verification:")
        for label, search_str in VERIFY_STRINGS.items():
            results = client.query(
                collection_name,
                filter=f'text like "%{search_str}%"',
                output_fields=["text", "filename"],
                limit=5,
            )
            found = len(results)
            source = results[0]["filename"] if results else "NOT FOUND"
            status = "✓" if found > 0 else "✗ MISSING"
            print(f"    {status} {label} ({search_str[:30]}): {found} chunks, source={source}")

    client.close()

    # Also check total markdown size for monolithic experiment
    md_dir = OUTPUT_DIR / "markdown"
    total_chars = sum(f.stat().st_size for f in md_dir.glob("*.md"))
    total_tokens_est = total_chars // 4  # rough estimate
    print(f"\n  Monolithic experiment: {total_chars:,} chars (~{total_tokens_est:,} tokens) across {len(list(md_dir.glob('*.md')))} markdown files")


# ── Main ────────────────────────────────────────────────────────

def main():
    t0 = time.time()

    docs, report = parse_all_documents()
    all_chunks = chunk_all_documents(docs)
    embed_and_store(all_chunks)
    verify_indexes()

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"COMPLETE in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"DB: {MILVUS_DB}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
