#!/usr/bin/env python3
"""
Chunk Statistics Analyzer
==========================
Analyzes chunk sizes and token counts from Milvus database.
Provides comprehensive statistical analysis including:
- Token count statistics (mean, median, percentiles, distribution)
- Character count statistics
- Distribution by document type
- Visualization of distributions
"""

import argparse
import json
from pymilvus import MilvusClient
import tiktoken
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path

def count_tokens(text, encoding_name="cl100k_base"):
    """Count tokens using tiktoken (GPT-4/GPT-3.5 encoding)."""
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))

def analyze_chunks(db_path="milvus_all_docs.db", collection_name="rag_all_docs"):
    """Analyze all chunks from Milvus database."""

    print(f"Connecting to database: {db_path}")
    client = MilvusClient(db_path)

    # Query all chunks (an empty filter needs an iterator; plain query() requires a limit)
    print(f"Querying collection: {collection_name}")
    results = []
    iterator = client.query_iterator(
        collection_name=collection_name,
        batch_size=1000,
        filter="",
        output_fields=["text", "source_file", "doc_type", "chunk_id", "total_chunks"]
    )
    while True:
        batch = iterator.next()
        if not batch:
            iterator.close()
            break
        results.extend(batch)

    print(f"Found {len(results)} chunks")

    # Collect statistics
    token_counts = []
    char_counts = []
    chunks_by_type = defaultdict(list)
    chunks_by_doc = defaultdict(list)

    print("\nAnalyzing chunks...")
    for chunk in results:
        text = chunk.get('text', '')
        doc_type = chunk.get('doc_type', 'unknown')
        source = chunk.get('source_file', 'unknown')

        # Count tokens and characters
        token_count = count_tokens(text)
        char_count = len(text)

        token_counts.append(token_count)
        char_counts.append(char_count)
        chunks_by_type[doc_type].append(token_count)
        chunks_by_doc[source].append(token_count)

    # Calculate statistics
    token_counts = np.array(token_counts)
    char_counts = np.array(char_counts)

    stats = {
        "total_chunks": len(token_counts),
        "token_statistics": {
            "mean": float(np.mean(token_counts)),
            "median": float(np.median(token_counts)),
            "std": float(np.std(token_counts)),
            "min": int(np.min(token_counts)),
            "max": int(np.max(token_counts)),
            "percentile_10": float(np.percentile(token_counts, 10)),
            "percentile_25": float(np.percentile(token_counts, 25)),
            "percentile_75": float(np.percentile(token_counts, 75)),
            "percentile_90": float(np.percentile(token_counts, 90)),
            "percentile_95": float(np.percentile(token_counts, 95)),
        },
        "character_statistics": {
            "mean": float(np.mean(char_counts)),
            "median": float(np.median(char_counts)),
            "std": float(np.std(char_counts)),
            "min": int(np.min(char_counts)),
            "max": int(np.max(char_counts)),
        },
        "chunks_by_size_bucket": {
            "0-50 tokens": int(np.sum(token_counts < 50)),
            "50-100 tokens": int(np.sum((token_counts >= 50) & (token_counts < 100))),
            "100-200 tokens": int(np.sum((token_counts >= 100) & (token_counts < 200))),
            "200-500 tokens": int(np.sum((token_counts >= 200) & (token_counts < 500))),
            "500-1000 tokens": int(np.sum((token_counts >= 500) & (token_counts < 1000))),
            "1000+ tokens": int(np.sum(token_counts >= 1000)),
        },
        "by_document_type": {}
    }

    # Statistics by document type
    for doc_type, tokens in chunks_by_type.items():
        tokens = np.array(tokens)
        stats["by_document_type"][doc_type] = {
            "count": len(tokens),
            "mean_tokens": float(np.mean(tokens)),
            "median_tokens": float(np.median(tokens)),
            "min_tokens": int(np.min(tokens)),
            "max_tokens": int(np.max(tokens)),
        }

    # Find smallest chunks (potential issues)
    small_chunk_threshold = 20
    small_chunks = token_counts < small_chunk_threshold
    stats["small_chunks_analysis"] = {
        "threshold": small_chunk_threshold,
        "count": int(np.sum(small_chunks)),
        "percentage": float(np.sum(small_chunks) / len(token_counts) * 100)
    }

    # Print results
    print("\n" + "="*70)
    print("CHUNK STATISTICS ANALYSIS")
    print("="*70)

    print(f"\nTotal Chunks: {stats['total_chunks']:,}")

    print("\nTOKEN COUNT STATISTICS:")
    print(f"  Mean:     {stats['token_statistics']['mean']:.1f} tokens")
    print(f"  Median:   {stats['token_statistics']['median']:.1f} tokens")
    print(f"  Std Dev:  {stats['token_statistics']['std']:.1f} tokens")
    print(f"  Min:      {stats['token_statistics']['min']:,} tokens")
    print(f"  Max:      {stats['token_statistics']['max']:,} tokens")

    print("\nPERCENTILES:")
    print(f"  10th:  {stats['token_statistics']['percentile_10']:.1f} tokens")
    print(f"  25th:  {stats['token_statistics']['percentile_25']:.1f} tokens")
    print(f"  75th:  {stats['token_statistics']['percentile_75']:.1f} tokens")
    print(f"  90th:  {stats['token_statistics']['percentile_90']:.1f} tokens")
    print(f"  95th:  {stats['token_statistics']['percentile_95']:.1f} tokens")

    print("\nCHUNK SIZE DISTRIBUTION:")
    for bucket, count in stats['chunks_by_size_bucket'].items():
        pct = count / stats['total_chunks'] * 100
        print(f"  {bucket:20s}: {count:5,} ({pct:5.1f}%)")

    print("\nSMALL CHUNKS ANALYSIS:")
    print(f"  Chunks under {small_chunk_threshold} tokens: {stats['small_chunks_analysis']['count']:,} ({stats['small_chunks_analysis']['percentage']:.1f}%)")

    print("\nBY DOCUMENT TYPE:")
    for doc_type, type_stats in stats['by_document_type'].items():
        print(f"\n  {doc_type.upper()}:")
        print(f"    Count:  {type_stats['count']:,}")
        print(f"    Mean:   {type_stats['mean_tokens']:.1f} tokens")
        print(f"    Median: {type_stats['median_tokens']:.1f} tokens")
        print(f"    Range:  {type_stats['min_tokens']}-{type_stats['max_tokens']} tokens")

    # Save to JSON
    output_file = "chunk_statistics_report.json"
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\n✓ Detailed report saved to: {output_file}")

    # Create visualizations
    create_visualizations(token_counts, chunks_by_type, stats)

    return stats, token_counts, chunks_by_type

def create_visualizations(token_counts, chunks_by_type, stats):
    """Create distribution plots."""

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Overall token distribution histogram
    ax = axes[0, 0]
    ax.hist(token_counts, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(stats['token_statistics']['mean'], color='red', linestyle='--',
               label=f"Mean: {stats['token_statistics']['mean']:.1f}")
    ax.axvline(stats['token_statistics']['median'], color='green', linestyle='--',
               label=f"Median: {stats['token_statistics']['median']:.1f}")
    ax.set_xlabel('Token Count')
    ax.set_ylabel('Number of Chunks')
    ax.set_title('Overall Token Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Box plot by document type
    ax = axes[0, 1]
    data_by_type = [tokens for tokens in chunks_by_type.values()]
    labels_by_type = list(chunks_by_type.keys())
    bp = ax.boxplot(data_by_type, labels=labels_by_type, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax.set_ylabel('Token Count')
    ax.set_title('Token Distribution by Document Type')
    ax.grid(True, alpha=0.3)

    # 3. Size bucket bar chart
    ax = axes[1, 0]
    buckets = list(stats['chunks_by_size_bucket'].keys())
    counts = list(stats['chunks_by_size_bucket'].values())
    colors = ['#ff6b6b', '#ffd93d', '#6bcf7f', '#4d96ff', '#a78bfa', '#ec4899']
    bars = ax.bar(range(len(buckets)), counts, color=colors, edgecolor='black')
    ax.set_xticks(range(len(buckets)))
    ax.set_xticklabels(buckets, rotation=45, ha='right')
    ax.set_ylabel('Number of Chunks')
    ax.set_title('Chunks by Size Bucket')
    ax.grid(True, alpha=0.3, axis='y')

    # Add count labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=9)

    # 4. Cumulative distribution
    ax = axes[1, 1]
    sorted_tokens = np.sort(token_counts)
    cumulative = np.arange(1, len(sorted_tokens) + 1) / len(sorted_tokens) * 100
    ax.plot(sorted_tokens, cumulative, linewidth=2)
    ax.axhline(50, color='red', linestyle='--', alpha=0.5, label='50th percentile')
    ax.axhline(90, color='orange', linestyle='--', alpha=0.5, label='90th percentile')
    ax.set_xlabel('Token Count')
    ax.set_ylabel('Cumulative Percentage')
    ax.set_title('Cumulative Distribution of Token Counts')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = 'chunk_statistics_plots.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Visualizations saved to: {output_file}")

    # Also create a focused view for small chunks
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    small_tokens = token_counts[token_counts <= 200]
    ax2.hist(small_tokens, bins=50, edgecolor='black', alpha=0.7, color='coral')
    ax2.axvline(np.mean(small_tokens), color='red', linestyle='--',
                label=f"Mean: {np.mean(small_tokens):.1f}")
    ax2.axvline(np.median(small_tokens), color='green', linestyle='--',
                label=f"Median: {np.median(small_tokens):.1f}")
    ax2.set_xlabel('Token Count')
    ax2.set_ylabel('Number of Chunks')
    ax2.set_title('Token Distribution for Chunks ≤ 200 Tokens (Detailed View)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    output_file2 = 'small_chunks_detail.png'
    plt.savefig(output_file2, dpi=300, bbox_inches='tight')
    print(f"✓ Small chunks detail plot saved to: {output_file2}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Token/character statistics for chunks stored in Milvus Lite.")
    parser.add_argument("--db", default="milvus_all_docs.db", help="Milvus Lite database file (default: milvus_all_docs.db)")
    parser.add_argument("--collection", default="rag_all_docs", help="Collection name (default: rag_all_docs)")
    args = parser.parse_args()

    stats, token_counts, chunks_by_type = analyze_chunks(args.db, args.collection)

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
