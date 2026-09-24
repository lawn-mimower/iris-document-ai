#!/usr/bin/env python3
"""
Enhanced Visual Grounding with Document Highlighting
====================================================
Retrieves chunks from Milvus and highlights them on the original documents.
Supports: PDF, Images (JPG/PNG), with bounding box visualization.
"""

import argparse
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import BytesIO
import numpy as np

class VisualGrounder:
    """Retrieve chunks and visualize them on source documents."""

    def __init__(self, db_path="oneshot.db", collection_name="oneshot", data_dir="all_data"):
        self.model = SentenceTransformer("BAAI/bge-large-en-v1.5")
        self.client = MilvusClient(db_path)
        self.collection_name = collection_name
        self.data_dir = Path(data_dir)

    def search_and_visualize(self, query, limit=5, output_dir="visual_grounding_outputs"):
        """Search for chunks and visualize them on source documents."""

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        print(f"\n{'='*80}")
        print(f"QUERY: {query}")
        print(f"{'='*80}\n")

        # Encode query and search
        query_embedding = self.model.encode([query])[0].tolist()
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            limit=limit,
            output_fields=["text", "source_file", "filename", "doc_type", "page_number",
                          "bbox_l", "bbox_t", "bbox_r", "bbox_b", "item_type"]
        )

        if not results or not results[0]:
            print("No results found.")
            return

        # Process each result
        for idx, result in enumerate(results[0], 1):
            entity = result['entity']
            score = result['distance']

            print(f"\n--- Result {idx} (Score: {score:.4f}) ---")
            print(f"File: {entity.get('source_file') or entity.get('filename', 'N/A')}")
            print(f"Type: {entity.get('doc_type', 'N/A')}")
            print(f"Page: {entity.get('page_number', 'N/A')}")
            print(f"Item Type: {entity.get('item_type', 'N/A')}")
            print(f"BBox: ({entity.get('bbox_l', 0):.2f}, {entity.get('bbox_t', 0):.2f}, "
                  f"{entity.get('bbox_r', 0):.2f}, {entity.get('bbox_b', 0):.2f})")
            print(f"Text: {entity.get('text', 'N/A')[:200]}...")

            # Visualize on source document
            try:
                self.visualize_chunk(entity, idx, query, output_path)
            except Exception as e:
                print(f"  ⚠ Error visualizing: {e}")

    def visualize_chunk(self, entity, result_idx, query, output_path):
        """Visualize a single chunk on its source document."""

        # Collections built by oneshot_ingestion.py store 'filename' (with extension) and no 'doc_type'
        source_file = entity.get('source_file') or entity.get('filename', '')
        doc_type = (entity.get('doc_type') or Path(source_file).suffix.lstrip('.')).lower()
        page_num = entity.get('page_number', 0)

        # Find the source file
        source_path = None
        for ext in ["", f".{doc_type}", ".pdf", ".jpg", ".png", ".xlsx", ".docx"]:
            potential_path = self.data_dir / f"{source_file}{ext}"
            if potential_path.exists():
                source_path = potential_path
                break

        if not source_path:
            print(f"  ⚠ Source file not found: {source_file}")
            return

        # Handle different file types
        if doc_type == 'pdf':
            self.visualize_pdf(source_path, entity, result_idx, query, output_path)
        elif doc_type in ['jpg', 'png', 'jpeg']:
            self.visualize_image(source_path, entity, result_idx, query, output_path)
        else:
            print(f"  ⓘ Visualization not supported for {doc_type} files")

    def visualize_pdf(self, pdf_path, entity, result_idx, query, output_path):
        """Highlight bbox on PDF page."""

        page_num = int(entity.get('page_number', 0))
        page_idx = page_num - 1 if page_num > 0 else 0  # docling page numbers start at 1
        bbox = [
            entity.get('bbox_l', 0),
            entity.get('bbox_t', 0),
            entity.get('bbox_r', 0),
            entity.get('bbox_b', 0)
        ]

        # Open PDF
        doc = fitz.open(pdf_path)

        if page_idx < 0 or page_idx >= len(doc):
            print(f"  ⚠ Invalid page number: {page_num}")
            doc.close()
            return

        page = doc[page_idx]

        # docling PDF boxes use a bottom-left origin (top > bottom); convert to top-left
        if bbox[1] > bbox[3]:
            page_height = page.rect.height
            bbox = [bbox[0], page_height - bbox[1], bbox[2], page_height - bbox[3]]

        # Convert page to image
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 16))
        ax.imshow(img)

        # Scale bbox coordinates to image size
        # PDF coordinates are in points, need to scale by the zoom factor
        zoom = 2.0
        rect = patches.Rectangle(
            (bbox[0] * zoom, bbox[1] * zoom),
            (bbox[2] - bbox[0]) * zoom,
            (bbox[3] - bbox[1]) * zoom,
            linewidth=3,
            edgecolor='red',
            facecolor='yellow',
            alpha=0.3
        )
        ax.add_patch(rect)

        # Add annotation
        ax.text(
            bbox[0] * zoom,
            bbox[1] * zoom - 10,
            f"Result {result_idx}",
            color='red',
            fontsize=12,
            fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )

        ax.axis('off')
        ax.set_title(f"Query: {query[:80]}...\nPage {page_idx + 1} of {pdf_path.name}",
                    fontsize=10, pad=20)

        # Save
        output_file = output_path / f"result_{result_idx}_{pdf_path.stem}_page{page_num}.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Saved visualization: {output_file}")

        doc.close()

    def visualize_image(self, img_path, entity, result_idx, query, output_path):
        """Highlight bbox on image."""

        bbox = [
            entity.get('bbox_l', 0),
            entity.get('bbox_t', 0),
            entity.get('bbox_r', 0),
            entity.get('bbox_b', 0)
        ]

        # Open image
        img = Image.open(img_path)

        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 16))
        ax.imshow(img)

        # Add bounding box
        rect = patches.Rectangle(
            (bbox[0], bbox[1]),
            bbox[2] - bbox[0],
            bbox[3] - bbox[1],
            linewidth=3,
            edgecolor='red',
            facecolor='yellow',
            alpha=0.3
        )
        ax.add_patch(rect)

        # Add annotation
        ax.text(
            bbox[0],
            bbox[1] - 10,
            f"Result {result_idx}",
            color='red',
            fontsize=12,
            fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )

        ax.axis('off')
        ax.set_title(f"Query: {query[:80]}...\n{img_path.name}",
                    fontsize=10, pad=20)

        # Save
        output_file = output_path / f"result_{result_idx}_{img_path.stem}.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Saved visualization: {output_file}")


def main():
    """Main execution."""

    parser = argparse.ArgumentParser(description="Retrieve chunks and highlight them on the source documents.")
    parser.add_argument("--db", default="oneshot.db", help="Milvus Lite database file (default: oneshot.db)")
    parser.add_argument("--collection", default="oneshot", help="Collection name (default: oneshot)")
    parser.add_argument("--data-dir", default="all_data", help="Folder with the original documents (default: all_data)")
    parser.add_argument("--output-dir", default="visual_grounding_outputs", help="Where to save the images")
    parser.add_argument("--query", "-q", action="append", help="Query text (repeatable; default: built-in form field queries)")
    parser.add_argument("--limit", type=int, default=3, help="Results per query (default: 3)")
    args = parser.parse_args()

    grounder = VisualGrounder(db_path=args.db, collection_name=args.collection, data_dir=args.data_dir)

    # Define queries
    queries = args.query or [
        "Corporate Identity Number (CIN) of company",
        "Name of the company",
        "Address of the registered office of the company",
        "Financial year to which financial statement relates",
        "Date of Board of Director's meeting in which financial statements are approved",
        "Whether Annual General Meeting (AGM) held",
        "Type of Industry",
        "Whether consolidated Financial Statements are also being filed",
        "Authorised capital of the company",
        "Whether Secretarial Audit is applicable"
    ]

    print("\n" + "="*80)
    print("VISUAL GROUNDING: QUERY TO SOURCE DOCUMENT HIGHLIGHTING")
    print("="*80)

    # Process each query
    for query in queries:
        grounder.search_and_visualize(query, limit=args.limit, output_dir=args.output_dir)

    print("\n" + "="*80)
    print(f"COMPLETED: Check '{args.output_dir}/' folder for highlighted documents")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
