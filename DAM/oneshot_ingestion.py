#!/usr/bin/env python3
"""
One-Shot Document Ingestion Pipeline
=====================================
Processes all files from all_data/ folder:
1. Converts to Docling documents (auto-detects format)
2. Chunks hierarchically with HierarchicalChunker
3. Extracts full visual grounding metadata (bbox, page, position, item_type)
4. Embeds with BGE-large (1024 dims)
5. Stores in Milvus 'oneshot.db' collection

Usage: python oneshot_ingestion.py
"""

import torch
from sentence_transformers import SentenceTransformer
import json
from pathlib import Path
from pymilvus import MilvusClient
from tqdm import tqdm
from docling.document_converter import DocumentConverter
from docling.chunking import HierarchicalChunker
from docling_core.types.doc import DoclingDocument
import time
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OneShotIngestion:
    """Complete document ingestion pipeline with visual grounding metadata."""

    def __init__(
        self,
        input_dir="all_data",
        db_name="oneshot.db",
        collection_name="oneshot",
        batch_size=32,
        embedding_model="BAAI/bge-large-en-v1.5"
    ):
        self.input_dir = Path(input_dir)
        self.db_name = db_name
        self.collection_name = collection_name
        self.batch_size = batch_size
        self.embedding_model_name = embedding_model
        self.embedding_dim = 1024

        # Statistics tracking
        self.stats = {
            'total_files': 0,
            'processed_files': 0,
            'failed_files': 0,
            'total_chunks': 0,
            'chunks_by_filetype': {},
            'chunks_by_file': {},
            'start_time': None,
            'end_time': None,
            'processing_time': None
        }

        # Initialize components
        self.embedding_model = None
        self.milvus_client = None
        self.converter = None
        self.chunker = None

    def check_gpu(self):
        """Check GPU availability and print info."""
        logger.info("="*80)
        logger.info("GPU CHECK")
        logger.info("="*80)
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"GPU Device: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            return "cuda"
        else:
            logger.warning("CUDA not available, using CPU (slower)")
            return "cpu"

    def load_embedding_model(self):
        """Load BGE embedding model."""
        logger.info("="*80)
        logger.info("LOADING EMBEDDING MODEL")
        logger.info("="*80)

        device = self.check_gpu()

        logger.info(f"Loading {self.embedding_model_name} on {device}...")
        self.embedding_model = SentenceTransformer(self.embedding_model_name, device=device)

        # Test embedding
        test_embedding = self.embedding_model.encode(["test"])
        actual_dim = len(test_embedding[0])

        logger.info(f"✓ Model loaded successfully!")
        logger.info(f"  Model: {self.embedding_model_name}")
        logger.info(f"  Device: {device}")
        logger.info(f"  Embedding dimension: {actual_dim}")
        logger.info(f"  Max sequence length: {self.embedding_model.max_seq_length}")

        self.embedding_dim = actual_dim

    def discover_files(self):
        """Scan input directory for all files (non-recursive)."""
        logger.info("="*80)
        logger.info("FILE DISCOVERY")
        logger.info("="*80)

        if not self.input_dir.exists():
            logger.error(f"Input directory not found: {self.input_dir}")
            raise FileNotFoundError(f"Directory {self.input_dir} does not exist")

        # Get all files (non-recursive)
        files = [f for f in self.input_dir.iterdir() if f.is_file()]

        logger.info(f"Found {len(files)} files in {self.input_dir}/")

        # Group by extension
        by_ext = {}
        for f in files:
            ext = f.suffix.lower() or "no_extension"
            by_ext[ext] = by_ext.get(ext, 0) + 1

        logger.info("\nBreakdown by file type:")
        for ext, count in sorted(by_ext.items()):
            logger.info(f"  {ext}: {count} files")

        logger.info("\nFiles to process:")
        for i, f in enumerate(files, 1):
            logger.info(f"  {i:2d}. {f.name}")

        self.stats['total_files'] = len(files)
        return files

    def initialize_converter_and_chunker(self):
        """Initialize DocumentConverter and HierarchicalChunker."""
        logger.info("="*80)
        logger.info("INITIALIZING CONVERTER & CHUNKER")
        logger.info("="*80)

        self.converter = DocumentConverter()
        self.chunker = HierarchicalChunker()

        logger.info("✓ DocumentConverter initialized")
        logger.info("✓ HierarchicalChunker initialized")

    def extract_metadata_from_chunk(self, chunk, doc, filename):
        """
        Extract visual grounding metadata from chunk.

        Returns dict with: page_number, bbox (l,t,r,b), position (x,y), item_type
        """
        metadata = {
            'filename': filename,
            'page_number': None,
            'bbox_l': None,
            'bbox_t': None,
            'bbox_r': None,
            'bbox_b': None,
            'position_x': None,
            'position_y': None,
            'item_type': 'unknown'
        }

        # Try to extract metadata from chunk's meta or provenance
        # The chunk object may have different attributes depending on Docling version

        # Try to access chunk metadata if available
        if hasattr(chunk, 'meta') and chunk.meta:
            # Extract from meta
            meta = chunk.meta
            if hasattr(meta, 'doc_items') and meta.doc_items:
                # Get first doc item for primary metadata
                first_item = meta.doc_items[0]
                if hasattr(first_item, 'prov') and first_item.prov:
                    prov = first_item.prov[0] if isinstance(first_item.prov, list) else first_item.prov

                    # Extract page number
                    if hasattr(prov, 'page_no'):
                        metadata['page_number'] = prov.page_no

                    # Extract bbox
                    if hasattr(prov, 'bbox') and prov.bbox:
                        bbox = prov.bbox
                        metadata['bbox_l'] = getattr(bbox, 'l', None)
                        metadata['bbox_t'] = getattr(bbox, 't', None)
                        metadata['bbox_r'] = getattr(bbox, 'r', None)
                        metadata['bbox_b'] = getattr(bbox, 'b', None)

                        # Calculate center position
                        if all(v is not None for v in [metadata['bbox_l'], metadata['bbox_r'],
                                                        metadata['bbox_t'], metadata['bbox_b']]):
                            metadata['position_x'] = (metadata['bbox_l'] + metadata['bbox_r']) / 2
                            metadata['position_y'] = (metadata['bbox_t'] + metadata['bbox_b']) / 2

                # Extract item type/label
                if hasattr(first_item, 'label'):
                    metadata['item_type'] = first_item.label

        return metadata

    def process_documents(self, files):
        """
        Convert files to Docling documents, chunk, and extract metadata.

        Returns list of dicts with text and metadata.
        """
        logger.info("="*80)
        logger.info("DOCUMENT CONVERSION & CHUNKING")
        logger.info("="*80)

        all_chunks = []

        for file_path in tqdm(files, desc="Processing files"):
            try:
                logger.info(f"\nProcessing: {file_path.name}")

                # Convert to Docling Document
                result = self.converter.convert(str(file_path))
                doc = result.document

                # Apply HierarchicalChunker
                chunk_iter = self.chunker.chunk(doc)
                doc_chunks = list(chunk_iter)

                logger.info(f"  ✓ Converted and chunked into {len(doc_chunks)} chunks")

                # Extract chunks with metadata
                file_type = file_path.suffix.lower() or "no_ext"
                for idx, chunk in enumerate(doc_chunks):
                    chunk_text = chunk.text

                    # Extract visual grounding metadata
                    metadata = self.extract_metadata_from_chunk(chunk, doc, file_path.name)

                    # Add chunk info
                    metadata['chunk_id'] = idx
                    metadata['total_chunks'] = len(doc_chunks)

                    all_chunks.append({
                        'text': chunk_text,
                        'metadata': metadata
                    })

                # Update statistics
                self.stats['processed_files'] += 1
                self.stats['chunks_by_file'][file_path.name] = len(doc_chunks)
                self.stats['chunks_by_filetype'][file_type] = \
                    self.stats['chunks_by_filetype'].get(file_type, 0) + len(doc_chunks)

            except Exception as e:
                logger.error(f"  ✗ Failed to process {file_path.name}: {str(e)}")
                self.stats['failed_files'] += 1
                continue

        self.stats['total_chunks'] = len(all_chunks)

        logger.info(f"\n{'='*80}")
        logger.info(f"CONVERSION COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"Files processed: {self.stats['processed_files']}/{self.stats['total_files']}")
        logger.info(f"Files failed: {self.stats['failed_files']}")
        logger.info(f"Total chunks: {len(all_chunks)}")

        logger.info(f"\nChunks by file type:")
        for ftype, count in sorted(self.stats['chunks_by_filetype'].items()):
            logger.info(f"  {ftype}: {count} chunks")

        return all_chunks

    def embed_chunks(self, chunks):
        """Batch embed all chunks using BGE model."""
        logger.info("="*80)
        logger.info("BATCH EMBEDDING")
        logger.info("="*80)

        # Extract texts
        all_texts = [chunk['text'] for chunk in chunks]

        logger.info(f"Embedding {len(all_texts)} chunks with batch_size={self.batch_size}...")
        start_time = time.time()

        # Batch encode
        embeddings = self.embedding_model.encode(
            all_texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        # Convert to list
        embeddings_list = embeddings.tolist()

        elapsed_time = time.time() - start_time

        logger.info(f"\n{'='*80}")
        logger.info(f"EMBEDDING COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"Total chunks embedded: {len(embeddings_list)}")
        logger.info(f"Time taken: {elapsed_time:.2f} seconds")
        logger.info(f"Average time per chunk: {elapsed_time/len(embeddings_list)*1000:.2f} ms")
        logger.info(f"Embedding shape: ({len(embeddings_list)}, {len(embeddings_list[0])})")

        return embeddings_list

    def setup_milvus(self):
        """Initialize Milvus client and create collection."""
        logger.info("="*80)
        logger.info("MILVUS DATABASE SETUP")
        logger.info("="*80)

        # Initialize client
        self.milvus_client = MilvusClient(self.db_name)
        logger.info(f"✓ Milvus client initialized: {self.db_name}")

        # Drop existing collection
        if self.milvus_client.has_collection(self.collection_name):
            logger.info(f"Dropping existing collection: {self.collection_name}")
            self.milvus_client.drop_collection(self.collection_name)

        # Create new collection
        logger.info(f"Creating collection: {self.collection_name}")
        self.milvus_client.create_collection(
            collection_name=self.collection_name,
            dimension=self.embedding_dim,
            metric_type="IP",  # Inner Product for cosine similarity
            consistency_level="Strong"
        )

        logger.info(f"✓ Collection created successfully!")
        logger.info(f"  Collection: {self.collection_name}")
        logger.info(f"  Dimension: {self.embedding_dim}")
        logger.info(f"  Metric: IP (Inner Product)")
        logger.info(f"  Consistency: Strong")

    def insert_data(self, chunks, embeddings):
        """Insert chunks with embeddings and metadata into Milvus."""
        logger.info("="*80)
        logger.info("DATA INSERTION")
        logger.info("="*80)

        # Prepare data
        data_to_insert = []
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            metadata = chunk['metadata']

            data_to_insert.append({
                'id': idx,
                'vector': embedding,
                'text': chunk['text'],
                'filename': metadata['filename'],
                'page_number': metadata['page_number'] or 0,
                'bbox_l': metadata['bbox_l'] or 0.0,
                'bbox_t': metadata['bbox_t'] or 0.0,
                'bbox_r': metadata['bbox_r'] or 0.0,
                'bbox_b': metadata['bbox_b'] or 0.0,
                'position_x': metadata['position_x'] or 0.0,
                'position_y': metadata['position_y'] or 0.0,
                'item_type': metadata['item_type'],
                'chunk_id': metadata['chunk_id']
            })

        logger.info(f"Inserting {len(data_to_insert)} records into Milvus...")
        start_time = time.time()

        # Insert
        insert_result = self.milvus_client.insert(
            collection_name=self.collection_name,
            data=data_to_insert
        )

        elapsed_time = time.time() - start_time

        logger.info(f"\n{'='*80}")
        logger.info(f"INSERTION COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"Records inserted: {insert_result['insert_count']}")
        logger.info(f"Time taken: {elapsed_time:.2f} seconds")

    def generate_report(self):
        """Generate and save processing report."""
        logger.info("="*80)
        logger.info("GENERATING REPORT")
        logger.info("="*80)

        # Get collection stats
        stats = self.milvus_client.get_collection_stats(self.collection_name)

        report = {
            'processing_info': {
                'timestamp': datetime.now().isoformat(),
                'input_directory': str(self.input_dir),
                'total_files_found': self.stats['total_files'],
                'files_processed': self.stats['processed_files'],
                'files_failed': self.stats['failed_files'],
                'total_chunks_created': self.stats['total_chunks'],
                'processing_time_seconds': self.stats['processing_time']
            },
            'embedding_info': {
                'model': self.embedding_model_name,
                'dimension': self.embedding_dim,
                'batch_size': self.batch_size
            },
            'milvus_info': {
                'database_file': self.db_name,
                'collection_name': self.collection_name,
                'total_vectors': stats['row_count'],
                'metric_type': 'IP'
            },
            'chunking_statistics': {
                'average_chunks_per_file': self.stats['total_chunks'] / max(self.stats['processed_files'], 1),
                'chunks_by_filetype': self.stats['chunks_by_filetype'],
                'chunks_by_file': self.stats['chunks_by_file']
            },
            'metadata_schema': {
                'filename': 'Original filename',
                'page_number': 'Page number where chunk appears',
                'bbox': 'Bounding box coordinates (l, t, r, b)',
                'position': 'Center coordinates (x, y)',
                'item_type': 'Element type (text, table, section_header, etc.)',
                'chunk_id': 'Position within document'
            }
        }

        # Save report
        report_path = Path("oneshot_ingestion_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"✓ Report saved to: {report_path}")

        # Print summary
        logger.info(f"\n{'='*80}")
        logger.info(f"INGESTION SUMMARY")
        logger.info(f"{'='*80}")
        logger.info(f"Total files processed: {self.stats['processed_files']}/{self.stats['total_files']}")
        logger.info(f"Total chunks created: {self.stats['total_chunks']}")
        logger.info(f"Average chunks per file: {report['chunking_statistics']['average_chunks_per_file']:.1f}")
        logger.info(f"Total processing time: {self.stats['processing_time']:.2f} seconds")
        logger.info(f"\nDatabase: {self.db_name}")
        logger.info(f"Collection: {self.collection_name}")
        logger.info(f"Vectors stored: {stats['row_count']}")
        logger.info(f"\nReport: {report_path}")
        logger.info(f"{'='*80}")

        return report

    def run(self):
        """Execute complete ingestion pipeline."""
        logger.info("\n" + "="*80)
        logger.info("ONE-SHOT DOCUMENT INGESTION PIPELINE")
        logger.info("="*80)

        self.stats['start_time'] = time.time()

        try:
            # Step 1: Load embedding model
            self.load_embedding_model()

            # Step 2: Discover files
            files = self.discover_files()

            if not files:
                logger.warning("No files found to process!")
                return

            # Step 3: Initialize converter and chunker
            self.initialize_converter_and_chunker()

            # Step 4: Process documents
            chunks = self.process_documents(files)

            if not chunks:
                logger.warning("No chunks created!")
                return

            # Step 5: Embed chunks
            embeddings = self.embed_chunks(chunks)

            # Step 6: Setup Milvus
            self.setup_milvus()

            # Step 7: Insert data
            self.insert_data(chunks, embeddings)

            # Calculate processing time
            self.stats['end_time'] = time.time()
            self.stats['processing_time'] = self.stats['end_time'] - self.stats['start_time']

            # Step 8: Generate report
            self.generate_report()

            logger.info("\n✓✓✓ ONE-SHOT INGESTION COMPLETE ✓✓✓\n")

        except Exception as e:
            logger.error(f"\n✗✗✗ INGESTION FAILED ✗✗✗")
            logger.error(f"Error: {str(e)}")
            raise


def main():
    """Main entry point."""
    pipeline = OneShotIngestion(
        input_dir="all_data",
        db_name="oneshot.db",
        collection_name="oneshot",
        batch_size=32
    )

    pipeline.run()


if __name__ == "__main__":
    main()
