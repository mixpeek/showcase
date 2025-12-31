#!/usr/bin/env python3
"""
GPT Baseline Pipeline
---------------------
Simple approach: PDF → PyMuPDF text extraction → OpenAI embeddings → Vector search

This represents the "naive" approach most people use.
"""

import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import fitz  # PyMuPDF
import numpy as np
from openai import OpenAI
from tqdm import tqdm


@dataclass
class Document:
    """A document chunk for indexing."""
    doc_id: str
    source_file: str
    page_number: int
    chunk_index: int
    text: str
    embedding: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None


class GPTBaselinePipeline:
    """
    Baseline pipeline using:
    - PyMuPDF for text extraction (no OCR, just embedded text)
    - OpenAI text-embedding-3-small for embeddings
    - Cosine similarity for retrieval
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        embedding_model: str = "text-embedding-3-small",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.client = OpenAI(api_key=openai_api_key or os.getenv("OPENAI_API_KEY"))
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.documents: List[Document] = []
        self.embeddings_matrix: Optional[np.ndarray] = None

    def extract_text_from_pdf(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """Extract text from PDF using PyMuPDF (no OCR)."""
        pages = []
        try:
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text("text")
                pages.append({
                    "page_number": page_num + 1,
                    "text": text.strip(),
                    "char_count": len(text),
                })
            doc.close()
        except Exception as e:
            print(f"Error extracting {pdf_path}: {e}")
        return pages

    def chunk_text(self, text: str, source_file: str, page_number: int) -> List[Document]:
        """Split text into overlapping chunks."""
        if not text or len(text) < 50:
            return []

        chunks = []
        start = 0
        chunk_idx = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]

            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk_text.rfind(". ")
                if last_period > self.chunk_size // 2:
                    chunk_text = chunk_text[: last_period + 1]
                    end = start + last_period + 1

            if chunk_text.strip():
                doc_id = hashlib.md5(
                    f"{source_file}:{page_number}:{chunk_idx}".encode()
                ).hexdigest()[:12]

                chunks.append(
                    Document(
                        doc_id=doc_id,
                        source_file=source_file,
                        page_number=page_number,
                        chunk_index=chunk_idx,
                        text=chunk_text.strip(),
                        metadata={"pipeline": "gpt_baseline"},
                    )
                )
                chunk_idx += 1

            start = end - self.chunk_overlap
            if start >= len(text):
                break

        return chunks

    def get_embedding(self, text: str) -> List[float]:
        """Get OpenAI embedding for text."""
        # Truncate if too long (8191 tokens max)
        text = text[:30000]  # Rough character limit
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text,
        )
        return response.data[0].embedding

    def get_embeddings_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """Get embeddings for multiple texts in batches."""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            # Truncate each text
            batch = [t[:30000] for t in batch]
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=batch,
            )
            all_embeddings.extend([d.embedding for d in response.data])
        return all_embeddings

    def index_pdfs(self, pdf_paths: List[Path], show_progress: bool = True) -> Dict[str, Any]:
        """Index multiple PDFs."""
        stats = {
            "total_pdfs": len(pdf_paths),
            "total_pages": 0,
            "total_chunks": 0,
            "failed_pdfs": [],
        }

        all_chunks = []

        # Extract and chunk
        iterator = tqdm(pdf_paths, desc="Extracting PDFs") if show_progress else pdf_paths
        for pdf_path in iterator:
            pages = self.extract_text_from_pdf(pdf_path)
            stats["total_pages"] += len(pages)

            for page in pages:
                if page["text"]:
                    chunks = self.chunk_text(
                        page["text"],
                        pdf_path.name,
                        page["page_number"],
                    )
                    all_chunks.extend(chunks)

        stats["total_chunks"] = len(all_chunks)

        if not all_chunks:
            print("No text extracted from PDFs!")
            return stats

        # Generate embeddings
        print(f"Generating embeddings for {len(all_chunks)} chunks...")
        texts = [c.text for c in all_chunks]
        embeddings = self.get_embeddings_batch(texts)

        for chunk, embedding in zip(all_chunks, embeddings):
            chunk.embedding = embedding

        self.documents = all_chunks
        self.embeddings_matrix = np.array(embeddings)

        return stats

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant documents."""
        if not self.documents:
            return []

        # Get query embedding
        query_embedding = np.array(self.get_embedding(query))

        # Cosine similarity
        similarities = np.dot(self.embeddings_matrix, query_embedding) / (
            np.linalg.norm(self.embeddings_matrix, axis=1) * np.linalg.norm(query_embedding)
        )

        # Get top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            doc = self.documents[idx]
            results.append({
                "doc_id": doc.doc_id,
                "source_file": doc.source_file,
                "page_number": doc.page_number,
                "text": doc.text[:500] + "..." if len(doc.text) > 500 else doc.text,
                "score": float(similarities[idx]),
                "metadata": doc.metadata,
            })

        return results

    def save_index(self, output_path: Path):
        """Save the index to disk."""
        data = {
            "documents": [asdict(d) for d in self.documents],
            "config": {
                "embedding_model": self.embedding_model,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
            },
        }
        with open(output_path, "w") as f:
            json.dump(data, f)

        # Save embeddings separately as numpy
        if self.embeddings_matrix is not None:
            np.save(output_path.with_suffix(".npy"), self.embeddings_matrix)

    def load_index(self, index_path: Path):
        """Load index from disk."""
        with open(index_path) as f:
            data = json.load(f)

        self.documents = [Document(**d) for d in data["documents"]]
        self.embeddings_matrix = np.load(index_path.with_suffix(".npy"))


def main():
    """Test the baseline pipeline."""
    import random

    # Sample 5 random PDFs
    data_dir = Path(__file__).parent.parent.parent / "data"
    pdf_files = list(data_dir.glob("*.pdf"))

    if not pdf_files:
        print("No PDFs found in data directory")
        return

    sample_pdfs = random.sample(pdf_files, min(5, len(pdf_files)))
    print(f"Testing with {len(sample_pdfs)} PDFs:")
    for p in sample_pdfs:
        print(f"  - {p.name}")

    # Initialize and index
    pipeline = GPTBaselinePipeline()
    stats = pipeline.index_pdfs(sample_pdfs)
    print(f"\nIndexing stats: {stats}")

    # Test search
    test_queries = [
        "FBI surveillance activities",
        "anti-war protests",
        "communist party connections",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print("=" * 60)
        results = pipeline.search(query, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"\n[{i}] Score: {r['score']:.4f} | {r['source_file']} (p.{r['page_number']})")
            print(f"    {r['text'][:200]}...")


if __name__ == "__main__":
    main()
