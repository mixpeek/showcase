#!/usr/bin/env python3
"""
CLIP Experiment
---------------
Test whether adding CLIP image embeddings improves retrieval on FBI vault documents.

Hypothesis: CLIP may help with:
1. Finding photos/images in documents
2. Handwritten notes where OCR fails
3. Visual queries about document structure
"""

import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import fitz
from PIL import Image
import io

sys.path.insert(0, str(Path(__file__).parent / "pipeline" / "advanced"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from sota_pipeline import SOTAPipeline, DocumentObject, BoundingBox
import random

# Import CLIP
try:
    from sentence_transformers import SentenceTransformer
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("sentence-transformers not available, skipping CLIP")


class CLIPEnhancedPipeline(SOTAPipeline):
    """Pipeline with optional CLIP image embeddings."""

    def __init__(self, use_clip: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.use_clip = use_clip and CLIP_AVAILABLE

        if self.use_clip:
            print("Loading CLIP model...")
            self.clip_model = SentenceTransformer('clip-ViT-B-32')
            print("CLIP model loaded")
        else:
            self.clip_model = None

        self.image_embeddings_matrix: Optional[np.ndarray] = None

    def _get_region_image(self, pdf_path: Path, page_num: int, bbox: BoundingBox) -> Optional[Image.Image]:
        """Extract image of a document region."""
        try:
            doc = fitz.open(pdf_path)
            page = doc[page_num - 1]

            # Render at higher DPI for better CLIP understanding
            mat = fitz.Matrix(2.0, 2.0)  # 2x zoom = ~144 DPI
            clip_rect = fitz.Rect(bbox.x0, bbox.y0, bbox.x1, bbox.y1)

            # Expand clip slightly for context
            clip_rect = clip_rect + fitz.Rect(-5, -5, 5, 5)
            clip_rect = clip_rect & page.rect  # Intersect with page bounds

            pix = page.get_pixmap(matrix=mat, clip=clip_rect)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            doc.close()
            return img
        except Exception as e:
            print(f"Error getting region image: {e}")
            return None

    def _get_clip_embedding(self, image: Image.Image) -> List[float]:
        """Get CLIP embedding for an image."""
        if not self.clip_model:
            return [0.0] * 512

        try:
            embedding = self.clip_model.encode(image)
            return embedding.tolist()
        except Exception as e:
            print(f"CLIP embedding error: {e}")
            return [0.0] * 512

    def _get_clip_text_embedding(self, text: str) -> np.ndarray:
        """Get CLIP text embedding for a query."""
        if not self.clip_model:
            return np.zeros(512)

        try:
            return self.clip_model.encode(text)
        except Exception as e:
            print(f"CLIP text embedding error: {e}")
            return np.zeros(512)

    def index_pdfs_with_clip(self, pdf_paths: List[Path], show_progress: bool = True) -> Dict[str, Any]:
        """Index PDFs with both text and CLIP embeddings."""
        # First do normal indexing
        stats = self.index_pdfs(pdf_paths, show_progress=show_progress)

        if not self.use_clip or not self.objects:
            return stats

        # Now add CLIP embeddings
        print(f"\nGenerating CLIP embeddings for {len(self.objects)} objects...")

        image_embeddings = []
        pdf_cache = {}  # Cache open PDFs

        for i, obj in enumerate(self.objects):
            if i % 100 == 0:
                print(f"  CLIP progress: {i}/{len(self.objects)}")

            # Get the PDF (cached)
            pdf_path = None
            for p in pdf_paths:
                if p.name == obj.source_file:
                    pdf_path = p
                    break

            if pdf_path:
                img = self._get_region_image(pdf_path, obj.page_number, obj.bbox)
                if img:
                    emb = self._get_clip_embedding(img)
                else:
                    emb = [0.0] * 512
            else:
                emb = [0.0] * 512

            image_embeddings.append(emb)
            obj.image_embedding = emb

        self.image_embeddings_matrix = np.array(image_embeddings)
        stats["clip_embeddings"] = len(image_embeddings)

        return stats

    def search_multimodal(
        self,
        query: str,
        top_k: int = 5,
        text_weight: float = 0.7,
        clip_weight: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """Search using both text and CLIP embeddings."""
        if not self.objects:
            return []

        # Get text-based results first (uses hybrid search internally)
        text_results = self.search(query, top_k=top_k * 2, use_hybrid=True)

        if not self.use_clip or self.image_embeddings_matrix is None:
            return text_results[:top_k]

        # Get CLIP query embedding
        clip_query = self._get_clip_text_embedding(query)

        # Compute CLIP similarities
        clip_scores = np.dot(self.image_embeddings_matrix, clip_query) / (
            np.linalg.norm(self.image_embeddings_matrix, axis=1) * np.linalg.norm(clip_query) + 1e-8
        )

        # Normalize CLIP scores to 0-1
        if clip_scores.max() > clip_scores.min():
            clip_scores = (clip_scores - clip_scores.min()) / (clip_scores.max() - clip_scores.min())

        # Get text scores for all objects
        text_scores = {}
        for r in text_results:
            text_scores[r['doc_id']] = r['score']

        # Combine scores
        combined = []
        for i, obj in enumerate(self.objects):
            text_score = text_scores.get(obj.object_id, 0.0)
            clip_score = clip_scores[i]

            combined_score = text_weight * text_score + clip_weight * clip_score

            combined.append({
                "doc_id": obj.object_id,
                "object_type": obj.object_type.value,
                "source_file": obj.source_file,
                "page_number": obj.page_number,
                "bbox": obj.bbox.to_list(),
                "text": obj.get_retrieval_text()[:500],
                "score": float(combined_score),
                "text_score": float(text_score),
                "clip_score": float(clip_score),
                "confidence_tag": obj.confidence_tag.value,
            })

        # Sort by combined score
        combined.sort(key=lambda x: x['score'], reverse=True)
        return combined[:top_k]


# Test queries - some should benefit from CLIP, some shouldn't
TEST_QUERIES = [
    # Text-heavy queries (CLIP may not help much)
    ("What did the FBI know about Abbie Hoffman?", "text"),
    ("surveillance of anti-war activists", "text"),
    ("communist party infiltration", "text"),

    # Visual/structural queries (CLIP might help)
    ("organizational charts", "visual"),
    ("photographs of protesters", "visual"),
    ("handwritten notes", "visual"),
    ("FBI letterhead documents", "visual"),
    ("redacted sections", "visual"),

    # Mixed queries
    ("diagrams showing connections between groups", "mixed"),
    ("forms with personal information", "mixed"),
]


def run_experiment():
    """Run CLIP vs text-only comparison."""
    print("=" * 80)
    print("CLIP EXPERIMENT: Text-only vs Multi-modal Retrieval")
    print("=" * 80)

    data_dir = Path(__file__).parent / "data"
    pdf_files = list(data_dir.glob("*.pdf"))

    if not pdf_files:
        print("No PDFs found")
        return

    # Use smaller sample for faster experimentation
    random.seed(42)
    sample_pdfs = random.sample(pdf_files, min(5, len(pdf_files)))

    print(f"\nUsing {len(sample_pdfs)} PDFs for experiment")
    for p in sample_pdfs:
        print(f"  - {p.name}")

    # Initialize pipelines
    print("\n" + "-" * 80)
    print("Indexing with text-only pipeline...")
    text_only = SOTAPipeline(fast_mode=True)
    text_stats = text_only.index_pdfs(sample_pdfs)
    print(f"  Objects indexed: {text_stats['total_objects']}")

    print("\n" + "-" * 80)
    print("Indexing with CLIP-enhanced pipeline...")
    clip_enhanced = CLIPEnhancedPipeline(use_clip=True, fast_mode=True)
    clip_stats = clip_enhanced.index_pdfs_with_clip(sample_pdfs)
    print(f"  Objects indexed: {clip_stats['total_objects']}")
    print(f"  CLIP embeddings: {clip_stats.get('clip_embeddings', 0)}")

    # Run queries and compare
    print("\n" + "=" * 80)
    print("QUERY COMPARISON")
    print("=" * 80)

    results_summary = []

    for query, query_type in TEST_QUERIES:
        print(f"\n{'─' * 80}")
        print(f"QUERY: \"{query}\" [{query_type}]")
        print("─" * 80)

        # Text-only results
        text_results = text_only.search(query, top_k=3)

        # Multi-modal results
        clip_results = clip_enhanced.search_multimodal(query, top_k=3)

        print("\n▶ TEXT-ONLY (top result):")
        if text_results:
            r = text_results[0]
            print(f"  Score: {r['score']:.4f}")
            print(f"  Source: {r['source_file']} p.{r['page_number']}")
            print(f"  Text: {r['text'][:150]}...")
        else:
            print("  No results")

        print("\n▶ CLIP-ENHANCED (top result):")
        if clip_results:
            r = clip_results[0]
            print(f"  Score: {r['score']:.4f} (text: {r['text_score']:.3f}, clip: {r['clip_score']:.3f})")
            print(f"  Source: {r['source_file']} p.{r['page_number']}")
            print(f"  Text: {r['text'][:150]}...")
        else:
            print("  No results")

        # Check if results differ
        if text_results and clip_results:
            same_top = text_results[0]['doc_id'] == clip_results[0]['doc_id']
            print(f"\n  Same top result: {'Yes' if same_top else 'NO - CLIP changed ranking'}")

            results_summary.append({
                "query": query,
                "type": query_type,
                "text_top_score": text_results[0]['score'],
                "clip_top_score": clip_results[0]['score'],
                "same_top": same_top,
                "clip_boost": clip_results[0]['clip_score'],
            })

    # Summary
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)

    changed = sum(1 for r in results_summary if not r['same_top'])
    print(f"\nQueries where CLIP changed top result: {changed}/{len(results_summary)}")

    # By query type
    for qtype in ["text", "visual", "mixed"]:
        type_results = [r for r in results_summary if r['type'] == qtype]
        if type_results:
            avg_clip_boost = sum(r['clip_boost'] for r in type_results) / len(type_results)
            changed_type = sum(1 for r in type_results if not r['same_top'])
            print(f"\n{qtype.upper()} queries:")
            print(f"  - Changed rankings: {changed_type}/{len(type_results)}")
            print(f"  - Avg CLIP score contribution: {avg_clip_boost:.3f}")


if __name__ == "__main__":
    run_experiment()
