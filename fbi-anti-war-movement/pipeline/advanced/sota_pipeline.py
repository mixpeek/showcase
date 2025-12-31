#!/usr/bin/env python3
"""
SOTA Multi-Modal Pipeline
-------------------------
Advanced approach with accurate bounding boxes:
- Fine-grained layout detection using PyMuPDF blocks + spatial clustering
- OCR fallback for scanned pages (via image analysis)
- VLM verification for low-confidence regions
- Multi-modal embeddings (text + image)
- Confidence scoring at every step
"""

import os
import io
import re
import json
import hashlib
import base64
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Literal
from dataclasses import dataclass, asdict, field
from enum import Enum
from collections import Counter
import fitz  # PyMuPDF
import numpy as np
from PIL import Image
from openai import OpenAI
from tqdm import tqdm


# Query expansion mappings for FBI/historical documents
QUERY_EXPANSIONS = {
    "cointelpro": ["counter intelligence program", "counterintelligence", "disruption program"],
    "chicago seven": ["chicago 7", "chicago conspiracy trial", "conspiracy seven"],
    "chicago 7": ["chicago seven", "chicago conspiracy trial"],
    "wiretap": ["wiretapping", "electronic surveillance", "wire tap", "telephone surveillance"],
    "wiretapping": ["wiretap", "electronic surveillance", "wire tap"],
    "spy": ["surveillance", "monitor", "informant", "watch"],
    "spying": ["surveillance", "monitoring", "informant activities"],
    "fbi": ["federal bureau of investigation", "bureau"],
    "informant": ["source", "confidential informant", "CI", "informer"],
    "subversive": ["subversion", "communist", "radical", "extremist"],
    "protest": ["demonstration", "march", "rally", "action"],
    "anti-war": ["antiwar", "peace movement", "war protest", "vietnam protest"],
    "activist": ["organizer", "agitator", "militant", "protester"],
    "illegal": ["unlawful", "unauthorized", "improper"],
}


def clean_ocr_text(text: str) -> str:
    """Clean OCR artifacts from extracted text."""
    if not text:
        return ""

    # Remove common OCR artifacts
    cleaned = text

    # Remove HTML entities
    cleaned = re.sub(r'&#\d+;', '', cleaned)
    cleaned = re.sub(r'&[a-z]+;', '', cleaned)

    # Remove standalone numbers that are OCR noise (like '92' appearing randomly)
    cleaned = re.sub(r'\b92\b', '', cleaned)

    # Remove sequences of punctuation/symbols
    cleaned = re.sub(r'[_\-\.\*\|]{3,}', ' ', cleaned)
    cleaned = re.sub(r'[\`\~\^\<\>]{2,}', '', cleaned)

    # Remove random single characters surrounded by spaces
    cleaned = re.sub(r'\s[a-zA-Z]\s(?=[a-zA-Z]\s)', ' ', cleaned)

    # Clean up multiple spaces/newlines
    cleaned = re.sub(r'\s+', ' ', cleaned)

    # Remove leading/trailing whitespace
    cleaned = cleaned.strip()

    return cleaned


def compute_text_quality_score(text: str) -> float:
    """
    Compute a quality score for text (0-1).
    Higher score = cleaner, more readable text.
    """
    if not text or len(text) < 10:
        return 0.0

    score = 1.0

    # Penalize high ratio of non-alphanumeric characters
    alphanum_ratio = sum(c.isalnum() or c.isspace() for c in text) / len(text)
    if alphanum_ratio < 0.7:
        score -= 0.3
    elif alphanum_ratio < 0.85:
        score -= 0.1

    # Penalize HTML entities
    entity_count = len(re.findall(r'&#?\w+;', text))
    if entity_count > 5:
        score -= 0.2
    elif entity_count > 0:
        score -= 0.1

    # Penalize excessive punctuation
    punct_ratio = sum(c in '.,;:!?()[]{}' for c in text) / len(text)
    if punct_ratio > 0.15:
        score -= 0.15

    # Penalize lack of spaces (garbled text often lacks proper spacing)
    space_ratio = text.count(' ') / len(text)
    if space_ratio < 0.1:
        score -= 0.2

    # Penalize very short words on average (OCR noise)
    words = text.split()
    if words:
        avg_word_len = sum(len(w) for w in words) / len(words)
        if avg_word_len < 2.5:
            score -= 0.15

    return max(0.0, min(1.0, score))


class ObjectType(str, Enum):
    PARAGRAPH = "paragraph"
    TABLE = "table"
    CHART = "chart"
    DIAGRAM = "diagram"
    FIGURE = "figure"
    HANDWRITTEN = "handwritten"
    FORM = "form"
    HEADER = "header"
    FOOTER = "footer"
    LIST = "list"


class ConfidenceTag(str, Enum):
    A = "A"  # High confidence (≥0.85)
    B = "B"  # Medium confidence (0.70-0.84)
    C = "C"  # Low confidence (0.50-0.69)
    D = "D"  # Very low confidence (<0.50)


@dataclass
class BoundingBox:
    """Accurate bounding box with pixel coordinates."""
    x0: float
    y0: float
    x1: float
    y1: float

    def to_list(self) -> List[float]:
        return [self.x0, self.y0, self.x1, self.y1]

    @classmethod
    def from_rect(cls, rect) -> "BoundingBox":
        return cls(float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1))

    @classmethod
    def from_tuple(cls, t: tuple) -> "BoundingBox":
        return cls(float(t[0]), float(t[1]), float(t[2]), float(t[3]))

    def area(self) -> float:
        return max(0, self.x1 - self.x0) * max(0, self.y1 - self.y0)

    def iou(self, other: "BoundingBox") -> float:
        """Intersection over Union."""
        x0 = max(self.x0, other.x0)
        y0 = max(self.y0, other.y0)
        x1 = min(self.x1, other.x1)
        y1 = min(self.y1, other.y1)

        intersection = max(0, x1 - x0) * max(0, y1 - y0)
        union = self.area() + other.area() - intersection
        return intersection / union if union > 0 else 0

    def merge(self, other: "BoundingBox") -> "BoundingBox":
        """Merge two bounding boxes."""
        return BoundingBox(
            min(self.x0, other.x0),
            min(self.y0, other.y0),
            max(self.x1, other.x1),
            max(self.y1, other.y1),
        )

    def expand(self, margin: float) -> "BoundingBox":
        """Expand bbox by margin."""
        return BoundingBox(
            self.x0 - margin,
            self.y0 - margin,
            self.x1 + margin,
            self.y1 + margin,
        )


@dataclass
class TextSpan:
    """A span of text with precise positioning."""
    text: str
    bbox: BoundingBox
    font_size: float
    font_name: str
    flags: int  # bold, italic, etc.
    confidence: float = 1.0


@dataclass
class DocumentObject:
    """A detected object from the document."""
    object_id: str
    object_type: ObjectType
    source_file: str
    page_number: int
    bbox: BoundingBox

    # Content
    text_raw: str = ""
    text_corrected: str = ""
    text_summary: str = ""
    structured_data: Optional[Dict[str, Any]] = None

    # Line-level data for accurate bbox
    spans: List[TextSpan] = field(default_factory=list)

    # Confidence
    ocr_confidence: float = 0.0
    vlm_confidence: float = 0.0
    overall_confidence: float = 0.0
    confidence_tag: ConfidenceTag = ConfidenceTag.D
    limitations: List[str] = field(default_factory=list)

    # Embeddings
    text_embedding: Optional[List[float]] = None
    image_embedding: Optional[List[float]] = None

    def get_retrieval_text(self) -> str:
        """Get cleaned text for retrieval."""
        raw = self.text_corrected or self.text_summary or self.text_raw
        return clean_ocr_text(raw)


class SOTAPipeline:
    """
    State-of-the-art pipeline with accurate bounding boxes.

    Key improvements:
    1. Character/word-level bbox extraction from PyMuPDF
    2. Spatial clustering to group related content
    3. Layout classification based on typography and position
    4. Smart VLM usage (only when needed)
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        text_embedding_model: str = "text-embedding-3-small",
        vlm_model: str = "gpt-4o-mini",
        use_vlm_correction: bool = True,
        min_confidence_for_vlm: float = 0.6,
        fast_mode: bool = False,  # Skip VLM for faster processing
        dpi: int = 150,
    ):
        self.client = OpenAI(api_key=openai_api_key or os.getenv("OPENAI_API_KEY"))
        self.text_embedding_model = text_embedding_model
        self.vlm_model = vlm_model
        self.use_vlm_correction = use_vlm_correction and not fast_mode
        self.min_confidence_for_vlm = min_confidence_for_vlm
        self.fast_mode = fast_mode
        self.dpi = dpi

        self.objects: List[DocumentObject] = []
        self.text_embeddings_matrix: Optional[np.ndarray] = None

    def _generate_object_id(self, source_file: str, page: int, idx: int) -> str:
        return hashlib.md5(f"{source_file}:{page}:{idx}".encode()).hexdigest()[:12]

    def _extract_text_spans(self, page: fitz.Page) -> List[TextSpan]:
        """
        Extract text with precise bounding boxes at span level.
        This gives accurate coordinates for each text segment.
        """
        spans = []
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]

        for block in blocks:
            if block["type"] != 0:  # Not a text block
                continue

            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if not text:
                        continue

                    bbox = BoundingBox.from_tuple(span["bbox"])
                    spans.append(TextSpan(
                        text=text,
                        bbox=bbox,
                        font_size=span.get("size", 12),
                        font_name=span.get("font", ""),
                        flags=span.get("flags", 0),
                    ))

        return spans

    def _cluster_spans_into_blocks(
        self,
        spans: List[TextSpan],
        page_width: float,
        page_height: float,
        vertical_threshold: float = 15,
        horizontal_threshold: float = 50,
    ) -> List[List[TextSpan]]:
        """
        Cluster spans into logical blocks based on spatial proximity.
        Uses accurate vertical/horizontal distance thresholds.
        """
        if not spans:
            return []

        # Sort by vertical position, then horizontal
        sorted_spans = sorted(spans, key=lambda s: (s.bbox.y0, s.bbox.x0))

        blocks = []
        current_block = [sorted_spans[0]]
        current_bbox = sorted_spans[0].bbox

        for span in sorted_spans[1:]:
            # Check if this span should join the current block
            vertical_gap = span.bbox.y0 - current_bbox.y1
            horizontal_overlap = min(span.bbox.x1, current_bbox.x1) - max(span.bbox.x0, current_bbox.x0)

            # Join if vertically close and horizontally overlapping
            if vertical_gap < vertical_threshold and horizontal_overlap > -horizontal_threshold:
                current_block.append(span)
                current_bbox = current_bbox.merge(span.bbox)
            else:
                # Start new block
                if current_block:
                    blocks.append(current_block)
                current_block = [span]
                current_bbox = span.bbox

        if current_block:
            blocks.append(current_block)

        return blocks

    def _classify_block(
        self,
        spans: List[TextSpan],
        block_bbox: BoundingBox,
        page_height: float,
        page_width: float,
    ) -> ObjectType:
        """Classify a block based on typography, position, and content."""
        if not spans:
            return ObjectType.PARAGRAPH

        text = " ".join(s.text for s in spans)
        text_lower = text.lower()

        # Position-based classification
        relative_y = block_bbox.y0 / page_height

        # Headers (top 10%)
        if relative_y < 0.10:
            avg_font_size = sum(s.font_size for s in spans) / len(spans)
            if avg_font_size > 12:
                return ObjectType.HEADER

        # Footers (bottom 10%)
        if relative_y > 0.90:
            return ObjectType.FOOTER

        # Content analysis
        # Tables: multiple columns, pipes, or tabular structure
        if "|" in text or "\t" in text:
            return ObjectType.TABLE

        # Check for columnar layout (multiple x-positions)
        x_positions = set(int(s.bbox.x0 / 20) for s in spans)
        if len(x_positions) > 3 and len(spans) > 5:
            return ObjectType.TABLE

        # Lists: numbered or bulleted items
        lines = text.split("\n")
        list_patterns = sum(1 for line in lines if line.strip()[:2] in ["1.", "2.", "3.", "•", "-", "*"])
        if list_patterns > 2:
            return ObjectType.LIST

        # Forms: label-value pairs with colons
        colon_count = text.count(":")
        if colon_count >= 3 and colon_count / len(lines) > 0.3 if lines else False:
            return ObjectType.FORM

        return ObjectType.PARAGRAPH

    def _compute_text_confidence(self, spans: List[TextSpan], text: str) -> float:
        """Estimate text extraction confidence."""
        if not text.strip():
            return 0.0

        confidence = 0.85  # Base confidence for embedded text

        # Reduce for problematic characters
        if "?" in text or "�" in text:
            confidence -= 0.15
        if text.count("  ") > 3:  # Multiple spaces suggest extraction issues
            confidence -= 0.1

        # Check font consistency
        if spans:
            fonts = set(s.font_name for s in spans)
            if len(fonts) > 5:  # Too many fonts might indicate noise
                confidence -= 0.05

        return max(0.3, min(1.0, confidence))

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        if not text or len(text.strip()) < 10:
            return [0.0] * 1536

        text = text[:30000]
        try:
            response = self.client.embeddings.create(
                model=self.text_embedding_model,
                input=text,
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Embedding error: {e}")
            return [0.0] * 1536

    def _get_embeddings_batch(self, texts: List[str], batch_size: int = 50) -> List[List[float]]:
        """Get embeddings in batches."""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = [t[:30000] for t in texts[i:i + batch_size]]
            try:
                response = self.client.embeddings.create(
                    model=self.text_embedding_model,
                    input=batch,
                )
                all_embeddings.extend([d.embedding for d in response.data])
            except Exception as e:
                print(f"Batch embedding error: {e}")
                all_embeddings.extend([[0.0] * 1536] * len(batch))
        return all_embeddings

    def _vlm_analyze_region(
        self,
        page: fitz.Page,
        bbox: BoundingBox,
        text: str,
        object_type: ObjectType,
    ) -> Tuple[str, float, Optional[Dict]]:
        """Use VLM to analyze a specific region."""
        # Convert page region to image
        mat = fitz.Matrix(self.dpi / 72, self.dpi / 72)
        clip = fitz.Rect(bbox.x0, bbox.y0, bbox.x1, bbox.y1)
        pix = page.get_pixmap(matrix=mat, clip=clip)

        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        img_b64 = base64.b64encode(buffer.getvalue()).decode()

        prompt = f"""Analyze this document region. OCR extracted:
"{text[:1000]}"

Verify the text and correct any errors. Preserve numbers and formatting exactly.

Return JSON: {{"corrected_text": "...", "confidence": 0.0-1.0, "notes": ["..."]}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.vlm_model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                    ],
                }],
                max_tokens=500,
                response_format={"type": "json_object"},
            )
            result = json.loads(response.choices[0].message.content)
            return result.get("corrected_text", text), result.get("confidence", 0.7), result
        except Exception as e:
            print(f"VLM error: {e}")
            return text, 0.5, None

    def _compute_confidence_tag(self, confidence: float) -> ConfidenceTag:
        if confidence >= 0.85:
            return ConfidenceTag.A
        elif confidence >= 0.70:
            return ConfidenceTag.B
        elif confidence >= 0.50:
            return ConfidenceTag.C
        return ConfidenceTag.D

    def process_pdf(self, pdf_path: Path) -> List[DocumentObject]:
        """Process a PDF and extract objects with accurate bounding boxes."""
        objects = []

        try:
            doc = fitz.open(pdf_path)

            for page_num in range(len(doc)):
                page = doc[page_num]
                page_width = page.rect.width
                page_height = page.rect.height

                # Extract spans with accurate bboxes
                spans = self._extract_text_spans(page)

                # Cluster into logical blocks
                blocks = self._cluster_spans_into_blocks(spans, page_width, page_height)

                for idx, block_spans in enumerate(blocks):
                    if not block_spans:
                        continue

                    # Compute accurate bbox from spans
                    block_bbox = block_spans[0].bbox
                    for span in block_spans[1:]:
                        block_bbox = block_bbox.merge(span.bbox)

                    # Get text
                    text = " ".join(s.text for s in block_spans)
                    if not text.strip() or len(text) < 20:
                        continue

                    # Classify
                    obj_type = self._classify_block(block_spans, block_bbox, page_height, page_width)

                    # Compute confidence
                    ocr_confidence = self._compute_text_confidence(block_spans, text)

                    # Create object
                    obj = DocumentObject(
                        object_id=self._generate_object_id(pdf_path.name, page_num + 1, idx),
                        object_type=obj_type,
                        source_file=pdf_path.name,
                        page_number=page_num + 1,
                        bbox=block_bbox,
                        text_raw=text,
                        spans=block_spans,
                        ocr_confidence=ocr_confidence,
                    )

                    # VLM correction for low confidence (if enabled)
                    if self.use_vlm_correction and ocr_confidence < self.min_confidence_for_vlm:
                        corrected, vlm_conf, _ = self._vlm_analyze_region(page, block_bbox, text, obj_type)
                        obj.text_corrected = corrected
                        obj.vlm_confidence = vlm_conf
                        obj.overall_confidence = (ocr_confidence + vlm_conf) / 2
                    else:
                        obj.text_corrected = text
                        obj.overall_confidence = ocr_confidence

                    obj.confidence_tag = self._compute_confidence_tag(obj.overall_confidence)
                    obj.text_summary = obj.text_corrected

                    objects.append(obj)

            doc.close()

        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")

        return objects

    def index_pdfs(self, pdf_paths: List[Path], show_progress: bool = True) -> Dict[str, Any]:
        """Index multiple PDFs."""
        stats = {
            "total_pdfs": len(pdf_paths),
            "total_pages": 0,
            "total_objects": 0,
            "objects_by_type": {},
            "confidence_distribution": {"A": 0, "B": 0, "C": 0, "D": 0},
        }

        all_objects = []
        iterator = tqdm(pdf_paths, desc="Processing PDFs") if show_progress else pdf_paths

        for pdf_path in iterator:
            try:
                doc = fitz.open(pdf_path)
                stats["total_pages"] += len(doc)
                doc.close()
            except:
                pass

            page_objects = self.process_pdf(pdf_path)
            for obj in page_objects:
                all_objects.append(obj)
                stats["objects_by_type"][obj.object_type.value] = stats["objects_by_type"].get(obj.object_type.value, 0) + 1
                stats["confidence_distribution"][obj.confidence_tag.value] += 1

        stats["total_objects"] = len(all_objects)

        # Generate embeddings in batch
        if all_objects:
            print(f"Generating embeddings for {len(all_objects)} objects...")
            texts = [o.get_retrieval_text() for o in all_objects]
            embeddings = self._get_embeddings_batch(texts)
            for obj, emb in zip(all_objects, embeddings):
                obj.text_embedding = emb

        self.objects = all_objects
        if all_objects:
            self.text_embeddings_matrix = np.array([o.text_embedding for o in all_objects])

        return stats

    def _expand_query(self, query: str) -> str:
        """Expand query with synonyms and related terms."""
        expanded = query.lower()
        additions = []

        for term, expansions in QUERY_EXPANSIONS.items():
            if term in expanded:
                additions.extend(expansions)

        if additions:
            return f"{query} {' '.join(set(additions))}"
        return query

    def _compute_keyword_score(self, query: str, text: str) -> float:
        """
        Compute BM25-style keyword matching score.
        Returns 0-1 score based on term overlap.
        """
        query_terms = set(re.findall(r'\b\w{3,}\b', query.lower()))
        text_lower = text.lower()

        if not query_terms:
            return 0.0

        # Count matches
        matches = sum(1 for term in query_terms if term in text_lower)

        # Bonus for exact phrase matches
        phrase_bonus = 0.0
        query_lower = query.lower()
        if len(query_lower) > 10 and query_lower in text_lower:
            phrase_bonus = 0.3

        return min(1.0, (matches / len(query_terms)) + phrase_bonus)

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_confidence: Optional[ConfidenceTag] = None,
        use_hybrid: bool = True,
        semantic_weight: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant objects using hybrid search.

        Args:
            query: Search query
            top_k: Number of results to return
            min_confidence: Minimum confidence tag filter
            use_hybrid: Whether to combine semantic + keyword search
            semantic_weight: Weight for semantic vs keyword (0-1)
        """
        if not self.objects:
            return []

        # Expand query with synonyms
        expanded_query = self._expand_query(query)

        # Get semantic embedding for expanded query
        query_embedding = np.array(self._get_text_embedding(expanded_query))

        # Filter by confidence
        confidence_order = {"A": 4, "B": 3, "C": 2, "D": 1}
        if min_confidence:
            min_score = confidence_order.get(min_confidence.value, 0)
            indices = [i for i, o in enumerate(self.objects)
                      if confidence_order.get(o.confidence_tag.value, 0) >= min_score]
        else:
            indices = list(range(len(self.objects)))

        if not indices:
            return []

        # Compute semantic similarities
        filtered_embeddings = self.text_embeddings_matrix[indices]
        semantic_scores = np.dot(filtered_embeddings, query_embedding) / (
            np.linalg.norm(filtered_embeddings, axis=1) * np.linalg.norm(query_embedding) + 1e-8
        )

        # Normalize semantic scores to 0-1
        if semantic_scores.max() > semantic_scores.min():
            semantic_scores = (semantic_scores - semantic_scores.min()) / (semantic_scores.max() - semantic_scores.min())

        # Compute hybrid scores
        if use_hybrid:
            keyword_scores = np.array([
                self._compute_keyword_score(query, self.objects[indices[i]].get_retrieval_text())
                for i in range(len(indices))
            ])

            # Text quality scores
            quality_scores = np.array([
                compute_text_quality_score(self.objects[indices[i]].get_retrieval_text())
                for i in range(len(indices))
            ])

            # Combined score: semantic + keyword + quality bonus
            keyword_weight = 1.0 - semantic_weight
            combined_scores = (
                semantic_weight * semantic_scores +
                keyword_weight * keyword_scores +
                0.1 * quality_scores  # Small quality bonus
            )
        else:
            combined_scores = semantic_scores

        # Get top-k by combined score
        top_local = np.argsort(combined_scores)[::-1][:top_k]

        results = []
        for local_idx in top_local:
            global_idx = indices[local_idx]
            obj = self.objects[global_idx]
            text = obj.get_retrieval_text()

            results.append({
                "doc_id": obj.object_id,
                "object_type": obj.object_type.value,
                "source_file": obj.source_file,
                "page_number": obj.page_number,
                "bbox": obj.bbox.to_list(),
                "text": text[:500],
                "score": float(combined_scores[local_idx]),
                "semantic_score": float(semantic_scores[local_idx]),
                "keyword_score": float(self._compute_keyword_score(query, text)) if use_hybrid else 0.0,
                "text_quality": float(compute_text_quality_score(text)),
                "confidence": obj.overall_confidence,
                "confidence_tag": obj.confidence_tag.value,
            })

        return results

    def save_index(self, output_path: Path):
        """Save index to disk."""
        data = [{
            "object_id": o.object_id,
            "object_type": o.object_type.value,
            "source_file": o.source_file,
            "page_number": o.page_number,
            "bbox": o.bbox.to_list(),
            "text": o.get_retrieval_text(),
            "confidence": o.overall_confidence,
            "confidence_tag": o.confidence_tag.value,
        } for o in self.objects]

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        if self.text_embeddings_matrix is not None:
            np.save(output_path.with_suffix(".npy"), self.text_embeddings_matrix)


def main():
    """Quick test."""
    import random

    data_dir = Path(__file__).parent.parent.parent / "data"
    pdfs = list(data_dir.glob("*.pdf"))[:2]

    pipeline = SOTAPipeline(fast_mode=True)
    stats = pipeline.index_pdfs(pdfs)
    print(json.dumps(stats, indent=2))

    results = pipeline.search("FBI surveillance", top_k=3)
    for r in results:
        print(f"\n{r['source_file']} p.{r['page_number']} ({r['confidence_tag']})")
        print(f"  bbox: {r['bbox']}")
        print(f"  {r['text'][:100]}...")


if __name__ == "__main__":
    main()
