# Product Requirements Document: Document Indexing & Retrieval Pipeline

## Overview

A production-grade pipeline for indexing and retrieving information from scanned government documents (FBI vault files). The system processes PDFs, extracts content with accurate bounding boxes, classifies document regions, and enables semantic search with confidence scoring.

---

## Problem Statement

FBI vault documents and similar archival materials present unique challenges:
- Mixed content types (paragraphs, tables, forms, handwritten notes)
- Variable scan quality and OCR accuracy
- No existing structure or metadata
- Need for precise source attribution (page, location)

Current approaches (simple text extraction + embeddings) lose structural information and provide no confidence signals about extraction quality.

---

## Goals

1. **Accurate Extraction**: Extract text with precise bounding boxes at the object level
2. **Content Classification**: Identify paragraphs, tables, forms, headers, footers
3. **Confidence Scoring**: Provide reliability signals for each extracted object
4. **Semantic Retrieval**: Enable natural language search with ranked results
5. **Auditability**: Every result traceable to exact page and location

### Success Metrics

| Metric | Baseline | Target |
|--------|----------|--------|
| MRR (Mean Reciprocal Rank) | 0.39 | 0.50+ |
| Precision@3 | 0.37 | 0.45+ |
| High-confidence extractions (A/B) | N/A | >90% |
| Indexing throughput | 100 pages/min | 50 pages/min (acceptable for quality) |

---

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INDEXING PIPELINE                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────┐    ┌──────────────┐    ┌─────────────┐    ┌─────────┐ │
│  │   PDF   │───▶│ Text Span    │───▶│  Spatial    │───▶│ Layout  │ │
│  │  Input  │    │ Extraction   │    │ Clustering  │    │ Classify│ │
│  └─────────┘    └──────────────┘    └─────────────┘    └─────────┘ │
│                        │                   │                 │      │
│                        ▼                   ▼                 ▼      │
│                 ┌─────────────────────────────────────────────────┐ │
│                 │              Document Objects                   │ │
│                 │  - object_id, type, bbox, text, confidence     │ │
│                 └─────────────────────────────────────────────────┘ │
│                                      │                              │
│                        ┌─────────────┴─────────────┐                │
│                        ▼                           ▼                │
│                 ┌─────────────┐            ┌─────────────┐          │
│                 │    VLM      │            │  Embedding  │          │
│                 │ Correction  │            │ Generation  │          │
│                 │ (optional)  │            │             │          │
│                 └─────────────┘            └─────────────┘          │
│                        │                           │                │
│                        └─────────────┬─────────────┘                │
│                                      ▼                              │
│                         ┌───────────────────────┐                   │
│                         │    Vector Index       │                   │
│                         │  + Metadata Store     │                   │
│                         └───────────────────────┘                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                       RETRIEVAL PIPELINE                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────┐    ┌──────────────┐    ┌─────────────┐    ┌─────────┐ │
│  │  Query  │───▶│   Embed      │───▶│   Vector    │───▶│ Rerank  │ │
│  │         │    │   Query      │    │   Search    │    │ +Filter │ │
│  └─────────┘    └──────────────┘    └─────────────┘    └─────────┘ │
│                                                               │     │
│                                                               ▼     │
│                                          ┌───────────────────────┐  │
│                                          │   Results with:       │  │
│                                          │   - source_file       │  │
│                                          │   - page_number       │  │
│                                          │   - bbox [x0,y0,x1,y1]│  │
│                                          │   - text              │  │
│                                          │   - confidence_tag    │  │
│                                          │   - object_type       │  │
│                                          └───────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Indexing Pipeline

### Stage 1: Text Span Extraction

Extract text with character-level precision using PyMuPDF.

**Input**: PDF file path
**Output**: List of `TextSpan` objects

```python
@dataclass
class TextSpan:
    text: str
    bbox: BoundingBox  # [x0, y0, x1, y1] in PDF coordinates
    font_size: float
    font_name: str
    flags: int  # bold, italic, etc.
```

**Implementation**:
```python
def extract_text_spans(page: fitz.Page) -> List[TextSpan]:
    blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
    spans = []
    for block in blocks:
        if block["type"] != 0:  # Skip non-text
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                spans.append(TextSpan(
                    text=span["text"],
                    bbox=BoundingBox.from_tuple(span["bbox"]),
                    font_size=span["size"],
                    font_name=span["font"],
                    flags=span["flags"],
                ))
    return spans
```

### Stage 2: Spatial Clustering

Group spans into logical blocks based on proximity.

**Input**: List of `TextSpan`
**Output**: List of `List[TextSpan]` (blocks)

**Algorithm**:
1. Sort spans by vertical position (y0), then horizontal (x0)
2. Initialize first span as current block
3. For each subsequent span:
   - Calculate vertical gap from current block
   - Calculate horizontal overlap
   - If vertically close AND horizontally overlapping → add to block
   - Otherwise → start new block
4. Return all blocks

**Parameters**:
- `vertical_threshold`: 15pt (adjustable based on document type)
- `horizontal_threshold`: 50pt

### Stage 3: Layout Classification

Classify each block by content type.

**Input**: Block of spans, page dimensions
**Output**: `ObjectType` enum

```python
class ObjectType(Enum):
    PARAGRAPH = "paragraph"
    TABLE = "table"
    FORM = "form"
    LIST = "list"
    HEADER = "header"
    FOOTER = "footer"
    FIGURE = "figure"
    HANDWRITTEN = "handwritten"
```

**Classification Rules**:

| Signal | Classification |
|--------|----------------|
| Position y < 10% page height | HEADER |
| Position y > 90% page height | FOOTER |
| Contains `\|` or `\t` | TABLE |
| Multiple x-positions (>3 columns) | TABLE |
| Lines starting with `1.`, `•`, `-` | LIST |
| High colon density (`:`) | FORM |
| Default | PARAGRAPH |

### Stage 4: Confidence Scoring

Compute extraction confidence for each object.

**Factors**:
- Base confidence: 0.85 (embedded text)
- Penalties:
  - Contains `?` or `�`: -0.15
  - Multiple consecutive spaces: -0.10
  - Many font variations (>5): -0.05

**Confidence Tags**:
| Tag | Range | Meaning |
|-----|-------|---------|
| A | ≥0.85 | High confidence, use directly |
| B | 0.70-0.84 | Medium confidence, reliable |
| C | 0.50-0.69 | Low confidence, may need verification |
| D | <0.50 | Very low, consider VLM correction |

### Stage 5: VLM Correction (Optional)

For low-confidence objects, use vision-language model to verify/correct.

**Trigger**: `confidence < 0.6` OR `object_type in [TABLE, CHART]`

**Input**: Cropped image of region + OCR text
**Output**: Corrected text + VLM confidence

**Prompt Template**:
```
Analyze this document region. OCR extracted:
"{ocr_text}"

Verify the text and correct any errors. Preserve numbers and formatting exactly.

Return JSON: {"corrected_text": "...", "confidence": 0.0-1.0, "notes": [...]}
```

### Stage 6: Embedding Generation

Generate vector embeddings for retrieval.

**Model**: `text-embedding-3-small` (1536 dimensions)
**Batch Size**: 50 texts per API call
**Truncation**: 30,000 characters max

**Output**: `DocumentObject` with embedding attached

```python
@dataclass
class DocumentObject:
    object_id: str
    object_type: ObjectType
    source_file: str
    page_number: int
    bbox: BoundingBox
    text_raw: str
    text_corrected: str
    overall_confidence: float
    confidence_tag: ConfidenceTag
    text_embedding: List[float]
```

---

## Retrieval Pipeline

### Query Processing

1. **Embed Query**: Same model as indexing (`text-embedding-3-small`)
2. **Vector Search**: Cosine similarity against all object embeddings
3. **Filter** (optional): By confidence tag, object type, source file
4. **Rank**: Return top-k by similarity score

### Response Format

```json
{
  "results": [
    {
      "object_id": "a1b2c3d4e5f6",
      "source_file": "abbie-hoffman-part-01.pdf",
      "page_number": 12,
      "bbox": [72.0, 144.5, 540.0, 289.3],
      "object_type": "paragraph",
      "text": "The subject was observed at the demonstration...",
      "score": 0.847,
      "confidence": 0.82,
      "confidence_tag": "B"
    }
  ],
  "query": "FBI surveillance activities",
  "total_results": 5254,
  "returned": 5
}
```

---

## Data Model

### Document Object Schema

```json
{
  "object_id": "string (12-char hash)",
  "object_type": "paragraph|table|form|list|header|footer|figure",
  "source_file": "string",
  "page_number": "integer",
  "bbox": {
    "x0": "float",
    "y0": "float",
    "x1": "float",
    "y1": "float"
  },
  "text_raw": "string (original extraction)",
  "text_corrected": "string (VLM-corrected if applicable)",
  "ocr_confidence": "float 0-1",
  "vlm_confidence": "float 0-1 (if VLM used)",
  "overall_confidence": "float 0-1",
  "confidence_tag": "A|B|C|D",
  "limitations": ["array of noted issues"],
  "structured_data": "object (for tables/charts)",
  "text_embedding": "array[1536] (stored separately)"
}
```

### Index Storage

| Component | Format | Location |
|-----------|--------|----------|
| Object metadata | JSON | `index.json` |
| Text embeddings | NumPy | `index.npy` |
| Image embeddings (optional) | NumPy | `index.img_emb.npy` |

---

## API Specification

### Index PDFs

```python
def index_pdfs(
    pdf_paths: List[Path],
    use_vlm_correction: bool = True,
    min_confidence_for_vlm: float = 0.6,
    fast_mode: bool = False,
) -> IndexStats
```

**Returns**:
```python
{
    "total_pdfs": 10,
    "total_pages": 985,
    "total_objects": 5254,
    "objects_by_type": {
        "paragraph": 1845,
        "table": 2686,
        "header": 213,
        "footer": 445,
        "form": 65
    },
    "confidence_distribution": {
        "A": 2231,
        "B": 3014,
        "C": 9,
        "D": 0
    }
}
```

### Search

```python
def search(
    query: str,
    top_k: int = 5,
    min_confidence: Optional[ConfidenceTag] = None,
    object_types: Optional[List[ObjectType]] = None,
    source_files: Optional[List[str]] = None,
) -> List[SearchResult]
```

---

## Performance Characteristics

### Indexing

| Metric | Value |
|--------|-------|
| Throughput (fast mode) | ~15 pages/second |
| Throughput (with VLM) | ~1 page/second |
| Memory usage | ~500MB per 1000 pages |
| Embedding API calls | 1 per 50 objects |

### Retrieval

| Metric | Value |
|--------|-------|
| Query latency | 150-200ms |
| Embedding call | 100-150ms |
| Vector search | 10-50ms |

---

## Configuration

```python
class PipelineConfig:
    # Extraction
    text_embedding_model: str = "text-embedding-3-small"
    vlm_model: str = "gpt-4o-mini"
    dpi: int = 150  # For image extraction

    # Clustering
    vertical_threshold: float = 15.0
    horizontal_threshold: float = 50.0
    min_text_length: int = 20

    # Confidence
    base_confidence: float = 0.85
    min_confidence_for_vlm: float = 0.6

    # VLM
    use_vlm_correction: bool = True
    vlm_max_tokens: int = 500

    # Retrieval
    default_top_k: int = 5
```

---

## Future Enhancements

### Phase 2
- [ ] OCR fallback for scanned-only pages (Tesseract/PaddleOCR)
- [ ] Image embedding for figures (CLIP)
- [ ] Table structure extraction (row/column detection)
- [ ] Handwriting detection and separate handling

### Phase 3
- [ ] Cross-document entity linking
- [ ] Temporal relationship extraction
- [ ] Named entity recognition (people, organizations, dates)
- [ ] Relationship graph construction

### Phase 4
- [ ] Real-time incremental indexing
- [ ] Distributed processing for large archives
- [ ] Query understanding (semantic parsing)
- [ ] Multi-modal retrieval (image + text)

---

## Dependencies

```
# Core
openai>=1.0.0
pymupdf>=1.23.0
numpy>=1.24.0

# Optional (Phase 2+)
pytesseract>=0.3.10
paddleocr>=2.7.0
sentence-transformers>=2.2.0
chromadb>=0.4.0
```

---

## Appendix: Benchmark Results

Tested on 10 sampled FBI vault PDFs (985 pages, 5254 objects).

| Metric | GPT Baseline | SOTA Pipeline | Improvement |
|--------|-------------|---------------|-------------|
| MRR | 0.392 | 0.420 | +7.2% |
| Precision@3 | 0.367 | 0.333 | -9.1% |
| Easy queries | 0.444 | 0.667 | **+22.2%** |
| Hard queries | 0.333 | 0.400 | +6.7% |
| Objects indexed | 2,020 | 5,254 | +160% |
| High-conf (A/B) | N/A | 99.8% | - |
