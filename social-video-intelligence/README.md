# Social Video Intelligence

**Turn hours of video into actionable brand intelligence in minutes.**

This showcase demonstrates how to build a complete video intelligence pipeline using [Mixpeek](https://mixpeek.com) - monitoring brand mentions, tracking sentiment, and discovering emerging narratives across social media and YouTube content.

## The Problem

Marketing and brand teams need to monitor video content across platforms, but:
- Manual video review doesn't scale (hours of content uploaded every minute)
- Keyword search misses visual brand appearances (logos, products in frame)
- Sentiment context matters (is that brand mention positive or negative?)
- Trends emerge across videos, not within a single one

## The Solution

This showcase builds a **multi-modal video intelligence system** that:

1. **Extracts signals in parallel** - Visual scenes, audio transcription, and on-screen text
2. **Classifies automatically** - Brand detection and sentiment analysis via semantic matching
3. **Enables complex queries** - "Videos where Nike appears visually with negative sentiment in the US"
4. **Surfaces trends** - Cluster similar content to discover emerging narratives

```
"Find videos where [brand] appears visually + negative sentiment + trending in [region]"
```

This is **semantic understanding**, not keyword matching. The system understands that a Nike swoosh in frame relates to "athletic footwear" even if no one says "Nike."

## How Mixpeek Powers This

Mixpeek handles the heavy lifting:

| Mixpeek Component | What It Does Here |
|-------------------|-------------------|
| **Buckets** | Store video metadata and trigger processing |
| **Collections** | Run 3 parallel extractors (visual, audio, text) on each video |
| **Feature Extractors** | Generate embeddings: 1408D visual, 1024D text |
| **Taxonomies** | Auto-classify: brand detection + sentiment analysis |
| **Retrievers** | Cross-modal search across all signal layers |
| **Clusters** | Group similar content to find emerging themes |
| **Alerts** | Real-time notifications for brand mentions |

### Architecture

```
                                    ┌─────────────────────────────────────┐
                                    │         MIXPEEK PLATFORM            │
┌─────────────┐                     │                                     │
│   Videos    │                     │  ┌─────────────────────────────┐   │
│  (YouTube,  │──── Upload ────────▶│  │     Video Bucket            │   │
│   TikTok,   │                     │  │  (metadata + video files)   │   │
│   etc.)     │                     │  └──────────┬──────────────────┘   │
└─────────────┘                     │             │                       │
                                    │             ▼                       │
                                    │  ┌─────────────────────────────┐   │
                                    │  │   Parallel Collections       │   │
                                    │  │                              │   │
                                    │  │  ┌────────┐ ┌────────┐ ┌────┐│   │
                                    │  │  │Visual  │ │ Audio  │ │Text││   │
                                    │  │  │Scenes  │ │Content │ │OCR ││   │
                                    │  │  └───┬────┘ └───┬────┘ └─┬──┘│   │
                                    │  └──────┼──────────┼────────┼───┘   │
                                    │         │          │        │       │
                                    │         ▼          ▼        ▼       │
                                    │  ┌─────────────────────────────┐   │
                                    │  │      Taxonomies             │   │
                                    │  │  • Brand Detection          │   │
                                    │  │  • Sentiment Classification │   │
                                    │  └──────────┬──────────────────┘   │
                                    │             │                       │
                                    │             ▼                       │
                                    │  ┌─────────────────────────────┐   │
                                    │  │      Retrievers             │   │
                                    │  │  • Cross-modal Search       │   │
                                    │  │  • Brand Monitoring         │   │
                                    │  └──────────┬──────────────────┘   │
                                    │             │                       │
                                    └─────────────┼───────────────────────┘
                                                  │
                                                  ▼
                                    ┌─────────────────────────────┐
                                    │     Query Results           │
                                    │  "Nike + negative + US"     │
                                    │  → 23 video segments        │
                                    └─────────────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.9+
- A Mixpeek API key ([get one free](https://mixpeek.com/dashboard))

### 1. Install Dependencies

```bash
cd social-video-intelligence
pip install -r requirements.txt
```

### 2. Set Your API Key

```bash
export MIXPEEK_API_KEY=your_api_key_here
```

Or create a `.env` file:
```bash
cp .env.example .env
# Edit .env and add your key
```

### 3. Run the Demo

```bash
# One command to set up everything and run sample queries
python run_demo.py
```

This will:
- Create a namespace with 3 extraction collections
- Set up brand and sentiment taxonomies
- Ingest sample videos
- Run example queries

### 4. Try Your Own Queries

```bash
# Search across all modalities
python cli.py search "basketball highlights"
python cli.py search "product unboxing" --sentiment positive

# Monitor specific brands
python cli.py monitor "Apple iPhone"
python cli.py monitor "Tesla" --sentiment negative

# Discover trending themes
python cli.py clusters --execute
```

## Use Cases

### Brand Crisis Detection

Immediately find negative brand mentions across platforms:

```bash
python cli.py monitor "YourBrand" --sentiment negative
```

Returns video segments where your brand appears (visually or verbally) in a negative context.

### Competitive Intelligence

Track how competitors appear in content:

```bash
python cli.py search "CompetitorBrand launch announcement"
```

Find product launches, reviews, and mentions - even when the brand name isn't spoken.

### Content Trend Discovery

See what themes are emerging across your video corpus:

```bash
python cli.py clusters --execute
```

Auto-clusters videos by visual and semantic similarity, surfacing emerging narratives.

### Regional Analysis

Filter insights by geography:

```bash
python cli.py search "viral trend" --region US
python cli.py search "product review" --region APAC
```

## How It Works

### Step 1: Parallel Extraction

When you ingest a video, Mixpeek processes it through three collections simultaneously:

| Collection | Split Method | Extracts |
|------------|--------------|----------|
| **visual_scenes** | Scene detection | Visual embeddings, thumbnails, OCR, descriptions |
| **audio_content** | Silence detection | Transcription, text embeddings, sentiment |
| **text_content** | Time intervals (5s) | OCR text, text embeddings |

### Step 2: Taxonomy Enrichment

During processing, taxonomies automatically classify each segment:

- **Brand Taxonomy**: Semantic matching against known brands (Nike, Apple, etc.)
- **Sentiment Taxonomy**: Classifies as positive, neutral, negative, or mixed

### Step 3: Cross-Modal Retrieval

The retriever searches across all collections simultaneously:

```python
# This query searches visual, audio, AND text signals
results = client.retrievers.execute(
    retriever_id="ret_unified_search",
    inputs={
        "query": "exciting product reveal",
        "sentiment": "positive",
        "platform": "youtube"
    }
)
```

### Step 4: Narrative Clustering

K-means clustering on visual embeddings groups similar content:

```python
clusters = client.clusters.execute(cluster_id="clust_narrative")
# Returns: [
#   {"label": "Product Unboxings", "size": 47, "documents": [...]},
#   {"label": "Sports Highlights", "size": 23, "documents": [...]},
#   ...
# ]
```

## API Reference

### Direct SDK Usage

```python
from mixpeek import Mixpeek

client = Mixpeek(api_key="your_api_key")

# Upload a video
client.bucket_objects.create(
    bucket_id="bkt_videos",
    blobs=[{"property": "video", "type": "video", "data": "https://example.com/video.mp4"}],
    metadata={"title": "Product Review", "platform": "youtube", "region": "US"},
    _headers={"X-Namespace": "ns_your_namespace"}
)

# Execute cross-modal search
results = client.retrievers.execute(
    retriever_id="ret_unified_search",
    inputs={"query": "basketball highlights", "sentiment": "positive"},
    _headers={"X-Namespace": "ns_your_namespace"}
)

# Get narrative clusters
clusters = client.clusters.execute(
    cluster_id="clust_narrative",
    _headers={"X-Namespace": "ns_your_namespace"}
)
```

### REST API

```bash
# Search videos
curl -X POST "https://api.mixpeek.com/v1/retrievers/ret_xxx/execute" \
  -H "Authorization: Bearer $MIXPEEK_API_KEY" \
  -H "X-Namespace: ns_xxx" \
  -H "Content-Type: application/json" \
  -d '{"inputs": {"query": "product review", "sentiment": "negative"}}'
```

## File Structure

```
social-video-intelligence/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── .env.example              # Environment template
├── run_demo.py               # One-command full demo
├── cli.py                    # CLI for queries and management
├── test_e2e_requests.py      # E2E test (verified against prod API)
├── setup/
│   ├── config.py             # Configuration management
│   └── setup_all.py          # Infrastructure setup
├── demo/
│   ├── ingest_videos.py      # Video ingestion helpers
│   ├── cross_modal_search.py # Search demo
│   ├── brand_monitoring.py   # Brand monitoring demo
│   └── narrative_discovery.py# Clustering demo
├── data/
│   ├── sample_videos.json    # Sample video URLs
│   ├── brands.json           # Brand reference data
│   └── sentiments.json       # Sentiment labels
└── tests/
    └── test_setup.py         # Unit tests
```

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `MIXPEEK_API_KEY` | Your Mixpeek API key | Yes |
| `MIXPEEK_API_BASE` | API base URL (default: https://api.mixpeek.com) | No |

### Extraction Parameters

Customize in `setup/setup_all.py`:

```python
# Scene detection sensitivity (0.3 = more scenes, 0.7 = fewer)
"scene_detection_threshold": 0.5

# Audio silence threshold in dB (lower = more sensitive)
"silence_db_threshold": -40

# OCR check interval in seconds
"time_split_interval": 5
```

## Troubleshooting

### "No results found"

Videos take time to process. Check status:
```bash
python cli.py status
```

### "Retriever not found"

Run setup first:
```bash
python cli.py setup
```

### "API key invalid"

Verify your key is set:
```bash
echo $MIXPEEK_API_KEY
```

## E2E Test

The showcase includes a verified end-to-end test:

```bash
python test_e2e_requests.py
```

This creates a complete pipeline and verifies all components work with the production API.

## Learn More

- [Mixpeek Documentation](https://docs.mixpeek.com)
- [Python SDK](https://pypi.org/project/mixpeek/)
- [API Reference](https://api.mixpeek.com/docs)
- [More Showcases](https://github.com/mixpeek/showcase)

## License

MIT License - see LICENSE file.
