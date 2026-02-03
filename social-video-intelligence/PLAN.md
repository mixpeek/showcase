# Social Video Intelligence Showcase - E2E Plan

## Overview

This showcase demonstrates **multimodal video intelligence for social/YouTube content** - running parallel extractors across video content, indexing each signal layer separately while enabling unified cross-modal queries.

### Use Case
Brand monitoring and content intelligence across social video:
- "Find videos where [brand logo] appears visually + negative audio sentiment + trending in [region]"
- Not keyword matching - semantic understanding across visual, audio, and text modalities

### Key Capabilities Demonstrated
1. **Parallel extraction layers** (scene detection, visual embeddings, audio transcription, OCR, sentiment)
2. **Multi-layer indexing** (each modality indexed separately)
3. **Cross-modal retrieval** (query across all layers simultaneously)
4. **Clustering** (surface emerging narratives automatically)
5. **Taxonomies** (map to brand safety, content categories, sentiment)
6. **Alerts** (real-time monitoring for brand mentions or sentiment shifts)

---

## Architecture

### Data Flow
```
YouTube/Social URLs
       │
       ▼
┌──────────────────────────────────────────────────────────────────┐
│                    BUCKET: social_videos                          │
│  Schema: video_url, title, platform, region, channel, upload_date│
└──────────────────────────────────────────────────────────────────┘
       │
       │ Parallel Processing (same source, multiple collections)
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         COLLECTIONS (Parallel Extractors)                    │
│                                                                             │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐  │
│  │ visual_scenes       │  │ audio_content       │  │ text_content        │  │
│  │                     │  │                     │  │                     │  │
│  │ multimodal_extractor│  │ multimodal_extractor│  │ text_extractor      │  │
│  │ - scene detection   │  │ - transcription     │  │ - OCR text          │  │
│  │ - visual embeddings │  │ - audio embeddings  │  │ - text embeddings   │  │
│  │ - thumbnails        │  │ - sentiment (LLM)   │  │ - brand mentions    │  │
│  │                     │  │                     │  │                     │  │
│  │ Vector: 1408D       │  │ Vector: 1024D       │  │ Vector: 1024D       │  │
│  └─────────────────────┘  └─────────────────────┘  └─────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              TAXONOMIES                                      │
│                                                                             │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐  │
│  │ brand_taxonomy      │  │ sentiment_taxonomy  │  │ content_taxonomy    │  │
│  │                     │  │                     │  │                     │  │
│  │ - Brand A           │  │ - Positive          │  │ - Entertainment     │  │
│  │ - Brand B           │  │ - Neutral           │  │ - Education         │  │
│  │ - Brand C           │  │ - Negative          │  │ - News              │  │
│  │ - Competitor X      │  │ - Mixed             │  │ - Sports            │  │
│  └─────────────────────┘  └─────────────────────┘  └─────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              RETRIEVERS                                      │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ unified_video_search                                                   │  │
│  │                                                                        │  │
│  │ Stage 1: semantic_search (visual_scenes - find brand logos)           │  │
│  │ Stage 2: semantic_search (audio_content - find sentiment context)     │  │
│  │ Stage 3: semantic_search (text_content - find brand mentions in OCR)  │  │
│  │ Stage 4: fusion (combine results with custom weights)                 │  │
│  │ Stage 5: filter (apply metadata filters: region, date, platform)      │  │
│  │ Stage 6: rerank (LLM reranking for final relevance)                   │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ brand_monitoring                                                       │  │
│  │                                                                        │  │
│  │ Stage 1: semantic_search (find brand visually)                        │  │
│  │ Stage 2: attribute_filter (sentiment = negative)                      │  │
│  │ Stage 3: taxonomy_enrich (add brand safety labels)                    │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLUSTERING                                      │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ narrative_clusters                                                     │  │
│  │                                                                        │  │
│  │ - Algorithm: k-means                                                   │  │
│  │ - Features: multimodal_embedding (1408D)                              │  │
│  │ - Auto-labels: LLM-generated cluster descriptions                     │  │
│  │ - Use: Surface emerging narratives across content                     │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ALERTS                                          │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ negative_brand_mention                                                 │  │
│  │                                                                        │  │
│  │ - Trigger: New video with brand + negative sentiment                  │  │
│  │ - Retriever: brand_monitoring                                         │  │
│  │ - Notification: webhook → Slack                                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Steps

### Phase 1: Setup Infrastructure

#### 1.1 Create Namespace
```python
namespace = client.namespaces.create(
    namespace_name="social_video_intelligence",
    description="Multi-modal video intelligence for brand monitoring"
)
```

#### 1.2 Create Bucket with Video Schema
```python
bucket = client.buckets.create(
    bucket_name="social_videos",
    description="Social media video content",
    bucket_schema={
        "properties": {
            "video": {"type": "video", "required": True},
            "title": {"type": "string", "required": True},
            "description": {"type": "text"},
            "platform": {"type": "string"},  # youtube, tiktok, instagram
            "channel": {"type": "string"},
            "region": {"type": "string"},   # US, EU, APAC
            "upload_date": {"type": "string"},
            "tags": {"type": "array", "items": {"type": "string"}},
            "view_count": {"type": "integer"},
            "engagement_score": {"type": "number"}
        }
    }
)
```

### Phase 2: Create Extraction Collections

#### 2.1 Visual Scenes Collection (Scene Detection + Visual Embeddings)
```python
visual_collection = client.collections.create(
    collection_name="visual_scenes",
    description="Visual scene detection and embeddings",
    source={"type": "bucket", "bucket_ids": [bucket_id]},
    feature_extractor={
        "feature_extractor_name": "multimodal_extractor",
        "version": "v1",
        "input_mappings": {"video": "video"},
        "parameters": {
            "extractor_type": "multimodal_extractor",
            "split_method": "scene",  # Scene detection
            "scene_detection_threshold": 0.5,
            "run_multimodal_embedding": True,  # 1408D visual embeddings
            "run_video_description": True,     # AI descriptions for search
            "run_ocr": True,                   # Text in frames (logos, etc.)
            "enable_thumbnails": True,
            "description_prompt": "Describe the visual content, noting any visible brands, logos, products, or text. Include the emotional tone of the scene."
        },
        "field_passthrough": [
            {"source_path": "metadata.title"},
            {"source_path": "metadata.platform"},
            {"source_path": "metadata.region"},
            {"source_path": "metadata.channel"},
            {"source_path": "metadata.upload_date"},
            {"source_path": "metadata.view_count"}
        ]
    }
)
```

#### 2.2 Audio Content Collection (Transcription + Sentiment)
```python
audio_collection = client.collections.create(
    collection_name="audio_content",
    description="Audio transcription with sentiment analysis",
    source={"type": "bucket", "bucket_ids": [bucket_id]},
    feature_extractor={
        "feature_extractor_name": "multimodal_extractor",
        "version": "v1",
        "input_mappings": {"video": "video"},
        "parameters": {
            "extractor_type": "multimodal_extractor",
            "split_method": "silence",        # Split on audio pauses
            "silence_db_threshold": -40,
            "run_transcription": True,        # Speech-to-text
            "run_transcription_embedding": True,  # 1024D text embeddings
            "run_multimodal_embedding": False,    # Skip visual
            "run_video_description": True,
            "description_prompt": "Analyze the spoken content. Identify: 1) Main topics discussed, 2) Sentiment (positive/neutral/negative), 3) Any brand or product mentions, 4) Emotional tone of the speaker."
        },
        "field_passthrough": [
            {"source_path": "metadata.title"},
            {"source_path": "metadata.platform"},
            {"source_path": "metadata.region"},
            {"source_path": "metadata.channel"}
        ]
    }
)
```

#### 2.3 Text Content Collection (OCR + Captions)
```python
text_collection = client.collections.create(
    collection_name="text_content",
    description="Text extracted from video (OCR, captions)",
    source={"type": "bucket", "bucket_ids": [bucket_id]},
    feature_extractor={
        "feature_extractor_name": "multimodal_extractor",
        "version": "v1",
        "input_mappings": {"video": "video"},
        "parameters": {
            "extractor_type": "multimodal_extractor",
            "split_method": "time",
            "time_split_interval": 5,         # Check every 5 seconds
            "run_ocr": True,                  # Extract text from frames
            "run_multimodal_embedding": False,
            "run_transcription": False,
            "run_video_description": False
        },
        "field_passthrough": [
            {"source_path": "metadata.title"},
            {"source_path": "metadata.platform"},
            {"source_path": "metadata.region"}
        ]
    }
)
```

### Phase 3: Create Taxonomies

#### 3.1 Brand Taxonomy
```python
# First create a bucket with brand reference data
brand_bucket = client.buckets.create(
    bucket_name="brand_reference",
    bucket_schema={
        "properties": {
            "brand_text": {"type": "text", "required": True},
            "brand_id": {"type": "string"},
            "brand_name": {"type": "string"},
            "brand_category": {"type": "string"},  # owned, competitor, partner
            "brand_logo_url": {"type": "string"}
        }
    }
)

# Create collection for brand embeddings
brand_collection = client.collections.create(
    collection_name="brand_reference",
    source={"type": "bucket", "bucket_ids": [brand_bucket_id]},
    feature_extractor={
        "feature_extractor_name": "text_extractor",
        "version": "v1",
        "input_mappings": {"text": "brand_text"}
    }
)

# Create brand search retriever
brand_retriever = client.retrievers.create(
    retriever_name="brand_search",
    stages=[{
        "stage": "feature_search",
        "stage_name": "brand_match",
        "feature_extractor_name": "text_extractor",
        "collection_ids": [brand_collection_id],
        "query_input_key": "query",
        "top_k": 5,
        "min_score": 0.7,
        "field_passthrough": ["brand_id", "brand_name", "brand_category"]
    }],
    inputs={"query": {"type": "text"}}
)

# Create brand taxonomy
brand_taxonomy = client.taxonomies.create(
    taxonomy_name="brand_taxonomy",
    description="Identify brands in content",
    config={
        "taxonomy_type": "flat",
        "retriever_id": brand_retriever_id,
        "input_mappings": [{"input_key": "query", "source_type": "text", "path": "description"}],
        "enrichment_fields": [
            {"field_path": "brand_id", "merge_mode": "replace"},
            {"field_path": "brand_name", "merge_mode": "replace"},
            {"field_path": "brand_category", "merge_mode": "replace"}
        ],
        "collection_configuration": [
            {"collection_id": visual_collection_id, "apply_on_ingestion": True}
        ]
    }
)
```

#### 3.2 Sentiment Taxonomy
```python
# Create sentiment reference collection
sentiments = [
    ("positive", "Happy, excited, enthusiastic, satisfied, pleased, joyful, optimistic content"),
    ("neutral", "Informational, factual, objective, balanced, descriptive content"),
    ("negative", "Angry, frustrated, disappointed, critical, upset, pessimistic content"),
    ("mixed", "Content with both positive and negative elements")
]

# ... similar pattern to brand taxonomy

sentiment_taxonomy = client.taxonomies.create(
    taxonomy_name="sentiment_taxonomy",
    description="Classify content sentiment",
    config={
        "taxonomy_type": "flat",
        "retriever_id": sentiment_retriever_id,
        "input_mappings": [{"input_key": "query", "source_type": "text", "path": "description"}],
        "enrichment_fields": [
            {"field_path": "sentiment_label", "merge_mode": "replace"},
            {"field_path": "sentiment_score", "merge_mode": "replace"}
        ],
        "collection_configuration": [
            {"collection_id": audio_collection_id, "apply_on_ingestion": True}
        ]
    }
)
```

#### 3.3 Content Category Taxonomy (IAB-style)
```python
# Similar to adtech showcase - create IAB content categories
content_taxonomy = client.taxonomies.create(
    taxonomy_name="content_taxonomy",
    description="IAB content categories for reporting",
    config={
        "taxonomy_type": "hierarchical",
        "retriever_id": iab_retriever_id,
        # ... IAB hierarchy configuration
    }
)
```

### Phase 4: Create Retrievers

#### 4.1 Unified Cross-Modal Video Search
```python
unified_retriever = client.retrievers.create(
    retriever_name="unified_video_search",
    description="Search across all video modalities",
    stages=[
        # Stage 1: Visual search
        {
            "stage": "feature_search",
            "stage_name": "visual_search",
            "feature_extractor_name": "multimodal_extractor",
            "collection_ids": [visual_collection_id],
            "query_input_key": "query",
            "top_k": 50,
            "min_score": 0.3,
            "field_passthrough": ["title", "platform", "region", "thumbnail_url", "start_time", "end_time", "description"]
        },
        # Stage 2: Audio/transcript search
        {
            "stage": "feature_search",
            "stage_name": "audio_search",
            "feature_extractor_name": "multimodal_extractor",
            "feature_name": "transcription_embedding",
            "collection_ids": [audio_collection_id],
            "query_input_key": "query",
            "top_k": 50,
            "min_score": 0.3,
            "field_passthrough": ["title", "transcription", "sentiment_label", "start_time", "end_time"]
        },
        # Stage 3: Fusion - combine visual and audio results
        {
            "stage": "fusion",
            "stage_name": "cross_modal_fusion",
            "strategy": "rrf",  # Reciprocal Rank Fusion
            "weights": {"visual_search": 0.6, "audio_search": 0.4},
            "top_k": 30
        },
        # Stage 4: Apply filters
        {
            "stage": "attribute_filter",
            "stage_name": "apply_filters",
            "filters": [
                {"field": "platform", "op": "eq", "value_input_key": "platform"},
                {"field": "region", "op": "eq", "value_input_key": "region"},
                {"field": "sentiment_label", "op": "eq", "value_input_key": "sentiment"}
            ]
        },
        # Stage 5: LLM rerank for final relevance
        {
            "stage": "llm_rerank",
            "stage_name": "final_rerank",
            "model": "gpt-4o-mini",
            "prompt": "Rank these video segments by relevance to the query. Consider visual content, spoken content, and sentiment.",
            "top_k": 10
        }
    ],
    inputs={
        "query": {"type": "text", "description": "Search query (text or brand name)"},
        "platform": {"type": "string", "description": "Filter by platform", "required": False},
        "region": {"type": "string", "description": "Filter by region", "required": False},
        "sentiment": {"type": "string", "description": "Filter by sentiment", "required": False}
    }
)
```

#### 4.2 Brand Monitoring Retriever
```python
brand_monitoring_retriever = client.retrievers.create(
    retriever_name="brand_monitoring",
    description="Find brand mentions with sentiment context",
    stages=[
        # Stage 1: Find brand visually (logos, products)
        {
            "stage": "feature_search",
            "stage_name": "visual_brand_search",
            "feature_extractor_name": "multimodal_extractor",
            "collection_ids": [visual_collection_id],
            "query_input_key": "brand_query",
            "top_k": 100,
            "min_score": 0.4
        },
        # Stage 2: Enrich with brand taxonomy
        {
            "stage": "taxonomy_enrich",
            "stage_name": "brand_classify",
            "taxonomy_id": brand_taxonomy_id,
            "min_score": 0.6
        },
        # Stage 3: Filter by sentiment
        {
            "stage": "attribute_filter",
            "stage_name": "sentiment_filter",
            "filters": [
                {"field": "sentiment_label", "op": "eq", "value_input_key": "target_sentiment"}
            ]
        },
        # Stage 4: Group by source video
        {
            "stage": "transform",
            "stage_name": "group_results",
            "group_by": "source_video_url",
            "aggregate": {
                "segments": "collect",
                "sentiment_labels": "collect_distinct",
                "brand_mentions": "count"
            }
        }
    ],
    inputs={
        "brand_query": {"type": "text", "description": "Brand name or description to search for"},
        "target_sentiment": {"type": "string", "description": "Filter by sentiment (positive/neutral/negative)"}
    }
)
```

### Phase 5: Create Clustering

#### 5.1 Narrative Discovery Cluster
```python
narrative_cluster = client.clusters.create(
    cluster_name="narrative_discovery",
    description="Discover emerging narratives across video content",
    config={
        "algorithm": "kmeans",
        "num_clusters": 20,  # Adjust based on content volume
        "feature_uri": "multimodal_extractor_v1_multimodal_embedding",
        "collections": [visual_collection_id],
        "enrichment": {
            "centroid_summary": True,  # Auto-generate cluster labels
            "summary_prompt": "Summarize the main theme or narrative of this cluster of video segments. What story or topic do they share?"
        },
        "schedule": {
            "type": "daily",
            "time": "02:00"  # Run clustering daily at 2 AM
        }
    }
)
```

### Phase 6: Create Alerts

#### 6.1 Negative Brand Mention Alert
```python
negative_brand_alert = client.alerts.create(
    name="negative_brand_mention",
    description="Alert when brand appears with negative sentiment",
    retriever_id=brand_monitoring_retriever_id,
    trigger_config={
        "inputs": {
            "brand_query": "Your Brand Name",
            "target_sentiment": "negative"
        },
        "min_results": 1  # Trigger if any results found
    },
    notification_config={
        "channels": [
            {
                "channel_type": "webhook",
                "channel_id": "wh_slack_alerts",
                "payload_template": {
                    "text": "Negative brand mention detected",
                    "blocks": [
                        {"type": "section", "text": "{{results[0].title}} - {{results[0].platform}}"},
                        {"type": "section", "text": "Sentiment: {{results[0].sentiment_label}}"}
                    ]
                }
            }
        ],
        "include_matches": True,
        "include_scores": True
    },
    enabled=True
)
```

### Phase 7: Demo Workflow

```python
def demo_social_video_intelligence():
    """Complete demo workflow."""

    # 1. Ingest sample videos
    sample_videos = [
        {"url": "https://example.com/product_review_positive.mp4", "platform": "youtube", "region": "US"},
        {"url": "https://example.com/brand_unboxing.mp4", "platform": "tiktok", "region": "EU"},
        {"url": "https://example.com/competitor_comparison.mp4", "platform": "youtube", "region": "US"},
        # ... more samples
    ]

    # Upload to bucket
    for video in sample_videos:
        client.bucket_objects.upload(bucket_id, video)

    # 2. Process through all collections
    batch = client.bucket_batches.create(bucket_id, object_ids)
    client.bucket_batches.submit(bucket_id, batch.batch_id)

    # Wait for processing...

    # 3. Run complex queries

    # Query: "Find videos where our brand appears with negative sentiment in the US"
    results = client.retrievers.execute(
        retriever_id=brand_monitoring_retriever_id,
        inputs={
            "brand_query": "Brand Logo visual appearance",
            "target_sentiment": "negative"
        }
    )

    # Query: "Cross-modal search for product launch content"
    results = client.retrievers.execute(
        retriever_id=unified_retriever_id,
        inputs={
            "query": "new product launch announcement exciting",
            "platform": "youtube",
            "sentiment": "positive"
        }
    )

    # 4. Get cluster insights
    cluster_results = client.clusters.execute(cluster_id)
    print("Emerging narratives:")
    for cluster in cluster_results.clusters:
        print(f"  - {cluster.label}: {cluster.document_count} videos")

    # 5. Check alerts
    alert_executions = client.alerts.list_executions(alert_id)
    print(f"Recent alerts: {len(alert_executions)} triggers")
```

---

## File Structure

```
showcase/social-video-intelligence/
├── PLAN.md                      # This file
├── README.md                    # User-facing documentation
├── requirements.txt             # Dependencies (mixpeek SDK)
├── .env.example                 # Environment template
│
├── setup/
│   ├── __init__.py
│   ├── config.py               # Configuration management
│   ├── create_namespace.py      # Create namespace
│   ├── create_bucket.py         # Create bucket with schema
│   ├── create_collections.py    # Create all extraction collections
│   ├── create_taxonomies.py     # Create brand/sentiment/content taxonomies
│   ├── create_retrievers.py     # Create unified + brand monitoring retrievers
│   ├── create_clusters.py       # Create narrative discovery cluster
│   ├── create_alerts.py         # Create brand monitoring alerts
│   └── setup_all.py             # Run complete setup
│
├── data/
│   ├── sample_videos.json       # Sample video URLs for demo
│   ├── brands.json              # Brand reference data
│   └── sentiments.json          # Sentiment labels
│
├── demo/
│   ├── __init__.py
│   ├── ingest_videos.py         # Upload and process sample videos
│   ├── cross_modal_search.py    # Demo unified search
│   ├── brand_monitoring.py      # Demo brand + sentiment filtering
│   ├── narrative_discovery.py   # Demo clustering insights
│   ├── export_report.py         # Export results to CSV/JSON
│   └── run_demo.py              # Interactive demo runner
│
├── cli.py                       # CLI entry point
└── tests/
    ├── test_setup.py
    └── test_queries.py
```

---

## API Key Handling

All code uses environment variables for API key management:

```python
import os
API_KEY = os.environ.get("MIXPEEK_API_KEY")
if not API_KEY:
    raise ValueError("Please set MIXPEEK_API_KEY environment variable")
```

Get your API key at: https://mixpeek.com/dashboard

---

## Success Criteria

1. **Setup completes successfully** - All resources created without errors
2. **Parallel extraction works** - Same video processed by 3 collections with different extractors
3. **Cross-modal search returns relevant results** - Query finds videos matching visual AND audio criteria
4. **Taxonomies enrich documents** - Brand/sentiment labels added to indexed documents
5. **Clustering surfaces insights** - Meaningful clusters with auto-generated labels
6. **Alerts trigger correctly** - Webhook fired when matching content ingested
7. **Latency acceptable** - Retriever execution < 2 seconds

---

## Next Steps

1. Create directory structure
2. Implement setup scripts
3. Create sample data
4. Build demo workflows
5. Add CLI interface
6. Write documentation
7. Test end-to-end
