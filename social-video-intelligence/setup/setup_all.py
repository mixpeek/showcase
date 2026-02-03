#!/usr/bin/env python3
"""
Complete setup for Social Video Intelligence showcase.

This script creates all necessary Mixpeek resources:
- Namespace
- Buckets (videos + reference data)
- Collections (visual, audio, text extraction)
- Taxonomies (brand, sentiment, content)
- Retrievers (unified search, brand monitoring)
- Clusters (narrative discovery)
- Alerts (negative brand mentions)

Usage:
    python -m setup.setup_all

Or with custom API key:
    MIXPEEK_API_KEY=your_key python -m setup.setup_all
"""

import sys
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mixpeek import Mixpeek

from setup.config import Config, get_config

console = Console()


# =============================================================================
# Reference Data
# =============================================================================

BRANDS = [
    {"id": "brand_coca_cola", "name": "Coca-Cola", "category": "competitor", "text": "Coca-Cola Coke red logo beverage soda drink refreshment"},
    {"id": "brand_pepsi", "name": "Pepsi", "category": "competitor", "text": "Pepsi blue logo beverage soda drink cola refreshment"},
    {"id": "brand_nike", "name": "Nike", "category": "tracked", "text": "Nike swoosh logo athletic sportswear shoes sneakers just do it"},
    {"id": "brand_adidas", "name": "Adidas", "category": "competitor", "text": "Adidas three stripes logo athletic sportswear shoes originals"},
    {"id": "brand_apple", "name": "Apple", "category": "tracked", "text": "Apple logo iPhone iPad Mac MacBook technology electronics"},
    {"id": "brand_samsung", "name": "Samsung", "category": "competitor", "text": "Samsung logo Galaxy smartphone tablet television electronics"},
    {"id": "brand_mcdonalds", "name": "McDonald's", "category": "tracked", "text": "McDonald's golden arches logo fast food burger fries restaurant"},
    {"id": "brand_starbucks", "name": "Starbucks", "category": "tracked", "text": "Starbucks green mermaid logo coffee cafe latte espresso"},
]

SENTIMENTS = [
    {"id": "sentiment_positive", "label": "positive", "text": "Happy excited enthusiastic satisfied pleased joyful optimistic amazing wonderful great excellent fantastic love"},
    {"id": "sentiment_neutral", "label": "neutral", "text": "Informational factual objective balanced descriptive neutral standard normal typical average"},
    {"id": "sentiment_negative", "label": "negative", "text": "Angry frustrated disappointed critical upset pessimistic terrible awful bad horrible hate disappointed"},
    {"id": "sentiment_mixed", "label": "mixed", "text": "Mixed feelings both positive and negative elements pros and cons advantages disadvantages"},
]


# =============================================================================
# Setup Functions
# =============================================================================


def create_namespace(client: Mixpeek, config: Config) -> str:
    """Create namespace for the showcase."""
    console.print("\n[bold blue]Creating namespace...[/bold blue]")

    result = client.namespaces.create(
        namespace_name=f"social_video_intelligence_{int(time.time())}",
        description="Multi-modal video intelligence for brand monitoring and content analysis",
        feature_extractors=[
            {"feature_extractor_name": "multimodal_extractor", "version": "v1"},
            {"feature_extractor_name": "text_extractor", "version": "v1"},
        ],
    )
    namespace_id = result.namespace_id
    config.namespace_id = namespace_id
    console.print(f"  [green]Created namespace:[/green] {namespace_id}")
    return namespace_id


def create_video_bucket(client: Mixpeek, config: Config) -> str:
    """Create bucket for video content."""
    console.print("\n[bold blue]Creating video bucket...[/bold blue]")

    result = client.buckets.create(
        bucket_name="social_videos",
        description="Social media video content for analysis",
        bucket_schema={
            "properties": {
                "video": {"type": "video", "required": True},
                "title": {"type": "string", "required": True},
                "description": {"type": "text"},
                "platform": {"type": "string"},
                "channel": {"type": "string"},
                "region": {"type": "string"},
                "upload_date": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "view_count": {"type": "integer"},
                "engagement_score": {"type": "number"},
            }
        },
        _headers={"X-Namespace": config.namespace_id},
    )
    bucket_id = result.bucket_id
    config.bucket_id = bucket_id
    console.print(f"  [green]Created bucket:[/green] {bucket_id}")
    return bucket_id


def create_extraction_collections(client: Mixpeek, config: Config) -> None:
    """Create collections for parallel extraction."""
    console.print("\n[bold blue]Creating extraction collections...[/bold blue]")

    # Visual Scenes Collection
    console.print("  Creating visual_scenes collection (scene detection + visual embeddings)...")
    result = client.collections.create(
        collection_name="visual_scenes",
        description="Visual scene detection and multimodal embeddings",
        source={"type": "bucket", "bucket_ids": [config.bucket_id]},
        feature_extractor={
            "feature_extractor_name": "multimodal_extractor",
            "version": "v1",
            "input_mappings": {"video": "video"},
            "parameters": {
                "extractor_type": "multimodal_extractor",
                "split_method": "scene",
                "scene_detection_threshold": 0.5,
                "run_multimodal_embedding": True,
                "run_video_description": True,
                "run_ocr": True,
                "enable_thumbnails": True,
                "description_prompt": (
                    "Describe the visual content, noting any visible brands, logos, products, or text. "
                    "Include the emotional tone of the scene and any notable visual elements."
                ),
            },
            "field_passthrough": [
                {"source_path": "metadata.title"},
                {"source_path": "metadata.platform"},
                {"source_path": "metadata.region"},
                {"source_path": "metadata.channel"},
                {"source_path": "metadata.upload_date"},
                {"source_path": "metadata.view_count"},
            ],
        },
        _headers={"X-Namespace": config.namespace_id},
    )
    config.visual_collection_id = result.collection_id
    console.print(f"    [green]Created:[/green] {result.collection_id}")

    # Audio Content Collection
    console.print("  Creating audio_content collection (transcription + sentiment)...")
    result = client.collections.create(
        collection_name="audio_content",
        description="Audio transcription with semantic embeddings",
        source={"type": "bucket", "bucket_ids": [config.bucket_id]},
        feature_extractor={
            "feature_extractor_name": "multimodal_extractor",
            "version": "v1",
            "input_mappings": {"video": "video"},
            "parameters": {
                "extractor_type": "multimodal_extractor",
                "split_method": "silence",
                "silence_db_threshold": -40,
                "run_transcription": True,
                "run_transcription_embedding": True,
                "run_multimodal_embedding": False,
                "run_video_description": True,
                "description_prompt": (
                    "Analyze the spoken content. Identify: "
                    "1) Main topics discussed, "
                    "2) Sentiment (positive/neutral/negative), "
                    "3) Any brand or product mentions, "
                    "4) Emotional tone of the speaker."
                ),
            },
            "field_passthrough": [
                {"source_path": "metadata.title"},
                {"source_path": "metadata.platform"},
                {"source_path": "metadata.region"},
                {"source_path": "metadata.channel"},
            ],
        },
        _headers={"X-Namespace": config.namespace_id},
    )
    config.audio_collection_id = result.collection_id
    console.print(f"    [green]Created:[/green] {result.collection_id}")

    # Text Content Collection
    console.print("  Creating text_content collection (OCR extraction)...")
    result = client.collections.create(
        collection_name="text_content",
        description="Text extracted from video frames (OCR)",
        source={"type": "bucket", "bucket_ids": [config.bucket_id]},
        feature_extractor={
            "feature_extractor_name": "multimodal_extractor",
            "version": "v1",
            "input_mappings": {"video": "video"},
            "parameters": {
                "extractor_type": "multimodal_extractor",
                "split_method": "time",
                "time_split_interval": 5,
                "run_ocr": True,
                "run_multimodal_embedding": True,
                "run_transcription": False,
                "run_video_description": False,
                "enable_thumbnails": True,
            },
            "field_passthrough": [
                {"source_path": "metadata.title"},
                {"source_path": "metadata.platform"},
                {"source_path": "metadata.region"},
            ],
        },
    )
    config.text_collection_id = result.collection_id
    console.print(f"    [green]Created:[/green] {result.collection_id}")


def create_brand_taxonomy(client: Mixpeek, config: Config) -> None:
    """Create brand reference data and taxonomy."""
    console.print("\n[bold blue]Creating brand taxonomy...[/bold blue]")

    # Create brand reference bucket
    console.print("  Creating brand reference bucket...")
    result = client.buckets.create(
        bucket_name="brand_reference",
        description="Brand reference data for matching",
        bucket_schema={
            "properties": {
                "brand_text": {"type": "text", "required": True},
                "brand_id": {"type": "string"},
                "brand_name": {"type": "string"},
                "brand_category": {"type": "string"},
            }
        },
    )
    config.brand_bucket_id = result.bucket_id
    console.print(f"    [green]Created bucket:[/green] {result.bucket_id}")

    # Create brand collection
    console.print("  Creating brand reference collection...")
    result = client.collections.create(
        collection_name="brand_reference",
        description="Brand embeddings for taxonomy matching",
        source={"type": "bucket", "bucket_ids": [config.brand_bucket_id]},
        feature_extractor={
            "feature_extractor_name": "text_extractor",
            "version": "v1",
            "input_mappings": {"text": "brand_text"},
        },
    )
    config.brand_collection_id = result.collection_id
    console.print(f"    [green]Created collection:[/green] {result.collection_id}")

    # Upload brand data
    console.print("  Uploading brand reference data...")
    object_ids = []
    for brand in BRANDS:
        result = client.bucket_objects.create(
            bucket_id=config.brand_bucket_id,
            object_id=brand["id"],
            blobs=[{"property": "brand_text", "type": "text", "data": brand["text"]}],
            metadata={
                "brand_id": brand["id"],
                "brand_name": brand["name"],
                "brand_category": brand["category"],
            },
        )
        object_ids.append(result.object_id)
        console.print(f"    [dim]Uploaded: {brand['name']}[/dim]")

    # Process brand data
    console.print("  Processing brand data...")
    batch = client.bucket_batches.create(
        bucket_id=config.brand_bucket_id,
        object_ids=object_ids,
    )
    client.bucket_batches.submit(
        bucket_id=config.brand_bucket_id,
        batch_id=batch.batch_id,
    )
    console.print(f"    [yellow]Batch submitted: {batch.batch_id}[/yellow]")

    # Create brand search retriever
    console.print("  Creating brand search retriever...")
    result = client.retrievers.create(
        retriever_name="brand_search",
        description="Search for brand matches",
        collection_identifiers=[config.brand_collection_id],
        input_schema={"query": {"type": "string", "required": True, "description": "Text to match against brands"}},
        stages=[
            {
                "stage_name": "brand_match",
                "stage_id": "feature_search",
                "parameters": {
                    "searches": [
                        {
                            "feature_uri": "mixpeek://text_extractor@v1/multilingual_e5_embedding",
                            "query": {"input_mode": "text", "text": "{{INPUT.query}}"},
                            "top_k": 5,
                            "min_score": 0.5,
                        }
                    ],
                    "final_top_k": 5,
                },
            }
        ],
    )
    config.brand_retriever_id = result.retriever_id
    console.print(f"    [green]Created retriever:[/green] {result.retriever_id}")

    # Create brand taxonomy
    console.print("  Creating brand taxonomy...")
    result = client.taxonomies.create(
        taxonomy_name="brand_taxonomy",
        description="Identify brands in video content",
        config={
            "taxonomy_type": "flat",
            "retriever_id": config.brand_retriever_id,
            "input_mappings": [
                {"input_key": "query", "source_type": "text", "path": "description"}
            ],
            "enrichment_fields": [
                {"field_path": "brand_id", "merge_mode": "replace"},
                {"field_path": "brand_name", "merge_mode": "replace"},
                {"field_path": "brand_category", "merge_mode": "replace"},
            ],
            "collection_configuration": [
                {"collection_id": config.visual_collection_id, "apply_on_ingestion": True}
            ],
        },
    )
    config.brand_taxonomy_id = result.taxonomy_id
    console.print(f"    [green]Created taxonomy:[/green] {result.taxonomy_id}")


def create_sentiment_taxonomy(client: Mixpeek, config: Config) -> None:
    """Create sentiment reference data and taxonomy."""
    console.print("\n[bold blue]Creating sentiment taxonomy...[/bold blue]")

    # Create sentiment reference bucket
    console.print("  Creating sentiment reference bucket...")
    result = client.buckets.create(
        bucket_name="sentiment_reference",
        description="Sentiment labels for classification",
        bucket_schema={
            "properties": {
                "sentiment_text": {"type": "text", "required": True},
                "sentiment_id": {"type": "string"},
                "sentiment_label": {"type": "string"},
            }
        },
    )
    config.sentiment_bucket_id = result.bucket_id
    console.print(f"    [green]Created bucket:[/green] {result.bucket_id}")

    # Create sentiment collection
    console.print("  Creating sentiment reference collection...")
    result = client.collections.create(
        collection_name="sentiment_reference",
        description="Sentiment embeddings for classification",
        source={"type": "bucket", "bucket_ids": [config.sentiment_bucket_id]},
        feature_extractor={
            "feature_extractor_name": "text_extractor",
            "version": "v1",
            "input_mappings": {"text": "sentiment_text"},
        },
    )
    config.sentiment_collection_id = result.collection_id
    console.print(f"    [green]Created collection:[/green] {result.collection_id}")

    # Upload sentiment data
    console.print("  Uploading sentiment reference data...")
    object_ids = []
    for sentiment in SENTIMENTS:
        result = client.bucket_objects.create(
            bucket_id=config.sentiment_bucket_id,
            object_id=sentiment["id"],
            blobs=[{"property": "sentiment_text", "type": "text", "data": sentiment["text"]}],
            metadata={
                "sentiment_id": sentiment["id"],
                "sentiment_label": sentiment["label"],
            },
        )
        object_ids.append(result.object_id)
        console.print(f"    [dim]Uploaded: {sentiment['label']}[/dim]")

    # Process sentiment data
    console.print("  Processing sentiment data...")
    batch = client.bucket_batches.create(
        bucket_id=config.sentiment_bucket_id,
        object_ids=object_ids,
    )
    client.bucket_batches.submit(
        bucket_id=config.sentiment_bucket_id,
        batch_id=batch.batch_id,
    )
    console.print(f"    [yellow]Batch submitted: {batch.batch_id}[/yellow]")

    # Create sentiment search retriever
    console.print("  Creating sentiment search retriever...")
    result = client.retrievers.create(
        retriever_name="sentiment_search",
        description="Classify sentiment from text",
        collection_identifiers=[config.sentiment_collection_id],
        input_schema={"query": {"type": "string", "required": True, "description": "Text to classify"}},
        stages=[
            {
                "stage_name": "sentiment_match",
                "stage_id": "feature_search",
                "parameters": {
                    "searches": [
                        {
                            "feature_uri": "mixpeek://text_extractor@v1/multilingual_e5_embedding",
                            "query": {"input_mode": "text", "text": "{{INPUT.query}}"},
                            "top_k": 1,
                            "min_score": 0.3,
                        }
                    ],
                    "final_top_k": 1,
                },
            }
        ],
    )
    config.sentiment_retriever_id = result.retriever_id
    console.print(f"    [green]Created retriever:[/green] {result.retriever_id}")

    # Create sentiment taxonomy
    console.print("  Creating sentiment taxonomy...")
    result = client.taxonomies.create(
        taxonomy_name="sentiment_taxonomy",
        description="Classify content sentiment",
        config={
            "taxonomy_type": "flat",
            "retriever_id": config.sentiment_retriever_id,
            "input_mappings": [
                {"input_key": "query", "source_type": "text", "path": "description"}
            ],
            "enrichment_fields": [
                {"field_path": "sentiment_label", "merge_mode": "replace"},
            ],
            "collection_configuration": [
                {"collection_id": config.audio_collection_id, "apply_on_ingestion": True}
            ],
        },
    )
    config.sentiment_taxonomy_id = result.taxonomy_id
    console.print(f"    [green]Created taxonomy:[/green] {result.taxonomy_id}")


def create_retrievers(client: Mixpeek, config: Config) -> None:
    """Create search retrievers."""
    console.print("\n[bold blue]Creating retrievers...[/bold blue]")

    # Unified Cross-Modal Search Retriever
    console.print("  Creating unified_video_search retriever...")
    result = client.retrievers.create(
        retriever_name="unified_video_search",
        description="Search across all video modalities (visual, audio, text)",
        collection_identifiers=[config.visual_collection_id, config.audio_collection_id],
        input_schema={
            "query": {"type": "string", "required": True, "description": "Search query"},
            "platform": {"type": "string", "required": False, "description": "Filter by platform"},
            "region": {"type": "string", "required": False, "description": "Filter by region"},
            "sentiment": {"type": "string", "required": False, "description": "Filter by sentiment"},
        },
        stages=[
            {
                "stage_name": "visual_search",
                "stage_id": "feature_search",
                "parameters": {
                    "searches": [
                        {
                            "feature_uri": "mixpeek://multimodal_extractor@v1/vertex_multimodal_embedding",
                            "query": {"input_mode": "text", "text": "{{INPUT.query}}"},
                            "top_k": 50,
                            "min_score": 0.3,
                        }
                    ],
                    "final_top_k": 50,
                },
            },
            {
                "stage_name": "audio_search",
                "stage_id": "feature_search",
                "parameters": {
                    "searches": [
                        {
                            "feature_uri": "mixpeek://multimodal_extractor@v1/multilingual_e5_embedding",
                            "query": {"input_mode": "text", "text": "{{INPUT.query}}"},
                            "top_k": 50,
                            "min_score": 0.3,
                        }
                    ],
                    "final_top_k": 50,
                },
            },
        ],
    )
    config.unified_retriever_id = result.retriever_id
    console.print(f"    [green]Created:[/green] {result.retriever_id}")

    # Brand Monitoring Retriever
    console.print("  Creating brand_monitoring retriever...")
    result = client.retrievers.create(
        retriever_name="brand_monitoring",
        description="Find brand mentions with sentiment context",
        collection_identifiers=[config.visual_collection_id],
        input_schema={
            "brand_query": {"type": "string", "required": True, "description": "Brand name or visual description"},
            "target_sentiment": {"type": "string", "required": False, "description": "Filter by sentiment"},
        },
        stages=[
            {
                "stage_name": "visual_brand_search",
                "stage_id": "feature_search",
                "parameters": {
                    "searches": [
                        {
                            "feature_uri": "mixpeek://multimodal_extractor@v1/vertex_multimodal_embedding",
                            "query": {"input_mode": "text", "text": "{{INPUT.brand_query}}"},
                            "top_k": 100,
                            "min_score": 0.4,
                        }
                    ],
                    "final_top_k": 100,
                },
            },
        ],
    )
    config.brand_monitoring_retriever_id = result.retriever_id
    console.print(f"    [green]Created:[/green] {result.retriever_id}")


def create_clustering(client: Mixpeek, config: Config) -> None:
    """Create clustering for narrative discovery."""
    console.print("\n[bold blue]Creating clustering...[/bold blue]")

    console.print("  Creating narrative_discovery cluster...")
    result = client.clusters.create(
        cluster_name="narrative_discovery",
        description="Discover emerging narratives across video content",
        collection_ids=[config.visual_collection_id],
        vector_config={
            "feature_uri": "mixpeek://multimodal_extractor@v1/vertex_multimodal_embedding",
            "clustering_method": "kmeans",
            "num_clusters": 10,
        },
    )
    config.narrative_cluster_id = result.cluster_id
    console.print(f"    [green]Created:[/green] {result.cluster_id}")


def create_alerts(client: Mixpeek, config: Config) -> None:
    """Create monitoring alerts."""
    console.print("\n[bold blue]Creating alerts...[/bold blue]")

    console.print("  Creating negative_brand_mention alert...")
    result = client.alerts.create(
        name="negative_brand_mention",
        description="Alert when tracked brand appears with negative sentiment",
        retriever_id=config.brand_monitoring_retriever_id,
        trigger_config={
            "inputs": {
                "brand_query": "Nike swoosh logo athletic",
                "target_sentiment": "negative",
            },
            "min_results": 1,
        },
        notification_config={
            "channels": [],  # Add webhook channels as needed
            "include_matches": True,
            "include_scores": True,
        },
        enabled=False,  # Enable after configuring webhooks
    )
    config.negative_brand_alert_id = result.alert_id
    console.print(f"    [green]Created:[/green] {result.alert_id}")
    console.print("    [yellow]Alert created but disabled. Enable after configuring webhook channels.[/yellow]")


def setup_all() -> Config:
    """Run complete setup."""
    console.print(
        Panel.fit(
            "[bold]Social Video Intelligence Showcase[/bold]\n"
            "Setting up multi-modal video analysis infrastructure",
            border_style="blue",
        )
    )

    config = get_config()
    client = Mixpeek(api_key=config.api_key)

    # Set namespace header for all requests
    # Note: SDK should handle this via context or headers

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Setting up infrastructure...", total=None)

        try:
            # Create namespace first
            create_namespace(client, config)

            # Create video bucket
            create_video_bucket(client, config)

            # Create extraction collections
            create_extraction_collections(client, config)

            # Create taxonomies
            create_brand_taxonomy(client, config)
            create_sentiment_taxonomy(client, config)

            # Create retrievers
            create_retrievers(client, config)

            # Create clustering
            create_clustering(client, config)

            # Create alerts
            create_alerts(client, config)

            # Save configuration
            config.save()

            progress.update(task, completed=True)

        except Exception as e:
            console.print(f"\n[red]Error during setup:[/red] {e}")
            raise

    # Print summary
    console.print("\n")
    console.print(
        Panel(
            f"""[bold green]Setup Complete![/bold green]

[bold]Namespace:[/bold] {config.namespace_id}

[bold]Collections:[/bold]
  - Visual Scenes: {config.visual_collection_id}
  - Audio Content: {config.audio_collection_id}
  - Text Content: {config.text_collection_id}

[bold]Taxonomies:[/bold]
  - Brand: {config.brand_taxonomy_id}
  - Sentiment: {config.sentiment_taxonomy_id}

[bold]Retrievers:[/bold]
  - Unified Search: {config.unified_retriever_id}
  - Brand Monitoring: {config.brand_monitoring_retriever_id}

[bold]Clustering:[/bold]
  - Narrative Discovery: {config.narrative_cluster_id}

[bold]Alerts:[/bold]
  - Negative Brand: {config.negative_brand_alert_id}

Configuration saved to: {config.config_file}

[bold]Next steps:[/bold]
1. Ingest sample videos: python -m demo.ingest_videos
2. Run searches: python -m demo.cross_modal_search
3. View clusters: python -m demo.narrative_discovery
""",
            title="Setup Summary",
            border_style="green",
        )
    )

    return config


if __name__ == "__main__":
    setup_all()
