#!/usr/bin/env python3
"""
Social Video Intelligence CLI

Command-line interface for the Social Video Intelligence showcase.
Demonstrates multi-modal video analysis, brand monitoring, and narrative discovery.

Usage:
    python cli.py setup        # Set up all infrastructure
    python cli.py ingest       # Ingest sample videos
    python cli.py search       # Run cross-modal search
    python cli.py monitor      # Brand monitoring
    python cli.py clusters     # Narrative discovery
"""

import click
from rich.console import Console

console = Console()


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """
    Social Video Intelligence - Multi-modal video analysis showcase.

    Demonstrates running parallel extractors across social/YouTube content,
    indexing each layer separately, and querying together for insights like:

    "Find videos where [brand] appears visually + negative sentiment + trending in [region]"

    Built with the Mixpeek SDK.
    """
    pass


@cli.command()
def setup():
    """Set up all Mixpeek infrastructure."""
    from setup.setup_all import setup_all
    setup_all()


@cli.command()
@click.option("--video-url", multiple=True, help="Custom video URL to ingest")
@click.option("--use-samples/--no-samples", default=True, help="Include sample videos")
def ingest(video_url, use_samples):
    """Ingest videos for processing."""
    from demo.ingest_videos import main as ingest_main
    # Invoke with click context
    from click.testing import CliRunner
    runner = CliRunner()

    args = []
    if not use_samples:
        args.append("--no-samples")
    for url in video_url:
        args.extend(["--video-url", url])

    # Import and call directly
    import sys
    sys.argv = ["ingest"] + args
    ingest_main(standalone_mode=False)


@cli.command()
@click.argument("query")
@click.option("--platform", help="Filter by platform (youtube, tiktok, etc.)")
@click.option("--region", help="Filter by region (US, EU, APAC)")
@click.option("--sentiment", type=click.Choice(["positive", "neutral", "negative", "mixed"]))
@click.option("--limit", default=10, help="Maximum results")
@click.option("--raw", is_flag=True, help="Output raw JSON")
def search(query, platform, region, sentiment, limit, raw):
    """
    Search videos across all modalities.

    Examples:
        python cli.py search "basketball highlights"
        python cli.py search "product review" --sentiment negative
        python cli.py search "tech unboxing" --platform youtube --region US
    """
    from demo.cross_modal_search import main as search_main
    import sys
    args = [query]
    if platform:
        args.extend(["--platform", platform])
    if region:
        args.extend(["--region", region])
    if sentiment:
        args.extend(["--sentiment", sentiment])
    args.extend(["--limit", str(limit)])
    if raw:
        args.append("--raw")
    sys.argv = ["search"] + args
    search_main(standalone_mode=False)


@cli.command()
@click.argument("brand_query")
@click.option("--sentiment", type=click.Choice(["positive", "neutral", "negative"]))
@click.option("--limit", default=20, help="Maximum results")
@click.option("--raw", is_flag=True, help="Output raw JSON")
def monitor(brand_query, sentiment, limit, raw):
    """
    Monitor brand mentions with sentiment analysis.

    Examples:
        python cli.py monitor "Nike swoosh logo"
        python cli.py monitor "Apple iPhone" --sentiment negative
        python cli.py monitor "Coca-Cola" --sentiment positive
    """
    from demo.brand_monitoring import main as monitor_main
    import sys
    args = [brand_query]
    if sentiment:
        args.extend(["--sentiment", sentiment])
    args.extend(["--limit", str(limit)])
    if raw:
        args.append("--raw")
    sys.argv = ["monitor"] + args
    monitor_main(standalone_mode=False)


@cli.command()
@click.option("--execute", is_flag=True, help="Run fresh clustering")
@click.option("--raw", is_flag=True, help="Output raw JSON")
def clusters(execute, raw):
    """
    Discover emerging narratives through clustering.

    Examples:
        python cli.py clusters              # View existing clusters
        python cli.py clusters --execute    # Run fresh clustering
    """
    from demo.narrative_discovery import main as cluster_main
    import sys
    args = []
    if execute:
        args.append("--execute")
    if raw:
        args.append("--raw")
    sys.argv = ["clusters"] + args
    cluster_main(standalone_mode=False)


@cli.command()
def status():
    """Check processing status and resource health."""
    from setup.config import get_config

    config = get_config()
    console.print("\n[bold blue]Social Video Intelligence Status[/bold blue]\n")

    # Check configuration
    console.print("[bold]Configuration:[/bold]")
    console.print(f"  Namespace: {config.namespace_id or '[red]Not set[/red]'}")
    console.print(f"  Bucket: {config.bucket_id or '[red]Not set[/red]'}")

    console.print("\n[bold]Collections:[/bold]")
    console.print(f"  Visual Scenes: {config.visual_collection_id or '[red]Not set[/red]'}")
    console.print(f"  Audio Content: {config.audio_collection_id or '[red]Not set[/red]'}")
    console.print(f"  Text Content: {config.text_collection_id or '[red]Not set[/red]'}")

    console.print("\n[bold]Taxonomies:[/bold]")
    console.print(f"  Brand: {config.brand_taxonomy_id or '[red]Not set[/red]'}")
    console.print(f"  Sentiment: {config.sentiment_taxonomy_id or '[red]Not set[/red]'}")

    console.print("\n[bold]Retrievers:[/bold]")
    console.print(f"  Unified Search: {config.unified_retriever_id or '[red]Not set[/red]'}")
    console.print(f"  Brand Monitoring: {config.brand_monitoring_retriever_id or '[red]Not set[/red]'}")

    console.print("\n[bold]Clustering:[/bold]")
    console.print(f"  Narrative Discovery: {config.narrative_cluster_id or '[red]Not set[/red]'}")

    console.print("\n[bold]Alerts:[/bold]")
    console.print(f"  Negative Brand: {config.negative_brand_alert_id or '[red]Not set[/red]'}")

    if not config.namespace_id:
        console.print("\n[yellow]Run setup to initialize:[/yellow]")
        console.print("  python cli.py setup")


if __name__ == "__main__":
    cli()
