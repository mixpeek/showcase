#!/usr/bin/env python3
"""
Social Video Intelligence - Complete Demo Runner

This script runs the complete end-to-end demonstration:
1. Sets up all infrastructure (namespace, buckets, collections, taxonomies, retrievers, clusters, alerts)
2. Ingests sample videos
3. Waits for processing
4. Runs sample queries
5. Shows results

Usage:
    # With environment variable
    export MIXPEEK_API_KEY=your_api_key
    python run_demo.py

    # Or pass as argument
    python run_demo.py --api-key your_api_key
"""

import os
import sys
import time
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

# Ensure imports work
sys.path.insert(0, str(Path(__file__).parent))

console = Console()


def wait_with_progress(seconds: int, message: str):
    """Wait with a progress indicator."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(message, total=seconds)
        for _ in range(seconds):
            time.sleep(1)
            progress.advance(task)


@click.command()
@click.option("--api-key", envvar="MIXPEEK_API_KEY", help="Mixpeek API key")
@click.option("--skip-setup", is_flag=True, help="Skip setup if already done")
@click.option("--skip-ingest", is_flag=True, help="Skip video ingestion")
@click.option("--wait-time", default=60, help="Seconds to wait for processing")
def main(api_key: str, skip_setup: bool, skip_ingest: bool, wait_time: int):
    """Run the complete Social Video Intelligence demo."""

    console.print(Panel.fit(
        "[bold blue]Social Video Intelligence Demo[/bold blue]\n"
        "Multi-modal video analysis for brand monitoring and content intelligence",
        border_style="blue"
    ))

    # Validate API key
    if not api_key:
        console.print("\n[red]Error: Mixpeek API key is required.[/red]")
        console.print("\nSet it via environment variable:")
        console.print("  export MIXPEEK_API_KEY=your_api_key")
        console.print("\nOr pass as argument:")
        console.print("  python run_demo.py --api-key your_api_key")
        console.print("\nGet your API key at: https://mixpeek.com/dashboard")
        sys.exit(1)

    # Set API key in environment for submodules
    os.environ["MIXPEEK_API_KEY"] = api_key

    # Step 1: Setup
    if not skip_setup:
        console.print("\n" + "=" * 60)
        console.print("[bold]Step 1: Setting up infrastructure[/bold]")
        console.print("=" * 60)

        from setup.setup_all import setup_all
        config = setup_all()
    else:
        console.print("\n[yellow]Skipping setup (--skip-setup)[/yellow]")
        from setup.config import get_config
        config = get_config()

    # Step 2: Ingest videos
    if not skip_ingest:
        console.print("\n" + "=" * 60)
        console.print("[bold]Step 2: Ingesting sample videos[/bold]")
        console.print("=" * 60)

        from mixpeek import Mixpeek
        from demo.ingest_videos import SAMPLE_VIDEOS, ingest_video, process_batch

        client = Mixpeek(api_key=api_key)

        console.print(f"\nUploading {len(SAMPLE_VIDEOS)} sample videos...")
        object_ids = []
        for video in SAMPLE_VIDEOS[:5]:  # Limit to 5 for demo
            console.print(f"  Uploading: {video['title']}")
            obj_id = ingest_video(client, config.bucket_id, video)
            if obj_id:
                object_ids.append(obj_id)
                console.print(f"    [green]OK[/green]")

        if object_ids:
            console.print(f"\nCreating batch for {len(object_ids)} videos...")
            batch_id = process_batch(client, config.bucket_id, object_ids)
            console.print(f"[green]Batch submitted: {batch_id}[/green]")
    else:
        console.print("\n[yellow]Skipping ingestion (--skip-ingest)[/yellow]")

    # Step 3: Wait for processing
    console.print("\n" + "=" * 60)
    console.print("[bold]Step 3: Waiting for video processing[/bold]")
    console.print("=" * 60)
    console.print(f"\n[dim]Processing videos through 3 parallel extractors...[/dim]")

    wait_with_progress(wait_time, f"Processing (waiting {wait_time}s for indexing)...")

    # Step 4: Run sample queries
    console.print("\n" + "=" * 60)
    console.print("[bold]Step 4: Running sample queries[/bold]")
    console.print("=" * 60)

    from mixpeek import Mixpeek
    client = Mixpeek(api_key=api_key)

    # Query 1: Cross-modal search
    console.print("\n[cyan]Query 1: Cross-modal search for 'action adventure'[/cyan]")
    try:
        result = client.retrievers.execute(
            retriever_id=config.unified_retriever_id,
            inputs={"query": "action adventure excitement"},
            settings={"limit": 5}
        )
        docs = result.documents if hasattr(result, 'documents') else result.get('documents', [])
        console.print(f"  Found {len(docs)} results")
        for doc in docs[:3]:
            title = doc.get('title') if isinstance(doc, dict) else getattr(doc, 'title', 'N/A')
            score = doc.get('score') if isinstance(doc, dict) else getattr(doc, 'score', 0)
            console.print(f"    - {title} (score: {score:.3f})" if isinstance(score, float) else f"    - {title}")
    except Exception as e:
        console.print(f"  [yellow]Query not ready yet: {e}[/yellow]")

    # Query 2: Brand monitoring
    console.print("\n[cyan]Query 2: Brand monitoring for 'Subaru'[/cyan]")
    try:
        result = client.retrievers.execute(
            retriever_id=config.brand_monitoring_retriever_id,
            inputs={"brand_query": "Subaru automotive car vehicle"},
            settings={"limit": 5}
        )
        docs = result.documents if hasattr(result, 'documents') else result.get('documents', [])
        console.print(f"  Found {len(docs)} brand mentions")
        for doc in docs[:3]:
            title = doc.get('title') if isinstance(doc, dict) else getattr(doc, 'title', 'N/A')
            console.print(f"    - {title}")
    except Exception as e:
        console.print(f"  [yellow]Query not ready yet: {e}[/yellow]")

    # Summary
    console.print("\n" + "=" * 60)
    console.print("[bold green]Demo Complete![/bold green]")
    console.print("=" * 60)

    console.print(f"""
[bold]What was demonstrated:[/bold]
  1. Created namespace with 3 parallel extraction collections
  2. Set up brand and sentiment taxonomies
  3. Created cross-modal search and brand monitoring retrievers
  4. Ingested sample videos for processing
  5. Ran semantic queries across visual, audio, and text modalities

[bold]Try more queries:[/bold]
  python cli.py search "animated film nature"
  python cli.py search "technology" --sentiment positive
  python cli.py monitor "Google logo" --sentiment neutral
  python cli.py clusters --execute

[bold]Resources created:[/bold]
  - Namespace: {config.namespace_id}
  - Unified Retriever: {config.unified_retriever_id}
  - Brand Monitor: {config.brand_monitoring_retriever_id}
  - Cluster: {config.narrative_cluster_id}
""")


if __name__ == "__main__":
    main()
