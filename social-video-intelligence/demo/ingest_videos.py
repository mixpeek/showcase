#!/usr/bin/env python3
"""
Ingest sample videos for the Social Video Intelligence showcase.

This script uploads sample videos to the bucket and triggers processing
through all extraction collections (visual, audio, text).

Usage:
    python -m demo.ingest_videos

Or with custom videos:
    python -m demo.ingest_videos --video-url https://example.com/video.mp4
"""

import sys
import time
from pathlib import Path
from typing import List, Optional

import click
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))

from mixpeek import Mixpeek

from setup.config import get_config

console = Console()


# Sample video URLs for demo (public domain / Creative Commons)
SAMPLE_VIDEOS = [
    {
        "url": "https://storage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4",
        "title": "Big Buck Bunny",
        "description": "Animated short film about a rabbit and forest creatures",
        "platform": "youtube",
        "region": "US",
        "channel": "Blender Foundation",
        "tags": ["animation", "short film", "comedy"],
    },
    {
        "url": "https://storage.googleapis.com/gtv-videos-bucket/sample/ElephantsDream.mp4",
        "title": "Elephants Dream",
        "description": "Surrealist animated short film",
        "platform": "youtube",
        "region": "EU",
        "channel": "Blender Foundation",
        "tags": ["animation", "surreal", "art"],
    },
    {
        "url": "https://storage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4",
        "title": "For Bigger Blazes",
        "description": "Action movie trailer with explosions",
        "platform": "youtube",
        "region": "US",
        "channel": "Google",
        "tags": ["trailer", "action", "movie"],
    },
    {
        "url": "https://storage.googleapis.com/gtv-videos-bucket/sample/ForBiggerEscapes.mp4",
        "title": "For Bigger Escapes",
        "description": "Adventure travel documentary style video",
        "platform": "youtube",
        "region": "APAC",
        "channel": "Google",
        "tags": ["travel", "adventure", "documentary"],
    },
    {
        "url": "https://storage.googleapis.com/gtv-videos-bucket/sample/ForBiggerFun.mp4",
        "title": "For Bigger Fun",
        "description": "Entertainment and lifestyle content",
        "platform": "tiktok",
        "region": "US",
        "channel": "Google",
        "tags": ["entertainment", "lifestyle", "fun"],
    },
]


def ingest_video(
    client: Mixpeek,
    bucket_id: str,
    video: dict,
) -> Optional[str]:
    """Ingest a single video."""
    try:
        result = client.bucket_objects.create(
            bucket_id=bucket_id,
            blobs=[{"property": "video", "type": "url", "data": video["url"]}],
            metadata={
                "title": video["title"],
                "description": video.get("description", ""),
                "platform": video.get("platform", "unknown"),
                "region": video.get("region", "unknown"),
                "channel": video.get("channel", "unknown"),
                "tags": video.get("tags", []),
                "upload_date": time.strftime("%Y-%m-%d"),
                "view_count": video.get("view_count", 0),
                "engagement_score": video.get("engagement_score", 0.0),
            },
        )
        return result.object_id
    except Exception as e:
        console.print(f"[red]Error uploading {video['title']}:[/red] {e}")
        return None


def process_batch(client: Mixpeek, bucket_id: str, object_ids: List[str]) -> Optional[str]:
    """Create and submit a batch for processing."""
    try:
        batch = client.bucket_batches.create(
            bucket_id=bucket_id,
            object_ids=object_ids,
        )
        result = client.bucket_batches.submit(
            bucket_id=bucket_id,
            batch_id=batch.batch_id,
        )
        return batch.batch_id
    except Exception as e:
        console.print(f"[red]Error creating batch:[/red] {e}")
        return None


@click.command()
@click.option(
    "--video-url",
    multiple=True,
    help="Custom video URL to ingest (can be specified multiple times)",
)
@click.option(
    "--use-samples/--no-samples",
    default=True,
    help="Include sample videos",
)
def main(video_url: tuple, use_samples: bool):
    """Ingest videos into the Social Video Intelligence showcase."""
    config = get_config()
    client = Mixpeek(api_key=config.api_key)

    if not config.bucket_id:
        console.print("[red]Error: No bucket configured. Run setup first:[/red]")
        console.print("  python -m setup.setup_all")
        return

    console.print("\n[bold blue]Ingesting Videos[/bold blue]")
    console.print(f"Bucket: {config.bucket_id}")

    # Build video list
    videos = []
    if use_samples:
        videos.extend(SAMPLE_VIDEOS)
    for url in video_url:
        videos.append({
            "url": url,
            "title": f"Custom Video {len(videos) + 1}",
            "platform": "custom",
            "region": "unknown",
        })

    if not videos:
        console.print("[yellow]No videos to ingest. Use --video-url or --use-samples[/yellow]")
        return

    # Upload videos
    console.print(f"\nUploading {len(videos)} video(s)...")
    object_ids = []
    for video in videos:
        console.print(f"  Uploading: {video['title']}")
        obj_id = ingest_video(client, config.bucket_id, video)
        if obj_id:
            object_ids.append(obj_id)
            console.print(f"    [green]Success:[/green] {obj_id}")
        else:
            console.print(f"    [red]Failed[/red]")

    if not object_ids:
        console.print("[red]No videos were uploaded successfully[/red]")
        return

    # Create and submit batch
    console.print(f"\nCreating batch for {len(object_ids)} object(s)...")
    batch_id = process_batch(client, config.bucket_id, object_ids)
    if batch_id:
        console.print(f"[green]Batch submitted:[/green] {batch_id}")
    else:
        console.print("[red]Failed to submit batch[/red]")
        return

    # Summary table
    table = Table(title="Ingestion Summary")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Videos Uploaded", str(len(object_ids)))
    table.add_row("Batch ID", batch_id)
    table.add_row("Bucket ID", config.bucket_id)
    console.print("\n")
    console.print(table)

    console.print(
        "\n[yellow]Processing has started. Videos will be processed through:[/yellow]"
    )
    console.print(f"  - Visual Scenes ({config.visual_collection_id})")
    console.print(f"  - Audio Content ({config.audio_collection_id})")
    console.print(f"  - Text Content ({config.text_collection_id})")
    console.print(
        "\n[dim]Processing may take several minutes depending on video length.[/dim]"
    )
    console.print("\n[bold]Next steps:[/bold]")
    console.print("  - Check processing status: python -m demo.check_status")
    console.print("  - Run searches: python -m demo.cross_modal_search")


if __name__ == "__main__":
    main()
