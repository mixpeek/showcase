#!/usr/bin/env python3
"""
Cross-modal video search demo.

Demonstrates querying across visual, audio, and text modalities
to find relevant video content.

Usage:
    python -m demo.cross_modal_search "basketball game highlights"
    python -m demo.cross_modal_search "product review" --sentiment negative
    python -m demo.cross_modal_search "brand logo" --platform youtube --region US
"""

import json
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))

from mixpeek import Mixpeek

from setup.config import get_config

console = Console()


def execute_search(
    client: Mixpeek,
    retriever_id: str,
    query: str,
    platform: Optional[str] = None,
    region: Optional[str] = None,
    sentiment: Optional[str] = None,
    limit: int = 10,
) -> dict:
    """Execute a cross-modal search."""
    inputs = {"query": query}
    if platform:
        inputs["platform"] = platform
    if region:
        inputs["region"] = region
    if sentiment:
        inputs["sentiment"] = sentiment

    try:
        result = client.retrievers.execute(
            retriever_id=retriever_id,
            inputs=inputs,
            settings={"limit": limit},
        )
        return result.model_dump() if hasattr(result, 'model_dump') else result
    except Exception as e:
        return {"error": str(e)}


def display_results(results: dict, query: str) -> None:
    """Display search results in a formatted table."""
    if "error" in results:
        console.print(f"[red]Search error:[/red] {results['error']}")
        return

    documents = results.get("documents", results.get("results", []))

    if not documents:
        console.print(f"[yellow]No results found for query:[/yellow] {query}")
        return

    console.print(
        Panel(
            f"Found [bold]{len(documents)}[/bold] result(s) for: [italic]{query}[/italic]",
            title="Search Results",
            border_style="blue",
        )
    )

    for i, doc in enumerate(documents[:10], 1):
        console.print(f"\n[bold cyan]Result {i}[/bold cyan]")
        console.print(f"  [bold]Title:[/bold] {doc.get('title', 'N/A')}")
        console.print(f"  [bold]Score:[/bold] {doc.get('score', 'N/A'):.4f}" if isinstance(doc.get('score'), float) else f"  [bold]Score:[/bold] N/A")
        console.print(f"  [bold]Platform:[/bold] {doc.get('platform', 'N/A')}")
        console.print(f"  [bold]Region:[/bold] {doc.get('region', 'N/A')}")

        if doc.get("sentiment_label"):
            sentiment_color = {
                "positive": "green",
                "negative": "red",
                "neutral": "yellow",
                "mixed": "magenta",
            }.get(doc["sentiment_label"], "white")
            console.print(f"  [bold]Sentiment:[/bold] [{sentiment_color}]{doc['sentiment_label']}[/{sentiment_color}]")

        if doc.get("brand_name"):
            console.print(f"  [bold]Brand:[/bold] {doc['brand_name']} ({doc.get('brand_category', 'N/A')})")

        if doc.get("description"):
            desc = doc["description"][:200] + "..." if len(doc.get("description", "")) > 200 else doc.get("description", "")
            console.print(f"  [bold]Description:[/bold] {desc}")

        if doc.get("transcription"):
            trans = doc["transcription"][:200] + "..." if len(doc.get("transcription", "")) > 200 else doc.get("transcription", "")
            console.print(f"  [bold]Transcription:[/bold] {trans}")

        if doc.get("start_time") is not None and doc.get("end_time") is not None:
            console.print(f"  [bold]Segment:[/bold] {doc['start_time']:.1f}s - {doc['end_time']:.1f}s")

        if doc.get("thumbnail_url"):
            console.print(f"  [bold]Thumbnail:[/bold] [link={doc['thumbnail_url']}]{doc['thumbnail_url'][:50]}...[/link]")


@click.command()
@click.argument("query")
@click.option("--platform", help="Filter by platform (youtube, tiktok, etc.)")
@click.option("--region", help="Filter by region (US, EU, APAC)")
@click.option("--sentiment", type=click.Choice(["positive", "neutral", "negative", "mixed"]), help="Filter by sentiment")
@click.option("--limit", default=10, help="Maximum results to return")
@click.option("--raw", is_flag=True, help="Output raw JSON")
def main(query: str, platform: Optional[str], region: Optional[str], sentiment: Optional[str], limit: int, raw: bool):
    """
    Search videos across visual, audio, and text modalities.

    QUERY: Natural language search query (e.g., "product unboxing review")

    Examples:
        python -m demo.cross_modal_search "basketball game highlights"
        python -m demo.cross_modal_search "cooking recipe" --sentiment positive
        python -m demo.cross_modal_search "tech review" --platform youtube --region US
    """
    config = get_config()
    client = Mixpeek(api_key=config.api_key)

    if not config.unified_retriever_id:
        console.print("[red]Error: Retrievers not configured. Run setup first:[/red]")
        console.print("  python -m setup.setup_all")
        return

    console.print(f"\n[bold blue]Cross-Modal Video Search[/bold blue]")
    console.print(f"Query: [italic]{query}[/italic]")
    if platform:
        console.print(f"Platform filter: {platform}")
    if region:
        console.print(f"Region filter: {region}")
    if sentiment:
        console.print(f"Sentiment filter: {sentiment}")

    # Execute search
    results = execute_search(
        client,
        config.unified_retriever_id,
        query,
        platform=platform,
        region=region,
        sentiment=sentiment,
        limit=limit,
    )

    if raw:
        console.print(json.dumps(results, indent=2))
    else:
        display_results(results, query)


if __name__ == "__main__":
    main()
