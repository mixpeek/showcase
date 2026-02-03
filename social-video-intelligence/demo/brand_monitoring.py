#!/usr/bin/env python3
"""
Brand monitoring demo.

Find videos where specific brands appear, optionally filtered by sentiment.
Demonstrates the use case: "Find videos where [brand] appears visually + negative sentiment"

Usage:
    python -m demo.brand_monitoring "Nike swoosh logo"
    python -m demo.brand_monitoring "Apple iPhone" --sentiment negative
    python -m demo.brand_monitoring "Coca-Cola" --sentiment positive
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


def execute_brand_search(
    client: Mixpeek,
    retriever_id: str,
    brand_query: str,
    target_sentiment: Optional[str] = None,
    limit: int = 20,
) -> dict:
    """Execute a brand monitoring search."""
    inputs = {"brand_query": brand_query}
    if target_sentiment:
        inputs["target_sentiment"] = target_sentiment

    try:
        result = client.retrievers.execute(
            retriever_id=retriever_id,
            inputs=inputs,
            settings={"limit": limit},
        )
        return result.model_dump() if hasattr(result, 'model_dump') else result
    except Exception as e:
        return {"error": str(e)}


def display_brand_results(results: dict, brand_query: str, sentiment_filter: Optional[str]) -> None:
    """Display brand monitoring results."""
    if "error" in results:
        console.print(f"[red]Search error:[/red] {results['error']}")
        return

    documents = results.get("documents", results.get("results", []))

    # Build header
    header = f"Brand: [bold]{brand_query}[/bold]"
    if sentiment_filter:
        sentiment_color = {
            "positive": "green",
            "negative": "red",
            "neutral": "yellow",
        }.get(sentiment_filter, "white")
        header += f" | Sentiment: [{sentiment_color}]{sentiment_filter}[/{sentiment_color}]"

    if not documents:
        console.print(Panel(
            f"No mentions found.\n{header}",
            title="Brand Monitoring Results",
            border_style="yellow"
        ))
        return

    console.print(Panel(
        f"Found [bold]{len(documents)}[/bold] mention(s)\n{header}",
        title="Brand Monitoring Results",
        border_style="blue"
    ))

    # Group by source video
    videos = {}
    for doc in documents:
        video_url = doc.get("source_video_url", doc.get("title", "unknown"))
        if video_url not in videos:
            videos[video_url] = {
                "title": doc.get("title", "Unknown"),
                "platform": doc.get("platform", "N/A"),
                "region": doc.get("region", "N/A"),
                "segments": [],
                "sentiments": set(),
                "brands": set(),
            }
        videos[video_url]["segments"].append({
            "start": doc.get("start_time"),
            "end": doc.get("end_time"),
            "description": doc.get("description"),
            "score": doc.get("score"),
        })
        if doc.get("sentiment_label"):
            videos[video_url]["sentiments"].add(doc["sentiment_label"])
        if doc.get("brand_name"):
            videos[video_url]["brands"].add(doc["brand_name"])

    # Display grouped results
    for i, (video_url, video_data) in enumerate(videos.items(), 1):
        console.print(f"\n[bold cyan]Video {i}: {video_data['title']}[/bold cyan]")
        console.print(f"  Platform: {video_data['platform']} | Region: {video_data['region']}")
        console.print(f"  Segments with brand: {len(video_data['segments'])}")

        if video_data["sentiments"]:
            sentiments_str = ", ".join(video_data["sentiments"])
            console.print(f"  Sentiments detected: {sentiments_str}")

        if video_data["brands"]:
            brands_str = ", ".join(video_data["brands"])
            console.print(f"  Brands matched: {brands_str}")

        # Show top segments
        console.print("  [dim]Top segments:[/dim]")
        for seg in sorted(video_data["segments"], key=lambda x: x.get("score", 0), reverse=True)[:3]:
            time_str = f"{seg['start']:.1f}s-{seg['end']:.1f}s" if seg.get("start") is not None else "N/A"
            score_str = f"{seg['score']:.3f}" if seg.get("score") else "N/A"
            console.print(f"    [{score_str}] {time_str}")
            if seg.get("description"):
                desc = seg["description"][:100] + "..." if len(seg["description"]) > 100 else seg["description"]
                console.print(f"          {desc}")

    # Summary statistics
    console.print("\n")
    table = Table(title="Summary Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Total Videos", str(len(videos)))
    table.add_row("Total Segments", str(len(documents)))

    all_sentiments = set()
    for v in videos.values():
        all_sentiments.update(v["sentiments"])
    table.add_row("Sentiment Distribution", ", ".join(all_sentiments) if all_sentiments else "N/A")

    all_brands = set()
    for v in videos.values():
        all_brands.update(v["brands"])
    table.add_row("Brands Detected", ", ".join(all_brands) if all_brands else "N/A")

    console.print(table)


@click.command()
@click.argument("brand_query")
@click.option(
    "--sentiment",
    type=click.Choice(["positive", "neutral", "negative"]),
    help="Filter by sentiment"
)
@click.option("--limit", default=20, help="Maximum results to return")
@click.option("--raw", is_flag=True, help="Output raw JSON")
def main(brand_query: str, sentiment: Optional[str], limit: int, raw: bool):
    """
    Monitor brand mentions in video content.

    BRAND_QUERY: Brand name or visual description to search for

    Examples:
        python -m demo.brand_monitoring "Nike swoosh logo"
        python -m demo.brand_monitoring "Apple iPhone" --sentiment negative
        python -m demo.brand_monitoring "McDonald's golden arches" --sentiment positive
    """
    config = get_config()
    client = Mixpeek(api_key=config.api_key)

    if not config.brand_monitoring_retriever_id:
        console.print("[red]Error: Retrievers not configured. Run setup first:[/red]")
        console.print("  python -m setup.setup_all")
        return

    console.print(f"\n[bold blue]Brand Monitoring Search[/bold blue]")
    console.print(f"Brand Query: [italic]{brand_query}[/italic]")
    if sentiment:
        console.print(f"Sentiment Filter: {sentiment}")

    # Execute search
    results = execute_brand_search(
        client,
        config.brand_monitoring_retriever_id,
        brand_query,
        target_sentiment=sentiment,
        limit=limit,
    )

    if raw:
        console.print(json.dumps(results, indent=2))
    else:
        display_brand_results(results, brand_query, sentiment)

    # Alert suggestion
    if sentiment == "negative" and not raw:
        console.print("\n[yellow]Tip:[/yellow] Set up an alert to monitor this brand for negative sentiment:")
        console.print(f"  Alert ID: {config.negative_brand_alert_id}")


if __name__ == "__main__":
    main()
