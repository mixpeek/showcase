#!/usr/bin/env python3
"""
Narrative discovery demo using clustering.

Automatically surfaces emerging themes and narratives across video content
by clustering similar segments together.

Usage:
    python -m demo.narrative_discovery
    python -m demo.narrative_discovery --execute  # Run fresh clustering
"""

import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))

from mixpeek import Mixpeek

from setup.config import get_config

console = Console()


def get_cluster_results(client: Mixpeek, cluster_id: str) -> dict:
    """Get existing cluster results."""
    try:
        result = client.clusters.list_executions(cluster_id=cluster_id)
        return result.model_dump() if hasattr(result, 'model_dump') else result
    except Exception as e:
        return {"error": str(e)}


def execute_clustering(client: Mixpeek, cluster_id: str) -> dict:
    """Execute fresh clustering."""
    try:
        result = client.clusters.execute(cluster_id=cluster_id)
        return result.model_dump() if hasattr(result, 'model_dump') else result
    except Exception as e:
        return {"error": str(e)}


def display_clusters(results: dict) -> None:
    """Display cluster results."""
    if "error" in results:
        console.print(f"[red]Clustering error:[/red] {results['error']}")
        return

    executions = results.get("executions", [])
    if not executions:
        console.print("[yellow]No cluster executions found. Run with --execute to generate clusters.[/yellow]")
        return

    # Get most recent execution
    latest = executions[0] if executions else None
    if not latest:
        console.print("[yellow]No cluster data available.[/yellow]")
        return

    clusters = latest.get("clusters", [])
    console.print(Panel(
        f"Found [bold]{len(clusters)}[/bold] narrative clusters",
        title="Narrative Discovery Results",
        border_style="blue"
    ))

    # Display each cluster
    for i, cluster in enumerate(clusters, 1):
        cluster_id = cluster.get("cluster_id", f"cluster_{i}")
        label = cluster.get("label", "Unlabeled")
        doc_count = cluster.get("document_count", 0)
        summary = cluster.get("summary", "No summary available")

        console.print(f"\n[bold cyan]Cluster {i}: {label}[/bold cyan]")
        console.print(f"  Documents: {doc_count}")
        console.print(f"  Summary: {summary}")

        # Show sample documents if available
        sample_docs = cluster.get("sample_documents", [])
        if sample_docs:
            console.print("  [dim]Sample content:[/dim]")
            for doc in sample_docs[:3]:
                title = doc.get("title", "Untitled")
                console.print(f"    - {title}")

    # Summary table
    console.print("\n")
    table = Table(title="Cluster Statistics")
    table.add_column("Cluster", style="cyan")
    table.add_column("Label", style="white")
    table.add_column("Documents", style="green", justify="right")

    for i, cluster in enumerate(clusters, 1):
        table.add_row(
            str(i),
            cluster.get("label", "Unlabeled")[:30],
            str(cluster.get("document_count", 0))
        )
    console.print(table)


@click.command()
@click.option("--execute", is_flag=True, help="Execute fresh clustering")
@click.option("--raw", is_flag=True, help="Output raw JSON")
def main(execute: bool, raw: bool):
    """
    Discover emerging narratives through automatic clustering.

    Examples:
        python -m demo.narrative_discovery             # View existing clusters
        python -m demo.narrative_discovery --execute   # Run fresh clustering
    """
    config = get_config()
    client = Mixpeek(api_key=config.api_key)

    if not config.narrative_cluster_id:
        console.print("[red]Error: Clustering not configured. Run setup first:[/red]")
        console.print("  python -m setup.setup_all")
        return

    console.print(f"\n[bold blue]Narrative Discovery[/bold blue]")
    console.print(f"Cluster ID: {config.narrative_cluster_id}")

    if execute:
        console.print("\n[yellow]Executing fresh clustering...[/yellow]")
        results = execute_clustering(client, config.narrative_cluster_id)
        if "error" not in results:
            console.print("[green]Clustering complete![/green]")
    else:
        results = get_cluster_results(client, config.narrative_cluster_id)

    if raw:
        console.print(json.dumps(results, indent=2))
    else:
        display_clusters(results)


if __name__ == "__main__":
    main()
