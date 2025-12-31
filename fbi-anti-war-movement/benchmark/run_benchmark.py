#!/usr/bin/env python3
"""
Benchmark Runner
----------------
Compares GPT Baseline vs SOTA Pipeline on FBI documents.

Metrics:
- Retrieval precision (is relevant doc in top-k?)
- Mean Reciprocal Rank (MRR)
- Search latency
- Coverage (% of docs successfully indexed)
"""

import os
import sys
import json
import time
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

# Add pipeline directories to path
sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline" / "baseline"))
sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline" / "advanced"))

# Load environment
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from gpt_baseline import GPTBaselinePipeline
from sota_pipeline import SOTAPipeline


@dataclass
class BenchmarkQuery:
    """A benchmark query with expected results."""
    query: str
    category: str  # topic, entity, event, etc.
    expected_keywords: List[str]  # Keywords that should appear in good results
    expected_files: List[str]  # Files that are likely relevant (partial match ok)
    difficulty: str  # easy, medium, hard


# Benchmark queries designed for FBI anti-war documents
BENCHMARK_QUERIES = [
    # Easy - Direct entity mentions
    BenchmarkQuery(
        query="Abbie Hoffman arrest records",
        category="entity",
        expected_keywords=["hoffman", "arrest", "detained", "charges"],
        expected_files=["abbie-hoffman"],
        difficulty="easy",
    ),
    BenchmarkQuery(
        query="Kent State shooting incident",
        category="event",
        expected_keywords=["kent", "state", "shooting", "national guard", "students"],
        expected_files=["kent-state"],
        difficulty="easy",
    ),
    BenchmarkQuery(
        query="Howard Zinn FBI file",
        category="entity",
        expected_keywords=["zinn", "boston", "professor", "historian"],
        expected_files=["howard-zinn"],
        difficulty="easy",
    ),

    # Medium - Topical queries
    BenchmarkQuery(
        query="surveillance of anti-war activists",
        category="topic",
        expected_keywords=["surveillance", "monitor", "informant", "investigation"],
        expected_files=["abbie-hoffman", "howard-zinn", "mario-savio"],
        difficulty="medium",
    ),
    BenchmarkQuery(
        query="communist party infiltration",
        category="topic",
        expected_keywords=["communist", "party", "infiltrat", "subversive"],
        expected_files=["american-friends", "clergy-laity"],
        difficulty="medium",
    ),
    BenchmarkQuery(
        query="student protest movements 1960s",
        category="topic",
        expected_keywords=["student", "protest", "demonstration", "university"],
        expected_files=["mario-savio", "kent-state"],
        difficulty="medium",
    ),
    BenchmarkQuery(
        query="FBI informant reports",
        category="topic",
        expected_keywords=["informant", "source", "confidential", "report"],
        expected_files=[],  # Could be any file
        difficulty="medium",
    ),

    # Hard - Complex/implicit queries
    BenchmarkQuery(
        query="connections between peace organizations",
        category="relationship",
        expected_keywords=["peace", "organization", "connection", "member"],
        expected_files=["american-friends", "clergy-laity"],
        difficulty="hard",
    ),
    BenchmarkQuery(
        query="government response to Vietnam protests",
        category="event",
        expected_keywords=["vietnam", "protest", "response", "action"],
        expected_files=["clergy-laity", "kent-state"],
        difficulty="hard",
    ),
    BenchmarkQuery(
        query="civil liberties violations by FBI",
        category="topic",
        expected_keywords=["rights", "violation", "surveillance", "illegal"],
        expected_files=[],
        difficulty="hard",
    ),
]


def evaluate_result(result: Dict[str, Any], query: BenchmarkQuery) -> Tuple[float, Dict[str, Any]]:
    """
    Evaluate a single search result against expected criteria.

    Returns: (relevance_score, evaluation_details)
    """
    text = result.get("text", "").lower()
    source = result.get("source_file", "").lower()

    # Keyword matching (0.0 to 1.0)
    keyword_hits = sum(1 for kw in query.expected_keywords if kw.lower() in text)
    keyword_score = keyword_hits / len(query.expected_keywords) if query.expected_keywords else 0.5

    # File matching (0.0 to 1.0)
    file_match = any(exp.lower() in source for exp in query.expected_files) if query.expected_files else 0.5
    file_score = 1.0 if file_match else 0.0

    # Combined score (weighted)
    if query.expected_files:
        relevance = 0.6 * keyword_score + 0.4 * file_score
    else:
        relevance = keyword_score

    details = {
        "keyword_hits": keyword_hits,
        "keyword_total": len(query.expected_keywords),
        "keyword_score": keyword_score,
        "file_match": file_match,
        "file_score": file_score,
        "relevance": relevance,
    }

    return relevance, details


def compute_mrr(results: List[Dict[str, Any]], query: BenchmarkQuery) -> float:
    """Compute Mean Reciprocal Rank for a query."""
    for rank, result in enumerate(results, 1):
        relevance, _ = evaluate_result(result, query)
        if relevance >= 0.5:  # Consider relevant
            return 1.0 / rank
    return 0.0


def compute_precision_at_k(results: List[Dict[str, Any]], query: BenchmarkQuery, k: int = 3) -> float:
    """Compute Precision@K."""
    if not results:
        return 0.0

    relevant_count = 0
    for result in results[:k]:
        relevance, _ = evaluate_result(result, query)
        if relevance >= 0.5:
            relevant_count += 1

    return relevant_count / k


def run_pipeline_benchmark(
    pipeline,
    pipeline_name: str,
    queries: List[BenchmarkQuery],
    top_k: int = 5,
) -> Dict[str, Any]:
    """Run benchmark on a single pipeline."""
    results = {
        "pipeline": pipeline_name,
        "total_queries": len(queries),
        "queries": [],
        "aggregate": {
            "mrr": 0.0,
            "precision_at_3": 0.0,
            "precision_at_5": 0.0,
            "avg_latency_ms": 0.0,
            "by_difficulty": {"easy": [], "medium": [], "hard": []},
        },
    }

    total_mrr = 0.0
    total_p3 = 0.0
    total_p5 = 0.0
    total_latency = 0.0

    for query in queries:
        # Time the search
        start_time = time.time()
        search_results = pipeline.search(query.query, top_k=top_k)
        latency_ms = (time.time() - start_time) * 1000

        # Compute metrics
        mrr = compute_mrr(search_results, query)
        p3 = compute_precision_at_k(search_results, query, k=3)
        p5 = compute_precision_at_k(search_results, query, k=5)

        query_result = {
            "query": query.query,
            "category": query.category,
            "difficulty": query.difficulty,
            "mrr": mrr,
            "precision_at_3": p3,
            "precision_at_5": p5,
            "latency_ms": latency_ms,
            "num_results": len(search_results),
            "top_result": search_results[0] if search_results else None,
        }
        results["queries"].append(query_result)
        results["aggregate"]["by_difficulty"][query.difficulty].append(mrr)

        total_mrr += mrr
        total_p3 += p3
        total_p5 += p5
        total_latency += latency_ms

    # Compute aggregates
    n = len(queries)
    results["aggregate"]["mrr"] = total_mrr / n
    results["aggregate"]["precision_at_3"] = total_p3 / n
    results["aggregate"]["precision_at_5"] = total_p5 / n
    results["aggregate"]["avg_latency_ms"] = total_latency / n

    # Aggregate by difficulty
    for diff in ["easy", "medium", "hard"]:
        scores = results["aggregate"]["by_difficulty"][diff]
        results["aggregate"]["by_difficulty"][diff] = sum(scores) / len(scores) if scores else 0.0

    return results


def print_results_summary(baseline_results: Dict, sota_results: Dict):
    """Print a comparison summary."""
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 80)

    print(f"\n{'Metric':<25} {'GPT Baseline':<20} {'SOTA Pipeline':<20} {'Delta':<15}")
    print("-" * 80)

    metrics = [
        ("MRR", "mrr"),
        ("Precision@3", "precision_at_3"),
        ("Precision@5", "precision_at_5"),
        ("Avg Latency (ms)", "avg_latency_ms"),
    ]

    for name, key in metrics:
        baseline_val = baseline_results["aggregate"][key]
        sota_val = sota_results["aggregate"][key]
        delta = sota_val - baseline_val
        delta_pct = (delta / baseline_val * 100) if baseline_val > 0 else 0

        if key == "avg_latency_ms":
            print(f"{name:<25} {baseline_val:>15.1f} ms  {sota_val:>15.1f} ms  {delta:>+.1f} ms")
        else:
            sign = "+" if delta >= 0 else ""
            print(f"{name:<25} {baseline_val:>15.4f}     {sota_val:>15.4f}     {sign}{delta:.4f} ({sign}{delta_pct:.1f}%)")

    print("\n" + "-" * 80)
    print("Performance by Difficulty:")
    print("-" * 80)

    for diff in ["easy", "medium", "hard"]:
        baseline_mrr = baseline_results["aggregate"]["by_difficulty"][diff]
        sota_mrr = sota_results["aggregate"]["by_difficulty"][diff]
        delta = sota_mrr - baseline_mrr
        print(f"  {diff.capitalize():<10} GPT: {baseline_mrr:.4f}  SOTA: {sota_mrr:.4f}  Delta: {delta:+.4f}")


def main():
    """Run the full benchmark."""
    print("=" * 80)
    print("FBI Anti-War Movement Document Retrieval Benchmark")
    print("=" * 80)

    # Configuration
    data_dir = Path(__file__).parent.parent / "data"
    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    pdf_files = list(data_dir.glob("*.pdf"))

    if not pdf_files:
        print("No PDFs found in data directory")
        return

    # Sample PDFs for benchmark (use 10 to keep costs reasonable)
    random.seed(42)  # Reproducibility
    sample_size = min(10, len(pdf_files))
    sample_pdfs = random.sample(pdf_files, sample_size)

    print(f"\nBenchmark configuration:")
    print(f"  - Total PDFs available: {len(pdf_files)}")
    print(f"  - PDFs sampled: {sample_size}")
    print(f"  - Test queries: {len(BENCHMARK_QUERIES)}")
    print(f"\nSampled files:")
    for p in sample_pdfs:
        print(f"    - {p.name}")

    # Initialize pipelines
    print("\n" + "-" * 80)
    print("PHASE 1: GPT Baseline Pipeline")
    print("-" * 80)

    baseline = GPTBaselinePipeline()
    baseline_index_start = time.time()
    baseline_stats = baseline.index_pdfs(sample_pdfs)
    baseline_index_time = time.time() - baseline_index_start

    print(f"\nBaseline indexing completed in {baseline_index_time:.1f}s")
    print(f"  - Pages: {baseline_stats['total_pages']}")
    print(f"  - Chunks: {baseline_stats['total_chunks']}")

    # Run baseline benchmark
    print("\nRunning baseline benchmark queries...")
    baseline_results = run_pipeline_benchmark(baseline, "GPT Baseline", BENCHMARK_QUERIES)

    # SOTA Pipeline
    print("\n" + "-" * 80)
    print("PHASE 2: SOTA Multi-Modal Pipeline")
    print("-" * 80)

    # Fast mode for benchmarking (skip VLM), accurate bboxes still used
    sota = SOTAPipeline(fast_mode=True)
    sota_index_start = time.time()
    sota_stats = sota.index_pdfs(sample_pdfs)
    sota_index_time = time.time() - sota_index_start

    print(f"\nSOTA indexing completed in {sota_index_time:.1f}s")
    print(f"  - Pages: {sota_stats['total_pages']}")
    print(f"  - Objects: {sota_stats['total_objects']}")
    print(f"  - By type: {json.dumps(sota_stats.get('objects_by_type', {}))}")
    print(f"  - Confidence distribution: {json.dumps(sota_stats.get('confidence_distribution', {}))}")

    # Run SOTA benchmark
    print("\nRunning SOTA benchmark queries...")
    sota_results = run_pipeline_benchmark(sota, "SOTA Pipeline", BENCHMARK_QUERIES)

    # Print comparison
    print_results_summary(baseline_results, sota_results)

    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"benchmark_results_{timestamp}.json"

    full_results = {
        "timestamp": timestamp,
        "config": {
            "sample_size": sample_size,
            "num_queries": len(BENCHMARK_QUERIES),
            "sample_files": [p.name for p in sample_pdfs],
        },
        "indexing": {
            "baseline": {
                "time_seconds": baseline_index_time,
                "stats": baseline_stats,
            },
            "sota": {
                "time_seconds": sota_index_time,
                "stats": sota_stats,
            },
        },
        "baseline": baseline_results,
        "sota": sota_results,
    }

    with open(results_file, "w") as f:
        json.dump(full_results, f, indent=2, default=str)

    print(f"\n\nDetailed results saved to: {results_file}")

    # Final verdict
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)

    mrr_improvement = (sota_results["aggregate"]["mrr"] - baseline_results["aggregate"]["mrr"]) / baseline_results["aggregate"]["mrr"] * 100 if baseline_results["aggregate"]["mrr"] > 0 else 0

    if mrr_improvement > 10:
        print(f"\n  SOTA Pipeline outperforms GPT Baseline by {mrr_improvement:.1f}%")
    elif mrr_improvement > 0:
        print(f"\n  SOTA Pipeline marginally better (+{mrr_improvement:.1f}%)")
    else:
        print(f"\n  GPT Baseline performs comparably or better ({mrr_improvement:.1f}%)")

    print("\n  Key factors:")
    if sota_stats.get("confidence_distribution", {}).get("A", 0) > sota_stats.get("confidence_distribution", {}).get("D", 0):
        print("    - High confidence extractions (more A than D ratings)")
    print(f"    - Object-level indexing ({sota_stats.get('total_objects', 0)} objects vs {baseline_stats.get('total_chunks', 0)} chunks)")

    hard_improvement = sota_results["aggregate"]["by_difficulty"]["hard"] - baseline_results["aggregate"]["by_difficulty"]["hard"]
    if hard_improvement > 0.1:
        print(f"    - Significantly better on hard queries (+{hard_improvement:.2f} MRR)")


if __name__ == "__main__":
    main()
