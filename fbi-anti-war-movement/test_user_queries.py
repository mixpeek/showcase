#!/usr/bin/env python3
"""
Test with realistic user queries - the kind of questions someone would actually ask.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "pipeline" / "baseline"))
sys.path.insert(0, str(Path(__file__).parent / "pipeline" / "advanced"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from gpt_baseline import GPTBaselinePipeline
from sota_pipeline import SOTAPipeline
import random

# Realistic user queries - natural language questions
USER_QUERIES = [
    # Conversational questions
    "What did the FBI know about Abbie Hoffman?",
    "Why was Howard Zinn being watched?",
    "What happened at Kent State according to the FBI?",
    "Did the FBI spy on anti-war protesters?",
    "Who were the informants in the peace movement?",

    # Specific lookups
    "Show me surveillance reports from 1968",
    "Find mentions of the Chicago Seven",
    "What organizations were considered subversive?",

    # Broad research questions
    "How did the FBI monitor student activists?",
    "What was COINTELPRO's involvement in peace movements?",
    "Evidence of illegal wiretapping",
]

def print_result(result, idx, show_scores=False):
    """Pretty print a search result."""
    print(f"\n  [{idx}] Score: {result.get('score', 0):.4f}")
    if show_scores and 'semantic_score' in result:
        sem = result.get('semantic_score', 0)
        kw = result.get('keyword_score', 0)
        qual = result.get('text_quality', 0)
        print(f"      (semantic: {sem:.3f}, keyword: {kw:.3f}, quality: {qual:.2f})")
    print(f"      Source: {result.get('source_file', 'N/A')} (page {result.get('page_number', '?')})")
    if 'object_type' in result:
        print(f"      Type: {result.get('object_type')} | Confidence: {result.get('confidence_tag', 'N/A')}")
    if 'bbox' in result:
        bbox = result['bbox']
        if isinstance(bbox, dict):
            print(f"      BBox: [{bbox['x0']:.0f}, {bbox['y0']:.0f}, {bbox['x1']:.0f}, {bbox['y1']:.0f}]")
        elif isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            print(f"      BBox: [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]")
    text = result.get('text', '')[:300]
    print(f"      Text: {text}...")


def main():
    data_dir = Path(__file__).parent / "data"
    pdf_files = list(data_dir.glob("*.pdf"))

    if not pdf_files:
        print("No PDFs found")
        return

    # Use same sample as benchmark for consistency
    random.seed(42)
    sample_pdfs = random.sample(pdf_files, min(10, len(pdf_files)))

    print("=" * 80)
    print("USER QUERY TEST")
    print("=" * 80)
    print(f"\nIndexing {len(sample_pdfs)} PDFs...")

    # Index with both pipelines
    baseline = GPTBaselinePipeline()
    baseline.index_pdfs(sample_pdfs, show_progress=True)

    sota = SOTAPipeline(fast_mode=True)
    sota.index_pdfs(sample_pdfs, show_progress=True)

    print("\n" + "=" * 80)
    print("RUNNING USER QUERIES")
    print("=" * 80)

    for query in USER_QUERIES:
        print(f"\n{'─' * 80}")
        print(f"QUERY: \"{query}\"")
        print("─" * 80)

        # Baseline results
        print("\n▶ GPT BASELINE (top 3):")
        baseline_results = baseline.search(query, top_k=3)
        for i, r in enumerate(baseline_results, 1):
            print_result(r, i)

        # SOTA results
        print("\n▶ SOTA PIPELINE (top 3):")
        sota_results = sota.search(query, top_k=3)
        for i, r in enumerate(sota_results, 1):
            print_result(r, i, show_scores=True)

        print()


if __name__ == "__main__":
    main()
