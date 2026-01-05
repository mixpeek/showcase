#!/usr/bin/env python3
"""
Hacker News Downloader

Downloads posts (stories) and all comments from Hacker News.
Supports historical downloads via Algolia HN Search API.
Categorizes posts as 'text' (Ask HN, Show HN, text posts) vs 'url' (link posts).

Data Sources:
- Historical: https://hn.algolia.com/api/v1/ (Algolia HN Search API)
- Comments: https://hacker-news.firebaseio.com/v0/ (Firebase API)

License: Hacker News content is user-generated. Use responsibly.
"""

import os
import json
import time
import argparse
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Hacker News Firebase API (for comments)
HN_API_BASE = "https://hacker-news.firebaseio.com/v0"
HN_ITEM_URL = f"{HN_API_BASE}/item/{{item_id}}.json"

# Algolia HN Search API (for historical stories)
ALGOLIA_API_BASE = "https://hn.algolia.com/api/v1"
ALGOLIA_SEARCH_BY_DATE = f"{ALGOLIA_API_BASE}/search_by_date"

# Rate limiting
ALGOLIA_REQUESTS_PER_HOUR = 10000
FIREBASE_DELAY = 0.05  # 50ms between Firebase requests


def get_item(item_id: int, timeout: int = 30) -> Optional[Dict]:
    """Fetch a single item (story or comment) from HN Firebase API."""
    try:
        url = HN_ITEM_URL.format(item_id=item_id)
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.debug(f"Failed to fetch item {item_id}: {e}")
        return None


def categorize_post(item: Dict) -> str:
    """
    Categorize a post as 'text' or 'url'.

    - 'text': Ask HN, Show HN (without URL), or posts with text content only
    - 'url': Posts that link to external URLs
    """
    title = item.get("title", "").lower()
    has_url = bool(item.get("url"))

    if title.startswith("ask hn"):
        return "text"
    if title.startswith("show hn"):
        return "url" if has_url else "text"
    if has_url:
        return "url"
    return "text"


def fetch_comments_recursive(
    item_ids: List[int],
    delay: float = FIREBASE_DELAY,
    max_workers: int = 20
) -> List[Dict]:
    """
    Recursively fetch all comments for given item IDs.
    Returns a flat list of all comments with their hierarchy info.
    """
    if not item_ids:
        return []

    comments = []
    child_ids_to_fetch = []

    # Fetch comments in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(get_item, item_id): item_id for item_id in item_ids}

        for future in as_completed(futures):
            item_id = futures[future]
            try:
                item = future.result()
                if item and item.get("type") == "comment" and not item.get("dead") and not item.get("deleted"):
                    comment_data = {
                        "id": item.get("id"),
                        "by": item.get("by"),
                        "text": item.get("text", ""),
                        "time": item.get("time"),
                        "time_str": datetime.fromtimestamp(item.get("time", 0)).isoformat() if item.get("time") else None,
                        "parent": item.get("parent"),
                        "kids": item.get("kids", []),
                    }
                    comments.append(comment_data)

                    # Collect child IDs for batch fetching
                    if item.get("kids"):
                        child_ids_to_fetch.extend(item["kids"])

            except Exception as e:
                logger.debug(f"Error processing comment {item_id}: {e}")

    # Recursively fetch child comments
    if child_ids_to_fetch:
        time.sleep(delay)
        child_comments = fetch_comments_recursive(child_ids_to_fetch, delay, max_workers)
        comments.extend(child_comments)

    return comments


def build_comment_tree(comments: List[Dict], parent_id: int) -> List[Dict]:
    """Build a nested comment tree structure."""
    tree = []

    for comment in comments:
        if comment["parent"] == parent_id:
            comment_copy = comment.copy()
            comment_copy["replies"] = build_comment_tree(comments, comment["id"])
            tree.append(comment_copy)

    tree.sort(key=lambda x: x.get("time", 0))
    return tree


def search_algolia(
    start_timestamp: int,
    end_timestamp: int,
    page: int = 0,
    hits_per_page: int = 1000,
    tags: str = "story"
) -> Dict:
    """Search Algolia HN API for stories in a time range."""
    params = {
        "tags": tags,
        "numericFilters": f"created_at_i>={start_timestamp},created_at_i<{end_timestamp}",
        "hitsPerPage": hits_per_page,
        "page": page,
    }

    try:
        response = requests.get(ALGOLIA_SEARCH_BY_DATE, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Algolia API error: {e}")
        return {"hits": [], "nbPages": 0, "nbHits": 0}


def fetch_stories_for_month(
    year: int,
    month: int,
    output_dir: Path,
    include_comments: bool = True,
    skip_existing: bool = True,
    delay: float = 0.1,
    max_workers: int = 20,
) -> Dict:
    """Fetch all stories for a specific month."""
    start_date = datetime(year, month, 1)
    if month == 12:
        end_date = datetime(year + 1, 1, 1)
    else:
        end_date = datetime(year, month + 1, 1)

    start_ts = int(start_date.timestamp())
    end_ts = int(end_date.timestamp())

    month_dir = output_dir / f"{year}" / f"{month:02d}"
    month_dir.mkdir(parents=True, exist_ok=True)

    # Check for existing month summary
    month_summary_path = month_dir / "month_summary.json"

    stats = {"downloaded": 0, "skipped": 0, "failed": 0, "text_posts": 0, "url_posts": 0}
    all_stories = []

    # First, get total count
    initial_response = search_algolia(start_ts, end_ts, page=0, hits_per_page=1)
    total_hits = initial_response.get("nbHits", 0)
    total_pages = initial_response.get("nbPages", 0)

    if total_hits == 0:
        logger.info(f"  No stories found for {year}-{month:02d}")
        return stats

    logger.info(f"  Found {total_hits} stories across {total_pages} pages for {year}-{month:02d}")

    # Fetch all pages
    for page in range(total_pages):
        if page > 0 and page % 10 == 0:
            logger.info(f"    Processing page {page}/{total_pages}...")

        response = search_algolia(start_ts, end_ts, page=page, hits_per_page=1000)
        hits = response.get("hits", [])

        for hit in hits:
            story_id = hit.get("objectID")
            if not story_id:
                continue

            try:
                story_id = int(story_id)
            except ValueError:
                continue

            story_path = month_dir / f"story_{story_id}.json"

            if skip_existing and story_path.exists():
                stats["skipped"] += 1
                continue

            # Build story data from Algolia response
            title = hit.get("title", "")
            category = "text" if (
                title.lower().startswith("ask hn") or
                (title.lower().startswith("show hn") and not hit.get("url")) or
                not hit.get("url")
            ) else "url"

            story_data = {
                "id": story_id,
                "title": title,
                "by": hit.get("author", "[deleted]"),
                "score": hit.get("points", 0),
                "time": hit.get("created_at_i"),
                "time_str": hit.get("created_at"),
                "category": category,
                "descendants": hit.get("num_comments", 0),
                "url": hit.get("url"),
                "text": hit.get("story_text"),
                "comments": [],
                "comment_count": 0,
            }

            # Fetch comments if requested
            if include_comments and hit.get("num_comments", 0) > 0:
                # Get full item from Firebase to get comment IDs
                full_item = get_item(story_id)
                if full_item and full_item.get("kids"):
                    all_comments = fetch_comments_recursive(full_item["kids"], FIREBASE_DELAY, max_workers)
                    story_data["comments"] = build_comment_tree(all_comments, story_id)
                    story_data["comment_count"] = len(all_comments)
                time.sleep(delay)

            # Save story
            try:
                with open(story_path, 'w', encoding='utf-8') as f:
                    json.dump(story_data, f, indent=2, ensure_ascii=False)

                stats["downloaded"] += 1
                if category == "text":
                    stats["text_posts"] += 1
                else:
                    stats["url_posts"] += 1

                all_stories.append({
                    "id": story_id,
                    "title": title,
                    "category": category,
                    "score": story_data["score"],
                    "comment_count": story_data["comment_count"],
                })

            except Exception as e:
                logger.warning(f"Failed to save story {story_id}: {e}")
                stats["failed"] += 1

        time.sleep(delay)  # Rate limiting between pages

    # Save month summary
    summary_data = {
        "year": year,
        "month": month,
        "downloaded_at": datetime.now().isoformat(),
        "total_stories": len(all_stories),
        "text_posts": stats["text_posts"],
        "url_posts": stats["url_posts"],
        "stories": all_stories,
    }

    with open(month_summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    return stats


def download_historical(
    output_dir: Path,
    years: int = 10,
    include_comments: bool = True,
    skip_existing: bool = True,
    delay: float = 0.1,
    max_workers: int = 20,
    start_year: Optional[int] = None,
    start_month: Optional[int] = None,
):
    """Download all stories from the last N years."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now()

    # Calculate start date
    if start_year and start_month:
        start_date = datetime(start_year, start_month, 1)
    else:
        start_date = now - relativedelta(years=years)
        start_date = datetime(start_date.year, start_date.month, 1)

    end_date = datetime(now.year, now.month, 1)

    # Generate list of months to process
    months_to_process = []
    current = start_date
    while current < end_date:
        months_to_process.append((current.year, current.month))
        if current.month == 12:
            current = datetime(current.year + 1, 1, 1)
        else:
            current = datetime(current.year, current.month + 1, 1)

    total_months = len(months_to_process)
    logger.info(f"Processing {total_months} months from {start_date.strftime('%Y-%m')} to {end_date.strftime('%Y-%m')}")

    total_stats = {"downloaded": 0, "skipped": 0, "failed": 0, "text_posts": 0, "url_posts": 0}

    for i, (year, month) in enumerate(months_to_process):
        logger.info(f"\n[{i+1}/{total_months}] Processing {year}-{month:02d}...")

        stats = fetch_stories_for_month(
            year=year,
            month=month,
            output_dir=output_dir,
            include_comments=include_comments,
            skip_existing=skip_existing,
            delay=delay,
            max_workers=max_workers,
        )

        for key in total_stats:
            total_stats[key] += stats[key]

        logger.info(f"  Month complete: {stats['downloaded']} downloaded, {stats['skipped']} skipped")

    logger.info(f"\n=== Historical Download Complete ===")
    logger.info(f"Total: {total_stats['downloaded']} downloaded, {total_stats['skipped']} skipped, {total_stats['failed']} failed")
    logger.info(f"Text posts: {total_stats['text_posts']}, URL posts: {total_stats['url_posts']}")


def download_current(
    output_dir: Path,
    story_type: str = "top",
    limit: Optional[int] = None,
    include_comments: bool = True,
    delay: float = 0.2,
    max_workers: int = 20,
    skip_existing: bool = True,
    list_only: bool = False,
):
    """Download current stories from HN (top, new, best, etc.)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Endpoints for current stories
    endpoints = {
        "top": f"{HN_API_BASE}/topstories.json",
        "new": f"{HN_API_BASE}/newstories.json",
        "best": f"{HN_API_BASE}/beststories.json",
        "ask": f"{HN_API_BASE}/askstories.json",
        "show": f"{HN_API_BASE}/showstories.json",
        "job": f"{HN_API_BASE}/jobstories.json",
    }

    url = endpoints.get(story_type, endpoints["top"])

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        story_ids = response.json()
    except Exception as e:
        logger.error(f"Failed to fetch {story_type} stories: {e}")
        return

    if limit:
        story_ids = story_ids[:limit]

    logger.info(f"Found {len(story_ids)} {story_type} stories")

    stats = {"downloaded": 0, "skipped": 0, "failed": 0, "text_posts": 0, "url_posts": 0}
    all_stories = []

    for i, story_id in enumerate(story_ids):
        story_path = output_dir / f"story_{story_id}.json"

        if skip_existing and story_path.exists():
            logger.info(f"[{i+1}/{len(story_ids)}] Skipping {story_id} (exists)")
            stats["skipped"] += 1
            continue

        item = get_item(story_id)
        if not item:
            stats["failed"] += 1
            continue

        if list_only:
            category = categorize_post(item)
            print(f"[{category.upper()}] {item.get('title', 'No title')} (id: {story_id}, score: {item.get('score', 0)})")
            time.sleep(delay)
            continue

        logger.info(f"[{i+1}/{len(story_ids)}] Downloading story {story_id}...")

        category = categorize_post(item)

        story_data = {
            "id": item.get("id"),
            "title": item.get("title", ""),
            "by": item.get("by", "[deleted]"),
            "score": item.get("score", 0),
            "time": item.get("time"),
            "time_str": datetime.fromtimestamp(item.get("time", 0)).isoformat() if item.get("time") else None,
            "category": category,
            "descendants": item.get("descendants", 0),
            "url": item.get("url"),
            "text": item.get("text"),
            "comments": [],
            "comment_count": 0,
        }

        # Fetch comments if requested
        if include_comments and item.get("kids"):
            logger.info(f"  Fetching {item.get('descendants', 'unknown')} comments...")
            all_comments = fetch_comments_recursive(item["kids"], FIREBASE_DELAY, max_workers)
            story_data["comments"] = build_comment_tree(all_comments, story_id)
            story_data["comment_count"] = len(all_comments)

        # Save story
        with open(story_path, 'w', encoding='utf-8') as f:
            json.dump(story_data, f, indent=2, ensure_ascii=False)

        all_stories.append(story_data)
        stats["downloaded"] += 1

        if category == "text":
            stats["text_posts"] += 1
        else:
            stats["url_posts"] += 1

        logger.info(f"  Saved: {story_path.name} [{category}] ({story_data['comment_count']} comments)")
        time.sleep(delay)

    if not list_only and all_stories:
        combined_path = output_dir / f"{story_type}_stories.json"
        combined_data = {
            "type": story_type,
            "downloaded_at": datetime.now().isoformat(),
            "total_stories": len(all_stories),
            "text_posts": stats["text_posts"],
            "url_posts": stats["url_posts"],
            "stories": all_stories,
        }

        with open(combined_path, 'w', encoding='utf-8') as f:
            json.dump(combined_data, f, indent=2, ensure_ascii=False)

        logger.info(f"\nSaved combined file: {combined_path}")

    logger.info(f"\n=== Download Complete ===")
    logger.info(f"Downloaded: {stats['downloaded']}, Skipped: {stats['skipped']}, Failed: {stats['failed']}")
    logger.info(f"Text posts: {stats['text_posts']}, URL posts: {stats['url_posts']}")


def main():
    parser = argparse.ArgumentParser(
        description="Download Hacker News stories and comments"
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Current stories command
    current_parser = subparsers.add_parser("current", help="Download current stories (top, new, best, etc.)")
    current_parser.add_argument(
        '--type',
        choices=['top', 'new', 'best', 'ask', 'show', 'job'],
        default='top',
        help='Type of stories to download (default: top)'
    )
    current_parser.add_argument(
        '-n', '--limit',
        type=int,
        default=30,
        help='Number of stories to download (default: 30)'
    )
    current_parser.add_argument(
        '--no-comments',
        action='store_true',
        help='Skip downloading comments'
    )
    current_parser.add_argument(
        '-o', '--output',
        type=Path,
        default=Path('./data/current'),
        help='Output directory (default: ./data/current)'
    )
    current_parser.add_argument(
        '-d', '--delay',
        type=float,
        default=0.2,
        help='Delay between API requests in seconds (default: 0.2)'
    )
    current_parser.add_argument(
        '-w', '--workers',
        type=int,
        default=20,
        help='Max parallel workers for fetching comments (default: 20)'
    )
    current_parser.add_argument(
        '--no-skip',
        action='store_true',
        help='Re-download existing files'
    )
    current_parser.add_argument(
        '--list-only',
        action='store_true',
        help='Only list stories, do not download'
    )

    # Historical download command
    historical_parser = subparsers.add_parser("historical", help="Download historical stories (last N years)")
    historical_parser.add_argument(
        '--years',
        type=int,
        default=10,
        help='Number of years to download (default: 10)'
    )
    historical_parser.add_argument(
        '--start-year',
        type=int,
        help='Start from specific year (overrides --years)'
    )
    historical_parser.add_argument(
        '--start-month',
        type=int,
        choices=range(1, 13),
        help='Start from specific month (requires --start-year)'
    )
    historical_parser.add_argument(
        '--no-comments',
        action='store_true',
        help='Skip downloading comments (much faster)'
    )
    historical_parser.add_argument(
        '-o', '--output',
        type=Path,
        default=Path('./data/historical'),
        help='Output directory (default: ./data/historical)'
    )
    historical_parser.add_argument(
        '-d', '--delay',
        type=float,
        default=0.1,
        help='Delay between API requests in seconds (default: 0.1)'
    )
    historical_parser.add_argument(
        '-w', '--workers',
        type=int,
        default=20,
        help='Max parallel workers for fetching comments (default: 20)'
    )
    historical_parser.add_argument(
        '--no-skip',
        action='store_true',
        help='Re-download existing files'
    )

    args = parser.parse_args()

    # Default to current if no command specified (backwards compatibility)
    if args.command is None:
        parser.print_help()
        print("\nExamples:")
        print("  python download_hn.py current --type top -n 50")
        print("  python download_hn.py historical --years 10")
        print("  python download_hn.py historical --years 10 --no-comments")
        return

    if args.command == "current":
        download_current(
            output_dir=args.output,
            story_type=args.type,
            limit=args.limit,
            include_comments=not args.no_comments,
            delay=args.delay,
            max_workers=args.workers,
            skip_existing=not args.no_skip,
            list_only=args.list_only,
        )
    elif args.command == "historical":
        download_historical(
            output_dir=args.output,
            years=args.years,
            include_comments=not args.no_comments,
            skip_existing=not args.no_skip,
            delay=args.delay,
            max_workers=args.workers,
            start_year=args.start_year,
            start_month=args.start_month,
        )


if __name__ == '__main__':
    main()
