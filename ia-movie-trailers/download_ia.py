#!/usr/bin/env python3
"""
Internet Archive Movie Trailers + Posters Downloader

Downloads movie trailers and posters from Internet Archive collections.
Uses the internetarchive Python library for API access.

Data Sources:
- Movie Trailers: https://archive.org/details/movie_trailers
- Movie Posters: https://archive.org/details/movie-posters_202403
- Harry Ransom Center Posters: https://archive.org/details/HRC_Posters

License: Public Domain / Various open licenses
"""

import os
import json
import time
import argparse
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, List, Any
import requests

try:
    import internetarchive as ia
except ImportError:
    print("Error: internetarchive library not installed.")
    print("Install with: pip install internetarchive")
    exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default collections
TRAILER_COLLECTIONS = [
    'movie_trailers',
    'movietrailers',
]

# Poster items - these are individual items with many poster files inside
# (not collections containing other items)
POSTER_ITEMS = [
    'movie-posters_202403',
    'HRC_Posters',
    'Horror-Movie-Posters',
    'illustration_Vintage_Movie_Posters',
]

# File extensions to download
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mkv', '.mov', '.ogv', '.webm', '.mpeg', '.mpg'}
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.tiff', '.tif'}
THUMBNAIL_PATTERNS = ['__ia_thumb', 'thumb', 'poster', 'cover']


def sanitize_filename(name: str, max_length: int = 100) -> str:
    """Sanitize a string for use as a filename."""
    safe = "".join(c if c.isalnum() or c in ' -_.' else '_' for c in name)
    return safe[:max_length].strip('._')


def get_download_url(identifier: str, filename: str) -> str:
    """Construct download URL for an Internet Archive file."""
    return f"https://archive.org/download/{identifier}/{filename}"


def download_file(url: str, output_path: Path, timeout: int = 60) -> bool:
    """Download a file from a URL."""
    try:
        response = requests.get(url, timeout=timeout, stream=True)
        response.raise_for_status()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        logger.warning(f"Failed to download {url}: {e}")
        return False


def get_best_video_file(files: List[Dict]) -> Optional[Dict]:
    """Select the best video file from available files."""
    video_files = []
    for f in files:
        name = f.get('name', '').lower()
        ext = Path(name).suffix.lower()
        if ext in VIDEO_EXTENSIONS:
            # Prefer mp4 and larger files
            size = int(f.get('size', 0))
            priority = 0
            if ext == '.mp4':
                priority = 3
            elif ext == '.webm':
                priority = 2
            elif ext in {'.avi', '.mkv', '.mov'}:
                priority = 1
            video_files.append((priority, size, f))

    if not video_files:
        return None

    # Sort by priority descending, then size descending
    video_files.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return video_files[0][2]


def get_thumbnail_file(files: List[Dict]) -> Optional[Dict]:
    """Find thumbnail/poster image from item files."""
    for f in files:
        name = f.get('name', '').lower()
        ext = Path(name).suffix.lower()
        if ext in IMAGE_EXTENSIONS:
            # Check if it's a thumbnail
            for pattern in THUMBNAIL_PATTERNS:
                if pattern in name:
                    return f

    # Fallback: return first image file
    for f in files:
        name = f.get('name', '').lower()
        ext = Path(name).suffix.lower()
        if ext in IMAGE_EXTENSIONS:
            return f

    return None


def get_poster_files(files: List[Dict]) -> List[Dict]:
    """Get all poster/image files from an item."""
    poster_files = []
    for f in files:
        name = f.get('name', '').lower()
        ext = Path(name).suffix.lower()
        if ext in IMAGE_EXTENSIONS:
            # Skip small thumbnails and derivative files
            size = int(f.get('size', 0))
            if size > 10000:  # Skip files < 10KB
                poster_files.append(f)
    return poster_files


def search_collection(collection: str, mediatype: str = None, limit: int = None) -> List[str]:
    """Search for items in a collection."""
    query = f'collection:{collection}'
    if mediatype:
        query += f' AND mediatype:{mediatype}'

    logger.info(f"Searching: {query}")

    identifiers = []
    try:
        search = ia.search_items(query)
        for result in search:
            identifiers.append(result['identifier'])
            if limit and len(identifiers) >= limit:
                break
    except Exception as e:
        logger.error(f"Search failed: {e}")

    logger.info(f"Found {len(identifiers)} items in {collection}")
    return identifiers


def download_trailers(
    output_dir: Path,
    collections: List[str] = None,
    max_workers: int = 4,
    delay: float = 0.5,
    limit: int = None,
    skip_existing: bool = True,
    video_only: bool = False,
):
    """Download movie trailers from Internet Archive."""
    collections = collections or TRAILER_COLLECTIONS
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Gather all items from collections
    all_identifiers = []
    for collection in collections:
        identifiers = search_collection(collection, mediatype='movies', limit=limit)
        all_identifiers.extend(identifiers)
        if limit and len(all_identifiers) >= limit:
            all_identifiers = all_identifiers[:limit]
            break

    logger.info(f"Processing {len(all_identifiers)} trailer items...")

    downloaded = 0
    skipped = 0
    failed = 0

    def process_item(identifier: str) -> tuple:
        try:
            item = ia.get_item(identifier)
            metadata = item.item_metadata.get('metadata', {})
            files = item.item_metadata.get('files', [])

            title = metadata.get('title', identifier)
            safe_title = sanitize_filename(title)
            base_name = f"{identifier}_{safe_title}"

            # Check if already downloaded
            json_path = output_dir / f"{base_name}.json"
            if skip_existing and json_path.exists():
                return ('skipped', identifier)

            time.sleep(delay)

            # Find best video file
            video_file = get_best_video_file(files)
            video_path = None
            if video_file:
                video_name = video_file['name']
                video_ext = Path(video_name).suffix
                video_path = output_dir / f"{base_name}{video_ext}"

                if not video_path.exists():
                    url = get_download_url(identifier, video_name)
                    if not download_file(url, video_path):
                        video_path = None

            # Find and download thumbnail
            thumb_path = None
            if not video_only:
                thumb_file = get_thumbnail_file(files)
                if thumb_file:
                    thumb_name = thumb_file['name']
                    thumb_ext = Path(thumb_name).suffix
                    thumb_path = output_dir / f"{base_name}_thumb{thumb_ext}"

                    if not thumb_path.exists():
                        url = get_download_url(identifier, thumb_name)
                        if not download_file(url, thumb_path):
                            thumb_path = None

            # Save metadata
            item_metadata = {
                'identifier': identifier,
                'title': metadata.get('title'),
                'description': metadata.get('description'),
                'creator': metadata.get('creator'),
                'date': metadata.get('date'),
                'year': metadata.get('year'),
                'subject': metadata.get('subject'),
                'collection': metadata.get('collection'),
                'mediatype': metadata.get('mediatype'),
                'runtime': metadata.get('runtime'),
                'licenseurl': metadata.get('licenseurl'),
                'video_file': video_path.name if video_path and video_path.exists() else None,
                'thumbnail_file': thumb_path.name if thumb_path and thumb_path.exists() else None,
                'source': {
                    'archive': 'Internet Archive',
                    'url': f"https://archive.org/details/{identifier}",
                    'download_url': f"https://archive.org/download/{identifier}",
                },
                'files': [
                    {'name': f['name'], 'size': f.get('size'), 'format': f.get('format')}
                    for f in files
                ]
            }

            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(item_metadata, f, indent=2, ensure_ascii=False)

            return ('success', identifier)

        except Exception as e:
            logger.warning(f"Failed to process {identifier}: {e}")
            return ('failed', identifier)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_item, id): id for id in all_identifiers}

        for future in as_completed(futures):
            status, identifier = future.result()
            if status == 'success':
                downloaded += 1
            elif status == 'skipped':
                skipped += 1
            else:
                failed += 1

            total = downloaded + skipped + failed
            if total % 10 == 0:
                logger.info(f"Trailers: {total}/{len(all_identifiers)} (downloaded: {downloaded}, skipped: {skipped}, failed: {failed})")

    logger.info(f"Trailers complete! Downloaded: {downloaded}, Skipped: {skipped}, Failed: {failed}")


def download_posters(
    output_dir: Path,
    item_ids: List[str] = None,
    max_workers: int = 4,
    delay: float = 0.3,
    limit: int = None,
    skip_existing: bool = True,
):
    """Download movie posters from Internet Archive items.

    Note: Poster items are single IA items containing many poster files,
    not collections of separate items.
    """
    item_ids = item_ids or POSTER_ITEMS
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Processing {len(item_ids)} poster items...")

    total_downloaded = 0
    total_skipped = 0
    total_failed = 0

    for identifier in item_ids:
        try:
            logger.info(f"Fetching item: {identifier}")
            item = ia.get_item(identifier)
            metadata = item.item_metadata.get('metadata', {})
            files = item.item_metadata.get('files', [])

            # Get all poster files from this item
            poster_files = get_poster_files(files)
            logger.info(f"Found {len(poster_files)} poster files in {identifier}")

            if not poster_files:
                logger.warning(f"No poster files found in {identifier}")
                total_failed += 1
                continue

            # Apply limit per item
            if limit:
                poster_files = poster_files[:limit]

            downloaded = 0
            skipped = 0

            for idx, poster_file in enumerate(poster_files):
                poster_name = poster_file['name']
                poster_ext = Path(poster_name).suffix

                # Create safe filename from original name
                safe_name = sanitize_filename(Path(poster_name).stem)
                base_name = f"{identifier}_{safe_name}"

                poster_path = output_dir / f"{base_name}{poster_ext}"
                json_path = output_dir / f"{base_name}.json"

                # Check if already downloaded
                if skip_existing and poster_path.exists() and json_path.exists():
                    skipped += 1
                    continue

                time.sleep(delay)

                # Download poster
                if not poster_path.exists():
                    url = get_download_url(identifier, poster_name)
                    if not download_file(url, poster_path):
                        continue

                # Save metadata for this poster
                poster_metadata = {
                    'identifier': identifier,
                    'item_title': metadata.get('title'),
                    'description': metadata.get('description'),
                    'creator': metadata.get('creator'),
                    'date': metadata.get('date'),
                    'year': metadata.get('year'),
                    'subject': metadata.get('subject'),
                    'collection': metadata.get('collection'),
                    'poster_file': poster_path.name,
                    'original_filename': poster_name,
                    'file_size': poster_file.get('size'),
                    'source': {
                        'archive': 'Internet Archive',
                        'url': f"https://archive.org/details/{identifier}",
                        'download_url': f"https://archive.org/download/{identifier}/{poster_name}",
                    },
                }

                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(poster_metadata, f, indent=2, ensure_ascii=False)

                downloaded += 1

                if (downloaded + skipped) % 10 == 0:
                    logger.info(f"  {identifier}: {downloaded + skipped}/{len(poster_files)} (downloaded: {downloaded}, skipped: {skipped})")

            total_downloaded += downloaded
            total_skipped += skipped
            logger.info(f"Completed {identifier}: downloaded {downloaded}, skipped {skipped}")

        except Exception as e:
            logger.warning(f"Failed to process {identifier}: {e}")
            total_failed += 1

    logger.info(f"Posters complete! Downloaded: {total_downloaded}, Skipped: {total_skipped}, Failed items: {total_failed}")


def main():
    parser = argparse.ArgumentParser(
        description="Download movie trailers and posters from Internet Archive"
    )
    parser.add_argument(
        '--trailers',
        action='store_true',
        help='Download movie trailers'
    )
    parser.add_argument(
        '--posters',
        action='store_true',
        help='Download movie posters'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Download both trailers and posters'
    )
    parser.add_argument(
        '-o', '--output',
        type=Path,
        default=Path('./data'),
        help='Base output directory (default: ./data)'
    )
    parser.add_argument(
        '-w', '--workers',
        type=int,
        default=4,
        help='Number of parallel download workers'
    )
    parser.add_argument(
        '-d', '--delay',
        type=float,
        default=0.5,
        help='Delay between API requests (seconds)'
    )
    parser.add_argument(
        '-l', '--limit',
        type=int,
        default=None,
        help='Limit number of items to download per category'
    )
    parser.add_argument(
        '--collection',
        type=str,
        action='append',
        help='Specific collection(s) to download from'
    )
    parser.add_argument(
        '--no-skip',
        action='store_true',
        help='Re-download existing files'
    )
    parser.add_argument(
        '--list-only',
        action='store_true',
        help='Only list items, do not download'
    )

    args = parser.parse_args()

    # Default to downloading both if neither specified
    if not args.trailers and not args.posters and not args.all:
        args.all = True

    if args.all:
        args.trailers = True
        args.posters = True

    skip_existing = not args.no_skip

    if args.list_only:
        # Just list items
        if args.trailers:
            collections = args.collection or TRAILER_COLLECTIONS
            for collection in collections:
                identifiers = search_collection(collection, mediatype='movies', limit=args.limit)
                for id in identifiers:
                    print(f"[trailer] {id}")

        if args.posters:
            item_ids = args.collection or POSTER_ITEMS
            for item_id in item_ids:
                print(f"[poster item] {item_id}")
        return

    # Download trailers
    if args.trailers:
        trailer_dir = args.output / 'videos'
        collections = args.collection if args.collection else None
        download_trailers(
            output_dir=trailer_dir,
            collections=collections,
            max_workers=args.workers,
            delay=args.delay,
            limit=args.limit,
            skip_existing=skip_existing,
        )

    # Download posters
    if args.posters:
        poster_dir = args.output / 'posters'
        item_ids = args.collection if args.collection else None
        download_posters(
            output_dir=poster_dir,
            item_ids=item_ids,
            max_workers=args.workers,
            delay=args.delay,
            limit=args.limit,
            skip_existing=skip_existing,
        )


if __name__ == '__main__':
    main()
