#!/usr/bin/env python3
"""
NYPL Public Domain Photo Archive Downloader

Downloads public domain images from the New York Public Library's Digital Collections.
Uses NYPL's public domain data release on GitHub for metadata and image URLs.

Data Source: https://github.com/NYPL-publicdomain/data-and-utilities
License: CC0 (Public Domain)
"""

import os
import csv
import json
import time
import requests
import argparse
import gzip
import io
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# NYPL Public Domain Data URLs (GitHub)
GITHUB_BASE = "https://raw.githubusercontent.com/NYPL-publicdomain/data-and-utilities/master"

# NDJSON files containing all items (split into 4 parts, ~187k total items)
ITEMS_NDJSON_URLS = [
    f"{GITHUB_BASE}/items/pd_items_1.ndjson",
    f"{GITHUB_BASE}/items/pd_items_2.ndjson",
    f"{GITHUB_BASE}/items/pd_items_3.ndjson",
    f"{GITHUB_BASE}/items/pd_items_4.ndjson",
]

# CSV files (alternative, less complete metadata)
ITEMS_CSV_URLS = [
    f"{GITHUB_BASE}/items/pd_items_1.csv",
    f"{GITHUB_BASE}/items/pd_items_2.csv",
]

# NYPL Digital Collections image base URLs
IMAGE_BASE = "https://images.nypl.org/index.php"


def download_file(url: str, cache_path: Path, compressed: bool = False) -> Path:
    """Download a file with caching support."""
    if cache_path.exists():
        logger.info(f"Using cached {cache_path.name}")
        return cache_path

    logger.info(f"Downloading {url}...")
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if compressed:
        # Decompress gzip content
        content = gzip.decompress(response.content)
        with open(cache_path, 'wb') as f:
            f.write(content)
    else:
        with open(cache_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

    logger.info(f"Downloaded {cache_path.name}")
    return cache_path


def load_items_csv(csv_path: Path) -> list:
    """Load items from the NYPL public domain CSV file."""
    items = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            items.append(row)
    logger.info(f"Loaded {len(items)} items from CSV")
    return items




def download_items_json(json_url: str, cache_dir: Path) -> list:
    """Download and parse a JSON file containing item metadata."""
    filename = json_url.split('/')[-1]
    cache_path = cache_dir / filename

    if cache_path.exists():
        with open(cache_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    try:
        response = requests.get(json_url, timeout=60)
        response.raise_for_status()
        data = response.json()

        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(data, f)

        return data
    except Exception as e:
        logger.warning(f"Failed to download {json_url}: {e}")
        return []


def construct_image_url(image_id: str, size: str = "w") -> str:
    """
    Construct NYPL image URL.

    Size options:
    - "t": thumbnail (~150px)
    - "r": reduced (~300px)
    - "w": web/medium (~760px)
    - "q": large (~1600px)
    - "v": very large
    - "g": giant (full resolution)
    - "b": original bitonal
    """
    size_map = {
        "thumb": "t",
        "small": "r",
        "medium": "w",
        "large": "q",
        "xlarge": "v",
        "high": "g"
    }
    size_code = size_map.get(size, size)
    return f"{IMAGE_BASE}?id={image_id}&t={size_code}"


def extract_image_info(item: dict) -> dict:
    """Extract image information from an item record."""
    # The item structure varies, try different paths
    captures = item.get('captures', []) or item.get('capture', [])
    if isinstance(captures, dict):
        captures = [captures]

    image_links = []
    for capture in captures:
        if isinstance(capture, dict):
            # Try to get image ID from various fields
            image_id = capture.get('imageID') or capture.get('image_id') or capture.get('imageId')
            if image_id:
                image_links.append({
                    'image_id': image_id,
                    'uuid': capture.get('uuid'),
                    'title': capture.get('title'),
                    'sort_string': capture.get('sortString')
                })

    # Also check for direct imageLinks
    if 'imageLinks' in item:
        links = item['imageLinks']
        if isinstance(links, dict):
            for key, url in links.items():
                if url and 'id=' in url:
                    # Extract image ID from URL
                    image_id = url.split('id=')[1].split('&')[0]
                    image_links.append({
                        'image_id': image_id,
                        'link_type': key,
                        'url': url
                    })

    return image_links


def load_ndjson_file(file_path: Path) -> list:
    """Load items from an NDJSON (newline-delimited JSON) file."""
    items = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.debug(f"Failed to parse line: {e}")
    return items


def load_all_items(cache_dir: Path, limit: int = None) -> list:
    """Load all items from NYPL public domain NDJSON files."""
    all_items = []

    for url in ITEMS_NDJSON_URLS:
        filename = url.split('/')[-1]
        cache_path = cache_dir / filename

        if not cache_path.exists():
            logger.info(f"Downloading {filename}...")
            try:
                download_file(url, cache_path)
            except Exception as e:
                logger.warning(f"Failed to download {filename}: {e}")
                continue

        logger.info(f"Loading {filename}...")
        items = load_ndjson_file(cache_path)
        all_items.extend(items)
        logger.info(f"Loaded {len(items)} items from {filename}")

        # Early exit if we have enough items
        if limit and len(all_items) >= limit:
            all_items = all_items[:limit]
            break

    logger.info(f"Total items loaded: {len(all_items)}")

    if limit:
        all_items = all_items[:limit]

    return all_items


def download_item_details(uuid: str, cache_dir: Path) -> dict:
    """Download detailed item info from NYPL API (requires no auth for basic info)."""
    cache_path = cache_dir / f"{uuid}.json"

    if cache_path.exists():
        with open(cache_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    # Use the public NYPL Digital Collections endpoint
    url = f"https://api.repo.nypl.org/api/v2/items/{uuid}?publicDomainOnly=true"

    try:
        response = requests.get(url, timeout=30, headers={
            'Accept': 'application/json'
        })
        if response.status_code == 200:
            data = response.json()
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f)
            return data
    except Exception as e:
        logger.debug(f"Could not fetch details for {uuid}: {e}")

    return {}


def download_image(url: str, output_path: Path, timeout: int = 60) -> bool:
    """Download a single image."""
    try:
        response = requests.get(url, timeout=timeout, stream=True, headers={
            'User-Agent': 'NYPL-Public-Domain-Downloader/1.0'
        })
        response.raise_for_status()

        # Check if we got actual image data
        content_type = response.headers.get('Content-Type', '')
        if 'image' not in content_type and len(response.content) < 1000:
            logger.warning(f"Invalid response for {url}: {content_type}")
            return False

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        logger.warning(f"Failed to download {url}: {e}")
        return False


def save_metadata(metadata: dict, output_path: Path):
    """Save metadata as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def get_nested(d: dict, *keys, default=''):
    """Safely get nested dictionary values."""
    for key in keys:
        if isinstance(d, dict):
            d = d.get(key, {})
        else:
            return default
    return d if d else default


def extract_image_id_from_item(item: dict) -> str:
    """Extract the image ID from an item's captures array."""
    # Captures is an array of image URLs like:
    # ["http://images.nypl.org/index.php?id=434086&t=g"]
    captures = item.get('captures', [])
    if captures and isinstance(captures, list) and len(captures) > 0:
        capture_url = captures[0]
        if isinstance(capture_url, str) and 'id=' in capture_url:
            # Extract ID from URL
            return capture_url.split('id=')[1].split('&')[0]
        elif isinstance(capture_url, dict):
            # Handle dict format if present
            image_id = capture_url.get('imageID') or capture_url.get('imageId')
            if image_id:
                return image_id

    # Try imageLinks as fallback
    image_links = item.get('imageLinks', {})
    if isinstance(image_links, dict):
        for key in ['imageLink', 'thumbnailLink', 'wsThumbnailLink']:
            url = image_links.get(key, '')
            if url and 'id=' in url:
                return url.split('id=')[1].split('&')[0]

    return None


def process_item(item: dict, output_dir: Path, metadata_dir: Path,
                 size: str, cache_dir: Path, delay: float) -> tuple:
    """Process a single item: download image and save metadata."""
    uuid = item.get('UUID') or item.get('uuid') or item.get('id')
    if not uuid:
        return ('failed', None, 'No UUID')

    # Create safe filename from UUID
    safe_uuid = uuid.replace('-', '_')[:50]
    image_path = output_dir / f"{safe_uuid}.jpg"
    metadata_path = metadata_dir / f"{safe_uuid}.json"

    # Skip if already downloaded
    if image_path.exists() and metadata_path.exists():
        return ('skipped', uuid, None)

    time.sleep(delay)

    # Get image ID from captures or imageLinks
    image_id = extract_image_id_from_item(item)

    # If no image ID found, skip this item
    if not image_id:
        return ('failed', uuid, 'No image ID found')

    # Construct image URL
    image_url = construct_image_url(image_id, size)

    # Download image if needed
    if not image_path.exists():
        success = download_image(image_url, image_path)
        if not success:
            # Try alternative URL constructions
            alt_urls = [
                f"https://images.nypl.org/index.php?id={image_id}&t=w",
                f"https://images.nypl.org/index.php?id={uuid}&t=w",
            ]
            for alt_url in alt_urls:
                success = download_image(alt_url, image_path)
                if success:
                    break
            if not success:
                return ('failed', uuid, 'Image download failed')

    # Extract metadata from NDJSON structure
    # Title is directly at top level
    title = item.get('title', '')

    # Date - can be array or string
    date_data = item.get('date', [])
    date_created = date_data[0] if isinstance(date_data, list) and date_data else str(date_data) if date_data else ''
    date_start = item.get('dateStart', '')
    date_end = item.get('dateEnd', '')

    # Contributors - array of objects with contributorName
    contributors = []
    contrib_list = item.get('contributor', [])
    for contrib in contrib_list if isinstance(contrib_list, list) else []:
        if isinstance(contrib, dict):
            name = contrib.get('contributorName', '')
            if name:
                contributors.append(name)

    # Subjects
    subjects = {
        'topical': [s.get('text', '') for s in item.get('subjectTopical', []) if isinstance(s, dict)],
        'geographic': [s.get('text', '') for s in item.get('subjectGeographic', []) if isinstance(s, dict)],
        'name': [s.get('text', '') for s in item.get('subjectName', []) if isinstance(s, dict)],
        'temporal': [s.get('text', '') for s in item.get('subjectTemporal', []) if isinstance(s, dict)],
    }

    # Genres - can be empty array
    genres = item.get('genre', [])
    if not isinstance(genres, list):
        genres = [genres] if genres else []

    # Type of resource - array
    resource_types = item.get('resourceType', [])
    type_of_resource = resource_types[0] if resource_types else ''

    # Collection info
    collection_uuid = item.get('collectionUUID', '')
    collection_title = item.get('collectionTitle', '')

    # Build and save metadata
    metadata = {
        'uuid': uuid,
        'database_id': item.get('databaseID', ''),
        'title': title,
        'alternative_titles': item.get('alternativeTitle', []),
        'date': date_created,
        'date_start': date_start,
        'date_end': date_end,
        'contributors': contributors,
        'subjects': subjects,
        'genres': genres,
        'type_of_resource': type_of_resource,
        'language': item.get('language', []),
        'description': item.get('description', []),
        'notes': item.get('note', []),
        'identifiers': {
            'b_number': item.get('identifierBNumber', ''),
            'call_number': item.get('identifierCallNumber', ''),
            'oclc': item.get('identifierOCLCRLIN', ''),
        },
        'physical_description': {
            'extent': item.get('physicalDescriptionExtent', []),
            'form': item.get('physicalDescriptionForm', []),
        },
        'publisher': item.get('publisher', []),
        'place_of_publication': item.get('placeOfPublication', []),
        'collection': {
            'uuid': collection_uuid,
            'title': collection_title,
        },
        'container': {
            'uuid': item.get('containerUUID', ''),
            'title': item.get('containerTitle', ''),
        },
        'parent_hierarchy': item.get('parentHierarchy', ''),
        'num_captures': item.get('numberOfCaptures', 1),
        'image': {
            'id': image_id,
            'url': image_url,
            'local_path': str(image_path.name)
        },
        'source': {
            'institution': 'The New York Public Library',
            'url': item.get('digitalCollectionsURL', f"https://digitalcollections.nypl.org/items/{uuid}"),
            'license': 'CC0 Public Domain',
            'attribution': 'From The New York Public Library'
        }
    }

    save_metadata(metadata, metadata_path)

    return ('success', uuid, None)


def parse_pipe_delimited(value: str) -> list:
    """Parse a pipe-delimited string into a list."""
    if not value:
        return []
    return [v.strip() for v in value.split('|') if v.strip()]


def get_item_genres(item: dict) -> list:
    """Extract genres from an item for filtering."""
    genres = item.get('genre', [])
    if isinstance(genres, list):
        return [g.lower() for g in genres if isinstance(g, str)]
    elif isinstance(genres, str):
        return [genres.lower()]
    # Also check resourceType as fallback
    resource_types = item.get('resourceType', [])
    if resource_types:
        return [r.lower() for r in resource_types if isinstance(r, str)]
    return []


def get_item_title(item: dict) -> str:
    """Extract title from an item."""
    return item.get('title', 'Unknown')


def download_images(
    items: list,
    output_dir: Path,
    metadata_dir: Path,
    cache_dir: Path,
    size: str = "medium",
    genre_filter: str = None,
    max_workers: int = 8,
    delay: float = 0.05,
    limit: int = None
):
    """Download images with parallel workers."""

    # Filter by genre if specified
    if genre_filter:
        genre_lower = genre_filter.lower()
        items = [i for i in items if any(genre_lower in g for g in get_item_genres(i))]
        logger.info(f"Filtered to {len(items)} items matching genre '{genre_filter}'")

    # Apply limit
    if limit:
        items = items[:limit]

    if not items:
        logger.warning("No items to download")
        return

    logger.info(f"Downloading {len(items)} images...")

    downloaded = 0
    skipped = 0
    failed = 0

    def process_one(item):
        return process_item(item, output_dir, metadata_dir, size, cache_dir, delay)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_one, item): item for item in items}

        for future in as_completed(futures):
            try:
                status, uuid, error = future.result()
                if status == 'success':
                    downloaded += 1
                elif status == 'skipped':
                    skipped += 1
                else:
                    failed += 1
                    if error:
                        logger.debug(f"Failed {uuid}: {error}")

                total = downloaded + skipped + failed
                if total % 100 == 0:
                    logger.info(f"Progress: {total}/{len(items)} "
                               f"(downloaded: {downloaded}, skipped: {skipped}, failed: {failed})")
            except Exception as e:
                failed += 1
                logger.warning(f"Worker error: {e}")

    logger.info(f"Complete! Downloaded: {downloaded}, Skipped: {skipped}, Failed: {failed}")


def main():
    parser = argparse.ArgumentParser(
        description="Download public domain images from NYPL Digital Collections"
    )
    parser.add_argument(
        '-o', '--output',
        type=Path,
        default=Path('./data/images'),
        help='Output directory for downloaded images'
    )
    parser.add_argument(
        '-m', '--metadata-dir',
        type=Path,
        default=Path('./data/metadata'),
        help='Directory for JSON metadata files'
    )
    parser.add_argument(
        '-c', '--cache',
        type=Path,
        default=Path('./data/cache'),
        help='Cache directory for NYPL data files'
    )
    parser.add_argument(
        '-s', '--size',
        type=str,
        default='medium',
        choices=['thumb', 'small', 'medium', 'large', 'xlarge', 'high'],
        help='Image size to download'
    )
    parser.add_argument(
        '--genre',
        type=str,
        default=None,
        help='Filter by genre (e.g., "Photographs", "Prints")'
    )
    parser.add_argument(
        '-w', '--workers',
        type=int,
        default=8,
        help='Number of parallel download workers'
    )
    parser.add_argument(
        '-d', '--delay',
        type=float,
        default=0.05,
        help='Delay between requests (seconds)'
    )
    parser.add_argument(
        '-l', '--limit',
        type=int,
        default=None,
        help='Limit number of images to download'
    )
    parser.add_argument(
        '--list-only',
        action='store_true',
        help='Only list items, do not download'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create directories
    args.output.mkdir(parents=True, exist_ok=True)
    args.metadata_dir.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)

    # Load items
    logger.info("Loading NYPL public domain items...")
    items = load_all_items(args.cache, limit=args.limit if args.list_only else None)

    # Apply genre filter for listing
    if args.genre:
        genre_lower = args.genre.lower()
        items = [i for i in items if any(genre_lower in g for g in get_item_genres(i))]

    if args.list_only:
        # Just list the items
        for i, item in enumerate(items[:args.limit or len(items)]):
            uuid = item.get('UUID') or item.get('uuid', 'unknown')
            title = get_item_title(item)
            genres = get_item_genres(item)
            genre_str = ', '.join(genres[:2]) if genres else ''
            print(f"{i+1}. [{uuid[:20]}...] {title[:60]} ({genre_str})")

        print(f"\nTotal: {len(items)} items")
        return

    # Download images
    download_images(
        items=items,
        output_dir=args.output,
        metadata_dir=args.metadata_dir,
        cache_dir=args.cache,
        size=args.size,
        genre_filter=args.genre,
        max_workers=args.workers,
        delay=args.delay,
        limit=args.limit
    )


if __name__ == '__main__':
    main()
