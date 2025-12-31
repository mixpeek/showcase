#!/usr/bin/env python3
"""
NGA Portrait Gallery Image Downloader

Downloads open-access images from the National Gallery of Art's collection.
Uses NGA's Open Data CSV files and IIIF Image API.

Data Source: https://github.com/NationalGalleryOfArt/opendata
License: CC0 (Public Domain)
"""

import os
import csv
import json
import time
import requests
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# NGA Open Data URLs
OPENDATA_BASE = "https://raw.githubusercontent.com/NationalGalleryOfArt/opendata/main/data"
PUBLISHED_IMAGES_URL = f"{OPENDATA_BASE}/published_images.csv"
OBJECTS_URL = f"{OPENDATA_BASE}/objects.csv"
OBJECTS_TERMS_URL = f"{OPENDATA_BASE}/objects_terms.csv"

# IIIF Image API base
IIIF_BASE = "https://media.nga.gov/iiif/public/objects"


def download_csv(url: str, cache_dir: Path) -> Path:
    """Download a CSV file from NGA's opendata repo with caching."""
    filename = url.split("/")[-1]
    cache_path = cache_dir / filename

    if cache_path.exists():
        logger.info(f"Using cached {filename}")
        return cache_path

    logger.info(f"Downloading {filename}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    logger.info(f"Downloaded {filename}")
    return cache_path


def load_published_images(csv_path: Path) -> dict:
    """Load published images CSV and return mapping of object_id -> image info."""
    images = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # NGA uses 'depictstmsobjectid' to link to objects
            object_id = row.get('depictstmsobjectid') or row.get('objectid') or row.get('object_id')
            if object_id:
                images[object_id] = row
    logger.info(f"Loaded {len(images)} published images")
    return images


def load_objects(csv_path: Path) -> dict:
    """Load objects CSV and return mapping of object_id -> object info."""
    objects = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            object_id = row.get('objectid') or row.get('object_id') or row.get('objectId')
            if object_id:
                objects[object_id] = row
    logger.info(f"Loaded {len(objects)} objects")
    return objects


def load_object_terms(csv_path: Path) -> dict:
    """Load object terms CSV and return mapping of object_id -> list of terms."""
    terms = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            object_id = row.get('objectid') or row.get('object_id') or row.get('objectId')
            term = row.get('term') or row.get('termtype') or ''
            if object_id:
                if object_id not in terms:
                    terms[object_id] = []
                terms[object_id].append(row)
    logger.info(f"Loaded terms for {len(terms)} objects")
    return terms


def construct_iiif_url(image_info: dict, size: str = "full") -> str:
    """
    Construct IIIF Image API URL for downloading.

    Size options:
    - "full": Full resolution
    - "1024,": Max width 1024px
    - ",1024": Max height 1024px
    - "512,512": Fit within 512x512
    """
    # NGA published_images.csv has 'iiifurl' field with direct IIIF base URL
    # e.g., https://api.nga.gov/iiif/00007f61-4922-417b-8f27-893ea328206c
    iiif_url = image_info.get('iiifurl') or image_info.get('iiif_url') or image_info.get('iiifURL')

    if iiif_url:
        # Direct IIIF URL from NGA - append size parameters
        return f"{iiif_url}/full/{size}/0/default.jpg"

    # Fallback: construct from UUID if available
    uuid = image_info.get('uuid')
    if uuid:
        return f"https://api.nga.gov/iiif/{uuid}/full/{size}/0/default.jpg"

    return None


def filter_portraits(objects: dict, terms: dict) -> set:
    """Filter objects to only include portraits based on classification and terms."""
    portrait_ids = set()

    portrait_keywords = [
        'portrait', 'self-portrait', 'self portrait',
        'bust', 'head', 'face', 'likeness'
    ]

    for object_id, obj in objects.items():
        # Check classification field
        classification = (obj.get('classification') or '').lower()
        title = (obj.get('title') or '').lower()

        is_portrait = any(kw in classification for kw in portrait_keywords)
        is_portrait = is_portrait or any(kw in title for kw in portrait_keywords)

        # Check terms
        if object_id in terms:
            for term_row in terms[object_id]:
                term_text = (term_row.get('term') or '').lower()
                if any(kw in term_text for kw in portrait_keywords):
                    is_portrait = True
                    break

        if is_portrait:
            portrait_ids.add(object_id)

    return portrait_ids


def download_image(url: str, output_path: Path, timeout: int = 30) -> bool:
    """Download a single image from IIIF endpoint."""
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


def download_images(
    images: dict,
    objects: dict,
    terms: dict,
    output_dir: Path,
    size: str = "1024,",
    filter_ids: set = None,
    max_workers: int = 4,
    delay: float = 0.1,
    limit: int = None,
    save_metadata: bool = True
):
    """Download images with parallel workers and rate limiting."""

    # Filter images if specified
    if filter_ids:
        images = {k: v for k, v in images.items() if k in filter_ids}

    # Apply limit
    if limit:
        images = dict(list(images.items())[:limit])

    logger.info(f"Downloading {len(images)} images...")

    downloaded = 0
    skipped = 0
    failed = 0

    def download_one(item):
        object_id, image_info = item

        # Get object metadata for filename
        obj = objects.get(object_id, {})
        title = obj.get('title', f'image_{object_id}')
        # Sanitize filename
        safe_title = "".join(c if c.isalnum() or c in ' -_' else '_' for c in title)[:100]
        base_filename = f"{object_id}_{safe_title}"
        image_path = output_dir / f"{base_filename}.jpg"
        metadata_path = output_dir / f"{base_filename}.json"

        # Skip if already exists (check both image and metadata)
        if image_path.exists() and (not save_metadata or metadata_path.exists()):
            return ('skipped', object_id)

        url = construct_iiif_url(image_info, size)
        if not url:
            return ('failed', object_id)

        time.sleep(delay)  # Rate limiting

        # Download image
        if not image_path.exists():
            if not download_image(url, image_path):
                return ('failed', object_id)

        # Save metadata JSON
        if save_metadata and not metadata_path.exists():
            object_terms = terms.get(object_id, [])
            metadata = {
                'object_id': object_id,
                'title': obj.get('title'),
                'artist': obj.get('attribution'),
                'artist_inverted': obj.get('attributioninverted'),
                'date': obj.get('displaydate'),
                'year_start': obj.get('beginyear'),
                'year_end': obj.get('endyear'),
                'medium': obj.get('medium'),
                'dimensions': obj.get('dimensions'),
                'classification': obj.get('classification'),
                'subclassification': obj.get('subclassification'),
                'department': obj.get('departmentabbr'),
                'creditline': obj.get('creditline'),
                'accession_number': obj.get('accessionnum'),
                'provenance': obj.get('provenancetext'),
                'inscription': obj.get('inscription'),
                'wikidata_id': obj.get('wikidataid'),
                'image': {
                    'uuid': image_info.get('uuid'),
                    'iiif_url': image_info.get('iiifurl'),
                    'width': image_info.get('width'),
                    'height': image_info.get('height'),
                    'alt_text': image_info.get('assistivetext'),
                },
                'terms': [
                    {'type': t.get('termtype'), 'term': t.get('term')}
                    for t in object_terms
                ],
                'source': {
                    'institution': 'National Gallery of Art',
                    'url': f"https://www.nga.gov/collection/art-object-page.{object_id}.html",
                    'license': 'CC0 Public Domain',
                }
            }
            try:
                metadata_path.parent.mkdir(parents=True, exist_ok=True)
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
            except Exception as e:
                logger.warning(f"Failed to save metadata for {object_id}: {e}")

        return ('success', object_id)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_one, item): item for item in images.items()}

        for future in as_completed(futures):
            status, object_id = future.result()
            if status == 'success':
                downloaded += 1
            elif status == 'skipped':
                skipped += 1
            else:
                failed += 1

            total = downloaded + skipped + failed
            if total % 100 == 0:
                logger.info(f"Progress: {total}/{len(images)} (downloaded: {downloaded}, skipped: {skipped}, failed: {failed})")

    logger.info(f"Complete! Downloaded: {downloaded}, Skipped: {skipped}, Failed: {failed}")


def main():
    parser = argparse.ArgumentParser(
        description="Download images from National Gallery of Art's open access collection"
    )
    parser.add_argument(
        '-o', '--output',
        type=Path,
        default=Path('./data/images'),
        help='Output directory for downloaded images'
    )
    parser.add_argument(
        '-c', '--cache',
        type=Path,
        default=Path('./data/cache'),
        help='Cache directory for CSV files'
    )
    parser.add_argument(
        '-s', '--size',
        type=str,
        default='1024,',
        help='Image size (IIIF format): "full", "1024,", ",1024", "512,512"'
    )
    parser.add_argument(
        '--portraits-only',
        action='store_true',
        help='Only download portrait images'
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
        default=0.1,
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
        help='Only list images, do not download'
    )

    args = parser.parse_args()

    # Download CSV data files
    logger.info("Fetching NGA Open Data files...")
    images_csv = download_csv(PUBLISHED_IMAGES_URL, args.cache)
    objects_csv = download_csv(OBJECTS_URL, args.cache)
    terms_csv = download_csv(OBJECTS_TERMS_URL, args.cache)

    # Load data
    logger.info("Loading data...")
    images = load_published_images(images_csv)
    objects = load_objects(objects_csv)
    terms = load_object_terms(terms_csv)

    # Filter for portraits if requested
    filter_ids = None
    if args.portraits_only:
        logger.info("Filtering for portrait images...")
        filter_ids = filter_portraits(objects, terms)
        logger.info(f"Found {len(filter_ids)} portrait images")

    if args.list_only:
        # Just list the images
        count = 0
        for object_id in (filter_ids or images.keys()):
            if object_id in images:
                obj = objects.get(object_id, {})
                title = obj.get('title', 'Unknown')
                artist = obj.get('attribution', 'Unknown')
                print(f"{object_id}: {title} - {artist}")
                count += 1
                if args.limit and count >= args.limit:
                    break
        print(f"\nTotal: {count} images")
        return

    # Download images
    download_images(
        images=images,
        objects=objects,
        terms=terms,
        output_dir=args.output,
        size=args.size,
        filter_ids=filter_ids,
        max_workers=args.workers,
        delay=args.delay,
        limit=args.limit,
        save_metadata=True
    )


if __name__ == '__main__':
    main()
