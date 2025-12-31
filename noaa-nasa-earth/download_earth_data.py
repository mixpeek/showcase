#!/usr/bin/env python3
"""
NOAA/NASA Earth Dataset Downloader

Downloads public domain Earth observation data from NASA and NOAA APIs:
- NASA EPIC (Earth Polychromatic Imaging Camera) - Full Earth disc images
- NASA EONET (Earth Observatory Natural Event Tracker) - Natural events metadata
- NASA GIBS - Satellite imagery tiles for specific events

Data Sources:
- NASA EPIC: https://epic.gsfc.nasa.gov/about/api
- NASA EONET: https://eonet.gsfc.nasa.gov/docs/v3
- NASA GIBS: https://nasa-gibs.github.io/gibs-api-docs/

License: Public Domain (U.S. Government Works)
"""

import os
import json
import time
import requests
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# API Endpoints
EPIC_API_BASE = "https://epic.gsfc.nasa.gov/api"
EPIC_ARCHIVE_BASE = "https://epic.gsfc.nasa.gov/archive"
EONET_API_BASE = "https://eonet.gsfc.nasa.gov/api/v3"
GIBS_WMTS_BASE = "https://gibs.earthdata.nasa.gov/wmts/epsg4326/best"

# EONET Event Categories
EONET_CATEGORIES = {
    'drought': 'Drought',
    'dustHaze': 'Dust and Haze',
    'earthquakes': 'Earthquakes',
    'floods': 'Floods',
    'landslides': 'Landslides',
    'manmade': 'Manmade',
    'seaLakeIce': 'Sea and Lake Ice',
    'severeStorms': 'Severe Storms',
    'snow': 'Snow',
    'tempExtremes': 'Temperature Extremes',
    'volcanoes': 'Volcanoes',
    'waterColor': 'Water Color',
    'wildfires': 'Wildfires'
}


def fetch_json(url: str, timeout: int = 30) -> dict:
    """Fetch JSON data from an API endpoint."""
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.warning(f"Failed to fetch {url}: {e}")
        return None


def download_file(url: str, output_path: Path, timeout: int = 60) -> bool:
    """Download a file from URL to output path."""
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


# ============================================================================
# NASA EPIC Functions - Full Earth Disc Images
# ============================================================================

def get_epic_available_dates(collection: str = "natural") -> list:
    """Get list of all available dates for EPIC imagery."""
    url = f"{EPIC_API_BASE}/{collection}/all"
    data = fetch_json(url)
    if data:
        return [item['date'] for item in data]
    return []


def get_epic_images_for_date(date: str, collection: str = "natural") -> list:
    """Get EPIC image metadata for a specific date."""
    url = f"{EPIC_API_BASE}/{collection}/date/{date}"
    return fetch_json(url) or []


def construct_epic_image_url(image_data: dict, collection: str = "natural",
                             format: str = "png") -> str:
    """
    Construct download URL for EPIC image.

    Formats:
    - png: Full resolution (~2048x2048)
    - jpg: Half resolution
    - thumbs: Thumbnail
    """
    date = image_data['date'].split(' ')[0]
    year, month, day = date.split('-')
    filename = image_data['image']

    return f"{EPIC_ARCHIVE_BASE}/{collection}/{year}/{month}/{day}/{format}/{filename}.{format}"


def download_epic_images(
    output_dir: Path,
    collection: str = "natural",
    format: str = "png",
    days: int = 7,
    limit: int = None,
    max_workers: int = 4,
    delay: float = 0.2,
    save_metadata: bool = True
):
    """
    Download EPIC Earth images.

    Args:
        output_dir: Directory to save images
        collection: 'natural', 'enhanced', 'aerosol', or 'cloud'
        format: 'png' (full), 'jpg' (half), or 'thumbs'
        days: Number of recent days to download
        limit: Maximum number of images
        max_workers: Parallel download threads
        delay: Delay between requests
        save_metadata: Save JSON metadata alongside images
    """
    logger.info(f"Fetching EPIC {collection} imagery for last {days} days...")

    # Get available dates
    all_dates = get_epic_available_dates(collection)
    if not all_dates:
        logger.error("Could not fetch available dates")
        return

    # Filter to recent dates
    cutoff = datetime.now() - timedelta(days=days)
    recent_dates = [d for d in all_dates if datetime.strptime(d, '%Y-%m-%d') >= cutoff]

    logger.info(f"Found {len(recent_dates)} dates with imagery")

    # Collect all images
    all_images = []
    for date in recent_dates:
        images = get_epic_images_for_date(date, collection)
        for img in images:
            img['_date'] = date
            img['_collection'] = collection
        all_images.extend(images)
        time.sleep(0.1)  # Rate limit API calls

    if limit:
        all_images = all_images[:limit]

    logger.info(f"Downloading {len(all_images)} EPIC images...")

    downloaded = 0
    skipped = 0
    failed = 0

    def download_one(image_data):
        image_name = image_data['image']
        date_str = image_data['_date'].replace('-', '')

        image_path = output_dir / f"epic_{collection}_{image_name}.{format}"
        metadata_path = output_dir / f"epic_{collection}_{image_name}.json"

        # Skip if already exists
        if image_path.exists() and (not save_metadata or metadata_path.exists()):
            return ('skipped', image_name)

        time.sleep(delay)

        # Download image
        url = construct_epic_image_url(image_data, collection, format)
        if not download_file(url, image_path):
            return ('failed', image_name)

        # Save metadata
        if save_metadata and not metadata_path.exists():
            metadata = {
                'identifier': image_data.get('identifier'),
                'image': image_name,
                'date': image_data.get('date'),
                'caption': image_data.get('caption'),
                'centroid_coordinates': image_data.get('centroid_coordinates'),
                'dscovr_j2000_position': image_data.get('dscovr_j2000_position'),
                'lunar_j2000_position': image_data.get('lunar_j2000_position'),
                'sun_j2000_position': image_data.get('sun_j2000_position'),
                'attitude_quaternions': image_data.get('attitude_quaternions'),
                'collection': collection,
                'format': format,
                'source': {
                    'mission': 'DSCOVR',
                    'instrument': 'EPIC (Earth Polychromatic Imaging Camera)',
                    'institution': 'NASA',
                    'api_url': f"{EPIC_API_BASE}/{collection}",
                    'license': 'Public Domain (U.S. Government Work)',
                }
            }
            try:
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
            except Exception as e:
                logger.warning(f"Failed to save metadata for {image_name}: {e}")

        return ('success', image_name)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_one, img): img for img in all_images}

        for future in as_completed(futures):
            status, name = future.result()
            if status == 'success':
                downloaded += 1
            elif status == 'skipped':
                skipped += 1
            else:
                failed += 1

            total = downloaded + skipped + failed
            if total % 10 == 0:
                logger.info(f"EPIC Progress: {total}/{len(all_images)} "
                           f"(downloaded: {downloaded}, skipped: {skipped}, failed: {failed})")

    logger.info(f"EPIC Complete! Downloaded: {downloaded}, Skipped: {skipped}, Failed: {failed}")


# ============================================================================
# NASA EONET Functions - Natural Events (Hurricanes, Volcanoes, etc.)
# ============================================================================

def get_eonet_categories() -> list:
    """Get all EONET event categories."""
    url = f"{EONET_API_BASE}/categories"
    return fetch_json(url) or []


def get_eonet_events(
    category: str = None,
    status: str = "all",
    days: int = 365,
    limit: int = None
) -> list:
    """
    Get EONET natural events.

    Args:
        category: Filter by category (e.g., 'severeStorms', 'wildfires')
        status: 'open', 'closed', or 'all'
        days: Number of days to look back
        limit: Maximum events to return
    """
    params = {
        'status': status,
        'days': days,
    }
    if limit:
        params['limit'] = limit

    if category:
        url = f"{EONET_API_BASE}/categories/{category}"
    else:
        url = f"{EONET_API_BASE}/events"

    query_string = '&'.join(f"{k}={v}" for k, v in params.items())
    full_url = f"{url}?{query_string}"

    data = fetch_json(full_url)
    if data and 'events' in data:
        return data['events']
    return []


def get_eonet_event_geojson(
    category: str = None,
    status: str = "all",
    days: int = 365,
    limit: int = None
) -> dict:
    """Get EONET events in GeoJSON format."""
    params = {
        'status': status,
        'days': days,
    }
    if limit:
        params['limit'] = limit

    if category:
        url = f"{EONET_API_BASE}/categories/{category}/geojson"
    else:
        url = f"{EONET_API_BASE}/events/geojson"

    query_string = '&'.join(f"{k}={v}" for k, v in params.items())
    full_url = f"{url}?{query_string}"

    return fetch_json(full_url)


def download_eonet_events(
    output_dir: Path,
    categories: list = None,
    days: int = 365,
    limit: int = None,
    include_geojson: bool = True
):
    """
    Download EONET natural events metadata.

    Args:
        output_dir: Directory to save event data
        categories: List of categories to download (None = all)
        days: Number of days to look back
        limit: Maximum events per category
        include_geojson: Also save GeoJSON format
    """
    if categories is None:
        categories = list(EONET_CATEGORIES.keys())

    logger.info(f"Downloading EONET events for {len(categories)} categories...")

    all_events = []

    for category in categories:
        logger.info(f"Fetching {EONET_CATEGORIES.get(category, category)} events...")

        events = get_eonet_events(
            category=category,
            status='all',
            days=days,
            limit=limit
        )

        logger.info(f"  Found {len(events)} {category} events")

        for event in events:
            event['_category'] = category
        all_events.extend(events)

        # Save category-specific file
        category_path = output_dir / f"eonet_{category}.json"
        with open(category_path, 'w', encoding='utf-8') as f:
            json.dump({
                'category': category,
                'category_title': EONET_CATEGORIES.get(category, category),
                'days': days,
                'count': len(events),
                'events': events,
                'source': {
                    'api': 'NASA EONET v3',
                    'url': f"{EONET_API_BASE}/categories/{category}",
                    'license': 'Public Domain (U.S. Government Work)',
                }
            }, f, indent=2, ensure_ascii=False)

        # Save GeoJSON version
        if include_geojson:
            geojson = get_eonet_event_geojson(
                category=category,
                status='all',
                days=days,
                limit=limit
            )
            if geojson:
                geojson_path = output_dir / f"eonet_{category}.geojson"
                with open(geojson_path, 'w', encoding='utf-8') as f:
                    json.dump(geojson, f, indent=2, ensure_ascii=False)

        time.sleep(0.2)  # Rate limit

    # Save combined events file
    combined_path = output_dir / "eonet_all_events.json"
    with open(combined_path, 'w', encoding='utf-8') as f:
        json.dump({
            'categories': categories,
            'days': days,
            'total_count': len(all_events),
            'events': all_events,
            'source': {
                'api': 'NASA EONET v3',
                'url': EONET_API_BASE,
                'license': 'Public Domain (U.S. Government Work)',
            }
        }, f, indent=2, ensure_ascii=False)

    logger.info(f"EONET Complete! Total events: {len(all_events)}")
    return all_events


# ============================================================================
# NASA GIBS Functions - Satellite Imagery Tiles
# ============================================================================

def get_gibs_tile_url(
    layer: str,
    date: str,
    zoom: int,
    row: int,
    col: int,
    format: str = "jpg"
) -> str:
    """
    Construct GIBS WMTS tile URL.

    Popular layers:
    - MODIS_Terra_CorrectedReflectance_TrueColor
    - MODIS_Aqua_CorrectedReflectance_TrueColor
    - VIIRS_SNPP_CorrectedReflectance_TrueColor
    - GOES-East_ABI_Band2_Red_Visible_1km
    """
    return (f"{GIBS_WMTS_BASE}/{layer}/default/{date}/250m/"
            f"{zoom}/{row}/{col}.{format}")


def download_gibs_region(
    output_dir: Path,
    layer: str,
    date: str,
    bbox: tuple,  # (min_lon, min_lat, max_lon, max_lat)
    zoom: int = 6,
    format: str = "jpg"
):
    """
    Download GIBS imagery tiles for a geographic region.

    Note: This is a simplified implementation. For production use,
    consider using GDAL or dedicated GIBS tools.
    """
    # This is a placeholder - GIBS tile calculation is complex
    # and depends on projection and tile matrix set
    logger.warning("GIBS regional download is a placeholder - use NASA Worldview for bulk downloads")
    pass


# ============================================================================
# Integrated Download Functions
# ============================================================================

def download_hurricane_package(
    output_dir: Path,
    days: int = 90,
    limit: int = 50
):
    """
    Download a curated package focused on hurricanes/severe storms.

    Includes:
    - EONET severe storm events with coordinates
    - Recent EPIC Earth images (to visualize storm systems)
    """
    logger.info("=== Downloading Hurricane/Storm Package ===")

    # Create subdirectories
    events_dir = output_dir / "events"
    images_dir = output_dir / "images"
    events_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    # Download storm events
    logger.info("Downloading severe storm events...")
    events = download_eonet_events(
        output_dir=events_dir,
        categories=['severeStorms'],
        days=days,
        limit=limit
    )

    # Download EPIC images
    logger.info("Downloading EPIC Earth images...")
    download_epic_images(
        output_dir=images_dir,
        collection='natural',
        format='jpg',  # Smaller file size
        days=min(days, 30),  # Limit to recent imagery
        limit=limit,
        max_workers=4
    )

    logger.info("=== Hurricane Package Complete ===")


def download_full_earth_package(
    output_dir: Path,
    days: int = 30,
    epic_limit: int = 100,
    eonet_limit: int = None
):
    """
    Download comprehensive Earth observation package.

    Includes:
    - All EONET natural events (hurricanes, volcanoes, wildfires, etc.)
    - EPIC natural and enhanced Earth images
    """
    logger.info("=== Downloading Full Earth Package ===")

    # Create subdirectories
    events_dir = output_dir / "metadata"
    natural_dir = output_dir / "images" / "natural"
    enhanced_dir = output_dir / "images" / "enhanced"

    events_dir.mkdir(parents=True, exist_ok=True)
    natural_dir.mkdir(parents=True, exist_ok=True)
    enhanced_dir.mkdir(parents=True, exist_ok=True)

    # Download all EONET events
    logger.info("Downloading all EONET natural events...")
    download_eonet_events(
        output_dir=events_dir,
        categories=None,  # All categories
        days=days,
        limit=eonet_limit
    )

    # Download EPIC natural images
    logger.info("Downloading EPIC natural color images...")
    download_epic_images(
        output_dir=natural_dir,
        collection='natural',
        format='jpg',
        days=days,
        limit=epic_limit,
        max_workers=4
    )

    # Download EPIC enhanced images
    logger.info("Downloading EPIC enhanced color images...")
    download_epic_images(
        output_dir=enhanced_dir,
        collection='enhanced',
        format='jpg',
        days=days,
        limit=epic_limit,
        max_workers=4
    )

    logger.info("=== Full Earth Package Complete ===")


def main():
    parser = argparse.ArgumentParser(
        description="Download NOAA/NASA Earth observation data (public domain)"
    )

    # Output options
    parser.add_argument(
        '-o', '--output',
        type=Path,
        default=Path('./data'),
        help='Output directory for downloaded data'
    )

    # Data source options
    parser.add_argument(
        '--source',
        choices=['epic', 'eonet', 'hurricanes', 'full'],
        default='full',
        help='Data source to download'
    )

    # EPIC options
    parser.add_argument(
        '--epic-collection',
        choices=['natural', 'enhanced', 'aerosol', 'cloud'],
        default='natural',
        help='EPIC image collection type'
    )
    parser.add_argument(
        '--epic-format',
        choices=['png', 'jpg', 'thumbs'],
        default='jpg',
        help='EPIC image format (png=full res, jpg=half res)'
    )

    # EONET options
    parser.add_argument(
        '--eonet-category',
        choices=list(EONET_CATEGORIES.keys()),
        default=None,
        help='EONET category to download (default: all)'
    )

    # Common options
    parser.add_argument(
        '-d', '--days',
        type=int,
        default=30,
        help='Number of days of data to download'
    )
    parser.add_argument(
        '-l', '--limit',
        type=int,
        default=50,
        help='Maximum number of items to download'
    )
    parser.add_argument(
        '-w', '--workers',
        type=int,
        default=4,
        help='Number of parallel download workers'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=0.2,
        help='Delay between requests (seconds)'
    )
    parser.add_argument(
        '--list-only',
        action='store_true',
        help='List available data without downloading'
    )

    args = parser.parse_args()

    if args.list_only:
        # List mode
        if args.source in ['epic', 'full']:
            logger.info("=== Available EPIC Dates ===")
            dates = get_epic_available_dates(args.epic_collection)
            for date in dates[-10:]:  # Show last 10
                logger.info(f"  {date}")
            logger.info(f"  ... ({len(dates)} total dates available)")

        if args.source in ['eonet', 'hurricanes', 'full']:
            logger.info("\n=== EONET Categories ===")
            for cat_id, cat_name in EONET_CATEGORIES.items():
                events = get_eonet_events(category=cat_id, days=args.days, limit=5)
                logger.info(f"  {cat_name}: {len(events)}+ events")
        return

    # Download mode
    if args.source == 'epic':
        output_dir = args.output / 'images'
        output_dir.mkdir(parents=True, exist_ok=True)
        download_epic_images(
            output_dir=output_dir,
            collection=args.epic_collection,
            format=args.epic_format,
            days=args.days,
            limit=args.limit,
            max_workers=args.workers,
            delay=args.delay
        )

    elif args.source == 'eonet':
        output_dir = args.output / 'metadata'
        output_dir.mkdir(parents=True, exist_ok=True)
        categories = [args.eonet_category] if args.eonet_category else None
        download_eonet_events(
            output_dir=output_dir,
            categories=categories,
            days=args.days,
            limit=args.limit
        )

    elif args.source == 'hurricanes':
        download_hurricane_package(
            output_dir=args.output,
            days=args.days,
            limit=args.limit
        )

    elif args.source == 'full':
        download_full_earth_package(
            output_dir=args.output,
            days=args.days,
            epic_limit=args.limit,
            eonet_limit=None
        )


if __name__ == '__main__':
    main()
