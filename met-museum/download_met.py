#!/usr/bin/env python3
"""
The Met Museum Public Domain Artworks Downloader

Downloads public domain artworks from The Metropolitan Museum of Art's collection.
Uses The Met Collection API.

API Documentation: https://metmuseum.github.io/
License: CC0 (Public Domain) for public domain artworks
"""

import os
import json
import time
import random
import requests
import argparse
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import cycle
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Met Museum API endpoints
API_BASE = "https://collectionapi.metmuseum.org/public/collection/v1"
OBJECTS_ENDPOINT = f"{API_BASE}/objects"
SEARCH_ENDPOINT = f"{API_BASE}/search"
DEPARTMENTS_ENDPOINT = f"{API_BASE}/departments"

# Default headers to avoid being blocked
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json,text/html,*/*",
    "Accept-Language": "en-US,en;q=0.9",
}

# User agents for rotation
USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
]


class ProxyManager:
    """Manages proxy rotation for requests."""

    def __init__(self, proxy_file: str = None, proxy_list: list = None):
        self.proxies = []
        self.proxy_cycle = None
        self.lock = threading.Lock()
        self.failed_proxies = set()

        if proxy_file and Path(proxy_file).exists():
            self._load_from_file(proxy_file)
        elif proxy_list:
            self.proxies = proxy_list
        else:
            self._fetch_free_proxies()

        if self.proxies:
            random.shuffle(self.proxies)
            self.proxy_cycle = cycle(self.proxies)
            logger.info(f"Loaded {len(self.proxies)} proxies")

    def _load_from_file(self, filepath: str):
        """Load proxies from file (one per line, format: ip:port or http://ip:port)"""
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if not line.startswith('http'):
                        line = f"http://{line}"
                    self.proxies.append(line)

    def _fetch_free_proxies(self):
        """Fetch free proxy list from public sources."""
        proxy_urls = [
            "https://raw.githubusercontent.com/TheSpeedX/PROXY-List/master/http.txt",
            "https://raw.githubusercontent.com/clarketm/proxy-list/master/proxy-list-raw.txt",
        ]

        for url in proxy_urls:
            try:
                resp = requests.get(url, timeout=10)
                if resp.status_code == 200:
                    for line in resp.text.strip().split('\n'):
                        line = line.strip()
                        if line and ':' in line:
                            proxy = f"http://{line.split()[0]}" if ' ' in line else f"http://{line}"
                            if proxy not in self.proxies:
                                self.proxies.append(proxy)
                    logger.info(f"Fetched proxies from {url}")
            except Exception as e:
                logger.debug(f"Failed to fetch proxies from {url}: {e}")

        # Limit to reasonable number
        if len(self.proxies) > 500:
            self.proxies = random.sample(self.proxies, 500)

    def get_proxy(self) -> dict:
        """Get next proxy in rotation."""
        if not self.proxy_cycle:
            return None

        with self.lock:
            # Try to get a working proxy
            for _ in range(min(10, len(self.proxies))):
                proxy = next(self.proxy_cycle)
                if proxy not in self.failed_proxies:
                    return {"http": proxy, "https": proxy}

            # If all tried proxies failed, reset and try again
            self.failed_proxies.clear()
            proxy = next(self.proxy_cycle)
            return {"http": proxy, "https": proxy}

    def mark_failed(self, proxy_dict: dict):
        """Mark a proxy as failed."""
        if proxy_dict:
            with self.lock:
                self.failed_proxies.add(proxy_dict.get("http"))

    def has_proxies(self) -> bool:
        return len(self.proxies) > 0


# Global proxy manager (initialized in main)
proxy_manager: ProxyManager = None

# Create a session with default headers
session = requests.Session()
session.headers.update(DEFAULT_HEADERS)


def make_request(url: str, timeout: int = 30, stream: bool = False, params: dict = None) -> requests.Response:
    """Make a request with optional proxy rotation and user-agent rotation."""
    headers = dict(DEFAULT_HEADERS)
    headers["User-Agent"] = random.choice(USER_AGENTS)

    proxies = None
    if proxy_manager and proxy_manager.has_proxies():
        proxies = proxy_manager.get_proxy()

    try:
        response = requests.get(
            url,
            headers=headers,
            proxies=proxies,
            timeout=timeout,
            stream=stream,
            params=params
        )

        # Check if we got blocked (Incapsula returns HTML)
        if response.status_code == 200 and 'incapsula' in response.text.lower()[:1000]:
            if proxies:
                proxy_manager.mark_failed(proxies)
            raise requests.exceptions.RequestException("Blocked by Incapsula")

        return response

    except requests.exceptions.RequestException as e:
        if proxies:
            proxy_manager.mark_failed(proxies)
        raise

# Department IDs for reference
DEPARTMENTS = {
    1: "American Decorative Arts",
    3: "Ancient Near Eastern Art",
    4: "Arms and Armor",
    5: "Arts of Africa, Oceania, and the Americas",
    6: "Asian Art",
    7: "Cloisters",
    8: "Costume Institute",
    9: "Drawings and Prints",
    10: "Egyptian Art",
    11: "European Paintings",
    12: "European Sculpture and Decorative Arts",
    13: "Greek and Roman Art",
    14: "Islamic Art",
    15: "Lehman Collection",
    16: "Library",
    17: "Medieval Art",
    18: "Musical Instruments",
    19: "Photographs",
    21: "Modern and Contemporary Art",
}


def get_public_domain_object_ids(
    cache_dir: Path,
    department_ids: list = None,
    has_images: bool = True,
    use_cache: bool = True
) -> list:
    """
    Get all public domain object IDs from The Met.

    Note: The Met API doesn't have a direct isPublicDomain filter on search,
    so we search with hasImages=true and filter by isPublicDomain in metadata.
    """
    # Create cache filename based on department filter
    if department_ids:
        dept_suffix = "_".join(str(d) for d in sorted(department_ids))
        cache_file = cache_dir / f"object_ids_dept_{dept_suffix}.json"
    else:
        cache_file = cache_dir / "object_ids_all.json"

    if use_cache and cache_file.exists():
        logger.info("Using cached object IDs")
        with open(cache_file, 'r') as f:
            return json.load(f)

    # Search for objects with images
    params = {"hasImages": "true", "q": "*"}
    if department_ids:
        params["departmentIds"] = "|".join(str(d) for d in department_ids)

    logger.info("Fetching object IDs from Met API...")
    response = make_request(SEARCH_ENDPOINT, params=params, timeout=60)
    response.raise_for_status()

    data = response.json()
    object_ids = data.get("objectIDs", [])

    logger.info(f"Found {len(object_ids)} objects with images")

    # Cache the results
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_file, 'w') as f:
        json.dump(object_ids, f)

    return object_ids


def get_object_metadata(object_id: int, timeout: int = 30, max_retries: int = 5) -> dict:
    """Fetch metadata for a single object with retry logic and proxy rotation."""
    url = f"{OBJECTS_ENDPOINT}/{object_id}"

    for attempt in range(max_retries):
        try:
            response = make_request(url, timeout=timeout)
            if response.status_code == 429:  # Rate limited
                wait_time = (2 ** attempt) + 1
                logger.debug(f"Rate limited on {object_id}, waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) * 0.5 + random.uniform(0, 1)
                time.sleep(wait_time)
            else:
                logger.debug(f"Failed to fetch metadata for {object_id}: {e}")
    return None


def download_image(url: str, output_path: Path, timeout: int = 60, max_retries: int = 5) -> bool:
    """Download a single image with retry logic and proxy rotation."""
    for attempt in range(max_retries):
        try:
            response = make_request(url, timeout=timeout, stream=True)
            if response.status_code == 429:  # Rate limited
                wait_time = (2 ** attempt) + 1
                time.sleep(wait_time)
                continue
            response.raise_for_status()

            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) * 0.5 + random.uniform(0, 1)
                time.sleep(wait_time)
            else:
                logger.debug(f"Failed to download {url}: {e}")
    return False


def sanitize_filename(text: str, max_length: int = 80) -> str:
    """Sanitize a string for use as a filename."""
    if not text:
        return ""
    # Replace problematic characters
    safe = "".join(c if c.isalnum() or c in ' -_' else '_' for c in text)
    # Collapse multiple underscores/spaces
    while '  ' in safe:
        safe = safe.replace('  ', ' ')
    while '__' in safe:
        safe = safe.replace('__', '_')
    return safe.strip()[:max_length]


def process_object(
    object_id: int,
    images_dir: Path,
    metadata_dir: Path,
    size: str = "primary",
    delay: float = 0.05,
    download_additional: bool = False
) -> dict:
    """
    Process a single object: fetch metadata, download image(s).

    Returns status dict with keys: status, object_id, message
    """
    time.sleep(delay)  # Rate limiting

    # Fetch metadata
    metadata = get_object_metadata(object_id)
    if not metadata:
        return {"status": "failed", "object_id": object_id, "message": "Failed to fetch metadata"}

    # Check if public domain
    if not metadata.get("isPublicDomain", False):
        return {"status": "skipped", "object_id": object_id, "message": "Not public domain"}

    # Check for images
    primary_image = metadata.get("primaryImage", "")
    if not primary_image:
        return {"status": "skipped", "object_id": object_id, "message": "No image available"}

    # Build filename
    title = sanitize_filename(metadata.get("title", ""))
    artist = sanitize_filename(metadata.get("artistDisplayName", ""))
    if title and artist:
        base_name = f"{object_id}_{artist}_{title}"
    elif title:
        base_name = f"{object_id}_{title}"
    else:
        base_name = f"{object_id}"
    base_name = base_name[:150]  # Limit total length

    # Paths
    image_path = images_dir / f"{base_name}.jpg"
    metadata_path = metadata_dir / f"{base_name}.json"

    # Skip if already downloaded
    if image_path.exists() and metadata_path.exists():
        return {"status": "exists", "object_id": object_id, "message": "Already downloaded"}

    # Download primary image
    if size == "small":
        image_url = metadata.get("primaryImageSmall", primary_image)
    else:
        image_url = primary_image

    if not image_path.exists():
        if not download_image(image_url, image_path):
            return {"status": "failed", "object_id": object_id, "message": "Failed to download image"}

    # Download additional images if requested
    additional_paths = []
    if download_additional:
        additional_images = metadata.get("additionalImages", [])
        for i, add_url in enumerate(additional_images[:5]):  # Limit to 5 additional
            add_path = images_dir / f"{base_name}_additional_{i+1}.jpg"
            if not add_path.exists():
                if download_image(add_url, add_path):
                    additional_paths.append(str(add_path.name))

    # Save metadata
    if not metadata_path.exists():
        clean_metadata = {
            "object_id": metadata.get("objectID"),
            "title": metadata.get("title"),
            "artist": metadata.get("artistDisplayName"),
            "artist_bio": metadata.get("artistDisplayBio"),
            "artist_nationality": metadata.get("artistNationality"),
            "artist_begin_date": metadata.get("artistBeginDate"),
            "artist_end_date": metadata.get("artistEndDate"),
            "date": metadata.get("objectDate"),
            "date_begin": metadata.get("objectBeginDate"),
            "date_end": metadata.get("objectEndDate"),
            "medium": metadata.get("medium"),
            "dimensions": metadata.get("dimensions"),
            "classification": metadata.get("classification"),
            "department": metadata.get("department"),
            "culture": metadata.get("culture"),
            "period": metadata.get("period"),
            "dynasty": metadata.get("dynasty"),
            "reign": metadata.get("reign"),
            "portfolio": metadata.get("portfolio"),
            "credit_line": metadata.get("creditLine"),
            "accession_number": metadata.get("accessionNumber"),
            "accession_year": metadata.get("accessionYear"),
            "geography_type": metadata.get("geographyType"),
            "city": metadata.get("city"),
            "state": metadata.get("state"),
            "county": metadata.get("county"),
            "country": metadata.get("country"),
            "region": metadata.get("region"),
            "subregion": metadata.get("subregion"),
            "locale": metadata.get("locale"),
            "locus": metadata.get("locus"),
            "excavation": metadata.get("excavation"),
            "river": metadata.get("river"),
            "tags": [t.get("term") for t in (metadata.get("tags") or []) if t],
            "is_public_domain": metadata.get("isPublicDomain"),
            "is_highlight": metadata.get("isHighlight"),
            "gallery_number": metadata.get("GalleryNumber"),
            "primary_image": metadata.get("primaryImage"),
            "primary_image_small": metadata.get("primaryImageSmall"),
            "additional_images": metadata.get("additionalImages", []),
            "downloaded_images": {
                "primary": str(image_path.name),
                "additional": additional_paths
            },
            "source": {
                "institution": "The Metropolitan Museum of Art",
                "url": metadata.get("objectURL"),
                "api_url": f"{OBJECTS_ENDPOINT}/{object_id}",
                "wikidata_url": metadata.get("objectWikidata_URL"),
                "license": "CC0 Public Domain"
            }
        }

        try:
            metadata_dir.mkdir(parents=True, exist_ok=True)
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(clean_metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to save metadata for {object_id}: {e}")

    return {"status": "success", "object_id": object_id, "message": "Downloaded"}


def download_artworks(
    object_ids: list,
    images_dir: Path,
    metadata_dir: Path,
    size: str = "primary",
    max_workers: int = 8,
    delay: float = 0.05,
    limit: int = None,
    download_additional: bool = False
):
    """Download artworks with parallel workers."""

    if limit:
        object_ids = object_ids[:limit]

    logger.info(f"Processing {len(object_ids)} objects...")

    stats = {"success": 0, "exists": 0, "skipped": 0, "failed": 0}

    def process_one(obj_id):
        return process_object(
            obj_id,
            images_dir=images_dir,
            metadata_dir=metadata_dir,
            size=size,
            delay=delay,
            download_additional=download_additional
        )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_one, obj_id): obj_id for obj_id in object_ids}

        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            status = result["status"]
            stats[status] = stats.get(status, 0) + 1

            if i % 100 == 0 or i == len(object_ids):
                logger.info(
                    f"Progress: {i}/{len(object_ids)} | "
                    f"Success: {stats['success']} | "
                    f"Exists: {stats['exists']} | "
                    f"Skipped: {stats['skipped']} | "
                    f"Failed: {stats['failed']}"
                )

    logger.info(f"\nComplete!")
    logger.info(f"  New downloads: {stats['success']}")
    logger.info(f"  Already existed: {stats['exists']}")
    logger.info(f"  Skipped (not public domain / no image): {stats['skipped']}")
    logger.info(f"  Failed: {stats['failed']}")


def list_departments():
    """Fetch and display all departments."""
    try:
        response = make_request(DEPARTMENTS_ENDPOINT, timeout=30)
        response.raise_for_status()
        data = response.json()
        print("\nThe Met Museum Departments:\n")
        for dept in data.get("departments", []):
            print(f"  {dept['departmentId']:2d}: {dept['displayName']}")
        print()
    except Exception as e:
        logger.error(f"Failed to fetch departments: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Download public domain artworks from The Metropolitan Museum of Art"
    )
    parser.add_argument(
        '-o', '--output',
        type=Path,
        default=Path('./data'),
        help='Output directory for downloads (default: ./data)'
    )
    parser.add_argument(
        '-s', '--size',
        choices=['primary', 'small'],
        default='primary',
        help='Image size: "primary" (full res) or "small" (default: primary)'
    )
    parser.add_argument(
        '-d', '--departments',
        type=str,
        default=None,
        help='Department IDs (comma-separated, e.g., "11,19" for Paintings and Photographs)'
    )
    parser.add_argument(
        '-w', '--workers',
        type=int,
        default=8,
        help='Number of parallel download workers (default: 8)'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=0.05,
        help='Delay between API requests in seconds (default: 0.05)'
    )
    parser.add_argument(
        '-l', '--limit',
        type=int,
        default=None,
        help='Limit number of objects to process'
    )
    parser.add_argument(
        '--additional-images',
        action='store_true',
        help='Also download additional images (up to 5 per object)'
    )
    parser.add_argument(
        '--list-departments',
        action='store_true',
        help='List all departments and exit'
    )
    parser.add_argument(
        '--list-only',
        action='store_true',
        help='Only list object IDs, do not download'
    )
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Do not use cached object IDs'
    )
    parser.add_argument(
        '--proxy-file',
        type=str,
        default=None,
        help='File containing proxy list (one per line: ip:port)'
    )
    parser.add_argument(
        '--use-proxies',
        action='store_true',
        help='Use rotating proxies (fetches free proxy list if no --proxy-file)'
    )
    parser.add_argument(
        '--no-proxies',
        action='store_true',
        help='Disable proxy rotation even if previously enabled'
    )

    args = parser.parse_args()

    # Initialize proxy manager
    global proxy_manager
    if args.use_proxies and not args.no_proxies:
        logger.info("Initializing proxy rotation...")
        proxy_manager = ProxyManager(proxy_file=args.proxy_file)
        if not proxy_manager.has_proxies():
            logger.warning("No proxies available, continuing without proxy rotation")
            proxy_manager = None

    if args.list_departments:
        list_departments()
        return

    # Parse department IDs
    department_ids = None
    if args.departments:
        department_ids = [int(d.strip()) for d in args.departments.split(',')]
        logger.info(f"Filtering by departments: {department_ids}")

    # Setup directories
    images_dir = args.output / "images"
    metadata_dir = args.output / "metadata"
    cache_dir = args.output / "cache"

    for d in [images_dir, metadata_dir, cache_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Get object IDs
    object_ids = get_public_domain_object_ids(
        cache_dir=cache_dir,
        department_ids=department_ids,
        use_cache=not args.no_cache
    )

    if args.list_only:
        count = 0
        for obj_id in object_ids:
            print(obj_id)
            count += 1
            if args.limit and count >= args.limit:
                break
        print(f"\nTotal: {count} object IDs")
        return

    # Download artworks
    download_artworks(
        object_ids=object_ids,
        images_dir=images_dir,
        metadata_dir=metadata_dir,
        size=args.size,
        max_workers=args.workers,
        delay=args.delay,
        limit=args.limit,
        download_additional=args.additional_images
    )


if __name__ == '__main__':
    main()
