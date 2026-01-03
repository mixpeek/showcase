#!/usr/bin/env python3
"""
CS50 Course Content Downloader

Downloads all course materials from Internet Archive CS50 2017 collection:
- Videos (MP4 format)
- Slides (PDF format)
- Code (ZIP archives)
- Metadata (JSON)

Source: https://archive.org/download/academictorrents_52da574b6412862e199abeaea63e51bf8cea2140
"""

import os
import json
import requests
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote
import time

# Configuration
BASE_URL = "https://archive.org/download/academictorrents_52da574b6412862e199abeaea63e51bf8cea2140"
OUTPUT_DIR = Path(__file__).parent / "data"
VIDEOS_DIR = OUTPUT_DIR / "videos"
SLIDES_DIR = OUTPUT_DIR / "slides"
CODE_DIR = OUTPUT_DIR / "code"
METADATA_DIR = OUTPUT_DIR / "metadata"
CACHE_DIR = OUTPUT_DIR / "cache"

# Download settings
MAX_WORKERS = 4
CHUNK_SIZE = 1024 * 1024  # 1MB chunks
RETRY_COUNT = 3
RETRY_DELAY = 2

# Ensure directories exist
for directory in [VIDEOS_DIR, SLIDES_DIR, CODE_DIR, METADATA_DIR, CACHE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


def get_file_list():
    """Fetch the file listing from Internet Archive."""
    print("Fetching file list from Internet Archive...")

    # Use the metadata API to get file listing
    metadata_url = "https://archive.org/metadata/academictorrents_52da574b6412862e199abeaea63e51bf8cea2140"

    try:
        response = requests.get(metadata_url, timeout=30)
        response.raise_for_status()
        data = response.json()

        # Save metadata
        metadata_path = METADATA_DIR / "archive_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"✓ Saved archive metadata to {metadata_path}")

        return data.get('files', [])

    except Exception as e:
        print(f"✗ Error fetching file list: {e}")
        return []


def categorize_files(files):
    """Organize files by type (video, slides, code)."""
    categorized = {
        'videos': [],
        'slides': [],
        'code': [],
        'other': []
    }

    for file_info in files:
        name = file_info.get('name', '')

        # Skip metadata and derivative files
        if any(x in name.lower() for x in ['.xml', '.sqlite', '_meta.', 'torrent', '.ogv', '.txt', '.djvu', 'jp2', '.abbyy']):
            continue

        # Categorize by extension
        if name.endswith('.mp4'):
            categorized['videos'].append(file_info)
        elif name.endswith('.pdf'):
            categorized['slides'].append(file_info)
        elif name.endswith('.zip'):
            categorized['code'].append(file_info)
        else:
            categorized['other'].append(file_info)

    return categorized


def parse_lecture_number(filename):
    """Extract lecture number from filename for sorting."""
    match = re.search(r'^(\d+)', filename)
    return int(match.group(1)) if match else 999


def download_file(file_info, output_dir, category):
    """Download a single file with progress tracking."""
    filename = file_info['name']
    file_url = f"{BASE_URL}/{quote(filename)}"
    output_path = output_dir / filename

    # Skip if already exists
    if output_path.exists():
        file_size = output_path.stat().st_size
        expected_size = int(file_info.get('size', 0))
        if file_size == expected_size:
            print(f"⊘ Skipping {filename} (already exists)")
            return {'status': 'skipped', 'file': filename}

    # Try downloading with retries
    for attempt in range(RETRY_COUNT):
        try:
            print(f"↓ Downloading {filename} ({file_info.get('size', 'unknown')} bytes)...")

            response = requests.get(file_url, stream=True, timeout=60)
            response.raise_for_status()

            # Download with progress
            downloaded = 0
            total_size = int(response.headers.get('content-length', 0))

            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            print(f"  Progress: {progress:.1f}%", end='\r')

            print(f"✓ Downloaded {filename} ({downloaded:,} bytes)")

            # Save metadata
            metadata_path = METADATA_DIR / f"{filename}.json"
            with open(metadata_path, 'w') as f:
                json.dump({
                    'filename': filename,
                    'category': category,
                    'size': file_info.get('size'),
                    'format': file_info.get('format'),
                    'md5': file_info.get('md5'),
                    'source_url': file_url,
                    'lecture_number': parse_lecture_number(filename)
                }, f, indent=2)

            return {'status': 'success', 'file': filename}

        except Exception as e:
            if attempt < RETRY_COUNT - 1:
                print(f"✗ Error downloading {filename}: {e}. Retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            else:
                print(f"✗ Failed to download {filename} after {RETRY_COUNT} attempts: {e}")
                return {'status': 'failed', 'file': filename, 'error': str(e)}

    return {'status': 'failed', 'file': filename}


def download_category(files, output_dir, category_name):
    """Download all files in a category."""
    if not files:
        print(f"\nNo {category_name} files found.")
        return

    print(f"\n{'='*60}")
    print(f"Downloading {len(files)} {category_name} file(s)")
    print(f"{'='*60}")

    # Sort by lecture number
    files_sorted = sorted(files, key=lambda x: parse_lecture_number(x['name']))

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(download_file, file_info, output_dir, category_name): file_info
            for file_info in files_sorted
        }

        for future in as_completed(futures):
            result = future.result()
            results.append(result)

    # Summary
    success = sum(1 for r in results if r['status'] == 'success')
    skipped = sum(1 for r in results if r['status'] == 'skipped')
    failed = sum(1 for r in results if r['status'] == 'failed')

    print(f"\n{category_name.capitalize()} Summary:")
    print(f"  ✓ Downloaded: {success}")
    print(f"  ⊘ Skipped: {skipped}")
    print(f"  ✗ Failed: {failed}")


def main():
    """Main download orchestrator."""
    print("CS50 Course Content Downloader")
    print("="*60)

    # Fetch file list
    files = get_file_list()
    if not files:
        print("No files found. Exiting.")
        return

    # Categorize files
    categorized = categorize_files(files)

    print(f"\nFound:")
    print(f"  - {len(categorized['videos'])} video files")
    print(f"  - {len(categorized['slides'])} slide files")
    print(f"  - {len(categorized['code'])} code archives")

    # Download each category
    download_category(categorized['videos'], VIDEOS_DIR, 'videos')
    download_category(categorized['slides'], SLIDES_DIR, 'slides')
    download_category(categorized['code'], CODE_DIR, 'code')

    print("\n" + "="*60)
    print("Download complete!")
    print(f"Files saved to: {OUTPUT_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
