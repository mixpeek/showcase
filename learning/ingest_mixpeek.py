#!/usr/bin/env python3
"""
Mixpeek Ingestion Script for CS50 Learning Content

This script uploads course materials (videos, slides, code) with metadata.
Uses batch upload API for efficiency.

Docs: https://docs.mixpeek.com
"""

import json
import os
import time
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration - Set these environment variables or update them here
API_KEY = os.environ.get("MIXPEEK_API_KEY", "your_api_key_here")
NAMESPACE = os.environ.get("MIXPEEK_NAMESPACE", "your_namespace_here")
API_BASE = "https://api.mixpeek.com"
BUCKET_ID = None  # Set to your bucket ID or leave as None to create new
BATCH_SIZE = 20  # Files per batch
DEBUG_FIRST_REQUEST = True  # Print first request payload

# Parallelism settings
MAX_WORKERS = 5  # Concurrent uploads/confirms (lower for large video files)

# Throttling
DELAY_BETWEEN_BATCHES = 0.5  # Delay between batches

# Data directories
BASE_DIR = Path(__file__).parent / "data"
VIDEOS_DIR = BASE_DIR / "videos"
SLIDES_DIR = BASE_DIR / "slides"
CODE_DIR = BASE_DIR / "code"
METADATA_DIR = BASE_DIR / "metadata"

# Reusable session for connection pooling
session = requests.Session()
session.headers.update({
    "Authorization": f"Bearer {API_KEY}",
    "X-Namespace": NAMESPACE,
    "Content-Type": "application/json"
})

# Separate session for S3 uploads (no auth headers)
s3_session = requests.Session()


def get_headers():
    """Get common headers for API requests."""
    return {
        "Authorization": f"Bearer {API_KEY}",
        "X-Namespace": NAMESPACE,
        "Content-Type": "application/json"
    }


def api_request(method, endpoint, data=None, quiet=False):
    """Make an API request and return the response."""
    url = f"{API_BASE}{endpoint}"

    if method == "GET":
        response = session.get(url, timeout=60)
    elif method == "POST":
        response = session.post(url, json=data, timeout=60)
    elif method == "DELETE":
        response = session.delete(url, timeout=60)
    else:
        raise ValueError(f"Unsupported method: {method}")

    if not quiet:
        print(f"{method} {endpoint} -> {response.status_code}")

    if response.status_code >= 400 and not quiet:
        print(f"Error: {response.text}")

    return response


def get_or_create_bucket():
    """Get existing bucket or create a new one."""
    print("\n=== Getting/Creating Bucket ===")

    bucket_data = {
        "bucket_name": "cs50-learning",
        "description": "Harvard CS50 course materials - videos, slides, and code",
        "bucket_schema": {
            "properties": {
                "video": {"type": "video"},
                "document": {"type": "document"},
                "code": {"type": "text"},
                "file_id": {"type": "string"},
                "filename": {"type": "string"},
                "lecture_number": {"type": "number"},
                "content_type": {"type": "string"},
                "category": {"type": "string"},
                "file_size": {"type": "number"},
                "format": {"type": "string"},
                "md5": {"type": "string"},
                "source_url": {"type": "string"}
            }
        },
        "unique_key": {
            "fields": ["file_id"],
            "default_policy": "upsert"
        }
    }

    response = api_request("POST", "/v1/buckets", bucket_data)

    if response.status_code in (200, 201):
        bucket = response.json()
        print(f"Created bucket: {bucket.get('bucket_id')}")
        return bucket

    # Try to get existing bucket
    print("Bucket may already exist, trying to list...")
    list_response = api_request("POST", "/v1/buckets/list", {})
    if list_response.status_code == 200:
        buckets = list_response.json()
        for bucket in buckets.get("results", []):
            if bucket.get("bucket_name") == "cs50-learning":
                print(f"Using existing bucket: {bucket.get('bucket_id')}")
                return bucket

    return None


def get_content_files():
    """Get sorted list of content files with their metadata."""
    print("\n=== Scanning Course Materials ===")

    files = []

    # Scan videos
    if VIDEOS_DIR.exists():
        for video_file in VIDEOS_DIR.glob("*.mp4"):
            metadata_file = METADATA_DIR / f"{video_file.name}.json"
            if metadata_file.exists():
                files.append({
                    "file_path": video_file,
                    "metadata_path": metadata_file,
                    "category": "video"
                })

    # Scan slides
    if SLIDES_DIR.exists():
        for slide_file in SLIDES_DIR.glob("*.pdf"):
            metadata_file = METADATA_DIR / f"{slide_file.name}.json"
            if metadata_file.exists():
                files.append({
                    "file_path": slide_file,
                    "metadata_path": metadata_file,
                    "category": "slides"
                })

    # Scan code
    if CODE_DIR.exists():
        for code_file in CODE_DIR.glob("*.zip"):
            metadata_file = METADATA_DIR / f"{code_file.name}.json"
            if metadata_file.exists():
                files.append({
                    "file_path": code_file,
                    "metadata_path": metadata_file,
                    "category": "code"
                })

    print(f"Found {len(files)} files with metadata")
    print(f"  Videos: {sum(1 for f in files if f['category'] == 'video')}")
    print(f"  Slides: {sum(1 for f in files if f['category'] == 'slides')}")
    print(f"  Code: {sum(1 for f in files if f['category'] == 'code')}")

    return sorted(files, key=lambda x: x["file_path"].name)


def load_content(file_info):
    """Load a single content file with its metadata."""
    file_path = file_info["file_path"]
    metadata_path = file_info["metadata_path"]

    if not file_path.exists():
        return None

    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    return {
        "file_path": file_path,
        "metadata": metadata,
        "category": file_info["category"]
    }


first_request_printed = False

def create_batch_uploads(bucket_id, samples):
    """Create batch upload requests to get presigned URLs."""
    global first_request_printed

    uploads = []
    for sample in samples:
        metadata = sample["metadata"]
        file_path = sample["file_path"]
        category = sample["category"]

        # Determine blob type and property based on category
        if category == "video":
            blob_type = "video"
            blob_property = "video"
            content_type = "video/mp4"
        elif category == "slides":
            blob_type = "document"
            blob_property = "document"
            content_type = "application/pdf"
        elif category == "code":
            blob_type = "text"
            blob_property = "code"
            content_type = "application/zip"
        else:
            continue  # Skip unknown types

        # Build object_metadata
        object_metadata = {
            "file_id": metadata.get("filename"),
            "filename": metadata.get("filename"),
            "lecture_number": metadata.get("lecture_number", 999),
            "content_type": content_type,
            "category": category,
            "file_size": metadata.get("size"),
            "format": metadata.get("format"),
            "md5": metadata.get("md5"),
            "source_url": metadata.get("source_url")
        }

        # Remove None values
        object_metadata = {k: v for k, v in object_metadata.items() if v is not None}

        uploads.append({
            "filename": file_path.name,
            "content_type": content_type,
            "file_size_bytes": file_path.stat().st_size,
            "blob_property": blob_property,
            "blob_type": blob_type,
            "create_object_on_confirm": True,
            "object_metadata": object_metadata
        })

    batch_request = {"uploads": uploads}

    # Debug: print first request payload
    if DEBUG_FIRST_REQUEST and not first_request_printed:
        print("\n=== DEBUG: First batch request payload ===")
        print(json.dumps(batch_request, indent=2, default=str))
        print("=== END DEBUG ===\n")
        first_request_printed = True

    # Retry with backoff for rate limits
    max_retries = 3
    for attempt in range(max_retries):
        response = api_request("POST", f"/v1/buckets/{bucket_id}/uploads/batch", batch_request)

        if response.status_code == 429:
            wait_time = 60 * (attempt + 1)  # 60s, 120s, 180s
            print(f"Rate limited. Waiting {wait_time}s before retry...")
            time.sleep(wait_time)
            continue
        elif response.status_code in (200, 201):
            return response.json().get("uploads", [])
        else:
            break  # Other error, don't retry

    return []


def upload_to_s3(presigned_url, file_path, content_type, max_retries=3):
    """Upload file to S3 using presigned URL with retry logic."""
    with open(file_path, "rb") as f:
        file_data = f.read()

    for attempt in range(max_retries):
        try:
            response = s3_session.put(
                presigned_url,
                data=file_data,
                headers={"Content-Type": content_type},
                timeout=180  # Longer timeout for large files
            )
            return response.status_code in (200, 201, 204)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # 1s, 2s, 4s
                print(f"  Retry {attempt + 1} after connection error...")
                time.sleep(wait_time)
            else:
                return False
    return False


def confirm_upload(bucket_id, upload_id):
    """Confirm upload completion."""
    response = api_request(
        "POST",
        f"/v1/buckets/{bucket_id}/uploads/{upload_id}/confirm",
        {},
        quiet=True
    )
    if response.status_code in (200, 201):
        return response.json().get("object_id")
    return None


def upload_and_confirm(bucket_id, sample, upload_info):
    """Upload a single file to S3 and confirm. Returns (success, skipped, filename)."""
    filename = sample["file_path"].name[:50]
    category = sample["category"]

    # Check if skipped due to unique key (already exists)
    if upload_info.get("skipped_unique_key"):
        return (False, True, filename, upload_info.get("existing_object_id", "?"))

    upload_id = upload_info.get("upload_id")
    presigned_url = upload_info.get("presigned_url")

    if not presigned_url:
        return (False, False, filename, "no URL")

    # Determine content type
    if category == "video":
        content_type = "video/mp4"
    elif category == "slides":
        content_type = "application/pdf"
    elif category == "code":
        content_type = "application/zip"
    else:
        content_type = "application/octet-stream"

    # Upload to S3
    if upload_to_s3(presigned_url, sample["file_path"], content_type):
        # Confirm upload
        object_id = confirm_upload(bucket_id, upload_id)
        if object_id:
            return (True, False, filename, object_id)
        return (False, False, filename, "confirm failed")
    return (False, False, filename, "S3 failed")


def process_batch(bucket_id, samples, batch_num, total_batches):
    """Process a batch of uploads in parallel."""
    print(f"\n--- Batch {batch_num}/{total_batches} ({len(samples)} files) ---")

    # Step 1: Create batch uploads to get presigned URLs
    upload_responses = create_batch_uploads(bucket_id, samples)

    if not upload_responses:
        print("Failed to create batch uploads")
        return 0, 0

    # Step 2: Upload files to S3 and confirm in parallel
    uploaded = 0
    skipped = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(upload_and_confirm, bucket_id, sample, upload_info): i
            for i, (sample, upload_info) in enumerate(zip(samples, upload_responses), 1)
        }

        for future in as_completed(futures):
            i = futures[future]
            try:
                success, was_skipped, filename, info = future.result()
                if was_skipped:
                    skipped += 1
                    print(f"  [{i}] {filename}... EXISTS ({info})")
                elif success:
                    uploaded += 1
                    print(f"  [{i}] {filename}... OK")
                else:
                    print(f"  [{i}] {filename}... FAILED ({info})")
            except Exception as e:
                print(f"  [{i}] Error: {e}")

    return uploaded, skipped


def upload_objects(bucket_id, file_infos):
    """Upload all files in batches. Loads metadata lazily per batch."""
    total = len(file_infos)
    total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"\n=== Uploading {total} Files in {total_batches} Batches ===")

    total_uploaded = 0
    total_skipped = 0

    for batch_num in range(total_batches):
        start_idx = batch_num * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, total)
        batch_infos = file_infos[start_idx:end_idx]

        # Load content for this batch only (lazy loading)
        batch_samples = []
        for file_info in batch_infos:
            sample = load_content(file_info)
            if sample:
                batch_samples.append(sample)

        if not batch_samples:
            print(f"\n--- Batch {batch_num + 1}/{total_batches} - no valid samples ---")
            continue

        uploaded, skipped = process_batch(bucket_id, batch_samples, batch_num + 1, total_batches)
        total_uploaded += uploaded
        total_skipped += skipped

        # Clear batch from memory
        del batch_samples

        print(f"Batch complete: {uploaded} uploaded, {skipped} skipped, {total_uploaded + total_skipped} processed")

        # Delay between batches
        if batch_num < total_batches - 1 and DELAY_BETWEEN_BATCHES > 0:
            time.sleep(DELAY_BETWEEN_BATCHES)

    return total_uploaded, total_skipped


def main():
    print("=" * 60)
    print("Mixpeek CS50 Learning Content Ingestion")
    print("=" * 60)

    # Step 1: Use existing bucket or create new one
    if BUCKET_ID:
        bucket_id = BUCKET_ID
        print(f"Using existing bucket: {bucket_id}")
    else:
        bucket = get_or_create_bucket()
        if not bucket:
            print("Failed to create/get bucket")
            return
        bucket_id = bucket.get("bucket_id")
        print(f"Created bucket: {bucket_id}")

    # Step 2: Get list of files (without loading content into memory)
    file_infos = get_content_files()
    if not file_infos:
        print("No content found")
        return

    # Step 3: Upload objects in batches (loads metadata lazily per batch)
    total_uploaded, total_skipped = upload_objects(bucket_id, file_infos)

    print("\n" + "=" * 60)
    print("COMPLETE!")
    print(f"Bucket ID: {bucket_id}")
    print(f"New uploads: {total_uploaded}")
    print(f"Already existed (skipped): {total_skipped}")
    print("=" * 60)


if __name__ == "__main__":
    main()
