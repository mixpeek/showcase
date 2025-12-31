#!/usr/bin/env python3
"""
Mixpeek Ingestion Script for Portrait Gallery

This script uploads all images from data/images directory with their metadata.
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
SKIP_FIRST = 0  # API now handles deduplication via skipped_unique_key
BATCH_SIZE = 50  # Files per batch (increased from 10)
DEBUG_FIRST_REQUEST = True  # Print first request payload

# Parallelism settings
MAX_WORKERS = 10  # Concurrent uploads/confirms

# Throttling (reduced - only if needed)
DELAY_BETWEEN_CONFIRMS = 0  # Removed - using parallel instead
DELAY_BETWEEN_BATCHES = 0.2  # Reduced from 1.0s

# Data directory
DATA_DIR = Path(__file__).parent / "data" / "images"

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
        "bucket_name": "portrait-gallery-4",
        "description": "National Gallery of Art portrait images with metadata",
        "bucket_schema": {
            "properties": {
                "image": {"type": "image"},
                "nga_object_id": {"type": "string"},
                "title": {"type": "string"},
                "artist": {"type": "string"},
                "date": {"type": "string"},
                "medium": {"type": "string"},
                "classification": {"type": "string"},
                "dimensions": {"type": "string"},
                "department": {"type": "string"},
                "creditline": {"type": "string"},
                "accession_number": {"type": "string"},
                "wikidata_id": {"type": "string"},
                "terms": {"type": "array"},
                "source": {"type": "object"},
                "image_info": {"type": "object"}
            }
        },
        "unique_key": {
            "fields": ["nga_object_id"],
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
            if bucket.get("bucket_name") == "portrait-gallery":
                print(f"Using existing bucket: {bucket.get('bucket_id')}")
                return bucket

    return None


def get_sample_files():
    """Get sorted list of JSON files (without loading content)."""
    print("\n=== Scanning Images ===")
    json_files = sorted(DATA_DIR.glob("*.json"))
    print(f"Found {len(json_files)} JSON metadata files")
    return json_files


def load_sample(json_path):
    """Load a single sample on demand."""
    image_path = json_path.with_suffix(".jpg")

    if not image_path.exists():
        return None

    with open(json_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    return {
        "image_path": image_path,
        "json_path": json_path,
        "metadata": metadata
    }


first_request_printed = False

def create_batch_uploads(bucket_id, samples):
    """Create batch upload requests to get presigned URLs."""
    global first_request_printed

    uploads = []
    for sample in samples:
        metadata = sample["metadata"]
        image_path = sample["image_path"]

        # Build object_metadata, filtering out None values
        object_metadata = {}

        # nga_object_id is required for unique key (map from source "object_id")
        object_metadata["nga_object_id"] = metadata.get("object_id")

        # String fields - only include if not None
        for field in ["title", "artist", "date", "medium", "classification",
                      "dimensions", "department", "creditline", "accession_number", "wikidata_id"]:
            value = metadata.get(field)
            if value is not None:
                object_metadata[field] = value

        # Array field - default to empty list
        terms = metadata.get("terms")
        object_metadata["terms"] = terms if terms is not None else []

        # Object fields - default to empty dict
        source = metadata.get("source")
        object_metadata["source"] = source if source is not None else {}

        image_info = metadata.get("image")
        object_metadata["image_info"] = image_info if image_info is not None else {}

        uploads.append({
            "filename": image_path.name,
            "content_type": "image/jpeg",
            "file_size_bytes": image_path.stat().st_size,
            "blob_property": "image",
            "blob_type": "image",
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


def upload_to_s3(presigned_url, image_path, max_retries=3):
    """Upload file to S3 using presigned URL with retry logic."""
    with open(image_path, "rb") as f:
        file_data = f.read()

    for attempt in range(max_retries):
        try:
            response = s3_session.put(
                presigned_url,
                data=file_data,
                headers={"Content-Type": "image/jpeg"},
                timeout=60
            )
            return response.status_code in (200, 201, 204)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # 1s, 2s, 4s
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
    """Upload a single file to S3 and confirm. Returns (success, skipped, title)."""
    title = (sample["metadata"].get("title") or "Unknown")[:40]

    # Check if skipped due to unique key (already exists)
    if upload_info.get("skipped_unique_key"):
        return (False, True, title, upload_info.get("existing_object_id", "?"))

    upload_id = upload_info.get("upload_id")
    presigned_url = upload_info.get("presigned_url")

    if not presigned_url:
        return (False, False, title, "no URL")

    # Upload to S3
    if upload_to_s3(presigned_url, sample["image_path"]):
        # Confirm upload
        object_id = confirm_upload(bucket_id, upload_id)
        if object_id:
            return (True, False, title, object_id)
        return (False, False, title, "confirm failed")
    return (False, False, title, "S3 failed")


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
                success, was_skipped, title, info = future.result()
                if was_skipped:
                    skipped += 1
                    print(f"  [{i}] {title}... EXISTS ({info})")
                elif success:
                    uploaded += 1
                    print(f"  [{i}] {title}... OK")
                else:
                    print(f"  [{i}] {title}... FAILED ({info})")
            except Exception as e:
                print(f"  [{i}] Error: {e}")

    return uploaded, skipped


def upload_objects(bucket_id, json_files):
    """Upload all samples in batches. Loads metadata lazily per batch."""
    total = len(json_files)
    total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"\n=== Uploading {total} Objects in {total_batches} Batches ===")

    total_uploaded = 0
    total_skipped = 0

    for batch_num in range(total_batches):
        start_idx = batch_num * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, total)
        batch_files = json_files[start_idx:end_idx]

        # Load samples for this batch only (lazy loading)
        batch_samples = []
        for json_path in batch_files:
            sample = load_sample(json_path)
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

        # Delay between batches to let MongoDB connections fully release
        if batch_num < total_batches - 1 and DELAY_BETWEEN_BATCHES > 0:
            time.sleep(DELAY_BETWEEN_BATCHES)

    return total_uploaded, total_skipped


def main():
    print("=" * 60)
    print("Mixpeek Portrait Gallery Ingestion")
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
    json_files = get_sample_files()
    if not json_files:
        print("No samples found")
        return

    # Step 2.5: Skip already-uploaded items (resume point)
    if SKIP_FIRST > 0:
        print(f"\nResuming: skipping first {SKIP_FIRST} items")
        json_files = json_files[SKIP_FIRST:]
        print(f"Remaining: {len(json_files)} items to upload")

    # Step 3: Upload objects in batches (loads metadata lazily per batch)
    total_uploaded, total_skipped = upload_objects(bucket_id, json_files)

    print("\n" + "=" * 60)
    print("COMPLETE!")
    print(f"Bucket ID: {bucket_id}")
    print(f"New uploads: {total_uploaded}")
    print(f"Already existed (skipped): {total_skipped}")
    print("=" * 60)


if __name__ == "__main__":
    main()
