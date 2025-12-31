#!/usr/bin/env python3
"""
Mixpeek Ingestion Script for Internet Archive Movie Trailers + Posters

This script:
1. Creates buckets for videos and posters with metadata schemas
2. Creates collections with video and image feature extractors
3. Uploads random samples as bucket objects
4. Creates and submits batches for processing

Docs: https://docs.mixpeek.com
"""

import json
import random
import requests
from pathlib import Path

# Configuration - UPDATE THESE VALUES
API_KEY = "YOUR_API_KEY_HERE"
NAMESPACE = "YOUR_NAMESPACE_HERE"
API_BASE = "https://api.mixpeek.com"

# Data directories
VIDEO_DIR = Path(__file__).parent / "data" / "videos"
POSTER_DIR = Path(__file__).parent / "data" / "posters"


def get_headers():
    """Get common headers for API requests."""
    return {
        "Authorization": f"Bearer {API_KEY}",
        "X-Namespace": NAMESPACE,
        "Content-Type": "application/json"
    }


def api_request(method, endpoint, data=None):
    """Make an API request and return the response."""
    url = f"{API_BASE}{endpoint}"
    headers = get_headers()

    if method == "GET":
        response = requests.get(url, headers=headers)
    elif method == "POST":
        response = requests.post(url, headers=headers, json=data)
    elif method == "DELETE":
        response = requests.delete(url, headers=headers)
    else:
        raise ValueError(f"Unsupported method: {method}")

    print(f"{method} {endpoint} -> {response.status_code}")

    if response.status_code >= 400:
        print(f"Error: {response.text}")

    return response


def create_video_bucket():
    """Create a bucket for movie trailers with video + metadata schema."""
    print("\n=== Creating Video Bucket ===")

    bucket_data = {
        "bucket_name": "movie-trailers",
        "description": "Internet Archive movie trailers with metadata",
        "bucket_schema": {
            "properties": {
                "video": {
                    "type": "video",
                    "required": True,
                    "description": "Movie trailer video file"
                },
                "thumbnail": {
                    "type": "image",
                    "description": "Video thumbnail image"
                },
                "title": {
                    "type": "text",
                    "required": True,
                    "description": "Trailer title"
                },
                "description": {
                    "type": "text",
                    "description": "Trailer description"
                },
                "date": {
                    "type": "text",
                    "description": "Release date"
                },
                "creator": {
                    "type": "text",
                    "description": "Creator/studio"
                },
                "subject": {
                    "type": "json",
                    "description": "Subject tags and keywords"
                },
                "runtime": {
                    "type": "text",
                    "description": "Video runtime"
                },
                "source": {
                    "type": "json",
                    "description": "Source information from Internet Archive"
                }
            }
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
            if bucket.get("bucket_name") == "movie-trailers":
                print(f"Using existing bucket: {bucket.get('bucket_id')}")
                return bucket

    return response.json() if response.status_code < 400 else None


def create_poster_bucket():
    """Create a bucket for movie posters with image + metadata schema."""
    print("\n=== Creating Poster Bucket ===")

    bucket_data = {
        "bucket_name": "movie-posters",
        "description": "Internet Archive movie posters with metadata",
        "bucket_schema": {
            "properties": {
                "image": {
                    "type": "image",
                    "required": True,
                    "description": "Movie poster image"
                },
                "title": {
                    "type": "text",
                    "description": "Poster title or movie name"
                },
                "description": {
                    "type": "text",
                    "description": "Poster description"
                },
                "creator": {
                    "type": "text",
                    "description": "Creator/studio"
                },
                "date": {
                    "type": "text",
                    "description": "Date"
                },
                "subject": {
                    "type": "json",
                    "description": "Subject tags"
                },
                "source": {
                    "type": "json",
                    "description": "Source information from Internet Archive"
                }
            }
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
            if bucket.get("bucket_name") == "movie-posters":
                print(f"Using existing bucket: {bucket.get('bucket_id')}")
                return bucket

    return response.json() if response.status_code < 400 else None


def create_video_collection(bucket_id):
    """Create a collection with video feature extractor."""
    print("\n=== Creating Video Collection ===")

    collection_data = {
        "collection_name": "movie-trailers-collection",
        "description": "Searchable movie trailers with video embeddings",
        "source": {
            "type": "bucket",
            "bucket_ids": [bucket_id]
        },
        "feature_extractor": {
            "feature_extractor_name": "multimodal_extractor",
            "version": "v1",
            "input_mappings": {
                "video": "video"
            },
            "field_passthrough": [
                {"source_path": "metadata.title"},
                {"source_path": "metadata.description"},
                {"source_path": "metadata.date"},
                {"source_path": "metadata.creator"},
                {"source_path": "metadata.subject"},
                {"source_path": "metadata.source"}
            ]
        }
    }

    response = api_request("POST", "/v1/collections", collection_data)

    if response.status_code in (200, 201):
        collection = response.json()
        print(f"Created collection: {collection.get('collection_id')}")
        return collection

    # Try to get existing collection
    print("Collection may already exist, trying to list...")
    list_response = api_request("POST", "/v1/collections/list", {})
    if list_response.status_code == 200:
        collections = list_response.json()
        for collection in collections.get("results", []):
            if collection.get("collection_name") == "movie-trailers-collection":
                print(f"Using existing collection: {collection.get('collection_id')}")
                return collection

    return response.json() if response.status_code < 400 else None


def create_poster_collection(bucket_id):
    """Create a collection with image feature extractor."""
    print("\n=== Creating Poster Collection ===")

    collection_data = {
        "collection_name": "movie-posters-collection",
        "description": "Searchable movie posters with image embeddings",
        "source": {
            "type": "bucket",
            "bucket_ids": [bucket_id]
        },
        "feature_extractor": {
            "feature_extractor_name": "multimodal_extractor",
            "version": "v1",
            "input_mappings": {
                "image": "image"
            },
            "field_passthrough": [
                {"source_path": "metadata.title"},
                {"source_path": "metadata.description"},
                {"source_path": "metadata.date"},
                {"source_path": "metadata.creator"},
                {"source_path": "metadata.subject"},
                {"source_path": "metadata.source"}
            ]
        }
    }

    response = api_request("POST", "/v1/collections", collection_data)

    if response.status_code in (200, 201):
        collection = response.json()
        print(f"Created collection: {collection.get('collection_id')}")
        return collection

    # Try to get existing collection
    print("Collection may already exist, trying to list...")
    list_response = api_request("POST", "/v1/collections/list", {})
    if list_response.status_code == 200:
        collections = list_response.json()
        for collection in collections.get("results", []):
            if collection.get("collection_name") == "movie-posters-collection":
                print(f"Using existing collection: {collection.get('collection_id')}")
                return collection

    return response.json() if response.status_code < 400 else None


def get_video_samples(n=5):
    """Get n random video/JSON pairs from the data directory."""
    print(f"\n=== Getting {n} Random Video Samples ===")

    json_files = list(VIDEO_DIR.glob("*.json"))

    if len(json_files) < n:
        print(f"Warning: Only found {len(json_files)} files")
        n = len(json_files)

    samples = random.sample(json_files, n)

    result = []
    for json_path in samples:
        with open(json_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        video_file = metadata.get("video_file")
        if not video_file:
            print(f"Warning: No video file in {json_path}")
            continue

        video_path = VIDEO_DIR / video_file
        if not video_path.exists():
            print(f"Warning: Video not found: {video_path}")
            continue

        thumb_file = metadata.get("thumbnail_file")
        thumb_path = VIDEO_DIR / thumb_file if thumb_file else None

        result.append({
            "video_path": video_path,
            "thumb_path": thumb_path if thumb_path and thumb_path.exists() else None,
            "json_path": json_path,
            "metadata": metadata
        })
        print(f"  Selected: {metadata.get('title', 'Unknown')[:50]}...")

    return result


def get_poster_samples(n=5):
    """Get n random poster/JSON pairs from the data directory."""
    print(f"\n=== Getting {n} Random Poster Samples ===")

    json_files = list(POSTER_DIR.glob("*.json"))

    if len(json_files) < n:
        print(f"Warning: Only found {len(json_files)} files")
        n = len(json_files)

    samples = random.sample(json_files, n)

    result = []
    for json_path in samples:
        with open(json_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        poster_file = metadata.get("poster_file")
        if not poster_file:
            print(f"Warning: No poster file in {json_path}")
            continue

        poster_path = POSTER_DIR / poster_file
        if not poster_path.exists():
            print(f"Warning: Poster not found: {poster_path}")
            continue

        result.append({
            "poster_path": poster_path,
            "json_path": json_path,
            "metadata": metadata
        })
        print(f"  Selected: {metadata.get('item_title', metadata.get('title', 'Unknown'))[:50]}...")

    return result


def upload_video_objects(bucket_id, samples):
    """Upload video samples as bucket objects."""
    print("\n=== Uploading Video Objects ===")

    object_ids = []

    for sample in samples:
        metadata = sample["metadata"]
        video_path = sample["video_path"]
        file_size = video_path.stat().st_size
        filename = video_path.name

        # Skip very large files (>100MB) for demo
        if file_size > 100 * 1024 * 1024:
            print(f"  Skipping large file: {filename} ({file_size / 1024 / 1024:.1f}MB)")
            continue

        print(f"  Processing: {metadata.get('title', 'Unknown')[:40]}...")

        # Request presigned URL
        upload_request = {
            "filename": filename,
            "content_type": "video/mp4",
            "file_size_bytes": file_size,
            "blob_property": "video",
            "create_object_on_confirm": True,
            "object_metadata": {
                "title": metadata.get("title"),
                "description": metadata.get("description"),
                "date": metadata.get("date"),
                "creator": metadata.get("creator"),
                "subject": metadata.get("subject", []),
                "runtime": metadata.get("runtime"),
                "source": metadata.get("source", {})
            }
        }

        response = api_request(
            "POST",
            f"/v1/buckets/{bucket_id}/uploads",
            upload_request
        )

        if response.status_code not in (200, 201):
            print(f"    Failed to get presigned URL")
            continue

        upload_data = response.json()
        presigned_url = upload_data.get("presigned_url")
        upload_id = upload_data.get("upload_id")

        # Upload file
        with open(video_path, "rb") as f:
            video_bytes = f.read()

        put_response = requests.put(
            presigned_url,
            data=video_bytes,
            headers={"Content-Type": "video/mp4"},
            timeout=300
        )

        if put_response.status_code not in (200, 201, 204):
            print(f"    Failed to upload: {put_response.status_code}")
            continue

        # Confirm upload
        confirm_response = api_request(
            "POST",
            f"/v1/buckets/{bucket_id}/uploads/{upload_id}/confirm",
            {}
        )

        if confirm_response.status_code in (200, 201):
            result = confirm_response.json()
            object_id = result.get("object_id")
            if object_id:
                object_ids.append(object_id)
                print(f"    Uploaded: {object_id}")
        else:
            print(f"    Failed to confirm upload")

    return object_ids


def upload_poster_objects(bucket_id, samples):
    """Upload poster samples as bucket objects."""
    print("\n=== Uploading Poster Objects ===")

    object_ids = []

    for sample in samples:
        metadata = sample["metadata"]
        poster_path = sample["poster_path"]
        file_size = poster_path.stat().st_size
        filename = poster_path.name

        print(f"  Processing: {metadata.get('item_title', metadata.get('title', 'Unknown'))[:40]}...")

        # Determine content type
        ext = poster_path.suffix.lower()
        content_type = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }.get(ext, 'image/jpeg')

        # Request presigned URL
        upload_request = {
            "filename": filename,
            "content_type": content_type,
            "file_size_bytes": file_size,
            "blob_property": "image",
            "create_object_on_confirm": True,
            "object_metadata": {
                "title": metadata.get("item_title") or metadata.get("title"),
                "description": metadata.get("description"),
                "date": metadata.get("date"),
                "creator": metadata.get("creator"),
                "subject": metadata.get("subject", []),
                "source": metadata.get("source", {})
            }
        }

        response = api_request(
            "POST",
            f"/v1/buckets/{bucket_id}/uploads",
            upload_request
        )

        if response.status_code not in (200, 201):
            print(f"    Failed to get presigned URL")
            continue

        upload_data = response.json()
        presigned_url = upload_data.get("presigned_url")
        upload_id = upload_data.get("upload_id")

        # Upload file
        with open(poster_path, "rb") as f:
            image_bytes = f.read()

        put_response = requests.put(
            presigned_url,
            data=image_bytes,
            headers={"Content-Type": content_type},
            timeout=60
        )

        if put_response.status_code not in (200, 201, 204):
            print(f"    Failed to upload: {put_response.status_code}")
            continue

        # Confirm upload
        confirm_response = api_request(
            "POST",
            f"/v1/buckets/{bucket_id}/uploads/{upload_id}/confirm",
            {}
        )

        if confirm_response.status_code in (200, 201):
            result = confirm_response.json()
            object_id = result.get("object_id")
            if object_id:
                object_ids.append(object_id)
                print(f"    Uploaded: {object_id}")
        else:
            print(f"    Failed to confirm upload")

    return object_ids


def create_and_submit_batch(bucket_id, object_ids, name=""):
    """Create a batch with object IDs and submit it."""
    print(f"\n=== Creating and Submitting {name} Batch ===")

    if not object_ids:
        print("No objects to batch")
        return None

    batch_data = {
        "object_ids": object_ids
    }

    response = api_request(
        "POST",
        f"/v1/buckets/{bucket_id}/batches",
        batch_data
    )

    if response.status_code not in (200, 201):
        print("Failed to create batch")
        return None

    batch = response.json()
    batch_id = batch.get("batch_id")
    print(f"Created batch: {batch_id}")

    submit_data = {
        "include_processing_history": True
    }

    response = api_request(
        "POST",
        f"/v1/buckets/{bucket_id}/batches/{batch_id}/submit",
        submit_data
    )

    if response.status_code in (200, 201, 202):
        result = response.json()
        print(f"Submitted batch: {batch_id}")
        print(f"Task ID: {result.get('task_id')}")
        return result
    else:
        print("Failed to submit batch")
        return None


def main():
    print("=" * 60)
    print("Mixpeek Movie Trailers + Posters Ingestion")
    print("=" * 60)

    if API_KEY == "YOUR_API_KEY_HERE":
        print("\nERROR: Please update API_KEY and NAMESPACE in this script")
        print("Get your API key from: https://mixpeek.com")
        return

    results = {}

    # ===== VIDEO INGESTION =====
    print("\n" + "=" * 60)
    print("VIDEO INGESTION")
    print("=" * 60)

    video_bucket = create_video_bucket()
    if video_bucket:
        video_bucket_id = video_bucket.get("bucket_id")
        print(f"Using video bucket: {video_bucket_id}")

        video_collection = create_video_collection(video_bucket_id)
        if video_collection:
            results["video_collection_id"] = video_collection.get("collection_id")

        video_samples = get_video_samples(3)  # 3 videos (they're large)
        if video_samples:
            video_object_ids = upload_video_objects(video_bucket_id, video_samples)
            if video_object_ids:
                video_result = create_and_submit_batch(video_bucket_id, video_object_ids, "Video")
                if video_result:
                    results["video_task_id"] = video_result.get("task_id")
                    results["video_objects"] = len(video_object_ids)

    # ===== POSTER INGESTION =====
    print("\n" + "=" * 60)
    print("POSTER INGESTION")
    print("=" * 60)

    poster_bucket = create_poster_bucket()
    if poster_bucket:
        poster_bucket_id = poster_bucket.get("bucket_id")
        print(f"Using poster bucket: {poster_bucket_id}")

        poster_collection = create_poster_collection(poster_bucket_id)
        if poster_collection:
            results["poster_collection_id"] = poster_collection.get("collection_id")

        poster_samples = get_poster_samples(5)
        if poster_samples:
            poster_object_ids = upload_poster_objects(poster_bucket_id, poster_samples)
            if poster_object_ids:
                poster_result = create_and_submit_batch(poster_bucket_id, poster_object_ids, "Poster")
                if poster_result:
                    results["poster_task_id"] = poster_result.get("task_id")
                    results["poster_objects"] = len(poster_object_ids)

    # ===== SUMMARY =====
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if results:
        print("\nSuccess! Resources created:")
        for key, value in results.items():
            print(f"  {key}: {value}")
        print("\nPoll task status to track progress:")
        if results.get("video_task_id"):
            print(f"  GET /v1/tasks/{results['video_task_id']}")
        if results.get("poster_task_id"):
            print(f"  GET /v1/tasks/{results['poster_task_id']}")
    else:
        print("\nNo resources were created")


if __name__ == "__main__":
    main()
