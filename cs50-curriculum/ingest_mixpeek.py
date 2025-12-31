#!/usr/bin/env python3
"""
Mixpeek Ingestion Script for CS50 Course Curriculum

This script:
1. Creates buckets for lecture videos and PDF slides with metadata schemas
2. Creates collections with video and document feature extractors
3. Uploads sample lectures as bucket objects
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
DATA_DIR = Path(__file__).parent / "data"


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


def create_lecture_video_bucket():
    """Create a bucket for CS50 lecture videos with metadata schema."""
    print("\n=== Creating Lecture Video Bucket ===")

    bucket_data = {
        "bucket_name": "cs50-lectures",
        "description": "CS50 lecture videos with metadata",
        "bucket_schema": {
            "properties": {
                "video": {
                    "type": "video",
                    "required": True,
                    "description": "Lecture video file"
                },
                "lecture_id": {
                    "type": "integer",
                    "required": True,
                    "description": "Lecture number (0-9)"
                },
                "title": {
                    "type": "text",
                    "required": True,
                    "description": "Lecture title"
                },
                "description": {
                    "type": "text",
                    "description": "Lecture description"
                },
                "topics": {
                    "type": "json",
                    "description": "Topics covered in lecture"
                },
                "is_topic_video": {
                    "type": "boolean",
                    "description": "Whether this is a topic short (true) or main lecture (false)"
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

    print("Bucket may already exist, trying to list...")
    list_response = api_request("POST", "/v1/buckets/list", {})
    if list_response.status_code == 200:
        buckets = list_response.json()
        for bucket in buckets.get("results", []):
            if bucket.get("bucket_name") == "cs50-lectures":
                print(f"Using existing bucket: {bucket.get('bucket_id')}")
                return bucket

    return response.json() if response.status_code < 400 else None


def create_slides_bucket():
    """Create a bucket for CS50 PDF slides with metadata schema."""
    print("\n=== Creating Slides Bucket ===")

    bucket_data = {
        "bucket_name": "cs50-slides",
        "description": "CS50 lecture slides (PDF) with metadata",
        "bucket_schema": {
            "properties": {
                "document": {
                    "type": "document",
                    "required": True,
                    "description": "PDF slide file"
                },
                "lecture_id": {
                    "type": "integer",
                    "required": True,
                    "description": "Lecture number (0-9)"
                },
                "title": {
                    "type": "text",
                    "required": True,
                    "description": "Lecture title"
                },
                "description": {
                    "type": "text",
                    "description": "Slide content description"
                },
                "topics": {
                    "type": "json",
                    "description": "Topics covered in slides"
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

    print("Bucket may already exist, trying to list...")
    list_response = api_request("POST", "/v1/buckets/list", {})
    if list_response.status_code == 200:
        buckets = list_response.json()
        for bucket in buckets.get("results", []):
            if bucket.get("bucket_name") == "cs50-slides":
                print(f"Using existing bucket: {bucket.get('bucket_id')}")
                return bucket

    return response.json() if response.status_code < 400 else None


def create_lecture_collection(bucket_id):
    """Create a collection with video feature extractor for lectures."""
    print("\n=== Creating Lecture Video Collection ===")

    collection_data = {
        "collection_name": "cs50-lectures-collection",
        "description": "Searchable CS50 lectures with video embeddings",
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
                {"source_path": "metadata.lecture_id"},
                {"source_path": "metadata.title"},
                {"source_path": "metadata.description"},
                {"source_path": "metadata.topics"},
                {"source_path": "metadata.is_topic_video"},
                {"source_path": "metadata.source"}
            ]
        }
    }

    response = api_request("POST", "/v1/collections", collection_data)

    if response.status_code in (200, 201):
        collection = response.json()
        print(f"Created collection: {collection.get('collection_id')}")
        return collection

    print("Collection may already exist, trying to list...")
    list_response = api_request("POST", "/v1/collections/list", {})
    if list_response.status_code == 200:
        collections = list_response.json()
        for collection in collections.get("results", []):
            if collection.get("collection_name") == "cs50-lectures-collection":
                print(f"Using existing collection: {collection.get('collection_id')}")
                return collection

    return response.json() if response.status_code < 400 else None


def create_slides_collection(bucket_id):
    """Create a collection with document feature extractor for slides."""
    print("\n=== Creating Slides Collection ===")

    collection_data = {
        "collection_name": "cs50-slides-collection",
        "description": "Searchable CS50 slides with document embeddings",
        "source": {
            "type": "bucket",
            "bucket_ids": [bucket_id]
        },
        "feature_extractor": {
            "feature_extractor_name": "multimodal_extractor",
            "version": "v1",
            "input_mappings": {
                "document": "document"
            },
            "field_passthrough": [
                {"source_path": "metadata.lecture_id"},
                {"source_path": "metadata.title"},
                {"source_path": "metadata.description"},
                {"source_path": "metadata.topics"},
                {"source_path": "metadata.source"}
            ]
        }
    }

    response = api_request("POST", "/v1/collections", collection_data)

    if response.status_code in (200, 201):
        collection = response.json()
        print(f"Created collection: {collection.get('collection_id')}")
        return collection

    print("Collection may already exist, trying to list...")
    list_response = api_request("POST", "/v1/collections/list", {})
    if list_response.status_code == 200:
        collections = list_response.json()
        for collection in collections.get("results", []):
            if collection.get("collection_name") == "cs50-slides-collection":
                print(f"Using existing collection: {collection.get('collection_id')}")
                return collection

    return response.json() if response.status_code < 400 else None


def get_lecture_samples(n=3):
    """Get n random lecture video/metadata pairs from the data directory."""
    print(f"\n=== Getting {n} Random Lecture Samples ===")

    samples = []

    # Find all lecture directories
    lecture_dirs = sorted(DATA_DIR.glob("lecture_*"))

    if not lecture_dirs:
        print("No lecture directories found. Run download_cs50.py first.")
        return []

    # Select random lectures
    selected_dirs = random.sample(lecture_dirs, min(n, len(lecture_dirs)))

    for lecture_dir in selected_dirs:
        # Find JSON metadata file
        json_files = list(lecture_dir.glob("lecture_*.json"))
        if not json_files:
            continue

        json_path = json_files[0]
        with open(json_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        # Find main lecture video
        video_file = metadata.get("files", {}).get("video")
        if not video_file:
            print(f"  No video in {lecture_dir.name}")
            continue

        video_path = lecture_dir / video_file
        if not video_path.exists():
            print(f"  Video not found: {video_path}")
            continue

        samples.append({
            "video_path": video_path,
            "json_path": json_path,
            "metadata": metadata,
            "is_topic": False
        })
        print(f"  Selected: Lecture {metadata.get('lecture_id')}: {metadata.get('title')}")

    return samples


def get_slide_samples(n=3):
    """Get n random PDF slide/metadata pairs from the data directory."""
    print(f"\n=== Getting {n} Random Slide Samples ===")

    samples = []

    # Find all lecture directories
    lecture_dirs = sorted(DATA_DIR.glob("lecture_*"))

    if not lecture_dirs:
        print("No lecture directories found. Run download_cs50.py first.")
        return []

    # Select random lectures
    selected_dirs = random.sample(lecture_dirs, min(n, len(lecture_dirs)))

    for lecture_dir in selected_dirs:
        # Find JSON metadata file
        json_files = list(lecture_dir.glob("lecture_*.json"))
        if not json_files:
            continue

        json_path = json_files[0]
        with open(json_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        # Find PDF file
        pdf_file = metadata.get("files", {}).get("pdf")
        if not pdf_file:
            print(f"  No PDF in {lecture_dir.name}")
            continue

        pdf_path = lecture_dir / pdf_file
        if not pdf_path.exists():
            print(f"  PDF not found: {pdf_path}")
            continue

        samples.append({
            "pdf_path": pdf_path,
            "json_path": json_path,
            "metadata": metadata
        })
        print(f"  Selected: Lecture {metadata.get('lecture_id')}: {metadata.get('title')}")

    return samples


def upload_lecture_objects(bucket_id, samples):
    """Upload lecture video samples as bucket objects."""
    print("\n=== Uploading Lecture Video Objects ===")

    object_ids = []

    for sample in samples:
        metadata = sample["metadata"]
        video_path = sample["video_path"]
        file_size = video_path.stat().st_size

        # Skip very large files (>200MB) for demo
        if file_size > 200 * 1024 * 1024:
            print(f"  Skipping large file: {video_path.name} ({file_size / 1024 / 1024:.1f}MB)")
            continue

        print(f"  Processing: Lecture {metadata.get('lecture_id')}: {metadata.get('title')}...")

        # Request presigned URL
        upload_request = {
            "filename": video_path.name,
            "content_type": "video/mp4",
            "file_size_bytes": file_size,
            "blob_property": "video",
            "create_object_on_confirm": True,
            "object_metadata": {
                "lecture_id": metadata.get("lecture_id"),
                "title": metadata.get("title"),
                "description": metadata.get("description"),
                "topics": metadata.get("topics", []),
                "is_topic_video": sample.get("is_topic", False),
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
        print(f"    Uploading {file_size / (1024*1024):.1f}MB...")
        with open(video_path, "rb") as f:
            video_bytes = f.read()

        put_response = requests.put(
            presigned_url,
            data=video_bytes,
            headers={"Content-Type": "video/mp4"},
            timeout=600
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


def upload_slide_objects(bucket_id, samples):
    """Upload PDF slide samples as bucket objects."""
    print("\n=== Uploading Slide Objects ===")

    object_ids = []

    for sample in samples:
        metadata = sample["metadata"]
        pdf_path = sample["pdf_path"]
        file_size = pdf_path.stat().st_size

        print(f"  Processing: Lecture {metadata.get('lecture_id')}: {metadata.get('title')}...")

        # Request presigned URL
        upload_request = {
            "filename": pdf_path.name,
            "content_type": "application/pdf",
            "file_size_bytes": file_size,
            "blob_property": "document",
            "create_object_on_confirm": True,
            "object_metadata": {
                "lecture_id": metadata.get("lecture_id"),
                "title": metadata.get("title"),
                "description": metadata.get("description"),
                "topics": metadata.get("topics", []),
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
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        put_response = requests.put(
            presigned_url,
            data=pdf_bytes,
            headers={"Content-Type": "application/pdf"},
            timeout=120
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
    print("Mixpeek CS50 Course Curriculum Ingestion")
    print("=" * 60)

    if API_KEY == "YOUR_API_KEY_HERE":
        print("\nERROR: Please update API_KEY and NAMESPACE in this script")
        print("Get your API key from: https://mixpeek.com")
        return

    if not DATA_DIR.exists():
        print(f"\nERROR: Data directory not found: {DATA_DIR}")
        print("Run download_cs50.py first to download course materials")
        return

    results = {}

    # ===== LECTURE VIDEO INGESTION =====
    print("\n" + "=" * 60)
    print("LECTURE VIDEO INGESTION")
    print("=" * 60)

    video_bucket = create_lecture_video_bucket()
    if video_bucket:
        video_bucket_id = video_bucket.get("bucket_id")
        print(f"Using video bucket: {video_bucket_id}")

        video_collection = create_lecture_collection(video_bucket_id)
        if video_collection:
            results["video_collection_id"] = video_collection.get("collection_id")

        video_samples = get_lecture_samples(3)
        if video_samples:
            video_object_ids = upload_lecture_objects(video_bucket_id, video_samples)
            if video_object_ids:
                video_result = create_and_submit_batch(video_bucket_id, video_object_ids, "Lecture Video")
                if video_result:
                    results["video_task_id"] = video_result.get("task_id")
                    results["video_objects"] = len(video_object_ids)

    # ===== SLIDE INGESTION =====
    print("\n" + "=" * 60)
    print("SLIDE INGESTION")
    print("=" * 60)

    slides_bucket = create_slides_bucket()
    if slides_bucket:
        slides_bucket_id = slides_bucket.get("bucket_id")
        print(f"Using slides bucket: {slides_bucket_id}")

        slides_collection = create_slides_collection(slides_bucket_id)
        if slides_collection:
            results["slides_collection_id"] = slides_collection.get("collection_id")

        slide_samples = get_slide_samples(3)
        if slide_samples:
            slide_object_ids = upload_slide_objects(slides_bucket_id, slide_samples)
            if slide_object_ids:
                slide_result = create_and_submit_batch(slides_bucket_id, slide_object_ids, "Slides")
                if slide_result:
                    results["slides_task_id"] = slide_result.get("task_id")
                    results["slides_objects"] = len(slide_object_ids)

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
        if results.get("slides_task_id"):
            print(f"  GET /v1/tasks/{results['slides_task_id']}")
    else:
        print("\nNo resources were created")


if __name__ == "__main__":
    main()
