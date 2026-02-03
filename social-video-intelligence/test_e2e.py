#!/usr/bin/env python3
"""
E2E Test for Social Video Intelligence Showcase

Tests the complete flow against the Mixpeek production API.
"""

import json
import os
import sys
import time
from pathlib import Path

# API Key from environment variable
API_KEY = os.environ.get("MIXPEEK_API_KEY")
API_BASE = os.environ.get("MIXPEEK_API_BASE", "https://api.mixpeek.com")

if not API_KEY:
    print("Error: MIXPEEK_API_KEY environment variable is required")
    print("Set it with: export MIXPEEK_API_KEY=your_api_key")
    sys.exit(1)

# Use installed SDK (not local dev version which may have issues)
try:
    from mixpeek import Mixpeek, ApiException
except ImportError:
    print("Error: mixpeek SDK not installed. Run: pip install mixpeek")
    sys.exit(1)

def test_api_connection():
    """Test basic API connection."""
    print("\n" + "=" * 60)
    print("TEST 1: API Connection")
    print("=" * 60)

    try:
        client = Mixpeek(api_key=API_KEY)
        # Try listing namespaces to verify connection
        result = client.namespaces.list()
        print(f"✓ Connected to Mixpeek API")
        print(f"  Found {len(result.namespaces) if hasattr(result, 'namespaces') else 0} existing namespaces")
        return client
    except ApiException as e:
        print(f"✗ API Error: {e.status} - {e.reason}")
        return None
    except Exception as e:
        print(f"✗ Connection Error: {e}")
        return None


def test_create_namespace(client):
    """Test creating a namespace."""
    print("\n" + "=" * 60)
    print("TEST 2: Create Namespace")
    print("=" * 60)

    namespace_name = f"svi_test_{int(time.time())}"

    try:
        result = client.namespaces.create(
            namespace_name=namespace_name,
            description="Social Video Intelligence E2E Test",
        )
        namespace_id = result.namespace_id
        print(f"✓ Created namespace: {namespace_id}")
        return namespace_id
    except ApiException as e:
        print(f"✗ Failed to create namespace: {e.status} - {e.reason}")
        if hasattr(e, 'body'):
            print(f"  Body: {e.body}")
        return None
    except Exception as e:
        print(f"✗ Error: {e}")
        return None


def test_create_bucket(client, namespace_id):
    """Test creating a bucket with video schema."""
    print("\n" + "=" * 60)
    print("TEST 3: Create Video Bucket")
    print("=" * 60)

    try:
        result = client.buckets.create(
            bucket_name="test_videos",
            description="Test video bucket",
            bucket_schema={
                "properties": {
                    "video": {"type": "video", "required": True},
                    "title": {"type": "string"},
                    "platform": {"type": "string"},
                    "region": {"type": "string"},
                }
            },
            _headers={"X-Namespace": namespace_id}
        )
        bucket_id = result.bucket_id
        print(f"✓ Created bucket: {bucket_id}")
        return bucket_id
    except ApiException as e:
        print(f"✗ Failed to create bucket: {e.status} - {e.reason}")
        if hasattr(e, 'body'):
            print(f"  Body: {e.body}")
        return None
    except Exception as e:
        print(f"✗ Error: {e}")
        return None


def test_create_collection(client, namespace_id, bucket_id):
    """Test creating a collection with multimodal extractor."""
    print("\n" + "=" * 60)
    print("TEST 4: Create Collection (Visual Scenes)")
    print("=" * 60)

    try:
        result = client.collections.create(
            collection_name="visual_scenes_test",
            description="Visual scene detection test",
            source={"type": "bucket", "bucket_ids": [bucket_id]},
            feature_extractor={
                "feature_extractor_name": "multimodal_extractor",
                "version": "v1",
                "input_mappings": {"video": "video"},
                "parameters": {
                    "extractor_type": "multimodal_extractor",
                    "split_method": "time",
                    "time_split_interval": 10,
                    "run_multimodal_embedding": True,
                    "run_transcription": True,
                    "run_transcription_embedding": True,
                    "enable_thumbnails": True,
                },
                "field_passthrough": [
                    {"source_path": "metadata.title"},
                    {"source_path": "metadata.platform"},
                    {"source_path": "metadata.region"},
                ],
            },
            _headers={"X-Namespace": namespace_id}
        )
        collection_id = result.collection_id
        print(f"✓ Created collection: {collection_id}")
        return collection_id
    except ApiException as e:
        print(f"✗ Failed to create collection: {e.status} - {e.reason}")
        if hasattr(e, 'body'):
            print(f"  Body: {e.body}")
        return None
    except Exception as e:
        print(f"✗ Error: {e}")
        return None


def test_upload_video(client, namespace_id, bucket_id):
    """Test uploading a video object."""
    print("\n" + "=" * 60)
    print("TEST 5: Upload Video Object")
    print("=" * 60)

    # Using a short public domain video
    video_url = "https://storage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4"

    try:
        result = client.bucket_objects.create(
            bucket_id=bucket_id,
            blobs=[{"property": "video", "type": "url", "data": video_url}],
            metadata={
                "title": "For Bigger Blazes",
                "platform": "youtube",
                "region": "US",
            },
            _headers={"X-Namespace": namespace_id}
        )
        object_id = result.object_id
        print(f"✓ Uploaded video object: {object_id}")
        return object_id
    except ApiException as e:
        print(f"✗ Failed to upload video: {e.status} - {e.reason}")
        if hasattr(e, 'body'):
            print(f"  Body: {e.body}")
        return None
    except Exception as e:
        print(f"✗ Error: {e}")
        return None


def test_create_batch(client, namespace_id, bucket_id, object_ids):
    """Test creating and submitting a batch."""
    print("\n" + "=" * 60)
    print("TEST 6: Create and Submit Batch")
    print("=" * 60)

    try:
        # Create batch
        batch_result = client.bucket_batches.create(
            bucket_id=bucket_id,
            object_ids=object_ids,
            _headers={"X-Namespace": namespace_id}
        )
        batch_id = batch_result.batch_id
        print(f"✓ Created batch: {batch_id}")

        # Submit batch
        submit_result = client.bucket_batches.submit(
            bucket_id=bucket_id,
            batch_id=batch_id,
            _headers={"X-Namespace": namespace_id}
        )
        task_id = getattr(submit_result, 'task_id', None)
        print(f"✓ Submitted batch, task: {task_id}")
        return batch_id
    except ApiException as e:
        print(f"✗ Failed batch operation: {e.status} - {e.reason}")
        if hasattr(e, 'body'):
            print(f"  Body: {e.body}")
        return None
    except Exception as e:
        print(f"✗ Error: {e}")
        return None


def test_create_retriever(client, namespace_id, collection_id):
    """Test creating a retriever."""
    print("\n" + "=" * 60)
    print("TEST 7: Create Retriever")
    print("=" * 60)

    try:
        result = client.retrievers.create(
            retriever_name="video_search_test",
            description="Test video search retriever",
            stages=[
                {
                    "stage": "feature_search",
                    "stage_name": "visual_search",
                    "feature_extractor_name": "multimodal_extractor",
                    "collection_ids": [collection_id],
                    "query_input_key": "query",
                    "top_k": 10,
                    "min_score": 0.3,
                    "field_passthrough": ["title", "platform", "region", "description", "start_time", "end_time"],
                }
            ],
            inputs={
                "query": {"type": "text", "description": "Search query"},
            },
            _headers={"X-Namespace": namespace_id}
        )
        retriever_id = result.retriever_id
        print(f"✓ Created retriever: {retriever_id}")
        return retriever_id
    except ApiException as e:
        print(f"✗ Failed to create retriever: {e.status} - {e.reason}")
        if hasattr(e, 'body'):
            print(f"  Body: {e.body}")
        return None
    except Exception as e:
        print(f"✗ Error: {e}")
        return None


def test_execute_retriever(client, namespace_id, retriever_id, query="action explosion"):
    """Test executing a retriever."""
    print("\n" + "=" * 60)
    print(f"TEST 8: Execute Retriever (query: '{query}')")
    print("=" * 60)

    try:
        result = client.retrievers.execute(
            retriever_id=retriever_id,
            inputs={"query": query},
            settings={"limit": 5},
            _headers={"X-Namespace": namespace_id}
        )

        documents = result.documents if hasattr(result, 'documents') else []
        print(f"✓ Retriever executed, found {len(documents)} results")

        for i, doc in enumerate(documents[:3], 1):
            title = doc.title if hasattr(doc, 'title') else doc.get('title', 'N/A')
            score = doc.score if hasattr(doc, 'score') else doc.get('score', 0)
            print(f"  {i}. {title} (score: {score:.3f})")

        return len(documents) > 0
    except ApiException as e:
        print(f"✗ Failed to execute retriever: {e.status} - {e.reason}")
        if hasattr(e, 'body'):
            print(f"  Body: {e.body}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_create_taxonomy_bucket(client, namespace_id):
    """Test creating a taxonomy reference bucket."""
    print("\n" + "=" * 60)
    print("TEST 9: Create Taxonomy Reference Data")
    print("=" * 60)

    try:
        # Create bucket for sentiment labels
        bucket_result = client.buckets.create(
            bucket_name="sentiment_reference",
            description="Sentiment labels for classification",
            bucket_schema={
                "properties": {
                    "sentiment_text": {"type": "text", "required": True},
                    "sentiment_label": {"type": "string"},
                }
            },
            _headers={"X-Namespace": namespace_id}
        )
        bucket_id = bucket_result.bucket_id
        print(f"✓ Created sentiment bucket: {bucket_id}")

        # Create collection
        collection_result = client.collections.create(
            collection_name="sentiment_reference",
            description="Sentiment embeddings",
            source={"type": "bucket", "bucket_ids": [bucket_id]},
            feature_extractor={
                "feature_extractor_name": "text_extractor",
                "version": "v1",
                "input_mappings": {"text": "sentiment_text"},
            },
            _headers={"X-Namespace": namespace_id}
        )
        collection_id = collection_result.collection_id
        print(f"✓ Created sentiment collection: {collection_id}")

        # Upload sentiment labels
        sentiments = [
            ("positive", "Happy excited enthusiastic satisfied pleased joyful optimistic amazing wonderful great"),
            ("negative", "Angry frustrated disappointed critical upset pessimistic terrible awful bad horrible"),
            ("neutral", "Informational factual objective balanced descriptive neutral standard normal typical"),
        ]

        object_ids = []
        for label, text in sentiments:
            obj_result = client.bucket_objects.create(
                bucket_id=bucket_id,
                blobs=[{"property": "sentiment_text", "type": "text", "data": text}],
                metadata={"sentiment_label": label},
                _headers={"X-Namespace": namespace_id}
            )
            object_ids.append(obj_result.object_id)
            print(f"  ✓ Uploaded: {label}")

        # Process
        batch = client.bucket_batches.create(
            bucket_id=bucket_id,
            object_ids=object_ids,
            _headers={"X-Namespace": namespace_id}
        )
        client.bucket_batches.submit(
            bucket_id=bucket_id,
            batch_id=batch.batch_id,
            _headers={"X-Namespace": namespace_id}
        )
        print(f"✓ Submitted batch: {batch.batch_id}")

        return bucket_id, collection_id
    except ApiException as e:
        print(f"✗ Failed: {e.status} - {e.reason}")
        return None, None
    except Exception as e:
        print(f"✗ Error: {e}")
        return None, None


def cleanup(client, namespace_id):
    """Clean up test resources."""
    print("\n" + "=" * 60)
    print("CLEANUP: Deleting test namespace")
    print("=" * 60)

    try:
        client.namespaces.delete(namespace_id=namespace_id)
        print(f"✓ Deleted namespace: {namespace_id}")
    except Exception as e:
        print(f"✗ Cleanup failed: {e}")


def run_e2e_test():
    """Run the complete E2E test."""
    print("\n" + "=" * 60)
    print("SOCIAL VIDEO INTELLIGENCE - E2E TEST")
    print("=" * 60)

    # Track created resources for cleanup
    namespace_id = None

    try:
        # Test 1: API Connection
        client = test_api_connection()
        if not client:
            print("\n[FAIL] Cannot proceed without API connection")
            return False

        # Test 2: Create Namespace
        namespace_id = test_create_namespace(client)
        if not namespace_id:
            print("\n[FAIL] Cannot proceed without namespace")
            return False

        # Test 3: Create Bucket
        bucket_id = test_create_bucket(client, namespace_id)
        if not bucket_id:
            print("\n[FAIL] Cannot proceed without bucket")
            cleanup(client, namespace_id)
            return False

        # Test 4: Create Collection
        collection_id = test_create_collection(client, namespace_id, bucket_id)
        if not collection_id:
            print("\n[FAIL] Cannot proceed without collection")
            cleanup(client, namespace_id)
            return False

        # Test 5: Upload Video
        object_id = test_upload_video(client, namespace_id, bucket_id)
        if not object_id:
            print("\n[FAIL] Cannot proceed without video object")
            cleanup(client, namespace_id)
            return False

        # Test 6: Create and Submit Batch
        batch_id = test_create_batch(client, namespace_id, bucket_id, [object_id])
        if not batch_id:
            print("\n[FAIL] Cannot proceed without batch")
            cleanup(client, namespace_id)
            return False

        # Test 7: Create Retriever
        retriever_id = test_create_retriever(client, namespace_id, collection_id)
        if not retriever_id:
            print("\n[FAIL] Cannot proceed without retriever")
            cleanup(client, namespace_id)
            return False

        # Wait for processing
        print("\n" + "=" * 60)
        print("Waiting 30s for video processing...")
        print("=" * 60)
        time.sleep(30)

        # Test 8: Execute Retriever
        has_results = test_execute_retriever(client, namespace_id, retriever_id, "action movie trailer")

        # Test 9: Create Taxonomy Reference Data
        sentiment_bucket_id, sentiment_collection_id = test_create_taxonomy_bucket(client, namespace_id)

        # Summary
        print("\n" + "=" * 60)
        print("E2E TEST SUMMARY")
        print("=" * 60)
        print(f"""
Resources Created:
  - Namespace: {namespace_id}
  - Video Bucket: {bucket_id}
  - Video Collection: {collection_id}
  - Video Object: {object_id}
  - Batch: {batch_id}
  - Retriever: {retriever_id}
  - Sentiment Bucket: {sentiment_bucket_id}
  - Sentiment Collection: {sentiment_collection_id}

Search Results: {'YES' if has_results else 'Not yet (processing may still be running)'}
        """)

        # Ask about cleanup
        print("\n[Note] Keeping namespace for further testing.")
        print(f"  To cleanup: delete namespace {namespace_id}")

        return True

    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        if namespace_id:
            cleanup(client, namespace_id)
        return False


if __name__ == "__main__":
    success = run_e2e_test()
    sys.exit(0 if success else 1)
