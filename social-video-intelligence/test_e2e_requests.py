#!/usr/bin/env python3
"""
E2E Test for Social Video Intelligence Showcase

Uses requests directly to avoid any SDK import issues.
Tests the complete flow against the Mixpeek production API.
"""

import json
import os
import time
import requests

# API Configuration - use environment variable
API_KEY = os.environ.get("MIXPEEK_API_KEY")
API_BASE = os.environ.get("MIXPEEK_API_BASE", "https://api.mixpeek.com")

if not API_KEY:
    print("Error: MIXPEEK_API_KEY environment variable is required")
    print("Set it with: export MIXPEEK_API_KEY=your_api_key")
    exit(1)


def api_call(method, endpoint, data=None, namespace_id=None, timeout=60, debug=False):
    """Make an API call."""
    url = f"{API_BASE}{endpoint}"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    if namespace_id:
        headers["X-Namespace"] = namespace_id

    try:
        if method == "GET":
            resp = requests.get(url, headers=headers, timeout=timeout)
        elif method == "POST":
            resp = requests.post(url, headers=headers, json=data, timeout=timeout)
        elif method == "DELETE":
            resp = requests.delete(url, headers=headers, timeout=timeout)
        else:
            return {"error": f"Unknown method: {method}"}

        if debug:
            print(f"  [DEBUG] Status: {resp.status_code}")
            print(f"  [DEBUG] Response: {resp.text[:500]}")

        if resp.status_code in [200, 201]:
            return resp.json()
        else:
            return {"error": f"{resp.status_code}: {resp.text[:500]}"}
    except Exception as e:
        return {"error": str(e)}


def test_api_connection():
    """Test basic API connection."""
    print("\n" + "=" * 60)
    print("TEST 1: API Connection")
    print("=" * 60)

    # Use POST to /v1/namespaces/list (not GET to /v1/namespaces)
    result = api_call("POST", "/v1/namespaces/list", {})
    if result.get("error"):
        print(f"✗ API Error: {result['error']}")
        return False

    namespaces = result.get("namespaces", result.get("results", []))
    print(f"✓ Connected to Mixpeek API")
    print(f"  Found {len(namespaces)} existing namespaces")
    return True


def test_create_namespace():
    """Test creating a namespace."""
    print("\n" + "=" * 60)
    print("TEST 2: Create Namespace")
    print("=" * 60)

    namespace_name = f"svi_e2e_test_{int(time.time())}"

    result = api_call("POST", "/v1/namespaces", {
        "namespace_name": namespace_name,
        "description": "Social Video Intelligence E2E Test",
        "feature_extractors": [
            {
                "feature_extractor_name": "multimodal_extractor",
                "version": "v1",
            },
            {
                "feature_extractor_name": "text_extractor",
                "version": "v1",
            }
        ]
    })

    if result.get("error"):
        print(f"✗ Failed: {result['error']}")
        return None

    namespace_id = result.get("namespace_id")
    print(f"✓ Created namespace: {namespace_id}")
    return namespace_id


def test_create_bucket(namespace_id):
    """Test creating a bucket."""
    print("\n" + "=" * 60)
    print("TEST 3: Create Video Bucket")
    print("=" * 60)

    result = api_call("POST", "/v1/buckets", {
        "bucket_name": "test_videos",
        "description": "Test video bucket for E2E",
        "bucket_schema": {
            "properties": {
                "video": {"type": "video", "required": True},
                "title": {"type": "string"},
                "platform": {"type": "string"},
                "region": {"type": "string"},
            }
        }
    }, namespace_id)

    if result.get("error"):
        print(f"✗ Failed: {result['error']}")
        return None

    bucket_id = result.get("bucket_id")
    print(f"✓ Created bucket: {bucket_id}")
    return bucket_id


def test_create_collection(namespace_id, bucket_id):
    """Test creating a collection with multimodal extractor."""
    print("\n" + "=" * 60)
    print("TEST 4: Create Collection (Visual + Audio)")
    print("=" * 60)

    result = api_call("POST", "/v1/collections", {
        "collection_name": "video_analysis",
        "description": "Video analysis with multimodal extractor",
        "source": {"type": "bucket", "bucket_ids": [bucket_id]},
        "feature_extractor": {
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
                "run_video_description": True,
                "enable_thumbnails": True,
                "description_prompt": "Describe this video segment. Note any visible brands, products, people, or text. Describe the emotional tone."
            },
            "field_passthrough": [
                {"source_path": "metadata.title"},
                {"source_path": "metadata.platform"},
                {"source_path": "metadata.region"},
            ],
        }
    }, namespace_id)

    if result.get("error"):
        print(f"✗ Failed: {result['error']}")
        return None

    collection_id = result.get("collection_id")
    print(f"✓ Created collection: {collection_id}")
    return collection_id


def test_upload_video(namespace_id, bucket_id):
    """Test uploading a video."""
    print("\n" + "=" * 60)
    print("TEST 5: Upload Video Object")
    print("=" * 60)

    # Short sample video
    video_url = "https://storage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4"

    # Correct format: blobs with property, type, and data (not url)
    result = api_call("POST", f"/v1/buckets/{bucket_id}/objects", {
        "blobs": [{"property": "video", "type": "video", "data": video_url}],
        "title": "For Bigger Blazes - Action Trailer",
        "platform": "youtube",
        "region": "US",
    }, namespace_id, debug=True)

    # Check for actual error (not just the presence of "error" key)
    if result.get("error"):
        print(f"✗ Failed: {result['error']}")
        return None

    object_id = result.get("object_id")
    if not object_id:
        print(f"✗ No object_id in response")
        print(f"  Response: {json.dumps(result, indent=2)[:300]}")
        return None

    print(f"✓ Uploaded video: {object_id}")
    print(f"  URL: {video_url}")
    return object_id


def test_create_and_submit_batch(namespace_id, bucket_id, object_ids):
    """Test creating and submitting a batch."""
    print("\n" + "=" * 60)
    print("TEST 6: Create and Submit Batch")
    print("=" * 60)

    # Create batch
    result = api_call("POST", f"/v1/buckets/{bucket_id}/batches", {
        "object_ids": object_ids
    }, namespace_id)

    if result.get("error"):
        print(f"✗ Failed to create batch: {result['error']}")
        return None

    batch_id = result.get("batch_id")
    print(f"✓ Created batch: {batch_id}")

    # Submit batch
    result = api_call("POST", f"/v1/buckets/{bucket_id}/batches/{batch_id}/submit", {}, namespace_id)

    if result.get("error"):
        print(f"✗ Failed to submit batch: {result['error']}")
        return batch_id

    task_id = result.get("task_id")
    print(f"✓ Submitted batch, task: {task_id}")
    return batch_id


def test_create_retriever(namespace_id, collection_id):
    """Test creating a retriever."""
    print("\n" + "=" * 60)
    print("TEST 7: Create Retriever")
    print("=" * 60)

    # Note: collection_ids is at RETRIEVER level (not inside stage)
    result = api_call("POST", "/v1/retrievers", {
        "retriever_name": "video_search",
        "description": "Search video content by visual and audio",
        "collection_identifiers": [collection_id],  # Collections at retriever level
        "input_schema": {
            "query": {"type": "string", "required": True, "description": "Search query"}
        },
        "stages": [
            {
                "stage_name": "visual_search",
                "stage_id": "feature_search",  # Built-in stage type
                "parameters": {
                    "searches": [
                        {
                            "feature_uri": "mixpeek://multimodal_extractor@v1/vertex_multimodal_embedding",
                            "query": {
                                "input_mode": "text",
                                "text": "{{INPUT.query}}"
                            },
                            "top_k": 20,
                            "min_score": 0.3
                        }
                    ],
                    "final_top_k": 10
                }
            }
        ]
    }, namespace_id, debug=True)

    if result.get("error"):
        print(f"✗ Failed: {result['error']}")
        return None

    # Handle nested response structure
    if "retriever" in result:
        retriever_id = result["retriever"].get("retriever_id")
    else:
        retriever_id = result.get("retriever_id")
    print(f"✓ Created retriever: {retriever_id}")
    return retriever_id


def test_execute_retriever(namespace_id, retriever_id, query="action explosion movie"):
    """Test executing a retriever."""
    print("\n" + "=" * 60)
    print(f"TEST 8: Execute Retriever (query: '{query}')")
    print("=" * 60)

    result = api_call("POST", f"/v1/retrievers/{retriever_id}/execute", {
        "inputs": {"query": query},
        "settings": {"limit": 10}
    }, namespace_id)

    if result.get("error"):
        print(f"✗ Failed: {result['error']}")
        return 0

    documents = result.get("documents", result.get("results", []))
    print(f"✓ Retriever executed, found {len(documents)} results")

    for i, doc in enumerate(documents[:5], 1):
        title = doc.get("title", "N/A")
        score = doc.get("score", 0)
        start = doc.get("start_time", "?")
        end = doc.get("end_time", "?")
        print(f"  {i}. [{score:.3f}] {title} ({start}s-{end}s)")
        if doc.get("description"):
            desc = doc["description"][:100] + "..." if len(doc.get("description", "")) > 100 else doc.get("description", "")
            print(f"     {desc}")

    return len(documents)


def test_create_taxonomy_resources(namespace_id):
    """Test creating taxonomy reference data (brands/sentiment)."""
    print("\n" + "=" * 60)
    print("TEST 9: Create Taxonomy Reference Data")
    print("=" * 60)

    # Create bucket for brands
    result = api_call("POST", "/v1/buckets", {
        "bucket_name": "brand_reference",
        "description": "Brand reference for taxonomy",
        "bucket_schema": {
            "properties": {
                "brand_text": {"type": "text", "required": True},
                "brand_id": {"type": "string"},
                "brand_name": {"type": "string"},
                "brand_category": {"type": "string"},
            }
        }
    }, namespace_id)

    if result.get("error"):
        print(f"✗ Failed to create brand bucket: {result['error']}")
        return None, None

    brand_bucket_id = result.get("bucket_id")
    print(f"✓ Created brand bucket: {brand_bucket_id}")

    # Create collection
    result = api_call("POST", "/v1/collections", {
        "collection_name": "brand_reference",
        "description": "Brand embeddings for semantic matching",
        "source": {"type": "bucket", "bucket_ids": [brand_bucket_id]},
        "feature_extractor": {
            "feature_extractor_name": "text_extractor",
            "version": "v1",
            "input_mappings": {"text": "brand_text"},
        }
    }, namespace_id)

    if result.get("error"):
        print(f"✗ Failed to create brand collection: {result['error']}")
        return brand_bucket_id, None

    brand_collection_id = result.get("collection_id")
    print(f"✓ Created brand collection: {brand_collection_id}")

    # Upload brand data
    brands = [
        ("brand_nike", "Nike", "tracked", "Nike swoosh logo athletic sportswear shoes sneakers just do it"),
        ("brand_apple", "Apple", "tracked", "Apple logo iPhone iPad Mac technology electronics smartphone"),
        ("brand_starbucks", "Starbucks", "tracked", "Starbucks green mermaid logo coffee cafe latte espresso"),
    ]

    object_ids = []
    for brand_id, brand_name, category, text in brands:
        result = api_call("POST", f"/v1/buckets/{brand_bucket_id}/objects", {
            "blobs": [{"property": "brand_text", "type": "text", "data": text}],
            "metadata": {
                "brand_id": brand_id,
                "brand_name": brand_name,
                "brand_category": category,
            }
        }, namespace_id)
        if "error" not in result:
            object_ids.append(result.get("object_id"))
            print(f"  ✓ Uploaded: {brand_name}")

    # Submit batch
    if object_ids:
        batch_result = api_call("POST", f"/v1/buckets/{brand_bucket_id}/batches", {
            "object_ids": object_ids
        }, namespace_id)
        if "error" not in batch_result:
            batch_id = batch_result.get("batch_id")
            api_call("POST", f"/v1/buckets/{brand_bucket_id}/batches/{batch_id}/submit", {}, namespace_id)
            print(f"✓ Submitted brand batch: {batch_id}")

    return brand_bucket_id, brand_collection_id


def test_create_cluster(namespace_id, collection_id):
    """Test creating a cluster for narrative discovery."""
    print("\n" + "=" * 60)
    print("TEST 10: Create Cluster (Narrative Discovery)")
    print("=" * 60)

    result = api_call("POST", "/v1/clusters", {
        "cluster_name": "narrative_discovery",
        "description": "Discover emerging narratives in video content",
        "collection_ids": [collection_id],
        "vector_config": {
            "feature_uri": "mixpeek://multimodal_extractor@v1/vertex_multimodal_embedding",
            "clustering_method": "kmeans",
            "num_clusters": 5,
        }
    }, namespace_id)

    if result.get("error"):
        print(f"✗ Failed: {result['error']}")
        return None

    # Handle nested response structure
    if "cluster" in result:
        cluster_id = result["cluster"].get("cluster_id")
    else:
        cluster_id = result.get("cluster_id")
    print(f"✓ Created cluster: {cluster_id}")
    return cluster_id


def run_e2e_test():
    """Run the complete E2E test."""
    print("\n" + "=" * 70)
    print("  SOCIAL VIDEO INTELLIGENCE - END-TO-END TEST")
    print("=" * 70)
    print(f"\nAPI Base: {API_BASE}")
    print(f"API Key: {API_KEY[:20]}...")

    namespace_id = None

    try:
        # Test 1: API Connection
        if not test_api_connection():
            print("\n[FAIL] Cannot proceed without API connection")
            return False

        # Test 2: Create Namespace
        namespace_id = test_create_namespace()
        if not namespace_id:
            return False

        # Test 3: Create Bucket
        bucket_id = test_create_bucket(namespace_id)
        if not bucket_id:
            return False

        # Test 4: Create Collection
        collection_id = test_create_collection(namespace_id, bucket_id)
        if not collection_id:
            return False

        # Test 5: Upload Video
        object_id = test_upload_video(namespace_id, bucket_id)
        if not object_id:
            return False

        # Test 6: Create and Submit Batch
        batch_id = test_create_and_submit_batch(namespace_id, bucket_id, [object_id])
        if not batch_id:
            return False

        # Test 7: Create Retriever
        retriever_id = test_create_retriever(namespace_id, collection_id)
        if not retriever_id:
            return False

        # Wait for processing
        print("\n" + "=" * 60)
        print("Waiting 45 seconds for video processing...")
        print("(Video will be split into segments, embedded, transcribed)")
        print("=" * 60)
        for i in range(45):
            print(f"\r  {i+1}/45 seconds...", end="", flush=True)
            time.sleep(1)
        print(" Done!")

        # Test 8: Execute Retriever
        result_count = test_execute_retriever(namespace_id, retriever_id, "action explosion fire")

        # Test 9: Create Taxonomy Resources
        brand_bucket_id, brand_collection_id = test_create_taxonomy_resources(namespace_id)

        # Test 10: Create Cluster
        cluster_id = test_create_cluster(namespace_id, collection_id)

        # Summary
        print("\n" + "=" * 70)
        print("  E2E TEST SUMMARY")
        print("=" * 70)
        print(f"""
[bold]Resources Created:[/bold]
  Namespace:          {namespace_id}
  Video Bucket:       {bucket_id}
  Video Collection:   {collection_id}
  Video Object:       {object_id}
  Batch:              {batch_id}
  Retriever:          {retriever_id}
  Brand Bucket:       {brand_bucket_id}
  Brand Collection:   {brand_collection_id}
  Cluster:            {cluster_id}

[bold]Search Results:[/bold] {result_count} documents found

[bold]Status:[/bold] {'SUCCESS' if result_count > 0 else 'PROCESSING (results may appear shortly)'}

[bold]Notes:[/bold]
  - Namespace kept for further testing
  - Video processing continues in background
  - Run more queries with: python cli.py search "your query"
  - To cleanup: DELETE /v1/namespaces/{namespace_id}
        """)

        return True

    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Test cancelled by user")
        return False
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    success = run_e2e_test()
    sys.exit(0 if success else 1)
