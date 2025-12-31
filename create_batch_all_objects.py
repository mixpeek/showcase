#!/usr/bin/env python3
"""
Create a single batch of all objects in a Mixpeek bucket.

Currently using offset pagination for production API.
TODO: Switch to cursor pagination after deploying cursor fixes to production.
"""
import requests

# Configuration
API_KEY = "sk_wmqi_kbP7NfKOKxk3TNblc5JPo9TXATOJEnYlYqT9BdRlAuo6ARbD0MjoPYrV7kbQf0"
NAMESPACE = "image-galleries"
API_BASE = "https://api.mixpeek.com"  # Local dev server
BUCKET_ID = "bkt_037376e0"

def get_bucket_info(bucket_id):
    """Get bucket info to determine total object count."""
    response = requests.get(
        f"{API_BASE}/v1/buckets/{bucket_id}",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "X-Namespace": NAMESPACE,
        },
        timeout=30
    )

    if response.status_code == 200:
        return response.json()
    return None


def get_all_object_ids(bucket_id):
    """Fetch all object IDs using offset-based pagination.

    Note: Using offset pagination for production until cursor pagination is deployed.
    Production API doesn't have the cursor pagination fixes yet.
    """
    print("=== Fetching all object IDs ===")

    # Get total object count from bucket info
    bucket_info = get_bucket_info(bucket_id)
    if bucket_info:
        total_objects = bucket_info.get("object_count", 0)
        print(f"Bucket has {total_objects:,} total objects")
    else:
        total_objects = 120000  # Fallback estimate
        print(f"⚠️  Could not fetch bucket info, using estimate: {total_objects:,}")

    all_object_ids = []
    limit = 1000
    offset = 0
    page_num = 0

    # Use while loop to continue until we have all objects
    while len(all_object_ids) < total_objects:
        page_num += 1
        print(f"Fetching page {page_num} (offset {offset})...")

        response = requests.post(
            f"{API_BASE}/v1/buckets/{bucket_id}/objects/list",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "X-Namespace": NAMESPACE,
                "Content-Type": "application/json"
            },
            json={
                "limit": limit,
                "offset": offset
            },
            timeout=60
        )

        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            print(response.text)
            break

        data = response.json()
        results = data.get("results", [])

        if not results:
            print(f"  No more results - stopping")
            break

        # Extract object IDs
        object_ids = [obj["object_id"] for obj in results if "object_id" in obj]
        all_object_ids.extend(object_ids)

        print(f"  Fetched {len(object_ids)} objects (total: {len(all_object_ids)}/{total_objects})")

        # Move to next page by ACTUAL number of results (not requested limit)
        offset += len(results)

        # Progress update every 100 pages
        if page_num % 100 == 0:
            percentage = (len(all_object_ids) / total_objects) * 100
            print(f"  📊 Progress: {percentage:.1f}% complete")

    print(f"\n✓ Total object IDs fetched: {len(all_object_ids)}")
    return all_object_ids


def create_and_submit_batch(bucket_id, object_ids):
    """Create a batch and submit it."""
    print(f"\n=== Creating batch with {len(object_ids)} objects ===")

    # Create batch
    response = requests.post(
        f"{API_BASE}/v1/buckets/{bucket_id}/batches",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "X-Namespace": NAMESPACE,
            "Content-Type": "application/json"
        },
        json={"object_ids": object_ids},
        timeout=120
    )

    if response.status_code not in (200, 201):
        print(f"✗ Failed to create batch: {response.status_code}")
        print(response.text)
        return None

    batch = response.json()
    batch_id = batch.get("batch_id")
    print(f"✓ Created batch: {batch_id}")

    # Submit batch
    print(f"\n=== Submitting batch ===")
    response = requests.post(
        f"{API_BASE}/v1/buckets/{bucket_id}/batches/{batch_id}/submit",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "X-Namespace": NAMESPACE,
            "Content-Type": "application/json"
        },
        json={"include_processing_history": True},
        timeout=120
    )

    if response.status_code in (200, 201, 202):
        result = response.json()
        task_id = result.get("task_id")
        print(f"✓ Batch submitted successfully!")
        print(f"\n{'='*60}")
        print(f"BATCH ID: {batch_id}")
        print(f"TASK ID:  {task_id}")
        print(f"OBJECTS:  {len(object_ids)}")
        print(f"{'='*60}")
        return result
    else:
        print(f"✗ Failed to submit batch: {response.status_code}")
        print(response.text)
        return None


def main():
    print("="*60)
    print("Create Batch of All Objects")
    print("="*60)
    print(f"Bucket: {BUCKET_ID}")
    print(f"Namespace: {NAMESPACE}")
    print("="*60)

    # Get all object IDs
    object_ids = get_all_object_ids(BUCKET_ID)

    if not object_ids:
        print("\n✗ No objects found")
        return

    print(f"\n✓ Successfully fetched {len(object_ids):,} object IDs")
    print(f"Ready to create batch (currently disabled for testing)")

    # Uncomment to create and submit batch:
    # create_and_submit_batch(BUCKET_ID, object_ids)


if __name__ == "__main__":
    main()
