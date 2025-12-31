#!/usr/bin/env python3
"""
Quick test to verify cursor-based pagination is working.
"""
import requests

# Configuration
API_KEY = "sk_f8TwRhcOt0RG3qnUnJ9bfstMraHM_5mcvbkA4tAWtK_ok7qmyN_F_8Qr6dXEW0_TZZ8"
NAMESPACE = "test-namespace"
API_BASE = "http://localhost:8000"

def test_cursor_pagination():
    """Test that cursor-based pagination works correctly."""
    print("="*60)
    print("Testing Cursor-Based Pagination")
    print("="*60)

    # First, create a test bucket
    print("\n1. Creating test bucket...")
    response = requests.post(
        f"{API_BASE}/v1/buckets",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "X-Namespace": NAMESPACE,
            "Content-Type": "application/json"
        },
        json={
            "bucket_name": f"cursor-test-bucket",
            "description": "Test bucket for cursor pagination"
        },
        timeout=10
    )

    if response.status_code not in (200, 201):
        print(f"✗ Failed to create bucket: {response.status_code}")
        print(response.text)
        return

    bucket = response.json()
    bucket_id = bucket.get("bucket_id")
    print(f"✓ Created bucket: {bucket_id}")

    # Test cursor-based pagination
    print("\n2. Testing cursor pagination (no include_total)...")
    response = requests.post(
        f"{API_BASE}/v1/buckets/{bucket_id}/objects/list",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "X-Namespace": NAMESPACE,
            "Content-Type": "application/json"
        },
        params={"limit": 10},  # Cursor-based, no include_total
        timeout=10
    )

    if response.status_code != 200:
        print(f"✗ Failed to list objects: {response.status_code}")
        print(response.text)
        return

    data = response.json()
    pagination = data.get("pagination", {})

    print(f"✓ List succeeded")
    print(f"  - Results: {len(data.get('results', []))}")
    print(f"  - next_cursor: {pagination.get('next_cursor')}")
    print(f"  - total (should be None): {pagination.get('total')}")

    # Verify total is NOT present (we didn't request it)
    if pagination.get('total') is None:
        print("✓ PASS: total is None (no COUNT query ran)")
    else:
        print(f"✗ FAIL: total should be None, got {pagination.get('total')}")

    # Test with include_total=true
    print("\n3. Testing cursor pagination WITH include_total...")
    response = requests.post(
        f"{API_BASE}/v1/buckets/{bucket_id}/objects/list",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "X-Namespace": NAMESPACE,
            "Content-Type": "application/json"
        },
        params={"limit": 10, "include_total": "true"},
        timeout=10
    )

    if response.status_code != 200:
        print(f"✗ Failed to list objects: {response.status_code}")
        print(response.text)
        return

    data = response.json()
    pagination = data.get("pagination", {})

    print(f"✓ List succeeded")
    print(f"  - Results: {len(data.get('results', []))}")
    print(f"  - next_cursor: {pagination.get('next_cursor')}")
    print(f"  - total: {pagination.get('total')}")
    print(f"  - page: {pagination.get('page')}")
    print(f"  - total_pages: {pagination.get('total_pages')}")

    # Verify total IS present when requested
    if pagination.get('total') is not None:
        print("✓ PASS: total is present (COUNT query ran)")
    else:
        print("✗ FAIL: total should be present, got None")

    # Test that offset parameter is rejected/ignored
    print("\n4. Testing that offset is no longer supported...")
    response = requests.post(
        f"{API_BASE}/v1/buckets/{bucket_id}/objects/list",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "X-Namespace": NAMESPACE,
            "Content-Type": "application/json"
        },
        params={"limit": 10, "offset": 100},  # Try to use offset
        timeout=10
    )

    if response.status_code == 200:
        print("✓ Request succeeded (offset parameter ignored)")
    else:
        print(f"Response: {response.status_code}")

    # Cleanup
    print("\n5. Cleaning up test bucket...")
    response = requests.delete(
        f"{API_BASE}/v1/buckets/{bucket_id}",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "X-Namespace": NAMESPACE,
        },
        timeout=10
    )

    if response.status_code in (200, 204):
        print("✓ Bucket deleted")
    else:
        print(f"Note: Cleanup status {response.status_code}")

    print("\n" + "="*60)
    print("Cursor Pagination Test Complete!")
    print("="*60)


if __name__ == "__main__":
    test_cursor_pagination()
