#!/usr/bin/env python3
"""
Test cursor pagination fix - ensure it doesn't loop infinitely.
"""
import requests

API_KEY = "sk_f8TwRhcOt0RG3qnUnJ9bfstMraHM_5mcvbkA4tAWtK_ok7qmyN_F_8Qr6dXEW0_TZZ8"
NAMESPACE = "test-namespace"
API_BASE = "http://localhost:8000"

def test_cursor_pagination():
    print("=" * 60)
    print("Testing Cursor Pagination Fix")
    print("=" * 60)
    
    # Create test bucket
    print("\n1. Creating test bucket...")
    response = requests.post(
        f"{API_BASE}/v1/buckets",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "X-Namespace": NAMESPACE,
            "Content-Type": "application/json"
        },
        json={
            "bucket_name": f"cursor-fix-test",
            "description": "Test cursor pagination fix",
            "bucket_schema": {
                "properties": {
                    "title": {"type": "string"},
                }
            }
        },
        timeout=10
    )
    
    if response.status_code != 200:
        print(f"✗ Failed to create bucket: {response.status_code}")
        print(response.text)
        return
    
    bucket = response.json()
    bucket_id = bucket.get("bucket_id")
    print(f"✓ Created bucket: {bucket_id}")
    
    # Add 25 objects (will create 3 pages with limit=10)
    print("\n2. Adding 25 test objects...")
    for i in range(25):
        response = requests.post(
            f"{API_BASE}/v1/buckets/{bucket_id}/objects",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "X-Namespace": NAMESPACE,
                "Content-Type": "application/json"
            },
            json={
                "title": f"Test Object {i+1}"
            },
            timeout=10
        )
        if response.status_code not in (200, 201):
            print(f"✗ Failed to create object {i+1}")
            return
    print("✓ Created 25 objects")
    
    # Test cursor pagination
    print("\n3. Testing cursor pagination...")
    cursor = None
    page_num = 1
    total_fetched = 0
    cursors_seen = set()
    
    while True:
        print(f"\nPage {page_num}:")
        
        # Build params
        params = {"limit": 10}
        if cursor:
            params["cursor"] = cursor
            
        response = requests.post(
            f"{API_BASE}/v1/buckets/{bucket_id}/objects/list",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "X-Namespace": NAMESPACE,
            },
            params=params,
            timeout=10
        )
        
        if response.status_code != 200:
            print(f"✗ Failed: {response.status_code}")
            print(response.text)
            break
        
        data = response.json()
        results = data.get("results", [])
        pagination = data.get("pagination", {})
        next_cursor = pagination.get("next_cursor")
        
        print(f"  Fetched {len(results)} objects")
        print(f"  next_cursor: {next_cursor}")
        
        total_fetched += len(results)
        
        # Check for infinite loop - if we've seen this cursor before, we're looping
        if cursor and cursor in cursors_seen:
            print(f"\n✗ FAIL: Infinite loop detected! Cursor repeated: {cursor}")
            break
        
        if cursor:
            cursors_seen.add(cursor)
        
        # If no results or no next cursor, we're done
        if not results or next_cursor is None:
            print(f"\n✓ PASS: Pagination complete")
            print(f"  Total objects fetched: {total_fetched}")
            print(f"  Total pages: {page_num}")
            print(f"  Final next_cursor: {next_cursor}")
            
            if total_fetched == 25:
                print(f"\n✓ SUCCESS: Fetched all 25 objects without looping!")
            else:
                print(f"\n⚠️  WARNING: Expected 25 objects, got {total_fetched}")
            break
        
        # Move to next page
        cursor = next_cursor
        page_num += 1
        
        # Safety: prevent infinite loop in test
        if page_num > 10:
            print(f"\n✗ FAIL: Too many pages (>10), likely infinite loop")
            break
    
    # Cleanup
    print("\n4. Cleaning up...")
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
    
    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)

if __name__ == "__main__":
    test_cursor_pagination()
