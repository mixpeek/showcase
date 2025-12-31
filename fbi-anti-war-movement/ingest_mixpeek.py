#!/usr/bin/env python3
"""
Mixpeek Ingestion Script for FBI Anti-War Movement Documents

This script uploads all PDF files from data directory with their metadata.
Uses batch upload API for efficiency.

Source: FBI Vault - Anti-War Movement Collection
https://vault.fbi.gov/anti-war-movement

Docs: https://docs.mixpeek.com
"""

import time
import requests
from pathlib import Path

# Configuration
API_KEY = "sk_6zci28-aEK9zVa_ou5YLSVyF1eNVFNu--D5pGKVzYP1Z391_LSxTAZ3lRnvj2bAsqFc"
NAMESPACE = "ns_ea38d330ad"
API_BASE = "https://api.mixpeek.com"
BATCH_SIZE = 5  # Smaller batches for larger PDF files
DEBUG_FIRST_REQUEST = True

# Throttling to prevent MongoDB connection exhaustion
DELAY_BETWEEN_CONFIRMS = 0.2
DELAY_BETWEEN_BATCHES = 2.0

# Data directory
DATA_DIR = Path(__file__).parent / "data"

# Subject metadata from FBI Vault
SUBJECT_METADATA = {
    "mario-savio": {
        "subject_name": "Mario Savio",
        "years_lived": "1942-1996",
        "description": "Political and human rights activist from the University of California at Berkeley who became the voice of the Free Speech Movement.",
        "investigation_period": "July 1964 - January 1975",
        "investigation_reason": "Following his arrest in March 1964 at a civil rights demonstration in San Francisco.",
        "category": "Activist",
        "keywords": ["Free Speech Movement", "UC Berkeley", "civil rights", "student activism"]
    },
    "abbie-hoffman": {
        "subject_name": "Abbie Hoffman",
        "years_lived": "1936-1989",
        "description": "Political and social activist, especially in the 1960s. Leader in the Yippie Party (Youth International Party).",
        "investigation_period": "1968-1973",
        "investigation_reason": "Investigated under anti-riot laws and as a domestic security threat. Role in anti-war protests at the 1968 Democratic National Convention in Chicago.",
        "category": "Activist",
        "keywords": ["Yippie Party", "Youth International Party", "Chicago Eight", "1968 DNC protests", "anti-riot laws"]
    },
    "kent-state": {
        "subject_name": "Kent State",
        "years_lived": None,
        "description": "FBI investigation into the Kent State shootings of May 4, 1970, where Ohio National Guard killed four students during anti-war protests.",
        "investigation_period": "1970",
        "investigation_reason": "Investigation of the Kent State University shootings.",
        "category": "Event",
        "keywords": ["Kent State shootings", "Ohio National Guard", "1970", "student protests", "Vietnam War protests"]
    },
    "clergy-laity-vietnam": {
        "subject_name": "Clergy and Laity Concerned about Vietnam",
        "years_lived": None,
        "description": "Religious organization that opposed the Vietnam War, bringing together clergy and laypeople from various faiths.",
        "investigation_period": None,
        "investigation_reason": "FBI monitoring of anti-war religious organizations.",
        "category": "Organization",
        "keywords": ["religious anti-war", "clergy", "Vietnam War", "peace movement"]
    },
    "american-friends-service-committee": {
        "subject_name": "American Friends Service Committee",
        "years_lived": None,
        "description": "Quaker peace and social justice organization founded in 1917. Monitored by FBI for anti-war activities.",
        "investigation_period": None,
        "investigation_reason": "FBI monitoring of peace organizations.",
        "category": "Organization",
        "keywords": ["Quakers", "peace organization", "social justice", "conscientious objectors"]
    },
    "howard-zinn": {
        "subject_name": "Howard Zinn",
        "years_lived": "1922-2010",
        "description": "Historian, author, playwright, and social activist. Best known for 'A People's History of the United States'.",
        "investigation_period": None,
        "investigation_reason": "FBI monitoring of activist academics.",
        "category": "Activist",
        "keywords": ["historian", "academic", "civil rights", "anti-war", "author"]
    },
    "cardinal-francis-spellman": {
        "subject_name": "Cardinal Francis Spellman",
        "years_lived": "1889-1967",
        "description": "Roman Catholic cardinal in the diocese of New York. Files consist of correspondence between Spellman and the FBI and investigation into the bombing of his home.",
        "investigation_period": None,
        "investigation_reason": "Correspondence with FBI; investigation into bombing of cardinal's home in New York.",
        "category": "Religious Figure",
        "keywords": ["Catholic Church", "New York", "cardinal", "bombing investigation"]
    },
    "jane-addams": {
        "subject_name": "Jane Addams",
        "years_lived": "1860-1935",
        "description": "Internationally known social worker, activist, and Nobel Peace Prize winner. Founding member of Women's International League for Peace and Freedom.",
        "investigation_period": "1924",
        "investigation_reason": "Treason investigation involving the Women's International League for Peace and Freedom.",
        "category": "Activist",
        "keywords": ["Hull House", "Nobel Peace Prize", "social worker", "Women's International League for Peace and Freedom", "pacifist"]
    },
    "edward-abbey": {
        "subject_name": "Edward Abbey",
        "years_lived": None,
        "description": "Author investigated by the FBI for sedition. While attending school in Pennsylvania, publicly proposed destruction of draft cards. Served in U.S. military 1945-1947.",
        "investigation_period": None,
        "investigation_reason": "Sedition investigation; Loyalty of Government Employee investigation while working for National Forest Service.",
        "category": "Author",
        "keywords": ["author", "sedition", "draft cards", "National Forest Service", "environmentalist"]
    }
}


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


def get_or_create_bucket():
    """Get existing bucket or create a new one."""
    print("\n=== Getting/Creating Bucket ===")

    bucket_data = {
        "bucket_name": "fbi-anti-war-movement",
        "description": "FBI Vault documents from the Anti-War Movement collection",
        "bucket_schema": {
            "properties": {
                "document": {"type": "pdf"},
                "file_id": {"type": "string"},
                "filename": {"type": "string"},
                "subject_name": {"type": "string"},
                "subject_slug": {"type": "string"},
                "part_number": {"type": "integer"},
                "years_lived": {"type": "string"},
                "description": {"type": "string"},
                "investigation_period": {"type": "string"},
                "investigation_reason": {"type": "string"},
                "category": {"type": "string"},
                "keywords": {"type": "array"},
                "source_collection": {"type": "string"},
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
            if bucket.get("bucket_name") == "fbi-anti-war-movement":
                print(f"Using existing bucket: {bucket.get('bucket_id')}")
                return bucket

    return None


def parse_filename(filename):
    """Parse filename to extract subject and part number."""
    # Format: subject-name-part-XX.pdf
    name = filename.replace(".pdf", "")
    parts = name.rsplit("-part-", 1)

    if len(parts) == 2:
        subject_slug = parts[0]
        part_num = int(parts[1])
    else:
        subject_slug = name
        part_num = 1

    return subject_slug, part_num


def get_all_samples():
    """Get all PDF files from the data directory."""
    print("\n=== Loading All PDFs ===")

    pdf_files = list(DATA_DIR.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files")

    result = []
    for pdf_path in sorted(pdf_files):
        subject_slug, part_num = parse_filename(pdf_path.name)

        # Get subject metadata
        subject_meta = SUBJECT_METADATA.get(subject_slug, {})

        result.append({
            "pdf_path": pdf_path,
            "subject_slug": subject_slug,
            "part_number": part_num,
            "subject_metadata": subject_meta
        })

    print(f"Loaded {len(result)} PDF files")
    return result


first_request_printed = False


def create_batch_uploads(bucket_id, samples):
    """Create batch upload requests to get presigned URLs."""
    global first_request_printed

    uploads = []
    for sample in samples:
        pdf_path = sample["pdf_path"]
        subject_slug = sample["subject_slug"]
        part_num = sample["part_number"]
        meta = sample["subject_metadata"]

        # Build object_metadata
        object_metadata = {
            "file_id": pdf_path.stem,  # filename without extension
            "filename": pdf_path.name,
            "subject_name": meta.get("subject_name", subject_slug.replace("-", " ").title()),
            "subject_slug": subject_slug,
            "part_number": part_num,
            "category": meta.get("category", "Unknown"),
            "keywords": meta.get("keywords", []),
            "source_collection": "FBI Vault - Anti-War Movement",
            "source_url": "https://vault.fbi.gov/anti-war-movement"
        }

        # Add optional fields if present
        if meta.get("years_lived"):
            object_metadata["years_lived"] = meta["years_lived"]
        if meta.get("description"):
            object_metadata["description"] = meta["description"]
        if meta.get("investigation_period"):
            object_metadata["investigation_period"] = meta["investigation_period"]
        if meta.get("investigation_reason"):
            object_metadata["investigation_reason"] = meta["investigation_reason"]

        uploads.append({
            "filename": pdf_path.name,
            "content_type": "application/pdf",
            "file_size_bytes": pdf_path.stat().st_size,
            "blob_property": "document",
            "create_object_on_confirm": True,
            "object_metadata": object_metadata
        })

    batch_request = {"uploads": uploads}

    # Debug: print first request payload
    if DEBUG_FIRST_REQUEST and not first_request_printed:
        import json
        print("\n=== DEBUG: First batch request payload ===")
        print(json.dumps(batch_request, indent=2, default=str))
        print("=== END DEBUG ===\n")
        first_request_printed = True

    # Retry with backoff for rate limits
    max_retries = 3
    for attempt in range(max_retries):
        response = api_request("POST", f"/v1/buckets/{bucket_id}/uploads/batch", batch_request)

        if response.status_code == 429:
            wait_time = 60 * (attempt + 1)
            print(f"Rate limited. Waiting {wait_time}s before retry...")
            time.sleep(wait_time)
            continue
        elif response.status_code in (200, 201):
            return response.json().get("uploads", [])
        else:
            break

    return []


def upload_to_s3(presigned_url, pdf_path, max_retries=3):
    """Upload file to S3 using presigned URL with retry logic."""
    with open(pdf_path, "rb") as f:
        file_data = f.read()

    for attempt in range(max_retries):
        try:
            response = requests.put(
                presigned_url,
                data=file_data,
                headers={"Content-Type": "application/pdf"},
                timeout=120  # Longer timeout for larger PDFs
            )
            return response.status_code in (200, 201, 204)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"    S3 upload failed ({type(e).__name__}), retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"    S3 upload failed after {max_retries} attempts: {e}")
                return False
    return False


def confirm_upload(bucket_id, upload_id):
    """Confirm upload completion."""
    response = api_request(
        "POST",
        f"/v1/buckets/{bucket_id}/uploads/{upload_id}/confirm",
        {}
    )
    if response.status_code in (200, 201):
        return response.json().get("object_id")
    return None


def process_batch(bucket_id, samples, batch_num, total_batches):
    """Process a batch of uploads."""
    print(f"\n--- Batch {batch_num}/{total_batches} ({len(samples)} files) ---")

    # Step 1: Create batch uploads to get presigned URLs
    print("Creating batch upload requests...")
    upload_responses = create_batch_uploads(bucket_id, samples)

    if not upload_responses:
        print("Failed to create batch uploads")
        return []

    print(f"Got {len(upload_responses)} presigned URLs")

    # Step 2: Upload files to S3 and confirm
    object_ids = []
    for i, (sample, upload_info) in enumerate(zip(samples, upload_responses), 1):
        filename = sample["pdf_path"].name
        upload_id = upload_info.get("upload_id")
        presigned_url = upload_info.get("presigned_url")

        if not presigned_url:
            print(f"  [{i}] {filename}... SKIPPED (no URL)")
            continue

        # Upload to S3
        print(f"  [{i}] Uploading {filename}...", end=" ", flush=True)
        if upload_to_s3(presigned_url, sample["pdf_path"]):
            # Confirm upload
            object_id = confirm_upload(bucket_id, upload_id)
            if object_id:
                object_ids.append(object_id)
                print("OK")
            else:
                print("CONFIRM FAILED")
        else:
            print("S3 FAILED")

        # Small delay between confirms
        if DELAY_BETWEEN_CONFIRMS > 0:
            time.sleep(DELAY_BETWEEN_CONFIRMS)

    return object_ids


def upload_objects(bucket_id, samples):
    """Upload all samples in batches."""
    total = len(samples)
    total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"\n=== Uploading {total} Objects in {total_batches} Batches ===")

    all_object_ids = []

    for batch_num in range(total_batches):
        start_idx = batch_num * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, total)
        batch_samples = samples[start_idx:end_idx]

        object_ids = process_batch(bucket_id, batch_samples, batch_num + 1, total_batches)
        all_object_ids.extend(object_ids)

        print(f"Batch complete: {len(object_ids)} uploaded, {len(all_object_ids)} total")

        # Delay between batches
        if batch_num < total_batches - 1 and DELAY_BETWEEN_BATCHES > 0:
            time.sleep(DELAY_BETWEEN_BATCHES)

    return all_object_ids


def main():
    print("=" * 60)
    print("Mixpeek FBI Anti-War Movement Ingestion")
    print("=" * 60)

    # Step 1: Get or create bucket
    bucket = get_or_create_bucket()
    if not bucket:
        print("Failed to create/get bucket")
        return

    bucket_id = bucket.get("bucket_id")
    print(f"Using bucket: {bucket_id}")

    # Step 2: Load all samples
    samples = get_all_samples()
    if not samples:
        print("No samples found")
        return

    # Step 3: Upload objects in batches
    object_ids = upload_objects(bucket_id, samples)

    print("\n" + "=" * 60)
    if object_ids:
        print("SUCCESS!")
        print(f"Bucket ID: {bucket_id}")
        print(f"Objects uploaded: {len(object_ids)}")
    else:
        print("No objects were uploaded")
    print("=" * 60)


if __name__ == "__main__":
    main()
