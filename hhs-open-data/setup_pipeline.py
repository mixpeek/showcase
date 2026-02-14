#!/usr/bin/env python3
"""
Ask HHS — Multimodal Search Engine Over Government Health Datasets

Ingests FDA adverse event reports (FAERS + CAERS) and drug recalls from the
openFDA API into Mixpeek, creating a public retriever that lets anyone search
government health data with natural language.

Datasets:
  - FAERS (FDA Adverse Event Reporting System) — drug side-effect reports
  - CAERS/Food Events — food, supplement, and cosmetic adverse events
  - FDA Drug Recalls — enforcement actions and product recalls

Usage:
    python customers/hhs-open-data/setup_pipeline.py

Environment:
    MIXPEEK_API_KEY  — Mixpeek API key (required)
    MIXPEEK_API_URL  — API base URL (default: https://api.mixpeek.com)
    MAX_RECORDS      — Max records per dataset (default: 500)
"""

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

# ─── Configuration ───

API_KEY = os.getenv(
    "MIXPEEK_API_KEY",
    "mxp_sk_qqgwtRt53bzgWJOEMS_Q1JA0O_0q52bp8aZhpra9JkGBKwUnxpu_3sDnNdxV5ZSBz9I",
)
BASE_URL = os.getenv("MIXPEEK_API_URL", "https://api.mixpeek.com")
MAX_RECORDS = int(os.getenv("MAX_RECORDS", "500"))
OPENFDA_PAGE_SIZE = 100

NAMESPACE_NAME = "ask_hhs"

# ─── API Helpers ───


def headers(namespace_id=None):
    h = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    if namespace_id:
        h["X-Namespace"] = namespace_id
    return h


def api(method, path, namespace_id=None, allow_conflict=False, **kwargs):
    url = f"{BASE_URL}/v1{path}"
    resp = requests.request(method, url, headers=headers(namespace_id), timeout=120, **kwargs)
    if resp.status_code >= 400:
        if allow_conflict and resp.status_code == 409:
            print(f"  (already exists, skipping)")
            return resp.json()
        print(f"  ERROR {resp.status_code}: {resp.text[:500]}")
        sys.exit(1)
    return resp.json()


def poll_batch(batch_id, namespace_id, timeout=3600, interval=15):
    """Poll batch status until completed or failed."""
    start = time.time()
    while time.time() - start < timeout:
        resp = requests.get(
            f"{BASE_URL}/v1/batches/{batch_id}",
            headers=headers(namespace_id),
            timeout=30,
        )
        if resp.status_code != 200:
            print(f"  Batch poll error: {resp.status_code}")
            time.sleep(interval)
            continue
        data = resp.json()
        status = data.get("status", "unknown")
        progress = data.get("progress", {})
        completed = progress.get("completed", 0)
        total = progress.get("total", 0)
        print(f"  Batch {batch_id}: {status} ({completed}/{total})")
        if status in ("COMPLETED", "completed"):
            return data
        if status in ("FAILED", "failed", "ERROR", "error"):
            print(f"  Batch failed: {data}")
            sys.exit(1)
        time.sleep(interval)
    print(f"  Timeout waiting for batch {batch_id}")
    sys.exit(1)


# ─── openFDA Data Fetchers ───


def fetch_openfda(endpoint, max_records, transform_fn):
    """Fetch records from openFDA API with pagination."""
    records = []
    skip = 0
    while len(records) < max_records:
        limit = min(OPENFDA_PAGE_SIZE, max_records - len(records))
        url = f"https://api.fda.gov/{endpoint}?limit={limit}&skip={skip}"
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code != 200:
                print(f"  openFDA error {resp.status_code} at skip={skip}")
                break
            data = resp.json()
            results = data.get("results", [])
            if not results:
                break
            for r in results:
                transformed = transform_fn(r)
                if transformed:
                    records.append(transformed)
            skip += limit
            if skip >= data.get("meta", {}).get("results", {}).get("total", 0):
                break
        except Exception as e:
            print(f"  openFDA fetch error: {e}")
            break
    return records


def transform_drug_event(record):
    """Transform FAERS drug adverse event into text blob + metadata."""
    patient = record.get("patient", {})

    # Extract drug names
    drugs = patient.get("drug", [])
    drug_names = []
    drug_indications = []
    for d in drugs:
        name = d.get("medicinalproduct", "").strip()
        if name:
            drug_names.append(name)
        indication = d.get("drugindication", "").strip()
        if indication:
            drug_indications.append(indication)

    # Extract reactions
    reactions = patient.get("reaction", [])
    reaction_terms = [r.get("reactionmeddrapt", "") for r in reactions if r.get("reactionmeddrapt")]

    # Patient demographics
    age = patient.get("patientonsetage", "")
    age_unit = patient.get("patientonsetageunit", "")
    sex_map = {"1": "Male", "2": "Female"}
    sex = sex_map.get(patient.get("patientsex", ""), "Unknown")
    weight = patient.get("patientweight", "")

    # Outcomes
    serious = record.get("serious", "")
    outcomes = []
    if record.get("seriousnessdeath") == "1":
        outcomes.append("Death")
    if record.get("seriousnesshospitalization") == "1":
        outcomes.append("Hospitalization")
    if record.get("seriousnesslifethreatening") == "1":
        outcomes.append("Life-Threatening")
    if record.get("seriousnessdisabling") == "1":
        outcomes.append("Disability")

    if not drug_names or not reaction_terms:
        return None

    # Build narrative text for embedding
    text_parts = [
        f"Drug Adverse Event Report",
        f"Drugs: {', '.join(drug_names)}",
    ]
    if drug_indications:
        text_parts.append(f"Indications: {', '.join(drug_indications)}")
    text_parts.append(f"Adverse Reactions: {', '.join(reaction_terms)}")
    if age:
        age_label = {"801": "years", "802": "months", "803": "weeks", "804": "days"}.get(age_unit, "")
        text_parts.append(f"Patient: {age} {age_label} old {sex}")
    if weight:
        text_parts.append(f"Weight: {weight} kg")
    if outcomes:
        text_parts.append(f"Outcomes: {', '.join(outcomes)}")
    if serious == "1":
        text_parts.append("This was classified as a SERIOUS adverse event.")

    receive_date = record.get("receivedate", "")
    if receive_date and len(receive_date) == 8:
        receive_date = f"{receive_date[:4]}-{receive_date[4:6]}-{receive_date[6:8]}"

    return {
        "text": "\n".join(text_parts),
        "metadata": {
            "dataset": "FAERS",
            "report_type": "Drug Adverse Event",
            "drug_names": ", ".join(drug_names[:5]),
            "primary_drug": drug_names[0] if drug_names else "",
            "reactions": ", ".join(reaction_terms[:5]),
            "primary_reaction": reaction_terms[0] if reaction_terms else "",
            "patient_sex": sex,
            "patient_age": f"{age} {age_label}" if age else "",
            "serious": "Yes" if serious == "1" else "No",
            "outcomes": ", ".join(outcomes) if outcomes else "Non-serious",
            "report_date": receive_date,
            "report_id": record.get("safetyreportid", ""),
        },
    }


def transform_food_event(record):
    """Transform CAERS food/supplement adverse event into text blob + metadata."""
    products = record.get("products", [])
    product_names = []
    industry_names = []
    for p in products:
        name = p.get("name_brand", "").strip()
        if name:
            product_names.append(name)
        industry = p.get("industry_name", "").strip()
        if industry:
            industry_names.append(industry)

    reactions_list = record.get("reactions", [])
    outcomes_list = record.get("outcomes", [])

    if not product_names and not reactions_list:
        return None

    text_parts = [
        "Food/Supplement Adverse Event Report",
    ]
    if product_names:
        text_parts.append(f"Products: {', '.join(product_names[:5])}")
    if industry_names:
        text_parts.append(f"Industry: {', '.join(set(industry_names[:3]))}")
    if reactions_list:
        text_parts.append(f"Adverse Reactions: {', '.join(reactions_list[:10])}")
    if outcomes_list:
        text_parts.append(f"Outcomes: {', '.join(outcomes_list[:5])}")

    consumer = record.get("consumer", {})
    age = consumer.get("age", "")
    sex = consumer.get("gender", "")

    if age:
        text_parts.append(f"Consumer Age: {age}")
    if sex:
        text_parts.append(f"Consumer Sex: {sex}")

    date_started = record.get("date_started", "")
    report_date = record.get("date_created", "")

    return {
        "text": "\n".join(text_parts),
        "metadata": {
            "dataset": "CAERS",
            "report_type": "Food/Supplement Adverse Event",
            "product_names": ", ".join(product_names[:3]),
            "primary_product": product_names[0] if product_names else "",
            "industry": ", ".join(set(industry_names[:2])) if industry_names else "",
            "reactions": ", ".join(reactions_list[:5]),
            "primary_reaction": reactions_list[0] if reactions_list else "",
            "outcomes": ", ".join(outcomes_list[:3]) if outcomes_list else "",
            "report_date": report_date[:10] if report_date else "",
            "report_id": record.get("report_number", ""),
        },
    }


def transform_recall(record):
    """Transform FDA drug recall/enforcement into text blob + metadata."""
    reason = (record.get("reason_for_recall") or "").strip()
    product = (record.get("product_description") or "").strip()
    firm = (record.get("recalling_firm") or "").strip()
    classification = (record.get("classification") or "").strip()
    recall_status = (record.get("status") or "").strip()
    distribution = (record.get("distribution_pattern") or "").strip()
    voluntary = (record.get("voluntary_mandated") or "").strip()

    if not reason and not product:
        return None

    text_parts = ["FDA Drug Recall / Enforcement Action"]
    if product:
        text_parts.append(f"Product: {product[:500]}")
    if firm:
        text_parts.append(f"Recalling Firm: {firm}")
    if reason:
        text_parts.append(f"Reason for Recall: {reason}")
    if classification:
        text_parts.append(f"Classification: {classification}")
    if recall_status:
        text_parts.append(f"Status: {recall_status}")
    if distribution:
        text_parts.append(f"Distribution: {distribution[:200]}")
    if voluntary:
        text_parts.append(f"Type: {voluntary}")

    recall_date = record.get("recall_initiation_date", "")
    if recall_date and len(recall_date) == 8:
        recall_date = f"{recall_date[:4]}-{recall_date[4:6]}-{recall_date[6:8]}"

    return {
        "text": "\n".join(text_parts),
        "metadata": {
            "dataset": "FDA_RECALLS",
            "report_type": "Drug Recall",
            "product_description": product[:200] if product else "",
            "recalling_firm": firm,
            "reason": reason[:200] if reason else "",
            "classification": classification,
            "recall_status": recall_status,
            "recall_date": recall_date,
            "recall_number": record.get("recall_number", ""),
        },
    }


# ─── Upload Helpers ───


def upload_record(bucket_id, namespace_id, record):
    """Upload a single record as a bucket object."""
    obj = {
        "blobs": [
            {
                "property": "content",
                "type": "text",
                "data": record["text"],
            }
        ],
    }
    # Add all metadata fields at root level
    for k, v in record["metadata"].items():
        obj[k] = v

    try:
        resp = requests.post(
            f"{BASE_URL}/v1/buckets/{bucket_id}/objects",
            headers=headers(namespace_id),
            json=obj,
            timeout=30,
        )
        if resp.status_code >= 400:
            return False
        return True
    except Exception:
        return False


def upload_records_parallel(bucket_id, namespace_id, records, max_workers=10):
    """Upload records in parallel with progress reporting."""
    success = 0
    failed = 0
    total = len(records)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(upload_record, bucket_id, namespace_id, r): i
            for i, r in enumerate(records)
        }
        for future in as_completed(futures):
            if future.result():
                success += 1
            else:
                failed += 1
            done = success + failed
            if done % 50 == 0 or done == total:
                print(f"  Uploaded {done}/{total} (ok={success}, fail={failed})")

    return success, failed


# ─── Main Pipeline ───


def main():
    print("=" * 60)
    print("Ask HHS — Setting Up Public Health Data Search")
    print("=" * 60)
    print(f"  API: {BASE_URL}")
    print(f"  Max records per dataset: {MAX_RECORDS}")
    print()

    # ─── Step 1: Fetch Data from openFDA ───

    print("[1/7] Fetching data from openFDA API...")

    print("  Fetching FAERS drug adverse events...")
    drug_events = fetch_openfda("drug/event.json", MAX_RECORDS, transform_drug_event)
    print(f"  -> {len(drug_events)} drug adverse events")

    print("  Fetching food/supplement adverse events...")
    food_events = fetch_openfda("food/event.json", MAX_RECORDS, transform_food_event)
    print(f"  -> {len(food_events)} food adverse events")

    print("  Fetching drug recalls...")
    recalls = fetch_openfda("drug/enforcement.json", MAX_RECORDS, transform_recall)
    print(f"  -> {len(recalls)} drug recalls")

    total = len(drug_events) + len(food_events) + len(recalls)
    print(f"\n  Total records to ingest: {total}")

    # ─── Step 2: Create Namespace ───

    print("\n[2/7] Creating namespace...")

    ns_resp = api("POST", "/namespaces", json={
        "namespace_name": NAMESPACE_NAME,
        "description": (
            "Ask HHS: Searchable database of FDA adverse event reports and drug recalls. "
            "Powered by openFDA data — FAERS drug adverse events, CAERS food/supplement "
            "adverse events, and FDA enforcement actions."
        ),
        "feature_extractors": [
            {"feature_extractor_name": "text_extractor", "version": "v1"},
        ],
        "payload_indexes": [
            {"field_name": "dataset", "type": "keyword"},
            {"field_name": "report_type", "type": "keyword"},
            {"field_name": "primary_drug", "type": "keyword"},
            {"field_name": "primary_product", "type": "keyword"},
            {"field_name": "primary_reaction", "type": "keyword"},
            {"field_name": "classification", "type": "keyword"},
            {"field_name": "serious", "type": "keyword"},
            {"field_name": "recalling_firm", "type": "keyword"},
        ],
    })
    namespace_id = ns_resp.get("namespace_id")
    print(f"  Namespace: {namespace_id}")

    # ─── Step 3: Create Buckets ───

    print("\n[3/7] Creating buckets...")

    drug_bucket = api("POST", "/buckets", namespace_id=namespace_id, json={
        "bucket_name": "drug_adverse_events",
        "description": "FDA Adverse Event Reporting System (FAERS) — drug side-effect reports",
        "bucket_schema": {
            "properties": {
                "content": {"type": "text", "description": "Adverse event narrative"},
            },
        },
    })
    drug_bucket_id = drug_bucket.get("bucket_id")
    print(f"  Drug adverse events bucket: {drug_bucket_id}")

    food_bucket = api("POST", "/buckets", namespace_id=namespace_id, json={
        "bucket_name": "food_adverse_events",
        "description": "CFSAN Adverse Event Reporting System (CAERS) — food/supplement adverse events",
        "bucket_schema": {
            "properties": {
                "content": {"type": "text", "description": "Adverse event narrative"},
            },
        },
    })
    food_bucket_id = food_bucket.get("bucket_id")
    print(f"  Food adverse events bucket: {food_bucket_id}")

    recall_bucket = api("POST", "/buckets", namespace_id=namespace_id, json={
        "bucket_name": "drug_recalls",
        "description": "FDA Drug Recalls and Enforcement Actions",
        "bucket_schema": {
            "properties": {
                "content": {"type": "text", "description": "Recall details"},
            },
        },
    })
    recall_bucket_id = recall_bucket.get("bucket_id")
    print(f"  Drug recalls bucket: {recall_bucket_id}")

    # ─── Step 4: Upload Records ───

    print("\n[4/7] Uploading records to buckets...")

    print(f"\n  Uploading {len(drug_events)} drug adverse events...")
    ok1, fail1 = upload_records_parallel(drug_bucket_id, namespace_id, drug_events)
    print(f"  -> Drug events: {ok1} uploaded, {fail1} failed")

    print(f"\n  Uploading {len(food_events)} food adverse events...")
    ok2, fail2 = upload_records_parallel(food_bucket_id, namespace_id, food_events)
    print(f"  -> Food events: {ok2} uploaded, {fail2} failed")

    print(f"\n  Uploading {len(recalls)} drug recalls...")
    ok3, fail3 = upload_records_parallel(recall_bucket_id, namespace_id, recalls)
    print(f"  -> Recalls: {ok3} uploaded, {fail3} failed")

    # ─── Step 5: Create Collections ───

    print("\n[5/7] Creating collections...")

    drug_collection = api("POST", "/collections", namespace_id=namespace_id, json={
        "collection_name": "drug_adverse_events",
        "description": "FAERS drug adverse event reports with semantic text embeddings",
        "source": {
            "type": "bucket",
            "bucket_ids": [drug_bucket_id],
        },
        "feature_extractor": {
            "feature_extractor_name": "text_extractor",
            "version": "v1",
            "input_mappings": {"text": "content"},
            "field_passthrough": [
                {"source_path": "dataset"},
                {"source_path": "report_type"},
                {"source_path": "drug_names"},
                {"source_path": "primary_drug"},
                {"source_path": "reactions"},
                {"source_path": "primary_reaction"},
                {"source_path": "patient_sex"},
                {"source_path": "patient_age"},
                {"source_path": "serious"},
                {"source_path": "outcomes"},
                {"source_path": "report_date"},
                {"source_path": "report_id"},
            ],
        },
    })
    drug_col_id = drug_collection.get("collection_id")
    print(f"  Drug adverse events collection: {drug_col_id}")

    food_collection = api("POST", "/collections", namespace_id=namespace_id, json={
        "collection_name": "food_adverse_events",
        "description": "CAERS food/supplement adverse event reports with semantic text embeddings",
        "source": {
            "type": "bucket",
            "bucket_ids": [food_bucket_id],
        },
        "feature_extractor": {
            "feature_extractor_name": "text_extractor",
            "version": "v1",
            "input_mappings": {"text": "content"},
            "field_passthrough": [
                {"source_path": "dataset"},
                {"source_path": "report_type"},
                {"source_path": "product_names"},
                {"source_path": "primary_product"},
                {"source_path": "industry"},
                {"source_path": "reactions"},
                {"source_path": "primary_reaction"},
                {"source_path": "outcomes"},
                {"source_path": "report_date"},
                {"source_path": "report_id"},
            ],
        },
    })
    food_col_id = food_collection.get("collection_id")
    print(f"  Food adverse events collection: {food_col_id}")

    recall_collection = api("POST", "/collections", namespace_id=namespace_id, json={
        "collection_name": "drug_recalls",
        "description": "FDA drug recall and enforcement action reports with semantic text embeddings",
        "source": {
            "type": "bucket",
            "bucket_ids": [recall_bucket_id],
        },
        "feature_extractor": {
            "feature_extractor_name": "text_extractor",
            "version": "v1",
            "input_mappings": {"text": "content"},
            "field_passthrough": [
                {"source_path": "dataset"},
                {"source_path": "report_type"},
                {"source_path": "product_description"},
                {"source_path": "recalling_firm"},
                {"source_path": "reason"},
                {"source_path": "classification"},
                {"source_path": "recall_status"},
                {"source_path": "recall_date"},
                {"source_path": "recall_number"},
            ],
        },
    })
    recall_col_id = recall_collection.get("collection_id")
    print(f"  Drug recalls collection: {recall_col_id}")

    # ─── Step 6: Trigger Collection Processing ───

    print("\n[6/7] Triggering collection processing...")

    for col_name, col_id in [
        ("drug_adverse_events", drug_col_id),
        ("food_adverse_events", food_col_id),
        ("drug_recalls", recall_col_id),
    ]:
        print(f"  Triggering {col_name} ({col_id})...")
        trigger_resp = api("POST", f"/collections/{col_id}/trigger", namespace_id=namespace_id, json={})
        batch_id = trigger_resp.get("batch_id")
        if batch_id:
            print(f"  -> Batch: {batch_id}")
        else:
            print(f"  -> Response: {trigger_resp}")

    # Wait for processing
    print("\n  Processing triggered! Batches are running asynchronously.")
    print("  Text extraction typically takes 10-20 minutes for 1500 records.")
    print("  You can check batch status via the Mixpeek dashboard.")
    print("  Proceeding to create retriever (it will return results once processing completes)...")

    # ─── Step 7: Create & Publish Retriever ───

    print("\n[7/7] Creating and publishing retriever...")

    retriever = api("POST", "/retrievers", namespace_id=namespace_id, json={
        "retriever_name": "ask_hhs_search",
        "description": "Search FDA adverse event reports and drug recalls with natural language",
        "collection_identifiers": [drug_col_id, food_col_id, recall_col_id],
        "stages": [
            {
                "stage_name": "semantic_search",
                "stage_type": "filter",
                "config": {
                    "stage_id": "feature_search",
                    "parameters": {
                        "searches": [
                            {
                                "feature_uri": "mixpeek://text_extractor@v1/multilingual_e5_large_instruct_v1",
                                "query": {
                                    "input_mode": "text",
                                    "text": "{{INPUT.query}}",
                                },
                                "top_k": 50,
                            }
                        ],
                        "final_top_k": 30,
                    },
                },
            },
        ],
        "input_schema": {
            "query": {
                "type": "text",
                "description": "Search FDA health data — try 'aspirin side effects' or 'recalled baby formula'",
                "required": True,
            },
        },
    })
    retriever_resp = retriever.get("retriever", retriever)
    retriever_id = retriever_resp.get("retriever_id")
    print(f"  Retriever: {retriever_id}")

    # Publish as public retriever
    print("  Publishing retriever...")
    publish_resp = api("POST", f"/retrievers/{retriever_id}/publish", namespace_id=namespace_id, json={
        "public_name": "ask-hhs",
        "description": (
            "Search 1,500+ FDA adverse event reports and drug recalls with natural language. "
            "Data sourced from FAERS (drug adverse events), CAERS (food/supplement adverse events), "
            "and FDA enforcement actions via openFDA."
        ),
        "display_config": {
            "title": "Ask HHS",
            "description": (
                "Search FDA adverse event reports and drug recalls. "
                "Try: 'aspirin side effects', 'recalled medications', "
                "'food supplement allergic reactions', 'baby formula recall'"
            ),
            "theme": {
                "primary_color": "#1a5276",
                "secondary_color": "#2e86c1",
                "background_color": "#f8f9fa",
                "surface_color": "#ffffff",
            },
            "inputs": [
                {
                    "field_name": "query",
                    "field_schema": {"type": "string"},
                    "label": "Search FDA Health Data",
                    "placeholder": "e.g. 'aspirin side effects' or 'recalled baby formula'",
                    "required": True,
                    "order": 0,
                    "component_type": "text_input",
                },
            ],
            "layout": {
                "columns": 3,
                "full_width": True,
                "gap": "12px",
            },
            "exposed_fields": [
                "dataset",
                "report_type",
                "drug_names",
                "primary_drug",
                "primary_product",
                "reactions",
                "primary_reaction",
                "outcomes",
                "serious",
                "classification",
                "recalling_firm",
                "reason",
                "product_description",
                "report_date",
                "report_id",
                "recall_number",
            ],
            "field_config": {
                "dataset": {
                    "format": "text",
                    "format_options": {"label": "Source"},
                },
                "report_type": {
                    "format": "text",
                    "format_options": {"label": "Type"},
                },
                "primary_drug": {
                    "format": "text",
                    "format_options": {"label": "Drug"},
                },
                "primary_product": {
                    "format": "text",
                    "format_options": {"label": "Product"},
                },
                "primary_reaction": {
                    "format": "text",
                    "format_options": {"label": "Reaction"},
                },
                "serious": {
                    "format": "text",
                    "format_options": {"label": "Serious"},
                },
                "classification": {
                    "format": "text",
                    "format_options": {"label": "Recall Class"},
                },
                "recalling_firm": {
                    "format": "text",
                    "format_options": {"label": "Firm"},
                },
                "report_date": {
                    "format": "text",
                    "format_options": {"label": "Date"},
                },
            },
            "components": {
                "result_card": {
                    "layout": "vertical",
                    "show_thumbnail": False,
                    "card_style": "default",
                    "card_click_action": "viewDetails",
                    "title_field": "report_type",
                    "subtitle_field": "dataset",
                    "card_fields": [
                        "primary_drug",
                        "primary_product",
                        "primary_reaction",
                        "serious",
                        "classification",
                        "report_date",
                    ],
                },
            },
            "field_mappings": {},
            "template_type": "default",
        },
        "tags": ["hhs", "fda", "adverse-events", "recalls", "public-health", "open-data"],
        "category": "public-health",
    })
    public_url = publish_resp.get("public_url", publish_resp.get("short_url", ""))
    print(f"  Published! URL: {public_url}")

    # ─── Summary ───

    print("\n" + "=" * 60)
    print("ASK HHS — SETUP COMPLETE!")
    print("=" * 60)
    print(f"""
Resources created:
  Namespace:    {namespace_id} ({NAMESPACE_NAME})
  Buckets:      {drug_bucket_id} (drug_adverse_events)
                {food_bucket_id} (food_adverse_events)
                {recall_bucket_id} (drug_recalls)
  Collections:  {drug_col_id} (drug_adverse_events)
                {food_col_id} (food_adverse_events)
                {recall_col_id} (drug_recalls)
  Retriever:    {retriever_id} (ask_hhs_search)
  Published:    ask-hhs

Data loaded:
  Drug adverse events (FAERS): {len(drug_events)}
  Food adverse events (CAERS): {len(food_events)}
  Drug recalls:                {len(recalls)}
  Total:                       {total}

Public URL: {public_url}
""")


if __name__ == "__main__":
    main()
