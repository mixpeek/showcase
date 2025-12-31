#!/usr/bin/env python3
"""
USPTO Patent Downloader

Downloads patent images and metadata from USPTO sources:
- Patent metadata (abstracts, titles, inventors) from PatentsView TSV files
- Patent PDF images from USPTO image-ppubs endpoint

Data Sources:
- PatentsView: https://patentsview.org/download/data-download-tables
- USPTO Image Server: https://image-ppubs.uspto.gov

License: Patent data is in the public domain
"""

import os
import csv
import json
import time
import requests
import argparse
import zipfile
import io
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# PatentsView Data URLs (2024 data files)
PATENTSVIEW_BASE = "https://s3.amazonaws.com/data.patentsview.org/download"

# Core patent tables
PATENT_TSV_URL = f"{PATENTSVIEW_BASE}/g_patent.tsv.zip"
ABSTRACT_TSV_URL = f"{PATENTSVIEW_BASE}/g_brf_sum_text.tsv.zip"
CLAIMS_TSV_URL = f"{PATENTSVIEW_BASE}/g_claims.tsv.zip"  # Large file
INVENTOR_TSV_URL = f"{PATENTSVIEW_BASE}/g_inventor.tsv.zip"
ASSIGNEE_TSV_URL = f"{PATENTSVIEW_BASE}/g_assignee.tsv.zip"
CPC_TSV_URL = f"{PATENTSVIEW_BASE}/g_cpc_current.tsv.zip"

# USPTO Image Server base URL
USPTO_IMAGE_BASE = "https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf"


def download_and_extract_tsv(url: str, cache_dir: Path, filename: str = None) -> Path:
    """Download a zipped TSV file and extract it."""
    if filename is None:
        filename = url.split("/")[-1].replace(".zip", "")

    tsv_path = cache_dir / filename
    zip_path = cache_dir / f"{filename}.zip"

    if tsv_path.exists():
        logger.info(f"Using cached {filename}")
        return tsv_path

    logger.info(f"Downloading {filename}...")
    try:
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()

        cache_dir.mkdir(parents=True, exist_ok=True)

        # Extract directly from response
        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            # Get the first TSV file in the archive
            tsv_names = [n for n in zf.namelist() if n.endswith('.tsv')]
            if tsv_names:
                with zf.open(tsv_names[0]) as src, open(tsv_path, 'wb') as dst:
                    dst.write(src.read())
                logger.info(f"Extracted {filename}")
                return tsv_path
            else:
                logger.error(f"No TSV file found in {url}")
                return None

    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return None


def load_patents(tsv_path: Path, limit: int = None, year_min: int = None, year_max: int = None) -> dict:
    """Load patent data from PatentsView TSV file."""
    patents = {}
    count = 0

    logger.info(f"Loading patents from {tsv_path}...")

    with open(tsv_path, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            patent_id = row.get('patent_id', '').strip()
            if not patent_id:
                continue

            # Filter by year if specified
            patent_date = row.get('patent_date', '')
            if patent_date and (year_min or year_max):
                try:
                    year = int(patent_date.split('-')[0])
                    if year_min and year < year_min:
                        continue
                    if year_max and year > year_max:
                        continue
                except (ValueError, IndexError):
                    continue

            patents[patent_id] = {
                'patent_id': patent_id,
                'patent_type': row.get('patent_type', ''),
                'patent_date': patent_date,
                'patent_title': row.get('patent_title', ''),
                'num_claims': row.get('num_claims', ''),
            }

            count += 1
            if limit and count >= limit:
                break

    logger.info(f"Loaded {len(patents)} patents")
    return patents


def load_abstracts(tsv_path: Path, patent_ids: set = None) -> dict:
    """Load patent abstracts from PatentsView TSV file."""
    abstracts = {}

    if not tsv_path or not tsv_path.exists():
        return abstracts

    logger.info(f"Loading abstracts from {tsv_path}...")

    with open(tsv_path, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            patent_id = row.get('patent_id', '').strip()
            if not patent_id:
                continue
            if patent_ids and patent_id not in patent_ids:
                continue

            text = row.get('summary_text', '') or row.get('brf_sum_text', '')
            if text:
                abstracts[patent_id] = text

    logger.info(f"Loaded {len(abstracts)} abstracts")
    return abstracts


def load_inventors(tsv_path: Path, patent_ids: set = None) -> dict:
    """Load inventor data from PatentsView TSV file."""
    inventors = {}

    if not tsv_path or not tsv_path.exists():
        return inventors

    logger.info(f"Loading inventors from {tsv_path}...")

    with open(tsv_path, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            patent_id = row.get('patent_id', '').strip()
            if not patent_id:
                continue
            if patent_ids and patent_id not in patent_ids:
                continue

            inventor = {
                'name_first': row.get('inventor_name_first', '') or row.get('name_first', ''),
                'name_last': row.get('inventor_name_last', '') or row.get('name_last', ''),
                'city': row.get('inventor_city', '') or row.get('city', ''),
                'state': row.get('inventor_state', '') or row.get('state', ''),
                'country': row.get('inventor_country', '') or row.get('country', ''),
            }

            if patent_id not in inventors:
                inventors[patent_id] = []
            inventors[patent_id].append(inventor)

    logger.info(f"Loaded inventors for {len(inventors)} patents")
    return inventors


def load_assignees(tsv_path: Path, patent_ids: set = None) -> dict:
    """Load assignee data from PatentsView TSV file."""
    assignees = {}

    if not tsv_path or not tsv_path.exists():
        return assignees

    logger.info(f"Loading assignees from {tsv_path}...")

    with open(tsv_path, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            patent_id = row.get('patent_id', '').strip()
            if not patent_id:
                continue
            if patent_ids and patent_id not in patent_ids:
                continue

            assignee = {
                'organization': row.get('organization', '') or row.get('assignee_organization', ''),
                'type': row.get('assignee_type', ''),
                'city': row.get('assignee_city', '') or row.get('city', ''),
                'state': row.get('assignee_state', '') or row.get('state', ''),
                'country': row.get('assignee_country', '') or row.get('country', ''),
            }

            if patent_id not in assignees:
                assignees[patent_id] = []
            assignees[patent_id].append(assignee)

    logger.info(f"Loaded assignees for {len(assignees)} patents")
    return assignees


def load_cpc_codes(tsv_path: Path, patent_ids: set = None) -> dict:
    """Load CPC classification codes from PatentsView TSV file."""
    cpc_codes = {}

    if not tsv_path or not tsv_path.exists():
        return cpc_codes

    logger.info(f"Loading CPC codes from {tsv_path}...")

    with open(tsv_path, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            patent_id = row.get('patent_id', '').strip()
            if not patent_id:
                continue
            if patent_ids and patent_id not in patent_ids:
                continue

            cpc = {
                'section': row.get('cpc_section', ''),
                'class': row.get('cpc_class', ''),
                'subclass': row.get('cpc_subclass', ''),
                'group': row.get('cpc_group', ''),
                'subgroup': row.get('cpc_subgroup', ''),
                'category': row.get('cpc_category', ''),
                'sequence': row.get('cpc_sequence', ''),
            }

            if patent_id not in cpc_codes:
                cpc_codes[patent_id] = []
            cpc_codes[patent_id].append(cpc)

    logger.info(f"Loaded CPC codes for {len(cpc_codes)} patents")
    return cpc_codes


def format_patent_number(patent_id: str) -> str:
    """Format patent number for USPTO image URL."""
    # Remove any leading zeros and repad appropriately
    patent_id = patent_id.strip()

    # Design patents start with 'D'
    if patent_id.upper().startswith('D'):
        num = patent_id[1:].lstrip('0')
        return f"D{num.zfill(6)}"

    # Plant patents start with 'PP'
    if patent_id.upper().startswith('PP'):
        num = patent_id[2:].lstrip('0')
        return f"PP{num.zfill(5)}"

    # Reissue patents start with 'RE'
    if patent_id.upper().startswith('RE'):
        num = patent_id[2:].lstrip('0')
        return f"RE{num.zfill(5)}"

    # Regular utility patents - pad to 7 or 8 digits
    num = patent_id.lstrip('0')
    if len(num) <= 7:
        return num.zfill(7)
    return num.zfill(8)


def construct_pdf_url(patent_id: str) -> str:
    """Construct USPTO PDF download URL for a patent."""
    formatted = format_patent_number(patent_id)
    return f"{USPTO_IMAGE_BASE}/{formatted}"


def download_patent_pdf(patent_id: str, output_path: Path, timeout: int = 60) -> bool:
    """Download patent PDF from USPTO."""
    url = construct_pdf_url(patent_id)

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Accept': 'application/pdf',
        }
        response = requests.get(url, headers=headers, timeout=timeout, stream=True)
        response.raise_for_status()

        # Verify it's a PDF
        content_type = response.headers.get('content-type', '')
        if 'pdf' not in content_type.lower() and not response.content[:4] == b'%PDF':
            logger.warning(f"Non-PDF response for {patent_id}: {content_type}")
            return False

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            logger.debug(f"Patent PDF not found: {patent_id}")
        else:
            logger.warning(f"HTTP error downloading {patent_id}: {e}")
        return False
    except Exception as e:
        logger.warning(f"Failed to download {patent_id}: {e}")
        return False


def download_patents(
    patents: dict,
    abstracts: dict,
    inventors: dict,
    assignees: dict,
    cpc_codes: dict,
    output_dir: Path,
    max_workers: int = 4,
    delay: float = 0.5,
    download_pdf: bool = True,
):
    """Download patent PDFs and save metadata."""

    logger.info(f"Processing {len(patents)} patents...")

    downloaded = 0
    skipped = 0
    failed = 0

    def process_one(item):
        patent_id, patent_data = item

        # Create safe filename
        safe_id = patent_id.replace('/', '_')
        pdf_path = output_dir / f"{safe_id}.pdf"
        metadata_path = output_dir / f"{safe_id}.json"

        # Skip if already exists
        if metadata_path.exists() and (not download_pdf or pdf_path.exists()):
            return ('skipped', patent_id)

        time.sleep(delay)  # Rate limiting

        # Download PDF
        if download_pdf and not pdf_path.exists():
            if not download_patent_pdf(patent_id, pdf_path):
                # Still save metadata even if PDF fails
                pass

        # Build metadata
        metadata = {
            'patent_id': patent_id,
            'type': patent_data.get('patent_type', ''),
            'date': patent_data.get('patent_date', ''),
            'title': patent_data.get('patent_title', ''),
            'num_claims': patent_data.get('num_claims', ''),
            'abstract': abstracts.get(patent_id, ''),
            'inventors': inventors.get(patent_id, []),
            'assignees': assignees.get(patent_id, []),
            'cpc_classifications': cpc_codes.get(patent_id, []),
            'source': {
                'database': 'USPTO via PatentsView',
                'url': f"https://patents.google.com/patent/US{patent_id}",
                'pdf_url': construct_pdf_url(patent_id),
                'license': 'Public Domain',
            }
        }

        # Save metadata
        try:
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to save metadata for {patent_id}: {e}")
            return ('failed', patent_id)

        # Check if PDF exists (either downloaded or already existed)
        if download_pdf and pdf_path.exists():
            return ('success', patent_id)
        elif not download_pdf:
            return ('success', patent_id)
        else:
            return ('partial', patent_id)  # Metadata saved but PDF failed

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_one, item): item for item in patents.items()}

        for future in as_completed(futures):
            status, patent_id = future.result()
            if status == 'success':
                downloaded += 1
            elif status == 'skipped':
                skipped += 1
            elif status == 'partial':
                downloaded += 1  # Count as success since metadata was saved
            else:
                failed += 1

            total = downloaded + skipped + failed
            if total % 50 == 0:
                logger.info(f"Progress: {total}/{len(patents)} (processed: {downloaded}, skipped: {skipped}, failed: {failed})")

    logger.info(f"Complete! Processed: {downloaded}, Skipped: {skipped}, Failed: {failed}")


def main():
    parser = argparse.ArgumentParser(
        description="Download USPTO patent images and metadata"
    )
    parser.add_argument(
        '-o', '--output',
        type=Path,
        default=Path('./data/patents'),
        help='Output directory for downloaded patents'
    )
    parser.add_argument(
        '-c', '--cache',
        type=Path,
        default=Path('./data/cache'),
        help='Cache directory for PatentsView TSV files'
    )
    parser.add_argument(
        '-l', '--limit',
        type=int,
        default=100,
        help='Limit number of patents to download (default: 100)'
    )
    parser.add_argument(
        '--year-min',
        type=int,
        default=None,
        help='Minimum patent year (e.g., 2020)'
    )
    parser.add_argument(
        '--year-max',
        type=int,
        default=None,
        help='Maximum patent year (e.g., 2024)'
    )
    parser.add_argument(
        '-w', '--workers',
        type=int,
        default=4,
        help='Number of parallel download workers'
    )
    parser.add_argument(
        '-d', '--delay',
        type=float,
        default=0.5,
        help='Delay between requests (seconds)'
    )
    parser.add_argument(
        '--no-pdf',
        action='store_true',
        help='Skip PDF downloads, only download metadata'
    )
    parser.add_argument(
        '--list-only',
        action='store_true',
        help='Only list patents, do not download'
    )
    parser.add_argument(
        '--skip-abstracts',
        action='store_true',
        help='Skip downloading abstracts (faster)'
    )
    parser.add_argument(
        '--skip-inventors',
        action='store_true',
        help='Skip downloading inventor data (faster)'
    )
    parser.add_argument(
        '--skip-assignees',
        action='store_true',
        help='Skip downloading assignee data (faster)'
    )
    parser.add_argument(
        '--skip-cpc',
        action='store_true',
        help='Skip downloading CPC classification codes (faster)'
    )

    args = parser.parse_args()

    # Download PatentsView data files
    logger.info("Fetching PatentsView data files...")

    patent_tsv = download_and_extract_tsv(PATENT_TSV_URL, args.cache, "g_patent.tsv")
    if not patent_tsv:
        logger.error("Failed to download patent data. Exiting.")
        return

    # Load patents
    patents = load_patents(
        patent_tsv,
        limit=args.limit,
        year_min=args.year_min,
        year_max=args.year_max
    )

    if not patents:
        logger.error("No patents found matching criteria")
        return

    patent_ids = set(patents.keys())

    # Load additional metadata (conditionally)
    abstracts = {}
    inventors = {}
    assignees = {}
    cpc_codes = {}

    if not args.skip_abstracts:
        abstract_tsv = download_and_extract_tsv(ABSTRACT_TSV_URL, args.cache, "g_brf_sum_text.tsv")
        if abstract_tsv:
            abstracts = load_abstracts(abstract_tsv, patent_ids)

    if not args.skip_inventors:
        inventor_tsv = download_and_extract_tsv(INVENTOR_TSV_URL, args.cache, "g_inventor.tsv")
        if inventor_tsv:
            inventors = load_inventors(inventor_tsv, patent_ids)

    if not args.skip_assignees:
        assignee_tsv = download_and_extract_tsv(ASSIGNEE_TSV_URL, args.cache, "g_assignee.tsv")
        if assignee_tsv:
            assignees = load_assignees(assignee_tsv, patent_ids)

    if not args.skip_cpc:
        cpc_tsv = download_and_extract_tsv(CPC_TSV_URL, args.cache, "g_cpc_current.tsv")
        if cpc_tsv:
            cpc_codes = load_cpc_codes(cpc_tsv, patent_ids)

    if args.list_only:
        # Just list the patents
        for patent_id, data in patents.items():
            title = data.get('patent_title', 'Unknown')
            date = data.get('patent_date', '')
            abstract = abstracts.get(patent_id, '')[:100] + '...' if abstracts.get(patent_id, '') else ''
            print(f"{patent_id} ({date}): {title}")
            if abstract:
                print(f"  Abstract: {abstract}")
        print(f"\nTotal: {len(patents)} patents")
        return

    # Download patents
    download_patents(
        patents=patents,
        abstracts=abstracts,
        inventors=inventors,
        assignees=assignees,
        cpc_codes=cpc_codes,
        output_dir=args.output,
        max_workers=args.workers,
        delay=args.delay,
        download_pdf=not args.no_pdf,
    )


if __name__ == '__main__':
    main()
