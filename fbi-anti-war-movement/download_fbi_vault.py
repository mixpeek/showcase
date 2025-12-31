#!/usr/bin/env python3
"""
FBI Vault PDF Downloader
Downloads all PDF documents from an FBI Vault subject page.
Uses cloudscraper to bypass Cloudflare protection.
"""

import os
import re
import urllib.parse
from pathlib import Path
import cloudscraper
from bs4 import BeautifulSoup

# Configuration
BASE_URL = "https://vault.fbi.gov"
SUBJECT_URL = f"{BASE_URL}/Edward%20Abbey"
OUTPUT_DIR = Path(__file__).parent


def main():
    print(f"FBI Vault PDF Downloader")
    print(f"Target: {SUBJECT_URL}")
    print(f"Output: {OUTPUT_DIR}")
    print("-" * 50)

    # Create a cloudscraper session
    scraper = cloudscraper.create_scraper(
        browser={
            'browser': 'chrome',
            'platform': 'darwin',
            'desktop': True,
        }
    )

    # Navigate to main subject page
    print(f"\nFetching: {SUBJECT_URL}")
    response = scraper.get(SUBJECT_URL)

    if response.status_code != 200:
        print(f"Failed to fetch main page: HTTP {response.status_code}")
        return

    soup = BeautifulSoup(response.text, 'html.parser')

    # Find all part links
    all_links = soup.find_all('a', href=True)

    part_info = []
    for link in all_links:
        href = link.get('href', '')
        text = link.get_text(strip=True)

        if 'edward%20abbey' in href.lower() and 'part' in (href.lower() + text.lower()):
            if href.startswith('/'):
                href = BASE_URL + href
            # Extract part number
            decoded = urllib.parse.unquote(href + " " + text)
            match = re.search(r'Part[_\s]*(\d+)', decoded, re.IGNORECASE)
            part_num = match.group(1) if match else None
            if part_num:
                # Get the folder URL, not the /view URL
                folder_url = href.replace('/view', '')
                part_info.append((folder_url, part_num, text.strip()))

    # Remove duplicates
    seen = set()
    unique_parts = []
    for item in part_info:
        key = item[1]
        if key not in seen:
            seen.add(key)
            unique_parts.append(item)

    print(f"\nFound {len(unique_parts)} parts:")
    for url, part_num, text in sorted(unique_parts, key=lambda x: int(x[1])):
        print(f"  Part {part_num}: {text}")

    # Download each part
    for folder_url, part_num, _ in sorted(unique_parts, key=lambda x: int(x[1])):
        print(f"\n{'='*50}")
        print(f"Processing Part {part_num}...")

        filename = f"edward-abbey-part-{part_num.zfill(2)}.pdf"
        filepath = OUTPUT_DIR / filename

        if filepath.exists():
            print(f"  Skipping (exists): {filename}")
            continue

        # Navigate to folder page
        print(f"  Fetching folder: {folder_url}")
        folder_response = scraper.get(folder_url)

        if folder_response.status_code != 200:
            print(f"  Failed: HTTP {folder_response.status_code}")
            continue

        folder_soup = BeautifulSoup(folder_response.text, 'html.parser')

        # Look for download link
        download_link = None

        # Try various patterns
        for link in folder_soup.find_all('a', href=True):
            href = link.get('href', '')
            text = link.get_text(strip=True).lower()

            if 'at_download' in href or 'download' in text or href.endswith('.pdf'):
                download_link = href if href.startswith('http') else BASE_URL + href
                print(f"  Found download link: {download_link}")
                break

        if not download_link:
            # Construct download URL from folder URL
            download_link = folder_url + '/at_download/file'
            print(f"  Trying constructed URL: {download_link}")

        # Download the PDF
        print(f"  Downloading...")
        try:
            pdf_response = scraper.get(download_link, stream=True)

            if pdf_response.status_code == 200:
                content_type = pdf_response.headers.get('content-type', '')
                if 'pdf' in content_type or 'octet-stream' in content_type:
                    with open(filepath, 'wb') as f:
                        for chunk in pdf_response.iter_content(chunk_size=8192):
                            f.write(chunk)

                    size_mb = filepath.stat().st_size / (1024 * 1024)
                    print(f"  Saved: {filename} ({size_mb:.2f} MB)")
                else:
                    print(f"  Not a PDF: content-type={content_type}")
            else:
                print(f"  Download failed: HTTP {pdf_response.status_code}")
        except Exception as e:
            print(f"  Error: {e}")

    print(f"\n{'='*50}")
    print("Download complete!")

    # List downloaded files
    pdfs = list(OUTPUT_DIR.glob("*.pdf"))
    print(f"\nDownloaded {len(pdfs)} PDF files:")
    for pdf in sorted(pdfs):
        size_mb = pdf.stat().st_size / (1024 * 1024)
        print(f"  {pdf.name} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
