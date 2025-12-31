#!/usr/bin/env python3
"""
CS50 Course Materials Downloader

Downloads lecture videos, slides (PDF), and source code (ZIP) from Internet Archive.

Data Source: https://archive.org/download/academictorrents_52da574b6412862e199abeaea63e51bf8cea2140

License: Educational / Public Access
"""

import os
import json
import time
import argparse
import logging
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, List, Any
from urllib.parse import quote
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Internet Archive item identifier
IA_IDENTIFIER = "academictorrents_52da574b6412862e199abeaea63e51bf8cea2140"
IA_BASE_URL = f"https://archive.org/download/{IA_IDENTIFIER}"
IA_METADATA_URL = f"https://archive.org/metadata/{IA_IDENTIFIER}"

# Lecture topics mapping
LECTURE_INFO = {
    0: {"name": "Scratch", "topics": ["visual programming", "blocks", "sprites"]},
    1: {"name": "C Fundamentals", "topics": ["command line", "data types", "conditionals", "loops", "operators", "IDE"]},
    2: {"name": "C Continued", "topics": ["arrays", "command line arguments", "debugging", "functions", "magic numbers", "variables", "scope"]},
    3: {"name": "Algorithms", "topics": ["linear search", "binary search", "bubble sort", "insertion sort", "selection sort", "merge sort", "recursion", "computational complexity", "GDB"]},
    4: {"name": "Memory", "topics": ["hexadecimal", "pointers", "call stacks", "dynamic memory", "file pointers"]},
    5: {"name": "Data Structures", "topics": ["structures", "custom types", "linked lists", "stacks", "queues", "hash tables", "tries"]},
    6: {"name": "HTTP", "topics": ["internet", "IP", "TCP", "HTTP", "HTML", "CSS"]},
    7: {"name": "Dynamic Programming", "topics": ["algorithms", "optimization"]},
    8: {"name": "Python", "topics": ["python basics", "syntax", "data types"]},
    9: {"name": "Python Continued", "topics": ["Flask", "web framework", "routing"]},
}

# File extensions to download
VIDEO_EXTENSIONS = {'.mp4', '.ogv'}
PDF_EXTENSIONS = {'.pdf'}
ZIP_EXTENSIONS = {'.zip'}


def sanitize_filename(name: str, max_length: int = 100) -> str:
    """Sanitize a string for use as a filename."""
    safe = "".join(c if c.isalnum() or c in ' -_.' else '_' for c in name)
    return safe[:max_length].strip('._')


def get_archive_files() -> List[Dict]:
    """Fetch file listing from Internet Archive metadata API."""
    logger.info(f"Fetching file list from Internet Archive...")

    try:
        response = requests.get(IA_METADATA_URL, timeout=30)
        response.raise_for_status()
        metadata = response.json()
        files = metadata.get('files', [])
        logger.info(f"Found {len(files)} total files in archive")
        return files
    except Exception as e:
        logger.error(f"Failed to fetch metadata: {e}")
        return []


def parse_lecture_number(filename: str) -> Optional[int]:
    """Extract lecture number from filename."""
    # Match patterns like "lecture0", "lecture_0", "week0", "week_0", etc.
    patterns = [
        r'lecture[_\s-]?(\d+)',
        r'week[_\s-]?(\d+)',
        r'^(\d+)[_\s-]',
    ]

    filename_lower = filename.lower()
    for pattern in patterns:
        match = re.search(pattern, filename_lower)
        if match:
            num = int(match.group(1))
            if 0 <= num <= 9:
                return num
    return None


def categorize_files(files: List[Dict]) -> Dict[int, Dict[str, List[Dict]]]:
    """Categorize files by lecture number and type."""
    lectures = {i: {"videos": [], "pdfs": [], "zips": [], "other": []} for i in range(10)}
    uncategorized = {"videos": [], "pdfs": [], "zips": [], "other": []}

    for f in files:
        name = f.get('name', '')
        ext = Path(name).suffix.lower()
        size = int(f.get('size', 0))

        # Skip very small files and metadata files
        if size < 1000:
            continue
        if ext in {'.xml', '.sqlite', '.torrent', '.txt'}:
            continue

        lecture_num = parse_lecture_number(name)

        # Determine file type
        if ext in VIDEO_EXTENSIONS:
            file_type = "videos"
        elif ext in PDF_EXTENSIONS:
            file_type = "pdfs"
        elif ext in ZIP_EXTENSIONS:
            file_type = "zips"
        else:
            file_type = "other"
            continue  # Skip other file types

        file_info = {
            "name": name,
            "size": size,
            "format": f.get('format', ''),
            "ext": ext
        }

        if lecture_num is not None:
            lectures[lecture_num][file_type].append(file_info)
        else:
            uncategorized[file_type].append(file_info)

    return lectures


def download_file(url: str, output_path: Path, timeout: int = 300) -> bool:
    """Download a file from a URL with progress indication."""
    try:
        response = requests.get(url, timeout=timeout, stream=True)
        response.raise_for_status()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        total_size = int(response.headers.get('content-length', 0))

        with open(output_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0 and downloaded % (1024 * 1024) == 0:
                    pct = (downloaded / total_size) * 100
                    logger.debug(f"  {downloaded / (1024*1024):.1f}MB / {total_size / (1024*1024):.1f}MB ({pct:.0f}%)")

        return True
    except Exception as e:
        logger.warning(f"Failed to download {url}: {e}")
        if output_path.exists():
            output_path.unlink()
        return False


def get_best_video(videos: List[Dict], prefer_mp4: bool = True) -> Optional[Dict]:
    """Select the best video file from available options."""
    if not videos:
        return None

    # Sort by format preference and size
    def score(v):
        ext = v.get('ext', '').lower()
        size = v.get('size', 0)
        # Prefer MP4, then larger files
        format_score = 2 if ext == '.mp4' else 1 if ext == '.ogv' else 0
        return (format_score if prefer_mp4 else 0, size)

    videos_sorted = sorted(videos, key=score, reverse=True)
    return videos_sorted[0]


def is_main_lecture(filename: str) -> bool:
    """Check if this is a main lecture video (not a topic short)."""
    name_lower = filename.lower()
    # Main lectures are typically larger and named simply "lecture_X" or "week_X"
    # Topic videos have specific names like "command_line", "data_types", etc.
    topic_indicators = [
        'command', 'data_type', 'loop', 'operator', 'conditional',
        'array', 'function', 'debug', 'variable', 'scope',
        'search', 'sort', 'recursion', 'complexity', 'gdb',
        'pointer', 'memory', 'stack', 'heap', 'file_pointer',
        'struct', 'linked_list', 'queue', 'hash', 'trie',
        'internet', 'tcp', 'http', 'html', 'css', 'ip_',
        'flask', 'python_'
    ]
    return not any(topic in name_lower for topic in topic_indicators)


def download_lectures(
    output_dir: Path,
    lecture_nums: List[int] = None,
    download_videos: bool = True,
    download_pdfs: bool = True,
    download_zips: bool = True,
    main_only: bool = False,
    max_workers: int = 4,
    delay: float = 0.5,
    skip_existing: bool = True,
    list_only: bool = False,
):
    """Download CS50 lecture materials."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get file listing
    all_files = get_archive_files()
    if not all_files:
        logger.error("No files found in archive")
        return

    # Categorize files
    lectures = categorize_files(all_files)

    # Filter lectures if specified
    if lecture_nums is None:
        lecture_nums = list(range(10))

    stats = {"downloaded": 0, "skipped": 0, "failed": 0}

    for lecture_num in lecture_nums:
        lecture_data = lectures.get(lecture_num, {})
        lecture_info = LECTURE_INFO.get(lecture_num, {"name": f"Lecture {lecture_num}", "topics": []})
        lecture_name = lecture_info["name"]

        logger.info(f"\n=== Lecture {lecture_num}: {lecture_name} ===")

        lecture_dir = output_dir / f"lecture_{lecture_num}"

        if list_only:
            print(f"\nLecture {lecture_num}: {lecture_name}")
            if download_videos:
                for v in lecture_data.get("videos", []):
                    main_tag = "[MAIN]" if is_main_lecture(v['name']) else "[TOPIC]"
                    print(f"  {main_tag} VIDEO: {v['name']} ({v['size'] / (1024*1024):.1f}MB)")
            if download_pdfs:
                for p in lecture_data.get("pdfs", []):
                    print(f"  PDF: {p['name']} ({p['size'] / (1024*1024):.1f}MB)")
            if download_zips:
                for z in lecture_data.get("zips", []):
                    print(f"  ZIP: {z['name']} ({z['size'] / (1024*1024):.1f}MB)")
            continue

        # Prepare metadata
        lecture_metadata = {
            "lecture_id": lecture_num,
            "title": lecture_name,
            "description": f"CS50 Lecture {lecture_num}: {lecture_name}",
            "topics": lecture_info["topics"],
            "files": {},
            "topic_videos": [],
            "source": {
                "archive": "Internet Archive",
                "url": IA_BASE_URL,
                "course": "CS50 2017",
                "identifier": IA_IDENTIFIER
            }
        }

        # Download videos
        if download_videos:
            videos = lecture_data.get("videos", [])

            if main_only:
                # Only download main lecture video
                main_videos = [v for v in videos if is_main_lecture(v['name'])]
                best = get_best_video(main_videos)
                if best:
                    videos = [best]
                else:
                    videos = []

            for video in videos:
                video_name = video['name']
                video_url = f"{IA_BASE_URL}/{quote(video_name)}"

                # Determine if main lecture or topic
                is_main = is_main_lecture(video_name)
                if is_main:
                    video_path = lecture_dir / f"lecture_{lecture_num}_{sanitize_filename(lecture_name)}.mp4"
                else:
                    topics_dir = lecture_dir / "topics"
                    safe_name = sanitize_filename(Path(video_name).stem)
                    video_path = topics_dir / f"{safe_name}.mp4"

                if skip_existing and video_path.exists():
                    logger.info(f"  Skipping (exists): {video_path.name}")
                    stats["skipped"] += 1
                    continue

                logger.info(f"  Downloading: {video_name} ({video['size'] / (1024*1024):.1f}MB)")
                time.sleep(delay)

                if download_file(video_url, video_path):
                    stats["downloaded"] += 1
                    if is_main:
                        lecture_metadata["files"]["video"] = video_path.name
                    else:
                        lecture_metadata["topic_videos"].append({
                            "name": Path(video_name).stem,
                            "file": f"topics/{video_path.name}",
                            "size": video['size']
                        })
                else:
                    stats["failed"] += 1

        # Download PDFs
        if download_pdfs:
            pdfs = lecture_data.get("pdfs", [])
            for pdf in pdfs:
                pdf_name = pdf['name']
                pdf_url = f"{IA_BASE_URL}/{quote(pdf_name)}"
                pdf_path = lecture_dir / f"lecture_{lecture_num}_{sanitize_filename(lecture_name)}.pdf"

                if skip_existing and pdf_path.exists():
                    logger.info(f"  Skipping (exists): {pdf_path.name}")
                    stats["skipped"] += 1
                    continue

                logger.info(f"  Downloading: {pdf_name} ({pdf['size'] / (1024*1024):.1f}MB)")
                time.sleep(delay)

                if download_file(pdf_url, pdf_path):
                    stats["downloaded"] += 1
                    lecture_metadata["files"]["pdf"] = pdf_path.name
                else:
                    stats["failed"] += 1

        # Download ZIPs
        if download_zips:
            zips = lecture_data.get("zips", [])
            for zip_file in zips:
                zip_name = zip_file['name']
                zip_url = f"{IA_BASE_URL}/{quote(zip_name)}"
                src_dir = lecture_dir / "src"
                zip_path = src_dir / f"lecture_{lecture_num}_src.zip"

                if skip_existing and zip_path.exists():
                    logger.info(f"  Skipping (exists): {zip_path.name}")
                    stats["skipped"] += 1
                    continue

                logger.info(f"  Downloading: {zip_name} ({zip_file['size'] / 1024:.1f}KB)")
                time.sleep(delay)

                if download_file(zip_url, zip_path):
                    stats["downloaded"] += 1
                    lecture_metadata["files"]["source_zip"] = f"src/{zip_path.name}"
                else:
                    stats["failed"] += 1

        # Save metadata
        if not list_only and (lecture_metadata["files"] or lecture_metadata["topic_videos"]):
            json_path = lecture_dir / f"lecture_{lecture_num}.json"
            lecture_dir.mkdir(parents=True, exist_ok=True)
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(lecture_metadata, f, indent=2, ensure_ascii=False)
            logger.info(f"  Saved metadata: {json_path.name}")

    if not list_only:
        logger.info(f"\n=== Download Complete ===")
        logger.info(f"Downloaded: {stats['downloaded']}, Skipped: {stats['skipped']}, Failed: {stats['failed']}")


def main():
    parser = argparse.ArgumentParser(
        description="Download CS50 course materials from Internet Archive"
    )
    parser.add_argument(
        '--videos',
        action='store_true',
        help='Download video files (MP4)'
    )
    parser.add_argument(
        '--pdfs',
        action='store_true',
        help='Download PDF slides'
    )
    parser.add_argument(
        '--zips',
        action='store_true',
        help='Download source code ZIPs'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Download all content types'
    )
    parser.add_argument(
        '--lectures',
        type=int,
        nargs='+',
        choices=range(10),
        metavar='N',
        help='Specific lecture numbers to download (0-9)'
    )
    parser.add_argument(
        '--main-only',
        action='store_true',
        help='Only download main lecture videos, skip topic shorts'
    )
    parser.add_argument(
        '-o', '--output',
        type=Path,
        default=Path('./data'),
        help='Base output directory (default: ./data)'
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
        '--no-skip',
        action='store_true',
        help='Re-download existing files'
    )
    parser.add_argument(
        '--list-only',
        action='store_true',
        help='Only list files, do not download'
    )

    args = parser.parse_args()

    # Default to downloading all if none specified
    if not args.videos and not args.pdfs and not args.zips and not args.all:
        args.all = True

    if args.all:
        args.videos = True
        args.pdfs = True
        args.zips = True

    skip_existing = not args.no_skip

    download_lectures(
        output_dir=args.output,
        lecture_nums=args.lectures,
        download_videos=args.videos,
        download_pdfs=args.pdfs,
        download_zips=args.zips,
        main_only=args.main_only,
        max_workers=args.workers,
        delay=args.delay,
        skip_existing=skip_existing,
        list_only=args.list_only,
    )


if __name__ == '__main__':
    main()
