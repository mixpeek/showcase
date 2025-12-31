# CS50 Computer Science Course Curriculum

Build a multimodal search engine for Harvard's CS50 course materials using Internet Archive's educational collection.

## Overview

This example demonstrates how to:
1. Download CS50 lecture videos, slides (PDF), source code (ZIP), and transcripts from Internet Archive
2. Organize content by lecture/course ID
3. Create a Mixpeek collection for storing multimodal educational assets
4. Configure feature extractors for video, document, and code embeddings
5. Build an index for similarity search across lecture materials

## Data Source

- **Archive**: [CS50 2017 Course Materials](https://archive.org/download/academictorrents_52da574b6412862e199abeaea63e51bf8cea2140)
- **Content**: Harvard CS50 Introduction to Computer Science (2017 edition)
- **License**: Educational / Public Access

### Course Structure

| Lecture | Topic | Content |
|---------|-------|---------|
| 0 | Scratch | Visual programming introduction |
| 1 | C Fundamentals | Command line, data types, loops, operators |
| 2 | C Continued | Arrays, functions, debugging, scope |
| 3 | Algorithms | Search, sort, recursion, complexity |
| 4 | Memory | Pointers, hexadecimal, dynamic allocation |
| 5 | Data Structures | Linked lists, stacks, queues, hash tables, tries |
| 6 | HTTP | Internet, IP, HTTP, HTML, CSS |
| 7 | Dynamic Programming | Advanced algorithms |
| 8 | Python | Python programming basics |
| 9 | Python Continued | Flask web framework |

## Prerequisites

```bash
pip install requests mixpeek
```

## Step 1: Download Content

The `download_cs50.py` script fetches course materials from Internet Archive.

### Download Everything

```bash
# Download all lectures (videos, PDFs, ZIPs)
python download_cs50.py --all -o ./data

# Specify content types
python download_cs50.py --videos --pdfs --zips -o ./data
```

### Download Specific Lectures

```bash
# Download only lecture 0 (Scratch)
python download_cs50.py --lectures 0

# Download lectures 1-3
python download_cs50.py --lectures 1 2 3

# Download only Python lectures
python download_cs50.py --lectures 8 9
```

### Download Specific Content Types

```bash
# Videos only
python download_cs50.py --videos

# PDFs only (slides)
python download_cs50.py --pdfs

# Source code ZIPs only
python download_cs50.py --zips

# Main lectures only (skip topic videos)
python download_cs50.py --videos --main-only
```

### Preview Available Files

```bash
# List files without downloading
python download_cs50.py --list-only
```

### Script Options

| Option | Description | Default |
|--------|-------------|---------|
| `--videos` | Download MP4 video files | `false` |
| `--pdfs` | Download PDF slide files | `false` |
| `--zips` | Download source code ZIPs | `false` |
| `--all` | Download all content types | `true` (if none specified) |
| `--lectures` | Specific lecture numbers (0-9) | All lectures |
| `--main-only` | Only main lecture videos, skip topics | `false` |
| `-o, --output` | Base output directory | `./data` |
| `-w, --workers` | Parallel download threads | `4` |
| `-d, --delay` | Delay between requests (sec) | `0.5` |
| `--no-skip` | Re-download existing files | `false` |
| `--list-only` | List files without downloading | `false` |

## Output Structure

```
cs50-curriculum/
├── README.md
├── download_cs50.py          # Content downloader script
├── ingest_mixpeek.py         # Mixpeek ingestion script
└── data/
    ├── lecture_0/            # Scratch
    │   ├── lecture_0_scratch.mp4
    │   ├── lecture_0_scratch.pdf
    │   ├── lecture_0_scratch.json
    │   └── src/
    │       └── lecture_0_src.zip
    ├── lecture_1/            # C Fundamentals
    │   ├── lecture_1_c.mp4
    │   ├── lecture_1_c.pdf
    │   ├── lecture_1_c.json
    │   ├── topics/           # Topic-specific videos
    │   │   ├── command_line.mp4
    │   │   ├── command_line.json
    │   │   ├── data_types.mp4
    │   │   ├── data_types.json
    │   │   └── ...
    │   └── src/
    │       └── lecture_1_src.zip
    └── ...
```

## Metadata Format

### Lecture Metadata (JSON)

```json
{
  "lecture_id": 1,
  "title": "C Fundamentals",
  "description": "Introduction to C programming language",
  "topics": ["command line", "data types", "loops", "operators"],
  "files": {
    "video": "lecture_1_c.mp4",
    "pdf": "lecture_1_c.pdf",
    "source_zip": "lecture_1_src.zip"
  },
  "topic_videos": [
    {
      "name": "Command Line",
      "file": "topics/command_line.mp4",
      "duration": "29:36"
    }
  ],
  "source": {
    "archive": "Internet Archive",
    "url": "https://archive.org/download/academictorrents_52da574b6412862e199abeaea63e51bf8cea2140",
    "course": "CS50 2017"
  }
}
```

## Step 2: Ingest into Mixpeek

The `ingest_mixpeek.py` script creates Mixpeek resources and uploads sample data.

### Setup

1. Get your API key from [mixpeek.com](https://mixpeek.com)
2. Update the script with your credentials:

```python
API_KEY = "your_api_key_here"
NAMESPACE = "your_namespace_here"
```

### Run Ingestion

```bash
python ingest_mixpeek.py
```

This will:
1. Create a `cs50-lectures` bucket for lecture videos
2. Create a `cs50-slides` bucket for PDF slides
3. Create collections with multimodal feature extractors
4. Upload sample lectures for processing
5. Submit batches for embedding generation

### Resources Created

| Resource | Name | Description |
|----------|------|-------------|
| Video Bucket | `cs50-lectures` | Lecture videos + metadata |
| PDF Bucket | `cs50-slides` | Lecture slides + metadata |
| Video Collection | `cs50-lectures-collection` | Searchable video embeddings |
| PDF Collection | `cs50-slides-collection` | Searchable document embeddings |

## Step 3: Search Examples

Once ingestion is complete, you can search using the Mixpeek API:

```python
import requests

# Search for content about pointers
response = requests.post(
    "https://api.mixpeek.com/v1/collections/search",
    headers={"Authorization": "Bearer YOUR_API_KEY"},
    json={
        "collection_id": "your_collection_id",
        "query": "how do pointers work in C",
        "top_k": 10
    }
)
```

## Use Cases

- **Educational Search**: Find specific programming concepts across lectures
- **Cross-Modal Learning**: Search videos using text queries about code
- **Concept Clustering**: Group related programming topics
- **Study Aid**: Find relevant lecture segments for specific topics
- **Code Examples**: Locate source code related to lecture content

## Attribution

Content sourced from Harvard's CS50 course, available on [Internet Archive](https://archive.org).

> CS50 is Harvard University's introduction to the intellectual enterprises of computer science and the art of programming.

## Resources

- [Internet Archive CS50 Collection](https://archive.org/download/academictorrents_52da574b6412862e199abeaea63e51bf8cea2140)
- [CS50 Official Website](https://cs50.harvard.edu)
- [Mixpeek Documentation](https://docs.mixpeek.com)
