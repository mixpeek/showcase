# CS50 Course Content Search

Build a multimodal search engine for educational content using Harvard's CS50 course materials from the Internet Archive.

## Overview

This example demonstrates how to:
1. Download complete course materials from Internet Archive (videos, slides, code)
2. Ingest multimodal content into Mixpeek with structured metadata
3. Create a searchable educational library using semantic embeddings
4. Query across video, slides, and code using natural language

## Data Source

Course materials are sourced from the [Internet Archive CS50 2017 Collection](https://archive.org/details/academictorrents_52da574b6412862e199abeaea63e51bf8cea2140):
- **License**: Educational use (Internet Archive)
- **Course**: Harvard CS50 2017 - Introduction to Computer Science
- **Total Content**: 12+ lectures with videos, slides, and source code
- **Video Format**: MP4 (193-428 MB per lecture)
- **Slides Format**: PDF (501 KB-136 MB)
- **Code Format**: ZIP archives

## Prerequisites

- Python 3.7+
- Mixpeek API Key ([Sign up for free](https://mixpeek.com))

```bash
pip install requests
```

Set your Mixpeek credentials as environment variables:

```bash
export MIXPEEK_API_KEY="your_api_key_here"
export MIXPEEK_NAMESPACE="your_namespace_here"
```

## Step 1: Download Course Materials

The [download_cs50.py](download_cs50.py) script fetches all course content from the Internet Archive.

### Download All Content

```bash
# Download videos, slides, and code
python download_cs50.py
```

The script will:
- Fetch metadata from Internet Archive API
- Download MP4 videos to `data/videos/`
- Download PDF slides to `data/slides/`
- Download ZIP code archives to `data/code/`
- Save metadata JSON for each file to `data/metadata/`
- Skip already downloaded files automatically
- Resume interrupted downloads

### Script Features

- **Parallel Downloads**: 4 concurrent workers for faster downloads
- **Progress Tracking**: Real-time progress for each file
- **Smart Resume**: Skips completed files, validates file sizes
- **Retry Logic**: 3 attempts per file with exponential backoff
- **Metadata Extraction**: Lecture numbers, file sizes, MD5 hashes

### Expected Output

```
data/
├── videos/          # MP4 lecture videos (~2-4 GB total)
├── slides/          # PDF presentation slides (~100-500 MB total)
├── code/            # ZIP source code archives (~50-200 MB total)
├── metadata/        # JSON metadata for each file
└── cache/           # Download cache and temp files
```

## Step 2: Ingest into Mixpeek

Upload the downloaded course materials to Mixpeek:

```bash
python ingest_mixpeek.py
```

The script will:
- Create a new bucket for CS50 content (or use existing)
- Upload videos with temporal embeddings for searchable segments
- Upload slides with page-level embeddings
- Upload code archives with file-level embeddings
- Attach structured metadata (lecture number, topic, format)
- Handle deduplication automatically
- Process content in parallel batches for efficiency

## Step 3: Search Your Collection

Once ingested, you can search across all course materials:

Example queries:
- **Cross-modal**: "explain bubble sort algorithm" → finds video segments, slide pages, and code examples
- **Code search**: "binary search implementation in C" → finds relevant code files and lecture segments
- **Concept search**: "memory allocation and pointers" → returns related videos and slides
- **Lecture discovery**: "web development with Flask" → finds Python web programming content

## Project Structure

```
learning/
├── README.md
├── download_cs50.py      # Download course materials from Archive.org
├── ingest_mixpeek.py     # Upload content to Mixpeek
└── data/
    ├── videos/           # Downloaded MP4 lecture videos
    ├── slides/           # Downloaded PDF slides
    ├── code/             # Downloaded ZIP code archives
    ├── metadata/         # JSON metadata for each file (lecture #, size, etc.)
    └── cache/            # Download cache (auto-created)
```

## Course Content

The CS50 2017 collection includes 13 lectures covering the complete introduction to computer science curriculum. See [syllabus.json](syllabus.json) for the complete course mapping.

### Lecture Overview

| Week | Lecture | Topics | Videos | Materials |
|------|---------|--------|--------|-----------|
| 0 | Scratch | Computational thinking, visual programming | 1 | slides |
| 1 | C | Data types, operators, conditionals, loops | 13 | slides, code |
| 2 | C, continued | Arrays, functions, debugging, cryptography | 12 | slides, code |
| 3 | Algorithms | Search, sort, recursion, Big O | 20 | slides, code |
| 4 | Memory | Pointers, heap, stack, file I/O | 10 | slides, code |
| 5 | Data Structures | Linked lists, hash tables, tries | 16 | slides, code |
| 6 | HTTP | Web protocols, HTML, CSS | 11 | slides |
| 7 | Dynamic Programming | Flask, MVC, web apps | 1 | slides |
| 8 | Python | Python basics, OOP, web scraping | 1 | slides |
| 9 | SQL | Databases, queries, relationships | 5 | slides, code |
| 10 | JavaScript | DOM, events, AJAX | 7 | slides, code |
| 11 | The End | Security, scalability, final projects | 5 | slides, code |
| 12 | Special Topics | AR/VR, computer vision, Java, more | 12 | slides, code |

Each lecture includes:
- Full video recording (MP4) + topic-specific segments
- Slide deck (PDF)
- Source code examples (ZIP)
- Structured metadata (lecture number, topic, file info)

## Attribution

Content from Harvard University's CS50 course, archived via Internet Archive and Academic Torrents.

> Harvard CS50 2017 - Introduction to Computer Science
> Internet Archive: academictorrents_52da574b6412862e199abeaea63e51bf8cea2140

## Resources

- [CS50 on Internet Archive](https://archive.org/details/academictorrents_52da574b6412862e199abeaea63e51bf8cea2140)
- [CS50 Official Website](https://cs50.harvard.edu/)
- [Mixpeek Documentation](https://docs.mixpeek.com)
