# Met Museum Public Domain Artworks

Build a semantic search engine for art using The Metropolitan Museum of Art's public domain collection.

## Overview

This example demonstrates how to:
1. Download ~470k public domain artworks from The Met's collection
2. Create a Mixpeek collection for storing image assets
3. Configure feature extractors for image embeddings
4. Build an index for similarity search
5. Create a retriever for natural language and image-based queries

## Data Source

Images are sourced from [The Met Collection API](https://metmuseum.github.io/):
- **License**: CC0 (Public Domain) for public domain works
- **Total Objects**: ~470,000+ objects with images
- **Public Domain Subset**: ~406,000+ public domain artworks

## Prerequisites

```bash
pip install requests mixpeek
```

## Step 1: Download Artworks

The `download_met.py` script fetches artworks from The Met's Collection API.

### Download All Public Domain Artworks

```bash
# Download all public domain artworks (full resolution)
python download_met.py -o ./data

# Download with smaller images (faster)
python download_met.py -o ./data -s small
```

### Download by Department

```bash
# List all departments
python download_met.py --list-departments

# Download only European Paintings (dept 11)
python download_met.py -o ./data -d 11

# Download Photographs and Prints (depts 19 and 9)
python download_met.py -o ./data -d 19,9

# Preview what will be downloaded
python download_met.py -d 11 --list-only -l 100
```

### Script Options

| Option | Description | Default |
|--------|-------------|---------|
| `-o, --output` | Output directory for downloads | `./data` |
| `-s, --size` | Image size: `primary` (full) or `small` | `primary` |
| `-d, --departments` | Department IDs (comma-separated) | All |
| `-w, --workers` | Parallel download threads | `8` |
| `--delay` | Delay between requests (sec) | `0.05` |
| `-l, --limit` | Max objects to process | `None` |
| `--additional-images` | Download extra images per object | `false` |
| `--list-departments` | List all departments and exit | - |
| `--list-only` | List object IDs without downloading | `false` |
| `--no-cache` | Don't use cached object IDs | `false` |

### Department Reference

| ID | Department |
|----|------------|
| 1 | American Decorative Arts |
| 3 | Ancient Near Eastern Art |
| 4 | Arms and Armor |
| 5 | Arts of Africa, Oceania, and the Americas |
| 6 | Asian Art |
| 7 | The Cloisters |
| 8 | Costume Institute |
| 9 | Drawings and Prints |
| 10 | Egyptian Art |
| 11 | European Paintings |
| 12 | European Sculpture and Decorative Arts |
| 13 | Greek and Roman Art |
| 14 | Islamic Art |
| 15 | Robert Lehman Collection |
| 16 | Library |
| 17 | Medieval Art |
| 18 | Musical Instruments |
| 19 | Photographs |
| 21 | Modern and Contemporary Art |

## Step 2: Create Mixpeek Resources

*Coming soon: Scripts for creating Mixpeek collection, feature extractors, index, and retriever.*

## Step 3: Search Examples

*Coming soon: Example queries and search interface.*

## Project Structure

```
met-museum/
├── README.md
├── download_met.py      # Artwork downloader script
├── setup.py             # Mixpeek resource setup (coming soon)
├── search.py            # Search interface (coming soon)
└── data/
    ├── cache/           # Cached object IDs
    ├── images/          # Downloaded artwork images
    └── metadata/        # JSON metadata for each artwork
```

## Metadata Fields

Each downloaded artwork includes a JSON metadata file with:

- **Object Info**: title, date, medium, dimensions, classification
- **Artist Info**: name, bio, nationality, dates
- **Geographic Info**: culture, period, dynasty, country, city
- **Museum Info**: department, accession number, credit line, gallery number
- **Tags**: subject tags assigned by The Met
- **Images**: URLs for primary and additional images
- **Links**: Object URL, API URL, Wikidata URL

## Attribution

While not required for CC0 works, The Met appreciates attribution:

> The Metropolitan Museum of Art, CC0

## API Rate Limits

The Met API allows up to 80 requests per second. The default settings are conservative to avoid issues:
- 8 parallel workers
- 0.05s delay between requests

## Resources

- [The Met Collection API](https://metmuseum.github.io/)
- [The Met Open Access Initiative](https://www.metmuseum.org/about-the-met/policies-and-documents/open-access)
- [The Met Collection](https://www.metmuseum.org/art/collection)
