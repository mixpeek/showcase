# NYPL Public Domain Photo Archive

Build a semantic image search engine using the New York Public Library's public domain photo collection.

## Overview

This example demonstrates how to:
1. Download ~187,000 public domain images from NYPL Digital Collections
2. Capture comprehensive metadata for each image
3. Create a Mixpeek collection for storing image assets
4. Configure feature extractors for image embeddings
5. Build an index for similarity search

## Data Source

Images are sourced from the [NYPL Public Domain Collections](https://www.nypl.org/research/resources/public-domain-collections):
- **License**: CC0 (Public Domain)
- **Total Images**: ~187,000 public domain items
- **Content**: Photographs, prints, maps, manuscripts, and more
- **Data Snapshot**: December 30, 2015 release + API updates

## Prerequisites

```bash
pip install requests tqdm
```

## Step 1: Download Images

The `download_nypl.py` script fetches images from NYPL's Digital Collections API.

### Download All Public Domain Images (~187k)

```bash
# Download images at medium resolution (recommended for ML)
python download_nypl.py -o ./data/images -s medium

# Download high resolution images (warning: very large files)
python download_nypl.py -o ./data/images -s high
```

### Download a Subset

```bash
# Download first 1000 images
python download_nypl.py -o ./data/images -l 1000

# Preview what will be downloaded
python download_nypl.py --list-only -l 100

# Download photos only (filter by genre)
python download_nypl.py --genre "Photographs" -l 5000
```

### Script Options

| Option | Description | Default |
|--------|-------------|---------|
| `-o, --output` | Output directory for images | `./data/images` |
| `-c, --cache` | Cache directory for metadata | `./data/cache` |
| `-m, --metadata-dir` | Directory for JSON metadata | `./data/metadata` |
| `-s, --size` | Image size: `thumb`, `medium`, `large`, `high` | `medium` |
| `--genre` | Filter by genre (e.g., "Photographs") | `None` |
| `-w, --workers` | Parallel download threads | `8` |
| `-d, --delay` | Delay between requests (sec) | `0.05` |
| `-l, --limit` | Max images to download | `None` |
| `--list-only` | List images without downloading | `false` |
| `--use-api` | Use live API instead of GitHub data | `false` |

### Size Options

| Size | Description |
|------|-------------|
| `thumb` | Thumbnail (~150px) |
| `medium` | Medium resolution (~760px) |
| `large` | Large resolution (~1600px) |
| `high` | Original high-res TIFF (200MB+ each) |

## Step 2: Create Mixpeek Resources

See `ingest_mixpeek.py` for creating Mixpeek collection, feature extractors, and index.

## Step 3: Search Examples

*Coming soon: Example queries and search interface.*

## Project Structure

```
nypl-photo-archive/
├── README.md
├── download_nypl.py     # Image downloader script
├── ingest_mixpeek.py    # Mixpeek resource setup
└── data/
    ├── cache/           # Cached NYPL metadata
    ├── metadata/        # Per-image JSON metadata
    └── images/          # Downloaded images
```

## Metadata Structure

Each image includes rich metadata:

```json
{
  "uuid": "510d47e1-5db3-a3d9-e040-e00a18064a99",
  "title": "Fifth Avenue, 42nd Street",
  "date": "1898",
  "contributors": ["Byron Company"],
  "subjects": ["Streets", "Architecture"],
  "genres": ["Photographs"],
  "collection": "Photographic views of New York City",
  "image_url": "https://images.nypl.org/...",
  "source": {
    "institution": "The New York Public Library",
    "url": "https://digitalcollections.nypl.org/items/...",
    "license": "CC0 Public Domain"
  }
}
```

## Attribution

While not required (CC0), NYPL requests consideration for attribution:

> From The New York Public Library

## Resources

- [NYPL Digital Collections](https://digitalcollections.nypl.org/)
- [NYPL Digital Collections API](https://api.repo.nypl.org/)
- [Public Domain Data on GitHub](https://github.com/NYPL-publicdomain/data-and-utilities)
- [About Public Domain Collections](https://www.nypl.org/research/resources/public-domain-collections)
