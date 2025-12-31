# National Portrait Gallery Search

Build a semantic image search engine for portrait photography using the National Gallery of Art's open-access collection.

![Demo](assets/npg-animation.webm)

**🔍 Try the live demo: [https://mxp.co/r/npg](https://mxp.co/r/npg)**

## Overview

This example demonstrates how to:
1. Download open-access portrait images from the National Gallery of Art
2. Ingest images into Mixpeek with metadata
3. Create a searchable collection using semantic embeddings
4. Query using natural language or image similarity

## Data Source

Images are sourced from the [National Gallery of Art Open Data Program](https://github.com/NationalGalleryOfArt/opendata):
- **License**: CC0 (Public Domain)
- **Total Images**: ~120,000 open-access images
- **Portrait Subset**: ~15,000+ portrait-specific images

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

## Step 1: Download Images

The `download_nga.py` script fetches images from NGA's IIIF Image API.

### Download All Open Access Images (~120k)

```bash
# Download all images at 1024px width
python download_nga.py -o ./data/images -s "1024,"

# Download full resolution (warning: very large files)
python download_nga.py -o ./data/images -s "full"
```

### Download Portraits Only (~15k)

```bash
# Filter for portrait images only
python download_nga.py -o ./data/images --portraits-only

# Preview what will be downloaded
python download_nga.py --portraits-only --list-only -l 100
```

### Script Options

| Option | Description | Default |
|--------|-------------|---------|
| `-o, --output` | Output directory for images | `./data/images` |
| `-c, --cache` | Cache directory for CSV metadata | `./data/cache` |
| `-s, --size` | IIIF size parameter | `1024,` |
| `--portraits-only` | Filter for portrait images | `false` |
| `-w, --workers` | Parallel download threads | `4` |
| `-d, --delay` | Delay between requests (sec) | `0.1` |
| `-l, --limit` | Max images to download | `None` |
| `--list-only` | List images without downloading | `false` |

### Size Options

| Format | Description |
|--------|-------------|
| `full` | Full resolution (can be 5000+ px) |
| `1024,` | Max width 1024px, proportional height |
| `,1024` | Max height 1024px, proportional width |
| `512,512` | Fit within 512x512 bounding box |

## Step 2: Ingest into Mixpeek

Upload the downloaded images to Mixpeek with their metadata:

```bash
python ingest_mixpeek.py
```

The script will:
- Create a new bucket (or use existing)
- Upload images with metadata (title, artist, date, etc.)
- Handle deduplication automatically
- Process images in parallel batches for efficiency

## Step 3: Search Your Collection

Once ingested, you can search your collection:

**Try the public demo retriever:** [https://mxp.co/r/npg](https://mxp.co/r/npg)

Example queries:
- "portraits of women wearing blue dresses"
- "elderly man with white beard"
- "young children playing"
- "formal royal portraits"

## Project Structure

```
portrait-gallery/
├── README.md
├── download_nga.py      # Download images from NGA
├── ingest_mixpeek.py    # Upload images to Mixpeek
└── data/
    ├── cache/           # Cached NGA metadata CSVs (auto-created)
    └── images/          # Downloaded images + JSON metadata (gitignored)
```

## Attribution

While not required (CC0), the National Gallery of Art requests consideration for attribution:

> National Gallery of Art Open Access Images

## Resources

- [NGA Open Data Program](https://github.com/NationalGalleryOfArt/opendata)
- [NGA Free Images & Open Access](https://www.nga.gov/artworks/free-images-and-open-access)
- [IIIF Image API](https://iiif.io/api/image/3.0/)
