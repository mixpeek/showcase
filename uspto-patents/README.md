# USPTO Patent Search

Build a multimodal search engine for U.S. patents using USPTO patent images and text data.

## Overview

This example demonstrates how to:
1. Download patent metadata (titles, abstracts, inventors, classifications) from PatentsView
2. Download patent PDF images from USPTO's image server
3. Create a Mixpeek collection for storing patent documents
4. Configure feature extractors for text and image embeddings
5. Build an index for similarity search
6. Create a retriever for natural language and image-based patent queries

## Data Sources

### PatentsView (Metadata)
- **Source**: [PatentsView Data Download](https://patentsview.org/download/data-download-tables)
- **License**: CC BY 4.0
- **Content**: Patent titles, abstracts, claims, inventors, assignees, CPC classifications
- **Coverage**: Granted patents from 1976 to present

### USPTO Image Server (Patent PDFs)
- **Source**: USPTO image-ppubs server
- **License**: Public Domain
- **Content**: Full patent documents as PDFs (drawings, specifications, claims)
- **Coverage**: All granted US patents

## Prerequisites

```bash
pip install requests mixpeek
```

## Step 1: Download Patents

The `download_patents.py` script fetches patent metadata and PDF images.

### Download Recent Patents (Default: 100)

```bash
# Download 100 recent patents with full metadata
python download_patents.py -o ./data/patents -l 100

# Download patents from a specific year range
python download_patents.py -o ./data/patents --year-min 2020 --year-max 2024 -l 500
```

### Download Metadata Only (Faster)

```bash
# Skip PDF downloads, only get metadata
python download_patents.py -o ./data/patents --no-pdf -l 1000

# Minimal metadata (skip abstracts, inventors, etc.)
python download_patents.py --no-pdf --skip-abstracts --skip-inventors -l 5000
```

### Preview Available Patents

```bash
# List patents without downloading
python download_patents.py --list-only -l 50

# List patents from 2023
python download_patents.py --list-only --year-min 2023 --year-max 2023 -l 100
```

### Script Options

| Option | Description | Default |
|--------|-------------|---------|
| `-o, --output` | Output directory for patents | `./data/patents` |
| `-c, --cache` | Cache directory for TSV files | `./data/cache` |
| `-l, --limit` | Max patents to download | `100` |
| `--year-min` | Minimum patent year | None |
| `--year-max` | Maximum patent year | None |
| `-w, --workers` | Parallel download threads | `4` |
| `-d, --delay` | Delay between requests (sec) | `0.5` |
| `--no-pdf` | Skip PDF downloads | `false` |
| `--skip-abstracts` | Skip abstract data | `false` |
| `--skip-inventors` | Skip inventor data | `false` |
| `--skip-assignees` | Skip assignee data | `false` |
| `--skip-cpc` | Skip CPC classifications | `false` |
| `--list-only` | List patents without downloading | `false` |

## Output Format

Each patent is saved with:
- `{patent_id}.pdf` - Full patent document (drawings, text, claims)
- `{patent_id}.json` - Structured metadata

### Metadata JSON Structure

```json
{
  "patent_id": "11234567",
  "type": "utility",
  "date": "2023-01-15",
  "title": "Method for Improved Machine Learning",
  "num_claims": "20",
  "abstract": "A system and method for...",
  "inventors": [
    {
      "name_first": "John",
      "name_last": "Smith",
      "city": "San Francisco",
      "state": "CA",
      "country": "US"
    }
  ],
  "assignees": [
    {
      "organization": "Tech Corp",
      "type": "2",
      "city": "Palo Alto",
      "state": "CA",
      "country": "US"
    }
  ],
  "cpc_classifications": [
    {
      "section": "G",
      "class": "06",
      "subclass": "N",
      "group": "3",
      "subgroup": "08"
    }
  ],
  "source": {
    "database": "USPTO via PatentsView",
    "url": "https://patents.google.com/patent/US11234567",
    "pdf_url": "https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11234567",
    "license": "Public Domain"
  }
}
```

## Step 2: Create Mixpeek Resources

*Coming soon: Scripts for creating Mixpeek collection, feature extractors, index, and retriever.*

## Step 3: Search Examples

*Coming soon: Example queries and search interface.*

### Search Use Cases

- **Prior Art Search**: Find similar patents to evaluate novelty
- **Technology Landscaping**: Explore patents in a technology domain
- **Competitor Analysis**: Find patents by specific assignees
- **Visual Search**: Find patents with similar drawings/diagrams
- **Semantic Search**: Natural language queries like "renewable energy storage systems"

## Project Structure

```
uspto-patents/
├── README.md
├── download_patents.py     # Patent downloader script
├── ingest_mixpeek.py       # Mixpeek ingestion (coming soon)
├── search.py               # Search interface (coming soon)
└── data/
    ├── cache/              # Cached PatentsView TSV files
    └── patents/            # Downloaded patent PDFs + JSON metadata
```

## CPC Classification Codes

Patents are classified using Cooperative Patent Classification (CPC):

| Section | Description |
|---------|-------------|
| A | Human Necessities |
| B | Operations and Transport |
| C | Chemistry and Metallurgy |
| D | Textiles |
| E | Fixed Constructions |
| F | Mechanical Engineering |
| G | Physics |
| H | Electricity |

## Notes

- **Rate Limiting**: The script includes delays to respect USPTO servers
- **Large Files**: PatentsView TSV files can be several GB; they're cached locally
- **PDF Availability**: Some older patents may not have PDFs available
- **Data Currency**: PatentsView data is updated quarterly

## Resources

- [PatentsView](https://patentsview.org/) - Patent data visualization and analysis
- [USPTO Open Data Portal](https://developer.uspto.gov/data) - Official USPTO data
- [Google Patents](https://patents.google.com/) - Patent search interface
- [CPC Classification](https://www.cooperativepatentclassification.org/) - Classification scheme
