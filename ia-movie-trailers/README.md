# Internet Archive Movie Trailers + Posters

Build a multimodal search engine for movie trailers and posters using Internet Archive's public collections.

## Overview

This example demonstrates how to:
1. Download movie trailers (video) and posters (images) from Internet Archive
2. Create a Mixpeek collection for storing multimodal assets
3. Configure feature extractors for video and image embeddings
4. Build an index for similarity search across both media types
5. Create a retriever for natural language queries

## Data Sources

Content is sourced from multiple Internet Archive collections:

### Movie Trailers
- **Collection**: [movie_trailers](https://archive.org/details/movie_trailers)
- **Content**: Classic and vintage movie trailers
- **License**: Public Domain / Various open licenses

### Movie Posters
- **Collection**: [movie-posters_202403](https://archive.org/details/movie-posters_202403)
- **Content**: 942+ movie posters, DVD covers, VHS covers
- **License**: Public Domain / Various open licenses

- **Collection**: [HRC_Posters](https://archive.org/details/HRC_Posters)
- **Content**: Harry Ransom Center classical movie posters
- **License**: Public Domain

- **Collection**: [Horror-Movie-Posters](https://archive.org/details/Horror-Movie-Posters)
- **Content**: 82+ horror movie posters
- **License**: Public Domain

## Prerequisites

```bash
pip install requests internetarchive mixpeek
```

## Step 1: Download Content

The `download_ia.py` script fetches trailers and posters from Internet Archive.

### Download Everything

```bash
# Download both trailers and posters (default)
python download_ia.py --all -o ./data

# Or specify individually
python download_ia.py --trailers --posters -o ./data
```

### Download Trailers Only

```bash
# Download movie trailers
python download_ia.py --trailers -o ./data

# Limit to first 100 trailers
python download_ia.py --trailers -l 100
```

### Download Posters Only

```bash
# Download movie posters
python download_ia.py --posters -o ./data

# From specific collection
python download_ia.py --posters --collection HRC_Posters
```

### Preview Available Items

```bash
# List items without downloading
python download_ia.py --all --list-only -l 50
```

### Script Options

| Option | Description | Default |
|--------|-------------|---------|
| `--trailers` | Download movie trailers | `false` |
| `--posters` | Download movie posters | `false` |
| `--all` | Download both trailers and posters | `true` (if neither specified) |
| `-o, --output` | Base output directory | `./data` |
| `-w, --workers` | Parallel download threads | `4` |
| `-d, --delay` | Delay between API requests (sec) | `0.5` |
| `-l, --limit` | Max items to download per category | `None` |
| `--collection` | Specific collection(s) to use | All defaults |
| `--no-skip` | Re-download existing files | `false` |
| `--list-only` | List items without downloading | `false` |

## Output Structure

```
ia-movie-trailers/
├── README.md
├── download_ia.py          # Content downloader script
├── ingest_mixpeek.py       # Mixpeek ingestion script
└── data/
    ├── videos/             # Downloaded trailers
    │   ├── {id}_{title}.mp4
    │   ├── {id}_{title}.json
    │   └── {id}_{title}_thumb.jpg
    ├── posters/            # Downloaded posters
    │   ├── {id}_{title}.jpg
    │   └── {id}_{title}.json
    └── cache/              # API response cache
```

## Metadata Format

### Trailer Metadata (JSON)

```json
{
  "identifier": "MovieTra1950",
  "title": "Movie Trailers 1950",
  "description": "Collection of classic 1950s movie trailers",
  "creator": "Various Studios",
  "date": "1950",
  "year": "1950",
  "subject": ["movie trailers", "1950s", "cinema"],
  "collection": ["movie_trailers"],
  "mediatype": "movies",
  "runtime": "10:30",
  "video_file": "MovieTra1950_Movie_Trailers_1950.mp4",
  "thumbnail_file": "MovieTra1950_Movie_Trailers_1950_thumb.jpg",
  "source": {
    "archive": "Internet Archive",
    "url": "https://archive.org/details/MovieTra1950",
    "download_url": "https://archive.org/download/MovieTra1950"
  }
}
```

### Poster Metadata (JSON)

```json
{
  "identifier": "movie-posters_202403",
  "title": "Casablanca Movie Poster",
  "description": "Original theatrical release poster",
  "creator": "Warner Bros",
  "date": "1942",
  "poster_file": "movie-posters_202403_Casablanca_0.jpg",
  "original_filename": "casablanca_poster.jpg",
  "file_size": "2456789",
  "source": {
    "archive": "Internet Archive",
    "url": "https://archive.org/details/movie-posters_202403",
    "download_url": "https://archive.org/download/movie-posters_202403/casablanca_poster.jpg"
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
1. Create a `movie-trailers` bucket for video content
2. Create a `movie-posters` bucket for image content
3. Create collections with multimodal feature extractors
4. Upload 3 random trailers and 5 random posters
5. Submit batches for processing

### Resources Created

| Resource | Name | Description |
|----------|------|-------------|
| Video Bucket | `movie-trailers` | Stores trailer videos + metadata |
| Poster Bucket | `movie-posters` | Stores poster images + metadata |
| Video Collection | `movie-trailers-collection` | Searchable video embeddings |
| Poster Collection | `movie-posters-collection` | Searchable image embeddings |

## Step 3: Search Examples

Once ingestion is complete, you can search using the Mixpeek API:

```python
import requests

# Search for horror trailers
response = requests.post(
    "https://api.mixpeek.com/v1/collections/search",
    headers={"Authorization": "Bearer YOUR_API_KEY"},
    json={
        "collection_id": "your_collection_id",
        "query": "scary horror movie with monster",
        "top_k": 10
    }
)
```

## Use Cases

- **Multimodal Movie Search**: Find trailers and posters using natural language
- **Visual Similarity**: Find movies with similar poster styles or trailer aesthetics
- **Genre Classification**: Cluster movies by visual themes
- **Era Detection**: Identify movies by their visual style period

## Attribution

While most content is public domain, attribution is appreciated:

> Content sourced from [Internet Archive](https://archive.org), a non-profit digital library.

## Resources

- [Internet Archive](https://archive.org)
- [Internet Archive Python Library](https://archive.org/developers/internetarchive/)
- [Movie Trailers Collection](https://archive.org/details/movie_trailers)
- [Movie Posters Collection](https://archive.org/details/movie-posters_202403)
