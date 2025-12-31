# NOAA/NASA Earth Dataset

Build a multimodal search engine for Earth observation data using NASA and NOAA public domain imagery and natural event metadata.

## Overview

This example demonstrates how to:
1. Download satellite imagery from NASA's EPIC (Earth Polychromatic Imaging Camera)
2. Retrieve natural event metadata from NASA's EONET (hurricanes, volcanoes, wildfires)
3. Create a Mixpeek collection for storing Earth observation assets
4. Build an index for searching by event type, location, or visual content

## Data Sources

### NASA EPIC - Full Earth Disc Images
- **Source**: [NASA DSCOVR EPIC](https://epic.gsfc.nasa.gov/)
- **License**: Public Domain (U.S. Government Work)
- **Content**: Full disc Earth images from the DSCOVR satellite at L1 Lagrange point
- **Resolution**: Up to 2048x2048 pixels
- **Collections**: Natural color, Enhanced color, Aerosol, Cloud
- **Update Frequency**: Daily (12-22 images per day)

### NASA EONET - Natural Events Tracker
- **Source**: [NASA EONET API](https://eonet.gsfc.nasa.gov/)
- **License**: Public Domain (U.S. Government Work)
- **Content**: Curated natural events with coordinates and metadata
- **Categories**: Severe Storms, Wildfires, Volcanoes, Earthquakes, Floods, Sea Ice, and more
- **Data Format**: JSON with GeoJSON support

### NASA GIBS - Satellite Imagery Tiles
- **Source**: [NASA GIBS](https://nasa-gibs.github.io/gibs-api-docs/)
- **License**: Public Domain
- **Content**: 1000+ satellite imagery products via WMTS/WMS
- **Use Case**: Regional imagery for specific events

## Prerequisites

```bash
pip install requests
```

## Step 1: Download Data

The `download_earth_data.py` script fetches data from NASA APIs.

### Download Full Earth Package (Recommended)

```bash
# Download 30 days of EPIC images + all EONET events
python download_earth_data.py --source full -d 30 -l 100

# Download with more images
python download_earth_data.py --source full -d 60 -l 200
```

### Download EPIC Images Only

```bash
# Natural color images (last 7 days)
python download_earth_data.py --source epic --epic-collection natural -d 7

# Enhanced color images (higher contrast)
python download_earth_data.py --source epic --epic-collection enhanced -d 7

# Full resolution PNG (larger files)
python download_earth_data.py --source epic --epic-format png -l 20
```

### Download Hurricane/Storm Data

```bash
# Focus on severe storms (hurricanes, typhoons, cyclones)
python download_earth_data.py --source hurricanes -d 90 -l 50

# Just the storm event metadata
python download_earth_data.py --source eonet --eonet-category severeStorms -d 365
```

### Download All Natural Events

```bash
# All EONET categories
python download_earth_data.py --source eonet -d 365

# Specific category
python download_earth_data.py --source eonet --eonet-category wildfires -d 90
python download_earth_data.py --source eonet --eonet-category volcanoes -d 365
```

### Preview Available Data

```bash
# List available dates and event counts without downloading
python download_earth_data.py --list-only
```

### Script Options

| Option | Description | Default |
|--------|-------------|---------|
| `-o, --output` | Output directory | `./data` |
| `--source` | Data source: `epic`, `eonet`, `hurricanes`, `full` | `full` |
| `--epic-collection` | EPIC type: `natural`, `enhanced`, `aerosol`, `cloud` | `natural` |
| `--epic-format` | Image format: `png`, `jpg`, `thumbs` | `jpg` |
| `--eonet-category` | Event category (see list below) | All |
| `-d, --days` | Days of data to download | `30` |
| `-l, --limit` | Maximum items to download | `50` |
| `-w, --workers` | Parallel download threads | `4` |
| `--delay` | Delay between requests (sec) | `0.2` |
| `--list-only` | Preview without downloading | `false` |

### EONET Categories

| Category | Description |
|----------|-------------|
| `severeStorms` | Hurricanes, typhoons, cyclones, tropical storms |
| `wildfires` | Forest fires, brush fires |
| `volcanoes` | Volcanic eruptions and activity |
| `earthquakes` | Significant earthquakes |
| `floods` | Major flooding events |
| `landslides` | Landslides and mudslides |
| `seaLakeIce` | Sea ice and lake ice changes |
| `dustHaze` | Dust storms and haze events |
| `drought` | Drought conditions |
| `snow` | Significant snow events |
| `tempExtremes` | Temperature extremes |
| `waterColor` | Algal blooms, sediment plumes |

## Step 2: Create Mixpeek Resources

*Coming soon: Scripts for creating Mixpeek collection, feature extractors, index, and retriever.*

## Step 3: Search Examples

*Coming soon: Example queries and search interface.*

## Project Structure

```
noaa-nasa-earth/
├── README.md
├── download_earth_data.py    # Data downloader script
├── ingest_mixpeek.py         # Mixpeek ingestion (coming soon)
├── search.py                 # Search interface (coming soon)
└── data/
    ├── images/
    │   ├── natural/          # EPIC natural color images
    │   └── enhanced/         # EPIC enhanced color images
    ├── metadata/             # EONET event JSON files
    │   ├── eonet_severeStorms.json
    │   ├── eonet_severeStorms.geojson
    │   ├── eonet_wildfires.json
    │   └── ...
    └── cache/                # API response cache
```

## Sample Data

After running the download, you'll have:

**EPIC Images** (`data/images/natural/`):
- `epic_natural_epic_1b_20241201042117.jpg` - Full Earth image
- `epic_natural_epic_1b_20241201042117.json` - Image metadata (coordinates, timestamp, satellite position)

**EONET Events** (`data/metadata/`):
- `eonet_severeStorms.json` - Storm events with track coordinates
- `eonet_severeStorms.geojson` - GeoJSON for mapping
- `eonet_all_events.json` - Combined events file

## Example Metadata

### EPIC Image Metadata
```json
{
  "identifier": "20241201042117",
  "image": "epic_1b_20241201042117",
  "date": "2024-12-01 04:21:17",
  "caption": "This image was taken by NASA's EPIC camera...",
  "centroid_coordinates": {
    "lat": -5.234,
    "lon": 156.789
  },
  "collection": "natural",
  "source": {
    "mission": "DSCOVR",
    "instrument": "EPIC (Earth Polychromatic Imaging Camera)",
    "institution": "NASA",
    "license": "Public Domain (U.S. Government Work)"
  }
}
```

### EONET Storm Event
```json
{
  "id": "EONET_6789",
  "title": "Tropical Storm Example",
  "categories": [{"id": "severeStorms", "title": "Severe Storms"}],
  "geometry": [
    {
      "date": "2024-11-15T12:00:00Z",
      "type": "Point",
      "coordinates": [-65.5, 18.2]
    }
  ],
  "sources": [
    {"id": "JTWC", "url": "https://..."}
  ]
}
```

## Attribution

While not required (public domain), NASA and NOAA request consideration for attribution:

> Earth imagery courtesy of NASA DSCOVR EPIC Team
> Natural event data from NASA Earth Observatory Natural Event Tracker (EONET)

## Resources

- [NASA EPIC Portal](https://epic.gsfc.nasa.gov/)
- [NASA EPIC API Documentation](https://epic.gsfc.nasa.gov/about/api)
- [NASA EONET Portal](https://eonet.gsfc.nasa.gov/)
- [NASA EONET API v3 Documentation](https://eonet.gsfc.nasa.gov/docs/v3)
- [NASA GIBS API Documentation](https://nasa-gibs.github.io/gibs-api-docs/)
- [NASA Worldview](https://worldview.earthdata.nasa.gov/) - Interactive imagery browser
- [NOAA Hurricane Imagery Archive](https://www.nesdis.noaa.gov/imagery/hurricanes/hurricane-imagery-archive)
