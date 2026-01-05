# Hacker News Posts & Comments

Download posts and comments from Hacker News, including full historical archives.
Categorizes posts by type (text vs URL).

## Overview

This module demonstrates how to:
1. Download current stories (top, new, best, ask, show, job) via Firebase API
2. Download historical stories (last N years) via Algolia HN Search API
3. Categorize posts as 'text' (Ask HN, Show HN, self-posts) vs 'url' (link posts)
4. Recursively fetch all comments with nested reply structure
5. Store everything in structured JSON format organized by year/month

## Data Sources

| API | Purpose | URL |
|-----|---------|-----|
| Algolia HN Search | Historical stories | https://hn.algolia.com/api/v1/ |
| Firebase HN API | Current stories & comments | https://hacker-news.firebaseio.com/v0/ |

## Prerequisites

```bash
pip install requests python-dateutil
```

## Commands

### Download Historical Data (Last 10 Years)

```bash
# Download all stories from the last 10 years (without comments - faster)
python download_hn.py historical --years 10 --no-comments

# Download with all comments (slower but complete)
python download_hn.py historical --years 10

# Download specific time range
python download_hn.py historical --start-year 2020 --start-month 1

# Resume interrupted download (skips existing files)
python download_hn.py historical --years 10 --no-comments
```

### Download Current Stories

```bash
# Download top 30 stories with comments
python download_hn.py current --type top -n 30

# Download Ask HN posts
python download_hn.py current --type ask -n 50

# List stories without downloading
python download_hn.py current --type top -n 20 --list-only
```

## Command Options

### `historical` - Download Historical Stories

| Option | Description | Default |
|--------|-------------|---------|
| `--years` | Number of years to download | `10` |
| `--start-year` | Start from specific year | - |
| `--start-month` | Start from specific month (1-12) | - |
| `--no-comments` | Skip downloading comments (much faster) | `false` |
| `-o, --output` | Output directory | `./data/historical` |
| `-d, --delay` | Delay between requests (sec) | `0.1` |
| `-w, --workers` | Parallel workers for comments | `20` |
| `--no-skip` | Re-download existing files | `false` |

### `current` - Download Current Stories

| Option | Description | Default |
|--------|-------------|---------|
| `--type` | Story type (top/new/best/ask/show/job) | `top` |
| `-n, --limit` | Number of stories | `30` |
| `--no-comments` | Skip downloading comments | `false` |
| `-o, --output` | Output directory | `./data/current` |
| `-d, --delay` | Delay between requests (sec) | `0.2` |
| `-w, --workers` | Parallel workers for comments | `20` |
| `--no-skip` | Re-download existing files | `false` |
| `--list-only` | List stories without downloading | `false` |

## Output Structure

### Historical Data (organized by year/month)

```
hacker-news/
├── download_hn.py
├── README.md
└── data/
    └── historical/
        ├── 2016/
        │   ├── 01/
        │   │   ├── month_summary.json
        │   │   ├── story_12345678.json
        │   │   └── ...
        │   ├── 02/
        │   └── ...
        ├── 2017/
        └── ...
```

### Current Data

```
data/
└── current/
    ├── top_stories.json
    ├── story_12345678.json
    └── ...
```

## Data Format

### Story JSON

```json
{
  "id": 12345678,
  "title": "Show HN: My new project",
  "by": "username",
  "score": 150,
  "time": 1704384000,
  "time_str": "2024-01-04T12:00:00",
  "category": "url",
  "descendants": 45,
  "url": "https://example.com/project",
  "text": null,
  "comment_count": 45,
  "comments": [
    {
      "id": 12345679,
      "by": "commenter",
      "text": "Great project!",
      "time": 1704384100,
      "time_str": "2024-01-04T12:01:40",
      "parent": 12345678,
      "kids": [12345680],
      "replies": [
        {
          "id": 12345680,
          "by": "another_user",
          "text": "I agree!",
          "replies": []
        }
      ]
    }
  ]
}
```

### Month Summary JSON

```json
{
  "year": 2024,
  "month": 1,
  "downloaded_at": "2024-01-15T10:00:00",
  "total_stories": 25000,
  "text_posts": 5000,
  "url_posts": 20000,
  "stories": [
    {"id": 12345678, "title": "...", "category": "url", "score": 150}
  ]
}
```

### Post Categories

| Category | Description | Examples |
|----------|-------------|----------|
| `text` | Self-posts with text content | Ask HN, text-only Show HN |
| `url` | Posts linking to external URLs | Articles, Show HN with links |

## Estimated Data Volume

| Time Range | Approx. Stories | Approx. Comments |
|------------|-----------------|------------------|
| 1 month | 25,000-30,000 | 500,000+ |
| 1 year | 300,000-400,000 | 6,000,000+ |
| 10 years | 3,000,000+ | 60,000,000+ |

**Note:** Downloading with comments is significantly slower. For initial bulk downloads, consider using `--no-comments` first, then selectively fetching comments for high-value stories.

## Performance Tips

1. **Start without comments**: Use `--no-comments` for initial bulk download
2. **Resume support**: Script automatically skips existing files
3. **Parallel workers**: Increase `-w` for faster comment fetching (default: 20)
4. **Rate limiting**: Algolia allows 10,000 requests/hour; adjust `-d` if needed

## Use Cases

- **Sentiment Analysis**: Analyze community reactions to tech announcements
- **Topic Modeling**: Cluster discussions by topic over time
- **Trend Analysis**: Track popular topics, technologies, companies over years
- **NLP Training**: Large-scale dataset for language models
- **Historical Research**: Study how tech discourse has evolved

## Attribution

Content sourced from [Hacker News](https://news.ycombinator.com) via Algolia HN Search API.

## Resources

- [Algolia HN Search API](https://hn.algolia.com/api)
- [HN Firebase API](https://github.com/HackerNews/API)
- [Hacker News](https://news.ycombinator.com)
