# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ZotWatch is a personalized academic paper recommendation system that builds a research interest profile from a user's Zotero library and continuously monitors academic sources for relevant new publications. It runs daily via GitHub Actions and outputs RSS/HTML feeds.

## Commands

```bash
# Install dependencies
uv sync

# Build/rebuild user profile from Zotero library (full rebuild)
uv run python -m src.cli profile --full

# Daily watch: fetch candidates, score, and generate RSS feed
uv run python -m src.cli watch --rss --top 20

# Generate both RSS and HTML report
uv run python -m src.cli watch --rss --report --top 20

# Push top recommendations back to Zotero
uv run python -m src.cli watch --rss --push --top 20

# Enable verbose logging
uv run python -m src.cli watch --verbose --rss
```

## Architecture

### Pipeline Flow

1. **Ingest** (`ingest_zotero_api.py`): Fetches items from Zotero Web API, stores in SQLite
2. **Profile Build** (`build_profile.py`): Vectorizes library items using Voyage AI API (voyage-3.5), builds FAISS index, extracts top authors/venues
3. **Candidate Fetch** (`fetch_new.py`): Pulls recent papers from Crossref, arXiv, bioRxiv/medRxiv, OpenAlex
4. **Deduplication** (`dedupe.py`): Filters out papers already in the user's library
5. **Scoring** (`score_rank.py`): Ranks candidates using weighted combination of similarity, recency, citations, journal quality, and whitelist bonuses
6. **Output** (`rss_writer.py`, `report_html.py`): Generates RSS feed and/or HTML report

### Key Data Artifacts

- `data/profile.sqlite`: SQLite database storing Zotero items and embeddings
- `data/faiss.index`: FAISS vector index for similarity search
- `data/profile.json`: Profile summary with top authors, venues, and centroid vector
- `data/cache/candidate_cache.json`: 12-hour cache of fetched candidates
- `data/journal_metrics.csv`: Optional SJR journal metrics for quality scoring

### Configuration Files (config/)

- `config.yaml`: Unified configuration file containing all settings:
  - `zotero`: Zotero API settings (user_id uses `${ZOTERO_USER_ID}` env var expansion)
  - `sources`: Data source toggles and parameters (window_days, categories)
  - `scoring`: Score weights, thresholds, decay settings, author/venue whitelists
  - `embedding`: Embedding model configuration (model name, API key env var, input type)

### Core Components

- `TextVectorizer` (`vectorizer.py`): Wraps Voyage AI API (voyage-3.5, 1024-dim embeddings)
- `FaissIndex` (`faiss_store.py`): Manages FAISS index for semantic similarity
- `ProfileStorage` (`storage.py`): SQLite abstraction for items and embeddings
- `Settings` (`settings.py`): Pydantic models for configuration with env var expansion

## Environment Variables

Required:
- `ZOTERO_API_KEY`: Zotero Web API key
- `ZOTERO_USER_ID`: Zotero user ID
- `VOYAGE_API_KEY`: Voyage AI API key for text embeddings

Optional:
- `OPENALEX_MAILTO`, `CROSSREF_MAILTO`: Polite API identifiers
- `ALTMETRIC_KEY`: Altmetric API access (if enabled)

## Key Constraints

- Preprint ratio is capped at 30% in final results (`_limit_preprints`)
- Results are filtered to papers within last 7 days (`_filter_recent`)
- Candidate cache expires after 12 hours
- GitHub Actions caches profile artifacts monthly to avoid full rebuilds
