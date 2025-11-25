import logging
import json
import html
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List

import feedparser
import requests

from .models import CandidateWork
from .settings import Settings
from .utils import ensure_isoformat, iso_to_datetime, utc_now

logger = logging.getLogger(__name__)


class CandidateFetcher:
    def __init__(self, settings: Settings, base_dir: Path):
        self.settings = settings
        self.session = requests.Session()
        self.base_dir = Path(base_dir)
        self.cache_path = self.base_dir / "data" / "cache" / "candidate_cache.json"
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.profile_path = self.base_dir / "data" / "profile.json"
        self.top_venues = self._load_top_venues()

    def fetch_all(self) -> List[CandidateWork]:
        cached = self._load_cache()
        if cached:
            fetched_at, candidates = cached
            age = datetime.now(timezone.utc) - fetched_at
            if age <= timedelta(hours=12):
                logger.info(
                    "Using cached candidate list from %s (age %.1f hours)",
                    fetched_at.isoformat(),
                    age.total_seconds() / 3600,
                )
                return candidates
            logger.info(
                "Candidate cache is stale (age %.1f hours); refreshing",
                age.total_seconds() / 3600,
            )
        results: List[CandidateWork] = []

        if self.settings.sources.openalex.enabled:
            results.extend(self._fetch_openalex())
        if self.settings.sources.crossref.enabled:
            results.extend(self._fetch_crossref())
            results.extend(self._fetch_crossref_top_venues())
        if self.settings.sources.arxiv.enabled:
            results.extend(self._fetch_arxiv())
        if self.settings.sources.biorxiv.enabled:
            results.extend(self._fetch_biorxiv())
        if self.settings.sources.medrxiv.enabled:
            results.extend(self._fetch_biorxiv(medrxiv=True))

        logger.info("Fetched %d candidate works", len(results))
        self._save_cache(results)
        return results

    def _load_top_venues(self) -> List[str]:
        if not self.profile_path.exists():
            return []
        try:
            data = json.loads(self.profile_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed to load profile when reading top venues: %s", exc)
            return []
        venues: List[str] = []
        for entry in data.get("top_venues", []):
            name = entry.get("venue") if isinstance(entry, dict) else None
            if name:
                venues.append(name)
        if venues:
            unique = list(dict.fromkeys(venues))
        else:
            unique = []
        if unique:
            logger.info("Loaded %d top venues from profile", len(unique))
        return unique[:20]

    def _load_cache(self):
        if not getattr(self, "cache_path", None) or not self.cache_path.exists():
            return None
        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed to read candidate cache: %s", exc)
            return None
        fetched_at = iso_to_datetime(payload.get("fetched_at"))
        if not fetched_at:
            return None
        items = payload.get("candidates", [])
        candidates: List[CandidateWork] = []
        for item in items:
            published = item.get("published")
            if published:
                item["published"] = _ensure_aware(iso_to_datetime(published))
            candidates.append(CandidateWork(**item))
        return fetched_at, candidates

    def _save_cache(self, candidates: List[CandidateWork]) -> None:
        if not getattr(self, "cache_path", None):
            return
        payload = {
            "fetched_at": ensure_isoformat(utc_now()),
            "candidates": [self._serialize_candidate(c) for c in candidates],
        }
        try:
            self.cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.warning("Failed to write candidate cache: %s", exc)

    @staticmethod
    def _serialize_candidate(candidate: CandidateWork) -> dict:
        data = candidate.model_dump()
        data["published"] = ensure_isoformat(candidate.published)
        return data

    def _fetch_openalex(self) -> List[CandidateWork]:
        config = self.settings.sources.openalex
        since = datetime.now(timezone.utc) - timedelta(days=config.days_back)
        url = "https://api.openalex.org/works"
        params = {
            "filter": f"from_publication_date:{since.date().isoformat()}",
            "sort": "publication_date:desc",
            "per-page": 200,
            "mailto": config.mailto,
        }
        logger.info("Fetching OpenAlex works since %s", since.date())
        resp = self.session.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        results = []
        for item in data.get("results", []):
            title = _clean_title(item.get("display_name"))
            if not title:
                continue
            work_id = item.get("id") or item.get("ids", {}).get("openalex")
            primary_location = item.get("primary_location") or {}
            source_info = primary_location.get("source") or {}
            landing_page = primary_location.get("landing_page_url")
            results.append(
                CandidateWork(
                    source="openalex",
                    identifier=work_id or item.get("doi") or title,
                    title=title,
                    abstract=_extract_openalex_abstract(item),
                    authors=[auth.get("author", {}).get("display_name", "") for auth in item.get("authorships", [])],
                    doi=item.get("doi"),
                    url=source_info.get("url") or landing_page,
                    published=_parse_date(item.get("publication_date")),
                    venue=source_info.get("display_name"),
                    metrics={"cited_by": float(item.get("cited_by_count", 0))},
                    extra={"concepts": [c.get("display_name") for c in item.get("concepts", [])]},
                )
            )
        return results

    def _fetch_crossref(self) -> List[CandidateWork]:
        config = self.settings.sources.crossref
        since = datetime.now(timezone.utc) - timedelta(days=config.days_back)
        url = "https://api.crossref.org/works"
        params = {
            "filter": f"from-pub-date:{since.date().isoformat()}",
            "sort": "created",
            "order": "desc",
            "rows": 200,
            "mailto": config.mailto,
        }
        logger.info("Fetching Crossref works since %s", since.date())
        resp = self.session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        message = resp.json().get("message", {})
        results = []
        for item in message.get("items", []):
            title = _clean_title((item.get("title") or [""])[0])
            if not title:
                continue
            doi = item.get("DOI")
            authors = [
                " ".join(filter(None, [p.get("given"), p.get("family")])).strip() for p in item.get("author", [])
            ]
            results.append(
                CandidateWork(
                    source="crossref",
                    identifier=doi or item.get("URL", "unknown"),
                    title=title,
                    abstract=_clean_crossref_abstract(item.get("abstract")),
                    authors=[a for a in authors if a],
                    doi=doi,
                    url=item.get("URL"),
                    published=_parse_date(item.get("created", {}).get("date-time")),
                    venue=(item.get("container-title") or [None])[0],
                    metrics={"is-referenced-by": float(item.get("is-referenced-by-count", 0))},
                    extra={"type": item.get("type")},
                )
            )
        return results

    def _fetch_crossref_top_venues(self) -> List[CandidateWork]:
        if not self.top_venues:
            return []
        config = self.settings.sources.crossref
        since = datetime.now(timezone.utc) - timedelta(days=config.days_back)
        results: List[CandidateWork] = []
        for venue in self.top_venues:
            params = {
                "filter": f"from-pub-date:{since.date().isoformat()},container-title:{venue}",
                "sort": "created",
                "order": "desc",
                "rows": 100,
                "mailto": config.mailto,
            }
            try:
                resp = self.session.get("https://api.crossref.org/works", params=params, timeout=30)
                resp.raise_for_status()
            except Exception as exc:
                logger.warning("Failed to fetch Crossref top venue %s: %s", venue, exc)
                continue
            message = resp.json().get("message", {})
            for item in message.get("items", []):
                title = _clean_title((item.get("title") or [""])[0])
                if not title:
                    continue
                doi = item.get("DOI")
                authors = [
                    " ".join(filter(None, [p.get("given"), p.get("family")])).strip() for p in item.get("author", [])
                ]
                results.append(
                    CandidateWork(
                        source="crossref",
                        identifier=doi or item.get("URL", "unknown"),
                        title=title,
                        abstract=_clean_crossref_abstract(item.get("abstract")),
                        authors=[a for a in authors if a],
                        doi=doi,
                        url=item.get("URL"),
                        published=_parse_date(item.get("created", {}).get("date-time")),
                        venue=venue,
                        metrics={"is-referenced-by": float(item.get("is-referenced-by-count", 0))},
                        extra={
                            "source": "top_venue",
                            "type": item.get("type"),
                        },
                    )
                )
        if results:
            logger.info("Fetched %d additional works from top venues", len(results))
        return results

    def _fetch_arxiv(self) -> List[CandidateWork]:
        config = self.settings.sources.arxiv
        categories = config.categories

        # Use submittedDate filter for date range
        to_date = datetime.now(timezone.utc)
        from_date = to_date - timedelta(days=config.days_back)
        date_filter = f"submittedDate:[{from_date:%Y%m%d}0000+TO+{to_date:%Y%m%d}2359]"

        cat_query = " OR ".join(f"cat:{cat}" for cat in categories)
        query = f"({cat_query})+AND+{date_filter}"

        url = "https://export.arxiv.org/api/query"
        params = {
            "search_query": query,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": config.max_results,
        }
        logger.info(
            "Fetching arXiv entries for categories: %s (last %d days, max %d)",
            ", ".join(categories),
            config.days_back,
            config.max_results,
        )
        resp = self.session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        feed = feedparser.parse(resp.text)
        results = []
        for entry in feed.entries:
            title = _clean_title(entry.get("title"))
            if not title:
                continue
            identifier = entry.get("id")
            published = _parse_date(entry.get("published"))
            results.append(
                CandidateWork(
                    source="arxiv",
                    identifier=identifier or title,
                    title=title,
                    abstract=(entry.get("summary") or "").strip() or None,
                    authors=[a.get("name") for a in entry.get("authors", [])],
                    doi=entry.get("arxiv_doi"),
                    url=entry.get("link"),
                    published=published,
                    venue="arXiv",
                    extra={"primary_category": entry.get("arxiv_primary_category", {}).get("term")},
                )
            )
        return results

    def _fetch_biorxiv(self, medrxiv: bool = False) -> List[CandidateWork]:
        if medrxiv:
            config = self.settings.sources.medrxiv
            base = "medrxiv"
        else:
            config = self.settings.sources.biorxiv
            base = "biorxiv"

        to_date = datetime.now(timezone.utc)
        from_date = to_date - timedelta(days=config.days_back)
        url = f"https://api.biorxiv.org/details/{base}/{from_date:%Y-%m-%d}/{to_date:%Y-%m-%d}"
        logger.info(
            "Fetching %s preprints from %s to %s",
            base,
            from_date.date(),
            to_date.date(),
        )
        resp = self.session.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        results = []
        for entry in data.get("collection", []):
            title = _clean_title(entry.get("title"))
            if not title:
                continue
            doi = entry.get("doi")
            rel_link = entry.get("rel_link") or entry.get("url")
            if not rel_link and doi:
                rel_link = f"https://doi.org/{doi}"
            results.append(
                CandidateWork(
                    source=base,
                    identifier=doi or entry.get("biorxiv_id") or title,
                    title=title,
                    abstract=entry.get("abstract"),
                    authors=[a.strip() for a in entry.get("authors", "").split(";") if a.strip()],
                    doi=doi,
                    url=rel_link,
                    published=_parse_date(entry.get("date")),
                    venue=base,
                    extra={
                        "category": entry.get("category"),
                        "version": entry.get("version"),
                    },
                )
            )
        return results


def _clean_title(value: str | None) -> str:
    if not value:
        return ""
    return value.strip()


def _extract_openalex_abstract(item: dict) -> str | None:
    abstract = item.get("abstract")
    if isinstance(abstract, dict):
        text = abstract.get("text")
        if text:
            return text
    if isinstance(abstract, str) and abstract.strip():
        return abstract.strip()
    inverted = item.get("abstract_inverted_index")
    if isinstance(inverted, dict) and inverted:
        try:
            size = max(pos for positions in inverted.values() for pos in positions) + 1
        except ValueError:
            size = 0
        tokens = ["" for _ in range(size)]
        for word, positions in inverted.items():
            for pos in positions:
                if 0 <= pos < size:
                    tokens[pos] = word
        summary = " ".join(filter(None, tokens)).strip()
        return summary or None
    return None


def _clean_crossref_abstract(value: str | None) -> str | None:
    if not value:
        return None
    text = html.unescape(value)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def _ensure_aware(dt):
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _parse_date(value):
    if not value:
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc)
    if isinstance(value, str):
        try:
            return _ensure_aware(datetime.fromisoformat(value.replace("Z", "+00:00")))
        except ValueError:
            try:
                return _ensure_aware(datetime.strptime(value, "%Y-%m-%d"))
            except ValueError:
                return None
    return None


__all__ = ["CandidateFetcher"]
