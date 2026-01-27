"""Crossref source implementation."""

import csv
import logging
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path

import requests

from zotwatch.config.settings import Settings
from zotwatch.core.constants import CROSSREF_API_PAGE_SIZE, DEFAULT_HTTP_TIMEOUT
from zotwatch.core.exceptions import SourceFetchError
from zotwatch.core.models import CandidateWork
from zotwatch.utils.datetime import utc_yesterday_end

from .base import BaseSource, SourceRegistry, clean_html, clean_title, is_non_article_title, parse_date

logger = logging.getLogger(__name__)


@SourceRegistry.register
class CrossrefSource(BaseSource):
    """Crossref journal articles source."""

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.config = settings.sources.crossref
        self.session = requests.Session()
        self._issn_whitelist: list[str] | None = None  # Lazy cache

    @property
    def name(self) -> str:
        return "crossref"

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    @property
    def issn_whitelist(self) -> list[str]:
        """Get ISSN whitelist (cached after first load)."""
        if self._issn_whitelist is None:
            self._issn_whitelist = self._load_issn_whitelist()
        return self._issn_whitelist

    def fetch(self, days_back: int | None = None) -> list[CandidateWork]:
        """Fetch Crossref works from journals in the ISSN whitelist."""
        if days_back is None:
            days_back = self.config.days_back

        max_results = self.config.max_results

        # Use cached ISSN whitelist
        issns = self.issn_whitelist
        if not issns:
            logger.warning("No ISSNs in whitelist, skipping Crossref fetch")
            return []

        logger.info("Using ISSN whitelist with %d journals", len(issns))
        return self._fetch_by_issn(days_back, issns, max_results)

    def _load_issn_whitelist(self) -> list[str]:
        """Load ISSN whitelist from CSV file."""
        # Look for whitelist in data directory
        path = Path(__file__).parents[3] / "data" / "journal_whitelist.csv"
        if not path.exists():
            logger.warning("Journal whitelist not found: %s", path)
            return []

        issns: list[str] = []
        try:
            with path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    issn = (row.get("issn") or "").strip()
                    # Skip empty and comment rows (ISSN starts with #)
                    if issn and not issn.startswith("#"):
                        issns.append(issn)
        except OSError as exc:
            logger.warning("Failed to read journal whitelist file %s: %s", path, exc)
            return []
        except csv.Error as exc:
            logger.warning("Failed to parse journal whitelist CSV %s: %s", path, exc)
            return []

        logger.info("Loaded %d ISSNs from whitelist", len(issns))
        return issns

    def _fetch_by_issn(
        self,
        days_back: int,
        issns: list[str],
        max_results: int,
    ) -> list[CandidateWork]:
        """Fetch works from specific journals by ISSN."""
        # Query complete past days only (not including today)
        yesterday = utc_yesterday_end()
        yesterday_start = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
        since = yesterday_start - timedelta(days=days_back - 1)
        until = yesterday_start  # End date is yesterday

        # Build filter string with ISSNs (OR logic)
        issn_filter = ",".join(f"issn:{issn}" for issn in issns)
        filter_str = (
            f"from-created-date:{since.date().isoformat()},until-created-date:{until.date().isoformat()},{issn_filter}"
        )

        params = {
            "filter": filter_str,
            "sort": "created",
            "order": "desc",
            "rows": min(CROSSREF_API_PAGE_SIZE, max_results),
            "mailto": self.config.mailto,
            "select": "DOI,title,author,abstract,container-title,created,URL,type,is-referenced-by-count,publisher,ISSN",
        }

        logger.info(
            "Fetching Crossref works from %s to %s from %d journals (max %d)",
            since.date(),
            until.date(),
            len(issns),
            max_results,
        )

        results, journal_counts = self._fetch_paginated(
            params,
            max_results,
            stat_key_fn=lambda item: (item.get("container-title") or ["Unknown"])[0],
        )

        logger.info("Fetched %d Crossref works from whitelisted journals", len(results))
        if journal_counts:
            for journal, count in sorted(journal_counts.items(), key=lambda x: -x[1])[:20]:
                logger.info("  - %s: %d articles", journal, count)

        return results

    def _fetch_paginated(
        self,
        params: dict,
        max_results: int,
        stat_key_fn: Callable[[dict], str] | None = None,
    ) -> tuple[list[CandidateWork], dict[str, int]]:
        """Fetch works with pagination.

        Args:
            params: Request parameters (will be modified with offset).
            max_results: Maximum number of results to fetch.
            stat_key_fn: Optional function to extract statistics key from item.

        Returns:
            Tuple of (results list, statistics dict).
        """
        url = "https://api.crossref.org/works"
        results: list[CandidateWork] = []
        stats: dict[str, int] = {}
        offset = 0

        while len(results) < max_results:
            params["offset"] = offset
            try:
                resp = self.session.get(url, params=params, timeout=DEFAULT_HTTP_TIMEOUT)
                resp.raise_for_status()
            except requests.exceptions.Timeout:
                raise SourceFetchError(
                    "crossref", f"Request timed out after {DEFAULT_HTTP_TIMEOUT}s (offset={offset})"
                ) from None
            except requests.exceptions.HTTPError as e:
                status = e.response.status_code if e.response is not None else "unknown"
                raise SourceFetchError("crossref", f"HTTP {status} error (offset={offset})") from e
            except requests.exceptions.RequestException as e:
                raise SourceFetchError("crossref", f"Network error: {type(e).__name__}") from e

            message = resp.json().get("message", {})
            items = message.get("items", [])
            if not items:
                break

            for item in items:
                if len(results) >= max_results:
                    break
                work = self._parse_crossref_item(item)
                if work:
                    results.append(work)
                    if stat_key_fn:
                        key = stat_key_fn(item)
                        stats[key] = stats.get(key, 0) + 1

            total = message.get("total-results", 0)
            offset += len(items)
            if offset >= total or offset >= max_results:
                break

        return results, stats

    def _parse_crossref_item(
        self,
        item: dict,
        venue_override: str | None = None,
    ) -> CandidateWork | None:
        """Parse Crossref API item to CandidateWork."""
        title = clean_title((item.get("title") or [""])[0])
        if not title:
            return None

        # Get venue for filtering
        venue = venue_override or (item.get("container-title") or [None])[0]

        # Filter out non-article entries (TOC, publication info, etc.)
        if is_non_article_title(title, venue):
            return None

        doi = item.get("DOI")
        authors = [" ".join(filter(None, [p.get("given"), p.get("family")])).strip() for p in item.get("author", [])]

        return CandidateWork(
            source="crossref",
            identifier=doi or item.get("URL", "unknown"),
            title=title,
            abstract=clean_html(item.get("abstract")),
            authors=[a for a in authors if a],
            doi=doi,
            url=item.get("URL"),
            published=parse_date(item.get("created", {}).get("date-time")),
            venue=venue,
            metrics={"is-referenced-by": float(item.get("is-referenced-by-count", 0))},
            extra={
                "type": item.get("type"),
                "issns": item.get("ISSN") or [],  # All ISSNs for matching
            },
        )


__all__ = ["CrossrefSource"]
