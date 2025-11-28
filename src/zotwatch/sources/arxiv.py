"""arXiv source implementation."""

import logging
from datetime import timedelta

import feedparser
import requests

from zotwatch.config.settings import Settings
from zotwatch.core.models import CandidateWork
from zotwatch.utils.datetime import utc_yesterday_end

from .base import BaseSource, SourceRegistry, clean_title, parse_date

logger = logging.getLogger(__name__)


@SourceRegistry.register
class ArxivSource(BaseSource):
    """arXiv preprint source."""

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.config = settings.sources.arxiv
        self.session = requests.Session()

    @property
    def name(self) -> str:
        return "arxiv"

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def fetch(self, days_back: int | None = None) -> list[CandidateWork]:
        """Fetch arXiv entries, filtering by primary category."""
        if days_back is None:
            days_back = self.config.days_back

        categories = self.config.categories
        categories_set = set(categories)  # For fast lookup
        # Query complete past days only (not including today)
        # This ensures consistent results regardless of when the program runs
        yesterday = utc_yesterday_end()
        yesterday_start = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
        from_date = yesterday_start - timedelta(days=days_back - 1)
        to_date = yesterday_start  # End date is yesterday
        max_results = self.config.max_results

        # Use submittedDate filter for date range
        # Note: arXiv API requires spaces around "TO" and "AND" keywords
        date_filter = f"submittedDate:[{from_date:%Y%m%d}0000 TO {to_date:%Y%m%d}2359]"
        cat_query = " OR ".join(f"cat:{cat}" for cat in categories)
        query = f"({cat_query}) AND {date_filter}"

        url = "https://export.arxiv.org/api/query"
        # Request more results to account for cross-listed papers filtering
        fetch_limit = min(max_results * 3, 2000)  # arXiv API has limits
        params = {
            "search_query": query,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": fetch_limit,
        }

        logger.info(
            "Fetching arXiv entries for categories: %s (%s to %s, max %d)",
            ", ".join(categories),
            from_date.strftime("%Y-%m-%d"),
            to_date.strftime("%Y-%m-%d"),
            max_results,
        )

        resp = self.session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        feed = feedparser.parse(resp.text)

        results: list[CandidateWork] = []
        category_counts: dict[str, int] = {}  # Count by category
        skipped_count = 0

        for entry in feed.entries:
            if len(results) >= max_results:
                break

            title = clean_title(entry.get("title"))
            if not title:
                continue

            primary_category = entry.get("arxiv_primary_category", {}).get("term")

            # Only include papers whose primary category is in our configured list
            if primary_category not in categories_set:
                skipped_count += 1
                continue

            identifier = entry.get("id")
            published = parse_date(entry.get("published"))

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
                    extra={"primary_category": primary_category},
                )
            )

            # Count by category
            category_counts[primary_category] = category_counts.get(primary_category, 0) + 1

        # Log total count and per-category statistics
        logger.info("Fetched %d arXiv entries (skipped %d cross-listed)", len(results), skipped_count)
        if category_counts:
            for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
                logger.info("  - %s: %d entries", cat, count)

        return results


__all__ = ["ArxivSource"]
