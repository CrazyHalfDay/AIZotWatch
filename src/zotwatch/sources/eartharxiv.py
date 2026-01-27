"""EarthArXiv source implementation via OSF API.

EarthArXiv is a preprint server for Earth sciences hosted on OSF (Open Science Framework).
API documentation: https://developer.osf.io/
"""

import logging
from datetime import datetime, timedelta, timezone

import requests

from zotwatch.config.settings import Settings
from zotwatch.core.constants import DEFAULT_HTTP_TIMEOUT
from zotwatch.core.exceptions import SourceFetchError
from zotwatch.core.models import CandidateWork
from zotwatch.utils.datetime import utc_yesterday_end

from .base import BaseSource, SourceRegistry, clean_title, parse_date

logger = logging.getLogger(__name__)

# OSF API base URL
OSF_API_BASE = "https://api.osf.io/v2"


@SourceRegistry.register
class EartharxivSource(BaseSource):
    """EarthArXiv preprint source via OSF API."""

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.config = settings.sources.eartharxiv
        self.session = requests.Session()
        # OSF API recommends setting User-Agent
        self.session.headers.update({
            "User-Agent": "ZotWatch/1.0 (https://github.com/CrazyHalfDay/AIZotWatch)",
            "Accept": "application/json",
        })

    @property
    def name(self) -> str:
        return "eartharxiv"

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def fetch(self, days_back: int | None = None) -> list[CandidateWork]:
        """Fetch EarthArXiv preprints from OSF API.

        Args:
            days_back: Number of days to look back (default from config).

        Returns:
            List of candidate works from EarthArXiv.
        """
        if days_back is None:
            days_back = self.config.days_back

        max_results = self.config.max_results

        # Calculate date range (complete past days only)
        yesterday = utc_yesterday_end()
        yesterday_start = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
        from_date = yesterday_start - timedelta(days=days_back - 1)

        logger.info(
            "Fetching EarthArXiv preprints from %s to %s (max %d)",
            from_date.strftime("%Y-%m-%d"),
            yesterday_start.strftime("%Y-%m-%d"),
            max_results,
        )

        results: list[CandidateWork] = []
        page = 1
        page_size = 100  # OSF API max page size

        while len(results) < max_results:
            try:
                preprints = self._fetch_page(page, page_size, from_date)
            except SourceFetchError:
                raise
            except Exception as e:
                logger.error("Error fetching EarthArXiv page %d: %s", page, e)
                break

            if not preprints:
                break

            for preprint in preprints:
                if len(results) >= max_results:
                    break

                work = self._parse_preprint(preprint)
                if work:
                    # Filter by date (API filter may not be exact)
                    if work.published and work.published >= from_date:
                        results.append(work)

            # Check if there are more pages
            if len(preprints) < page_size:
                break

            page += 1

        logger.info("Fetched %d EarthArXiv preprints", len(results))
        return results

    def _fetch_page(
        self,
        page: int,
        page_size: int,
        from_date: datetime,
    ) -> list[dict]:
        """Fetch a single page of preprints from OSF API.

        Args:
            page: Page number (1-indexed).
            page_size: Number of results per page.
            from_date: Filter preprints published after this date.

        Returns:
            List of preprint data dictionaries.
        """
        url = f"{OSF_API_BASE}/preprints/"
        params = {
            "filter[provider]": "eartharxiv",
            "filter[date_published][gte]": from_date.strftime("%Y-%m-%d"),
            "filter[reviews_state]": "accepted",  # Only accepted preprints
            "sort": "-date_published",  # Most recent first
            "page": page,
            "page[size]": page_size,
            "embed": "contributors",  # Embed author information in same request
        }

        try:
            resp = self.session.get(url, params=params, timeout=DEFAULT_HTTP_TIMEOUT)
            resp.raise_for_status()
        except requests.exceptions.Timeout:
            raise SourceFetchError(
                "eartharxiv",
                f"Request timed out after {DEFAULT_HTTP_TIMEOUT}s (page {page})",
            ) from None
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else "unknown"
            raise SourceFetchError(
                "eartharxiv",
                f"HTTP {status} error (page {page})",
            ) from e
        except requests.exceptions.RequestException as e:
            raise SourceFetchError(
                "eartharxiv",
                f"Network error: {type(e).__name__}",
            ) from e

        data = resp.json()
        return data.get("data", [])

    def _parse_preprint(self, preprint: dict) -> CandidateWork | None:
        """Parse OSF preprint data to CandidateWork.

        Args:
            preprint: Preprint data from OSF API.

        Returns:
            CandidateWork or None if parsing fails.
        """
        attrs = preprint.get("attributes", {})
        links = preprint.get("links", {})
        embeds = preprint.get("embeds", {})

        title = clean_title(attrs.get("title"))
        if not title:
            return None

        # Skip withdrawn preprints
        if attrs.get("reviews_state") == "withdrawn":
            return None

        # Get DOI from links or attributes
        doi = None
        preprint_doi_url = links.get("preprint_doi")
        if preprint_doi_url and "doi.org/" in preprint_doi_url:
            doi = preprint_doi_url.split("doi.org/")[-1]

        # Parse publication date
        published = parse_date(attrs.get("date_published"))

        # Get abstract (description in OSF API)
        abstract = (attrs.get("description") or "").strip() or None

        # Get tags as keywords
        tags = attrs.get("tags", [])

        # Get subjects (hierarchical)
        subjects = []
        for subject_list in attrs.get("subjects", []):
            for subject in subject_list:
                if isinstance(subject, dict):
                    subjects.append(subject.get("text", ""))

        # Build URL
        url = links.get("html") or links.get("self")

        # Get preprint ID
        identifier = preprint.get("id") or preprint.get("links", {}).get("iri", "")

        # Extract authors from embedded contributors
        authors = self._extract_authors(embeds.get("contributors", {}))

        return CandidateWork(
            source="eartharxiv",
            identifier=identifier or title,
            title=title,
            abstract=abstract,
            authors=authors,
            doi=doi,
            url=url,
            published=published,
            venue="EarthArXiv",
            extra={
                "tags": tags,
                "subjects": subjects,
                "version": attrs.get("version", 1),
            },
        )

    def _extract_authors(self, contributors_data: dict) -> list[str]:
        """Extract author names from embedded contributors data.

        Args:
            contributors_data: Embedded contributors response from OSF API.

        Returns:
            List of author full names, sorted by contribution index.
        """
        authors = []
        contributor_list = contributors_data.get("data", [])

        # Sort by index to maintain author order
        sorted_contributors = sorted(
            contributor_list,
            key=lambda c: c.get("attributes", {}).get("index", 999),
        )

        for contributor in sorted_contributors:
            attrs = contributor.get("attributes", {})

            # Only include bibliographic (citable) contributors
            if not attrs.get("bibliographic", True):
                continue

            # Try to get name from embedded user data
            user_embed = contributor.get("embeds", {}).get("users", {}).get("data", {})
            user_attrs = user_embed.get("attributes", {})

            full_name = user_attrs.get("full_name")

            # Fallback to unregistered_contributor field
            if not full_name:
                full_name = attrs.get("unregistered_contributor")

            if full_name:
                authors.append(full_name.strip())

        return authors


__all__ = ["EartharxivSource"]
