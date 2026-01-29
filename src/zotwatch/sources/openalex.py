"""OpenAlex API client for fetching papers by followed authors.

Uses the OpenAlex Works API with author filtering and cursor-based pagination.
Supports both OpenAlex Author IDs (e.g., A5023888391) and ORCIDs.

API docs: https://docs.openalex.org/api-entities/works
"""

import logging
import time
from datetime import datetime

import requests

from zotwatch.core.models import CandidateWork
from zotwatch.utils.datetime import ensure_aware

logger = logging.getLogger(__name__)

OPENALEX_API_BASE = "https://api.openalex.org"
DEFAULT_PER_PAGE = 200
DEFAULT_TIMEOUT = 30
POLITE_DELAY_S = 0.2  # 200ms between requests for polite pool


def _normalize_author_id(author_id: str) -> str:
    """Normalize author ID to OpenAlex format.

    Accepts:
      - OpenAlex ID: "A5023888391" or "a5023888391"
      - Full URL: "https://openalex.org/A5023888391"
      - ORCID: "0000-0001-2345-6789" or "https://orcid.org/0000-0001-2345-6789"

    Returns:
        Normalized filter value for OpenAlex API.
    """
    author_id = author_id.strip()

    # Full OpenAlex URL
    if "openalex.org/" in author_id:
        return author_id.split("/")[-1]

    # ORCID format (contains dashes in pattern XXXX-XXXX-XXXX-XXXX)
    if "-" in author_id and len(author_id.replace("-", "")) == 16:
        orcid = author_id.replace("https://orcid.org/", "")
        return f"https://orcid.org/{orcid}"

    # Already an OpenAlex ID
    return author_id


def _reconstruct_abstract(inverted_index: dict | None) -> str | None:
    """Reconstruct abstract from OpenAlex inverted index format.

    OpenAlex stores abstracts as {word: [positions]} to save space.
    We reconstruct the full text by placing words at their positions.
    """
    if not inverted_index:
        return None

    # Collect (position, word) pairs
    position_words: list[tuple[int, str]] = []
    for word, positions in inverted_index.items():
        for pos in positions:
            position_words.append((pos, word))

    if not position_words:
        return None

    # Sort by position and join
    position_words.sort(key=lambda x: x[0])
    return " ".join(word for _, word in position_words)


def _parse_work(work: dict, author_name: str) -> CandidateWork | None:
    """Parse an OpenAlex work object into a CandidateWork.

    Args:
        work: Raw work dict from OpenAlex API.
        author_name: Name of the followed author.

    Returns:
        CandidateWork or None if the work is invalid.
    """
    title = work.get("title")
    if not title:
        return None

    # Extract DOI (remove URL prefix)
    doi_raw = work.get("doi") or ""
    doi = doi_raw.replace("https://doi.org/", "") if doi_raw else None

    # Build URL: prefer DOI, fall back to OpenAlex landing page
    url = doi_raw or work.get("id")

    # Extract authors
    authors = []
    for authorship in work.get("authorships", []):
        name = authorship.get("author", {}).get("display_name")
        if name:
            authors.append(name)

    # Extract venue from primary location
    venue = None
    primary_loc = work.get("primary_location") or {}
    source_obj = primary_loc.get("source") or {}
    venue = source_obj.get("display_name")

    # Parse publication date
    published = None
    pub_date_str = work.get("publication_date")
    if pub_date_str:
        try:
            published = ensure_aware(datetime.fromisoformat(pub_date_str))
        except (ValueError, TypeError):
            pass

    # Reconstruct abstract
    abstract = _reconstruct_abstract(work.get("abstract_inverted_index"))

    # Extract OpenAlex work ID
    openalex_id = work.get("id", "").replace("https://openalex.org/", "")

    # Identifier: prefer DOI, fall back to OpenAlex ID
    identifier = doi or openalex_id
    if not identifier:
        return None

    return CandidateWork(
        source="openalex",
        identifier=identifier,
        title=title,
        abstract=abstract,
        authors=authors,
        doi=doi,
        url=url,
        published=published,
        venue=venue,
        metrics={"cited_by_count": work.get("cited_by_count", 0)},
        extra={
            "openalex_id": openalex_id,
            "followed_author": author_name,
            "work_type": work.get("type", ""),
        },
    )


class OpenAlexAuthorFetcher:
    """Fetch papers by specific authors from OpenAlex API.

    Supports full and incremental fetching with cursor-based pagination.
    """

    def __init__(
        self,
        polite_email: str = "",
        per_page: int = DEFAULT_PER_PAGE,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        self.session = requests.Session()
        self.per_page = per_page
        self.timeout = timeout

        # Set up polite pool headers
        headers = {"Accept": "application/json"}
        if polite_email:
            headers["User-Agent"] = f"ZotWatch/1.0 (mailto:{polite_email})"
        self.session.headers.update(headers)

    def fetch_author_works(
        self,
        author_id: str,
        author_name: str,
        *,
        from_date: str | None = None,
        max_results: int = 10000,
    ) -> list[CandidateWork]:
        """Fetch all works by a specific author.

        Args:
            author_id: OpenAlex author ID or ORCID.
            author_name: Display name for the author.
            from_date: Only fetch works published on or after this date (YYYY-MM-DD).
            max_results: Maximum number of works to fetch.

        Returns:
            List of CandidateWork objects.
        """
        normalized_id = _normalize_author_id(author_id)

        # Build filter
        if normalized_id.startswith("https://orcid.org/"):
            filter_str = f"author.orcid:{normalized_id}"
        else:
            filter_str = f"author.id:{normalized_id}"

        if from_date:
            filter_str += f",from_publication_date:{from_date}"

        works: list[CandidateWork] = []
        cursor = "*"
        page = 0

        logger.info(
            "Fetching works for author '%s' (id=%s)%s",
            author_name,
            author_id,
            f" from {from_date}" if from_date else " (full)",
        )

        while cursor and len(works) < max_results:
            params = {
                "filter": filter_str,
                "sort": "publication_date:desc",
                "per_page": min(self.per_page, max_results - len(works)),
                "cursor": cursor,
            }

            try:
                resp = self.session.get(
                    f"{OPENALEX_API_BASE}/works",
                    params=params,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()
            except requests.exceptions.RequestException as e:
                logger.warning(
                    "OpenAlex API error for author '%s': %s", author_name, e
                )
                break

            results = data.get("results", [])
            if not results:
                break

            for raw_work in results:
                candidate = _parse_work(raw_work, author_name)
                if candidate:
                    works.append(candidate)

            # Get next cursor
            meta = data.get("meta", {})
            cursor = meta.get("next_cursor")
            page += 1

            logger.debug(
                "Page %d: fetched %d works (total: %d)",
                page,
                len(results),
                len(works),
            )

            # Polite delay between requests
            time.sleep(POLITE_DELAY_S)

        logger.info(
            "Fetched %d works for author '%s'", len(works), author_name
        )
        return works

    def fetch_all_authors(
        self,
        authors: list[dict],
        *,
        last_dates: dict[str, str] | None = None,
        max_results_per_author: int = 10000,
    ) -> list[CandidateWork]:
        """Fetch works for all configured authors.

        Args:
            authors: List of author dicts with 'name' and 'id' keys.
            last_dates: Dict mapping author ID to last fetch date (YYYY-MM-DD).
            max_results_per_author: Max works per author.

        Returns:
            Combined list of CandidateWork from all authors.
        """
        if not authors:
            return []

        last_dates = last_dates or {}
        all_works: list[CandidateWork] = []

        for author in authors:
            author_id = author.get("id", "")
            author_name = author.get("name", author_id)

            if not author_id:
                logger.warning("Skipping author with no ID: %s", author)
                continue

            from_date = last_dates.get(author_id)
            works = self.fetch_author_works(
                author_id,
                author_name,
                from_date=from_date,
                max_results=max_results_per_author,
            )
            all_works.extend(works)

        return all_works


__all__ = ["OpenAlexAuthorFetcher"]
