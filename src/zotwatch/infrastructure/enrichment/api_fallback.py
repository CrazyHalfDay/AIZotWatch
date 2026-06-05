"""API-based abstract fallback via Crossref and OpenAlex.

These endpoints are fast (sub-second) and free, so they are tried before the
expensive browser scraper. Crossref returns abstracts as JATS XML (when the
publisher deposits them); OpenAlex stores them as an inverted index that we
reconstruct. Notably, Elsevier does not deposit abstracts to Crossref but is
often covered by OpenAlex, which makes OpenAlex the more valuable fallback.
"""

import logging
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from zotwatch.sources.openalex import _reconstruct_abstract
from zotwatch.utils.text import clean_html

logger = logging.getLogger(__name__)

CROSSREF_WORKS_URL = "https://api.crossref.org/works/{doi}"
OPENALEX_WORK_URL = "https://api.openalex.org/works/https://doi.org/{doi}"
ELSEVIER_ARTICLE_URL = "https://api.elsevier.com/content/article/doi/{doi}"
SPRINGER_META_URL = "https://api.springernature.com/meta/v2/json"

# Elsevier DOIs share this registrant prefix; only these are worth an API call
ELSEVIER_DOI_PREFIX = "10.1016/"

# Springer Nature registrant prefixes (Springer journals + Nature-branded titles)
SPRINGER_DOI_PREFIXES = ("10.1007/", "10.1038/")

# Abstracts shorter than this are treated as noise (e.g. stray fragments)
MIN_ABSTRACT_CHARS = 80

# Callback: (doi, abstract_or_none, source_or_none) -> None
ApiResultCallback = Callable[[str, str | None, str | None], None]


class ApiAbstractFetcher:
    """Fetches abstracts from Crossref and OpenAlex APIs for a batch of DOIs."""

    def __init__(
        self,
        mailto: str = "",
        use_crossref: bool = True,
        use_openalex: bool = True,
        use_elsevier: bool = True,
        elsevier_api_key: str = "",
        use_springer: bool = True,
        springer_api_key: str = "",
        timeout: float = 15.0,
        max_workers: int = 8,
        publisher_max_workers: int = 2,
    ):
        """Initialize the API fetcher.

        Args:
            mailto: Email for Crossref/OpenAlex polite pools.
            use_crossref: Whether to query Crossref.
            use_openalex: Whether to query OpenAlex.
            use_elsevier: Whether to query the Elsevier Article API.
            elsevier_api_key: X-ELS-APIKey for the Elsevier Article API.
            use_springer: Whether to query the Springer Nature Meta API.
            springer_api_key: api_key for the Springer Nature Meta API.
            timeout: Per-request timeout in seconds.
            max_workers: Concurrent workers for batch fetching.
            publisher_max_workers: Max concurrent publisher-API (Elsevier/
                Springer) requests. Kept low to avoid rate-limit (429) drops
                that would otherwise fall through to the browser scraper.
        """
        self.mailto = mailto.strip()
        self.use_crossref = use_crossref
        self.use_openalex = use_openalex
        self.elsevier_api_key = elsevier_api_key.strip()
        self.use_elsevier = use_elsevier and bool(self.elsevier_api_key)
        self.springer_api_key = springer_api_key.strip()
        self.use_springer = use_springer and bool(self.springer_api_key)
        self.timeout = timeout
        self.max_workers = max(1, max_workers)
        # Throttle publisher APIs independently of the pool: Crossref/OpenAlex
        # stay fully parallel while Elsevier/Springer are capped to a few
        # in-flight requests so a burst does not trip their rate limits.
        self._publisher_semaphore = threading.Semaphore(max(1, publisher_max_workers))
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self._user_agent()})

    def _user_agent(self) -> str:
        ua = "ZotWatch/1.0 (abstract-enrichment)"
        if self.mailto:
            ua += f"; mailto:{self.mailto}"
        return ua

    def _params(self) -> dict[str, str]:
        return {"mailto": self.mailto} if self.mailto else {}

    def fetch_one(self, doi: str) -> tuple[str | None, str | None]:
        """Fetch an abstract for a single DOI, trying Crossref then OpenAlex.

        Returns:
            Tuple of (abstract, source) where source is
            "crossref"/"elsevier"/"springer"/"openalex", or (None, None).
        """
        if self.use_crossref:
            abstract = self._fetch_crossref(doi)
            if abstract:
                return abstract, "crossref"
        # Commercial publishers do not deposit abstracts to Crossref and
        # aggregators lag for fresh articles, so the publisher API is the only
        # reliable source. Route each DOI by its registrant prefix.
        if self.use_elsevier and doi.lower().startswith(ELSEVIER_DOI_PREFIX):
            abstract = self._fetch_elsevier(doi)
            if abstract:
                return abstract, "elsevier"
        if self.use_springer and doi.lower().startswith(SPRINGER_DOI_PREFIXES):
            abstract = self._fetch_springer(doi)
            if abstract:
                return abstract, "springer"
        if self.use_openalex:
            abstract = self._fetch_openalex(doi)
            if abstract:
                return abstract, "openalex"
        return None, None

    def _fetch_crossref(self, doi: str) -> str | None:
        """Fetch and clean an abstract from Crossref's single-work endpoint."""
        url = CROSSREF_WORKS_URL.format(doi=doi)
        try:
            resp = self.session.get(url, params=self._params(), timeout=self.timeout)
            if resp.status_code != 200:
                return None
            message = resp.json().get("message", {})
            abstract = clean_html(message.get("abstract"))
        except (requests.RequestException, ValueError) as exc:
            logger.debug("Crossref fallback failed for %s: %s", doi, exc)
            return None
        if abstract and len(abstract) >= MIN_ABSTRACT_CHARS:
            return abstract
        return None

    def _fetch_elsevier(self, doi: str) -> str | None:
        """Fetch an abstract from the Elsevier Article API (``view=META_ABS``).

        Elsevier abstracts are absent from Crossref and the open aggregators
        (licensing + indexing lag), so the publisher API is the only reliable
        source for freshly published ScienceDirect articles.
        """
        url = ELSEVIER_ARTICLE_URL.format(doi=doi)
        headers = {"X-ELS-APIKey": self.elsevier_api_key, "Accept": "application/json"}
        try:
            with self._publisher_semaphore:
                resp = self.session.get(
                    url, headers=headers, params={"view": "META_ABS"}, timeout=self.timeout
                )
            if resp.status_code != 200:
                return None
            coredata = resp.json().get("full-text-retrieval-response", {}).get("coredata", {})
            abstract = clean_html(coredata.get("dc:description"))
        except (requests.RequestException, ValueError, AttributeError) as exc:
            logger.debug("Elsevier fallback failed for %s: %s", doi, exc)
            return None
        if abstract and len(abstract) >= MIN_ABSTRACT_CHARS:
            return abstract
        return None

    def _fetch_springer(self, doi: str) -> str | None:
        """Fetch an abstract from the Springer Nature Meta API.

        Covers both Springer (10.1007) and Nature-branded (10.1038) journals,
        including subscription titles whose abstracts are absent from Crossref
        and lag in the open aggregators. The free key returns metadata only.
        """
        params = {"q": f"doi:{doi}", "api_key": self.springer_api_key}
        try:
            with self._publisher_semaphore:
                resp = self.session.get(SPRINGER_META_URL, params=params, timeout=self.timeout)
            if resp.status_code != 200:
                return None
            records = resp.json().get("records") or []
            abstract = clean_html(records[0].get("abstract")) if records else None
        except (requests.RequestException, ValueError, AttributeError, IndexError) as exc:
            logger.debug("Springer fallback failed for %s: %s", doi, exc)
            return None
        if abstract and len(abstract) >= MIN_ABSTRACT_CHARS:
            return abstract
        return None

    def _fetch_openalex(self, doi: str) -> str | None:
        """Fetch an abstract from OpenAlex and reconstruct it from the index."""
        url = OPENALEX_WORK_URL.format(doi=doi)
        try:
            resp = self.session.get(url, params=self._params(), timeout=self.timeout)
            if resp.status_code != 200:
                return None
            work = resp.json()
            abstract = _reconstruct_abstract(work.get("abstract_inverted_index"))
        except (requests.RequestException, ValueError) as exc:
            logger.debug("OpenAlex fallback failed for %s: %s", doi, exc)
            return None
        if abstract and len(abstract) >= MIN_ABSTRACT_CHARS:
            return abstract
        return None

    def fetch_batch(
        self,
        dois: list[str],
        on_result: ApiResultCallback | None = None,
    ) -> dict[str, str]:
        """Fetch abstracts for multiple DOIs concurrently.

        Args:
            dois: DOIs to look up.
            on_result: Optional callback invoked per DOI as results complete,
                with signature (doi, abstract_or_none, source_or_none).

        Returns:
            Dict mapping DOI to abstract for DOIs that were resolved.
        """
        if not dois:
            return {}

        results: dict[str, str] = {}
        total = len(dois)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_doi = {executor.submit(self.fetch_one, doi): doi for doi in dois}
            completed = 0
            for future in as_completed(future_to_doi):
                doi = future_to_doi[future]
                completed += 1
                try:
                    abstract, source = future.result()
                except Exception as exc:  # noqa: BLE001 - never break the batch
                    logger.debug("API fallback error for %s: %s", doi, exc)
                    abstract, source = None, None
                if abstract:
                    results[doi] = abstract
                    logger.info(
                        "API fallback [%d/%d]: %s via %s (%d chars)",
                        completed,
                        total,
                        doi,
                        source,
                        len(abstract),
                    )
                else:
                    logger.debug("API fallback [%d/%d]: %s (no abstract)", completed, total, doi)
                if on_result:
                    on_result(doi, abstract, source)

        return results

    def close(self) -> None:
        """Close the underlying HTTP session."""
        self.session.close()


__all__ = ["ApiAbstractFetcher"]
