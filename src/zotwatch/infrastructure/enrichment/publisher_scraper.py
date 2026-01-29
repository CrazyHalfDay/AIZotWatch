"""Abstract scraper with Camoufox browser and rule-based + LLM extraction.

Features:
- Concurrent batch fetching with configurable parallelism
- Publisher-specific rule-based extraction (ACM, IEEE, Springer, Elsevier, etc.)
- LLM fallback for unknown publishers or failed rules
- Cloudflare bypass via camoufox-captcha

Flow: DOI -> doi.org redirect -> HTML -> Rules extract -> (LLM fallback) -> abstract
"""

import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from zotwatch.llm.base import BaseLLMProvider

from .llm_extractor import LLMAbstractExtractor
from .publisher_extractors import PublisherExtractor, extract_abstract
from .stealth_browser import StealthBrowser

logger = logging.getLogger(__name__)

# Type alias for result callback: (doi, abstract_or_none) -> None
ResultCallback = Callable[[str, str | None], None]

# Default max workers for concurrent fetching
DEFAULT_MAX_WORKERS = 3


class AbstractScraper:
    """DOI-based abstract scraper with concurrent fetching.

    Features:
    - Concurrent batch processing with configurable parallelism
    - Rate limiting per worker to avoid overloading servers
    - Rule-based extraction for major publishers
    - LLM fallback when rules fail
    - Cloudflare bypass via Camoufox
    """

    def __init__(
        self,
        llm: BaseLLMProvider | None = None,
        rate_limit_delay: float = 1.0,
        timeout: int = 60000,
        max_retries: int = 2,
        max_html_chars: int = 15000,
        llm_max_tokens: int = 1024,
        llm_temperature: float = 0.1,
        use_llm_fallback: bool = True,
        max_workers: int | None = None,
    ):
        """Initialize the abstract scraper.

        Args:
            llm: LLM provider for fallback extraction. Optional if use_llm_fallback=False.
            rate_limit_delay: Minimum seconds between requests (per worker).
            timeout: Page load timeout in milliseconds.
            max_retries: Maximum retry attempts for Cloudflare challenges.
            max_html_chars: Maximum HTML chars to send to LLM.
            llm_max_tokens: Maximum tokens for LLM response.
            llm_temperature: LLM temperature for extraction.
            use_llm_fallback: Whether to use LLM when rules fail.
            max_workers: Maximum concurrent workers for batch fetching.
        """
        self.rate_limit_delay = rate_limit_delay
        self.timeout = timeout
        self.max_retries = max_retries
        self.use_llm_fallback = use_llm_fallback
        self.max_workers = max_workers or DEFAULT_MAX_WORKERS

        # Publisher-specific extractor
        self.publisher_extractor = PublisherExtractor(use_llm_fallback=use_llm_fallback)

        # LLM extractor (optional fallback)
        self.llm_extractor: LLMAbstractExtractor | None = None
        if llm and use_llm_fallback:
            self.llm_extractor = LLMAbstractExtractor(
                llm=llm,
                max_html_chars=max_html_chars,
                max_tokens=llm_max_tokens,
                temperature=llm_temperature,
            )

        # Thread-safe rate limiting
        self._rate_lock = threading.Lock()
        self._last_request_time = 0.0

    def _wait_for_rate_limit(self):
        """Respect rate limit between requests (thread-safe)."""
        with self._rate_lock:
            elapsed = time.time() - self._last_request_time
            if elapsed < self.rate_limit_delay:
                sleep_time = self.rate_limit_delay - elapsed
                logger.debug("Rate limiting: sleeping %.1fs", sleep_time)
                time.sleep(sleep_time)
            self._last_request_time = time.time()

    def _extract_abstract(
        self,
        html: str,
        url: str,
        title: str | None = None,
    ) -> str | None:
        """Extract abstract using rules first, then LLM fallback.

        Args:
            html: Page HTML content.
            url: Final page URL (for publisher detection).
            title: Optional paper title for LLM context.

        Returns:
            Extracted abstract or None.
        """
        # Try rule-based extraction first
        abstract = extract_abstract(html, url)
        if abstract:
            return abstract

        # LLM fallback
        if self.llm_extractor and self.use_llm_fallback:
            logger.debug("Rule extraction failed, trying LLM fallback")
            abstract = self.llm_extractor.extract(html, title)
            if abstract:
                return abstract

        return None

    def fetch_abstract(
        self,
        doi: str,
        title: str | None = None,
    ) -> str | None:
        """Fetch abstract for a single DOI.

        Args:
            doi: Digital Object Identifier.
            title: Optional paper title for better extraction.

        Returns:
            Extracted abstract or None.
        """
        self._wait_for_rate_limit()

        doi_url = f"https://doi.org/{doi}"
        html, final_url = StealthBrowser.fetch_page(
            doi_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

        if not html:
            logger.debug("Failed to fetch page for DOI %s", doi)
            return None

        logger.debug("DOI %s resolved to %s", doi, final_url)

        abstract = self._extract_abstract(html, final_url or doi_url, title)

        if abstract:
            logger.info("Extracted abstract for %s (%d chars)", doi, len(abstract))

        return abstract

    def fetch_batch(
        self,
        items: list[dict[str, str]],
        on_result: ResultCallback | None = None,
    ) -> dict[str, str]:
        """Fetch abstracts for multiple DOIs with concurrent processing.

        Uses ThreadPoolExecutor for parallel fetching while respecting
        rate limits through a shared lock.

        Args:
            items: List of dicts with 'doi' and optional 'title'.
            on_result: Optional callback called for each result as it completes.
                       Signature: (doi: str, abstract: Optional[str]) -> None

        Returns:
            Dict mapping DOI to abstract.
        """
        if not items:
            return {}

        # Filter out items without DOI
        valid_items = [item for item in items if item.get("doi")]
        if not valid_items:
            return {}

        total = len(valid_items)
        workers = min(self.max_workers, total)

        logger.info("Batch fetching %d DOIs with %d concurrent workers", total, workers)

        results: dict[str, str] = {}
        results_lock = threading.Lock()

        def fetch_one(item: dict[str, str]) -> tuple[str, str | None]:
            """Fetch a single DOI. Returns (doi, abstract_or_none)."""
            doi = item["doi"]
            title = item.get("title")
            abstract = self.fetch_abstract(doi, title)
            return (doi, abstract)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Submit all tasks
            futures = {executor.submit(fetch_one, item): item for item in valid_items}

            # Process results as they complete
            completed = 0
            for future in as_completed(futures):
                completed += 1
                try:
                    doi, abstract = future.result()
                    if abstract:
                        with results_lock:
                            results[doi] = abstract
                        logger.info(
                            "Fetched [%d/%d]: %s (%d chars)",
                            completed,
                            total,
                            doi,
                            len(abstract),
                        )
                    else:
                        logger.info("Fetched [%d/%d]: %s (no abstract)", completed, total, doi)

                    # Call callback for each result
                    if on_result:
                        on_result(doi, abstract)

                except Exception as e:
                    item = futures[future]
                    doi = item.get("doi", "unknown")
                    logger.warning("Failed to fetch %s: %s", doi, e)
                    if on_result:
                        on_result(doi, None)

        return results

    def close(self):
        """Clean up browser resources."""
        StealthBrowser.close()


__all__ = [
    "AbstractScraper",
    "ResultCallback",
]
