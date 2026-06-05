"""Abstract enrichment pipeline for candidates with missing abstracts."""

import logging
from dataclasses import dataclass
from pathlib import Path

from zotwatch.config.settings import Settings
from zotwatch.core.models import CandidateWork
from zotwatch.infrastructure.enrichment.cache import MetadataCache
from zotwatch.infrastructure.enrichment.publisher_scraper import AbstractScraper
from zotwatch.llm.base import BaseLLMProvider

logger = logging.getLogger(__name__)


@dataclass
class EnrichmentStats:
    """Statistics from the enrichment process."""

    total_candidates: int
    with_abstract: int
    missing_abstracts: int
    skipped_no_doi: int
    cache_hits: int
    api_fetched: int = 0  # Abstracts fetched via Crossref/OpenAlex APIs
    scraper_fetched: int = 0  # Abstracts fetched via scraper
    enriched: int = 0
    failed: int = 0

    @property
    def original_rate(self) -> float:
        """Original abstract completeness rate before enrichment."""
        if self.total_candidates == 0:
            return 0.0
        return self.with_abstract / self.total_candidates * 100

    @property
    def final_rate(self) -> float:
        """Final abstract completeness rate after enrichment."""
        if self.total_candidates == 0:
            return 0.0
        return (self.with_abstract + self.enriched) / self.total_candidates * 100


class AbstractEnricher:
    """Enriches candidates with missing abstracts from multiple sources.

    Uses an API-first, multi-tier strategy:
    1. Check local cache first (SQLite-backed)
    2. API fallback (Crossref + OpenAlex) - fast and free
    3. Browser scraper (Camoufox + rules + LLM) for whatever the APIs miss
    4. Negative-cache DOIs no source could resolve, to skip re-fetching dead
       links on subsequent runs
    """

    def __init__(
        self,
        settings: Settings,
        base_dir: Path,
        llm: BaseLLMProvider | None = None,
        cache: MetadataCache | None = None,
    ):
        """Initialize the enricher.

        Args:
            settings: Application settings.
            base_dir: Base directory for data files.
            llm: LLM provider for scraper extraction fallback.
            cache: Optional pre-configured cache (for testing).
        """
        self.config = settings.sources.scraper
        self.fallback_config = settings.sources.abstract_fallback
        self.base_dir = Path(base_dir)
        self.llm = llm

        # Initialize cache
        if cache is not None:
            self.cache = cache
        else:
            cache_path = self.base_dir / "data" / "metadata.sqlite"
            self.cache = MetadataCache(cache_path)

        # Ensure Camoufox profile lives under project data directory
        from zotwatch.infrastructure.enrichment.stealth_browser import StealthBrowser

        StealthBrowser.set_profile_path(self.base_dir / "data" / "camoufox_profile")

    def enrich(self, candidates: list[CandidateWork]) -> tuple[list[CandidateWork], EnrichmentStats]:
        """Enrich candidates with missing abstracts.

        Args:
            candidates: List of candidate works.

        Returns:
            Tuple of (enriched candidates, statistics).
        """
        scraper_enabled = self.config.enabled
        api_enabled = self.fallback_config.enabled
        if not scraper_enabled and not api_enabled:
            logger.debug("Abstract enrichment is disabled (scraper + API fallback off)")
            with_abstract = sum(1 for c in candidates if c.abstract)
            return candidates, EnrichmentStats(
                total_candidates=len(candidates),
                with_abstract=with_abstract,
                missing_abstracts=len(candidates) - with_abstract,
                skipped_no_doi=0,
                cache_hits=0,
                enriched=0,
                failed=0,
            )

        # Categorize candidates
        with_abstract = []
        needs_enrichment = []
        no_doi = []

        for c in candidates:
            if c.abstract:
                with_abstract.append(c)
            elif c.doi:
                needs_enrichment.append(c)
            else:
                no_doi.append(c)

        logger.info(
            "Abstract status: %d/%d (%.1f%%) have abstracts, %d need enrichment, %d have no DOI",
            len(with_abstract),
            len(candidates),
            len(with_abstract) / len(candidates) * 100 if candidates else 0,
            len(needs_enrichment),
            len(no_doi),
        )

        if not needs_enrichment:
            return candidates, EnrichmentStats(
                total_candidates=len(candidates),
                with_abstract=len(with_abstract),
                missing_abstracts=0,
                skipped_no_doi=len(no_doi),
                cache_hits=0,
                enriched=0,
                failed=0,
            )

        # Step 1: Check cache (positive hits only)
        dois_to_check = [c.doi for c in needs_enrichment]
        cached_abstracts = self.cache.get_batch(dois_to_check)
        cache_hits = len(cached_abstracts)

        logger.debug("Cache hits: %d/%d", cache_hits, len(dois_to_check))

        # Skip DOIs that are negative-cached (every source failed within TTL)
        uncached_dois = [doi for doi in dois_to_check if doi not in cached_abstracts]
        failed_cached = self.cache.get_failed_dois(uncached_dois)
        if failed_cached:
            logger.info("Skipping %d DOIs negative-cached as unresolved", len(failed_cached))
        to_fetch = [doi for doi in uncached_dois if doi not in failed_cached]

        # Step 2: API fallback first (Crossref + OpenAlex) - fast and free
        api_abstracts: dict[str, str] = {}
        if api_enabled and to_fetch:
            logger.info("API fallback: fetching %d papers via Crossref/OpenAlex...", len(to_fetch))
            api_abstracts = self._fetch_with_api(to_fetch, needs_enrichment)

        # Step 3: Browser scraper for whatever the APIs could not resolve
        remaining = [doi for doi in to_fetch if doi not in api_abstracts]
        scraper_abstracts: dict[str, str] = {}
        if scraper_enabled and remaining:
            logger.info("Scraper: fetching %d papers...", len(remaining))
            scraper_abstracts = self._fetch_with_scraper(remaining, needs_enrichment)

        # Step 4: Negative-cache DOIs that no source could resolve
        still_missing = [doi for doi in remaining if doi not in scraper_abstracts]
        if still_missing:
            self._negative_cache(still_missing)

        # Merge results from all sources
        all_abstracts = {**cached_abstracts, **api_abstracts, **scraper_abstracts}

        # Step 3: Apply abstracts to candidates
        enriched_count = 0
        for candidate in needs_enrichment:
            if candidate.doi in all_abstracts:
                candidate.abstract = all_abstracts[candidate.doi]
                enriched_count += 1

        failed = len(needs_enrichment) - enriched_count

        stats = EnrichmentStats(
            total_candidates=len(candidates),
            with_abstract=len(with_abstract),
            missing_abstracts=len(needs_enrichment),
            skipped_no_doi=len(no_doi),
            cache_hits=cache_hits,
            api_fetched=len(api_abstracts),
            scraper_fetched=len(scraper_abstracts),
            enriched=enriched_count,
            failed=failed,
        )

        logger.info(
            "Enrichment complete: %d/%d abstracts added "
            "(cache: %d, api: %d, scraper: %d, not found: %d)",
            stats.enriched,
            stats.missing_abstracts,
            stats.cache_hits,
            stats.api_fetched,
            stats.scraper_fetched,
            stats.failed,
        )

        # Provide helpful context about unindexed papers
        if stats.failed > 0 and stats.failed > stats.enriched:
            logger.info(
                "Note: %d papers not found - this may be due to access restrictions or extraction failures",
                stats.failed,
            )

        logger.info(
            "Abstract rate: %.1f%% -> %.1f%%",
            stats.original_rate,
            stats.final_rate,
        )

        return candidates, stats

    def _fetch_with_api(
        self,
        dois: list[str],
        candidates: list[CandidateWork],
    ) -> dict[str, str]:
        """Fetch abstracts via Crossref/OpenAlex APIs, caching each result.

        Args:
            dois: List of DOIs to fetch.
            candidates: List of candidates (for title context when caching).

        Returns:
            Dict mapping DOI to abstract.
        """
        from zotwatch.infrastructure.enrichment.api_fallback import ApiAbstractFetcher

        doi_to_title = {c.doi: c.title for c in candidates if c.doi}

        fetcher = ApiAbstractFetcher(
            mailto=self.fallback_config.mailto,
            use_crossref=self.fallback_config.use_crossref,
            use_openalex=self.fallback_config.use_openalex,
            use_elsevier=self.fallback_config.use_elsevier,
            elsevier_api_key=self.fallback_config.elsevier_api_key,
            use_springer=self.fallback_config.use_springer,
            springer_api_key=self.fallback_config.springer_api_key,
            timeout=self.fallback_config.timeout,
            max_workers=self.fallback_config.max_workers,
        )

        # Cache each resolved abstract immediately, tagged with its source
        def on_result(doi: str, abstract: str | None, source: str | None) -> None:
            if abstract:
                self.cache.put(
                    doi=doi,
                    abstract=abstract,
                    source=source or "api",
                    title=doi_to_title.get(doi),
                    ttl_days=30,
                )

        try:
            results = fetcher.fetch_batch(dois, on_result=on_result)
            if results:
                logger.info("API fallback: fetched %d/%d abstracts", len(results), len(dois))
            return results
        finally:
            fetcher.close()

    def _negative_cache(self, dois: list[str]) -> None:
        """Record DOIs no source could resolve, with a short TTL.

        Stored as rows with a NULL abstract so positive lookups ignore them,
        while ``get_failed_dois`` can skip re-fetching them until they expire.
        """
        ttl = self.fallback_config.negative_cache_ttl_days
        if ttl <= 0:
            return
        for doi in dois:
            self.cache.put(doi=doi, abstract=None, source="failed", ttl_days=ttl)
        logger.info("Negative-cached %d unresolved DOIs for %d days", len(dois), ttl)

    def _fetch_with_scraper(
        self,
        dois: list[str],
        candidates: list[CandidateWork],
    ) -> dict[str, str]:
        """Fetch abstracts using scraper (Camoufox + rules + LLM fallback).

        Uses sequential fetching with rate limiting.
        Caches each result immediately as it completes to prevent data loss.

        Args:
            dois: List of DOIs to fetch.
            candidates: List of candidates (for title context).

        Returns:
            Dict mapping DOI to abstract.
        """
        # Create DOI -> title mapping for extraction context
        doi_to_title = {c.doi: c.title for c in candidates if c.doi}

        # Build items list for batch processing
        items = [{"doi": doi, "title": doi_to_title.get(doi)} for doi in dois]

        scraper = AbstractScraper(
            llm=self.llm,
            rate_limit_delay=self.config.rate_limit_delay,
            timeout=self.config.timeout,
            max_retries=self.config.max_retries,
            max_html_chars=self.config.max_html_chars,
            llm_max_tokens=self.config.llm_max_tokens,
            llm_temperature=self.config.llm_temperature,
            use_llm_fallback=self.config.use_llm_fallback,
            max_workers=self.config.max_workers,
            block_resources=self.config.block_resources,
        )

        # Callback to cache results immediately as they complete
        def on_result(doi: str, abstract: str | None) -> None:
            if abstract:
                title = doi_to_title.get(doi)
                self.cache.put(
                    doi=doi,
                    abstract=abstract,
                    source="scraper",
                    title=title,
                    ttl_days=30,
                )

        try:
            # Fetch abstracts sequentially with immediate caching via callback
            results = scraper.fetch_batch(items, on_result=on_result)

            if results:
                logger.info("Scraper: fetched %d/%d abstracts", len(results), len(dois))
            return results
        finally:
            scraper.close()


def enrich_candidates(
    candidates: list[CandidateWork],
    settings: Settings,
    base_dir: Path,
    llm: BaseLLMProvider | None = None,
) -> tuple[list[CandidateWork], EnrichmentStats]:
    """Convenience function to enrich candidates with missing abstracts.

    Args:
        candidates: List of candidate works.
        settings: Application settings.
        base_dir: Base directory for data files.
        llm: Optional LLM provider for scraper fallback.

    Returns:
        Tuple of (enriched candidates, statistics).
    """
    enricher = AbstractEnricher(settings, base_dir, llm=llm)
    return enricher.enrich(candidates)


__all__ = ["AbstractEnricher", "EnrichmentStats", "enrich_candidates"]
