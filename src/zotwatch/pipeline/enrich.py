"""Abstract enrichment pipeline for candidates with missing abstracts."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

    Uses a two-tier strategy:
    1. Check local cache first (SQLite-backed)
    2. Use abstract scraper (Camoufox + rules + LLM) for cache misses
    """

    def __init__(
        self,
        settings: Settings,
        base_dir: Path,
        llm: Optional[BaseLLMProvider] = None,
        cache: Optional[MetadataCache] = None,
    ):
        """Initialize the enricher.

        Args:
            settings: Application settings.
            base_dir: Base directory for data files.
            llm: LLM provider for scraper extraction fallback.
            cache: Optional pre-configured cache (for testing).
        """
        self.config = settings.sources.scraper
        self.base_dir = Path(base_dir)
        self.llm = llm

        # Initialize cache
        if cache is not None:
            self.cache = cache
        else:
            cache_path = self.base_dir / "data" / "metadata.sqlite"
            self.cache = MetadataCache(cache_path)

    def enrich(self, candidates: List[CandidateWork]) -> Tuple[List[CandidateWork], EnrichmentStats]:
        """Enrich candidates with missing abstracts.

        Args:
            candidates: List of candidate works.

        Returns:
            Tuple of (enriched candidates, statistics).
        """
        if not self.config.enabled:
            logger.debug("Abstract scraper enrichment is disabled")
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

        # Step 1: Check cache
        dois_to_check = [c.doi for c in needs_enrichment]
        cached_abstracts = self.cache.get_batch(dois_to_check)
        cache_hits = len(cached_abstracts)

        logger.debug("Cache hits: %d/%d", cache_hits, len(dois_to_check))

        # Step 2: Scraper for cache misses
        uncached_dois = [doi for doi in dois_to_check if doi not in cached_abstracts]
        scraper_abstracts: Dict[str, str] = {}

        if uncached_dois:
            logger.info("Scraper: fetching %d papers...", len(uncached_dois))
            scraper_abstracts = self._fetch_with_scraper(uncached_dois, needs_enrichment)

        # Merge results from all sources
        all_abstracts = {**cached_abstracts, **scraper_abstracts}

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
            scraper_fetched=len(scraper_abstracts),
            enriched=enriched_count,
            failed=failed,
        )

        logger.info(
            "Enrichment complete: %d/%d abstracts added (cache: %d, scraper: %d, not found: %d)",
            stats.enriched,
            stats.missing_abstracts,
            stats.cache_hits,
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

    def _fetch_with_scraper(
        self,
        dois: List[str],
        candidates: List[CandidateWork],
    ) -> Dict[str, str]:
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
        )

        # Callback to cache results immediately as they complete
        def on_result(doi: str, abstract: Optional[str]) -> None:
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
    candidates: List[CandidateWork],
    settings: Settings,
    base_dir: Path,
    llm: Optional[BaseLLMProvider] = None,
) -> Tuple[List[CandidateWork], EnrichmentStats]:
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
