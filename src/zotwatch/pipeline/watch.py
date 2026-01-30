"""Watch pipeline orchestrator.

Consolidates all business logic from cli/main.py watch command
into a single, testable pipeline class.
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from zotwatch.config.settings import Settings
from zotwatch.core.models import (
    CandidateWork,
    ClusteredProfile,
    InterestWork,
    OverallSummary,
    RankedWork,
    ResearcherProfile,
)
from zotwatch.infrastructure.embedding import (
    CachingEmbeddingProvider,
    EmbeddingCache,
    create_embedding_provider,
    create_reranker,
)
from zotwatch.infrastructure.enrichment.cache import MetadataCache
from zotwatch.infrastructure.storage import ProfileStorage
from zotwatch.llm import (
    InterestRefiner,
    LibraryAnalyzer,
    OverallSummarizer,
    PaperDomainClassifier,
    PaperRelevanceFilter,
    PaperSummarizer,
    TitleTranslator,
)
from zotwatch.llm.base import BaseLLMProvider
from zotwatch.llm.factory import create_llm_client
from zotwatch.pipeline import DedupeEngine, InterestRanker, ProfileBuilder, ProfileRanker, ProfileStatsExtractor
from zotwatch.pipeline.enrich import AbstractEnricher, EnrichmentStats
from zotwatch.pipeline.fetch import CandidateFetcher
from zotwatch.pipeline.filters import (
    exclude_by_keywords,
    filter_by_interest_similarity,
    filter_recent,
    filter_without_abstract,
    include_by_keywords,
    limit_preprints,
)
from zotwatch.pipeline.profile_ranker import ComputedThresholds
from zotwatch.sources.zotero import ZoteroIngestor

logger = logging.getLogger(__name__)


@dataclass
class WatchConfig:
    """Configuration for watch pipeline execution.

    Can be overridden by CLI arguments or loaded from settings.
    """

    top_k: int = 20
    recent_days: int = 7
    max_preprint_ratio: float = 0.9
    require_abstract: bool = True
    generate_summaries: bool = True
    translate_titles: bool = False


@dataclass
class WatchStats:
    """Statistics from watch pipeline execution."""

    candidates_fetched: int = 0
    candidates_after_dedupe: int = 0
    candidates_after_include_filter: int = 0
    candidates_after_keyword_filter: int = 0  # After exclude_keywords filtering
    candidates_after_abstract_filter: int = 0
    candidates_after_semantic_filter: int = 0
    candidates_after_llm_filter: int = 0
    candidates_after_ranking: int = 0  # After similarity gate in ProfileRanker
    candidates_after_recent_filter: int = 0
    abstracts_enriched: int = 0
    summaries_generated: int = 0
    interest_papers_selected: int = 0


@dataclass
class WatchResult:
    """Complete result from watch pipeline execution."""

    ranked_works: list[RankedWork] = field(default_factory=list)
    interest_works: list[InterestWork] = field(default_factory=list)
    followed_works: list[RankedWork] = field(default_factory=list)
    researcher_profile: ResearcherProfile | None = None
    overall_summaries: dict[str, OverallSummary] = field(default_factory=dict)
    stats: WatchStats = field(default_factory=WatchStats)
    computed_thresholds: ComputedThresholds | None = None


class WatchPipeline:
    """Orchestrates the complete watch workflow.

    Extracts business logic from cli/main.py into a testable,
    reusable pipeline class.
    """

    def __init__(
        self,
        base_dir: Path | str,
        settings: Settings,
        config: WatchConfig | None = None,
        embedding_cache: EmbeddingCache | None = None,
    ):
        """Initialize watch pipeline.

        Args:
            base_dir: Base directory for data files.
            settings: Application settings.
            config: Pipeline configuration (uses settings defaults if None).
            embedding_cache: Optional shared embedding cache.
        """
        self.base_dir = Path(base_dir)
        self.settings = settings

        # Merge config with settings defaults
        if config is None:
            config = WatchConfig(
                top_k=settings.watch.top_k,
                recent_days=settings.watch.recent_days,
                max_preprint_ratio=settings.watch.max_preprint_ratio,
                require_abstract=settings.watch.require_abstract,
                generate_summaries=settings.llm.enabled,
                translate_titles=settings.llm.enabled and settings.llm.translation.enabled,
            )
        self.config = config

        # Lazy-initialized resources
        self._llm_client: BaseLLMProvider | None = None
        self._storage: ProfileStorage | None = None
        self._embedding_cache = embedding_cache

    def _get_storage(self) -> ProfileStorage:
        """Get or create storage instance."""
        if self._storage is None:
            self._storage = ProfileStorage(self.base_dir / "data" / "profile.sqlite")
            self._storage.initialize()
        return self._storage

    def _get_embedding_cache(self) -> EmbeddingCache:
        """Get or create embedding cache."""
        if self._embedding_cache is None:
            cache_db_path = self.base_dir / "data" / "embeddings.sqlite"
            self._embedding_cache = EmbeddingCache(cache_db_path)
        return self._embedding_cache

    def _get_llm_client(self) -> BaseLLMProvider | None:
        """Get or create LLM client (lazy singleton)."""
        if self._llm_client is None and self.settings.llm.enabled:
            self._llm_client = create_llm_client(self.settings.llm)
        return self._llm_client

    def _ensure_profile_exists(
        self,
        on_progress: Callable[[str, str], None] | None = None,
    ) -> bool:
        """Check if profile exists, return True if it was built."""
        storage = self._get_storage()
        current_signature = self.settings.embedding.signature
        stored_signature = storage.get_metadata("embedding_signature")
        faiss_path = self.base_dir / "data" / "faiss.index"
        sqlite_path = self.base_dir / "data" / "profile.sqlite"

        # Rebuild when artifacts are missing or embedding provider/model changed
        has_artifacts = faiss_path.exists() and sqlite_path.exists()
        signature_mismatch = stored_signature != current_signature

        if has_artifacts and not signature_mismatch:
            return False

        if signature_mismatch:
            logger.info(
                "Embedding provider/model changed (was '%s', now '%s'); rebuilding profile embeddings and FAISS index",
                stored_signature or "<unknown>",
                current_signature,
            )
        else:
            logger.info("Profile artifacts missing; rebuilding profile embeddings and FAISS index")

        # Build profile
        logger.info("Building profile from Zotero library...")
        self._build_profile(full=True, on_progress=on_progress)
        return True

    def _build_profile(
        self,
        full: bool = True,
        on_progress: Callable[[str, str], None] | None = None,
    ) -> None:
        """Build user profile from Zotero library (ingest + embeddings)."""
        storage = self._get_storage()
        self._run_ingest(storage, full=full, on_progress=on_progress)
        self._build_profile_from_storage(full=full)

    def _build_profile_from_storage(self, *, full: bool = False) -> None:
        """Build embeddings + FAISS index from items already in storage."""
        embedding_cache = self._get_embedding_cache()

        vectorizer = create_embedding_provider(self.settings.embedding)

        builder = ProfileBuilder(
            self.base_dir,
            self._get_storage(),
            self.settings,
            vectorizer=vectorizer,
            embedding_cache=embedding_cache,
        )
        builder.run(full=full)

    def _run_ingest(
        self,
        storage: ProfileStorage,
        *,
        full: bool,
        on_progress: Callable[[str, str], None] | None = None,
    ):
        """Run Zotero ingest with optional progress callbacks."""
        ingestor = ZoteroIngestor(storage, self.settings)
        return ingestor.run(full=full, on_progress=on_progress)

    def run(
        self,
        on_progress: Callable[[str, str], None] | None = None,
    ) -> WatchResult:
        """Execute the complete watch pipeline.

        Args:
            on_progress: Optional callback for progress updates.
                        Called with (stage_name: str, message: str).

        Returns:
            WatchResult containing ranked works, statistics, and optional
            researcher profile, summaries, and interest works.

        Note:
            Resources are cleaned up in a finally block to ensure cleanup
            happens even if an exception occurs during pipeline execution.
        """
        result = WatchResult()
        storage = self._get_storage()
        embedding_cache = self._get_embedding_cache()
        pipeline_start = time.time()

        def progress(stage: str, msg: str) -> None:
            logger.info("[%s] %s", stage, msg)
            if on_progress:
                on_progress(stage, msg)

        try:
            progress("start", "Starting ZotWatch pipeline...")

            # 1. Ensure profile exists
            profile_built = self._ensure_profile_exists(on_progress=progress)
            if profile_built:
                progress("profile", "Profile built from Zotero library")

            ingest_stats = None

            # 2. Incremental Zotero sync (skip if we just did a full rebuild)
            if not profile_built:
                progress("sync", "Syncing with Zotero...")
                ingest_stats = self._run_ingest(storage, full=False, on_progress=progress)

                # 2.1 Refresh profile artifacts when library changed
                if ingest_stats and (ingest_stats.fetched > 0 or ingest_stats.removed > 0):
                    progress("profile", "Updating profile embeddings and FAISS index...")
                    self._build_profile_from_storage(full=False)

            # 3. Analyze researcher profile (optional)
            if self.settings.llm.enabled:
                result.researcher_profile = self._analyze_profile(storage, progress)

            # 3.5 Fetch followed authors (skip scoring, only dedupe)
            followed_cfg = self.settings.sources.followed_authors
            if followed_cfg.enabled and followed_cfg.authors:
                result.followed_works = self._fetch_followed_authors(storage, progress)

            # 4. Fetch candidates
            fetch_start = time.time()
            progress("fetch", "Fetching candidates from configured sources...")
            fetcher = CandidateFetcher(self.settings, self.base_dir)
            candidates = fetcher.fetch_all()
            result.stats.candidates_fetched = len(candidates)
            fetch_elapsed = time.time() - fetch_start
            progress("fetch", f"Found {len(candidates)} candidates ({fetch_elapsed:.1f}s)")

            # 5. Enrich abstracts (optional)
            if self.settings.sources.scraper.enabled:
                candidates, enrich_stats = self._enrich_abstracts(candidates, progress)
                result.stats.abstracts_enriched = enrich_stats.enriched

            # 6. Deduplicate
            progress("dedupe", "Filtering duplicates against library...")
            dedupe = DedupeEngine(storage, title_threshold=self.settings.watch.dedupe_threshold)
            before_dedupe = len(candidates)
            candidates = dedupe.filter(candidates)
            result.stats.candidates_after_dedupe = len(candidates)
            progress("dedupe", f"Removed {before_dedupe - len(candidates)} duplicates ({len(candidates)} remaining)")

            # 6.5 Exclude by keywords (if configured)
            interests_config = self.settings.scoring.interests
            if interests_config.include_keywords:
                candidates, removed = include_by_keywords(
                    candidates,
                    interests_config.include_keywords,
                    min_matches=interests_config.include_min_matches,
                    fields=tuple(interests_config.include_match_fields),
                )
                result.stats.candidates_after_include_filter = len(candidates)
                if removed > 0:
                    progress("filter", f"Removed {removed} candidates missing include keywords")
            else:
                result.stats.candidates_after_include_filter = len(candidates)

            if interests_config.exclude_keywords:
                candidates, removed = exclude_by_keywords(
                    candidates, interests_config.exclude_keywords
                )
                result.stats.candidates_after_keyword_filter = len(candidates)
                if removed > 0:
                    progress("filter", f"Excluded {removed} candidates by keywords")
            else:
                result.stats.candidates_after_keyword_filter = len(candidates)

            # 7. Filter without abstract (if required)
            if self.config.require_abstract:
                candidates, removed = filter_without_abstract(candidates)
                result.stats.candidates_after_abstract_filter = len(candidates)
                if removed > 0:
                    progress("filter", f"Removed {removed} candidates without abstracts")
            else:
                result.stats.candidates_after_abstract_filter = len(candidates)

            # 7.25 Semantic interest filter (optional)
            if (
                interests_config.semantic_filter_enabled
                and interests_config.description.strip()
                and candidates
            ):
                progress("filter", "Applying semantic interest filter...")
                semantic_vectorizer = CachingEmbeddingProvider(
                    provider=create_embedding_provider(self.settings.embedding),
                    cache=embedding_cache,
                    source_type="candidate",
                    ttl_days=self.settings.embedding.candidate_ttl_days,
                )
                candidates, removed = filter_by_interest_similarity(
                    candidates,
                    query=interests_config.description,
                    vectorizer=semantic_vectorizer,
                    min_similarity=interests_config.semantic_filter_min_similarity,
                    max_candidates=interests_config.semantic_filter_max_candidates,
                )
                result.stats.candidates_after_semantic_filter = len(candidates)
                if removed > 0:
                    progress("filter", f"Removed {removed} candidates by semantic similarity")
            else:
                result.stats.candidates_after_semantic_filter = len(candidates)

            # 7.5 LLM relevance filter (optional)
            if interests_config.llm_relevance_filter_enabled and self.settings.llm.enabled:
                llm_client = self._get_llm_client()
                if llm_client:
                    progress("filter", "Applying LLM relevance filter...")
                    llm_filter = PaperRelevanceFilter(
                        llm_client,
                        model=self.settings.llm.model,
                        batch_size=interests_config.llm_relevance_batch_size,
                        max_candidates=interests_config.llm_relevance_max_candidates,
                    )
                    candidates, removed = llm_filter.filter_candidates(
                        candidates,
                        user_interests=interests_config.description,
                    )
                    result.stats.candidates_after_llm_filter = len(candidates)
                    if removed > 0:
                        progress("filter", f"Removed {removed} candidates by LLM relevance")
                else:
                    result.stats.candidates_after_llm_filter = len(candidates)
            else:
                result.stats.candidates_after_llm_filter = len(candidates)

            # 8. Interest-based selection (optional)
            if interests_config.enabled and interests_config.description.strip():
                result.interest_works = self._select_interest_papers(candidates, embedding_cache, progress)
                result.stats.interest_papers_selected = len(result.interest_works)

            # 9. Rank by profile similarity
            rank_start = time.time()
            progress("rank", "Computing relevance scores...")
            ranker = ProfileRanker(self.base_dir, self.settings, embedding_cache=embedding_cache)
            ranked = ranker.rank(candidates)
            result.computed_thresholds = ranker.computed_thresholds
            result.stats.candidates_after_ranking = len(ranked)
            rank_elapsed = time.time() - rank_start
            progress("rank", f"Scored {len(ranked)} candidates ({rank_elapsed:.1f}s)")

            # 10. Apply filters
            ranked = filter_recent(ranked, days=self.config.recent_days)
            result.stats.candidates_after_recent_filter = len(ranked)
            ranked = limit_preprints(ranked, max_ratio=self.config.max_preprint_ratio)

            # 11. Apply top_k limit
            if self.config.top_k and len(ranked) > self.config.top_k:
                ranked = ranked[: self.config.top_k]

            result.ranked_works = ranked
            progress("rank", f"Final: {len(ranked)} recommendations")

            # 12. Generate AI summaries (optional)
            if self.config.generate_summaries and self.settings.llm.enabled and ranked:
                self._generate_summaries(result, storage, progress)

            # 13. Translate titles (optional)
            if self.config.translate_titles and self.settings.llm.enabled:
                self._translate_titles(result, storage, progress)

            # 14. Classify domains (optional)
            if self.settings.llm.domain_classification.enabled and self.settings.llm.enabled:
                self._classify_domains(result, progress)

            # Report total elapsed time and pipeline funnel
            total_elapsed = time.time() - pipeline_start
            progress("done", f"Pipeline complete: {len(result.ranked_works)} recommendations in {total_elapsed:.1f}s")

            # Log pipeline funnel summary for diagnosis
            s = result.stats
            funnel_lines = [
                f"  Fetched:          {s.candidates_fetched}",
                f"  After dedupe:     {s.candidates_after_dedupe}",
                f"  After include kw: {s.candidates_after_include_filter}",
                f"  After exclude kw: {s.candidates_after_keyword_filter}",
                f"  After abstract:   {s.candidates_after_abstract_filter}",
                f"  After semantic:   {s.candidates_after_semantic_filter}",
                f"  After LLM:        {s.candidates_after_llm_filter}",
                f"  After ranking:    {s.candidates_after_ranking}",
                f"  After recency:    {s.candidates_after_recent_filter}",
                f"  Final output:     {len(result.ranked_works)}",
            ]
            progress("stats", "Pipeline funnel:\n" + "\n".join(funnel_lines))

            return result

        finally:
            # Always cleanup caches, even if an exception occurred
            try:
                self._cleanup_caches(embedding_cache, progress)
            except Exception as e:
                logger.warning("Cache cleanup failed: %s", e)

    def _analyze_profile(
        self,
        storage: ProfileStorage,
        progress: Callable[[str, str], None],
    ) -> ResearcherProfile | None:
        """Analyze researcher profile from library."""
        progress("profile", "Analyzing researcher profile...")
        all_items = storage.get_all_items()

        if not all_items:
            progress("profile", "No items in library, skipping")
            return None

        # Check cache
        stats_extractor = ProfileStatsExtractor()
        current_hash = stats_extractor.compute_library_hash(all_items)
        cached_profile = storage.get_profile_analysis(current_hash)

        if cached_profile:
            # Check if LLM model changed - need to regenerate AI insights
            # Only regenerate if LLM is enabled and model differs from cached
            current_model = self.settings.llm.model if self.settings.llm.enabled else None
            should_regenerate = self.settings.llm.enabled and cached_profile.model_used != current_model
            if not should_regenerate:
                progress("profile", "Using cached profile analysis")
                # Still need to load clustered profile even with cached base profile
                self._load_clustered_profile(cached_profile, storage, progress)
                return cached_profile
            else:
                logger.info(
                    "LLM model changed (%s -> %s), regenerating AI insights",
                    cached_profile.model_used,
                    current_model,
                )
                progress("profile", "LLM model changed, regenerating AI insights...")

        # Extract statistics
        progress("profile", "Extracting library statistics...")
        profile = stats_extractor.extract_all(
            all_items,
            author_min_count=self.settings.profile.author_min_count,
        )

        # Use LLM for insights
        llm_client = self._get_llm_client()
        if llm_client:
            try:
                analyzer = LibraryAnalyzer(llm_client, model=self.settings.llm.model)
                progress("profile", "Classifying research domains...")
                profile.domains = analyzer.classify_domains(all_items)
                progress("profile", "Generating AI insights...")
                profile.insights = analyzer.generate_insights(profile)
                profile.model_used = self.settings.llm.model
                storage.save_profile_analysis(profile)
                progress("profile", "Profile analysis complete and cached")
            except Exception as e:
                logger.warning("Failed to generate profile insights: %s", e)
                progress("profile", f"AI insights skipped (error: {e})")

        # Load clustered profile if available
        self._load_clustered_profile(profile, storage, progress)

        return profile

    def _load_clustered_profile(
        self,
        profile: ResearcherProfile,
        storage: ProfileStorage,
        progress: Callable[[str, str], None],
    ) -> None:
        """Load clustered profile and optionally generate LLM labels."""
        if not self.settings.profile.clustering.enabled:
            return

        embedding_signature = self.settings.embedding.signature
        clustered = storage.get_clustered_profile(embedding_signature)

        if not clustered or clustered.valid_cluster_count == 0:
            return

        # Generate LLM labels for clusters if enabled
        llm_client = self._get_llm_client()
        if self.settings.profile.clustering.generate_labels and llm_client:
            try:
                self._label_clusters(clustered, llm_client, progress)
                # Update cached clustered profile with labels
                storage.save_clustered_profile(clustered)
            except Exception as e:
                logger.warning("Failed to generate cluster labels: %s", e)

        profile.clustered_profile = clustered
        progress(
            "profile",
            f"Loaded {clustered.valid_cluster_count} research clusters",
        )

    def _label_clusters(
        self,
        clustered: ClusteredProfile,
        llm_client: BaseLLMProvider,
        progress: Callable[[str, str], None],
    ) -> None:
        """Generate LLM labels for clusters."""
        from zotwatch.llm.cluster_labeler import ClusterLabeler

        # Only label clusters that don't have labels yet
        unlabeled = [c for c in clustered.clusters if not c.label]
        if not unlabeled:
            return

        progress("profile", f"Generating labels for {len(unlabeled)} clusters...")
        labeler = ClusterLabeler(llm_client, model=self.settings.llm.model)

        # Use batch labeling for efficiency
        labels = labeler.label_clusters_batch(unlabeled)
        for cluster, label in zip(unlabeled, labels):
            cluster.label = label

    def _enrich_abstracts(
        self,
        candidates: list[CandidateWork],
        progress: Callable[[str, str], None],
    ) -> tuple[list[CandidateWork], EnrichmentStats]:
        """Enrich missing abstracts via scraper."""
        progress("enrich", "Enriching missing abstracts...")

        llm_for_enrichment = None
        if self.settings.llm.enabled and self.settings.sources.scraper.use_llm_fallback:
            try:
                llm_for_enrichment = self._get_llm_client()
            except Exception as e:
                logger.warning("Failed to create LLM client for enrichment: %s", e)

        enricher = AbstractEnricher(self.settings, self.base_dir, llm=llm_for_enrichment)
        candidates, stats = enricher.enrich(candidates)

        progress(
            "enrich",
            f"Enriched {stats.enriched} abstracts (cache: {stats.cache_hits}, scraper: {stats.scraper_fetched})",
        )

        return candidates, stats

    def _select_interest_papers(
        self,
        candidates: list[CandidateWork],
        embedding_cache: EmbeddingCache,
        progress: Callable[[str, str], None],
    ) -> list[InterestWork]:
        """Select papers based on user interests."""
        progress("interest", "Selecting interest-based papers...")

        try:
            llm_client = self._get_llm_client()
            if not llm_client:
                return []

            refiner = InterestRefiner(llm_client, model=self.settings.llm.model)
            reranker = create_reranker(
                self.settings.scoring.rerank,
                self.settings.embedding,
            )

            # Create cached embedding provider (reuses same cache as ProfileRanker)
            base_vectorizer = create_embedding_provider(self.settings.embedding)
            cached_vectorizer = CachingEmbeddingProvider(
                provider=base_vectorizer,
                cache=embedding_cache,
                source_type="candidate",
                ttl_days=self.settings.embedding.candidate_ttl_days,
            )

            selector = InterestRanker(
                settings=self.settings,
                vectorizer=cached_vectorizer,
                reranker=reranker,
                interest_refiner=refiner,
                base_dir=self.base_dir,
            )
            interest_works = selector.select(candidates)
            progress("interest", f"Selected {len(interest_works)} interest papers")
            return interest_works

        except Exception as e:
            logger.warning("Interest selection failed: %s", e)
            progress("interest", f"Interest selection skipped (error: {e})")
            return []

    def _generate_summaries(
        self,
        result: WatchResult,
        storage: ProfileStorage,
        progress: Callable[[str, str], None],
    ) -> None:
        """Generate AI summaries for ranked works."""
        llm_client = self._get_llm_client()
        if not llm_client:
            return

        # Summarize ranked works
        summary_start = time.time()
        # Estimate ~2s per paper for LLM summarization
        est_time = len(result.ranked_works) * 2
        progress("summary", f"Generating summaries for {len(result.ranked_works)} papers (est. ~{est_time}s)...")
        summarizer = PaperSummarizer(llm_client, storage, model=self.settings.llm.model)
        max_workers = self.settings.watch.summarizer_max_workers
        summary_result = summarizer.summarize_batch(result.ranked_works, max_workers=max_workers)
        result.stats.summaries_generated = summary_result.success_count

        if summary_result.failed_ids:
            logger.warning(
                "Failed to summarize %d ranked papers: %s",
                summary_result.failure_count,
                summary_result.failed_ids[:5],  # Log first 5 failed IDs
            )

        # Attach summaries to works
        summary_map = {s.paper_id: s for s in summary_result.summaries}
        for work in result.ranked_works:
            if work.identifier in summary_map:
                work.summary = summary_map[work.identifier]

        # Summarize interest works
        if result.interest_works:
            progress("summary", f"Generating summaries for {len(result.interest_works)} interest papers...")
            interest_result = summarizer.summarize_batch(result.interest_works, max_workers=max_workers)
            result.stats.summaries_generated += interest_result.success_count

            if interest_result.failed_ids:
                logger.warning(
                    "Failed to summarize %d interest papers: %s",
                    interest_result.failure_count,
                    interest_result.failed_ids[:5],
                )

            interest_map = {s.paper_id: s for s in interest_result.summaries}
            for work in result.interest_works:
                if work.identifier in interest_map:
                    work.summary = interest_map[work.identifier]

        # Summarize followed works
        if result.followed_works:
            progress("summary", f"Generating summaries for {len(result.followed_works)} followed author papers...")
            followed_result = summarizer.summarize_batch(result.followed_works, max_workers=max_workers)
            result.stats.summaries_generated += followed_result.success_count

            followed_map = {s.paper_id: s for s in followed_result.summaries}
            for work in result.followed_works:
                if work.identifier in followed_map:
                    work.summary = followed_map[work.identifier]

        # Generate overall summaries
        progress("summary", "Generating overall summaries...")
        overall_summarizer = OverallSummarizer(llm_client, model=self.settings.llm.model)

        if result.interest_works:
            result.overall_summaries["interest"] = overall_summarizer.summarize_section(
                result.interest_works, "interest"
            )

        if result.ranked_works:
            result.overall_summaries["similarity"] = overall_summarizer.summarize_section(
                result.ranked_works, "similarity"
            )

        summary_elapsed = time.time() - summary_start
        progress("summary", f"Generated {result.stats.summaries_generated} summaries ({summary_elapsed:.1f}s)")

    def _translate_titles(
        self,
        result: WatchResult,
        storage: ProfileStorage,
        progress: Callable[[str, str], None],
    ) -> None:
        """Translate paper titles."""
        llm_client = self._get_llm_client()
        if not llm_client:
            return

        all_works = result.ranked_works + (result.interest_works or []) + (result.followed_works or [])
        if not all_works:
            return

        translate_start = time.time()
        progress("translate", f"Translating {len(all_works)} titles...")
        translator = TitleTranslator(llm_client, storage, model=self.settings.llm.model)
        translations = translator.translate_batch(all_works)

        for work in result.ranked_works:
            if work.identifier in translations:
                work.translated_title = translations[work.identifier]

        for work in result.interest_works or []:
            if work.identifier in translations:
                work.translated_title = translations[work.identifier]

        for work in result.followed_works or []:
            if work.identifier in translations:
                work.translated_title = translations[work.identifier]

        translate_elapsed = time.time() - translate_start
        progress("translate", f"Translated {len(translations)} titles ({translate_elapsed:.1f}s)")

    def _classify_domains(
        self,
        result: WatchResult,
        progress: Callable[[str, str], None],
    ) -> None:
        """Classify papers into research domains using LLM."""
        llm_client = self._get_llm_client()
        if not llm_client:
            return

        all_works = result.ranked_works + (result.interest_works or []) + (result.followed_works or [])
        if not all_works:
            return

        classify_start = time.time()
        progress("classify", f"Classifying {len(all_works)} papers into domains...")

        domain_config = self.settings.llm.domain_classification
        classifier = PaperDomainClassifier(
            llm_client,
            model=self.settings.llm.model,
            domains=domain_config.domains if domain_config.domains else None,
            batch_size=domain_config.batch_size,
            max_workers=domain_config.max_workers,
        )

        # Classify all papers
        classifications = classifier.classify_papers(all_works)

        # Apply classifications to works
        for work in result.ranked_works:
            if work.identifier in classifications:
                work.domain = classifications[work.identifier]

        for work in result.interest_works or []:
            if work.identifier in classifications:
                work.domain = classifications[work.identifier]

        for work in result.followed_works or []:
            if work.identifier in classifications:
                work.domain = classifications[work.identifier]

        classify_elapsed = time.time() - classify_start
        progress("classify", f"Classified {len(classifications)} papers ({classify_elapsed:.1f}s)")

    def _fetch_followed_authors(
        self,
        storage: ProfileStorage,
        progress: Callable[[str, str], None],
    ) -> list[RankedWork]:
        """Fetch papers by followed authors from OpenAlex.

        Papers skip scoring/ranking/TopK. Only deduplication is applied.
        First run does a full pull; subsequent runs use incremental fetching.
        """
        from zotwatch.infrastructure.storage import ArchiveStorage
        from zotwatch.sources.openalex import OpenAlexAuthorFetcher

        followed_cfg = self.settings.sources.followed_authors
        progress("followed", f"Fetching papers for {len(followed_cfg.authors)} followed authors...")

        fetch_start = time.time()

        # Initialize archive for state tracking
        archive_db = self.base_dir / "data" / "archive.sqlite"
        archive = ArchiveStorage(archive_db)
        archive.initialize()

        try:
            # Get last fetch dates for incremental fetching
            last_dates = archive.get_all_followed_author_states()

            # Get known identifiers for deduplication
            known_identifiers = archive.get_known_identifiers()

            # Also get identifiers from the user's Zotero library for dedup
            library_dois: set[str] = set()
            library_titles: set[str] = set()
            for item in storage.iter_items():
                if item.doi:
                    library_dois.add(item.doi.lower().strip())
                library_titles.add(item.title.lower().strip())

            # Fetch from OpenAlex
            fetcher = OpenAlexAuthorFetcher(
                polite_email=followed_cfg.polite_email,
            )

            author_dicts = [
                {"id": a.id, "name": a.name} for a in followed_cfg.authors
            ]
            candidates = fetcher.fetch_all_authors(
                author_dicts,
                last_dates=last_dates,
                max_results_per_author=followed_cfg.max_results_per_author,
            )

            progress("followed", f"Fetched {len(candidates)} papers from OpenAlex")

            # Deduplicate against archive and library
            new_works: list[RankedWork] = []
            seen: set[str] = set()
            for candidate in candidates:
                # Skip if already in archive
                if candidate.identifier in known_identifiers:
                    continue
                # Skip if already seen in this batch
                if candidate.identifier in seen:
                    continue
                # Skip if DOI already in library
                if candidate.doi and candidate.doi.lower().strip() in library_dois:
                    continue
                # Skip if title already in library (exact match)
                if candidate.title.lower().strip() in library_titles:
                    continue

                seen.add(candidate.identifier)

                # Convert to RankedWork with label="followed", no scoring
                followed_author = candidate.extra.get("followed_author", "")
                work = RankedWork(
                    source=candidate.source,
                    identifier=candidate.identifier,
                    title=candidate.title,
                    abstract=candidate.abstract,
                    authors=candidate.authors,
                    doi=candidate.doi,
                    url=candidate.url,
                    published=candidate.published,
                    venue=candidate.venue,
                    metrics=candidate.metrics,
                    extra={
                        **candidate.extra,
                        "followed_author": followed_author,
                    },
                    score=0.0,
                    similarity=0.0,
                    label="followed",
                )
                new_works.append(work)

            # Update state for each author
            today = time.strftime("%Y-%m-%d")
            for author in followed_cfg.authors:
                author_works = [
                    w for w in new_works
                    if w.extra.get("followed_author") == author.name
                ]
                existing_state = archive.get_followed_author_state(author.id)
                prev_total = existing_state["total_works"] if existing_state else 0
                archive.update_followed_author_state(
                    author_id=author.id,
                    author_name=author.name,
                    last_fetch_date=today,
                    total_works=prev_total + len(author_works),
                )

            fetch_elapsed = time.time() - fetch_start
            progress(
                "followed",
                f"Found {len(new_works)} new papers from followed authors ({fetch_elapsed:.1f}s)",
            )

            return new_works

        finally:
            archive.close()

    def _cleanup_caches(
        self,
        embedding_cache: EmbeddingCache,
        progress: Callable[[str, str], None],
    ) -> None:
        """Cleanup expired cache entries."""
        removed = embedding_cache.cleanup_expired()
        if removed > 0:
            progress("cleanup", f"Cleaned up {removed} expired embedding cache entries")

        metadata_cache = MetadataCache(self.base_dir / "data" / "metadata.sqlite")
        removed_meta = metadata_cache.cleanup_expired()
        if removed_meta > 0:
            progress("cleanup", f"Cleaned up {removed_meta} expired metadata cache entries")
        metadata_cache.close()


__all__ = ["WatchPipeline", "WatchConfig", "WatchResult", "WatchStats", "ComputedThresholds"]
