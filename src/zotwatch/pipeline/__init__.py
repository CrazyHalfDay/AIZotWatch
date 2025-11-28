"""Processing pipeline components."""

from .dedupe import DedupeEngine
from .enrich import AbstractEnricher, EnrichmentStats, enrich_candidates
from .featured import FeaturedSelector
from .fetch import fetch_candidates
from .ingest import ingest_zotero
from .profile import ProfileBuilder
from .profile_stats import ProfileStatsExtractor
from .score import WorkRanker

__all__ = [
    "ingest_zotero",
    "ProfileBuilder",
    "ProfileStatsExtractor",
    "fetch_candidates",
    "AbstractEnricher",
    "EnrichmentStats",
    "enrich_candidates",
    "DedupeEngine",
    "WorkRanker",
    "FeaturedSelector",
]
