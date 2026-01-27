"""Candidate filtering functions.

Extracted from cli/main.py to enable reuse and testing.
"""

import logging
import re
from datetime import timedelta
from functools import lru_cache

from zotwatch.core.models import CandidateWork, RankedWork
from zotwatch.utils.datetime import utc_today_start

logger = logging.getLogger(__name__)

# Preprint sources for ratio limiting
PREPRINT_SOURCES = frozenset({"arxiv", "biorxiv", "medrxiv", "eartharxiv"})


def filter_recent(ranked: list[RankedWork], *, days: int = 7) -> list[RankedWork]:
    """Filter to papers published within recent days.

    Args:
        ranked: List of ranked works to filter.
        days: Number of days to look back. If <= 0, no filtering is applied.

    Returns:
        Filtered list containing only papers published within the specified days.
    """
    if days <= 0:
        return ranked

    cutoff = utc_today_start() - timedelta(days=days)
    kept = [work for work in ranked if work.published and work.published >= cutoff]
    removed = len(ranked) - len(kept)

    if removed > 0:
        logger.info("Dropped %d items older than %d days", removed, days)

    return kept


def limit_preprints(ranked: list[RankedWork], *, max_ratio: float = 0.9) -> list[RankedWork]:
    """Limit preprints to a maximum ratio of total results.

    Prevents arXiv/bioRxiv/medRxiv from dominating recommendations.

    Args:
        ranked: List of ranked works to filter (should be sorted by score).
        max_ratio: Maximum ratio of preprints allowed (0.0 to 1.0).

    Returns:
        Filtered list respecting the preprint ratio limit.
    """
    if not ranked or max_ratio <= 0:
        return ranked

    filtered: list[RankedWork] = []
    preprint_count = 0

    for work in ranked:
        source = work.source.lower()
        proposed_total = len(filtered) + 1

        if source in PREPRINT_SOURCES:
            proposed_preprints = preprint_count + 1
            if (proposed_preprints / proposed_total) > max_ratio:
                continue
            preprint_count = proposed_preprints

        filtered.append(work)

    removed = len(ranked) - len(filtered)
    if removed > 0:
        logger.info(
            "Preprint cap removed %d items to respect %.0f%% limit",
            removed,
            max_ratio * 100,
        )

    return filtered


def filter_without_abstract(
    candidates: list[CandidateWork],
) -> tuple[list[CandidateWork], int]:
    """Remove candidates without abstracts.

    Abstracts are required for accurate similarity scoring.

    Args:
        candidates: List of candidate works to filter.

    Returns:
        Tuple of (filtered candidates, number removed).
    """
    filtered = [c for c in candidates if c.abstract]
    removed = len(candidates) - len(filtered)

    if removed > 0:
        logger.info("Removed %d candidates without abstracts", removed)

    return filtered, removed


@lru_cache(maxsize=128)
def _compile_keyword_pattern(keywords_tuple: tuple[str, ...]) -> re.Pattern:
    """Compile keywords into a single regex pattern for efficient matching.

    Uses word boundary matching to avoid partial matches.
    Cached for reuse across calls with same keywords.
    """
    # Escape special regex characters and join with alternation
    escaped = [re.escape(kw.lower()) for kw in keywords_tuple]
    # Use word boundaries for more accurate matching
    pattern = r"\b(" + "|".join(escaped) + r")\b"
    return re.compile(pattern, re.IGNORECASE)


def exclude_by_keywords(
    candidates: list[CandidateWork],
    exclude_keywords: list[str],
) -> tuple[list[CandidateWork], int]:
    """Remove candidates matching any exclude keyword.

    Optimized implementation using:
    1. Pre-compiled regex pattern for batch matching
    2. Word boundary matching to avoid false positives
    3. Caching of compiled patterns

    Args:
        candidates: List of candidate works to filter.
        exclude_keywords: List of keywords to exclude.

    Returns:
        Tuple of (filtered candidates, number removed).
    """
    if not exclude_keywords:
        return candidates, 0

    # Convert to tuple for hashability (enables lru_cache)
    keywords_tuple = tuple(exclude_keywords)

    # Use simple substring matching for short keyword lists (faster than regex)
    # Use regex for longer keyword lists (better scaling)
    if len(exclude_keywords) <= 10:
        # Simple substring matching for small keyword sets
        exclude_lower = frozenset(kw.lower() for kw in exclude_keywords)
        filtered = []
        for c in candidates:
            text = f"{c.title} {c.abstract or ''}".lower()
            if not any(kw in text for kw in exclude_lower):
                filtered.append(c)
    else:
        # Compiled regex for larger keyword sets
        pattern = _compile_keyword_pattern(keywords_tuple)
        filtered = []
        for c in candidates:
            text = f"{c.title} {c.abstract or ''}"
            if not pattern.search(text):
                filtered.append(c)

    removed = len(candidates) - len(filtered)

    if removed > 0:
        logger.info(
            "Excluded %d candidates matching keywords (from %d keywords)",
            removed,
            len(exclude_keywords),
        )

    return filtered, removed


__all__ = [
    "filter_recent",
    "limit_preprints",
    "filter_without_abstract",
    "exclude_by_keywords",
    "PREPRINT_SOURCES",
]
