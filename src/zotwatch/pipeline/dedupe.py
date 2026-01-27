"""Deduplication pipeline."""

import logging
import re
from collections.abc import Iterable

from rapidfuzz import fuzz

from zotwatch.core.models import CandidateWork
from zotwatch.infrastructure.storage import ProfileStorage

logger = logging.getLogger(__name__)

# Length tolerance for fuzzy matching pre-filter (fraction of title length)
LENGTH_TOLERANCE = 0.3


class DedupeEngine:
    """Deduplication engine for candidate works.

    Optimizations:
    - Uses set for exact title matching (O(1) lookup)
    - Pre-filters by title length before fuzzy matching
    - Short-circuits on exact matches
    """

    def __init__(self, storage: ProfileStorage, title_threshold: float = 0.9):
        self.storage = storage
        self.title_threshold = title_threshold
        self.existing_doi: set[str] = set()
        self.existing_ids: set[str] = set()
        self.existing_titles: list[str] = []
        # Optimization: exact title set for O(1) lookup
        self.existing_titles_set: set[str] = set()
        self._load_existing()

    def _load_existing(self) -> None:
        """Load existing items for deduplication."""
        for item in self.storage.iter_items():
            if item.doi:
                self.existing_doi.add(_normalize_identifier(item.doi))
            if item.url:
                self.existing_ids.add(_normalize_identifier(item.url))
            normalized = _normalize_title(item.title)
            self.existing_titles.append(normalized)
            # Add to set for fast exact matching
            self.existing_titles_set.add(normalized)

    def filter(self, candidates: Iterable[CandidateWork]) -> list[CandidateWork]:
        """Filter out duplicate candidates."""
        source = list(candidates)
        deduped: list[CandidateWork] = []
        candidate_titles: list[str] = []
        candidate_titles_set: set[str] = set()  # Fast lookup for batch dedup
        seen_keys: set[str] = set()

        for work in source:
            key = _normalize_identifier(work.identifier)
            doi = _normalize_identifier(work.doi) if work.doi else None
            title = _normalize_title(work.title)

            # Check DOI duplication
            if doi and doi in self.existing_doi:
                logger.debug("Skipping %s due to DOI duplication", work.identifier)
                continue
            if doi and doi in seen_keys:
                continue

            # Check identifier duplication
            if key in self.existing_ids or key in seen_keys:
                logger.debug("Skipping %s due to identifier duplication", work.identifier)
                continue

            # Check title similarity (optimized)
            if self._is_title_duplicate(title) or _is_title_in_list_optimized(
                title, candidate_titles, candidate_titles_set, self.title_threshold
            ):
                logger.debug("Skipping %s due to title similarity", work.identifier)
                continue

            deduped.append(work)
            candidate_titles.append(title)
            candidate_titles_set.add(title)
            seen_keys.add(key)
            if doi:
                seen_keys.add(doi)

        logger.info("Deduped candidates from %d to %d", len(source), len(deduped))
        return deduped

    def _is_title_duplicate(self, title: str) -> bool:
        """Check if title matches existing titles (optimized).

        1. First check exact match in set (O(1))
        2. Then do fuzzy matching with length pre-filter
        """
        # Fast path: exact match
        if title in self.existing_titles_set:
            return True

        # Slow path: fuzzy matching with length pre-filter
        return _is_title_in_list_with_length_filter(
            title, self.existing_titles, self.title_threshold
        )


def _normalize_identifier(value: str) -> str:
    """Normalize identifier for comparison."""
    return (value or "").lower().strip()


def _normalize_title(title: str) -> str:
    """Normalize title for comparison."""
    normalized = re.sub(r"\s+", " ", title or "").strip().lower()
    return normalized


def _is_title_in_list_optimized(
    title: str,
    title_list: list[str],
    title_set: set[str],
    threshold: float,
) -> bool:
    """Check if title matches any in list (optimized).

    1. First check exact match in set (O(1))
    2. Then do fuzzy matching with length pre-filter
    """
    # Fast path: exact match
    if title in title_set:
        return True

    # Slow path: fuzzy matching with length pre-filter
    return _is_title_in_list_with_length_filter(title, title_list, threshold)


def _is_title_in_list_with_length_filter(
    title: str,
    title_list: list[str],
    threshold: float,
) -> bool:
    """Check if title matches any in list using fuzzy matching with length pre-filter.

    Only compares titles whose lengths are within tolerance to reduce computation.
    """
    if not title:
        return False

    title_len = len(title)
    min_len = int(title_len * (1 - LENGTH_TOLERANCE))
    max_len = int(title_len * (1 + LENGTH_TOLERANCE))

    for existing in title_list:
        if not existing:
            continue
        # Length pre-filter: skip titles with very different lengths
        existing_len = len(existing)
        if existing_len < min_len or existing_len > max_len:
            continue
        # Fuzzy matching
        score = fuzz.token_set_ratio(title, existing) / 100.0
        if score >= threshold:
            return True
    return False


def _is_title_in_list(title: str, title_list: Iterable[str], threshold: float) -> bool:
    """Check if title matches any in list using fuzzy matching.

    Legacy function kept for backward compatibility.
    """
    for existing in title_list:
        if not existing:
            continue
        score = fuzz.token_set_ratio(title, existing) / 100.0
        if score >= threshold:
            return True
    return False


__all__ = ["DedupeEngine"]
