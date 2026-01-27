"""Journal impact factor scoring utilities."""

import csv
import logging
import math
from functools import lru_cache
from pathlib import Path

from zotwatch.config.settings import ScoringConfig
from zotwatch.core.models import CandidateWork

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_journal_whitelist_cached(path_str: str) -> dict[str, dict]:
    """Module-level cached journal whitelist loader.

    Uses lru_cache to avoid re-reading CSV file on each JournalScorer instantiation.
    The path is converted to string for hashability.

    Args:
        path_str: String path to journal_whitelist.csv.

    Returns:
        Dict mapping ISSN to journal info dict.
    """
    path = Path(path_str)
    whitelist: dict[str, dict] = {}

    if not path.exists():
        logger.warning("Journal whitelist not found: %s", path)
        return whitelist

    try:
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                issn = (row.get("issn") or "").strip()
                # Skip empty rows and comment rows (ISSN starts with #)
                if not issn or issn.startswith("#"):
                    continue
                category = row.get("category") or ""
                if_str = (row.get("impact_factor") or "").strip()
                whitelist[issn] = {
                    "title": row.get("title") or "",
                    "category": category,
                    "impact_factor": None if if_str in ("NA", "") else float(if_str),
                    "is_cn": "(CN)" in category,
                }
        logger.info("Loaded %d journals from whitelist (cached)", len(whitelist))
    except Exception as exc:
        logger.warning("Failed to load journal whitelist: %s", exc)

    return whitelist


class JournalScorer:
    """Computes journal impact factor scores for candidate works."""

    def __init__(
        self,
        base_dir: Path | str,
        config: ScoringConfig.JournalScoringConfig,
    ):
        """Initialize journal scorer.

        Args:
            base_dir: Base directory containing data/journal_whitelist.csv.
            config: Journal scoring configuration.
        """
        self.base_dir = Path(base_dir)
        self.config = config
        # Use module-level cached loader
        whitelist_path = self.base_dir / "data" / "journal_whitelist.csv"
        self._whitelist = _load_journal_whitelist_cached(str(whitelist_path))

    def compute_score(self, candidate: CandidateWork) -> tuple[float, float | None, bool]:
        """Compute IF score for a candidate.

        Args:
            candidate: The candidate work to score

        Returns:
            Tuple of (normalized_if_score, raw_impact_factor, is_chinese_core)
        """
        # arXiv papers get mid-range score
        if candidate.source == "arxiv":
            return (self.config.arxiv_score, None, False)

        # Try to find journal in whitelist by any of its ISSNs
        issns = candidate.extra.get("issns") or []
        for issn in issns:
            if issn and issn in self._whitelist:
                entry = self._whitelist[issn]
                if entry["is_cn"]:
                    return (self.config.chinese_core_score, None, True)
                if entry["impact_factor"] is not None:
                    raw_if = entry["impact_factor"]
                    normalized = math.log(raw_if + 1) / math.log(self.config.log_base)
                    return (min(normalized, 1.0), raw_if, False)

        # Unknown journal
        return (self.config.unknown_score, None, False)


__all__ = ["JournalScorer"]
