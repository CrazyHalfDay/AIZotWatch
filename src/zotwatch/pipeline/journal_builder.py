"""Build the journal whitelist CSV from library venues.

The LLM proposes journal titles, categories and impact factors; Crossref is
then queried to resolve authoritative ISSNs for each title. Journals that
cannot be verified on Crossref are skipped so the whitelist only ever
contains real ISSNs.
"""

import csv
import logging
import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import requests

from zotwatch.core.constants import DEFAULT_HTTP_TIMEOUT
from zotwatch.llm.journal_recommender import GeneratedJournal

logger = logging.getLogger(__name__)

CROSSREF_JOURNALS_URL = "https://api.crossref.org/journals"

# Minimum token-overlap ratio to accept a Crossref title as a match.
_TITLE_MATCH_THRESHOLD = 0.6


@dataclass
class JournalEntry:
    """A resolved whitelist row (one per ISSN)."""

    issn: str
    title: str
    category: str
    impact_factor: float | None


@dataclass
class BuildResult:
    """Outcome of building the whitelist."""

    entries: list[JournalEntry] = field(default_factory=list)
    verified: int = 0
    skipped: list[str] = field(default_factory=list)
    kept_existing: int = 0
    written: bool = False
    output_path: Path | None = None
    backup_path: Path | None = None


def _normalize_title(title: str) -> set[str]:
    """Tokenize a title into a normalized set of lowercase words."""
    cleaned = re.sub(r"[^a-z0-9\s]", " ", title.lower())
    return {tok for tok in cleaned.split() if tok}


def _title_match(query: str, candidate: str) -> bool:
    """Return True if two journal titles refer to the same journal."""
    q = _normalize_title(query)
    c = _normalize_title(candidate)
    if not q or not c:
        return False
    if q == c:
        return True
    # Token Jaccard handles minor wording/punctuation differences. Bare subset
    # matching is intentionally avoided: a one-word query like "Science" must
    # not match any longer title that merely contains that word.
    overlap = len(q & c) / len(q | c)
    return overlap >= _TITLE_MATCH_THRESHOLD


class CrossrefJournalVerifier:
    """Resolves authoritative ISSNs for journal titles via Crossref."""

    def __init__(self, mailto: str = "", timeout: float = DEFAULT_HTTP_TIMEOUT):
        self.session = requests.Session()
        self.mailto = mailto
        self.timeout = timeout

    def verify(self, title: str) -> tuple[list[str], str] | None:
        """Look up a journal title on Crossref.

        Args:
            title: Journal title to verify.

        Returns:
            Tuple of (issns, official_title) if found, else None.
        """
        # Strip punctuation from the query: characters like "&" otherwise make
        # Crossref's search return nothing (e.g. "... Earth & Environment").
        query = re.sub(r"[^\w\s]", " ", title).strip() or title
        params: dict[str, str | int] = {"query": query, "rows": 20}
        if self.mailto:
            params["mailto"] = self.mailto

        try:
            resp = self.session.get(CROSSREF_JOURNALS_URL, params=params, timeout=self.timeout)
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("Crossref lookup failed for '%s': %s", title, type(exc).__name__)
            return None

        items = resp.json().get("message", {}).get("items", [])
        target = _normalize_title(title)
        loose: tuple[list[str], str] | None = None
        for item in items:
            cr_title = item.get("title") or ""
            issns = [s.strip() for s in (item.get("ISSN") or []) if s and s.strip()]
            if not issns:
                continue
            # Prefer an exact normalized-title match: recovers flagship journals
            # (e.g. "Nature") that rank below generic look-alikes, and avoids
            # loose subset matches grabbing the wrong journal.
            if _normalize_title(cr_title) == target:
                return issns, cr_title
            if loose is None and _title_match(title, cr_title):
                loose = (issns, cr_title)
        return loose


class JournalWhitelistBuilder:
    """Builds and writes the journal whitelist CSV."""

    FIELDNAMES = ["issn", "title", "category", "impact_factor"]

    def __init__(
        self,
        csv_path: Path | str,
        verifier: CrossrefJournalVerifier | None = None,
    ):
        self.csv_path = Path(csv_path)
        self.verifier = verifier or CrossrefJournalVerifier()

    def build(
        self,
        generated: list[GeneratedJournal],
        *,
        merge: bool = True,
        dry_run: bool = False,
        on_progress: Callable[[str], None] | None = None,
    ) -> BuildResult:
        """Verify generated journals and write the whitelist CSV.

        Args:
            generated: Journals proposed by the LLM.
            merge: Preserve existing whitelist entries (by ISSN).
            dry_run: Compute the result without writing the file.
            on_progress: Optional progress callback.

        Returns:
            BuildResult describing the outcome.
        """
        result = BuildResult(output_path=self.csv_path)

        # Start from existing entries when merging.
        entries: dict[str, JournalEntry] = {}
        if merge:
            existing = self._read_existing()
            entries.update(existing)
            result.kept_existing = len(existing)

        for journal in generated:
            verified = self.verifier.verify(journal.title)
            if verified is None:
                result.skipped.append(journal.title)
                if on_progress:
                    on_progress(f"skip (not on Crossref): {journal.title}")
                continue

            issns, official_title = verified
            category = self._format_category(journal.category, journal.is_chinese)
            result.verified += 1
            if on_progress:
                on_progress(f"verified: {official_title} -> {', '.join(issns)}")

            # Add one row per ISSN to maximize candidate matching.
            for issn in issns:
                if merge and issn in entries:
                    # Keep manually curated entries untouched.
                    continue
                entries[issn] = JournalEntry(
                    issn=issn,
                    title=official_title or journal.title,
                    category=category,
                    impact_factor=journal.impact_factor,
                )

        result.entries = self._sorted_entries(entries)

        if not dry_run:
            result.backup_path = self._write(result.entries)
            result.written = True

        return result

    def _read_existing(self) -> dict[str, JournalEntry]:
        """Read existing whitelist rows keyed by ISSN."""
        entries: dict[str, JournalEntry] = {}
        if not self.csv_path.exists():
            return entries

        try:
            with self.csv_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    issn = (row.get("issn") or "").strip()
                    if not issn or issn.startswith("#"):
                        continue
                    if_str = (row.get("impact_factor") or "").strip()
                    entries[issn] = JournalEntry(
                        issn=issn,
                        title=(row.get("title") or "").strip(),
                        category=(row.get("category") or "").strip(),
                        impact_factor=None if if_str in ("NA", "") else float(if_str),
                    )
        except (OSError, csv.Error, ValueError) as exc:
            logger.warning("Failed to read existing whitelist: %s", exc)

        return entries

    @staticmethod
    def _format_category(category: str, is_chinese: bool) -> str:
        """Append the (CN) marker for Chinese core journals."""
        category = category.strip() or "GENERAL"
        if is_chinese and "(CN)" not in category:
            category = f"{category} (CN)"
        return category

    @staticmethod
    def _sorted_entries(entries: dict[str, JournalEntry]) -> list[JournalEntry]:
        """Sort by impact factor (desc, None last) then title."""
        return sorted(
            entries.values(),
            key=lambda e: (-(e.impact_factor or -1.0), e.title.lower()),
        )

    def _write(self, entries: list[JournalEntry]) -> Path | None:
        """Write entries to the CSV, backing up any existing file."""
        backup_path: Path | None = None
        if self.csv_path.exists():
            backup_path = self.csv_path.with_suffix(self.csv_path.suffix + ".bak")
            shutil.copy2(self.csv_path, backup_path)

        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with self.csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.FIELDNAMES)
            for entry in entries:
                if_value = "" if entry.impact_factor is None else f"{entry.impact_factor:g}"
                writer.writerow([entry.issn, entry.title, entry.category, if_value])

        return backup_path


__all__ = [
    "JournalWhitelistBuilder",
    "CrossrefJournalVerifier",
    "JournalEntry",
    "BuildResult",
]
