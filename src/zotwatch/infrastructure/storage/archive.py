"""Archive storage for ranked works.

Stores historical ranked works for archive page generation.
"""

import json
import sqlite3
from datetime import date, datetime
from pathlib import Path

from zotwatch.core.models import PaperSummary, RankedWork

ARCHIVE_SCHEMA = """
CREATE TABLE IF NOT EXISTS archive (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    identifier TEXT NOT NULL,
    run_date DATE NOT NULL,
    source TEXT NOT NULL,
    title TEXT NOT NULL,
    abstract TEXT,
    authors_json TEXT,
    doi TEXT,
    url TEXT,
    published DATE,
    venue TEXT,
    score REAL NOT NULL,
    similarity REAL NOT NULL,
    impact_factor_score REAL DEFAULT 0.0,
    impact_factor REAL,
    is_chinese_core INTEGER DEFAULT 0,
    label TEXT NOT NULL,
    translated_title TEXT,
    summary_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(identifier, run_date)
);

CREATE INDEX IF NOT EXISTS idx_archive_run_date ON archive(run_date);
CREATE INDEX IF NOT EXISTS idx_archive_source ON archive(source);
CREATE INDEX IF NOT EXISTS idx_archive_venue ON archive(venue);
CREATE INDEX IF NOT EXISTS idx_archive_label ON archive(label);
CREATE INDEX IF NOT EXISTS idx_archive_identifier ON archive(identifier);
"""


class ArchiveStorage:
    """SQLite storage for archived ranked works."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._conn: sqlite3.Connection | None = None

    def connect(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._conn is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.path))
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def initialize(self) -> None:
        """Initialize database schema."""
        conn = self.connect()
        conn.executescript(ARCHIVE_SCHEMA)
        conn.commit()

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def save_batch(
        self,
        works: list[RankedWork],
        run_date: date | None = None,
    ) -> int:
        """Save a batch of ranked works to archive.

        Args:
            works: List of RankedWork to save.
            run_date: Date of the run (defaults to today).

        Returns:
            Number of works saved (new or updated).
        """
        if not works:
            return 0

        run_date = run_date or date.today()
        conn = self.connect()

        rows = []
        for work in works:
            summary_json = None
            if work.summary:
                summary_json = work.summary.model_dump_json()

            rows.append((
                work.identifier,
                run_date.isoformat(),
                work.source,
                work.title,
                work.abstract,
                json.dumps(work.authors, ensure_ascii=False),
                work.doi,
                work.url,
                work.published.isoformat() if work.published else None,
                work.venue,
                work.score,
                work.similarity,
                work.impact_factor_score,
                work.impact_factor,
                1 if work.is_chinese_core else 0,
                work.label,
                work.translated_title,
                summary_json,
            ))

        conn.executemany(
            """
            INSERT OR REPLACE INTO archive (
                identifier, run_date, source, title, abstract,
                authors_json, doi, url, published, venue,
                score, similarity, impact_factor_score, impact_factor,
                is_chinese_core, label, translated_title, summary_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
        return len(rows)

    def get_all(
        self,
        days: int = 90,
        sources: list[str] | None = None,
        labels: list[str] | None = None,
    ) -> list[RankedWork]:
        """Get all archived works from the specified period.

        Args:
            days: Number of days to include.
            sources: Filter by sources (arXiv, Crossref, etc.).
            labels: Filter by labels (must_read, consider, etc.).

        Returns:
            List of RankedWork ordered by run_date desc, score desc.
        """
        conn = self.connect()
        query = """
            SELECT * FROM archive
            WHERE run_date >= date('now', ?)
        """
        params: list = [f"-{days} days"]

        if sources:
            placeholders = ",".join("?" * len(sources))
            query += f" AND source IN ({placeholders})"
            params.extend(sources)

        if labels:
            placeholders = ",".join("?" * len(labels))
            query += f" AND label IN ({placeholders})"
            params.extend(labels)

        query += " ORDER BY run_date DESC, score DESC"

        cursor = conn.execute(query, params)
        return [self._row_to_work(row) for row in cursor.fetchall()]

    def get_by_date(self, run_date: date) -> list[RankedWork]:
        """Get works from a specific run date."""
        conn = self.connect()
        cursor = conn.execute(
            "SELECT * FROM archive WHERE run_date = ? ORDER BY score DESC",
            (run_date.isoformat(),),
        )
        return [self._row_to_work(row) for row in cursor.fetchall()]

    def get_grouped_by_date(
        self,
        days: int = 90,
        sources: list[str] | None = None,
        labels: list[str] | None = None,
    ) -> dict[str, list[RankedWork]]:
        """Get works grouped by run date."""
        works = self.get_all(days=days, sources=sources, labels=labels)
        grouped: dict[str, list[RankedWork]] = {}
        for work in works:
            # Use run_date from DB, fall back to published date
            run_date = work.extra.get("run_date", "Unknown")
            if run_date not in grouped:
                grouped[run_date] = []
            grouped[run_date].append(work)
        return grouped

    def get_grouped_by_venue(
        self,
        days: int = 90,
        sources: list[str] | None = None,
        labels: list[str] | None = None,
    ) -> dict[str, list[RankedWork]]:
        """Get works grouped by venue."""
        works = self.get_all(days=days, sources=sources, labels=labels)
        grouped: dict[str, list[RankedWork]] = {}
        for work in works:
            venue = work.venue or "Unknown"
            if venue not in grouped:
                grouped[venue] = []
            grouped[venue].append(work)
        return grouped

    def get_grouped_by_source(
        self,
        days: int = 90,
        labels: list[str] | None = None,
    ) -> dict[str, list[RankedWork]]:
        """Get works grouped by source."""
        works = self.get_all(days=days, labels=labels)
        grouped: dict[str, list[RankedWork]] = {}
        for work in works:
            source = work.source
            if source not in grouped:
                grouped[source] = []
            grouped[source].append(work)
        return grouped

    def get_grouped_by_label(
        self,
        days: int = 90,
        sources: list[str] | None = None,
    ) -> dict[str, list[RankedWork]]:
        """Get works grouped by label."""
        works = self.get_all(days=days, sources=sources)
        grouped: dict[str, list[RankedWork]] = {}
        for work in works:
            label = work.label
            if label not in grouped:
                grouped[label] = []
            grouped[label].append(work)
        return grouped

    def get_stats(self, days: int = 90) -> dict:
        """Get archive statistics."""
        conn = self.connect()
        cursor = conn.execute(
            """
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN label = 'must_read' THEN 1 ELSE 0 END) as must_read,
                SUM(CASE WHEN label = 'consider' THEN 1 ELSE 0 END) as consider,
                COUNT(DISTINCT source) as source_count,
                COUNT(DISTINCT venue) as venue_count,
                MIN(run_date) as earliest,
                MAX(run_date) as latest
            FROM archive
            WHERE run_date >= date('now', ?)
            """,
            (f"-{days} days",),
        )
        row = cursor.fetchone()
        return {
            "total": row["total"] or 0,
            "must_read": row["must_read"] or 0,
            "consider": row["consider"] or 0,
            "source_count": row["source_count"] or 0,
            "venue_count": row["venue_count"] or 0,
            "earliest": row["earliest"],
            "latest": row["latest"],
        }

    def get_sources(self, days: int = 90) -> list[dict]:
        """Get source distribution."""
        conn = self.connect()
        cursor = conn.execute(
            """
            SELECT source, COUNT(*) as count
            FROM archive
            WHERE run_date >= date('now', ?)
            GROUP BY source
            ORDER BY count DESC
            """,
            (f"-{days} days",),
        )
        return [{"source": row["source"], "count": row["count"]} for row in cursor.fetchall()]

    def _row_to_work(self, row: sqlite3.Row) -> RankedWork:
        """Convert database row to RankedWork."""
        authors = json.loads(row["authors_json"]) if row["authors_json"] else []

        published = None
        if row["published"]:
            try:
                published = datetime.fromisoformat(row["published"])
            except ValueError:
                pass

        summary = None
        if row["summary_json"]:
            try:
                summary = PaperSummary.model_validate_json(row["summary_json"])
            except Exception:
                pass

        return RankedWork(
            source=row["source"],
            identifier=row["identifier"],
            title=row["title"],
            abstract=row["abstract"],
            authors=authors,
            doi=row["doi"],
            url=row["url"],
            published=published,
            venue=row["venue"],
            score=row["score"],
            similarity=row["similarity"],
            impact_factor_score=row["impact_factor_score"] or 0.0,
            impact_factor=row["impact_factor"],
            is_chinese_core=bool(row["is_chinese_core"]),
            label=row["label"],
            translated_title=row["translated_title"],
            summary=summary,
            extra={"run_date": row["run_date"]},
        )


__all__ = ["ArchiveStorage"]
