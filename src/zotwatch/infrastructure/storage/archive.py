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
    domain TEXT,
    followed_author TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(identifier, run_date)
);

CREATE INDEX IF NOT EXISTS idx_archive_run_date ON archive(run_date);
CREATE INDEX IF NOT EXISTS idx_archive_source ON archive(source);
CREATE INDEX IF NOT EXISTS idx_archive_venue ON archive(venue);
CREATE INDEX IF NOT EXISTS idx_archive_label ON archive(label);
CREATE INDEX IF NOT EXISTS idx_archive_identifier ON archive(identifier);

CREATE TABLE IF NOT EXISTS followed_author_state (
    author_id TEXT PRIMARY KEY,
    author_name TEXT NOT NULL,
    last_fetch_date TEXT,
    total_works INTEGER DEFAULT 0,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
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
        """Initialize database schema and apply migrations."""
        conn = self.connect()
        # Create table and base indexes first
        conn.executescript(ARCHIVE_SCHEMA)
        conn.commit()
        # Apply migrations (may add columns to existing tables)
        self._migrate(conn)
        # Create indexes that depend on migrated columns
        conn.execute("CREATE INDEX IF NOT EXISTS idx_archive_domain ON archive(domain)")
        conn.commit()

    def _migrate(self, conn: sqlite3.Connection) -> None:
        """Apply schema migrations for existing databases."""
        cursor = conn.execute("PRAGMA table_info(archive)")
        columns = {row["name"] for row in cursor.fetchall()}

        # V2: Add domain column
        if "domain" not in columns:
            conn.execute("ALTER TABLE archive ADD COLUMN domain TEXT")
            conn.commit()

        # V3: Add followed_author column
        if "followed_author" not in columns:
            conn.execute("ALTER TABLE archive ADD COLUMN followed_author TEXT")
            conn.commit()

        # V4: followed_author_state table is created in ARCHIVE_SCHEMA via executescript

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

            # Extract followed_author from extra dict
            followed_author = None
            if isinstance(work.extra, dict):
                followed_author = work.extra.get("followed_author")

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
                work.domain,
                followed_author,
            ))

        conn.executemany(
            """
            INSERT OR REPLACE INTO archive (
                identifier, run_date, source, title, abstract,
                authors_json, doi, url, published, venue,
                score, similarity, impact_factor_score, impact_factor,
                is_chinese_core, label, translated_title, summary_json,
                domain, followed_author
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        dedupe: bool = True,
    ) -> list[RankedWork]:
        """Get all archived works from the specified period.

        Args:
            days: Number of days to include.
            sources: Filter by sources (arXiv, Crossref, etc.).
            labels: Filter by labels (must_read, consider, etc.).
            dedupe: If True, only return the latest record for each paper.

        Returns:
            List of RankedWork ordered by run_date desc, score desc.
        """
        conn = self.connect()

        if dedupe:
            # Use subquery to get only the latest record for each identifier
            query = """
                SELECT a.* FROM archive a
                INNER JOIN (
                    SELECT identifier, MAX(run_date) as max_date
                    FROM archive
                    WHERE run_date >= date('now', ?)
                    GROUP BY identifier
                ) latest ON a.identifier = latest.identifier AND a.run_date = latest.max_date
                WHERE a.run_date >= date('now', ?)
            """
            params: list = [f"-{days} days", f"-{days} days"]
        else:
            query = """
                SELECT * FROM archive
                WHERE run_date >= date('now', ?)
            """
            params = [f"-{days} days"]

        if sources:
            placeholders = ",".join("?" * len(sources))
            query += f" AND a.source IN ({placeholders})" if dedupe else f" AND source IN ({placeholders})"
            params.extend(sources)

        if labels:
            placeholders = ",".join("?" * len(labels))
            query += f" AND a.label IN ({placeholders})" if dedupe else f" AND label IN ({placeholders})"
            params.extend(labels)

        query += " ORDER BY a.run_date DESC, a.score DESC" if dedupe else " ORDER BY run_date DESC, score DESC"

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

    def get_grouped_by_domain(
        self,
        days: int = 90,
        sources: list[str] | None = None,
        labels: list[str] | None = None,
    ) -> dict[str, list[RankedWork]]:
        """Get works grouped by domain."""
        works = self.get_all(days=days, sources=sources, labels=labels)
        grouped: dict[str, list[RankedWork]] = {}
        for work in works:
            domain = work.domain or "未分类"
            if domain not in grouped:
                grouped[domain] = []
            grouped[domain].append(work)
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
        """Get archive statistics (deduplicated by identifier)."""
        conn = self.connect()
        cursor = conn.execute(
            """
            WITH latest AS (
                SELECT a.*
                FROM archive a
                INNER JOIN (
                    SELECT identifier, MAX(run_date) as max_date
                    FROM archive
                    WHERE run_date >= date('now', ?)
                    GROUP BY identifier
                ) m ON a.identifier = m.identifier AND a.run_date = m.max_date
            )
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN label = 'must_read' THEN 1 ELSE 0 END) as must_read,
                SUM(CASE WHEN label = 'consider' THEN 1 ELSE 0 END) as consider,
                SUM(CASE WHEN label = 'followed' THEN 1 ELSE 0 END) as followed,
                COUNT(DISTINCT source) as source_count,
                COUNT(DISTINCT venue) as venue_count,
                MIN(run_date) as earliest,
                MAX(run_date) as latest
            FROM latest
            """,
            (f"-{days} days",),
        )
        row = cursor.fetchone()
        return {
            "total": row["total"] or 0,
            "must_read": row["must_read"] or 0,
            "consider": row["consider"] or 0,
            "followed": row["followed"] or 0,
            "source_count": row["source_count"] or 0,
            "venue_count": row["venue_count"] or 0,
            "earliest": row["earliest"],
            "latest": row["latest"],
        }

    def get_sources(self, days: int = 90) -> list[dict]:
        """Get source distribution (deduplicated by identifier)."""
        conn = self.connect()
        cursor = conn.execute(
            """
            WITH latest AS (
                SELECT a.*
                FROM archive a
                INNER JOIN (
                    SELECT identifier, MAX(run_date) as max_date
                    FROM archive
                    WHERE run_date >= date('now', ?)
                    GROUP BY identifier
                ) m ON a.identifier = m.identifier AND a.run_date = m.max_date
            )
            SELECT source, COUNT(*) as count
            FROM latest
            GROUP BY source
            ORDER BY count DESC
            """,
            (f"-{days} days",),
        )
        return [{"source": row["source"], "count": row["count"]} for row in cursor.fetchall()]

    def get_followed_author_state(self, author_id: str) -> dict | None:
        """Get state for a followed author.

        Returns:
            Dict with author_id, author_name, last_fetch_date, total_works,
            or None if not found.
        """
        conn = self.connect()
        cursor = conn.execute(
            "SELECT * FROM followed_author_state WHERE author_id = ?",
            (author_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None
        return {
            "author_id": row["author_id"],
            "author_name": row["author_name"],
            "last_fetch_date": row["last_fetch_date"],
            "total_works": row["total_works"],
        }

    def get_all_followed_author_states(self) -> dict[str, str]:
        """Get last fetch dates for all followed authors.

        Returns:
            Dict mapping author_id to last_fetch_date (YYYY-MM-DD).
        """
        conn = self.connect()
        cursor = conn.execute(
            "SELECT author_id, last_fetch_date FROM followed_author_state WHERE last_fetch_date IS NOT NULL"
        )
        return {row["author_id"]: row["last_fetch_date"] for row in cursor.fetchall()}

    def update_followed_author_state(
        self,
        author_id: str,
        author_name: str,
        last_fetch_date: str,
        total_works: int,
    ) -> None:
        """Update state for a followed author."""
        conn = self.connect()
        conn.execute(
            """
            INSERT INTO followed_author_state (author_id, author_name, last_fetch_date, total_works, updated_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(author_id) DO UPDATE SET
                author_name = excluded.author_name,
                last_fetch_date = excluded.last_fetch_date,
                total_works = excluded.total_works,
                updated_at = CURRENT_TIMESTAMP
            """,
            (author_id, author_name, last_fetch_date, total_works),
        )
        conn.commit()

    def get_known_identifiers(self, source: str | None = None) -> set[str]:
        """Get all known identifiers in archive (for deduplication).

        Args:
            source: Optional source filter (e.g., "openalex").

        Returns:
            Set of identifier strings.
        """
        conn = self.connect()
        if source:
            cursor = conn.execute(
                "SELECT DISTINCT identifier FROM archive WHERE source = ?",
                (source,),
            )
        else:
            cursor = conn.execute("SELECT DISTINCT identifier FROM archive")
        return {row["identifier"] for row in cursor.fetchall()}

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

        # Build extra dict
        extra: dict[str, object] = {"run_date": row["run_date"]}
        followed_author = row["followed_author"] if "followed_author" in row.keys() else None
        if followed_author:
            extra["followed_author"] = followed_author

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
            domain=row["domain"],
            extra=extra,
        )


__all__ = ["ArchiveStorage"]
