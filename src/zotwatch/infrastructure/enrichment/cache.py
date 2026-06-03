"""Metadata cache storage layer for paper enrichment."""

import json
import logging
from datetime import timedelta

from zotwatch.infrastructure.cache_base import BaseSQLiteCache
from zotwatch.utils.datetime import format_sqlite_datetime, utc_now

logger = logging.getLogger(__name__)


class MetadataCache(BaseSQLiteCache):
    """Cache for paper metadata from external APIs.

    Stores paper abstracts and other metadata with TTL support.
    Uses SQLite backend with thread-safe write operations.
    """

    def _ensure_schema(self) -> None:
        """Create metadata table if not exists."""
        conn = self._connect()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS paper_metadata (
                doi TEXT PRIMARY KEY,
                abstract TEXT,
                title TEXT,
                authors_json TEXT,
                citation_count INTEGER,
                source TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_meta_expires
                ON paper_metadata(expires_at) WHERE expires_at IS NOT NULL;

            CREATE INDEX IF NOT EXISTS idx_meta_source
                ON paper_metadata(source);
        """)
        conn.commit()

    def _get_expires_column(self) -> str:
        """Return the column name for expiration timestamps."""
        return "expires_at"

    def _get_table_name(self) -> str:
        """Return the main table name."""
        return "paper_metadata"

    def get_abstract(self, doi: str) -> str | None:
        """Get cached abstract for DOI.

        Args:
            doi: Digital Object Identifier.

        Returns:
            Abstract text if found and not expired, None otherwise.
        """
        conn = self._connect()
        cur = conn.execute(
            """
            SELECT abstract FROM paper_metadata
            WHERE doi = ?
              AND (expires_at IS NULL OR expires_at > datetime('now'))
            """,
            (doi.lower(),),
        )
        row = cur.fetchone()
        return row["abstract"] if row else None

    def get_batch(self, dois: list[str]) -> dict[str, str]:
        """Batch fetch cached abstracts.

        Args:
            dois: List of DOIs to fetch.

        Returns:
            Dict mapping DOI to abstract for found items.
        """
        if not dois:
            return {}

        # Normalize DOIs to lowercase
        normalized = [d.lower() for d in dois]
        doi_map = {d.lower(): d for d in dois}  # Map back to original case

        conn = self._connect()
        placeholders = ",".join("?" for _ in normalized)
        cur = conn.execute(
            f"""
            SELECT doi, abstract FROM paper_metadata
            WHERE doi IN ({placeholders})
              AND abstract IS NOT NULL
              AND (expires_at IS NULL OR expires_at > datetime('now'))
            """,
            normalized,
        )
        # Return with original DOI case
        return {doi_map.get(row["doi"], row["doi"]): row["abstract"] for row in cur}

    def get_failed_dois(self, dois: list[str]) -> set[str]:
        """Return DOIs negative-cached as unresolved (abstract IS NULL, not expired).

        These are DOIs that every enrichment source failed to resolve within the
        negative-cache TTL, so they should be skipped until the entry expires.

        Args:
            dois: List of DOIs to check.

        Returns:
            Set of DOIs (original case) with a live negative-cache entry.
        """
        if not dois:
            return set()

        normalized = [d.lower() for d in dois]
        doi_map = {d.lower(): d for d in dois}

        conn = self._connect()
        placeholders = ",".join("?" for _ in normalized)
        cur = conn.execute(
            f"""
            SELECT doi FROM paper_metadata
            WHERE doi IN ({placeholders})
              AND abstract IS NULL
              AND (expires_at IS NULL OR expires_at > datetime('now'))
            """,
            normalized,
        )
        return {doi_map.get(row["doi"], row["doi"]) for row in cur}

    def put(
        self,
        doi: str,
        abstract: str | None,
        source: str,
        title: str | None = None,
        authors: list[str] | None = None,
        citation_count: int | None = None,
        ttl_days: int = 30,
    ) -> None:
        """Store paper metadata with TTL (thread-safe).

        Args:
            doi: Digital Object Identifier.
            abstract: Paper abstract text.
            source: Source identifier (e.g., "semantic_scholar").
            title: Paper title.
            authors: List of author names.
            citation_count: Citation count.
            ttl_days: Time-to-live in days.
        """
        expires_at = format_sqlite_datetime(utc_now() + timedelta(days=ttl_days))
        authors_json = json.dumps(authors) if authors else None

        with self._write_lock:
            conn = self._connect()
            conn.execute(
                """
                INSERT OR REPLACE INTO paper_metadata
                    (doi, abstract, title, authors_json, citation_count, source, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (doi.lower(), abstract, title, authors_json, citation_count, source, expires_at),
            )
            conn.commit()

    def put_batch(
        self,
        items: list[tuple[str, str | None]],
        source: str,
        ttl_days: int = 30,
    ) -> None:
        """Batch store abstracts (thread-safe).

        Args:
            items: List of (doi, abstract) tuples.
            source: Source identifier.
            ttl_days: Time-to-live in days.
        """
        if not items:
            return

        expires_at = format_sqlite_datetime(utc_now() + timedelta(days=ttl_days))

        with self._write_lock:
            conn = self._connect()
            conn.executemany(
                """
                INSERT OR REPLACE INTO paper_metadata
                    (doi, abstract, source, expires_at)
                VALUES (?, ?, ?, ?)
                """,
                [(doi.lower(), abstract, source, expires_at) for doi, abstract in items],
            )
            conn.commit()

    def count(self, source: str | None = None) -> int:
        """Count cached metadata entries.

        Args:
            source: Optional filter by source.

        Returns:
            Number of cached entries.
        """
        conn = self._connect()
        if source:
            cur = conn.execute(
                "SELECT COUNT(*) FROM paper_metadata WHERE source = ?",
                (source,),
            )
        else:
            cur = conn.execute("SELECT COUNT(*) FROM paper_metadata")
        return cur.fetchone()[0]


__all__ = ["MetadataCache"]
