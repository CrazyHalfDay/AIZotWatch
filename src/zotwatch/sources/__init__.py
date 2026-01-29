"""Data source implementations."""

from .arxiv import ArxivSource
from .base import SourceRegistry, get_enabled_sources
from .crossref import CrossrefSource
from .eartharxiv import EartharxivSource
from .openalex import OpenAlexAuthorFetcher
from .zotero import ZoteroClient, ZoteroIngestor

__all__ = [
    "SourceRegistry",
    "get_enabled_sources",
    "ArxivSource",
    "CrossrefSource",
    "EartharxivSource",
    "OpenAlexAuthorFetcher",
    "ZoteroClient",
    "ZoteroIngestor",
]

