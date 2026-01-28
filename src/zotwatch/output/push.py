"""Push recommendations to Zotero."""

import logging
from collections.abc import Iterable

from zotwatch.config.settings import Settings
from zotwatch.core.constants import DEFAULT_HTTP_TIMEOUT, ZOTERO_API_PAGE_SIZE
from zotwatch.core.exceptions import StorageError
from zotwatch.core.models import RankedWork
from zotwatch.infrastructure.http import HTTPClient

logger = logging.getLogger(__name__)

API_BASE = "https://api.zotero.org"
COLLECTION_NAME = "AI Suggested"
RETRYABLE_STATUSES = {429, 500, 502, 503, 504}


class ZoteroPusher:
    """Push recommendations to Zotero library."""

    def __init__(self, settings: Settings):
        self.settings = settings
        api_key = settings.zotero.api.api_key
        self.http = HTTPClient(
            headers={
                "Zotero-API-Version": "3",
                "Authorization": f"Bearer {api_key}",
                "User-Agent": "ZotWatch/0.2",
                "Content-Type": "application/json",
            },
            timeout=DEFAULT_HTTP_TIMEOUT,
            max_retries=3,
            retryable_statuses=RETRYABLE_STATUSES,
        )
        self.base_url = f"{API_BASE}/users/{settings.zotero.api.user_id}"
        self._collection_key: str | None = None

    def push(self, works: Iterable[RankedWork], note_template: str | None = None) -> None:
        """Push recommendations as notes to Zotero.

        Args:
            works: Ranked works to push
            note_template: Optional template for note content
        """
        works_list = list(works)
        if not works_list:
            logger.info("No works provided for Zotero push")
            return

        collection_key = self._ensure_collection()
        payload = []
        for work in works_list:
            note = (note_template or "Recommended due to score {score:.3f}").format(**work.model_dump())
            payload.append(
                {
                    "itemType": "note",
                    "note": note,
                    "tags": [
                        {"tag": "ZotWatch"},
                        {"tag": work.label},
                    ],
                    "collections": [collection_key],
                }
            )

        url = f"{self.base_url}/items"
        logger.info("Pushing %d recommendation notes to Zotero", len(payload))
        resp = self.http.post(url, json=payload)
        resp.raise_for_status()
        logger.info("Successfully pushed notes to Zotero collection %s", collection_key)

    def _ensure_collection(self) -> str:
        """Ensure AI Suggested collection exists."""
        if self._collection_key:
            return self._collection_key

        collections_url = f"{self.base_url}/collections"
        resp = self.http.get(collections_url, params={"limit": ZOTERO_API_PAGE_SIZE})
        resp.raise_for_status()

        for collection in resp.json():
            data = collection.get("data", {})
            if data.get("name") == COLLECTION_NAME:
                self._collection_key = data.get("key")
                return self._collection_key

        # Create collection
        logger.info("Creating Zotero collection '%s'", COLLECTION_NAME)
        resp = self.http.post(collections_url, json=[{"name": COLLECTION_NAME}])
        resp.raise_for_status()
        created = resp.json()[0]
        self._collection_key = created.get("successful", {}).get("0", {}).get("data", {}).get("key")
        if not self._collection_key:
            raise StorageError(f"Failed to create or retrieve Zotero collection '{COLLECTION_NAME}'")
        return self._collection_key


__all__ = ["ZoteroPusher"]
