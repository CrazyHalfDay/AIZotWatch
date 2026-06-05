"""Title translation service."""

import json
import logging
from typing import TypeVar

from zotwatch.core.models import RankedWork
from zotwatch.infrastructure.storage import ProfileStorage

from .base import BaseLLMProvider
from .prompts import TITLE_TRANSLATION_PROMPT

logger = logging.getLogger(__name__)

WorkType = TypeVar("WorkType", bound=RankedWork)

# Language code to display name mapping
LANGUAGE_NAMES = {
    "zh-CN": "简体中文",
    "zh-TW": "繁体中文",
    "ja": "日语",
    "ko": "韩语",
    "de": "德语",
    "fr": "法语",
    "es": "西班牙语",
}


class TitleTranslator:
    """Service for translating and caching paper titles."""

    def __init__(
        self,
        llm: BaseLLMProvider,
        storage: ProfileStorage | None = None,
        model: str | None = None,
        target_language: str = "zh-CN",
        batch_size: int = 5,
    ):
        self.llm = llm
        self.storage = storage
        self.model = model
        self.target_language = target_language
        self.batch_size = batch_size

    def translate_batch(
        self,
        works: list[WorkType],
        *,
        force: bool = False,
    ) -> dict[str, str]:
        """Translate titles for multiple papers.

        Args:
            works: List of RankedWork objects.
            force: If True, skip cache and re-translate all titles.

        Returns:
            Dict mapping paper_id to translated title.
        """
        if not works:
            return {}

        translations: dict[str, str] = {}
        papers_to_translate: list[WorkType] = []

        # Check cache first
        if not force and self.storage:
            paper_ids = [w.identifier for w in works]
            cached = self.storage.get_translations_batch(paper_ids, self.target_language)
            translations.update(cached)

            # Filter out cached papers
            cached_ids = set(cached.keys())
            papers_to_translate = [w for w in works if w.identifier not in cached_ids]

            if cached:
                logger.info("Found %d cached translations", len(cached))
        else:
            papers_to_translate = list(works)

        if not papers_to_translate:
            return translations

        logger.info("Translating %d titles in batches of %d", len(papers_to_translate), self.batch_size)

        # Process in batches
        for i in range(0, len(papers_to_translate), self.batch_size):
            batch = papers_to_translate[i : i + self.batch_size]
            batch_num = i // self.batch_size + 1
            total_batches = (len(papers_to_translate) + self.batch_size - 1) // self.batch_size

            logger.info("Processing batch %d/%d (%d titles)", batch_num, total_batches, len(batch))

            batch_translations = self._translate_batch(batch)
            translations.update(batch_translations)

            # Cache results
            if self.storage and batch_translations:
                self._cache_translations(batch, batch_translations)

        return translations

    def _translate_batch(self, works: list[WorkType]) -> dict[str, str]:
        """Translate a batch of titles using LLM."""
        # Format titles list
        titles_list = "\n".join(f"{w.identifier}: {w.title}" for w in works)

        prompt = TITLE_TRANSLATION_PROMPT.format(
            target_language=self._get_language_name(self.target_language),
            titles_list=titles_list,
        )

        try:
            response = self.llm.complete(prompt, model=self.model)
            return self._parse_response(response.content)
        except Exception as e:
            logger.warning("Translation batch failed: %s", e)
            return {}

    def _parse_response(self, content: str | None) -> dict[str, str]:
        """Parse LLM response to extract translations."""
        if content is None:
            logger.warning("LLM returned None content for translation")
            return {}
        try:
            content = content.strip()
            # Remove markdown code blocks if present
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            data = json.loads(content)
            translations = {}
            for item in data.get("translations", []):
                paper_id = item.get("id")
                translated = item.get("translated")
                if paper_id and translated:
                    translations[paper_id] = translated
            return translations
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning("Failed to parse translation response: %s", e)
            return {}

    def _cache_translations(self, works: list[WorkType], translations: dict[str, str]) -> None:
        """Cache translations to storage."""
        if not self.storage:
            return

        batch_data = []
        for work in works:
            if work.identifier in translations:
                batch_data.append(
                    {
                        "paper_id": work.identifier,
                        "original": work.title,
                        "translated": translations[work.identifier],
                    }
                )

        if batch_data:
            self.storage.save_translations_batch(
                batch_data,
                self.target_language,
                self.model or "unknown",
            )
            logger.debug("Cached %d translations", len(batch_data))

    def _get_language_name(self, code: str) -> str:
        """Get human-readable language name from code."""
        return LANGUAGE_NAMES.get(code, code)


__all__ = ["TitleTranslator"]
