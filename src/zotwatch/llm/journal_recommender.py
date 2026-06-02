"""Target journal list generation service using LLM."""

import json
import logging
import re
from dataclasses import dataclass

from zotwatch.core.models import VenueStats

from .base import BaseLLMProvider
from .prompts import JOURNAL_GENERATION_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class GeneratedJournal:
    """A journal proposed by the LLM, before ISSN verification.

    The ISSN is intentionally not requested from the LLM; it is resolved
    later via Crossref to guarantee authoritative values.
    """

    title: str
    category: str
    impact_factor: float | None = None
    is_chinese: bool = False


class JournalRecommender:
    """Generates a target journal list from library venues using an LLM."""

    def __init__(
        self,
        llm: BaseLLMProvider,
        model: str | None = None,
    ):
        """Initialize the recommender.

        Args:
            llm: LLM provider instance.
            model: Optional model name to use.
        """
        self.llm = llm
        self.model = model

    def generate(
        self,
        venues: list[VenueStats],
        research_focus: str = "",
        max_tokens: int = 4096,
    ) -> list[GeneratedJournal]:
        """Generate candidate journals from the library's top venues.

        Args:
            venues: Venue statistics extracted from the user's library.
            research_focus: Optional description to steer recommendations.
            max_tokens: Token budget for the completion (journal lists can be long).

        Returns:
            List of proposed journals (without ISSN).
        """
        if not venues:
            return []

        venues_list = "\n".join(
            f"- {v.venue} [{v.venue_type}]: {v.paper_count} 篇" for v in venues
        )
        prompt = JOURNAL_GENERATION_PROMPT.format(
            venues_list=venues_list,
            research_focus=research_focus or "未提供",
        )

        response = self.llm.complete(prompt, model=self.model, max_tokens=max_tokens)
        logger.debug("LLM response for journal generation: %s", response.content)

        return self._parse_response(response.content)

    def _parse_response(self, content: str | None) -> list[GeneratedJournal]:
        """Parse LLM JSON response into a list of GeneratedJournal.

        Tries a strict parse first; if the response is wrapped in prose/reasoning
        or truncated (common with reasoning models hitting the token limit), falls
        back to salvaging every complete journal object found in the text.
        """
        if content is None:
            logger.warning("LLM returned None content for journal generation")
            return []

        entries = self._parse_entries(content)
        results: list[GeneratedJournal] = []
        for entry in entries:
            journal = self._build_journal(entry)
            if journal is not None:
                results.append(journal)
        return results

    def _parse_entries(self, content: str) -> list[dict]:
        """Extract journal entry dicts, tolerating fences, prose and truncation."""
        text = content.strip()
        # Strip a leading markdown code fence if present
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\n?", "", text)
            text = re.sub(r"\n?```$", "", text)

        # Fast path: a single well-formed JSON object
        try:
            data = json.loads(text)
            if isinstance(data, dict) and isinstance(data.get("journals"), list):
                return [e for e in data["journals"] if isinstance(e, dict)]
        except json.JSONDecodeError:
            pass

        # Salvage path: recover every complete {...} object in the original
        # content. This handles reasoning prose around the JSON as well as
        # responses truncated mid-array (we keep whatever objects completed).
        salvaged = [obj for obj in _extract_json_objects(content) if "title" in obj]
        if salvaged:
            logger.warning(
                "Journal response not valid JSON; salvaged %d complete entries",
                len(salvaged),
            )
        else:
            logger.warning("Failed to parse or salvage any journals from LLM response")
        return salvaged

    def _build_journal(self, entry: dict) -> GeneratedJournal | None:
        """Convert a parsed entry dict into a GeneratedJournal."""
        title = (entry.get("title") or "").strip()
        if not title:
            return None
        raw_if = entry.get("impact_factor")
        try:
            impact_factor = float(raw_if) if raw_if is not None else None
        except (TypeError, ValueError):
            impact_factor = None
        return GeneratedJournal(
            title=title,
            category=(entry.get("category") or "").strip(),
            impact_factor=impact_factor,
            is_chinese=bool(entry.get("is_chinese", False)),
        )


def _extract_json_objects(text: str) -> list[dict]:
    """Extract every balanced {...} JSON object from arbitrary text.

    Brace matching respects strings and escapes, so it recovers complete objects
    even when nested in prose or when the overall document is truncated before
    its closing brace (the inner, already-complete objects are still captured).
    """
    objects: list[dict] = []
    stack: list[int] = []
    in_str = False
    escape = False
    for i, ch in enumerate(text):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            stack.append(i)
        elif ch == "}" and stack:
            start = stack.pop()
            try:
                obj = json.loads(text[start : i + 1])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                objects.append(obj)
    return objects


__all__ = ["JournalRecommender", "GeneratedJournal"]
