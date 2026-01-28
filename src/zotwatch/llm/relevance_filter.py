"""LLM-based relevance filtering for candidate papers."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from zotwatch.core.models import CandidateWork

from .base import BaseLLMProvider
from .prompts import RELEVANCE_FILTER_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class RelevanceFilterResult:
    """Result of LLM relevance filtering."""

    relevant_ids: list[str]
    irrelevant_ids: list[str]


class PaperRelevanceFilter:
    """Filter candidates based on LLM relevance to user interests."""

    def __init__(
        self,
        llm: BaseLLMProvider,
        *,
        model: str | None = None,
        batch_size: int = 20,
        max_candidates: int = 200,
    ) -> None:
        self.llm = llm
        self.model = model
        self.batch_size = max(1, batch_size)
        self.max_candidates = max_candidates

    def filter_candidates(
        self,
        candidates: list[CandidateWork],
        *,
        user_interests: str,
    ) -> tuple[list[CandidateWork], int]:
        """Filter candidates with LLM relevance classification.

        Args:
            candidates: Candidate works to filter.
            user_interests: Interest description for relevance judgement.

        Returns:
            (filtered_candidates, removed_count)
        """
        if not candidates:
            return [], 0

        if not user_interests.strip():
            logger.warning("LLM relevance filter skipped: empty user interests")
            return candidates, 0

        candidates_to_check = candidates
        if self.max_candidates > 0 and len(candidates) > self.max_candidates:
            candidates_to_check = candidates[: self.max_candidates]
            logger.info(
                "LLM relevance filter: limiting to first %d candidates (out of %d)",
                self.max_candidates,
                len(candidates),
            )

        relevant_ids: set[str] = set()
        irrelevant_ids: set[str] = set()

        for i in range(0, len(candidates_to_check), self.batch_size):
            batch = candidates_to_check[i : i + self.batch_size]
            prompt = RELEVANCE_FILTER_PROMPT.format(
                user_interests=user_interests.strip(),
                papers_list=self._format_papers(batch),
            )
            response = self.llm.complete(prompt, model=self.model)
            parsed = self._parse_response(response.content)
            if parsed is None:
                logger.warning("LLM relevance filter: failed to parse response, keeping batch")
                for c in batch:
                    relevant_ids.add(c.identifier)
                continue
            relevant_ids.update(parsed.relevant_ids)
            irrelevant_ids.update(parsed.irrelevant_ids)

        if self.max_candidates > 0 and len(candidates) > self.max_candidates:
            # Keep unfiltered remainder to avoid accidental drops
            for c in candidates[self.max_candidates :]:
                relevant_ids.add(c.identifier)

        filtered = [c for c in candidates if c.identifier in relevant_ids]
        removed = len(candidates) - len(filtered)
        if removed > 0:
            logger.info("LLM relevance filter removed %d candidates", removed)

        return filtered, removed

    def _format_papers(self, batch: list[CandidateWork]) -> str:
        lines = []
        for c in batch:
            abstract = c.abstract or "No abstract available"
            lines.append(f"- {c.identifier}\n  Title: {c.title}\n  Abstract: {abstract}")
        return "\n".join(lines)

    def _parse_response(self, content: str) -> RelevanceFilterResult | None:
        try:
            text = content.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()
            data = json.loads(text)
            relevant = data.get("relevant_ids", []) or []
            irrelevant = data.get("irrelevant_ids", []) or []
            return RelevanceFilterResult(
                relevant_ids=[str(x) for x in relevant],
                irrelevant_ids=[str(x) for x in irrelevant],
            )
        except (json.JSONDecodeError, TypeError) as exc:
            logger.warning("Failed to parse relevance filter response: %s", exc)
            return None

