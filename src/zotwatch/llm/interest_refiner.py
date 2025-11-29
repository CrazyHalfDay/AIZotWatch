"""Interest refinement service."""

import json
import logging
import re

from zotwatch.core.models import RefinedInterests

from .base import BaseLLMProvider
from .prompts import INTEREST_REFINEMENT_PROMPT

logger = logging.getLogger(__name__)


class InterestRefiner:
    """Refines user interests using LLM."""

    def __init__(
        self,
        llm: BaseLLMProvider,
        model: str | None = None,
    ):
        self.llm = llm
        self.model = model

    def refine(self, user_interests: str) -> RefinedInterests:
        """Refine natural language interests into structured query.

        Args:
            user_interests: Natural language description of research interests

        Returns:
            RefinedInterests with query and keywords
        """
        prompt = INTEREST_REFINEMENT_PROMPT.format(user_interests=user_interests)
        response = self.llm.complete(prompt, model=self.model)

        logger.debug("LLM response for interest refinement: %s", response.content)

        result = self._parse_response(response.content)

        if result.exclude_keywords:
            logger.info("Exclude keywords generated: %s", result.exclude_keywords)

        return result

    def _parse_response(self, content: str) -> RefinedInterests:
        """Parse LLM JSON response into RefinedInterests."""
        try:
            # Try to extract JSON from response
            content = content.strip()

            # Remove markdown code blocks if present
            if content.startswith("```"):
                content = re.sub(r"^```(?:json)?\n?", "", content)
                content = re.sub(r"\n?```$", "", content)

            data = json.loads(content)

            return RefinedInterests(
                refined_query=data.get("refined_query", ""),
                include_keywords=data.get("include_keywords", []),
                exclude_keywords=data.get("exclude_keywords", []),
            )

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Failed to parse interest refinement response: %s", e)
            # Return a basic fallback
            return RefinedInterests(
                refined_query=content[:500] if content else "",
                include_keywords=[],
                exclude_keywords=[],
            )


__all__ = ["InterestRefiner"]
