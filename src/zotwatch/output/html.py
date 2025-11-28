"""HTML report generation."""

import logging
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from zotwatch.core.models import FeaturedWork, OverallSummary, RankedWork, ResearcherProfile

logger = logging.getLogger(__name__)


def _get_builtin_template_dir() -> Path:
    """Get path to built-in templates directory.

    Returns:
        Path to the templates directory within the package.
    """
    # Use importlib.resources for package-relative paths
    return Path(str(resources.files("zotwatch.templates")))


def render_html(
    works: list[RankedWork],
    output_path: Path | str,
    *,
    template_dir: Path | None = None,
    template_name: str = "report.html",
    featured_works: list[FeaturedWork] | None = None,
    overall_summaries: dict[str, OverallSummary] | None = None,
    researcher_profile: ResearcherProfile | None = None,
) -> Path:
    """Render HTML report from ranked works.

    Args:
        works: Ranked works to include.
        output_path: Path to write HTML file.
        template_dir: Directory containing templates. If None, uses built-in templates.
        template_name: Name of template file.
        featured_works: Optional list of featured works based on user interests.
        overall_summaries: Optional dict with "featured" and/or "similarity" OverallSummary.
        researcher_profile: Optional researcher profile analysis.

    Returns:
        Path to written HTML file.
    """
    generated_at = datetime.now(timezone.utc)

    # Determine template directory
    if template_dir is None:
        template_dir = _get_builtin_template_dir()

    template_path = template_dir / template_name

    if template_path.exists():
        env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            autoescape=select_autoescape(["html", "xml"]),
        )
        template = env.get_template(template_name)
    else:
        # Fallback: should not happen if package is installed correctly
        logger.warning(
            "Template %s not found in %s, report generation may fail",
            template_name,
            template_dir,
        )
        raise FileNotFoundError(f"Template {template_name} not found in {template_dir}")

    rendered = template.render(
        works=works,
        generated_at=generated_at,
        featured_works=featured_works or [],
        overall_summaries=overall_summaries or {},
        researcher_profile=researcher_profile,
    )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")
    logger.info("Wrote HTML report with %d items to %s", len(works), path)
    return path


__all__ = ["render_html"]
