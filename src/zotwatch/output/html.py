"""HTML report generation."""

import logging
from datetime import datetime
from importlib import resources
from pathlib import Path
from zoneinfo import ZoneInfo

from jinja2 import Environment, FileSystemLoader, select_autoescape

from zotwatch.core.models import InterestWork, OverallSummary, RankedWork, ResearcherProfile

logger = logging.getLogger(__name__)


def _get_builtin_template_dir() -> Path:
    """Get path to built-in templates directory.

    Returns:
        Path to the templates directory within the package.
    """
    # Use importlib.resources for package-relative paths
    return Path(str(resources.files("zotwatch.templates")))


def _convert_utc_to_tz(dt: datetime | None, target_tz: ZoneInfo) -> datetime | None:
    """Convert a datetime from UTC to target timezone.

    Args:
        dt: Datetime to convert (assumes naive datetime is UTC).
        target_tz: Target timezone.

    Returns:
        Converted datetime, or None if input is None.
    """
    if dt is None:
        return None
    if dt.tzinfo is None:
        # Assume naive datetime is UTC
        dt = dt.replace(tzinfo=ZoneInfo("UTC"))
    return dt.astimezone(target_tz)


def render_html(
    works: list[RankedWork],
    output_path: Path | str,
    *,
    template_dir: Path | None = None,
    template_name: str = "report.html",
    timezone_name: str = "UTC",
    interest_works: list[InterestWork] | None = None,
    overall_summaries: dict[str, OverallSummary] | None = None,
    researcher_profile: ResearcherProfile | None = None,
) -> Path:
    """Render HTML report from ranked works.

    Args:
        works: Ranked works to include.
        output_path: Path to write HTML file.
        template_dir: Directory containing templates. If None, uses built-in templates.
        template_name: Name of template file.
        timezone_name: IANA timezone name (e.g., "Asia/Shanghai"). Defaults to "UTC".
        interest_works: Optional list of interest-based works.
        overall_summaries: Optional dict with "interest" and/or "similarity" OverallSummary.
        researcher_profile: Optional researcher profile analysis.

    Returns:
        Path to written HTML file.
    """
    tz = ZoneInfo(timezone_name)
    generated_at = datetime.now(tz)

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

    # Convert profile generation time to user timezone
    profile_generated_at = None
    if researcher_profile:
        profile_generated_at = _convert_utc_to_tz(researcher_profile.generated_at, tz)

    rendered = template.render(
        works=works,
        generated_at=generated_at,
        timezone_name=timezone_name,
        interest_works=interest_works or [],
        overall_summaries=overall_summaries or {},
        researcher_profile=researcher_profile,
        profile_generated_at=profile_generated_at,
    )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")
    logger.info("Wrote HTML report with %d items to %s", len(works), path)
    return path


__all__ = ["render_html"]
