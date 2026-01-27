"""HTML report generation."""

import logging
import math
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


def _build_cluster_links(
    clustered_profile,
    threshold: float = 0.5,
    max_neighbors: int = 2,
) -> list[dict]:
    """Precompute inter-cluster similarity links using KNN + threshold strategy.

    Uses weighted_centroid when available. Falls back to centroid.

    Strategy: For each cluster, keep at most `max_neighbors` edges to its
    most similar neighbors, but only if similarity > threshold.

    Args:
        clustered_profile: ClusteredProfile with cluster centroids.
        threshold: Minimum similarity threshold for edges.
        max_neighbors: Maximum number of neighbors per cluster (K in KNN).

    Returns:
        List of edge dicts: {"source": id, "target": id, "value": similarity}
    """
    clusters = getattr(clustered_profile, "clusters", None) or []
    if not clusters:
        return []

    # Normalize centroids to avoid front-end recomputation bias
    normalized = []
    for c in clusters:
        vec = c.weighted_centroid or c.centroid or []
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0:
            continue
        normalized.append((c.cluster_id, [v / norm for v in vec]))

    n = len(normalized)
    if n < 2:
        return []

    # Compute all pairwise similarities
    similarity_matrix: dict[tuple[int, int], float] = {}
    for i in range(n):
        id_i, vec_i = normalized[i]
        for j in range(i + 1, n):
            id_j, vec_j = normalized[j]
            # dot product (cosine since normalized)
            sim = sum(a * b for a, b in zip(vec_i, vec_j))
            similarity_matrix[(id_i, id_j)] = sim
            similarity_matrix[(id_j, id_i)] = sim

    # For each cluster, find top-K neighbors above threshold
    selected_edges: set[tuple[int, int]] = set()

    for i in range(n):
        cluster_id = normalized[i][0]

        # Get all neighbors with their similarities
        neighbors = []
        for j in range(n):
            if i == j:
                continue
            neighbor_id = normalized[j][0]
            key = (cluster_id, neighbor_id) if cluster_id < neighbor_id else (neighbor_id, cluster_id)
            sim = similarity_matrix.get((cluster_id, neighbor_id), 0)
            if sim > threshold:
                neighbors.append((neighbor_id, sim, key))

        # Sort by similarity descending, take top K
        neighbors.sort(key=lambda x: x[1], reverse=True)
        for neighbor_id, sim, edge_key in neighbors[:max_neighbors]:
            selected_edges.add(edge_key)

    # Convert to link list
    links: list[dict] = []
    for id_i, id_j in selected_edges:
        sim = similarity_matrix.get((id_i, id_j), 0)
        links.append({"source": id_i, "target": id_j, "value": sim})

    return links


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
    cluster_links: list[dict] = []
    if researcher_profile:
        profile_generated_at = _convert_utc_to_tz(researcher_profile.generated_at, tz)
        if researcher_profile.clustered_profile:
            cluster_links = _build_cluster_links(researcher_profile.clustered_profile)

    rendered = template.render(
        works=works,
        generated_at=generated_at,
        timezone_name=timezone_name,
        interest_works=interest_works or [],
        overall_summaries=overall_summaries or {},
        researcher_profile=researcher_profile,
        profile_generated_at=profile_generated_at,
        cluster_links=cluster_links,
    )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")
    logger.info("Wrote HTML report with %d items to %s", len(works), path)
    return path


def render_archive(
    grouped_works: dict[str, list[RankedWork]],
    output_path: Path | str,
    *,
    group_by: str = "date",
    stats: dict | None = None,
    sources: list[dict] | None = None,
    template_dir: Path | None = None,
    template_name: str = "archive.html",
    timezone_name: str = "UTC",
) -> Path:
    """Render archive page with grouped works.

    Args:
        grouped_works: Works grouped by the specified dimension.
        output_path: Path to write HTML file.
        group_by: Grouping dimension (date, venue, source, label).
        stats: Archive statistics from ArchiveStorage.get_stats().
        sources: Source distribution from ArchiveStorage.get_sources().
        template_dir: Directory containing templates.
        template_name: Name of template file.
        timezone_name: IANA timezone name.

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
        raise FileNotFoundError(f"Template {template_name} not found in {template_dir}")

    # Default stats if not provided
    if stats is None:
        total_works = sum(len(works) for works in grouped_works.values())
        stats = {
            "total": total_works,
            "must_read": 0,
            "consider": 0,
        }

    rendered = template.render(
        grouped_works=grouped_works,
        generated_at=generated_at,
        timezone_name=timezone_name,
        group_by=group_by,
        stats=stats,
        sources=sources or [],
    )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")
    logger.info("Wrote archive page with %d groups to %s", len(grouped_works), path)
    return path


__all__ = ["render_html", "render_archive"]
