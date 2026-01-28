"""HTML report generation."""

import json
import logging
from datetime import datetime
from importlib import resources
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
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
    Optimized with NumPy vectorization for O(n²) matrix operations in C.

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

    # Collect valid clusters with non-zero centroids
    cluster_ids: list[int] = []
    vectors: list[list[float]] = []
    for c in clusters:
        vec = c.weighted_centroid or c.centroid or []
        if not vec:
            continue
        cluster_ids.append(c.cluster_id)
        vectors.append(vec)

    n = len(cluster_ids)
    if n < 2:
        return []

    # Convert to NumPy array and normalize (vectorized)
    mat = np.array(vectors, dtype=np.float32)  # shape: (n, dim)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms == 0, 1, norms)
    mat = mat / norms

    # Compute all pairwise cosine similarities via matrix multiplication
    # Result: sim_matrix[i, j] = cosine similarity between cluster i and j
    sim_matrix = mat @ mat.T  # shape: (n, n)

    # Set diagonal to -inf to exclude self-similarity from top-K selection
    np.fill_diagonal(sim_matrix, -np.inf)

    # For each cluster, find top-K neighbors above threshold
    selected_edges: set[tuple[int, int]] = set()

    for i in range(n):
        row = sim_matrix[i]
        # Use argpartition for O(n) partial sort to find top-K indices
        if n - 1 <= max_neighbors:
            # If fewer neighbors than max_neighbors, take all
            top_indices = np.where(row > threshold)[0]
        else:
            # Get indices of top max_neighbors values
            top_k_indices = np.argpartition(row, -max_neighbors)[-max_neighbors:]
            # Filter by threshold
            top_indices = top_k_indices[row[top_k_indices] > threshold]

        cluster_id = cluster_ids[i]
        for j in top_indices:
            neighbor_id = cluster_ids[j]
            # Use canonical edge key (smaller id first) to avoid duplicates
            edge_key = (cluster_id, neighbor_id) if cluster_id < neighbor_id else (neighbor_id, cluster_id)
            selected_edges.add(edge_key)

    # Convert to link list
    links: list[dict] = []
    for id_i, id_j in selected_edges:
        # Find indices for the cluster ids
        i = cluster_ids.index(id_i)
        j = cluster_ids.index(id_j)
        sim = float(sim_matrix[i, j])
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

    data_filename = f"{Path(output_path).stem}-data.json"
    data_path = Path(output_path).with_name(data_filename)

    archive_data = {
        "group_by": group_by,
        "generated_at": generated_at.isoformat(),
        "stats": stats,
        "sources": sources or [],
        "groups": [
            {
                "name": group_name,
                "count": len(works),
                "works": [_serialize_ranked_work(work) for work in works],
            }
            for group_name, works in grouped_works.items()
        ],
    }

    data_path.write_text(
        json.dumps(archive_data, ensure_ascii=False),
        encoding="utf-8",
    )

    rendered = template.render(
        grouped_works=grouped_works,
        generated_at=generated_at,
        timezone_name=timezone_name,
        group_by=group_by,
        stats=stats,
        sources=sources or [],
        data_url=data_filename,
    )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")
    logger.info("Wrote archive page with %d groups to %s", len(grouped_works), path)
    return path


def _serialize_ranked_work(work: RankedWork) -> dict:
    data = work.model_dump(exclude={"summary"})
    if data.get("published"):
        data["published"] = data["published"].isoformat()
    extra = data.get("extra") or {}
    if isinstance(extra, dict) and extra.get("run_date"):
        run_date = extra.get("run_date")
        if hasattr(run_date, "isoformat"):
            extra["run_date"] = run_date.isoformat()
    data["extra"] = extra
    return data


__all__ = ["render_html", "render_archive"]
