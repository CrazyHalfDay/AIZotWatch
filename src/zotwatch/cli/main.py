"""Main CLI entry point using Click."""

import logging
from pathlib import Path

import click
from dotenv import load_dotenv

from zotwatch import __version__
from zotwatch.config import Settings, load_settings
from zotwatch.infrastructure.embedding import EmbeddingCache, create_embedding_provider
from zotwatch.infrastructure.storage import ArchiveStorage, ProfileStorage
from zotwatch.output import render_archive, render_html, write_rss
from zotwatch.output.push import ZoteroPusher
from zotwatch.pipeline import ProfileBuilder, WatchConfig, WatchPipeline, WatchResult
from zotwatch.sources.zotero import ZoteroIngestor
from zotwatch.utils.datetime import utc_today_start
from zotwatch.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def _get_base_dir() -> Path:
    """Get base directory from current working directory or git root."""
    cwd = Path.cwd()
    # Check for config/config.yaml to identify project root
    if (cwd / "config" / "config.yaml").exists():
        return cwd
    # Try parent directories
    for parent in cwd.parents:
        if (parent / "config" / "config.yaml").exists():
            return parent
    return cwd


def _get_embedding_cache(base_dir: Path) -> EmbeddingCache:
    """Get or create embedding cache for the given base directory."""
    cache_db_path = base_dir / "data" / "embeddings.sqlite"
    return EmbeddingCache(cache_db_path)


@click.group()
@click.option("--base-dir", type=click.Path(exists=True), default=None, help="Repository base directory")
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging")
@click.version_option(version=__version__, prog_name="zotwatch")
@click.pass_context
def cli(ctx: click.Context, base_dir: str | None, verbose: bool) -> None:
    """ZotWatch - Personalized academic paper recommendations."""
    ctx.ensure_object(dict)

    base = Path(base_dir) if base_dir else _get_base_dir()
    load_dotenv(base / ".env")
    setup_logging(verbose=verbose)

    ctx.obj["base_dir"] = base
    ctx.obj["verbose"] = verbose

    # Load settings lazily (some commands may not need them)
    ctx.obj["_settings"] = None
    ctx.obj["_embedding_cache"] = None


def _get_settings(ctx: click.Context) -> Settings:
    """Get or load settings."""
    if ctx.obj["_settings"] is None:
        ctx.obj["_settings"] = load_settings(ctx.obj["base_dir"])
    return ctx.obj["_settings"]


def _get_cache(ctx: click.Context) -> EmbeddingCache:
    """Get or create embedding cache."""
    if ctx.obj["_embedding_cache"] is None:
        ctx.obj["_embedding_cache"] = _get_embedding_cache(ctx.obj["base_dir"])
    return ctx.obj["_embedding_cache"]


def _profile_exists(base_dir: Path) -> bool:
    """Check if profile artifacts exist."""
    faiss_path = base_dir / "data" / "faiss.index"
    sqlite_path = base_dir / "data" / "profile.sqlite"
    return faiss_path.exists() and sqlite_path.exists()


def _build_profile(
    base_dir: Path,
    settings: Settings,
    embedding_cache: EmbeddingCache,
    full: bool = True,
) -> None:
    """Build user profile from Zotero library."""
    storage = ProfileStorage(base_dir / "data" / "profile.sqlite")
    storage.initialize()

    # Progress callback for ingest
    def on_ingest_progress(stage: str, msg: str) -> None:
        click.echo(f"  [{stage}] {msg}")

    # Ingest from Zotero
    click.echo("Ingesting items from Zotero...")
    ingestor = ZoteroIngestor(storage, settings)
    stats = ingestor.run(full=full, on_progress=on_ingest_progress)
    click.echo(f"  Fetched: {stats.fetched}, Updated: {stats.updated}, Removed: {stats.removed}")

    # Count items
    total_items = storage.count_items()
    if total_items == 0:
        raise click.ClickException(
            "No items found in your Zotero library. Please add some papers to Zotero before running ZotWatch."
        )

    # Build profile with unified cache (incremental: skips if nothing changed)
    vectorizer = create_embedding_provider(settings.embedding)
    builder = ProfileBuilder(
        base_dir,
        storage,
        settings,
        vectorizer=vectorizer,
        embedding_cache=embedding_cache,
    )
    artifacts = builder.run(full=full)

    click.echo(f"Profile ready: {artifacts.faiss_path}")


@cli.command()
@click.option("--full", is_flag=True, help="Full rebuild of profile (recompute all embeddings)")
@click.pass_context
def profile(ctx: click.Context, full: bool) -> None:
    """Build or update user research profile.

    By default, uses cached embeddings where available.
    Use --full to invalidate cache and recompute all embeddings.
    """
    settings = _get_settings(ctx)
    base_dir = ctx.obj["base_dir"]
    storage = ProfileStorage(base_dir / "data" / "profile.sqlite")
    storage.initialize()
    embedding_cache = _get_cache(ctx)

    # Progress callback for ingest
    def on_ingest_progress(stage: str, msg: str) -> None:
        click.echo(f"  [{stage}] {msg}")

    # Ingest from Zotero
    click.echo("Ingesting items from Zotero...")
    ingestor = ZoteroIngestor(storage, settings)
    stats = ingestor.run(full=full, on_progress=on_ingest_progress)
    click.echo(f"  Fetched: {stats.fetched}, Updated: {stats.updated}, Removed: {stats.removed}")

    # Build profile with unified cache
    vectorizer = create_embedding_provider(settings.embedding)
    builder = ProfileBuilder(
        base_dir,
        storage,
        settings,
        vectorizer=vectorizer,
        embedding_cache=embedding_cache,
    )

    # Check if rebuild is needed before logging
    if full:
        click.echo("Building profile (full rebuild)...")
    elif not builder._can_skip_rebuild():
        total_items = storage.count_items()
        cached_profile = embedding_cache.count(source_type="profile", model=settings.embedding.model)
        if cached_profile < total_items:
            click.echo(f"Building profile ({total_items - cached_profile}/{total_items} items need embedding)...")
        else:
            click.echo(f"Building profile (all {total_items} embeddings cached, rebuilding FAISS index)...")
    else:
        click.echo("Profile is up to date, no rebuild needed.")

    artifacts = builder.run(full=full)

    click.echo(f"Profile ready:")
    click.echo(f"  SQLite: {artifacts.sqlite_path}")
    click.echo(f"  FAISS: {artifacts.faiss_path}")


@cli.command()
@click.option("--rss", is_flag=True, help="Generate RSS feed only")
@click.option("--report", is_flag=True, help="Generate HTML report only")
@click.option("--top", type=int, default=None, help="Number of top results (default: from config)")
@click.option("--push", is_flag=True, help="Push recommendations to Zotero")
@click.pass_context
def watch(
    ctx: click.Context,
    rss: bool,
    report: bool,
    top: int | None,
    push: bool,
) -> None:
    """Fetch, score, and output paper recommendations.

    By default, generates RSS feed and HTML report with AI summaries.
    Use --rss or --report to generate specific output formats.
    """
    # If none specified, generate all
    if not rss and not report:
        rss = True
        report = True

    settings = _get_settings(ctx)
    base_dir = ctx.obj["base_dir"]
    embedding_cache = _get_cache(ctx)

    # Build pipeline config from settings + CLI overrides
    config = WatchConfig(
        top_k=top if top is not None else settings.watch.top_k,
        recent_days=settings.watch.recent_days,
        max_preprint_ratio=settings.watch.max_preprint_ratio,
        require_abstract=settings.watch.require_abstract,
        generate_summaries=settings.llm.enabled,
        translate_titles=settings.llm.enabled and settings.llm.translation.enabled and report,
    )

    # Progress callback for CLI output
    def on_progress(stage: str, msg: str) -> None:
        click.echo(f"[{stage}] {msg}")

    # Run pipeline
    pipeline = WatchPipeline(base_dir, settings, config, embedding_cache)
    result = pipeline.run(on_progress=on_progress)

    # Handle empty results
    has_ranked = bool(result.ranked_works)
    has_followed = bool(result.followed_works)

    if not has_ranked and not has_followed:
        click.echo("No recommendations found")
        if rss:
            write_rss([], base_dir / "reports" / "feed.xml")
        if report:
            render_html([], base_dir / "reports" / "report-empty.html", timezone_name=settings.output.timezone)
        return

    # Display computed thresholds
    if result.computed_thresholds:
        t = result.computed_thresholds
        click.echo(f"\nThresholds ({t.mode}): must_read >= {t.must_read:.3f}, consider >= {t.consider:.3f}")

    # Display top recommendations
    if has_ranked:
        click.echo(f"\nTop {min(10, len(result.ranked_works))} recommendations:")
        for idx, work in enumerate(result.ranked_works[:10], start=1):
            click.echo(f"  {idx:02d} | {work.score:.3f} | {work.label} | {work.title[:60]}...")

    # Display followed author papers
    if has_followed:
        click.echo(f"\nFollowed authors: {len(result.followed_works)} new papers")
        for idx, work in enumerate(result.followed_works[:5], start=1):
            author = work.extra.get("followed_author", "")
            click.echo(f"  {idx:02d} | {author} | {work.title[:60]}...")

    # Generate outputs
    _output_results(result, base_dir, settings, rss, report, push)

    # Save to archive
    archive_db = base_dir / "data" / "archive.sqlite"
    with ArchiveStorage(archive_db) as archive:
        saved = archive.save_batch(result.ranked_works)
        click.echo(f"Saved {saved} ranked works to archive")
        if result.followed_works:
            saved_followed = archive.save_batch(result.followed_works)
            click.echo(f"Saved {saved_followed} followed author works to archive")


def _output_results(
    result: WatchResult,
    base_dir: Path,
    settings: Settings,
    rss: bool,
    report: bool,
    push: bool,
) -> None:
    """Generate output files from watch results."""
    if rss:
        rss_path = base_dir / "reports" / "feed.xml"
        write_rss(
            result.ranked_works,
            rss_path,
            title=settings.output.rss.title,
            link=settings.output.rss.link,
            description=settings.output.rss.description,
        )
        click.echo(f"RSS feed: {rss_path}")

    if report:
        report_name = f"report-{utc_today_start():%Y%m%d}.html"
        report_path = base_dir / "reports" / report_name
        template_dir = base_dir / "templates"
        render_html(
            result.ranked_works,
            report_path,
            template_dir=template_dir if template_dir.exists() else None,
            timezone_name=settings.output.timezone,
            interest_works=result.interest_works if result.interest_works else None,
            followed_works=result.followed_works if result.followed_works else None,
            overall_summaries=result.overall_summaries if result.overall_summaries else None,
            researcher_profile=result.researcher_profile,
        )
        click.echo(f"HTML report: {report_path}")

    if push:
        pusher = ZoteroPusher(settings)
        pusher.push(result.ranked_works)
        click.echo("Pushed recommendations to Zotero")


if __name__ == "__main__":
    cli()


@cli.command()
@click.option("--days", default=90, help="Number of days to include")
@click.option(
    "--group-by",
    type=click.Choice(["date", "venue", "source", "label", "domain", "author", "all"]),
    default="all",
    help="Grouping dimension ('all' generates all views)",
)
@click.pass_context
def archive(ctx: click.Context, days: int, group_by: str) -> None:
    """Generate archive page with historical recommendations."""
    settings = _get_settings(ctx)
    base_dir = ctx.obj["base_dir"]
    archive_db = base_dir / "data" / "archive.sqlite"

    if not archive_db.exists():
        raise click.ClickException(
            "No archive found. Run 'zotwatch watch' first to build the archive."
        )

    template_dir = base_dir / "templates"
    reports_dir = base_dir / "reports"

    # Determine which views to generate
    views = ["date", "venue", "source", "label", "domain", "author"] if group_by == "all" else [group_by]

    with ArchiveStorage(archive_db) as storage:
        stats = storage.get_stats(days=days)
        sources = storage.get_sources(days=days)

        for view in views:
            # Get grouped works based on dimension
            if view == "date":
                grouped = storage.get_grouped_by_date(days=days)
            elif view == "venue":
                grouped = storage.get_grouped_by_venue(days=days)
            elif view == "source":
                grouped = storage.get_grouped_by_source(days=days)
            elif view == "label":
                grouped = storage.get_grouped_by_label(days=days)
            elif view == "domain":
                grouped = storage.get_grouped_by_domain(days=days)
            elif view == "author":
                grouped = storage.get_grouped_by_author(days=days)
            else:
                grouped = storage.get_grouped_by_date(days=days)

            if not grouped:
                click.echo(f"No archived works found for view: {view}")
                continue

            # Render archive page
            if view == "date":
                archive_path = reports_dir / "archive.html"  # Default page
            else:
                archive_path = reports_dir / f"archive-{view}.html"

            render_archive(
                grouped,
                archive_path,
                group_by=view,
                stats=stats,
                sources=sources,
                template_dir=template_dir if template_dir.exists() else None,
                timezone_name=settings.output.timezone,
            )
            click.echo(f"Archive page ({view}): {archive_path}")

    click.echo(f"  Total: {stats['total']} papers, Must-read: {stats['must_read']}, Consider: {stats['consider']}")

