"""Daily push notifications for new recommendations.

Currently supports WeChat push via Server酱 (sctapi.ftqq.com): a compact
"today's new papers" digest is sent after each watch run, with a link back to
the full HTML report. The message is built from the structured RankedWork
results, so no HTML parsing is involved.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import requests

from zotwatch.core.models import RankedWork

if TYPE_CHECKING:
    from zotwatch.config.settings import Settings
    from zotwatch.pipeline.watch import WatchResult

logger = logging.getLogger(__name__)

SERVERCHAN_API = "https://sctapi.ftqq.com/{sendkey}.send"
_TITLE_MAX = 32  # Server酱 caps the title at 32 characters


def _work_link(work: RankedWork) -> str:
    """Resolve a clickable link for a work (prefer url, then DOI)."""
    if work.url:
        return work.url
    if work.doi:
        return f"https://doi.org/{work.doi}"
    return ""


def _work_line(idx: int, work: RankedWork) -> str:
    """Render one Markdown list line for a work."""
    title = work.translated_title or work.title
    link = _work_link(work)
    meta_bits = [f"评分 {work.score:.2f}"]
    if work.venue:
        meta_bits.append(work.venue)
    meta = " · ".join(meta_bits)
    head = f"[{title}]({link})" if link else title
    return f"{idx}. {head}\n   {meta}"


def build_message(
    ranked: list[RankedWork],
    flagship: list[RankedWork] | None,
    *,
    top_n: int,
    report_url: str,
    include_flagship: bool,
) -> tuple[str, str]:
    """Build the (title, markdown_body) for the daily push."""
    flagship = flagship or []
    show_flagship = include_flagship and bool(flagship)
    total = len(ranked) + (len(flagship) if show_flagship else 0)

    title = f"ZotWatch 今日 {total} 篇新文献"

    lines: list[str] = [f"## 今日推荐 {total} 篇\n"]
    if ranked:
        lines.append("### 📄 个性化推荐")
        lines.extend(_work_line(i, w) for i, w in enumerate(ranked[:top_n], start=1))
        lines.append("")
    if show_flagship:
        lines.append("### 🏆 顶刊地学速览")
        lines.extend(_work_line(i, w) for i, w in enumerate(flagship[:top_n], start=1))
        lines.append("")
    if report_url:
        lines.append(f"[👉 查看完整报告]({report_url})")

    return title, "\n".join(lines)


class ServerChanNotifier:
    """Push daily recommendations to WeChat via Server酱 (sctapi.ftqq.com)."""

    def __init__(self, sendkey: str, *, timeout: float = 15.0) -> None:
        if not sendkey:
            raise ValueError("Server酱 sendkey is required for notifications")
        self.sendkey = sendkey
        self.timeout = timeout

    def send(self, title: str, body: str) -> None:
        """Send one push; raises on transport or API-level failure."""
        url = SERVERCHAN_API.format(sendkey=self.sendkey)
        resp = requests.post(
            url,
            data={"title": title[:_TITLE_MAX], "desp": body},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        # Server酱 returns code 0 on success.
        if payload.get("code", 0) != 0:
            raise RuntimeError(f"Server酱 push failed: {payload}")
        logger.info("Server酱 push sent: %s", title)


def send_notification(result: WatchResult, settings: Settings) -> bool:
    """Send the daily push if enabled and there is content to report.

    Returns True if a push was actually sent, False if skipped (disabled or
    nothing to report).
    """
    cfg = settings.output.notify
    if not cfg.enabled:
        return False

    ranked = result.ranked_works or []
    flagship = result.flagship_works or []
    show_flagship = cfg.include_flagship and bool(flagship)
    if not ranked and not show_flagship:
        logger.info("No recommendations to notify; skipping push")
        return False

    if cfg.provider != "serverchan":
        raise ValueError(f"Unsupported notify provider: {cfg.provider!r}")

    report_url = cfg.report_url or settings.output.rss.link
    title, body = build_message(
        ranked,
        flagship,
        top_n=cfg.top_n,
        report_url=report_url,
        include_flagship=cfg.include_flagship,
    )
    ServerChanNotifier(cfg.sendkey).send(title, body)
    return True


__all__ = ["ServerChanNotifier", "build_message", "send_notification"]
