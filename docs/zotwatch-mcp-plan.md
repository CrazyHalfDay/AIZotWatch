# Plan: ZotWatch MCP server for WeChat ClawBot interaction

Tracking doc for building a ZotWatch MCP server so ZotWatch can be driven
interactively from WeChat via the official **ClawBot** plugin (→ OpenClaw → MCP).
(GitHub Issues are disabled on this repo, so this plan lives here.)

## Goal
Two-way interaction (beyond the daily push): chat in WeChat to query/trigger ZotWatch.

## Architecture
```
WeChat ClawBot  <->  OpenClaw (always-on gateway)  <->  ZotWatch MCP server  <->  zotwatch data/pipeline
```
- ClawBot = official WeChat plugin (launched ~2026-03), connects WeChat to a
  self-hosted OpenClaw agent gateway.
- OpenClaw natively supports MCP (via `~/.openclaw/openclaw.json` ACP mode, or
  the MCPorter middleware).

## Repo decision
- MCP server lives in its **own repo**: `CrazyHalfDay/zotwatch-mcp`
  (already created; public; default branch `main`).
- Depends on the `zotwatch` package (git dependency) — needs `ProfileRanker`,
  `ArchiveStorage`, `WatchPipeline`, embeddings/FAISS, `ZoteroPusher`, LLM client.
- **Build blocker:** the build session's repo scope must include `zotwatch-mcp`.
  A session locked to `aizotwatch` only gets "Access denied" on it, and the
  `add_repo` tool may be unavailable. Next session must select **both** repos so
  the server can be written directly into `zotwatch-mcp`.

## Phase 1 — MCP server (deliverable in `zotwatch-mcp`)
```
zotwatch-mcp/
├── pyproject.toml              # deps: zotwatch (git) + mcp SDK
├── README.md                   # deployment instructions
├── src/zotwatch_mcp/server.py  # MCP server + tools
└── examples/openclaw.json      # OpenClaw integration config example
```
Tools (all 4 requested):
- `today_recommendations` / `trigger_watch` — today's new papers / run a fresh watch
- `search_papers` — semantic search over candidates + library
- `favorite_paper` — favorite and sync back to Zotero
- `summarize_paper` — AI summary / Q&A about a specific paper

## Phase 2 — VPS deploy + WeChat link (user does, guided)
- VPS reqs: Linux, >=2 cores / >=4GB (Docker >=2GB), >=10GB disk, Node >=22 or Docker.
  No 16GB needed (uses cloud APIs, not local Ollama).
- Deploy: official script -> `openclaw onboard --install-daemon` ->
  `openclaw dashboard` -> connect the WeChat plugin.

## Open items / caveats
- **Overseas VPS <-> WeChat ClawBot**: the relevant link is *outbound* from the
  VPS to WeChat's domestic cloud (`ilinkai.weixin.qq.com`) + long-poll. WeChat
  services are globally reachable so it will *probably* work, but WeChat 风控 may
  reject a cross-border IP (phone in CN, bot on overseas IP). Only testable in
  Phase 2. (Inbound CN->VPS connectivity is confirmed, but that is the other
  direction.) Fallback: the MCP server also works with Claude Desktop / Cursor /
  other MCP clients, so it is not wasted if the WeChat link fails.
- **Data on the VPS**: the MCP server needs `data/*.sqlite` + `faiss.index`.
  Decide: sync artifacts from the daily GitHub Actions run, vs. run the pipeline
  on the VPS.

## Done so far (in aizotwatch)
- Daily WeChat push via Server酱 (`output/notify.py`, `watch --notify`) — shipped.
- Flagship real score/similarity — shipped.

## Next action
Start a web session with **both** `aizotwatch` and `zotwatch-mcp` in scope, then
say "开建 zotwatch-mcp".
