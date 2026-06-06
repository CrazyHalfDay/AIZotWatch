# Plan: ZotWatch WeChat bot (lightweight, no full OpenClaw)

Tracking doc for building a lightweight WeChat bot so ZotWatch can be driven
interactively from WeChat. (GitHub Issues are disabled on this repo, so this
plan lives here.)

> **Architecture decision (latest): Option A — lightweight direct bot.**
> Do NOT deploy full OpenClaw. The WeChat ClawBot link is just Tencent's iLink
> protocol (`ilinkai.weixin.qq.com`): plain HTTP/JSON + Bearer token + long-poll,
> no SDK. A small Python service handles it and calls `zotwatch` directly.

## Goal
Two-way interaction (beyond the daily push): chat in WeChat to query/trigger
ZotWatch, with a tiny VPS footprint.

## Architecture (lightweight)
```
WeChat ClawBot  <->  small Python service (iLink long-poll + command router)  <->  zotwatch
```
- Connect once via the official `npx @tencent-weixin/openclaw-weixin-cli install`
  (scan QR to link the WeChat account); then our service long-polls iLink.
- No OpenClaw gateway, no Node runtime needed at runtime, no 4GB requirement.

### Why not full OpenClaw / MCP (Option B, rejected for now)
- Heavier (Node + ~4GB + always-on agent gateway).
- B's only upside was MCP reusability (Claude Desktop/Cursor) as a fallback if
  the WeChat link fails. Accepted that risk to stay light. The action layer is
  kept as clean functions so a future MCP wrapper is still possible if wanted.

## Repo
- Lives in `CrazyHalfDay/zotwatch-mcp` (already created; public; `main`).
  Name says "mcp" but it's now a lightweight bot — fine to keep, or rename later.
- Depends on the `zotwatch` package (git dependency).
- **Build blocker:** the build session must have `zotwatch-mcp` in scope. A
  session locked to `aizotwatch` gets "Access denied" and has no `add_repo` tool.
  Next session: select **both** repos, then say "开建 zotwatch-mcp".

## Phase 1 — lightweight bot (✅ SHIPPED in `zotwatch-mcp`)
Delivered on branch `claude/focused-einstein-xls3P`. Final layout:
```
zotwatch-mcp/
├── pyproject.toml          # deps: zotwatch (git) + requests + numpy + qrcode
├── README.md               # setup + commands + Phase 2 deploy notes
├── examples/.env.example
└── src/zotwatch_bot/
    ├── ilink.py            # iLink client: QR login, getupdates, sendmessage
    ├── actions.py          # thin wrappers over zotwatch (all capabilities)
    ├── render.py           # plain-text list/result rendering for WeChat
    ├── router.py           # command parsing + per-chat last-list + LLM fallback
    ├── config.py           # env-driven BotConfig (ZOTWATCH_DIR, token file, …)
    └── main.py             # loop: poll -> route -> reply (slow ops off-thread)
```
All capabilities are thin wrappers over existing zotwatch components (no
duplicated pipeline): `今天`→ArchiveStorage, `抓取`→WatchPipeline, `搜`→FAISS
library + semantic re-rank of recent archived candidates, `收藏`→ZoteroPusher,
`总结`→PaperSummarizer, free text→LLM client. The action layer is WeChat-agnostic
so a future MCP wrapper can reuse it.
Capabilities / commands (all 4 requested):
- `今天` / `today` -> today's recommendations (read ArchiveStorage)
- `抓取` / `trigger` -> run a fresh watch (slow; reply "处理中" then result)
- `搜 <query>` -> semantic search over candidates + library
- `收藏 <n>` -> favorite the n-th paper from the last list, sync to Zotero
- `总结 <n>` / `<n>` -> AI summary / Q&A about a paper
- anything else -> LLM Q&A fallback (reuse zotwatch's LLM client)

### Implementation notes (resolved during Phase 1)
- iLink flow: no local daemon needed. QR login (`get_bot_qrcode` →
  `get_qrcode_status`) hands back a `bot_token` + `baseurl`; we then long-poll
  `ilink/bot/getupdates` (cursor `get_updates_buf`) and reply via
  `ilink/bot/sendmessage`, echoing the inbound `context_token`. Implemented in
  `ilink.py` from the community impl; **field names still need live
  verification in Phase 2.**
- Per-chat state: implemented as an in-memory `{user_id: [RankedWork]}` dict in
  `router.py`; `收藏 N` / `总结 N` map the index back to the last shown list.
- Async send: slow `抓取` replies "处理中…" immediately and pushes the result
  from a worker thread (`main.py`), so the poll loop never stalls. Reuses the
  inbound `context_token` for the follow-up send.

### Remaining for next session / Phase 2
- Verify the iLink field names against a live link (run the QR login once).
- `uv sync` + smoke test against a real ZotWatch `data/` (couldn't install the
  3.13 + faiss/voyage deps in the build session; code is source-verified only).

## References
- Protocol doc: https://github.com/hao-ji-xing/openclaw-weixin/blob/main/weixin-bot-api.md
- Community lib (no-OpenClaw): https://github.com/SiverKing/weixin-ClawBot-API
- Official CLI: `npx -y @tencent-weixin/openclaw-weixin-cli@latest install`

## Phase 2 — VPS deploy (user does, guided)
- Lightweight: just Python + the iLink connection. Tiny VPS is enough.
- Needs `zotwatch` installed + data artifacts (`data/*.sqlite`, `faiss.index`)
  and API keys on the VPS.
- **Data sync**: decide between syncing artifacts from the daily GitHub Actions
  run vs. running the pipeline on the VPS.

## Open caveats
- **Overseas VPS <-> WeChat 风控**: relevant link is *outbound* VPS ->
  `ilinkai.weixin.qq.com` (long-poll). WeChat is globally reachable so it will
  *probably* work, but cross-border IP (phone in CN, bot overseas) may be
  rejected by WeChat risk control. Only testable in Phase 2. (Inbound CN->VPS is
  confirmed, but that's the other direction.) If it fails, the bot is
  WeChat-only and has no fallback — would need a CN-located host.

## Done so far
- Daily WeChat push via Server酱 (`output/notify.py`, `watch --notify`) — shipped.
- Flagship real score/similarity — shipped.
- **Phase 1 lightweight bot in `zotwatch-mcp`** — shipped (see above).

## Next action
Phase 2: deploy on a VPS. `uv sync` next to a ZotWatch install with built
`data/` artifacts, run `zotwatch-bot`, scan the QR once, and verify the iLink
field names against the live link. Then sort out artifact sync (GitHub Actions
cache vs. running the pipeline on the VPS) and test the cross-border 风控 path.
