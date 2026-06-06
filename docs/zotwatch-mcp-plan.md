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

## Phase 1 — lightweight bot (deliverable in `zotwatch-mcp`)
```
zotwatch-mcp/
├── pyproject.toml          # deps: zotwatch (git) + requests
├── README.md               # setup + deployment
├── src/zotwatch_bot/
│   ├── ilink.py            # iLink long-poll client (receive + reply)
│   ├── actions.py          # thin wrappers over zotwatch (the 4 capabilities)
│   ├── router.py           # command parsing + optional LLM intent fallback
│   └── main.py             # loop: poll -> route -> reply
└── examples/.env.example
```
Capabilities / commands (all 4 requested):
- `今天` / `today` -> today's recommendations (read ArchiveStorage)
- `抓取` / `trigger` -> run a fresh watch (slow; reply "处理中" then result)
- `搜 <query>` -> semantic search over candidates + library
- `收藏 <n>` -> favorite the n-th paper from the last list, sync to Zotero
- `总结 <n>` / `<n>` -> AI summary / Q&A about a paper
- anything else -> LLM Q&A fallback (reuse zotwatch's LLM client)

### Implementation notes / research items for next session
- Nail down the iLink connection/auth flow: does `openclaw-weixin-cli install`
  run a local daemon to talk to, or just register + hand back a token to poll
  directly? Read the protocol doc + community impl before coding.
- Per-chat state: remember the "last shown list" so `收藏 3` / `总结 2` can map
  an index back to a paper (simple in-memory dict keyed by user).
- 5s reply window: fast queries reply synchronously; slow ops (`trigger`) reply
  "处理中…" then send the result when ready (verify iLink supports async send).

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

## Done so far (in aizotwatch)
- Daily WeChat push via Server酱 (`output/notify.py`, `watch --notify`) — shipped.
- Flagship real score/similarity — shipped.

## Next action
Start a web session with **both** `aizotwatch` and `zotwatch-mcp` in scope, then
say "开建 zotwatch-mcp".
