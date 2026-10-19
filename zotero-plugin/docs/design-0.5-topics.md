# ZotWatch 0.5 设计：课题追踪、未读提醒、覆盖统计

给实现者（Codex）的说明。目标是让插件从「多找文献」走到「知道遗漏在哪里、哪些值得继续追踪」。本文写清楚做什么、复用哪些现有代码、数据怎么存、界面放在哪里，以及验收标准。代码约定以仓库根目录的 `CLAUDE.md`（Zotero Plugin 一节）为准。

## 0. 原则

1. **复用调研引擎，不另起炉灶。** 课题就是一项开启了「持续追踪」的调研。种子、检索式、关键词、主题向量和候选池都已经在调研记录里了。
2. **不加新页面。** 课题放在「调研」页，概览页只加一张卡片。插件已有概览、推荐、调研、设置四页，再多就难找了。
3. **只用公开 API。** OpenAlex、Semantic Scholar、Crossref 都走现有的 `transport` 和共享限速器（`src/core/ratelimit.js`），OpenAlex 请求带 `api_key`。
   - **不得**调用 Connected Papers 的内部接口（`rest.connectedpapers.com` 等），也不得伪造请求头。
   - 每个服务只用一个 Key，不做多 Key 轮询。
4. **每周追踪默认不调用大模型。** 排序用现有的向量相似度。大模型只在用户主动要求时使用，例如对新增论文重新生成综述。
5. **运行环境约束不变：** Zotero 7–10，Firefox 115（没有 HTML `<select>`，外部文本只用 `textContent`，偏好设置的写入类型要和默认值一致，严格 XHTML）。
6. **不做的事：** 「完整度 xx%」之类的分数；把检索结果的分布当成领域本身的结论。

## 1. 课题追踪（第一优先）

### 1.1 概念

在调研记录上加一个 `tracking` 字段，开启后这项调研就成为一个「课题」：

```js
survey.tracking = {
  enabled: true,
  intervalDays: 7,          // 检查间隔，默认 7，可选 3 / 7 / 14 / 30
  createdAt, lastCheckAt,   // ms
  lastViewedAt,             // 用户上次打开「追踪」栏的时间，用于「自上次查看以来新增 N 篇」
}
```

只有已完成（`status === "done"`）、且有种子或检索式的调研才能开启。

### 1.2 存储

- **新文件 `tracking-<id>.json`**（每个课题一份，和 `survey-<id>.json` 放在同一个数据目录）。追踪结果不写进调研文件，避免和「重新生成综述」「保存到 Zotero」的写入互相覆盖（0.4.0 审查修过这类问题）。读写都经过 `src/survey/store.js` 的 `withSurveyLock(storage, …)`。

```js
{
  version: 1,
  surveyId,
  checks: [{ id, at, since, counts: { fetched, unique, known, inLibrary, kept }, errors: [] }],  // 最近 52 次
  found: {   // key：规范化 DOI，否则 OpenAlex id，否则规范化标题
    "<key>": {
      paper,              // 与调研候选相同的结构（title, doi, openalexId, year, venue, abstract, authors, citations, via, scores …）
      firstSeenAt, checkId,
      reasons: ["cites:seed1", "search:2", "rel:seed1"],   // 由哪个种子 / 哪条检索式找到
      inLibrary, itemKey, unread,                          // 见第 2 节
      status: "new" | "seen" | "added" | "dismissed",
    }
  }
}
```

- `found` 最多保留 1000 条，超出时先删最早的 `dismissed`，再删最早的 `seen`。
- **调研记录的 30 条上限不能删掉课题。** `store.js` 的 `writeSurvey` 现在会删除超出 `SURVEY_LIMIT` 的旧调研文件。要改成：开启追踪的调研不计入上限、永不自动删除；`indexEntry` 增加 `tracked: true` 和 `newCount`，供列表显示。
- 删除调研时，同时删除对应的 `tracking-<id>.json`。「清除缓存」（`resetData`）也要删掉它们，确认提示里写明。

### 1.3 一次检查做什么

新模块 `src/survey/tracking.js`，和 `src/survey/engine.js` 一样不依赖 Zotero（HTTP、storage、library 都注入），可以直接复用 `SurveyEngine` 的 `vectorsOf` / `queryText`，或抽出共用函数。

**时间窗：** `since = lastCheckAt ? lastCheckAt - 14 天 : survey.finishedAt - 30 天`。多出的 14 天用来覆盖 OpenAlex 收录的延迟；重复的结果靠 `found` 去重，不会出现两次。所有请求都加 `from_publication_date:<since 的日期>`（`sources.js` 的 `openAlexFilter({ yearFrom })` 要扩展成支持具体日期）。

**请求（只用 OpenAlex，第一版不用 Semantic Scholar，因为它每秒只能请求 1 次）：**

| 来源 | 请求 | `reasons` 标记 |
|---|---|---|
| 引用核心种子的新论文 | `filter=cites:W1\|W2\|…,from_publication_date:…`，按 `publication_date:desc` 排序，`per-page=100`，一页 | `cites:seed<n>`（看论文的 `referenced_works` 包含哪个种子；用 `select` 取 `referenced_works`） |
| 检索式的新结果 | 对 `survey.queries` 的每条做 `search=…`，`per-page=25` | `search:<i>` |
| 相关的新论文（文献耦合，类似 Connected Papers 的「相似研究」） | 每个种子取 3–5 篇被引次数最少（≥3）的参考文献，查引用它们的新论文（`cites:<ref>`，带 `referenced_works`），只保留和种子共同引用 ≥2 篇的；强度 = 共同引用数 / √(两者参考文献数之积) | `rel:seed<n>` |

- 实现前先验证 `cites:` 是否支持 `|` 多值。不支持时（HTTP 400）改为每个种子发一次请求，写法参考 `openAlexWorksByIds` 的过滤器回退。
- 预算：每个课题每次大约 15–30 次 OpenAlex 请求，不调用大模型。向量化最多 300 条（标题 + 摘要）。

**处理流程：**
1. 用 `PaperCollector`（`src/survey/pool.js`）按 DOI / OpenAlex id 合并。
2. 去掉 `survey.pool` 和 `found` 里已有的论文，以及种子本身。统计 `known`（已知数）。
3. 用 `dedupePool` 和文库比对：在文库里的不丢弃，标记 `inLibrary`，送到第 2 节的未读判断。
4. 向量化后，用和调研相同的中心（主题文字向量 + 种子向量的平均，见 `engine.js` 的 `stage_rank`）和 `scorePapers` 的权重打分。
5. 过滤：相似度（`cosine`）低于原调研候选池中位数的丢弃，防止检索式带进无关论文；剩下的按得分保留前 30 篇。
6. 缺摘要的用 `enrichMissingAbstracts`（`src/sources/abstracts.js`）补全，只补保留下来的。
7. 写入 `tracking-<id>.json`：新论文 `status: "new"`，以及这次检查的 `checks` 记录（各步数量写进 `counts`）。

### 1.4 调度

- 挂在现有调度器上（`src/zotero/addon.js` 的 `startScheduler` / `maybeAutoRun`）：每次定时检查时，如果没有正在进行的推荐运行（`this.running`）或调研（`this.surveyTask`），就检查到期的课题（`lastCheckAt + intervalDays` 天 ≤ 现在）。每轮最多检查 3 个，按到期时间先后，逐个在后台运行。
- 推荐运行或调研开始时，正在进行的检查要能取消（复用 `createCancelSignal`）。插件停用时同样要取消。
- 课题页有「立即检查」按钮。
- 后台检查有新增、且 ZotWatch 窗口没打开时，用 `notify()` 发通知，例如「课题『矿物碳酸化』新增 5 篇」，文本走 `plainNoticeText`。
- 通过 `this.emit({ type: "tracking", action: "checked" | "updated", id })` 通知界面刷新。

### 1.5 界面

**调研页（`addon/content/survey.js`）：**
- 已完成的调研，工具栏加开关「持续追踪」和间隔选择（3 / 7 / 14 / 30 天；用分段按钮或菜单，不能用 `<select>`）。
- 调研记录列表：追踪中的课题置顶，单独分一组「追踪中的课题」。每项显示上次检查时间和「新增 N」徽标。
- 课题详情在现有「综述 / 候选文献 / 主题」之外加一栏「追踪」：
  - 顶部写「自上次查看以来新增 N 篇 · 上次检查 …」和「立即检查」按钮。打开这一栏时，更新 `lastViewedAt`，并把 `new` 改成 `seen`。
  - 按检查日期分组，新的在前。每行显示标题、期刊和年份、来源标签（「引用种子 1」「检索 2」「相关」）、得分、「已在文库 / 已收藏·未读」，以及操作「加入 Zotero」（复用 `addSurveyPaper` 的导入路径）和「忽略」。
  - 底部显示每次检查的数量（第 4 节）。
- 左侧栏「调研」旁显示所有课题的新增总数，样式和「推荐」的数字徽标一致。

**概览页（`addon/content/overview.js`）：**
- 新卡片「课题追踪」：列出各课题的名称、新增数和上次检查时间，点击进入 `#survey/<id>`。没有课题时显示空状态，说明「在调研结果中打开『持续追踪』」。
- 「研究方向」卡片：每个方向加一个操作「建为课题」，打开调研表单并预填。主题填方向名，关键词填该方向的高频标签（`src/core/topics.js` 的 `topKeywords`），种子填该方向最接近中心的条目（最多 `MAX_SEEDS` 篇）。成员和代表条目从 `clusters.json` / `cluster-labels.json` 取，实现时先确认这两个文件的字段。

## 2. 「已收藏·未读」提醒（第二优先）

**判断规则：** Zotero 没有已读状态，要从信号推断。
- **已读**（满足任一）：有子笔记；任一附件有批注；带有设置里指定的已读标签。
- **未读**：带有设置里指定的未读标签；或者以上信号都没有。

新增偏好设置（字符串类型，默认空）`library.readTag`、`library.unreadTag`，以及开关 `library.unreadHints`（默认 true），放在「设置 → 文库」。

**实现：** 在 `src/zotero/env.js` 的 `createZoteroLibrary()` 增加 `getReadState(keys)`，批量返回 `{ key: { notes, annotations, unread } }`。
- 笔记数用 `item.getNotes()`；批注数对每个附件用 `attachment.getAnnotations?.()` 求和（要兼容 Zotero 7–10，方法不存在时当作 0）。
- 每次检查或调研只查一次，结果缓存在本次运行里。

**显示位置：**
1. **课题追踪：** 检查结果命中文库、且判断为未读的条目，不进入「新增」，单独列在「追踪」栏里的「文库里还没读的」一组，写明原因，例如「已收藏、无批注，引用了种子 1」。
2. **调研候选文献表：** 筛选里加「已收藏·未读」；行内标签「已收藏·未读」。
3. 推荐页暂不做（推荐的都是不在文库里的新论文）。

## 3. 覆盖统计（调研页，第三优先）

在调研详情的「调研过程」下方加「覆盖情况」，只用已有数据，不发请求，也不调用大模型：
- **年份分布：** 候选池按 5 年分桶，用 CSS 条形显示，旁边写数字。
- **来源：** OpenAlex / Semantic Scholar / Crossref 各占多少，以及检索 / 引用 / 被引 / 推荐 / 相关各占多少（数据在 `p.via`、`p.sources` 和 `stages.expand.byVia`）。
- **主题：** 各主题的篇数（`survey.themes`）。
- **研究方法：** 只统计精读过的论文。把 `p.read.method` 用简单的关键词规则（中英文）归到「实验 / 数值模拟 / 野外观测 / 样品分析 / 综述 / 其他」，注明「粗略分类，仅统计精读的 N 篇」。
- **固定说明：** 「这是检索结果的分布，不代表领域本身的研究分布；某类研究少，可能只是检索词没覆盖到。」

## 4. 记录扩展效果（随第 1 节一起做）

- 每次课题检查和每次调研扩展，都记录 `fetched → unique → known → inLibrary → kept`（课题检查存在 `checks[].counts`，调研存在 `stages.expand`）。
- 用户的后续操作（加入 Zotero、忽略）按 `reasons` 汇总，在「追踪」栏底部显示，例如「近 8 次检查：引用种子 1 → 新增 12，采纳 4；检索 3 → 新增 20，采纳 0」。
- 这一版只展示，不自动调整种子或预算。

## 5. 以后再做（不在 0.5 范围）

- **跨课题标签：** 有 ≥2 个课题后，一篇候选和两个课题中心的相似度都高，就标「跨课题」。
- **推荐卡片「相关文献」：** 按需展开，用 OpenAlex 文献耦合加 Semantic Scholar 推荐接口；结果缓存。
- **一键转入调研：** 调研种子现在只能是文库条目（`input.seedKeys`），需要扩展 `normalizeSurveyInput` 支持 `seedDois`。
- **按效果自动调整种子和请求预算。**

## 6. 测试与验收

**测试（node:test，仿照 `test/survey.test.js` 和 `test/survey-fixtures.js` 的 mock HTTP）：**
- 课题检查：
  - URL 带 `from_publication_date`、`cites:` 和 `api_key`；
  - 和候选池、`found`、种子去重；
  - 命中文库的条目进入未读分组；
  - 相似度过滤和前 30 篇的上限；
  - 取消时不写入半截结果；
  - 某个来源失败只记进 `errors`，不让整次检查失败。
- 存储：
  - 追踪中的调研不受 30 条上限影响；
  - 删除调研时删除 `tracking-<id>.json`；
  - 并发写入不丢数据（参考已有的 overlapping-saves 测试）。
- 调度：只检查到期的课题，每轮最多 3 个；推荐运行或调研进行中不检查。
- 未读判断：在 `test/addon.test.js` 的模拟 Zotero 里，覆盖笔记、批注、已读 / 未读标签各种组合。
- 界面：「追踪」栏渲染、「新增 N」徽标、`lastViewedAt` 更新（参考 `test/survey-page.test.js`）。所有外部文本都用 `textContent`。

**验收：**
1. 一项完成的调研打开「持续追踪」后，「立即检查」能列出新增论文。每篇写明由哪个种子或哪条检索式找到；可以加入 Zotero 或忽略，再次检查不会重复出现。
2. 每 7 天自动检查一次；有新增时，概览卡片、左侧栏徽标，以及窗口关闭时的通知都能看到。
3. 文库里没有笔记和批注的命中条目，显示为「已收藏·未读」并写明原因。
4. 调研页显示覆盖情况，以及每次检查各步的数量。
5. 追踪中的课题不会因为调研记录满 30 条被删除。
6. 只请求公开 API；每个课题每次检查的请求数在 30 次以内；默认不调用大模型。
7. `npm test` 和 `npm run build` 都通过。更新 README（新增「课题追踪」说明，隐私一节写明追踪会定期发送种子的 OpenAlex id 和检索式）、CHANGELOG（`## 0.5.0`）、`CLAUDE.md` 的 Zotero Plugin 一节；版本号改为 0.5.0。

## 7. 涉及的现有代码

| 位置 | 用途 |
|---|---|
| `src/survey/engine.js` | `SurveyEngine`：`queryText`、`vectorsOf`、`stage_rank` 的中心向量算法；`createSurvey` 的字段 |
| `src/survey/sources.js` | `openAlexSearch`、`openAlexCitingWorks`、`openAlexWorksByIds`、`openAlexFilter`（要支持具体日期）、`mapOpenAlexPaper` |
| `src/survey/pool.js` | `PaperCollector`、`dedupePool`、`linkedSeeds`、来源标记 |
| `src/survey/rank.js` | `scorePapers`、`centroidOf` |
| `src/survey/store.js` | `withSurveyLock`、`writeSurvey`（30 条上限）、`indexEntry`、`deleteSurvey`、`updateSurvey` |
| `src/sources/abstracts.js` | `enrichMissingAbstracts` |
| `src/zotero/addon.js` | 调度（`startScheduler`、`maybeAutoRun`）、`runNow`、`launchSurvey` / `assertNoSurvey`、`notify`、`emit`、`listSurveys` / `getSurvey`、`addSurveyPaper`、`resetData` |
| `src/zotero/env.js` | `createZoteroLibrary`、`itemToPlain`（加未读信号） |
| `addon/content/survey.js`、`survey.css` | 调研页：工具栏、记录列表、详情各栏 |
| `addon/content/overview.js` | `card()`、「研究方向」卡片 |
| `addon/content/main.js` | 左侧栏徽标 |
| `addon/prefs.js`、`src/settings.js`、`addon/content/settings.js` | 新偏好设置及其界面 |
