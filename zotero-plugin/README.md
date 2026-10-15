# ZotWatch for Zotero（Zotero 插件版）

把 [AIZotWatch](../README.md) 的「根据文库推荐新文献」功能直接做进 Zotero：插件读取你本地的 Zotero 文库构建研究画像，每天自动从 Crossref / arXiv / EarthArXiv 抓取最新论文，按与文库的语义相关性打分排序，在 Zotero 里直接浏览、一键加入文库。

不再需要 GitHub Actions、Zotero Web API Key 或服务器——Zotero 开着就能用。

## 功能

- **本地文库画像**：直接读取 Zotero 本地数据库中的条目（无需 Zotero API Key），增量向量化——只有新增或修改过的条目才会重新计算
- **多源抓取**：Crossref（按关注期刊 ISSN）、arXiv（按分类）、EarthArXiv（OAI-PMH），摘要缺失时自动从 OpenAlex 补全
- **关注期刊自动推断**：默认取文库中最常出现的期刊作为关注列表，也可导入 `journal_whitelist.csv`（含影响因子）
- **与 Python 版一致的打分算法**：k-NN 微观相似度 + 语义聚类宏观相似度融合（Silhouette 自适应选 k），叠加期刊影响因子，动态阈值分出「必读 / 推荐 / 一般」
- **推荐收件箱**：「待查看 / 已加入 / 已忽略」三栏，已处理的论文不会重复推荐；支持筛选必读、本次新增、按分数/日期排序、搜索
- **一键加入 Zotero**：通过 DOI / arXiv 号调用 Zotero 自带的「通过标识符添加」抓取完整元数据，自动放入指定分类并打标签
- **AI 增强（可选）**：标题翻译成中文、五要点结构化 AI 摘要（研究问题 / 方法 / 发现 / 创新点 / 相关性），可保存为子笔记；支持任何 OpenAI 兼容接口（OpenRouter、DeepSeek、Kimi、通义千问…）
- **每日自动运行**：Zotero 运行时在后台按间隔检查，发现新推荐时弹出提示
- **开箱即用**：默认使用无需 API Key 的本地向量化模式；配置 DashScope / Voyage 后推荐质量更高

## 安装

支持 **Zotero 7、8、9、10**（已对照 Zotero 7.0 / 10.0 源码逐项核对所用接口）。在 Zotero 9 及以上版本中，菜单通过官方的 `Zotero.MenuManager` 注册（带图标，随界面语言显示中文或英文）；Zotero 7、8 上则直接在「工具」菜单中插入纯文本菜单项（这两个版本里插件的本地化文件会影响主窗口其他文字的语言，所以不使用）。两种方式下菜单项都会显示待查看推荐的数量。

1. 下载 `.xpi` 文件：
   - 正式版本：仓库的 [Releases](https://github.com/CrazyHalfDay/AIZotWatch/releases) 页面（`zotero-plugin-v*` 标签）
   - 最新构建：仓库 Actions 中的「Zotero Plugin」工作流 → 最近一次运行 → Artifacts → `zotwatch-xpi`（下载后先解压出 `.xpi`）
   - 或自行构建，见下文「开发」
2. Zotero → 工具 → 插件（Plugins）→ 右上角齿轮 → 「从文件安装插件…」（Install Plugin From File…）→ 选择 `.xpi`

## 快速上手

1. 安装后打开 **Zotero 设置 → ZotWatch**：
   - **向量化服务**：推荐选择「阿里云 DashScope」并填入 API Key（与 AIZotWatch 的 `DASHSCOPE_API_KEY` 相同）；不填也能用本地模式
   - **Crossref 联系邮箱**：建议填写，进入 Crossref 的礼貌池，请求更稳定
   - **关注期刊**：默认自动关注文库中最常见的 40 种期刊；也可以点「导入 CSV 文件…」导入 AIZotWatch 的 `data/journal_whitelist.csv`，或点「载入地学期刊示例」
2. 点击 **「立即运行并查看推荐」**，或使用菜单 **工具 → ZotWatch：立即检查新文献**
3. 首次运行需要向量化整个文库（几千篇文献约几分钟，取决于 API 速度），之后每天只处理新增内容
4. 在 **工具 → ZotWatch 文献推荐…** 窗口中浏览推荐，点「加入 Zotero」或「忽略」

## 设置说明

| 分组 | 设置项 | 说明 |
|------|--------|------|
| 运行 | 每天自动检查 / 检查间隔 | Zotero 启动约 3 分钟后检查一次，之后每 30 分钟判断是否到期；失败后 2 小时内不会重试 |
| 向量化 | 服务商 | `本地模式`（免费、离线、效果一般）/ `DashScope`（text-embedding-v4，1024 维）/ `Voyage`（voyage-3.5）/ `OpenAI 兼容`（可接 OpenAI、硅基流动、本地 Ollama 等） |
| 向量化 | 模型 / 接口地址 | 留空使用默认；更换服务商或模型后会自动重新向量化文库 |
| 来源 | Crossref / arXiv / EarthArXiv | 回溯天数默认 7 / 7 / 14 天；arXiv 只保留主分类匹配的论文 |
| 关注期刊 | 自动关注文库常见期刊 | 取出现次数 ≥ 2 的前 N 种期刊；文库条目缺少 ISSN 时通过 Crossref 按刊名查询（结果缓存 90 天） |
| 关注期刊 | 期刊白名单 | 与 AIZotWatch 相同的 CSV 格式：`issn,title,category,impact_factor`；影响因子参与打分。列表保存为数据目录下的 `zotwatch/journals.csv`（体积可能超过偏好设置建议的 4 KB 上限，所以不放在偏好设置里），清除缓存时会保留 |
| 推荐 | 每次推荐数量 / 最近 N 天 / 预印本占比 | 对应 Python 版的 `watch.top_k`、`watch.recent_days`、`watch.max_preprint_ratio` |
| 推荐 | 最低相似度 | 相似度门槛（对应 `fusion.min_similarity`），留空时按向量化服务自动取值（API 模式 0.20，本地模式 0.06） |
| 推荐 | 包含 / 排除关键词 | 对应 `interests.include_keywords` / `exclude_keywords`，整词匹配，中文按子串匹配 |
| AI | 接口地址 / API Key / 模型 | 任意 OpenAI 兼容的 `/chat/completions` 接口；「快速填写」可一键填入常见服务商 |
| 加入 Zotero | 分类 / 标签 / 子笔记 | 默认保存到「ZotWatch 推荐」分类并打上 `ZotWatch` 标签 |

更多高级参数（打分权重、k-NN 邻居数、时间衰减半衰期、收件箱保留天数、使用的文库 ID 等）可在 **设置 → 高级 → 配置编辑器** 中搜索 `extensions.zotwatch.` 修改。

## 工作原理

```
Zotero 文库 ──► 画像（增量向量化 + 聚类）
                        │
Crossref / arXiv / EarthArXiv ──► 去重（DOI + 模糊标题）──► 关键词过滤 ──► OpenAlex 补摘要
                        │
                        ▼
             打分：score = 0.8 × 相似度 + 0.2 × 期刊分
             相似度 = 0.65 × k-NN 微观分 + 0.35 × 聚类宏观分
                        │
                        ▼
      相似度门槛 → 动态阈值标签 → 最近 N 天 → 预印本占比 → Top K → 收件箱
```

算法细节与 AIZotWatch 的 [docs/algorithm.md](../docs/algorithm.md) 一致：聚类采用球面 k-means，按 Silhouette 系数在 `[max(5, √N/3), min(35, √N)]` 范围内选择 k（偏向更细粒度）；宏观分为 `max_k(sim_k × ln(1+E_k)) / max_k ln(1+E_k)`；动态阈值取当批分数的第 95 / 70 百分位并设下限。

为了不阻塞 Zotero 界面，聚类和打分计算会分段让出主线程（单次阻塞 < 100 ms）。点击「取消」或停用/升级插件时，本次运行中正在进行和正在等待重试的网络请求都会被立即中止，之后也不会再发出新请求；你手动发起的操作（如单篇 AI 摘要、测试连接）不受影响。

后台自动运行发现新推荐时会弹出提示。Zotero 10 的提示框依附于主窗口（不再置顶），所以如果当时 Zotero 不在前台，提示会等到你切回 Zotero 时再显示，并一直保留到点击为止。

## 数据与隐私

- 插件数据保存在 **Zotero 数据目录下的 `zotwatch/` 文件夹**（向量缓存 `profile.*`、候选缓存 `candidates.*`、聚类 `clusters.json`、推荐记录 `inbox.json`、期刊白名单 `journals.csv`）；在设置中可一键清除缓存和推荐记录，不影响文库本身和期刊白名单
- API Key 保存在 Zotero 的偏好设置（profile 目录下的 `prefs.js`，明文），请勿与他人共享该目录
- 发送到外部服务的内容：
  - 向量化服务：文库条目和候选论文的**标题 + 摘要**（本地模式不发送）
  - 大模型（仅启用时）：推荐论文的标题、摘要、作者和期刊名
  - Crossref / arXiv / EarthArXiv / OpenAlex：只有查询条件（ISSN、分类、日期、DOI），不含文库内容
- 所有网络请求都通过 `Zotero.HTTP` 发出，遵循 Zotero 的代理设置；在 Zotero 9 及以上版本中请求不携带 Zotero 的 Cookie（匿名请求）
- 填写的 Crossref 联系邮箱会作为 `mailto` 参数出现在请求网址中，因此也会出现在 Zotero 的调试日志里；公开提交调试日志前请留意

## 与 Python 版 AIZotWatch 的对应关系

| AIZotWatch（Python） | 插件 |
|----------------------|------|
| `pipeline/ingest.py`（Zotero Web API） | 直接读取本地文库（`src/zotero/env.js`） |
| `pipeline/profile.py` + `profile_clusterer.py`（FAISS） | `src/engine.js` + `src/core/kmeans.js`（纯 JS） |
| `pipeline/profile_ranker.py` + `cluster_scorer.py` + `journal_scorer.py` | `src/core/scoring.js` |
| `sources/crossref.py` / `arxiv.py` / `eartharxiv.py` | `src/sources/*.js` |
| `pipeline/dedupe.py`（rapidfuzz） | `src/core/dedupe.js`（等价的 token_set_ratio） |
| `pipeline/filters.py` | `src/core/filters.js` |
| `llm/summarizer.py` / `translator.py` | `src/llm/llm.js`（相同的摘要提示词） |
| RSS / HTML 报告 / Server酱推送 | Zotero 内的推荐窗口 + 桌面通知 |
| `watch --push`（推送到 Zotero） | 「加入 Zotero」按钮 |

尚未移植：顶刊地学速览（flagship track）、关注作者（ORCID）、出版商网页摘要抓取（Camoufox）、研究兴趣重排序（rerank）、领域分类与总体综述。插件与 Python 版的缓存互不共享，可以同时使用。

## 开发

```bash
cd zotero-plugin
npm install        # 只有 esbuild 一个开发依赖
npm test           # 单元测试 + 模拟 Zotero 环境的完整生命周期测试
npm run build      # 生成 build/zotwatch-<version>.xpi 和 build/zotwatch-update.json
```

目录结构：

```
zotero-plugin/
├── addon/                  # 插件静态文件（打包进 .xpi）
│   ├── manifest.json       # 插件 ID、版本、兼容的 Zotero 版本
│   ├── bootstrap.js        # 生命周期入口，注册 chrome://zotwatch/ 并加载打包后的代码
│   ├── prefs.js            # 偏好设置默认值
│   └── content/            # 设置面板、推荐窗口（xhtml/js/css）、图标、示例期刊表
├── src/                    # 业务代码（esbuild 打包为 content/zotwatch.js）
│   ├── core/               # 纯算法：文本、XML、向量、k-means、打分、过滤、去重、期刊表
│   ├── sources/            # Crossref、arXiv、EarthArXiv、OpenAlex
│   ├── embedding/          # DashScope / Voyage / OpenAI 兼容 / 本地 TF-IDF
│   ├── llm/                # OpenAI 兼容对话接口、标题翻译、AI 摘要
│   ├── engine.js           # 完整流水线（不依赖 Zotero，可在 Node 中测试）
│   ├── settings.js         # 偏好设置 → 配置对象
│   └── zotero/             # Zotero 适配层：HTTP、文件存储、文库读取、条目创建、菜单与调度
├── test/                   # node:test 测试
└── scripts/build.mjs       # 打包 + 生成 .xpi（零依赖 zip 实现）
```

`src/engine.js` 的所有外部依赖（HTTP、存储、文库）都通过参数注入，因此核心流程可以完全在 Node 中测试；`test/addon.test.js` 则用模拟的 Zotero API 加载真正打包出来的插件，分别按 Zotero 10（MenuManager、设置面板脚本运行在独立沙箱）、Zotero 8 和 Zotero 7（直接插入菜单）走完「启动 → 注册设置面板 → 添加菜单 → 运行 → 取消 → 加入文库 → 关闭」的完整流程，并验证关闭插件时会先等待正在进行的运行结束。

### 发布

1. 修改 `package.json` 中的 `version`
2. 在 `CHANGELOG.md` 中新增 `## <version>` 一节（会作为 Release 说明，缺少时发布会失败）
3. 推送标签 `zotero-plugin-v<version>`（如 `zotero-plugin-v0.1.0`）
4. GitHub Actions 会运行测试、构建并创建 Release（标记为 latest），附带 `.xpi` 与 `zotwatch-update.json`；已安装的插件会通过 `manifest.json` 中的 `update_url` 自动检查更新

兼容的 Zotero 版本范围由 `addon/manifest.json` 的 `strict_min_version` / `strict_max_version`（当前 `10.*`）决定，构建时会同步写入 `zotwatch-update.json`（Zotero 会把更新清单里的范围当作兼容性更新应用，两处必须一致）。新的 Zotero 大版本发布并测试通过后，只需提高 manifest 中的上限。
