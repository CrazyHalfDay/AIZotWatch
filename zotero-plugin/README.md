# ZotWatch for Zotero（Zotero 插件版）

把 [AIZotWatch](../README.md) 的「根据文库推荐新文献」功能直接做进 Zotero：插件读取你本地的 Zotero 文库构建研究画像，每天自动从 Crossref / arXiv / EarthArXiv 抓取最新论文，按与文库的语义相关性打分排序，在 Zotero 里直接浏览、一键加入文库。

不再需要 GitHub Actions、Zotero Web API Key 或服务器——Zotero 开着就能用。

## 功能

- **本地文库画像**：直接读取 Zotero 本地数据库中的条目（无需 Zotero API Key），增量向量化——只有新增或修改过的条目才会重新计算
- **多源抓取**：Crossref（按关注期刊 ISSN）、arXiv（按分类）、EarthArXiv（OAI-PMH）
- **摘要补全**：论文缺少摘要时（很多出版商不向 Crossref 提交摘要），按 DOI 依次从 Crossref、Elsevier、Springer Nature（后两者需填写免费的 API Key）和 OpenAlex 查找，结果缓存在本地；关注作者的论文同样会补全
- **关注期刊自动推断**：默认取文库中最常出现的期刊作为关注列表，也可导入 `journal_whitelist.csv`（含影响因子）
- **与 Python 版一致的打分算法**：k-NN 微观相似度 + 语义聚类宏观相似度融合（Silhouette 自适应选 k），叠加期刊影响因子，动态阈值分出「必读 / 推荐 / 一般」
- **顶刊地学速览**：Nature、Science、PNAS、Nature Communications、Science Advances 等综合性顶刊单独抓取，不按文库相似度筛选，而是用语义「领域锚点」挑出其中的固体地球科学与古生物文章（介于两可之间的可交给大模型复核），单独列出（每次取最新发表的 N 篇）
- **关注作者**：按 ORCID（或 OpenAlex 作者 ID）追踪指定研究者的新论文。这些论文不参与打分、只与文库去重，每篇只出现一次；与个性推荐重合时不重复显示，而是在那条推荐上标注「关注：作者名」
- **概览仪表盘**：待查看篇数、本次新增与必读、最近 14 次运行的新增推荐、筛选漏斗、各数据源状态、研究方向和关注作者一目了然
- **推荐收件箱**：「待查看 / 已加入 / 已忽略」三栏，已处理的论文不会重复推荐；按来源分区显示（个性推荐、顶刊地学、关注作者），支持按来源筛选、筛选必读、本次新增、按分数/日期排序、搜索
- **一键加入 Zotero**：通过 DOI / arXiv 号调用 Zotero 自带的「通过标识符添加」抓取完整元数据，自动放入指定分类并打标签
- **AI 增强（可选）**：标题翻译成中文、五要点结构化 AI 摘要（研究问题 / 方法 / 发现 / 创新点 / 相关性），可保存为子笔记；支持任何 OpenAI 兼容接口（OpenRouter、DeepSeek、Kimi、通义千问…）
- **研究方向标签（需大模型）**：文库语义聚类得到的每个研究方向由大模型根据代表性标题和你的 Zotero 标签起一个简短的中文名称，每篇推荐（含顶刊地学文章）标注它最接近的研究方向「方向：…」；标签只生成一次，之后按方向成员沿用，文库变化后只为新出现的方向补充
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

1. 安装后打开 **工具 → ZotWatch 文献推荐…**（或 Zotero 设置 → ZotWatch → 「打开设置」），在 ZotWatch 窗口左侧点 **设置**：
   - **向量化**：推荐选择「阿里云 DashScope」并填入 API Key（与 AIZotWatch 的 `DASHSCOPE_API_KEY` 相同），点「测试向量化」确认可用；不填也能用本地模式
   - **文献来源**：建议填写 Crossref 联系邮箱，进入 Crossref 的礼貌池，请求更稳定
   - **关注期刊**：默认自动关注文库中最常见的 40 种期刊；也可以点「导入 CSV…」导入 AIZotWatch 的 `data/journal_whitelist.csv`，或点「载入地学期刊示例」
2. 点窗口右上角的 **「立即检查」**，或使用菜单 **工具 → ZotWatch：立即检查新文献**
3. 首次运行需要向量化整个文库（几千篇文献约几分钟，取决于 API 速度），之后每天只处理新增内容
4. 在 ZotWatch 窗口的 **推荐** 页浏览推荐，点「加入 Zotero」或「忽略」

## 概览

打开 ZotWatch 窗口时默认显示 **概览**（左侧栏「概览」，快捷键 Ctrl/⌘+1），汇总文库和最近几次检查的情况：

![ZotWatch 概览](docs/overview.png)

- **待查看**：推荐页中还没处理的篇数，按个性推荐 / 顶刊地学 / 关注作者细分，并标出其中的必读篇数；旁边四项指标为上次检查新增的推荐（按来源细分）、其中的必读、已加入 Zotero 的篇数（已处理的记录至少保留 60 天）和上次检查的时间与耗时
- **最近运行**：最近 14 次检查每次新增的推荐，按来源堆叠（蓝 = 个性推荐，橙 = 顶刊地学，绿 = 关注作者）
- **运行状态**：上次检查的结果（正常 / 有几个问题 / 运行失败）、下次自动检查的时间，以及上次运行中出错的来源和原因
- **筛选漏斗**：上次运行个性推荐每一步筛选后剩下的篇数（抓取 → 去重 → 关键词 → 有摘要 → 相似度门槛 → 最近 N 天 → 推荐），开启顶刊地学时另列它的候选、通过（灰区、大模型复核）和显示篇数，便于判断门槛是否过严
- **数据源**：各来源是否启用、上次运行是否出错以及抓取的篇数，还有摘要补全、向量化服务和大模型的状态
- **研究方向 / 关注作者**：文库语义聚类得到的最大 10 个研究方向（需开启大模型生成名称）；关注的作者及其 ORCID / OpenAlex ID、各自待查看的论文数，无法识别的行会单独提示

图表上悬停鼠标，或按 Tab 进入图表后用方向键切换，可以看到每一项的具体数值；每张图右上角的「表格」按钮把图表换成同样数字的表格。页面右上角的「测试连接」逐项检查向量化、大模型、Crossref 和已填写的出版商 API Key。

## 推荐

左侧栏 **推荐**（快捷键 Ctrl/⌘+2）是推荐收件箱，每篇论文一张卡片：

![ZotWatch 推荐](docs/inbox.png)

- 顶部按「待查看 / 已加入 / 已忽略」切换，标签旁是篇数；「全部」视图按来源分区（个性推荐、顶刊地学速览、关注作者），也可以只看某个来源、只看必读或本次检查新增的论文，按分数或发表日期排序，或在右上角搜索标题、作者、期刊和研究方向（Ctrl/⌘+F）
- 卡片上方是推荐等级（「必读」「推荐」「一般」，顶刊和关注作者的论文显示来源标记）、期刊与发表日期、影响因子和分数；下面是中文翻译标题、作者、关注作者与研究方向标记，摘要较长时点击展开
- 「加入 Zotero」按设置中的分类和标签保存（已生成的 AI 摘要可一并保存为子笔记），「忽略」移到已忽略；未生成摘要的论文可点「AI 摘要」单独生成；「打开原文」在浏览器中打开论文页面。已加入的论文可「在 Zotero 中显示」，已处理的论文可移回待查看
- 「全部忽略」忽略当前筛选下显示的所有待查看论文；上次运行有问题时，列表上方会列出出错的来源和原因

## 设置说明

所有设置都在 ZotWatch 窗口的 **设置** 页中（左侧栏「设置」，快捷键 Ctrl/⌘+3；也可以从 Zotero 设置 → ZotWatch → 「打开设置」进入，那里只保留状态摘要和「打开 ZotWatch」「打开设置」「立即检查」三个按钮）。设置首页按分区列出各项的当前状态（如「DashScope · 已配置」「未启用」），点开一个分区即可修改：

- 修改**自动保存**：开关和选项立即生效，文本框在停止输入约 0.6 秒后或离开输入框时保存，切换页面或关闭窗口前也会保存；改动在下次检查时生效
- 数值框旁标出单位和允许的范围，超出范围或不是数字时下方会提示实际使用的值（如「超出范围，将按 365 使用」）；留空的阈值显示按当前向量化服务自动取的值（如「自动（0.20）」）
- API Key 默认隐藏，点输入框右侧的眼睛图标可以查看；「向量化」「文献来源 → 摘要补全」「AI 助手」中的测试按钮会先保存刚输入的内容，再把检测结果显示在按钮旁

![ZotWatch 设置：向量化](docs/settings.png)

| 分区 | 设置项 | 说明 |
|------|--------|------|
| 常规 | 自动检查新文献 / 检查间隔 | Zotero 启动约 3 分钟后检查一次，之后每 30 分钟判断是否到期；失败后 2 小时内不会重试。间隔 1–720 小时（默认 24）；分区中同时显示上次和下次检查的时间 |
| 常规 | 推荐保留天数 | 待查看的推荐超过 N 天（默认 30，1–365）没有再次出现就会移除；已加入或已忽略的记录至少保留 60 天，以免重复推荐 |
| 向量化 | 服务商 | `本地模式`（免费、离线、效果一般）/ `阿里云 DashScope`（推荐，text-embedding-v4，1024 维）/ `Voyage AI`（voyage-3.5）/ `OpenAI 兼容接口`（可接 OpenAI、硅基流动、本地 Ollama 等）；只显示所选服务需要的输入项 |
| 向量化 | API Key / 模型 / 接口地址 / 向量维度 | 留空使用默认（输入框中显示默认模型；接口地址只用于 OpenAI 兼容接口，向量维度在「高级选项」中，只用于 DashScope）；更换服务商、模型或维度后会自动重新向量化文库。「测试向量化」用当前设置向量化一段测试文本 |
| 文献来源 | Crossref / arXiv / EarthArXiv | 每个来源一张卡片，右上角开关启用。回溯天数默认 7 / 7 / 14 天（Crossref 可改，1–60）；arXiv 按分类抓取，只保留主分类匹配的论文 |
| 文献来源 | 摘要补全 | 缺摘要的论文按 DOI 依次查询 Crossref（只用于预印本等不是从 Crossref 抓到的论文）→ Elsevier（`10.1016/`，需 API Key）→ Springer Nature（`10.1007/`、`10.1038/`，需 API Key）→ OpenAlex，取第一个不少于 80 个字符的摘要。找到的摘要缓存 30 天；各来源都明确没有的 DOI 7 天内不再查询，发表不到两周的新论文则只隔约一天（OpenAlex 等收录新论文需要几天）；网络错误、限流和无法解析的响应不算，新填 API Key 后也会重新查询。API Key 无效（HTTP 401）或某个来源连续出错时本次运行停用该来源，并在推荐页提示；「测试 API Key」会检查已填写的 Key。Key 申请：[dev.elsevier.com](https://dev.elsevier.com) / [dev.springernature.com](https://dev.springernature.com) |
| 关注期刊 | 期刊列表 | 与 AIZotWatch 相同的 CSV 格式：`issn,title,category,impact_factor`；影响因子参与打分（权重 20%），未列出的期刊按默认分计算。输入时上方实时显示识别到的期刊数和有影响因子的期刊数，第一列不是 ISSN 的行会单独提示。「从文库生成」把文库常见期刊合并进列表，「导入 CSV…」和「载入地学期刊示例」替换整个列表（列表非空时会先确认）。列表保存为数据目录下的 `zotwatch/journals.csv`（体积可能超过偏好设置建议的 4 KB 上限，所以不放在偏好设置里），清除缓存时会保留 |
| 关注期刊 | 自动关注文库中最常见的期刊 | 取出现次数 ≥ 2 的前 N 种期刊（默认 40，1–300），与期刊列表合并；文库条目缺少 ISSN 时通过 Crossref 按刊名查询（结果缓存 90 天） |
| 关注作者 | 作者列表 / 检查最近 N 天 | 每行一位：姓名 + ORCID（如 `汪在聪, 0000-0002-3584-1673`，也可粘贴 `https://orcid.org/…` 链接）或 OpenAlex 作者 ID（如 `A5023888391`），以 `#` 开头的行会被忽略；输入时下方实时列出识别到的作者及其 ORCID / OpenAlex ID，无法识别的行标为「无法识别」。每次运行在 OpenAlex 中查询这些作者最近 N 天（默认 60 天，7–365）发表的论文，已出现过的不会再推荐。这些论文不参与评分，不受关键词、摘要和「最近 N 天」过滤影响，只与文库去重；首次开启时会把最近 N 天的论文一次性列出。OpenAlex 的列表只给出论文的前 100 位作者，作者更多的论文会再查一次完整作者名单，以确定是哪位关注作者（查不到时不标注作者名） |
| 顶刊地学 | 开启 / 顶刊 ISSN | 默认关闭。开启后这些期刊（默认 12 个 ISSN：Nature、Science、Nature Communications、Science Advances、Communications Earth & Environment、PNAS、Nature Geoscience、Nature Reviews Earth & Environment 的印刷版与电子版）用单独的 Crossref 查询抓取（回溯天数与 Crossref 相同），并有自己的抓取上限（与关注期刊各自最多 3000 篇，即配置编辑器中的 `sources.crossref.maxResults`），发文量很大的综合期刊不会挤占关注期刊的名额；即使关闭了 Crossref 来源也会抓取。期刊的任一 ISSN 命中即算顶刊文章，它们不再进入个性推荐，也不受关键词、相似度门槛、预印本占比和推荐数量限制，但必须有摘要（会先经过摘要补全）且在「最近 N 天」内发表 |
| 顶刊地学 | 门控与阈值 | 每篇文章的摘要向量与「地学领域描述」（正向锚点）和每条「排除领域」（大气、材料、化学、生物、天文、工程）比较：离任一排除领域不比正向锚点远的直接排除；与正向锚点的相似度 ≥ 接受阈值的保留，< 灰区下限的排除，两者之间为灰区（灰区下限高于接受阈值时按接受阈值使用，输入框下方会提示）。阈值留空时自动取值，输入框中显示当前的自动值：API 模式 0.30 / 0.22（Python 版针对 DashScope text-embedding-v4 调校的值），本地模式 0.12 / 0.04（本地模式只比较字面词汇，与描述用词不同的地学文章几乎都会落在灰区以下：开启大模型复核时，除明显更接近某个排除领域的文章外，所有未直接保留的文章都交给大模型判断，请求会多一些；不复核时这类文章会被漏掉，建议改用 DashScope；本地模式下分区中会显示这段提示）。Voyage 的锚点按查询（query）向量化，与 Python 版一致 |
| 顶刊地学 | 大模型复核 / 每次最多显示 | 灰区文章按「大模型复核标准」（与 Python 版 `llm_boundary` 相同）每 20 篇一批交给大模型判断；未启用大模型或关闭复核时灰区全部保留。大模型请求失败或回复无法解析时保留该批的灰区文章（本地模式下额外交给大模型的文章不保留）并在推荐页提示，不影响本次运行。每次最多保留最新发表的 N 篇（默认 30，0 为不限），与其他分区一样按所选方式排序；卡片上显示与文库的相似度（仅供参考，与子笔记中的数值一致），悬停徽章可看到地学相关度和是否经大模型复核。ISSN、锚点和复核标准默认显示内置的默认值，可直接修改；清空或点「恢复默认」时恢复默认值（与默认值相同的内容不会被单独保存，以后版本更新默认值时会跟着更新） |
| 推荐与过滤 | 每次推荐数量 / 最近 N 天 / 预印本占比 | 对应 Python 版的 `watch.top_k`、`watch.recent_days`、`watch.max_preprint_ratio`（默认 30 / 7 / 0.7；最近 N 天填 0 为不限） |
| 推荐与过滤 | 最低相似度 | 相似度门槛（对应 `fusion.min_similarity`），留空时按向量化服务自动取值（API 模式 0.20，本地模式 0.06） |
| 推荐与过滤 | 包含 / 排除关键词 | 对应 `interests.include_keywords` / `exclude_keywords`，整词匹配，中文按子串匹配 |
| 推荐与过滤 | 阈值模式 / 时间衰减半衰期（高级选项） | 阈值模式：`动态`（默认）按每批分数的第 95 / 70 百分位划分「必读 / 推荐」并设下限，`固定` 使用 0.75 / 0.55。半衰期大于 0 时，较早加入文库的条目在研究画像中的权重按半衰期递减（对应 Python 版的时间衰减），默认 0 不衰减；修改后会重新聚类 |
| AI 助手 | 接口地址 / API Key / 模型 | 任意 OpenAI 兼容的 `/chat/completions` 接口；「快速填写」可一键填入 OpenRouter、DeepSeek、Kimi、阿里云百炼（通义千问）的接口地址和模型，当前使用的服务商会高亮。「测试大模型」发送一条测试消息并显示回复 |
| AI 助手 | 将英文标题翻译为中文 / 每次自动生成摘要的篇数 | 标题翻译覆盖本次所有新增论文；AI 摘要在个性推荐中取分数最高的 N 篇（默认 10，0–100），顶刊地学和关注作者各另取最新的 N 篇 |
| AI 助手 | 为文库研究方向生成主题标签 | 默认开启（启用大模型并填写 API Key 后生效）。每个研究方向（文库聚类）把最接近中心的 3 个标题和最常用的 5 个 Zotero 标签（不含 `ZotWatch` 输出标签、以 `/` 或 `_` 开头的工作流标签和 ⭐、✔️ 这类符号标签）发给大模型，一次请求为所有未命名的方向起名（Python 版的批量提示词）。重新聚类后，成员重合度（Jaccard）≥ 0.5 的方向沿用原来的标签，只为新方向补充；回复格式不对时不保存，下次运行重试，并在推荐页提示 |
| 加入 Zotero | 分类 / 标签 / 元数据 / 子笔记 | 默认保存到「ZotWatch 推荐」分类（不存在时自动创建）并打上 `ZotWatch` 标签，留空则不放入分类 / 不加标签；默认通过 DOI / arXiv 号抓取完整元数据（与「通过标识符添加条目」相同），并把已生成的 AI 摘要保存为子笔记 |
| 数据与维护 | 数据目录 / 清除缓存 / 版本 | 显示数据目录（可在文件管理器中打开）、「清除缓存与推荐记录…」（见下文「数据与隐私」）、插件版本和本说明的链接 |

更多高级参数（打分权重、k-NN 邻居数、各来源的抓取上限、使用的文库 ID 等）可在 **Zotero 设置 → 高级 → 配置编辑器** 中搜索 `extensions.zotwatch.` 修改。

## 工作原理

```
Zotero 文库 ──► 画像（增量向量化 + 聚类 + 研究方向标签）
                        │
Crossref / arXiv / EarthArXiv ──► 去重（DOI + 模糊标题）──► 补摘要 ──► 关键词过滤
                        │
                        ▼
             打分：score = 0.8 × 相似度 + 0.2 × 期刊分
             相似度 = 0.65 × k-NN 微观分 + 0.35 × 聚类宏观分
                        │
                        ▼
      相似度门槛 → 动态阈值标签 → 最近 N 天 → 预印本占比 → Top K ──► 个性推荐 ──────────┐
                                                                                        │
Crossref（顶刊 ISSN，单独查询）──► 去重 ──► 补摘要 ──► 有摘要 + 最近 N 天                │
      ──► 领域锚点门控（灰区交给大模型）──► 新→旧，最多 N 篇 ──► 顶刊地学速览 ──────────┼──► 收件箱
                                                                                        │
OpenAlex（关注作者的 ORCID）──► 去掉已出现过的 ──► 与文库去重 ──► 补摘要 ──► 关注作者 ──┘
```

「补摘要」对本次所有缺摘要的论文（候选论文和关注作者的新论文一起）按 DOI 依次查询 Crossref → Elsevier → Springer Nature → OpenAlex，取第一个足够长的摘要；它在关键词过滤之前进行（与 Python 版相同），所以关键词也能匹配补全的摘要。

顶刊地学速览与 Python 版的 `scoring.flagship` 相同：顶刊文章在去重后立即离开个性推荐流程，先补摘要、去掉无摘要和不在「最近 N 天」内的文章（Python 版在门控之后才按时间过滤，结果相同，但这里省下向量化和大模型调用），再与领域锚点比较。锚点每次运行重新向量化（一次很小的请求，不写入候选缓存）；文章向量与个性推荐共用候选缓存。入选文章照常计算与文库的相似度和期刊分，但只用于显示，不设门槛、不分「必读 / 推荐」。大模型复核用 Python 版的相关性提示词，只是把论文编号改为 P1、P2…（比让模型照抄 DOI 更可靠）；与 Python 版不同，复核失败不会中断运行，而是保留该批文章。

研究方向标签对应 Python 版的 `profile.clustering.generate_labels`：聚类按方向大小从大到小排列，每个方向的代表性标题按与成员均值（k-means 中心）的相似度选取，关键词是成员最常用的 Zotero 标签。与 Python 版不同，标签保存在独立的 `cluster-labels.json` 中，按方向成员（文库条目）匹配而不是随聚类结果一起清空，所以文库增删几篇论文、或本地模式每次重新聚类，都不会重新生成全部标签；大模型的回复必须是与方向数量相同的 JSON 字符串数组，否则不保存（Python 版会把「Cluster N」或解析错的行当作标签永久保存）。每篇论文的研究方向取中心最接近的方向（Python 版计算的是偏向大方向的宏观最佳方向，且没有在报告中使用）；新方向的标签在本次推荐写入收件箱之后生成，随后补到这些论文上。

关注作者的论文与个性推荐是同一篇时只保留个性推荐，并标注「关注：作者名」；已在收件箱中的论文只补上标注，不改变它的分数和状态。之后如果一篇「关注作者」论文出现在个性推荐的结果里，它会改为显示分数，但保留关注标注。

算法细节与 AIZotWatch 的 [docs/algorithm.md](../docs/algorithm.md) 一致：聚类采用球面 k-means，按 Silhouette 系数在 `[max(5, √N/3), min(35, √N)]` 范围内选择 k（偏向更细粒度）；宏观分为 `max_k(sim_k × ln(1+E_k)) / max_k ln(1+E_k)`；动态阈值取当批分数的第 95 / 70 百分位并设下限。

为了不阻塞 Zotero 界面，聚类和打分计算会分段让出主线程（单次阻塞 < 100 ms）。点击「取消」或停用/升级插件时，本次运行中正在进行和正在等待重试的网络请求都会被立即中止，之后也不会再发出新请求；你手动发起的操作（如单篇 AI 摘要、测试连接）不受影响。

后台自动运行发现新推荐时会弹出提示。Zotero 10 的提示框依附于主窗口（不再置顶），所以如果当时 Zotero 不在前台，提示会等到你切回 Zotero 时再显示，并一直保留到点击为止。

## 数据与隐私

- 插件数据保存在 **Zotero 数据目录下的 `zotwatch/` 文件夹**（向量缓存 `profile.*`、候选缓存 `candidates.*`、聚类 `clusters.json`、推荐记录 `inbox.json`、研究方向标签 `cluster-labels.json`、关注作者已出现过的论文 `followed.json`、摘要缓存 `abstracts-cache.json`、期刊 ISSN 查询缓存 `journals-cache.json`、期刊白名单 `journals.csv`）；在「设置 → 数据与维护」中可一键清除缓存和推荐记录，不影响文库本身和期刊白名单（清除后，关注作者最近 N 天的论文会再列出一次，研究方向标签也会重新生成）
- API Key 保存在 Zotero 的偏好设置（profile 目录下的 `prefs.js`，明文），请勿与他人共享该目录
- 发送到外部服务的内容：
  - 向量化服务：文库条目和候选论文的**标题 + 摘要**，开启顶刊地学速览时还有领域锚点文本（本地模式不发送）
  - 大模型（仅启用时）：推荐论文（包括顶刊地学和关注作者的论文）的标题、摘要、作者和期刊名；顶刊灰区复核时发送这些文章的标题和摘要以及复核标准；生成研究方向标签时发送**文库中**每个方向最多 3 个代表性标题和 5 个 Zotero 标签以及方向的篇数（可在设置中关闭）
  - Crossref / arXiv / EarthArXiv / OpenAlex：只有查询条件（ISSN、文库常见期刊缺 ISSN 时的刊名（发给 Crossref，用于查 ISSN）、分类、日期、DOI、关注作者的 ORCID / OpenAlex 作者 ID），不含文库条目的标题、摘要等内容
  - Elsevier / Springer Nature（仅在填写 API Key 时）：缺摘要论文的 DOI 和你的 API Key，不含文库内容
- 所有网络请求都通过 `Zotero.HTTP` 发出，遵循 Zotero 的代理设置；在 Zotero 9 及以上版本中请求不携带 Zotero 的 Cookie（匿名请求）
- 填写的 Crossref 联系邮箱会作为 `mailto` 参数出现在请求网址中，Springer Nature 的 API Key 也作为网址参数发送（Elsevier 的 Key 放在请求头中），因此可能出现在 Zotero 的调试日志里；公开提交调试日志前请留意

## 与 Python 版 AIZotWatch 的对应关系

| AIZotWatch（Python） | 插件 |
|----------------------|------|
| `pipeline/ingest.py`（Zotero Web API） | 直接读取本地文库（`src/zotero/env.js`） |
| `pipeline/profile.py` + `profile_clusterer.py`（FAISS） | `src/engine.js` + `src/core/kmeans.js`（纯 JS） |
| `pipeline/profile_ranker.py` + `cluster_scorer.py` + `journal_scorer.py` | `src/core/scoring.js` |
| `sources/crossref.py` / `arxiv.py` / `eartharxiv.py` | `src/sources/*.js` |
| `sources/openalex.py`（`followed_authors`，关注作者） | `src/sources/followed.js`；按 ORCID / OpenAlex 作者 ID 合并查询，每次查最近 N 天并用 `followed.json` 记录已出现的论文（不再首次拉取全部历史论文，也不会因某次运行失败而漏掉论文），跳过数据集、勘误等非论文类型 |
| `pipeline/flagship_filter.py` + `watch.py` 中的 `_select_flagship`（顶刊地学速览） | `src/core/flagship.js`（ISSN 匹配与锚点门控）+ `src/engine.js` 的 `selectFlagship` + `src/llm/llm.js` 的 `judgeRelevance`；相同的锚点、阈值和复核提示词（论文编号改为 P1、P2…），顶刊用单独的 Crossref 查询和抓取上限，ISSN 统一格式后匹配，按向量化服务自动校准阈值，大模型失败时保留灰区文章而不中断运行，AI 摘要每次取最新的 N 篇 |
| `infrastructure/enrichment/api_fallback.py` + `pipeline/enrich.py`（API 摘要补全） | `src/sources/abstracts.js`；相同的 Crossref → Elsevier → Springer Nature → OpenAlex 顺序和 80 字符下限，OpenAlex 每 50 个 DOI 合并查询；缓存为 `abstracts-cache.json`，网络错误和限流不会被记为「没有摘要」，API Key 无效时本次运行停用该来源并提示；在去重之后进行，不为文库中已有的论文浪费请求 |
| `pipeline/dedupe.py`（rapidfuzz） | `src/core/dedupe.js`（等价的 token_set_ratio） |
| `pipeline/filters.py` | `src/core/filters.js` |
| `llm/summarizer.py` / `translator.py` | `src/llm/llm.js`（相同的摘要提示词） |
| `llm/cluster_labeler.py` + `profile_clusterer.py` 中的代表性标题和关键词（研究方向标签） | `src/core/scoring.js` 的 `buildClusters`（成员、代表性标题、标签关键词，按大小排序）+ `src/core/topics.js`（标签筛选与按成员 Jaccard 沿用）+ `src/llm/llm.js` 的 `labelClusters`（相同的批量提示词，只接受数量正确的 JSON 数组）；标签存于 `cluster-labels.json`，重新聚类时不清空，每篇推荐标注中心最接近的方向 |
| RSS / HTML 报告 / Server酱推送 | Zotero 内的 ZotWatch 窗口（概览 + 推荐）+ 桌面通知 |
| `watch --push`（推送到 Zotero） | 「加入 Zotero」按钮 |

尚未移植：出版商网页摘要抓取（Camoufox）、研究兴趣重排序（rerank）、领域分类与总体综述、HTML 报告中的研究兴趣画像（文库分析与研究方向聚类图谱）。插件与 Python 版的缓存互不共享，可以同时使用。

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
│   └── content/            # ZotWatch 主窗口（main.xhtml：概览 / 推荐 / 设置）、Zotero 设置中的状态面板、图标、示例期刊表
├── src/                    # 业务代码（esbuild 打包为 content/zotwatch.js）
│   ├── core/               # 纯算法：文本、XML、向量、k-means、打分、过滤、去重、期刊表、作者解析、推荐分区、顶刊门控、研究方向标签
│   ├── sources/            # Crossref、arXiv、EarthArXiv、OpenAlex（关注作者）、摘要补全链
│   ├── embedding/          # DashScope / Voyage / OpenAI 兼容 / 本地 TF-IDF
│   ├── llm/                # OpenAI 兼容对话接口、标题翻译、AI 摘要、顶刊灰区复核、研究方向标签
│   ├── engine.js           # 完整流水线（不依赖 Zotero，可在 Node 中测试）
│   ├── settings.js         # 偏好设置 → 配置对象
│   └── zotero/             # Zotero 适配层：HTTP、文件存储、文库读取、条目创建、菜单与调度
├── docs/                   # README 中的界面截图（由 scripts/preview.mjs 生成）
├── test/                   # node:test 测试
└── scripts/
    ├── build.mjs           # 打包 + 生成 .xpi（零依赖 zip 实现）
    └── preview.mjs         # 界面预览（开发用，不参与打包和 CI）
```

界面预览：`npm run build && node scripts/preview.mjs <输出目录>` 用示例数据加载主窗口，截取各页面的浅色 / 深色截图（1120×820 与 820×700；需要全局安装的 Playwright，选项见脚本开头的注释）。

`src/engine.js` 的所有外部依赖（HTTP、存储、文库）都通过参数注入，因此核心流程可以完全在 Node 中测试；`test/addon.test.js` 则用模拟的 Zotero API 加载真正打包出来的插件，分别按 Zotero 10（MenuManager、设置面板脚本运行在独立沙箱）、Zotero 8 和 Zotero 7（直接插入菜单）走完「启动 → 注册设置面板 → 添加菜单 → 运行 → 取消 → 加入文库 → 关闭」的完整流程，并验证关闭插件时会先等待正在进行的运行结束。

### 发布

1. 修改 `package.json` 中的 `version`
2. 在 `CHANGELOG.md` 中新增 `## <version>` 一节（会作为 Release 说明，缺少时发布会失败）
3. 推送标签 `zotero-plugin-v<version>`（如 `zotero-plugin-v0.1.0`）；或者在 GitHub 的 Actions 页面手动运行「Zotero Plugin」工作流并勾选 `publish`，由工作流自动创建该标签
4. GitHub Actions 会运行测试、构建并创建 Release（标记为 latest），附带 `.xpi` 与 `zotwatch-update.json`；已安装的插件会通过 `manifest.json` 中的 `update_url` 自动检查更新

兼容的 Zotero 版本范围由 `addon/manifest.json` 的 `strict_min_version` / `strict_max_version`（当前 `10.*`）决定，构建时会同步写入 `zotwatch-update.json`（Zotero 会把更新清单里的范围当作兼容性更新应用，两处必须一致）。新的 Zotero 大版本发布并测试通过后，只需提高 manifest 中的上限。
