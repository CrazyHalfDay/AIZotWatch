# ZotWatch

ZotWatch 是一个基于 Zotero 文库构建个人研究兴趣画像，并持续监测学术信息源的智能文献推荐系统。支持 AI 摘要生成、领域分类、作者追踪、增量嵌入计算，可在本地手动执行或通过 GitHub Actions 自动运行。

## 致谢

本项目受 [Yorks0n/ZotWatch](https://github.com/Yorks0n/ZotWatch) 启发，并在此基础上进行了大量扩展和优化。

## 部署示例

- **在线演示**: [https://ehehe.cn/zotwatch/](https://ehehe.cn/zotwatch/)
- **文献归档**: [https://ehehe.cn/zotwatch/archive.html](https://ehehe.cn/zotwatch/archive.html)
- **RSS 订阅**: [https://ehehe.cn/zotwatch/feed.xml](https://ehehe.cn/zotwatch/feed.xml)

## 功能概览

### 核心功能

- **Zotero 同步**：通过 Zotero Web API 获取文库条目，支持增量更新
- **智能画像构建**：使用语义嵌入向量化条目，支持增量计算（仅处理新增/变更条目）
- **多源候选抓取**：支持 Crossref、arXiv、EarthArXiv 数据源
- **智能评分排序**：结合语义相似度、时间衰减、期刊质量及研究领域权重
- **兴趣驱动推荐**：基于用户描述的研究兴趣，使用语义重排序筛选最相关论文

### AI 增强

- **AI 摘要生成**：为推荐论文生成结构化摘要（研究问题、方法、发现、创新点）
- **标题翻译**：自动将英文论文标题翻译为中文
- **领域分类**：自动识别论文所属研究领域（可自定义领域列表）
- **摘要补全**：使用反检测浏览器从出版商网站抓取缺失摘要

### 作者追踪

- **关注作者**：通过 ORCID 追踪指定作者的所有新发表论文
- **独立通道**：关注作者的论文独立于推荐系统，不受评分阈值影响
- **增量同步**：首次全量抓取后，每日仅同步新发表论文

### 多格式输出

- **RSS 订阅源**：支持 Feedly、Inoreader 等 RSS 阅读器
- **HTML 报告**：响应式网页报告，支持移动端浏览
- **文献归档**：支持按日期/年份/期刊/来源/标签/领域/作者分组浏览
- **文献导出**：支持导出为 RIS、BibTeX 格式，导入 Zotero/EndNote/Mendeley
- **推送到 Zotero**：将推荐论文直接推送到 Zotero 文库

## 快速开始

### 1. 克隆仓库并安装依赖

```bash
git clone <your-repo-url>
cd ZotWatch
uv sync
```

### 2. 安装 Camoufox 浏览器

ZotWatch 使用 [Camoufox](https://github.com/nicholaswan/camoufox)（基于 Firefox 的反检测浏览器）从出版商网站抓取论文摘要：

```bash
uv run python -m camoufox fetch
```

> **注意**：首次下载约需 1-2 分钟，浏览器文件约 200MB。GitHub Actions 会自动处理此步骤并缓存。

### 3. 配置环境变量

复制 `.env.example` 为 `.env` 并填入你的 API 密钥：

```bash
cp .env.example .env
```

| 变量名 | 必需 | 说明 | 获取地址 |
|--------|------|------|----------|
| `ZOTERO_API_KEY` | ✅ | Zotero API 密钥 | [Zotero API Keys](https://www.zotero.org/settings/keys) |
| `ZOTERO_USER_ID` | ✅ | Zotero 用户 ID | 同上 |
| `VOYAGE_API_KEY` | ⚠️ | Voyage AI API 密钥 | [Voyage AI](https://dash.voyageai.com/) |
| `DASHSCOPE_API_KEY` | ⚠️ | 阿里云 DashScope API 密钥 | [阿里云百炼平台](https://bailian.console.aliyun.com/) |
| `MOONSHOT_API_KEY` | ⚠️ | Kimi API 密钥 | [Moonshot AI](https://platform.moonshot.cn/) |
| `OPENROUTER_API_KEY` | ⚠️ | OpenRouter API 密钥 | [OpenRouter](https://openrouter.ai/keys) |
| `DEEPSEEK_API_KEY` | ⚠️ | DeepSeek API 密钥 | [DeepSeek](https://platform.deepseek.com/) |
| `CROSSREF_MAILTO` | 推荐 | Crossref 礼貌池邮箱 | 你的邮箱地址 |

> **注意**：
> - **嵌入提供商**：`VOYAGE_API_KEY` 和 `DASHSCOPE_API_KEY` 二选一
> - **LLM 提供商**：`MOONSHOT_API_KEY`、`OPENROUTER_API_KEY`、`DEEPSEEK_API_KEY` 至少配置一个

### 4. 运行

```bash
# 首次全量画像构建
uv run zotwatch profile --full

# 日常监测（生成 RSS + HTML 报告 + AI 摘要）
uv run zotwatch watch

# 生成文献归档页面
uv run zotwatch archive
```

## CLI 命令

### `zotwatch profile`

构建或更新用户研究画像。

```bash
uv run zotwatch profile [OPTIONS]

Options:
  --full    全量重建（重新计算所有嵌入）
```

默认使用增量模式，仅对新增或内容变更的条目计算嵌入向量。当文库内容无变化时会自动跳过重建。

### `zotwatch watch`

获取、评分并输出论文推荐。

```bash
uv run zotwatch watch [OPTIONS]

Options:
  --rss        只生成 RSS 订阅源
  --report     只生成 HTML 报告
  --top N      保留前 N 条结果（默认 30）
  --push       推送推荐到 Zotero
```

### `zotwatch archive`

生成文献归档页面，支持多种分组视图。

```bash
uv run zotwatch archive [OPTIONS]

Options:
  --group-by TYPE   分组方式: date/year/venue/source/label/domain/author/all
  --days N          显示最近 N 天的文献（默认 90）
```

## 配置指南

所有配置集中在 `config/config.yaml`，支持环境变量替换（`${VAR_NAME}` 语法）。

### 数据源配置

```yaml
sources:
  # Crossref（学术期刊）
  crossref:
    enabled: true
    days_back: 7        # 抓取最近 N 天的论文
    max_results: 3000

  # arXiv（预印本，主要用于 CS/ML 领域）
  arxiv:
    enabled: false
    categories:
      - "physics.geo-ph"  # 地球物理
      - "astro-ph.EP"     # 地球与行星天体物理
    days_back: 7
    max_results: 500

  # EarthArXiv（地球科学预印本）
  eartharxiv:
    enabled: true
    days_back: 14
    max_results: 500
```

### 关注作者配置

通过 ORCID 追踪指定作者的论文：

```yaml
sources:
  followed_authors:
    enabled: true
    polite_email: "${CROSSREF_MAILTO}"
    max_results_per_author: 10000
    authors:
      - name: "张三"
        id: "0000-0001-2345-6789"   # ORCID
      - name: "John Doe"
        id: "0000-0002-3456-7890"
```

### 嵌入提供商配置

#### Voyage AI（推荐用于英文论文）

```yaml
embedding:
  provider: "voyage"
  model: "voyage-3.5"
  api_key: "${VOYAGE_API_KEY}"
  batch_size: 128

scoring:
  rerank:
    provider: "voyage"
    model: "rerank-2.5"
```

#### DashScope（推荐用于中文或中英混合）

```yaml
embedding:
  provider: "dashscope"
  model: "text-embedding-v4"
  api_key: "${DASHSCOPE_API_KEY}"
  batch_size: 10  # DashScope 限制 ≤10

scoring:
  rerank:
    provider: "dashscope"
    model: "qwen3-rerank"
```

> **重要**：当 `scoring.interests.enabled=true` 时，`embedding.provider` 和 `scoring.rerank.provider` 必须使用相同的提供商。

### LLM 提供商配置

支持三种 LLM 提供商：Kimi、OpenRouter、DeepSeek。

```yaml
llm:
  enabled: true
  provider: "openrouter"  # 或 "kimi" / "deepseek"
  api_key: "${OPENROUTER_API_KEY}"
  model: "anthropic/claude-3.5-sonnet"
  max_tokens: 5120
  temperature: 0.3
  translation:
    enabled: true  # 启用标题翻译
  domain_classification:
    enabled: true  # 启用领域分类
    domains:       # 自定义领域列表
      - "地球化学"
      - "岩石学"
      - "矿物学"
      - "火山学"
      - "构造地质学"
```

### 兴趣驱动推荐配置

```yaml
scoring:
  interests:
    enabled: true
    description: |
      我是一名地球化学/岩石学方向的研究者，重点关注：
      1) 深部地幔岩浆过程
      2) 俯冲带中金属元素与挥发分的循环
      3) 成矿过程与资源潜力

      请排除以下领域：材料科学、生物医学、环境工程

    # 包含关键词（至少匹配一个才保留）
    include_keywords:
      - "geochemistry"
      - "petrology"
      - "mantle"
      - "subduction"
      - "isotope"

    # 排除关键词（匹配任一则排除）
    exclude_keywords:
      - "nanomaterial"
      - "battery"
      - "drug delivery"
```

### 评分阈值配置

#### 动态阈值（推荐）

```yaml
scoring:
  thresholds:
    mode: "dynamic"
    dynamic:
      must_read_percentile: 95  # 前 5% 标记为必读
      consider_percentile: 70   # 70-95 百分位标记为推荐
      min_must_read: 0.55       # 必读最低分数
      min_consider: 0.35        # 推荐最低分数
```

#### 固定阈值

```yaml
scoring:
  thresholds:
    mode: "fixed"
    must_read: 0.75
    consider: 0.55
```

## 目录结构

```
ZotWatch/
├── src/zotwatch/           # 主包
│   ├── core/               # 核心模型和协议
│   ├── config/             # 配置管理
│   ├── infrastructure/     # 存储、嵌入、HTTP 客户端
│   ├── sources/            # 数据源（arXiv、Crossref、EarthArXiv、OpenAlex）
│   ├── llm/                # LLM 集成（Kimi、OpenRouter、DeepSeek）
│   ├── pipeline/           # 处理管道
│   ├── output/             # 输出生成（RSS、HTML、Zotero 推送）
│   ├── templates/          # HTML 模板
│   └── cli/                # Click CLI
├── config/
│   └── config.yaml         # 统一配置文件
├── data/                   # 画像/缓存（不纳入版本控制）
├── reports/                # 生成的 RSS/HTML 输出
└── .github/workflows/      # GitHub Actions 配置
```

## 数据文件

| 文件 | 说明 |
|------|------|
| `data/journal_whitelist.csv` | Crossref 期刊白名单（ISSN、期刊名、类别、影响因子） |
| `data/profile.sqlite` | Zotero 条目和元数据 |
| `data/faiss.index` | FAISS 向量索引 |
| `data/embeddings.sqlite` | 嵌入向量缓存 |
| `data/archive.sqlite` | 归档文献数据库 |

## GitHub Actions 自动运行

### 1. Fork 仓库并配置 Secrets

在仓库 **Settings → Secrets and variables → Actions** 中添加必要的 API 密钥。

### 2. 启用 GitHub Pages

进入 **Settings → Pages**，Source 选择 **GitHub Actions**。

### 3. 首次运行

在 **Actions** 标签页手动触发 **Daily Watch & Deploy**。

### 4. 访问结果

- **HTML 报告**：`https://[username].github.io/[repo]/`
- **文献归档**：`https://[username].github.io/[repo]/archive.html`
- **RSS 订阅**：`https://[username].github.io/[repo]/feed.xml`

## 常见问题

**Q: 推荐为空？**

检查以下可能原因：
- 所有候选都超出时间窗口（默认 7 天）
- `include_keywords` 过严导致过滤过多
- `similarity_gate` 阈值过高（默认 0.20）

**Q: 如何强制重新构建画像？**

```bash
uv run zotwatch profile --full
```

**Q: 切换嵌入提供商后需要做什么？**

切换后首次运行会自动检测并重建画像，无需手动操作。

**Q: 如何添加关注的作者？**

在 `config/config.yaml` 的 `sources.followed_authors.authors` 中添加作者的姓名和 ORCID。

**Q: 如何修改监测的期刊？**

编辑 `data/journal_whitelist.csv`，添加或删除期刊的 ISSN。

## 许可证

本项目基于 [MIT 许可证](LICENSE) 发布。
