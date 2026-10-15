# ZotWatch for Zotero 更新日志

发布新版本时，在这里新增一节 `## <版本号>`（与 `package.json` 的 `version` 一致）；发布流程会把这一节作为 GitHub Release 的说明。

## 0.1.0

首个版本：把 AIZotWatch 的文献推荐功能做成 Zotero 插件，支持 Zotero 7、8、9、10。

### 安装

下载下方的 `zotwatch-0.1.0.xpi`，在 Zotero 中打开「工具 → 插件」，点右上角齿轮选择「从文件安装插件…」。之后的版本会通过 Zotero 的插件更新自动推送。

### 功能

- 直接读取本地 Zotero 文库构建研究画像，增量向量化（只处理新增或修改过的条目），无需 Zotero API Key
- 每天自动从 Crossref（按关注期刊）、arXiv、EarthArXiv 抓取最新论文，摘要缺失时从 OpenAlex 补全
- 与 Python 版 AIZotWatch 一致的打分：k-NN 与语义聚类融合的相似度，加上期刊影响因子，按动态阈值分为「必读 / 推荐 / 一般」
- 推荐窗口（工具 → ZotWatch 文献推荐）：待查看 / 已加入 / 已忽略，支持筛选、排序和搜索；一键通过 DOI / arXiv 号加入 Zotero，放入指定分类并打标签
- 可选的 AI 功能：中文标题翻译、五要点结构化摘要（可保存为子笔记），支持任意 OpenAI 兼容接口（OpenRouter、DeepSeek、Kimi、通义千问等）
- 向量化服务：阿里云 DashScope、Voyage AI、OpenAI 兼容接口，或无需 API Key 的本地模式（默认）

### 快速上手

1. 在「Zotero 设置 → ZotWatch」中填写向量化服务的 API Key（推荐 DashScope），以及 Crossref 联系邮箱
2. 点「立即运行并查看推荐」。首次运行需要向量化整个文库，可能需要几分钟

完整说明见 [插件 README](https://github.com/CrazyHalfDay/AIZotWatch/blob/zotero-plugin-v0.1.0/zotero-plugin/README.md)。
