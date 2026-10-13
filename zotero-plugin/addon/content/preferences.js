/* global Zotero, ChromeUtils, Services, document, window */

// Preference pane logic. Inputs bind to prefs automatically via their
// `preference` attributes; this script only wires up the buttons.

var ZotWatchPrefs = {
  initialized: false,

  LLM_PRESETS: {
    openrouter: { baseURL: "https://openrouter.ai/api/v1", model: "deepseek/deepseek-chat" },
    deepseek: { baseURL: "https://api.deepseek.com/v1", model: "deepseek-chat" },
    kimi: { baseURL: "https://api.moonshot.cn/v1", model: "kimi-k2-turbo-preview" },
    dashscope: { baseURL: "https://dashscope.aliyuncs.com/compatible-mode/v1", model: "qwen-plus" },
  },

  get api() {
    return Zotero.ZotWatch;
  },

  init() {
    if (this.initialized || !document.getElementById("zw-prefs-root")) return;
    this.initialized = true;

    const on = (id, handler) => {
      const el = document.getElementById(id);
      if (el) el.addEventListener("command", () => handler.call(this));
    };
    on("zw-run", () => {
      this.api.openRecommendations();
      this.api.runNow().catch(() => {});
    });
    on("zw-test", () => this.show("正在测试连接…", () => this.api.testConnections()));
    on("zw-status-refresh", () => this.show("正在读取状态…", () => this.api.getStatus()));
    on("zw-journals-suggest", () =>
      this.show("正在分析文库期刊（可能需要查询 Crossref）…", async () => {
        this.setPref("journals.whitelist", await this.api.suggestJournals());
        return "已根据文库生成期刊列表（已合并原有条目）。可手动补充影响因子。";
      }),
    );
    on("zw-journals-preset", () =>
      this.show("正在载入…", async () => {
        this.setPref("journals.whitelist", await this.api.loadPresetJournals());
        return "已载入地学期刊示例列表（来自 AIZotWatch 项目）。";
      }),
    );
    on("zw-journals-import", () => this.importWhitelist());
    on("zw-reset", () => this.reset());

    const preset = document.getElementById("zw-llm-preset");
    if (preset) {
      preset.addEventListener("command", () => {
        const p = this.LLM_PRESETS[preset.value];
        if (!p) return;
        this.setPref("llm.baseURL", p.baseURL);
        this.setPref("llm.model", p.model);
      });
    }

    this.show("正在读取状态…", () => this.api.getStatus());
  },

  setPref(key, value) {
    Zotero.Prefs.set(`extensions.zotwatch.${key}`, value, true);
  },

  async show(pending, task) {
    const out = document.getElementById("zw-output");
    if (out) out.textContent = pending;
    try {
      const text = await task();
      if (out && text) out.textContent = text;
    } catch (e) {
      if (out) out.textContent = `出错：${e.message || e}`;
      Zotero.logError(e);
    }
  },

  async importWhitelist() {
    const { FilePicker } = ChromeUtils.importESModule("chrome://zotero/content/modules/filePicker.mjs");
    const fp = new FilePicker();
    fp.init(window, "选择期刊白名单 CSV", fp.modeOpen);
    fp.appendFilter("CSV", "*.csv");
    fp.appendFilters(fp.filterAll);
    if ((await fp.show()) !== fp.returnOK) return;
    const path = typeof fp.file === "string" ? fp.file : fp.file.path;
    await this.show("正在导入…", async () => {
      this.setPref("journals.whitelist", await Zotero.File.getContentsAsync(path));
      return `已导入：${path}`;
    });
  },

  async reset() {
    const ok = Services.prompt.confirm(
      window,
      "ZotWatch",
      "将删除向量缓存、聚类结果和全部推荐记录（不会影响 Zotero 文库）。下次运行会重新向量化文库。确定继续吗？",
    );
    if (!ok) return;
    await this.show("正在清除…", async () => {
      await this.api.resetData();
      return "已清除 ZotWatch 数据。";
    });
  },
};

// Markup is inserted after this script runs; init on the pane's load event
// (see onload in preferences.xhtml) or as soon as the markup shows up.
(function waitForPane(attempts = 0) {
  if (ZotWatchPrefs.initialized) return;
  if (document.getElementById("zw-prefs-root")) ZotWatchPrefs.init();
  else if (attempts < 100) setTimeout(() => waitForPane(attempts + 1), 100);
})();
