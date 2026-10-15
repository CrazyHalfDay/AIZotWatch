/* global Zotero, ChromeUtils, Services, document, window */

// Preference pane logic. Most inputs bind to prefs through their `preference`
// attributes; this script wires up the buttons and the journal whitelist, which
// is stored in a data file rather than a pref.
//
// Zotero 7 loads pane scripts into the preferences window, Zotero 8+ into a
// per-pane sandbox, so markup cannot call into this script through inline
// handlers. Instead, capture listeners on the document react to the 'load',
// 'showing' and 'unload' events Zotero dispatches to the pane root.

var ZotWatchPrefs = {
  initialized: false,
  disposed: false,
  saveTimer: null,

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
    on("zw-status-refresh", () => this.refreshStatus());
    on("zw-journals-suggest", () =>
      this.show("正在分析文库期刊（可能需要查询 Crossref）…", async () => {
        await this.setWhitelist(await this.api.suggestJournals());
        return "已根据文库生成期刊列表（已合并原有条目）。可手动补充影响因子。";
      }),
    );
    on("zw-journals-preset", () =>
      this.show("正在载入…", async () => {
        await this.setWhitelist(await this.api.loadPresetJournals());
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

    const whitelist = document.getElementById("zw-whitelist");
    if (whitelist) {
      whitelist.value = this.api.getJournalWhitelist();
      // Save shortly after typing stops, and immediately when the field loses focus
      whitelist.addEventListener("input", () => {
        clearTimeout(this.saveTimer);
        this.saveTimer = setTimeout(() => this.saveWhitelist(), 800);
      });
      whitelist.addEventListener("change", () => this.saveWhitelist());
    }
  },

  /** Pane became visible (not merely loaded for search or hover): show status. */
  onShowing() {
    this.refreshStatus();
  },

  onUnload() {
    if (this.saveTimer) {
      clearTimeout(this.saveTimer);
      this.saveWhitelist();
    }
    this.disposed = true;
  },

  setPref(key, value) {
    Zotero.Prefs.set(`extensions.zotwatch.${key}`, value, true);
  },

  saveWhitelist() {
    this.saveTimer = null;
    const whitelist = document.getElementById("zw-whitelist");
    if (!whitelist) return;
    this.api.setJournalWhitelist(whitelist.value).catch((e) => Zotero.logError(e));
  },

  async setWhitelist(text) {
    const whitelist = document.getElementById("zw-whitelist");
    if (whitelist) whitelist.value = text;
    await this.api.setJournalWhitelist(text);
  },

  refreshStatus() {
    this.show("正在读取状态…", () => this.api.getStatus());
  },

  async show(pending, task) {
    const output = () => (this.disposed ? null : document.getElementById("zw-output"));
    let out = output();
    if (out) out.textContent = pending;
    try {
      const text = await task();
      out = output();
      if (out && text) out.textContent = text;
    } catch (e) {
      out = output();
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
      await this.setWhitelist(await Zotero.File.getContentsAsync(path));
      return `已导入：${path}`;
    });
  },

  async reset() {
    const ok = Services.prompt.confirm(
      window,
      "ZotWatch",
      "将删除向量缓存、聚类结果和全部推荐记录（不会影响 Zotero 文库和期刊列表）。下次运行会重新向量化文库。确定继续吗？",
    );
    if (!ok) return;
    await this.show("正在清除…", async () => {
      await this.api.resetData();
      return "已清除 ZotWatch 数据。";
    });
  },
};

for (const [type, handler] of [
  ["load", () => ZotWatchPrefs.init()],
  ["showing", () => ZotWatchPrefs.onShowing()],
  ["unload", () => ZotWatchPrefs.onUnload()],
]) {
  document.addEventListener(
    type,
    (event) => {
      if (event.target?.id === "zw-prefs-root") handler();
    },
    true,
  );
}
