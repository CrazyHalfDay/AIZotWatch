/* global Zotero, document */

// Zotero settings → ZotWatch: a status summary with buttons that open the
// ZotWatch main window, where all settings live since 0.3.0.
//
// Zotero 7 loads pane scripts into the preferences window, Zotero 8+ into a
// per-pane sandbox, so markup cannot call into this script through inline
// handlers. Instead, capture listeners on the document react to the 'load',
// 'showing' and 'unload' events Zotero dispatches to the pane root.

var ZotWatchPrefs = {
  initialized: false,
  disposed: false,
  listener: null,

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
    on("zw-open", () => this.api?.openWindow("overview"));
    on("zw-open-settings", () => this.api?.openWindow("settings"));
    // Same as Tools → 立即检查新文献: show the recommendations and start a run
    on("zw-run", () => this.api?.runFromMenu());

    // Keep the status current while the pane is open
    this.listener = (event) => {
      if (["start", "done", "cancelled", "error", "inbox"].includes(event.type)) this.refreshStatus();
    };
    this.api?.addListener(this.listener);
  },

  /**
   * Zotero also dispatches 'showing' to every pane while the settings search
   * runs; only read the status when our pane is the one actually selected.
   */
  onShowing() {
    if (document.getElementById("prefs-search")?.value?.trim()) return;
    const selected = document.getElementById("prefs-navigation")?.value;
    if (selected && selected !== "zotwatch-prefpane") return;
    this.refreshStatus();
  },

  onUnload() {
    if (this.listener) this.api?.removeListener(this.listener);
    this.listener = null;
    this.disposed = true;
  },

  setText(id, text) {
    const el = this.disposed ? null : document.getElementById(id);
    if (el) el.textContent = text;
  },

  /** "2 小时前（2024-10-08 09:30）" */
  formatWhen(ms, now = Date.now()) {
    const diff = Math.max(0, now - ms);
    const relative =
      diff < 60000 ? "刚刚" : diff < 3600000 ? `${Math.floor(diff / 60000)} 分钟前` : diff < 86400000 ? `${Math.floor(diff / 3600000)} 小时前` : `${Math.floor(diff / 86400000)} 天前`;
    const d = new Date(ms);
    const pad = (n) => String(n).padStart(2, "0");
    return `${relative}（${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}）`;
  },

  async refreshStatus() {
    if (this.disposed || !this.api) return;
    try {
      const view = await this.api.getView();
      const pending = view.works.filter((w) => w.status === "new").length;
      const finished = view.lastRun ? view.lastRun.finishedAt || view.lastRun.runAt : 0;
      let last = finished ? this.formatWhen(finished) : "尚未运行";
      if (view.running) last = "正在检查…";
      else if (view.lastError && view.lastError.at > finished) last = `失败：${view.lastError.message}`;
      this.setText("zw-pending", `${pending} 篇`);
      this.setText("zw-last", last);
      this.setText("zw-embedding", this.api.embeddingInfo().label);
    } catch (e) {
      this.setText("zw-last", `出错：${e.message || e}`);
      Zotero.logError(e);
    }
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
