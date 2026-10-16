// Visual QA harness (dev only; not part of the build or CI).
//
// Serves build/addon/content/ over a local HTTP server (chrome://zotwatch/content/
// URLs rewritten to it, other chrome:// scripts stubbed), injects a mock ChromeUtils / Zotero / Services with the
// fixture data of scripts/preview-fixtures.mjs, and screenshots the main window
// pages in light and dark at 1120x820 and 820x700 with Playwright's Chromium.
// Chromium is not Firefox 115: this checks layout and visuals only.
//
//   npm run build && node scripts/preview.mjs <out-dir> [options]
//
// Options (comma-separated lists):
//   --pages overview,inbox,settings,settings/embedding   routes to capture
//   --schemes light,dark        --sizes 1120x820,820x700
//   --states default,running,error,empty                 extra states (light, first size only)
//   --styleguide                also capture the shared components (main.css)
//   --script <file.mjs>         run its default export `async (page, {route, scheme, size})`
//                               before each screenshot (e.g. to type into fields)
//
// Exits with status 1 on any page error, console error or XML parse error.

import { execSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync } from "node:fs";
import { createServer } from "node:http";
import { dirname, extname, join, normalize, resolve, sep } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import vm from "node:vm";

import { build } from "esbuild";

import { createEmptyFixtures, createFixtures } from "./preview-fixtures.mjs";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const contentDir = join(root, "build", "addon", "content");
const DEFAULT_PAGES = ["overview", "inbox", "settings", "settings/embedding", "settings/ai"];
const TYPES = {
  ".xhtml": "application/xhtml+xml; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".svg": "image/svg+xml",
  ".csv": "text/csv; charset=utf-8",
  ".png": "image/png",
};

// ------------------------------------------------------------------ args

function parseArgs(argv) {
  const opts = { out: null, pages: DEFAULT_PAGES, schemes: ["light", "dark"], sizes: ["1120x820", "820x700"], states: ["default"], styleguide: false, script: null };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    const list = () => argv[++i].split(",").map((s) => s.trim()).filter(Boolean);
    if (arg === "--pages") opts.pages = list();
    else if (arg === "--schemes") opts.schemes = list();
    else if (arg === "--sizes") opts.sizes = list();
    else if (arg === "--states") opts.states = list();
    else if (arg === "--styleguide") opts.styleguide = true;
    else if (arg === "--script") opts.script = resolve(argv[++i]);
    else if (!arg.startsWith("--")) opts.out = resolve(arg);
    else throw new Error(`Unknown option ${arg}`);
  }
  return opts;
}

async function loadPlaywright() {
  try {
    return await import("playwright");
  } catch {
    try {
      const globalRoot = execSync("npm root -g", { encoding: "utf8" }).trim();
      return await import(pathToFileURL(join(globalRoot, "playwright", "index.mjs")).href);
    } catch {
      console.error("Playwright is not installed. Install it with `npm install -g playwright` (and `npx playwright install chromium`).");
      process.exit(2);
    }
  }
}

// ---------------------------------------------------------------- server

function startServer() {
  const server = createServer((req, res) => {
    const path = decodeURIComponent(new URL(req.url, "http://localhost").pathname);
    if (path === "/favicon.ico") {
      res.writeHead(204).end();
      return;
    }
    // Firefox / Zotero scripts the window loads (the edit-menu overlay) do not exist here
    if (path.startsWith("/__chrome/")) {
      res.writeHead(200, { "Content-Type": TYPES[".js"], "Cache-Control": "no-store" }).end("// stub\n");
      return;
    }
    const file = normalize(join(contentDir, path));
    if (!file.startsWith(contentDir + sep) || !existsSync(file)) {
      res.writeHead(404).end("not found");
      return;
    }
    const type = TYPES[extname(file)] || "application/octet-stream";
    let body = readFileSync(file);
    if (/^(text|application\/xhtml)|svg/.test(type)) {
      body = Buffer.from(
        body
          .toString("utf8")
          .replaceAll("chrome://zotwatch/content/", "/")
          .replace(/chrome:\/\/(\w+)\/content\//g, "/__chrome/$1/"),
      );
    }
    res.writeHead(200, { "Content-Type": type, "Cache-Control": "no-store" }).end(body);
  });
  return new Promise((resolveServer) => server.listen(0, "127.0.0.1", () => resolveServer(server)));
}

/** Default prefs of the built plugin ({"extensions.zotwatch.x": value}). */
function prefDefaults() {
  const defaults = {};
  vm.runInNewContext(readFileSync(join(root, "build", "addon", "prefs.js"), "utf8"), { pref: (k, v) => (defaults[k] = v) });
  return defaults;
}

/** The pure helpers the mock API delegates to, bundled from src/ for the browser. */
async function coreBundle() {
  const result = await build({
    stdin: {
      contents: [
        'export { parseFollowedAuthors } from "./src/core/authors.js";',
        'export { summarizeWhitelist } from "./src/core/journals.js";',
        'export { autoValues } from "./src/settings.js";',
      ].join("\n"),
      resolveDir: root,
      loader: "js",
    },
    bundle: true,
    format: "iife",
    globalName: "ZotWatchPreviewCore",
    write: false,
    logLevel: "warning",
  });
  // Init scripts may run inside a function: publish the bundle on window explicitly
  return `${result.outputFiles[0].text}\nwindow.ZotWatchPreviewCore = ZotWatchPreviewCore;\n`;
}

// ------------------------------------------------------------------ mock

/** Runs in the page before any script: a mock Zotero with Zotero.ZotWatch over the fixtures. */
function installMock({ fixtures, defaults, state }) {
  const PREFIX = "extensions.zotwatch.";
  const core = window.ZotWatchPreviewCore;
  const data = JSON.parse(JSON.stringify(fixtures));
  const listeners = new Set();
  const prefs = new Map(Object.entries(defaults));
  for (const [k, v] of Object.entries(data.prefs || {})) prefs.set(PREFIX + k, v);
  const delay = (ms) => new Promise((r) => setTimeout(r, ms));
  const emit = (event) => [...listeners].forEach((fn) => fn(event));

  if (state === "running") {
    data.view.running = true;
    data.view.progress = { stage: "embed", message: "候选文献向量化 412/1012", current: 412, total: 1012 };
  }
  if (state === "error") data.view.lastError = { message: "网络连接失败：api.crossref.org（0x804b0014）", at: Date.now() - 10 * 60000 };

  const Prefs = {
    get: (key) => prefs.get(key),
    set(key, value) {
      if (!(key in defaults)) throw new Error(`No default for pref ${key}`);
      if (typeof value !== typeof defaults[key]) throw new Error(`Pref ${key} expects a ${typeof defaults[key]}, got a ${typeof value}`);
      prefs.set(key, value);
    },
  };

  let running = null;
  const ZotWatch = {
    lastPage: null,
    version: "0.3.0",
    async getView() {
      return JSON.parse(JSON.stringify(data.view));
    },
    async getDashboard() {
      await delay(120);
      return JSON.parse(JSON.stringify({ ...data.dashboard, running: data.view.running, lastError: data.view.lastError }));
    },
    async checkConnections(only = null) {
      await delay(700);
      return data.connections.filter((r) => !only || only.includes(r.key));
    },
    async testConnections() {
      return (await this.checkConnections()).map((r) => `${r.ok ? "✅" : r.ok === false ? "❌" : "ℹ️"} ${r.name}：${r.detail}`).join("\n");
    },
    addListener: (fn) => listeners.add(fn),
    removeListener: (fn) => listeners.delete(fn),
    runNow() {
      if (running) return running;
      running = (async () => {
        data.view.running = true;
        emit({ type: "start" });
        const steps = [["读取 Zotero 文库…"], ["抓取最新文献…"], ["候选文献向量化", 1012], ["计算相关性评分…"]];
        for (const [message, total] of steps) {
          for (let i = 0; i <= (total ? 4 : 0); i++) {
            const current = total ? Math.round((total * i) / 4) : undefined;
            data.view.progress = { message: total ? `${message} ${current}/${total}` : message, current, total };
            emit({ type: "progress", progress: data.view.progress });
            await delay(350);
          }
        }
        data.view.running = false;
        data.view.progress = null;
        running = null;
        emit({ type: "done" });
      })();
      return running;
    },
    cancelRun() {},
    async setStatus(ids, status) {
      for (const id of [].concat(ids)) {
        const work = data.view.works.find((w) => w.id === id);
        if (work) work.status = status;
      }
      emit({ type: "inbox" });
    },
    async addToLibrary(id) {
      await delay(600);
      const work = data.view.works.find((w) => w.id === id);
      work.status = "added";
      work.itemID = 99;
      emit({ type: "inbox" });
      return 99;
    },
    async summarize(id) {
      await delay(800);
      const work = data.view.works.find((w) => w.id === id);
      work.summary = data.view.works.find((w) => w.summary)?.summary;
      emit({ type: "inbox" });
      return work.summary;
    },
    async showItem() {},
    openUrl: (url) => console.log(`openUrl ${url}`),
    openPreferences: () => console.log("openPreferences"),
    openWindow: () => {},
    getJournalWhitelist: () => data.whitelist,
    async setJournalWhitelist(text) {
      data.whitelist = String(text);
    },
    async suggestJournals() {
      await delay(500);
      return data.whitelist;
    },
    async loadPresetJournals() {
      return data.whitelist;
    },
    async resetData() {
      data.view.works = [];
      data.view.lastRun = null;
      emit({ type: "inbox" });
    },
    async revealDataDir() {
      return true;
    },
    getDataDir: () => data.dashboard.dataDir,
    parseFollowedAuthors: (text) => core.parseFollowedAuthors(text),
    summarizeWhitelist: (text) => core.summarizeWhitelist(text),
    getAutoValues: () => core.autoValues(prefs.get(`${PREFIX}embedding.provider`)),
  };

  const Zotero = {
    ZotWatch,
    Prefs,
    isMac: false,
    version: "7.0.25",
    debug() {},
    logError: (e) => console.error(String(e?.stack || e)),
    launchURL: (url) => console.log(`launchURL ${url}`),
    UIProperties: {
      registerRoot(rootEl) {
        rootEl.style.fontSize = "1rem";
        rootEl.style.setProperty("--zotero-font-size", "1rem");
      },
    },
    File: { getContentsAsync: async () => data.whitelist, reveal: async () => {} },
  };
  window.ChromeUtils = {
    importESModule(url) {
      if (url.endsWith("zotero.mjs")) return { Zotero };
      if (url.endsWith("filePicker.mjs")) {
        return {
          FilePicker: class {
            init() {}
            appendFilter() {}
            appendFilters() {}
            async show() {
              return 1;
            }
          },
        };
      }
      throw new Error(`Unexpected module ${url}`);
    },
  };
  window.Services = { prompt: { confirm: () => true }, locale: { appLocaleAsBCP47: "zh-CN" } };
  window.__zwPreview = { emit, data };
}

/** Replaces the overview content with samples of the shared components (uses main.js helpers). */
function renderStyleguide() {
  /* global el, svgIcon, emptyState */
  const row = (...children) => el("div", { style: "display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:8px 0" }, ...children);
  const card = (title, ...body) => el("div", { class: "card", style: "margin-bottom:12px" }, el("div", { class: "card-header" }, el("h2", { class: "card-title", text: title })), el("div", { class: "card-body" }, ...body));
  const seg = el("div", { class: "seg" }, el("button", { type: "button", class: "active", "data-value": "a" }, "动态"), el("button", { type: "button", "data-value": "b" }, "固定"));
  const radio = (value, title, text, checked) =>
    el("label", { class: "radio-card" }, el("input", { type: "radio", name: "sg", value, checked: checked || null }), el("span", { class: "radio-card-body" }, el("span", { class: "radio-card-title", text: title }), el("span", { class: "radio-card-text", text })));
  const field = (label, help, control) => el("div", { class: "field" }, el("div", {}, el("div", { class: "field-label", text: label }), el("div", { class: "field-help", text: help })), el("div", { class: "field-control" }, control));
  const sw = (on) => {
    const input = el("input", { type: "checkbox", role: "switch", class: "switch" });
    input.checked = on;
    return input;
  };
  document.getElementById("overview-root").replaceChildren(
    card(
      "按钮与徽章",
      row(el("button", { type: "button", class: "btn primary" }, svgIcon("refresh"), "立即检查"), el("button", { type: "button", class: "btn" }, "测试连接"), el("button", { type: "button", class: "btn ghost" }, "取消"), el("button", { type: "button", class: "btn danger" }, "清除缓存…"), el("button", { type: "button", class: "btn small" }, "小按钮"), el("button", { type: "button", class: "btn", disabled: true }, "禁用")),
      row(el("span", { class: "badge must", text: "必读" }), el("span", { class: "badge consider", text: "推荐" }), el("span", { class: "badge ignore", text: "一般" }), el("span", { class: "badge flagship" }, el("span", { class: "dot flagship" }), "顶刊地学"), el("span", { class: "badge followed" }, el("span", { class: "dot followed" }), "关注"), el("span", { class: "chip", text: "方向：斑岩铜矿成矿" }), el("span", { class: "pill", text: "12 个期刊" })),
      row(el("span", { class: "status" }, el("span", { class: "dot good" }), "正常"), el("span", { class: "status" }, el("span", { class: "dot critical" }), "出错"), el("span", { class: "status" }, el("span", { class: "dot warning" }), "有 1 个问题"), el("span", { class: "status" }, el("span", { class: "dot off" }), "未启用"), el("span", { class: "status" }, el("span", { class: "dot running" }), "正在检查…")),
    ),
    card(
      "表单",
      field("自动检查", "Zotero 运行时在后台按间隔检查新文献。", sw(true)),
      field("排除没有摘要的文献", "没有摘要的论文无法可靠地计算相似度。", sw(false)),
      field("检查间隔", "1–720 小时", el("span", { class: "input-unit" }, el("input", { class: "input", style: "width:90px", value: "24" }), "小时")),
      field("阈值模式", "动态阈值按每批分数分布计算。", seg),
      el("div", { class: "radio-cards", style: "margin-top:8px" }, radio("local", "本地模式", "无需 API Key，效果一般", false), radio("dashscope", "阿里云 DashScope（推荐）", "text-embedding-v4", true), radio("voyage", "Voyage AI", "voyage-3.5", false)),
      el("textarea", { class: "textarea mono", rows: "3", style: "margin-top:12px", placeholder: "issn,title,category,impact_factor" }),
    ),
    el("div", { class: "callout warning", style: "margin-bottom:12px" }, svgIcon("alert"), el("div", { class: "callout-body" }, el("div", { class: "callout-title", text: "上次运行有 1 个问题" }), el("div", { class: "callout-text", text: "arXiv：HTTP 503" }))),
    el("div", { class: "card" }, emptyState({ icon: "chart", title: "运行几次后这里会显示每次新增的推荐", compact: true })),
  );
}

// ------------------------------------------------------------------- run

async function main() {
  const opts = parseArgs(process.argv.slice(2));
  if (!opts.out) {
    console.error("Usage: node scripts/preview.mjs <out-dir> [--pages a,b] [--schemes light,dark] [--sizes 1120x820,820x700] [--states default,running,error,empty] [--styleguide]");
    process.exit(2);
  }
  if (!existsSync(join(contentDir, "main.xhtml"))) {
    console.error("build/addon/content/main.xhtml not found: run `npm run build` first.");
    process.exit(2);
  }
  mkdirSync(opts.out, { recursive: true });
  const { chromium } = await loadPlaywright();
  const server = await startServer();
  const base = `http://127.0.0.1:${server.address().port}`;
  const defaults = prefDefaults();
  const core = await coreBundle();
  const executablePath = process.env.CHROMIUM || "/opt/pw-browsers/chromium";
  const browser = await chromium.launch(existsSync(executablePath) ? { executablePath } : {});
  const prepare = opts.script ? (await import(pathToFileURL(opts.script).href)).default : null;
  const problems = [];
  const shots = [];

  async function capture({ route, scheme, size, state = "default", styleguide = false }) {
    const [width, height] = size.split("x").map(Number);
    const context = await browser.newContext({ viewport: { width, height }, deviceScaleFactor: 1, colorScheme: scheme });
    const page = await context.newPage();
    const label = `${styleguide ? "styleguide" : route.replace("/", "-")}${state === "default" ? "" : `-${state}`}-${scheme}-${size}`;
    page.on("pageerror", (e) => problems.push(`${label}: page error: ${e.message}`));
    page.on("console", (msg) => {
      if (msg.type() === "error") problems.push(`${label}: console error: ${msg.text()}`);
    });
    const fixtures = state === "empty" ? createEmptyFixtures() : createFixtures();
    await page.addInitScript({ content: core });
    await page.addInitScript(installMock, { fixtures, defaults, state });
    await page.emulateMedia({ colorScheme: scheme });
    await page.goto(`${base}/main.xhtml#${route}`);
    const parseError = await page.evaluate(() => !!document.getElementsByTagNameNS("*", "parsererror").length || document.documentElement.localName === "parsererror");
    if (parseError) problems.push(`${label}: XML parse error in main.xhtml`);
    await page.waitForSelector("html[data-ready]", { timeout: 10000 }).catch(() => problems.push(`${label}: the window never became ready`));
    if (route.startsWith("overview")) {
      // The dashboard loads after the shell (api.getDashboard()); the styleguide replaces it once loaded
      await page.waitForSelector('#overview-root[aria-busy="false"]', { timeout: 10000 }).catch(() => problems.push(`${label}: the overview never loaded`));
    }
    if (styleguide) await page.evaluate(renderStyleguide);
    if (prepare) await prepare(page, { route, scheme, size, state });
    await page.evaluate(() => document.fonts.ready);
    await page.waitForTimeout(250);
    const file = join(opts.out, `${label}.png`);
    await page.screenshot({ path: file });
    shots.push(file);
    await context.close();
  }

  try {
    for (const scheme of opts.schemes) {
      for (const size of opts.sizes) {
        for (const route of opts.pages) await capture({ route, scheme, size });
      }
    }
    for (const state of opts.states.filter((s) => s !== "default")) {
      for (const route of ["overview", "inbox"]) {
        await capture({ route, scheme: opts.schemes[0], size: opts.sizes[0], state });
      }
    }
    if (opts.styleguide) {
      for (const scheme of opts.schemes) await capture({ route: "overview", scheme, size: opts.sizes[0], styleguide: true });
    }
  } finally {
    await browser.close();
    server.close();
  }

  for (const file of shots) console.log(file);
  if (problems.length) {
    console.error(`\n${problems.length} problem(s):\n${problems.join("\n")}`);
    process.exit(1);
  }
}

await main();
