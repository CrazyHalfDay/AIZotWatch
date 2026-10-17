/* global el, svgIcon */

// Search-and-add component of the main window (search.css): a text input with
// an inline result list, used by 关注期刊, 关注作者 and 顶刊地学 (settings.js)
// and reusable by any page. Typing searches after a pause (400 ms, at least
// two characters; Enter searches at once), a spinner shows while a search
// runs, answers to older queries are ignored, and every result row has one
// action button (「添加」, 「关注」, ...). Keyboard: ↑/↓ move through the
// results, Enter runs the action of the highlighted row, Esc closes the list
// (a second Esc clears the input). Combobox semantics: the input owns the
// listbox through aria-controls / aria-activedescendant. Result text is built
// by the caller with el() (textContent only).

const SEARCH_DELAY = 400;
const SEARCH_MIN_CHARS = 2;

/**
 * @param {object} options
 * @param {string} options.id - id of the input; the list is `${id}-results`
 * @param {string} options.label - accessible name of the input
 * @param {string} [options.placeholder]
 * @param {(query: string) => Promise<object[]>} options.search
 * @param {(result: object) => Node} options.render - the text of a result row
 * @param {(result: object) => {label: string, disabled?: boolean, title?: string}} options.action
 * @param {(result: object) => Promise<string|void>} options.pick - runs the action; a returned string is shown as the status
 * @param {(result: object) => string} [options.nameOf] - names a result in its button's label (default: title or name)
 * @param {string} [options.emptyText] - shown when a search finds nothing
 * @param {Node} [options.hint] - shown under the input (e.g. a missing-key notice)
 * @param {number} [options.delay]
 * @param {number} [options.minChars]
 * @returns {{node, input, clear(), refresh(), close(), searchNow(): Promise<void>}}
 */
function createSearchBox(options) {
  const { id, label, placeholder = "", search, render, action, pick, emptyText = "没有找到匹配的结果", hint = null } = options;
  const nameOf = options.nameOf || ((result) => result?.title || result?.name || "");
  const delay = options.delay ?? SEARCH_DELAY;
  const minChars = options.minChars ?? SEARCH_MIN_CHARS;
  const listId = `${id}-results`;
  const state = { seq: 0, timer: null, results: [], rows: [], active: -1, busy: false, query: "" };

  const input = el("input", {
    type: "text",
    class: "input sb-input",
    id,
    role: "combobox",
    "aria-label": label,
    "aria-autocomplete": "list",
    "aria-expanded": "false",
    "aria-controls": listId,
    placeholder,
    autocomplete: "off",
    spellcheck: "false",
    inputmode: "search",
  });
  const spinner = el("span", { class: "sb-spinner", hidden: true, "aria-hidden": "true" });
  const list = el("ul", { class: "sb-results", id: listId, role: "listbox", "aria-label": label, hidden: true });
  const status = el("p", { class: "sb-status", role: "status", hidden: true });
  const node = el(
    "div",
    { class: "sb" },
    el("div", { class: "sb-field" }, svgIcon("search", "icon sb-icon"), input, spinner),
    hint,
    status,
    list,
  );

  function setStatus(text, tone = "muted") {
    status.hidden = !text;
    status.className = `sb-status ${tone}`;
    status.textContent = text || "";
  }

  function setBusy(on) {
    state.busy = on;
    spinner.hidden = !on;
    node.setAttribute("aria-busy", on ? "true" : "false");
  }

  function open(on) {
    list.hidden = !on;
    input.setAttribute("aria-expanded", on ? "true" : "false");
    if (!on) setActive(-1);
  }

  function setActive(index) {
    state.active = index;
    state.rows.forEach((row, i) => {
      const on = i === index;
      row.li.classList.toggle("active", on);
      row.li.setAttribute("aria-selected", on ? "true" : "false");
    });
    if (index >= 0 && state.rows[index]) {
      input.setAttribute("aria-activedescendant", state.rows[index].li.getAttribute("id"));
      state.rows[index].li.scrollIntoView?.({ block: "nearest" });
    } else {
      input.removeAttribute("aria-activedescendant");
    }
  }

  /** Update every row's button (e.g. after the list it adds to changed). */
  function refresh() {
    for (const row of state.rows) {
      if (row.running) continue;
      const a = action(row.result) || { label: "" };
      row.button.textContent = a.label;
      row.button.disabled = !!a.disabled;
      if (a.title) row.button.setAttribute("title", a.title);
      else row.button.removeAttribute("title");
      row.button.setAttribute("aria-label", `${a.label}：${row.name}`);
    }
  }

  async function runPick(row) {
    const a = action(row.result) || {};
    if (row.running || a.disabled) return;
    row.running = true;
    row.button.disabled = true;
    try {
      const message = await pick(row.result);
      if (message) setStatus(message, "good");
    } catch (e) {
      if (!e?.cancelled) setStatus(`出错：${e?.message || e}`, "error");
    } finally {
      row.running = false;
      refresh();
    }
  }

  function renderResults(results) {
    state.results = results;
    state.rows = results.map((result, i) => {
      const button = el("button", { type: "button", class: "btn small sb-action", tabindex: "-1" });
      const content = render(result);
      const li = el("li", { class: "sb-result", role: "option", id: `${listId}-${i}`, "aria-selected": "false" }, el("div", { class: "sb-result-body" }, content), button);
      const row = { result, li, button, running: false, name: String(nameOf(result) || "") };
      button.addEventListener("click", () => runPick(row));
      li.addEventListener("mousemove", () => {
        if (state.active !== i) setActive(i);
      });
      return row;
    });
    list.replaceChildren(...state.rows.map((row) => row.li));
    refresh();
    open(results.length > 0);
  }

  async function searchNow() {
    clearTimeout(state.timer);
    state.timer = null;
    const query = String(input.value || "").trim();
    if (query.length < minChars) return;
    const seq = ++state.seq;
    state.query = query;
    setBusy(true);
    try {
      const results = (await search(query)) || [];
      if (seq !== state.seq) return;
      renderResults(results);
      setStatus(results.length ? "" : emptyText);
    } catch (e) {
      if (seq !== state.seq) return;
      renderResults([]);
      // A search replaced by a newer one is cancelled: say nothing
      setStatus(e?.cancelled ? "" : `搜索失败：${e?.message || e}`, "error");
    } finally {
      if (seq === state.seq) setBusy(false);
    }
  }

  function clear() {
    clearTimeout(state.timer);
    state.seq++;
    input.value = "";
    state.query = "";
    setBusy(false);
    renderResults([]);
    setStatus("");
  }

  input.addEventListener("input", () => {
    clearTimeout(state.timer);
    const query = String(input.value || "").trim();
    if (query.length < minChars) {
      // Too short: forget the results of a longer query
      state.seq++;
      setBusy(false);
      renderResults([]);
      setStatus("");
      return;
    }
    if (query === state.query && !state.busy && state.results.length) {
      open(true);
      return;
    }
    state.timer = setTimeout(searchNow, delay);
  });

  input.addEventListener("keydown", (event) => {
    const count = state.rows.length;
    switch (event.key) {
      case "ArrowDown":
        if (!count) return;
        event.preventDefault();
        if (list.hidden) open(true);
        setActive(state.active < count - 1 ? state.active + 1 : 0);
        break;
      case "ArrowUp":
        if (!count) return;
        event.preventDefault();
        if (list.hidden) open(true);
        setActive(state.active > 0 ? state.active - 1 : count - 1);
        break;
      case "Enter":
        event.preventDefault();
        if (!list.hidden && state.active >= 0 && state.rows[state.active]) runPick(state.rows[state.active]);
        else searchNow();
        break;
      case "Escape":
        // Handled here: the window's Esc (close) never sees it
        event.preventDefault();
        event.stopPropagation?.();
        if (!list.hidden) open(false);
        else if (input.value) clear();
        break;
      default:
    }
  });

  return { node, input, clear, refresh, close: () => open(false), searchNow };
}

/** Number with thousands separators ("20,000"), "" when missing. */
function searchCount(n) {
  return Number.isFinite(n) ? new Intl.NumberFormat("zh-CN").format(n) : "";
}

/** A result row: a title line (with optional trailing nodes) and a muted meta line joined by " · ". */
function searchResultText(title, meta, ...extra) {
  return el(
    "div",
    { class: "sb-text" },
    el("div", { class: "sb-title-line" }, el("span", { class: "sb-title", text: title }), ...extra),
    el("div", { class: "sb-meta", text: meta.filter(Boolean).join(" · ") }),
  );
}
