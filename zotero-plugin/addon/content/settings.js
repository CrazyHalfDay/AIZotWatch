/* global $, el, svgIcon, emptyState, api, registerPage, setSubtitle, setActiveSection, SETTINGS_SECTIONS */

// 设置 page: one card per section (SETTINGS_SECTIONS in main.js), reachable
// as #settings/<id>. Placeholder until the settings cards are built: each
// section shows its title and description; the settings themselves are still
// edited in Zotero's settings window.

var ZotWatchSettings = (() => {
  function sectionCard(section) {
    return el(
      "section",
      { class: "card settings-section", id: `settings-${section.id}`, "data-section": section.id },
      el(
        "div",
        { class: "card-header" },
        el("div", { class: "settings-section-heading" }, svgIcon(section.icon), el("h2", { class: "card-title", text: section.title })),
      ),
      el("div", { class: "card-body" }, el("p", { class: "settings-section-text", text: section.description })),
    );
  }

  function init() {
    $("settings-root").replaceChildren(
      emptyState({
        icon: "settings",
        title: "设置即将移到这里",
        text: "目前请在 Zotero 设置的 ZotWatch 页面中修改设置。",
        action: { label: "打开 Zotero 设置", onclick: () => api()?.openPreferences() },
        compact: true,
      }),
      ...SETTINGS_SECTIONS.map(sectionCard),
    );
  }

  function show({ section } = {}) {
    setSubtitle("settings", "所有修改自动保存");
    setActiveSection(section);
    if (section) $(`settings-${section}`)?.scrollIntoView({ block: "start" });
  }

  function refresh() {}

  return { init, show, refresh };
})();

registerPage("settings", ZotWatchSettings);
