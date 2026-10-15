/* global Components, Services, APP_SHUTDOWN, ZotWatchBundle */

// Zotero 7+ bootstrapped plugin entry point. The actual code is bundled into
// content/zotwatch.js (see scripts/build.mjs) and loaded via a chrome:// URL.

var chromeHandle;

function install() {}

async function startup({ id, version, rootURI }) {
  const aomStartup = Components.classes["@mozilla.org/addons/addon-manager-startup;1"].getService(
    Components.interfaces.amIAddonManagerStartup,
  );
  const manifestURI = Services.io.newURI(rootURI + "manifest.json");
  chromeHandle = aomStartup.registerChrome(manifestURI, [["content", "zotwatch", rootURI + "content/"]]);

  // Defines the global `ZotWatchBundle` in this sandbox
  Services.scriptloader.loadSubScriptWithOptions("chrome://zotwatch/content/zotwatch.js", {
    target: globalThis,
    ignoreCache: true,
  });
  await ZotWatchBundle.addon.startup({ id, version, rootURI });
}

function onMainWindowLoad({ window }) {
  globalThis.ZotWatchBundle?.addon.addToWindow(window);
}

function onMainWindowUnload({ window }) {
  globalThis.ZotWatchBundle?.addon.removeFromWindow(window);
}

// Zotero awaits the promise returned here, so a running pipeline can be
// cancelled and settled before the plugin scope goes away.
async function shutdown(data, reason) {
  const addon = globalThis.ZotWatchBundle?.addon;
  if (reason === APP_SHUTDOWN) {
    await addon?.shutdown({ appShutdown: true });
    return;
  }
  try {
    await addon?.shutdown();
  } finally {
    globalThis.ZotWatchBundle = undefined;
    if (chromeHandle) {
      chromeHandle.destruct();
      chromeHandle = null;
    }
  }
}

function uninstall() {}
