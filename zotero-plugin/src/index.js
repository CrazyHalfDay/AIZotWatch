// Bundle entry. esbuild wraps this module into the global `ZotWatchBundle`,
// which addon/bootstrap.js loads into the plugin sandbox.

import { ZotWatchAddon } from "./zotero/addon.js";

export const addon = new ZotWatchAddon();
