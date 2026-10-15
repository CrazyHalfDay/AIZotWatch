// Build the plugin: bundle src/ with esbuild, copy addon/, and zip it into an .xpi.
// Output: build/addon/ (unpacked), build/zotwatch-<version>.xpi, build/zotwatch-update.json

import { build } from "esbuild";
import { cpSync, mkdirSync, readFileSync, readdirSync, rmSync, statSync, writeFileSync } from "node:fs";
import { dirname, join, relative, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { deflateRawSync } from "node:zlib";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const pkg = JSON.parse(readFileSync(join(root, "package.json"), "utf8"));
const version = pkg.version;
const outDir = join(root, "build");
const addonOut = join(outDir, "addon");
const ADDON_ID = "zotwatch@crazyhalfday.github.io";
const TARGET = "firefox115"; // Zotero 7 runs on Firefox 115 ESR
const RELEASE_BASE = "https://github.com/CrazyHalfDay/AIZotWatch/releases/download";

function walk(dir) {
  return readdirSync(dir).flatMap((name) => {
    const p = join(dir, name);
    return statSync(p).isDirectory() ? walk(p) : [p];
  });
}

// ---------------------------------------------------------------- zip writer

const CRC_TABLE = Array.from({ length: 256 }, (_, n) => {
  let c = n;
  for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
  return c >>> 0;
});

function crc32(buf) {
  let c = 0xffffffff;
  for (const byte of buf) c = CRC_TABLE[(c ^ byte) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

function zip(files) {
  // Fixed timestamp (1980-01-01) keeps builds reproducible
  const DOS_TIME = 0;
  const DOS_DATE = (0 << 9) | (1 << 5) | 1;
  const locals = [];
  const centrals = [];
  let offset = 0;
  for (const { name, data } of files) {
    const nameBuf = Buffer.from(name, "utf8");
    const compressed = deflateRawSync(data, { level: 9 });
    const useDeflate = compressed.length < data.length;
    const body = useDeflate ? compressed : data;
    const crc = crc32(data);

    const local = Buffer.alloc(30);
    local.writeUInt32LE(0x04034b50, 0);
    local.writeUInt16LE(20, 4);
    local.writeUInt16LE(0x0800, 6);
    local.writeUInt16LE(useDeflate ? 8 : 0, 8);
    local.writeUInt16LE(DOS_TIME, 10);
    local.writeUInt16LE(DOS_DATE, 12);
    local.writeUInt32LE(crc, 14);
    local.writeUInt32LE(body.length, 18);
    local.writeUInt32LE(data.length, 22);
    local.writeUInt16LE(nameBuf.length, 26);
    local.writeUInt16LE(0, 28);
    locals.push(local, nameBuf, body);

    const central = Buffer.alloc(46);
    central.writeUInt32LE(0x02014b50, 0);
    central.writeUInt16LE(20, 4);
    central.writeUInt16LE(20, 6);
    central.writeUInt16LE(0x0800, 8);
    central.writeUInt16LE(useDeflate ? 8 : 0, 10);
    central.writeUInt16LE(DOS_TIME, 12);
    central.writeUInt16LE(DOS_DATE, 14);
    central.writeUInt32LE(crc, 16);
    central.writeUInt32LE(body.length, 20);
    central.writeUInt32LE(data.length, 24);
    central.writeUInt16LE(nameBuf.length, 28);
    central.writeUInt32LE(offset, 42);
    centrals.push(central, nameBuf);

    offset += local.length + nameBuf.length + body.length;
  }
  const centralBuf = Buffer.concat(centrals);
  const end = Buffer.alloc(22);
  end.writeUInt32LE(0x06054b50, 0);
  end.writeUInt16LE(files.length, 8);
  end.writeUInt16LE(files.length, 10);
  end.writeUInt32LE(centralBuf.length, 12);
  end.writeUInt32LE(offset, 16);
  return Buffer.concat([...locals, centralBuf, end]);
}

// --------------------------------------------------------------------- build

rmSync(outDir, { recursive: true, force: true });
mkdirSync(outDir, { recursive: true });
cpSync(join(root, "addon"), addonOut, { recursive: true });

const manifestPath = join(addonOut, "manifest.json");
writeFileSync(manifestPath, readFileSync(manifestPath, "utf8").replace("__VERSION__", version));
const zoteroApp = JSON.parse(readFileSync(manifestPath, "utf8")).applications.zotero;

await build({
  entryPoints: [join(root, "src", "index.js")],
  bundle: true,
  format: "iife",
  globalName: "ZotWatchBundle",
  target: TARGET,
  charset: "ascii",
  legalComments: "none",
  outfile: join(addonOut, "content", "zotwatch.js"),
  logLevel: "warning",
});

// Scripts loaded through loadSubScript are kept ASCII-only. Every non-ASCII
// UTF-16 unit becomes a \\uXXXX escape, which is valid in strings, templates,
// regular expressions and comments alike (our identifiers are all ASCII).
const asciiEscape = (code) => code.replace(/[^\x00-\x7f]/g, (ch) => `\\u${ch.charCodeAt(0).toString(16).padStart(4, "0")}`);
for (const file of walk(addonOut).filter((f) => f.endsWith(".js"))) {
  writeFileSync(file, asciiEscape(readFileSync(file, "utf8")));
}

const files = walk(addonOut)
  .map((f) => ({ name: relative(addonOut, f).split(sep).join("/"), data: readFileSync(f) }))
  .sort((a, b) => a.name.localeCompare(b.name));
const xpiName = `zotwatch-${version}.xpi`;
writeFileSync(join(outDir, xpiName), zip(files));

const update = {
  addons: {
    [ADDON_ID]: {
      updates: [
        {
          version,
          update_link: `${RELEASE_BASE}/zotero-plugin-v${version}/${xpiName}`,
          // Must match manifest.json: Firefox applies update-manifest ranges as
          // compatibility updates, so a missing max would silently become "*"
          applications: {
            zotero: { strict_min_version: zoteroApp.strict_min_version, strict_max_version: zoteroApp.strict_max_version },
          },
        },
      ],
    },
  },
};
writeFileSync(join(outDir, "zotwatch-update.json"), `${JSON.stringify(update, null, 2)}\n`);

console.log(`Built build/${xpiName} (${files.length} files)`);
