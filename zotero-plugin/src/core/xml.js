// Minimal namespace-agnostic XML parser, enough for Atom (arXiv) and OAI-PMH
// responses. Works identically in Zotero and in Node tests (no DOMParser).
// Element names keep their prefix in `qname`; `name` is the local name.

import { decodeEntities } from "./text.js";

class XmlElement {
  constructor(qname, attrs) {
    this.qname = qname;
    const idx = qname.indexOf(":");
    this.name = idx >= 0 ? qname.slice(idx + 1) : qname;
    this.attrs = attrs;
    this.children = [];
  }

  /** Concatenated text content of this element and its descendants. */
  get text() {
    let out = "";
    for (const child of this.children) {
      out += typeof child === "string" ? child : child.text;
    }
    return out;
  }

  get elements() {
    return this.children.filter((c) => typeof c !== "string");
  }

  /** First direct child with the given local name. */
  child(name) {
    return this.elements.find((e) => e.name === name) || null;
  }

  /** All direct children with the given local name. */
  childrenNamed(name) {
    return this.elements.filter((e) => e.name === name);
  }

  /** All descendants with the given local name (document order). */
  findAll(name, out = []) {
    for (const el of this.elements) {
      if (el.name === name) out.push(el);
      el.findAll(name, out);
    }
    return out;
  }

  find(name) {
    for (const el of this.elements) {
      if (el.name === name) return el;
      const nested = el.find(name);
      if (nested) return nested;
    }
    return null;
  }

  attr(name) {
    if (name in this.attrs) return this.attrs[name];
    for (const [key, value] of Object.entries(this.attrs)) {
      const idx = key.indexOf(":");
      if (idx >= 0 && key.slice(idx + 1) === name) return value;
    }
    return null;
  }
}

const ATTR_RE = /([^\s=/>]+)\s*=\s*("([^"]*)"|'([^']*)')/g;

function parseAttrs(source) {
  const attrs = {};
  ATTR_RE.lastIndex = 0;
  let m;
  while ((m = ATTR_RE.exec(source))) {
    attrs[m[1]] = decodeEntities(m[3] !== undefined ? m[3] : m[4]);
  }
  return attrs;
}

/**
 * Parse an XML string into a tree and return the root element.
 * With `strict`, mismatched or unclosed tags throw (used to validate our own XHTML).
 */
export function parseXml(xml, { strict = false } = {}) {
  const root = new XmlElement("#document", {});
  const stack = [root];
  let i = 0;
  const len = xml.length;

  const top = () => stack[stack.length - 1];

  while (i < len) {
    const lt = xml.indexOf("<", i);
    if (lt < 0) {
      top().children.push(decodeEntities(xml.slice(i)));
      break;
    }
    if (lt > i) top().children.push(decodeEntities(xml.slice(i, lt)));

    if (xml.startsWith("<!--", lt)) {
      const end = xml.indexOf("-->", lt + 4);
      if (end < 0) {
        if (strict) throw new Error("Unterminated comment");
        break;
      }
      i = end + 3;
    } else if (xml.startsWith("<![CDATA[", lt)) {
      const end = xml.indexOf("]]>", lt + 9);
      if (end < 0) {
        if (strict) throw new Error("Unterminated CDATA section");
        break;
      }
      top().children.push(xml.slice(lt + 9, end));
      i = end + 3;
    } else if (xml.startsWith("<?", lt)) {
      const end = xml.indexOf("?>", lt + 2);
      i = end < 0 ? len : end + 2;
    } else if (xml.startsWith("<!", lt)) {
      // DOCTYPE (possibly with an internal subset in brackets)
      let depth = 0;
      let j = lt + 2;
      for (; j < len; j++) {
        const ch = xml[j];
        if (ch === "[") depth++;
        else if (ch === "]") depth--;
        else if (ch === ">" && depth <= 0) break;
      }
      i = j + 1;
    } else if (xml[lt + 1] === "/") {
      const end = xml.indexOf(">", lt);
      const qname = xml.slice(lt + 2, end < 0 ? len : end).trim();
      i = end < 0 ? len : end + 1;
      // Pop to the matching element (tolerates sloppy markup when not strict)
      let idx = stack.length - 1;
      while (idx > 0 && stack[idx].qname !== qname) idx--;
      if (idx === 0 || idx !== stack.length - 1) {
        if (strict) throw new Error(`Mismatched closing tag </${qname}> (open: <${top().qname}>)`);
      }
      if (idx > 0) stack.length = idx;
    } else {
      // Start tag: find the closing ">" outside of quoted attribute values
      let j = lt + 1;
      let quote = null;
      for (; j < len; j++) {
        const ch = xml[j];
        if (quote) {
          if (ch === quote) quote = null;
        } else if (ch === '"' || ch === "'") {
          quote = ch;
        } else if (ch === ">") {
          break;
        }
      }
      if (j >= len && strict) throw new Error("Unterminated start tag");
      let body = xml.slice(lt + 1, j);
      const selfClosing = body.endsWith("/");
      if (selfClosing) body = body.slice(0, -1);
      const nameMatch = body.match(/^[^\s/>]+/);
      if (!nameMatch) {
        if (strict) throw new Error(`Invalid tag at offset ${lt}`);
        i = j + 1;
        continue;
      }
      const el = new XmlElement(nameMatch[0], parseAttrs(body.slice(nameMatch[0].length)));
      top().children.push(el);
      if (!selfClosing) stack.push(el);
      i = j + 1;
    }
  }

  if (strict && stack.length > 1) {
    throw new Error(`Unclosed tag <${top().qname}>`);
  }
  const rootEl = root.elements[0];
  if (!rootEl) throw new Error("No root element");
  return rootEl;
}
