// Loads the chat page scripts (vendor libraries, i18n.js, chat.js) into a
// node:vm context with a minimal fake DOM, for test_chat_*.cjs.
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const chatRoot = path.join(__dirname, '../../ironmlx-app/Sources/IronMLXAppCore/Resources/chat');
const read = name => fs.readFileSync(path.join(chatRoot, name), 'utf8');

function camel(name) { return name.replace(/-([a-z])/g, (_, c) => c.toUpperCase()); }

function decode(text) {
  return text.replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'").replace(/&amp;/g, '&');
}

// One compound selector: tag, #id, .class (any number), [attr] or [attr="value"].
function matches(el, selector) {
  if (!el || !el.tagName) return false;
  const re = /([a-zA-Z][\w-]*)|#([\w-]+)|\.([\w-]+)|\[([\w-]+)(?:="([^"]*)")?\]/g;
  let m;
  while ((m = re.exec(selector))) {
    if (m[1] && el.tagName.toLowerCase() !== m[1].toLowerCase()) return false;
    if (m[2] && el.id !== m[2]) return false;
    if (m[3] && !el.classList.contains(m[3])) return false;
    if (m[4]) {
      const value = el.getAttribute(m[4]);
      if (value === null) return false;
      if (m[5] !== undefined && value !== m[5]) return false;
    }
  }
  return true;
}

// A descendant chain of compound selectors, such as `#history .it .it-time`.
function matchesChain(el, selector) {
  const parts = selector.trim().split(/\s+/);
  if (!matches(el, parts[parts.length - 1])) return false;
  let index = parts.length - 2;
  for (let node = el.parentNode; index >= 0 && node && node.nodeType === 1; node = node.parentNode) {
    if (matches(node, parts[index])) index -= 1;
  }
  return index < 0;
}

class FakeNode {
  constructor(doc) { this.ownerDocument = doc; this.parentNode = null; }
  remove() { if (this.parentNode) this.parentNode.removeChild(this); }
}

class FakeText extends FakeNode {
  constructor(doc, text) { super(doc); this.nodeType = 3; this.data = String(text); }
  get textContent() { return this.data; }
  set textContent(value) { this.data = String(value); }
}

class FakeElement extends FakeNode {
  constructor(doc, tag) {
    super(doc);
    this.nodeType = 1;
    this.tagName = tag.toUpperCase();
    this.childNodes = [];
    this.attributes = {};
    this.dataset = {};
    this.listeners = {};
    this.classes = new Set();
    this.style = { setProperty(key, value) { this[key] = value; } };
    this.html = null;
    this.value = '';
    this.disabled = false;
    this.hidden = false;
    this.files = [];
    this.scrollTop = 0;
    this.scrollHeight = 0;
    this.clientHeight = 0;
  }
  get children() { return this.childNodes.filter(node => node.nodeType === 1); }
  get firstChild() { return this.childNodes[0] || null; }
  get className() { return [...this.classes].join(' '); }
  set className(value) { this.classes = new Set(String(value).split(/\s+/).filter(Boolean)); }
  get classList() {
    const set = this.classes;
    return {
      add: (...names) => names.forEach(name => set.add(name)),
      remove: (...names) => names.forEach(name => set.delete(name)),
      contains: name => set.has(name),
      toggle: (name, force) => {
        const on = force === undefined ? !set.has(name) : Boolean(force);
        if (on) set.add(name); else set.delete(name);
        return on;
      },
    };
  }
  get id() { return this.attributes.id || ''; }
  set id(value) { this.attributes.id = String(value); }
  setAttribute(key, value) {
    if (key === 'class') { this.className = value; return; }
    this.attributes[key] = String(value);
    if (key.startsWith('data-')) this.dataset[camel(key.slice(5))] = String(value);
  }
  getAttribute(key) {
    if (key === 'class') return this.className;
    return Object.prototype.hasOwnProperty.call(this.attributes, key) ? this.attributes[key] : null;
  }
  hasAttribute(key) { return this.getAttribute(key) !== null; }
  removeAttribute(key) { delete this.attributes[key]; }
  appendChild(node) {
    if (node.parentNode) node.parentNode.removeChild(node);
    node.parentNode = this;
    this.childNodes.push(node);
    this.html = null;
    return node;
  }
  append(...nodes) {
    nodes.forEach(node => this.appendChild(typeof node === 'string' ? new FakeText(this.ownerDocument, node) : node));
  }
  insertBefore(node, ref) {
    if (!ref) return this.appendChild(node);
    if (node.parentNode) node.parentNode.removeChild(node);
    node.parentNode = this;
    this.childNodes.splice(this.childNodes.indexOf(ref), 0, node);
    return node;
  }
  removeChild(node) {
    const index = this.childNodes.indexOf(node);
    if (index >= 0) this.childNodes.splice(index, 1);
    node.parentNode = null;
    return node;
  }
  replaceChildren(...nodes) {
    this.childNodes.forEach(node => { node.parentNode = null; });
    this.childNodes = [];
    this.html = null;
    this.append(...nodes);
  }
  get textContent() {
    if (this.html !== null) return decode(this.html.replace(/<[^>]*>/g, ''));
    return this.childNodes.map(node => node.textContent).join('');
  }
  set textContent(value) {
    this.replaceChildren();
    if (value !== '' && value !== null && value !== undefined) this.append(String(value));
  }
  get innerHTML() { return this.html !== null ? this.html : ''; }
  set innerHTML(value) { this.replaceChildren(); this.html = String(value); }
  addEventListener(type, fn) { (this.listeners[type] = this.listeners[type] || []).push(fn); }
  removeEventListener(type, fn) { this.listeners[type] = (this.listeners[type] || []).filter(item => item !== fn); }
  dispatch(type, extra = {}) {
    const event = Object.assign({
      type, target: this, currentTarget: this, defaultPrevented: false,
      preventDefault() { this.defaultPrevented = true; }, stopPropagation() {},
    }, extra);
    (this.listeners[type] || []).slice().forEach(fn => fn(event));
    return event;
  }
  click() { if (!this.disabled) this.dispatch('click'); }
  focus() { this.ownerDocument.activeElement = this; }
  blur() {}
  select() {}
  getBoundingClientRect() { return { x: 0, y: 0, width: 0, height: 0, top: 0, left: 0, right: 0, bottom: 0 }; }
  contains(node) {
    for (let current = node; current; current = current.parentNode) if (current === this) return true;
    return false;
  }
  matches(selector) { return matchesChain(this, selector); }
  closest(selector) {
    for (let current = this; current && current.nodeType === 1; current = current.parentNode) {
      if (matches(current, selector)) return current;
    }
    return null;
  }
  querySelectorAll(selector) {
    const found = [];
    const walk = node => node.children.forEach(child => { if (matchesChain(child, selector)) found.push(child); walk(child); });
    walk(this);
    return found;
  }
  querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
}

class FakeDocument {
  constructor() {
    this.listeners = {};
    this.documentElement = new FakeElement(this, 'html');
    this.body = new FakeElement(this, 'body');
    this.documentElement.appendChild(this.body);
    this.activeElement = this.body;
    this.readyState = 'complete';
    this.compatMode = 'CSS1Compat';
  }
  createElement(tag) { return new FakeElement(this, tag); }
  createTextNode(text) { return new FakeText(this, text); }
  getElementById(id) { return this.documentElement.querySelector('#' + id); }
  querySelector(selector) { return this.documentElement.querySelector(selector); }
  querySelectorAll(selector) { return this.documentElement.querySelectorAll(selector); }
  addEventListener(type, fn) { (this.listeners[type] = this.listeners[type] || []).push(fn); }
  removeEventListener() {}
  dispatch(type, extra = {}) {
    const event = Object.assign({
      type, target: this.body, defaultPrevented: false,
      preventDefault() { this.defaultPrevented = true; }, stopPropagation() {},
    }, extra);
    (this.listeners[type] || []).slice().forEach(fn => fn(event));
    return event;
  }
}

/// Reads finish at once, or wait in `reads` while `reads.held` is true.
function fileReaderClass(reads) {
  return class FakeFileReader {
    readAsDataURL(file) {
      const finish = () => {
        this.result = 'data:' + file.type + ';base64,' + Buffer.from(file.name).toString('base64');
        if (this.onload) this.onload({ target: this });
      };
      if (reads.held) reads.push(finish); else finish();
    }
  };
}

/// Loads the page. `lang` and `now` (ms) are optional.
function loadChat({ now = Date.now() } = {}) {
  const doc = new FakeDocument();
  const posted = [];
  const alerts = [];
  const purified = [];
  const queue = [];
  const reads = [];
  let clock = now;
  const window = {
    document: doc,
    navigator: { language: 'en' },
    console,
    webkit: { messageHandlers: { chat: { postMessage: message => posted.push(JSON.parse(JSON.stringify(message))) } } },
    alert: message => alerts.push(message),
    confirm: message => { alerts.push(message); return true; },
    setTimeout: (fn) => { queue.push(fn); return queue.length; },
    clearTimeout() {},
    setInterval: () => 0,
    clearInterval() {},
    requestAnimationFrame: fn => { queue.push(fn); return queue.length; },
    cancelAnimationFrame() {},
    addEventListener() {},
    FileReader: fileReaderClass(reads),
    __IRONCHAT_NOW__: () => clock,
  };
  window.window = window;
  window.self = window;
  const context = vm.createContext(window);
  for (const name of ['vendor/marked.min.js', 'vendor/purify.min.js', 'vendor/highlight.min.js', 'vendor/katex.min.js']) {
    vm.runInContext(read(name), context, { filename: name });
  }
  // DOMPurify needs a real DOM; under node it reports itself unsupported and
  // has no `sanitize`. The page must still route every model reply through
  // it, so a recording pass-through stands in here; the real library runs in
  // WebKit (ChatWindowControllerTests.pageSanitizesWithDOMPurify).
  const purifySupported = window.DOMPurify.isSupported;
  window.DOMPurify.sanitize = (html) => { purified.push(html); return html; };
  vm.runInContext(read('i18n.js'), context, { filename: 'i18n.js' });
  vm.runInContext(read('chat.js'), context, { filename: 'chat.js' });

  const page = {
    window, doc, posted, alerts, purified, purifySupported,
    i18n: window.CHAT_I18N,
    receive: message => window.ironChat.receive(JSON.parse(JSON.stringify(message))),
    flush() { while (queue.length) queue.shift()(); },
    holdReads() { reads.held = true; },
    finishReads() { reads.held = false; while (reads.length) reads.shift()(); },
    advance(ms) { clock += ms; },
    $: selector => doc.querySelector(selector),
    $$: selector => doc.querySelectorAll(selector),
    last: type => posted.filter(message => !type || message.type === type).slice(-1)[0],
    sent: type => posted.filter(message => message.type === type),
    state(overrides = {}) {
      const base = {
        type: 'state', language: 'zh-Hans',
        models: [{ id: 'mlx-community/Qwen3.5-4B-MLX-4bit', name: 'Qwen3.5 4B', sizeBytes: 3061132920, loaded: true, current: true, pinned: false }],
        model: { id: 'mlx-community/Qwen3.5-4B-MLX-4bit', name: 'Qwen3.5 4B', status: 'ready' },
        reasoning: null, reasoningChoice: 'off', supportsVision: true, generating: false,
      };
      page.receive(Object.assign(base, overrides));
      page.flush();
    },
    type(text) {
      const input = doc.querySelector('#input');
      input.value = text;
      input.dispatch('input');
    },
    send(text) {
      page.type(text);
      doc.querySelector('#send').click();
      page.flush();
    },
  };
  return page;
}

module.exports = { loadChat };
