const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const html = fs.readFileSync(path.join(__dirname,
  '../../ironmlx-app/Sources/IronMLXAppCore/Resources/dashboard2.html'), 'utf8');
const extract = (start, end) => html.slice(html.indexOf(`  function ${start}(`),
  html.indexOf(`  function ${end}(`));
function slice(start, end) {
  const from = html.indexOf(start);
  assert.ok(from >= 0, `missing ${start}`);
  return html.slice(from, html.indexOf(end, from));
}

const MODEL = 'mlx-community/Qwen3.6-35B-A3B-4bit';
const GB = 1024 ** 3;
// One load keeps its start time across updates.
const STARTED = Date.now() - 5000;

// A small DOM for the panel, indicator and overlay: parses the markup the
// page writes, so tests can hold element nodes and click real buttons.
const VOID_TAGS = new Set(['br', 'img', 'input', 'meta', 'link', 'hr']);
function escapeText(value) {
  return String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
function camel(name) { return name.replace(/-([a-z])/g, (_, c) => c.toUpperCase()); }
function decode(value) {
  return value.replace(/&quot;/g, '"').replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&amp;/g, '&');
}
class FakeText {
  constructor(text) { this.nodeType = 3; this.text = text; this.parentNode = null; }
  get outerHTML() { return this.text; }
  get textContent() { return decode(this.text); }
}
class FakeElement {
  constructor(doc, tag) {
    this.doc = doc; this.nodeType = 1; this.tagName = tag.toUpperCase();
    this.childNodes = []; this.attributes = new Map(); this.classes = new Set();
    this.dataset = {}; this.style = {}; this.hidden = false; this.disabled = false; this.parentNode = null;
  }
  get classList() {
    const set = this.classes;
    return {
      add: (...names) => names.forEach(name => set.add(name)),
      remove: (...names) => names.forEach(name => set.delete(name)),
      contains: name => set.has(name),
      toggle(name, force) {
        const on = force === undefined ? !set.has(name) : !!force;
        if (on) set.add(name); else set.delete(name);
        return on;
      },
    };
  }
  setAttribute(name, value) {
    value = String(value);
    if (name === 'class') { this.classes = new Set(value.split(/\s+/).filter(Boolean)); this.attributes.set('class', ''); return; }
    this.attributes.set(name, value);
    if (name.startsWith('data-')) this.dataset[camel(name.slice(5))] = decode(value);
    if (name === 'disabled') this.disabled = true;
  }
  getAttribute(name) {
    if (name === 'class') return [...this.classes].join(' ');
    return this.attributes.has(name) ? this.attributes.get(name) : null;
  }
  removeAttribute(name) { this.attributes.delete(name); }
  addEventListener() {}
  focus() {}
  appendChild(node) { node.parentNode = this; this.childNodes.push(node); return node; }
  get outerHTML() {
    const tag = this.tagName.toLowerCase();
    const attrs = [...this.attributes].map(([name, value]) => name === 'class'
      ? ` class="${this.getAttribute('class')}"` : (value === '' && name === 'disabled' ? ' disabled' : ` ${name}="${value}"`)).join('');
    return `<${tag}${attrs}>${this.innerHTML}</${tag}>`;
  }
  get innerHTML() { return this.childNodes.map(node => node.outerHTML).join(''); }
  set innerHTML(markup) {
    this.childNodes.forEach(node => { node.parentNode = null; });
    this.childNodes = [];
    const stack = [this];
    const re = /<\/([a-zA-Z][\w:-]*)\s*>|<([a-zA-Z][\w:-]*)((?:\s+[^\s=>\/]+(?:="[^"]*")?)*)\s*(\/?)>|([^<]+)/g;
    let match;
    while ((match = re.exec(String(markup)))) {
      const top = stack[stack.length - 1];
      if (match[1]) {
        const index = stack.map(el => el.tagName).lastIndexOf(match[1].toUpperCase());
        if (index > 0) stack.length = index;
      } else if (match[2]) {
        const el = new FakeElement(this.doc, match[2]);
        for (const [, name, value] of match[3].matchAll(/([^\s=]+)(?:="([^"]*)")?/g)) el.setAttribute(name, value === undefined ? '' : value);
        top.appendChild(el);
        if (!match[4] && !VOID_TAGS.has(match[2].toLowerCase())) stack.push(el);
      } else if (match[5]) {
        top.appendChild(new FakeText(match[5]));
      }
    }
  }
  get textContent() { return this.childNodes.map(node => node.textContent).join(''); }
  set textContent(value) { this.innerHTML = escapeText(value); }
  matches(selector) {
    const re = /([a-zA-Z][\w-]*)|\.([\w-]+)|\[([\w-]+)(?:="([^"]*)")?\]/g;
    let m;
    while ((m = re.exec(selector))) {
      if (m[1] && this.tagName !== m[1].toUpperCase()) return false;
      if (m[2] && !this.classes.has(m[2])) return false;
      if (m[3] && (this.getAttribute(m[3]) === null || (m[4] !== undefined && this.getAttribute(m[3]) !== m[4]))) return false;
    }
    return true;
  }
  querySelectorAll(selector) {
    const found = [];
    const walk = node => node.childNodes.forEach(child => {
      if (child.nodeType !== 1) return;
      if (child.matches(selector)) found.push(child);
      walk(child);
    });
    walk(this);
    return found;
  }
  querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
  contains(node) { for (let at = node; at; at = at.parentNode) if (at === this) return true; return false; }
  click() {
    const code = this.getAttribute('onclick');
    if (code && !this.disabled) vm.runInContext('Function', this.doc.context)('event', code).call(this, { target: this });
  }
}

function page(lang = 'zh-Hans') {
  const elements = new Map();
  const posted = [];
  const toasts = [];
  const rows = [];
  const navigation = [];
  let syncs = 0;
  const doc = {};
  const intervals = [];
  // Fake timeouts: they run only when a test calls `runTimers()`.
  let timeouts = new Map();
  let nextTimeout = 1;
  function element(id) {
    if (!elements.has(id)) {
      const el = new FakeElement(doc, 'div');
      el.setAttribute('id', id);
      el.hidden = true;
      elements.set(id, el);
    }
    return elements.get(id);
  }
  const tbody = { innerHTML: '', appendChild(row) { rows.push(row); } };
  let unloads = 0;
  const handler = type => ({ postMessage(body) { posted.push({ type, body }); } });
  const context = {
    currentLang: lang,
    console,
    window: {
      __MODEL_INTEGRITY_STATUS__: {},
      __LOCAL_MODELS__: {},
      webkit: { messageHandlers: {
        cancelModelLoad: handler('cancelModelLoad'),
        openModelLoadPanel: handler('openModelLoadPanel'),
        acknowledgeModelLoadFailure: handler('acknowledgeModelLoadFailure'),
        scanLocalModels: handler('scanLocalModels'),
      } },
    },
    document: {
      getElementById: element,
      querySelector: selector => selector === '#tab-models-manage tbody' ? tbody : null,
      querySelectorAll: selector => selector === '#tab-models-manage tbody tr' ? rows.slice() : [],
      addEventListener() {},
      createElement: tag => new FakeElement(doc, tag),
    },
    setInterval(fn) { intervals.push(fn); return intervals.length; },
    clearInterval(id) { intervals[id - 1] = null; },
    setTimeout(fn) { timeouts.set(nextTimeout, fn); return nextTimeout++; },
    clearTimeout(id) { timeouts.delete(id); },
    showToast: (message, type) => toasts.push({ message, type }),
    syncModelList() { syncs += 1; },
    unloadModel() { unloads += 1; },
    loadModel() {},
    navigateTo: page => navigation.push(page),
    escapeAttr: value => String(value || '').replace(/&/g, '&amp;').replace(/"/g, '&quot;')
      .replace(/</g, '&lt;').replace(/>/g, '&gt;'),
    escapeHTML: value => String(value).replace(/[&<>"']/g, c => `&#${c.charCodeAt(0)};`),
    closeModelMoreActions() {}, refreshSupportedModelActions() {}, hideModelTableTooltip() {},
    renderProfileModelOptions() {}, syncBenchmarkModels() {},
    renderMtpBadge: () => '', renderDFlash2Badge: () => '', renderModelType: () => '',
    renderQuantBadge: () => '', renderModelMoreActions: () => '',
    modelColumnTooltipAttributes: () => '', pinTooltip: () => '',
    formatSize: mb => `${mb} MB`,
    showSidebarTooltip() {}, hideSidebarTooltip() {},
  };
  vm.createContext(context);
  doc.context = context;
  vm.runInContext(slice('  const I18N =', '  const LANG_MAP =')
    + slice('  function t(key, fallback)', '  function setLanguage(')
    + slice('  // ── Model load progress', "  window.__CURRENT_PARAM_MODEL__ = '';")
    + slice('  function modelReadiness(', '  function updateModelIntegrityRow(')
    + slice('  function renderModelLoadActions(', '  // Called from Rust with local model scan results')
    + extract('onLocalModelsScanned', 'renderProfileModelOptions')
    + extract('toggleModelLoad', 'markModelActionLoading'), context);
  return {
    context, element, posted, toasts, rows, navigation,
    get syncs() { return syncs; },
    get unloads() { return unloads; },
    push(...loads) { context.onModelLoadsChanged(JSON.stringify({ loads })); },
    tick() { intervals.forEach(fn => fn && fn()); },
    get pendingTimers() { return timeouts.size; },
    runTimers() { const due = timeouts; timeouts = new Map(); due.forEach(fn => fn()); },
    scan(models) { rows.length = 0; context.onLocalModelsScanned(JSON.stringify(models)); return rows; },
  };
}

function activity(overrides = {}) {
  return {
    model: MODEL, variant: 'plain', kind: 'load', origin: 'dashboard', state: 'loading',
    stage: 'reading_weights', completedStages: [], bytesDone: 0, bytesTotal: 0,
    startedAt: STARTED, cancellable: true, autoOpenPanel: false,
    ...overrides,
  };
}

function scannedModel(overrides = {}) {
  return { id: MODEL, display_name: MODEL, type: 'llm', loaded: false, size_mb: 20070,
    readiness: { status: 'ready' }, ...overrides };
}

test('row_state_survives_rerender', () => {
  const p = page();
  p.push(activity({ bytesDone: 62 * GB, bytesTotal: 100 * GB }));
  // Switching pages re-renders the model list from the disk scan, where the
  // model is not loaded yet; the row must keep the load state.
  const [row] = p.scan([scannedModel()]);
  assert.match(row.innerHTML, /model-status-text">加载中<[\s\S]*62%/);
  assert.match(row.innerHTML, /onclick="cancelModelLoadFromUI\(this\.dataset\.model\)"[^>]*>取消</);
  assert.doesNotMatch(row.innerHTML, /toggleModelLoad/);
  assert.match(row.innerHTML, /model-load-rowprogress[\s\S]*width:62%/);

  // A reload keeps the loaded model's row and says so.
  p.push(activity({ kind: 'reload', startedAt: STARTED + 1, bytesDone: 4 * GB, bytesTotal: 10 * GB }));
  const [reloaded] = p.scan([scannedModel({ loaded: true })]);
  assert.match(reloaded.innerHTML, /重新加载中<[\s\S]*40%/);
  assert.match(reloaded.innerHTML, />取消</);
  assert.doesNotMatch(reloaded.innerHTML, /action-unload/);

  // Queued loads show the queue and can be cancelled.
  p.push(activity({ state: 'queued', stage: 'queued', startedAt: Date.now() }));
  const [queued] = p.scan([scannedModel()]);
  assert.match(queued.innerHTML, /排队中/);
  assert.match(queued.innerHTML, />取消</);

  // Once the load is gone the row falls back to the scan.
  p.push();
  const [idle] = p.scan([scannedModel()]);
  assert.match(idle.innerHTML, /toggleModelLoad/);
  assert.doesNotMatch(idle.innerHTML, /cancelModelLoadFromUI/);
});

test('percentages never decrease within one load', () => {
  const p = page();
  const startedAt = Date.now() - 1000;
  p.push(activity({ startedAt, bytesDone: 62 * GB, bytesTotal: 100 * GB }));
  // The draft's exact size arrives late and grows the total.
  p.push(activity({ startedAt, bytesDone: 62 * GB, bytesTotal: 120 * GB }));
  assert.match(p.scan([scannedModel()])[0].innerHTML, /<span>62%<\/span>/);
  assert.match(p.element('model-load-indicator').innerHTML, /62%/);
  // A new load starts from its own progress.
  p.push(activity({ startedAt: startedAt + 5000, bytesDone: 10 * GB, bytesTotal: 100 * GB }));
  assert.match(p.scan([scannedModel()])[0].innerHTML, /<span>10%<\/span>/);
});

test('cancel button sends the model and shows cancelling at once', () => {
  const p = page();
  p.push(activity({ bytesDone: 3 * GB, bytesTotal: 10 * GB }));
  p.context.cancelModelLoadFromUI(MODEL);
  assert.deepEqual(p.posted.filter(m => m.type === 'cancelModelLoad'), [{ type: 'cancelModelLoad', body: MODEL }]);
  const [row] = p.scan([scannedModel()]);
  assert.match(row.innerHTML, /正在取消…/);
  assert.match(row.innerHTML, /disabled/);
});

test('eta_hidden_until_ten_percent_and_three_seconds', () => {
  const p = page();
  const now = 1_000_000;
  const base = { startedAt: now - 10_000, stage: 'reading_weights', bytesTotal: 100 * GB };
  const early = p.context.formatLoadEta(activity({ ...base, bytesDone: 5 * GB,
    completedStages: [{ stage: 'preparing_memory', elapsedMs: 1000 }] }), now);
  assert.equal(early.elapsed, '0:10');
  assert.equal(early.remaining, '', 'below ten percent');
  assert.equal(early.text, '已用 0:10');

  const quick = p.context.formatLoadEta(activity({ ...base, bytesDone: 20 * GB,
    completedStages: [{ stage: 'preparing_memory', elapsedMs: 8000 }] }), now);
  assert.equal(quick.remaining, '', 'under three seconds in the stage');

  const ready = p.context.formatLoadEta(activity({ ...base, bytesDone: 20 * GB,
    completedStages: [{ stage: 'preparing_memory', elapsedMs: 1000 }] }), now);
  assert.equal(ready.remaining, '0:36');
  assert.equal(ready.text, '已用 0:10 · 约剩 0:36');
});

test('eta_only_in_weight_stages', () => {
  const p = page();
  const now = 2_000_000;
  for (const stage of ['preparing_memory', 'building_model', 'starting_engine', 'warming_up']) {
    const eta = p.context.formatLoadEta(activity({ stage, startedAt: now - 60_000,
      bytesDone: 50 * GB, bytesTotal: 100 * GB }), now);
    assert.equal(eta.remaining, '', stage);
    assert.equal(eta.elapsed, '1:00');
  }
  const draft = p.context.formatLoadEta(activity({ variant: 'dflash2', stage: 'reading_draft_weights',
    startedAt: now - 12_000, bytesDone: 60 * GB, bytesTotal: 100 * GB,
    completedStages: [{ stage: 'reading_weights', elapsedMs: 6000 }, { stage: 'building_model', elapsedMs: 2000 }] }), now);
  assert.equal(draft.remaining, '0:07', 'rate over both weight-reading stages');
  const cancelling = p.context.formatLoadEta(activity({ state: 'cancelling', startedAt: now - 20_000,
    bytesDone: 50 * GB, bytesTotal: 100 * GB }), now);
  assert.equal(cancelling.remaining, '');
});

test('panel_auto_opens_only_for_dashboard_origin', () => {
  const p = page();
  const overlay = p.element('model-load-overlay');
  p.push(activity({ origin: 'chat', bytesDone: GB, bytesTotal: 10 * GB }));
  assert.equal(overlay.hidden, true);
  assert.equal(p.posted.some(m => m.type === 'openModelLoadPanel' && m.body === MODEL), false);

  p.push(activity({ origin: 'dashboard', autoOpenPanel: true, bytesDone: GB, bytesTotal: 10 * GB }));
  assert.equal(overlay.hidden, false);
  assert.ok(p.posted.some(m => m.type === 'openModelLoadPanel' && m.body === MODEL));
  assert.match(p.element('model-load-panel').innerHTML, /正在加载模型/);
  assert.match(p.element('model-load-panel').innerHTML, /后台继续/);

  // Closing (×, Esc, "continue in background") never cancels, and a push
  // that still carries the flag does not reopen the same load.
  p.context.closeModelLoadPanel();
  assert.equal(overlay.hidden, true);
  p.push(activity({ origin: 'dashboard', autoOpenPanel: true, bytesDone: 2 * GB, bytesTotal: 10 * GB }));
  assert.equal(overlay.hidden, true);
  assert.equal(p.posted.some(m => m.type === 'cancelModelLoad'), false);

  // The indicator reopens it.
  p.context.openModelLoadPanel(MODEL);
  assert.equal(overlay.hidden, false);
});

test('panel closes when its load finishes', () => {
  const p = page();
  p.push(activity({ autoOpenPanel: true, bytesDone: GB, bytesTotal: 10 * GB }));
  assert.equal(p.element('model-load-overlay').hidden, false);
  p.push(activity({ state: 'loaded', bytesDone: 10 * GB, bytesTotal: 10 * GB }));
  assert.equal(p.element('model-load-overlay').hidden, true);
  assert.ok(p.syncs > 0, 'the model list refreshes from the scan');
});

test('indicator_shows_queue_count', () => {
  const p = page();
  const indicator = p.element('model-load-indicator');
  p.push(activity({ bytesDone: 74 * GB, bytesTotal: 100 * GB }),
    activity({ model: 'mlx-community/gemma-4-e4b-it-4bit', state: 'queued', stage: 'queued' }));
  assert.equal(indicator.hidden, false);
  assert.match(indicator.innerHTML, /正在加载 2 个模型/);
  assert.match(indicator.innerHTML, /74%/);
  assert.match(indicator.innerHTML, /另 1 个排队/);

  p.push(activity({ bytesDone: 81 * GB, bytesTotal: 100 * GB }));
  assert.match(indicator.innerHTML, /正在加载 · 81%/);
  assert.match(indicator.innerHTML, /Qwen3\.6-35B-A3B-4bit/);

  p.push();
  assert.equal(indicator.hidden, true);
});

test('indicator_turns_red_on_unacknowledged_failure', () => {
  const p = page();
  const indicator = p.element('model-load-indicator');
  p.push(activity({ bytesDone: GB, bytesTotal: 10 * GB }));
  p.push(activity({ state: 'failed', stage: 'preparing_memory', cancellable: false,
    message: 'not enough memory', completedStages: [{ stage: 'verifying_files', elapsedMs: 14000 }] }));
  assert.equal(indicator.hidden, false);
  assert.ok(indicator.classList.contains('is-failed'));
  assert.match(indicator.innerHTML, /加载失败/);

  // It stays red until the user opens it; closing the failure acknowledges it.
  p.push(activity({ state: 'failed', stage: 'preparing_memory', cancellable: false, message: 'not enough memory' }));
  assert.ok(indicator.classList.contains('is-failed'));
  p.context.openModelLoadPanel(MODEL);
  const panel = p.element('model-load-panel').innerHTML;
  assert.match(panel, /加载失败/);
  assert.match(panel, /not enough memory/);
  assert.match(panel, /查看日志/);
  p.context.closeModelLoadPanel();
  assert.deepEqual(p.posted.filter(m => m.type === 'acknowledgeModelLoadFailure'),
    [{ type: 'acknowledgeModelLoadFailure', body: MODEL }]);
});

test('cancel_reload_shows_previous_version_kept_toast', () => {
  const p = page();
  p.push(activity({ kind: 'reload', bytesDone: GB, bytesTotal: 10 * GB }));
  p.push(activity({ kind: 'reload', state: 'cancelled', code: 'reload_cancelled', cancellable: false }));
  assert.equal(p.toasts.length, 1);
  assert.match(p.toasts[0].message, /已取消重新加载 Qwen3\.6-35B-A3B-4bit，继续使用原版本/);
});

test('cancel_load_shows_memory_released_toast', () => {
  const p = page();
  // A cancelled load seen for the first time (Dashboard reopened) says nothing.
  p.push(activity({ model: 'other', state: 'cancelled', code: 'load_cancelled', cancellable: false }));
  assert.equal(p.toasts.length, 0);

  p.push(activity({ autoOpenPanel: true, bytesDone: GB, bytesTotal: 10 * GB }));
  p.push(activity({ state: 'cancelling', bytesDone: GB, bytesTotal: 10 * GB, cancellable: false }));
  assert.match(p.element('model-load-panel').innerHTML, /正在取消…/);
  p.push(activity({ state: 'cancelled', code: 'load_cancelled', cancellable: false }));
  assert.equal(p.toasts.length, 1);
  assert.equal(p.toasts[0].message, '已取消加载 Qwen3.6-35B-A3B-4bit，占用的内存已释放');
  assert.equal(p.element('model-load-overlay').hidden, true, 'the panel closes after cancelling');
  p.push(activity({ state: 'cancelled', code: 'load_cancelled', cancellable: false }));
  assert.equal(p.toasts.length, 1, 'one toast per cancellation');
});

test('dflash2 cancellations explain what the backend restored', () => {
  for (const [code, pattern] of [
    ['dflash2_cancelled_restored', /恢复之前的配置/],
    ['dflash2_startup_cancelled', /普通模式/],
  ]) {
    const p = page();
    p.push(activity({ variant: 'dflash2', stage: 'restarting_backend' }));
    p.push(activity({ variant: 'dflash2', state: 'cancelled', code, cancellable: false }));
    assert.match(p.toasts[0].message, pattern, code);
  }
});

test('dflash2_panel_lists_six_stages_with_progress', () => {
  const p = page();
  const panel = p.context.renderModelLoadPanel(activity({
    variant: 'dflash2', stage: 'reading_weights', bytesDone: 9.3 * GB, bytesTotal: 16.1 * GB,
    completedStages: [
      { stage: 'verifying_files', elapsedMs: 11000 },
      { stage: 'stopping_backend', elapsedMs: 2000 },
      { stage: 'restarting_backend', elapsedMs: 1000 },
    ],
  }));
  const steps = [...panel.matchAll(/<li class="model-load-step[^"]*"[^>]*>[\s\S]*?<\/li>/g)].map(match => match[0]);
  assert.equal(steps.length, 6);
  const labels = ['校验目标模型与草稿模型', '重启后端', '读取目标模型权重', '构建模型', '读取草稿模型权重', '预热'];
  labels.forEach((label, index) => assert.match(steps[index], new RegExp(label)));
  assert.match(steps[0], /is-done[\s\S]*0:11/);
  assert.match(steps[1], /is-done[\s\S]*0:03/, 'stop and restart share one step');
  assert.match(steps[2], /is-current/);
  assert.match(panel, /模型\+DFlash2/);
  assert.match(panel, /9\.3 GB<\/b> \/ 16\.1 GB · 57%/);
  assert.match(panel, /width:57%/);
  assert.match(panel, /恢复之前的配置并重启后端/);

  const restore = p.context.renderModelLoadPanel(activity({ variant: 'dflash2', origin: 'restore',
    stage: 'reading_draft_weights', bytesDone: 15 * GB, bytesTotal: 16 * GB,
    completedStages: [{ stage: 'restarting_backend', elapsedMs: 1000 }, { stage: 'reading_weights', elapsedMs: 20000 },
      { stage: 'building_model', elapsedMs: 3000 }] }));
  assert.doesNotMatch(restore, /校验目标模型/, 'skipped steps are not shown');
  assert.match(restore, /以普通模式重启/);
});

test('mtp panel names the draft weights and plain loads hide transient waits', () => {
  const p = page('en');
  const panel = p.context.renderModelLoadPanel(activity({ variant: 'mtp', stage: 'building_model',
    bytesDone: 4 * GB, bytesTotal: 4 * GB,
    completedStages: [{ stage: 'waiting_for_backend', elapsedMs: 40 }, { stage: 'queued', elapsedMs: 3 },
      { stage: 'preparing_memory', elapsedMs: 900 }, { stage: 'reading_weights', elapsedMs: 20000 }] }));
  assert.match(panel, /Model \+ MTP/);
  assert.match(panel, /Read MTP draft weights/);
  assert.doesNotMatch(panel, /Wait for the backend|Wait in queue/);
  assert.match(panel, /Start inference/);
});

test('load callback leaves cancellations to the load-state toast', () => {
  const p = page();
  vm.runInContext(extract('onModelLoaded', 'localizeErrorResult'), p.context);
  p.context.onModelLoaded(JSON.stringify({ success: false, error: 'cancelled', code: 'load_cancelled' }));
  assert.equal(p.toasts.length, 0);
});

test('all_new_strings_exist_in_five_languages', () => {
  const p = page();
  const translations = vm.runInContext('I18N', p.context);
  const keys = Object.keys(translations.en).filter(key => key.startsWith('model_load_'));
  assert.ok(keys.length >= 30, `found ${keys.length} model_load_ keys`);
  const section = slice('  // ── Model load progress', "  window.__CURRENT_PARAM_MODEL__ = '';");
  for (const [, key] of section.matchAll(/'(model_load_[a-z0-9_]+)'/g)) {
    assert.ok(keys.includes(key), `${key} is referenced but missing in en`);
  }
  for (const language of ['zh-Hans', 'zh-Hant', 'ja', 'ko']) {
    for (const key of keys) {
      assert.ok(translations[language][key], `${language} ${key}`);
      if (!/^\{|^OK$/.test(translations.en[key])) {
        assert.notEqual(translations[language][key], translations.en[key], `${language} ${key} is translated`);
      }
    }
  }
});

test('panel_controls_survive_ticks_and_pushes', () => {
  const p = page();
  p.push(activity({ autoOpenPanel: true, bytesDone: GB, bytesTotal: 10 * GB }));
  const panel = p.element('model-load-panel');
  const cancel = panel.querySelector('[data-load-action="cancel"]');
  const background = panel.querySelector('[data-load-action="background"]');
  const close = panel.querySelector('[data-load-action="close"]');
  const spinner = panel.querySelector('.model-load-step.is-current');
  assert.ok(cancel && background && close && spinner);
  // The 1 s timer and Swift's 0.5 s pushes must not replace the controls
  // (a click whose mouseup lands on a replaced node is lost).
  for (let index = 2; index <= 4; index += 1) {
    p.tick();
    p.push(activity({ bytesDone: index * GB, bytesTotal: 10 * GB }));
    p.tick();
  }
  assert.equal(panel.querySelector('[data-load-action="cancel"]'), cancel);
  assert.equal(panel.querySelector('[data-load-action="background"]'), background);
  assert.equal(panel.querySelector('[data-load-action="close"]'), close);
  assert.equal(panel.querySelector('.model-load-step.is-current'), spinner, 'the current step keeps its spinner');
  // Values update in place.
  assert.equal(panel.querySelector('.model-load-fill').style.width, '40%');
  assert.match(panel.querySelector('.model-load-amount').innerHTML, /<b>4\.0 GB<\/b> \/ 10\.0 GB · 40%/);
  assert.match(panel.querySelector('.model-load-eta').textContent, /^已用 0:0\d/);
  cancel.click();
  assert.deepEqual(p.posted.filter(m => m.type === 'cancelModelLoad'), [{ type: 'cancelModelLoad', body: MODEL }]);
  assert.match(panel.innerHTML, /正在取消…/);
});

test('load_state_rows_wrap_instead_of_truncating', () => {
  const p = page();
  const cell = row => row.innerHTML.match(/<td class="model-status-cell[^"]*"[\s\S]*?<\/td>/)[0];
  p.push(activity({ kind: 'reload', bytesDone: 4 * GB, bytesTotal: 10 * GB }));
  const reload = cell(p.scan([scannedModel({ loaded: true })])[0]);
  assert.match(reload, /class="model-status-cell is-load-state"/, 'load-state cells may wrap');
  assert.match(reload, /model-status-text">重新加载中</);
  p.push(activity({ state: 'failed', stage: 'preparing_memory', cancellable: false, message: 'x' }));
  const failed = cell(p.scan([scannedModel()])[0]);
  assert.match(failed, /class="model-status-cell is-load-state"/);
  assert.match(failed, /model-status-text is-failed">加载失败</, 'a failure reads as a failure, not as loading');
  assert.doesNotMatch(failed, /status-dot/);
  p.push();
  assert.doesNotMatch(cell(p.scan([scannedModel()])[0]), /is-load-state/);
});

test('acknowledged_external_failure_stays_dismissed', () => {
  const p = page();
  const failure = activity({ origin: 'external', state: 'failed', stage: 'reading_weights',
    cancellable: false, message: 'broken weights' });
  p.push(failure);
  const indicator = p.element('model-load-indicator');
  assert.ok(indicator.classList.contains('is-failed'));
  p.context.openModelLoadPanel(MODEL);
  p.context.closeModelLoadPanel();
  // The backend keeps listing it for its 30 s retention.
  p.push(failure);
  assert.equal(indicator.hidden, true);
  assert.doesNotMatch(p.scan([scannedModel()])[0].innerHTML, /加载失败/);
  // A later failure of a new load shows again.
  p.push(activity({ ...failure, startedAt: STARTED + 10_000 }));
  assert.equal(indicator.hidden, false);
  assert.ok(indicator.classList.contains('is-failed'));
});

test('classified_async_failures_show_localized_guidance', () => {
  for (const lang of ['en', 'zh-Hans', 'zh-Hant', 'ja', 'ko']) {
    const p = page(lang);
    vm.runInContext(extract('onModelLoaded', 'onModelUnloaded'), p.context);
    const dict = vm.runInContext('I18N', p.context)[lang];
    for (const code of ['max_loaded_models_reached', 'model_memory_limit_exceeded', 'total_memory_limit_exceeded',
      'gpu_memory_insufficient', 'kv_memory_budget_exceeded', 'mtp_incompatible', 'load_result_unknown']) {
      assert.ok(dict[`err_${code}`], `${lang} err_${code}`);
      // The registry reports the backend's English message with the code.
      const failed = activity({ state: 'failed', stage: 'preparing_memory', cancellable: false,
        code, message: 'English backend message' });
      assert.equal(p.context.modelLoadFailureText(failed), dict[`err_${code}`], `${lang} panel ${code}`);
      p.toasts.length = 0;
      p.context.onModelLoaded(JSON.stringify({ success: false, status: 'error', code, error: 'English backend message' }));
      assert.equal(p.toasts.at(-1).message, dict[`err_${code}`], `${lang} toast ${code}`);
    }
  }
});

test('row_progress_shrinks_inside_a_narrow_status_cell', () => {
  const rule = selector => {
    const match = html.match(new RegExp(`\\n\\s*${selector.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\s*\\{([^}]*)\\}`));
    assert.ok(match, `missing ${selector}`);
    return match[1];
  };
  // Every part of the row progress may shrink, so "73%" never leaves the card.
  assert.match(rule('.model-load-rowprogress'), /min-width:\s*0/);
  assert.match(rule('.model-load-rowprogress'), /max-width:\s*min\(128px,\s*100%\)/);
  assert.match(rule('.model-load-rowprogress .model-load-bar'), /min-width:\s*0/);
  assert.match(rule('.model-load-rowprogress > span'), /flex:\s*none/);
});

test('row_leaves_the_reload_state_as_soon_as_the_reload_ends', () => {
  const p = page();
  const [row] = p.scan([scannedModel({ loaded: true })]);
  p.push(activity({ kind: 'reload', bytesDone: 4 * GB, bytesTotal: 10 * GB }));
  assert.match(row.innerHTML, /重新加载中/);
  const staleCancel = row.querySelector('.action-cancel-load');
  assert.ok(staleCancel);
  const syncs = p.syncs;

  p.push(activity({ kind: 'reload', state: 'loaded', cancellable: false, bytesDone: 10 * GB, bytesTotal: 10 * GB }));
  // The row shows the loaded model again without waiting for a rescan.
  assert.doesNotMatch(row.innerHTML, /重新加载中|cancelModelLoadFromUI/);
  assert.match(row.innerHTML, /action-unload/);
  assert.ok(p.syncs > syncs, 'a rescan still confirms the backend state');

  // The cancel button left from the reload never reaches the backend or unloads.
  staleCancel.click();
  assert.equal(p.posted.filter(m => m.type === 'cancelModelLoad').length, 0);
  assert.equal(p.unloads, 0);
  // The unload button that replaced "Cancel" shows as disabled while the row
  // settles, so a click aimed at "Cancel" does nothing.
  const unload = row.querySelector('.action-unload');
  assert.equal(unload.disabled, true);
  assert.match(row.innerHTML, /class="action-btn action-unload"[^>]* disabled>/);
  unload.click();
  assert.equal(p.unloads, 0);
  p.runTimers();
  assert.equal(unload.disabled, false, 'enabled after the settle period');
  assert.doesNotMatch(row.innerHTML, /action-unload"[^>]* disabled/);
  unload.click();
  assert.equal(p.unloads, 1, 'later clicks unload as usual');
});

test('settling_unload_button_survives_rescans_and_removed_rows', () => {
  const p = page();
  p.scan([scannedModel({ loaded: true })]);
  p.push(activity({ kind: 'reload', bytesDone: GB, bytesTotal: 10 * GB }));
  p.push(activity({ kind: 'reload', state: 'loaded', cancellable: false }));
  // A rescan during the settle period keeps the button disabled...
  const [rescanned] = p.scan([scannedModel({ loaded: true })]);
  assert.equal(rescanned.querySelector('.action-unload').disabled, true);
  // ...and the end of the period enables the row that shows the model now.
  p.runTimers();
  assert.equal(rescanned.querySelector('.action-unload').disabled, false);
  // Rescans after it render the button enabled.
  assert.equal(p.scan([scannedModel({ loaded: true })])[0].querySelector('.action-unload').disabled, false);

  // A row removed during the settle period: the timer finds nothing to do.
  p.push(activity({ kind: 'reload', startedAt: STARTED + 1, bytesDone: GB, bytesTotal: 10 * GB }));
  p.push(activity({ kind: 'reload', startedAt: STARTED + 1, state: 'loaded', cancellable: false }));
  assert.equal(p.pendingTimers, 1);
  p.scan([]);
  assert.doesNotThrow(() => p.runTimers());
  // A second restore of the same row replaces the first timer.
  const [again] = p.scan([scannedModel({ loaded: true })]);
  p.push(activity({ kind: 'reload', startedAt: STARTED + 2, bytesDone: GB, bytesTotal: 10 * GB }));
  p.push(activity({ kind: 'reload', startedAt: STARTED + 2, state: 'loaded', cancellable: false }));
  p.push(activity({ kind: 'reload', startedAt: STARTED + 3, bytesDone: GB, bytesTotal: 10 * GB }));
  p.push(activity({ kind: 'reload', startedAt: STARTED + 3, state: 'loaded', cancellable: false }));
  assert.equal(p.pendingTimers, 1);
  p.runTimers();
  assert.equal(again.querySelector('.action-unload').disabled, false);
});

test('rows_of_ended_loads_return_to_their_own_state', () => {
  const p = page();
  const [row] = p.scan([scannedModel()]);
  // A cancelled new load leaves the model unloaded.
  p.push(activity({ bytesDone: GB, bytesTotal: 10 * GB }));
  assert.match(row.innerHTML, /cancelModelLoadFromUI/);
  p.push(activity({ state: 'cancelled', code: 'load_cancelled', cancellable: false }));
  assert.doesNotMatch(row.innerHTML, /cancelModelLoadFromUI/);
  assert.match(row.innerHTML, /action-load/);
  assert.doesNotMatch(row.innerHTML, /action-unload/);

  // A cancelled reload keeps the previous version loaded.
  const [loaded] = p.scan([scannedModel({ loaded: true })]);
  p.push(activity({ kind: 'reload', startedAt: STARTED + 1, bytesDone: GB, bytesTotal: 10 * GB }));
  assert.match(loaded.innerHTML, /重新加载中/);
  p.push(activity({ kind: 'reload', startedAt: STARTED + 1, state: 'cancelled', code: 'reload_cancelled', cancellable: false }));
  assert.match(loaded.innerHTML, /action-unload/);

  // A load that is no longer listed falls back to the last scan.
  const [idle] = p.scan([scannedModel()]);
  p.push(activity({ startedAt: STARTED + 2, bytesDone: GB, bytesTotal: 10 * GB }));
  const staleCancel = idle.querySelector('.action-cancel-load');
  p.push();
  assert.doesNotMatch(idle.innerHTML, /cancelModelLoadFromUI/);
  assert.match(idle.innerHTML, /action-load/);
  staleCancel.click();
  assert.equal(p.posted.filter(m => m.type === 'cancelModelLoad').length, 0);
  assert.equal(p.unloads, 0);
});

test('acknowledging_a_failure_clears_indicator_and_row_at_once', () => {
  const p = page();
  const [row] = p.scan([scannedModel()]);
  p.push(activity({ bytesDone: GB, bytesTotal: 10 * GB }));
  p.push(activity({ state: 'failed', stage: 'preparing_memory', cancellable: false,
    message: 'engine pool capacity reached' }));
  const indicator = p.element('model-load-indicator');
  assert.ok(indicator.classList.contains('is-failed'));
  assert.match(row.innerHTML, /加载失败/);
  const syncs = p.syncs;

  p.context.openModelLoadPanel(MODEL);
  p.element('model-load-panel').querySelector('[data-load-action="ok"]').click();
  // Both clear before any push from the App or rescan arrives.
  assert.equal(indicator.hidden, true);
  assert.doesNotMatch(row.innerHTML, /加载失败/);
  assert.match(row.innerHTML, /action-load/);
  assert.ok(p.syncs > syncs, 'a rescan still confirms the backend state');
});

test('model_sync_is_its_own_step', () => {
  for (const lang of ['en', 'zh-Hans', 'zh-Hant', 'ja', 'ko']) {
    const p = page(lang);
    const dict = vm.runInContext('I18N', p.context)[lang];
    assert.ok(dict.model_load_stage_syncing_models, `${lang} model_load_stage_syncing_models`);
    const panel = p.context.renderModelLoadPanel(activity({ stage: 'syncing_models',
      completedStages: [{ stage: 'waiting_for_backend', elapsedMs: 2000 }] }));
    const labels = [...panel.matchAll(/<li class="model-load-step is-([a-z]+)"[^>]*>[\s\S]*?step-label">([^<]*)</g)]
      .map(match => `${match[1]}:${match[2]}`);
    assert.deepEqual(labels.slice(0, 3), [
      `done:${dict.model_load_stage_waiting_for_backend}`,
      `current:${dict.model_load_stage_syncing_models}`,
      `pending:${dict.model_load_stage_preparing_memory}`,
    ], lang);
  }
  const p = page();
  assert.match(p.context.renderModelLoadPanel(activity({ stage: 'syncing_models' })), /同步本地模型/);
});

test('failure_marks_the_stage_it_stopped_in', () => {
  const p = page();
  const steps = markup => [...markup.matchAll(/<li class="model-load-step is-([a-z]+)"[^>]*>[\s\S]*?<\/li>/g)]
    .map(match => ({ status: match[1], html: match[0] }));
  // A backend stage: the registry also lists it as completed when the load ends.
  const backend = steps(p.context.renderModelLoadPanel(activity({ state: 'failed', stage: 'preparing_memory',
    cancellable: false, message: 'engine pool capacity reached',
    completedStages: [{ stage: 'verifying_files', elapsedMs: 3000 }, { stage: 'queued', elapsedMs: 2 },
      { stage: 'preparing_memory', elapsedMs: 40 }] })));
  assert.deepEqual(backend.map(step => step.status), ['done', 'failed']);
  assert.match(backend[1].html, /准备内存/);
  assert.match(backend[1].html, /model-load-step-mark">!</);
  // An App stage.
  const app = steps(p.context.renderModelLoadPanel(activity({ state: 'failed', stage: 'verifying_files',
    cancellable: false, message: 'checksum mismatch' })));
  assert.deepEqual(app.map(step => step.status), ['failed']);
  assert.match(app[0].html, /校验模型文件/);
  assert.match(app[0].html, /model-load-step-mark">!</);
});

test('unrecognized_failures_show_a_localized_message_with_the_raw_detail', () => {
  const raw = 'loading engine model `x`: <b>disk</b> & "fire"';
  const escaped = 'loading engine model `x`: &lt;b&gt;disk&lt;/b&gt; &amp; &quot;fire&quot;';
  for (const lang of ['en', 'zh-Hans', 'zh-Hant', 'ja', 'ko']) {
    const p = page(lang);
    vm.runInContext(extract('onModelLoaded', 'onModelUnloaded'), p.context);
    const dict = vm.runInContext('I18N', p.context)[lang];
    assert.ok(dict.err_load_failed, `${lang} err_load_failed`);
    if (lang !== 'en') assert.notEqual(dict.err_load_failed, vm.runInContext('I18N', p.context).en.err_load_failed);
    for (const code of ['load_failed', 'some_future_code', undefined]) {
      const failed = activity({ state: 'failed', stage: 'reading_weights', cancellable: false, code, message: raw });
      const panel = p.context.renderModelLoadPanel(failed);
      const primary = panel.match(/<div class="model-load-error">([\s\S]*?)<\/div>/);
      assert.ok(primary, `${lang} ${code} panel error`);
      assert.equal(primary[1], p.context.escapeAttr(dict.err_load_failed), `${lang} ${code} primary`);
      const detail = panel.match(/<div class="model-load-error-detail">([\s\S]*?)<\/div>/);
      assert.ok(detail, `${lang} ${code} panel detail`);
      assert.equal(detail[1], escaped, `${lang} ${code} detail is escaped`);
      assert.doesNotMatch(panel, /<b>disk/);

      p.push(failed);
      const row = p.context.modelLoadRowState(MODEL);
      assert.ok(row.title.startsWith(dict.err_load_failed), `${lang} ${code} row tooltip`);
      assert.ok(row.title.includes(raw), `${lang} ${code} row tooltip keeps the detail`);

      p.toasts.length = 0;
      p.context.onModelLoaded(JSON.stringify({ success: false, status: 'error', code, error: raw }));
      assert.equal(p.toasts.at(-1).message, dict.err_load_failed, `${lang} ${code} toast`);
    }
  }
  assert.equal(vm.runInContext('I18N', page().context)['zh-Hans'].err_load_failed,
    '模型加载失败，原因未能识别。请查看日志了解详情。');
});

test('classified_failures_have_no_raw_detail', () => {
  const p = page();
  vm.runInContext(extract('onModelLoaded', 'onModelUnloaded'), p.context);
  const dict = vm.runInContext('I18N', p.context)['zh-Hans'];
  const panel = p.context.renderModelLoadPanel(activity({ state: 'failed', stage: 'preparing_memory',
    cancellable: false, code: 'max_loaded_models_reached', message: 'English backend message' }));
  assert.match(panel, new RegExp(`<div class="model-load-error">${dict.err_max_loaded_models_reached}</div>`));
  assert.doesNotMatch(panel, /model-load-error-detail|English backend message/);
});

test('recognized_failures_without_their_own_string_keep_the_backend_reason', () => {
  const raw = 'DiffusionGemma supports max_tokens, temperature, and seed. Remove top_p, top_k, and repetition_penalty overrides and try again.';
  for (const lang of ['en', 'zh-Hans', 'zh-Hant', 'ja', 'ko']) {
    const p = page(lang);
    vm.runInContext(extract('onModelLoaded', 'onModelUnloaded'), p.context);
    const dict = vm.runInContext('I18N', p.context)[lang];
    const failed = activity({ state: 'failed', stage: 'preparing_memory', cancellable: false,
      code: 'diffusion_gemma_sampling_parameter_unsupported', message: raw });
    const panel = p.context.renderModelLoadPanel(failed);
    const primary = panel.match(/<div class="model-load-error">([\s\S]*?)<\/div>/);
    assert.equal(primary && primary[1], p.context.escapeAttr(dict.recovery_reason_config), `${lang} primary`);
    const detail = panel.match(/<div class="model-load-error-detail">([\s\S]*?)<\/div>/);
    assert.equal(detail && detail[1], raw, `${lang} detail names the setting to remove`);
    p.push(failed);
    const row = p.context.modelLoadRowState(MODEL);
    assert.equal(row.title, `${dict.recovery_reason_config}\n${raw}`, `${lang} row tooltip`);
  }
});

test('model_task_mismatch_has_its_own_message', () => {
  const en = vm.runInContext('I18N', page().context).en.err_model_task_mismatch;
  assert.ok(en, 'en err_model_task_mismatch');
  for (const lang of ['en', 'zh-Hans', 'zh-Hant', 'ja', 'ko']) {
    const p = page(lang);
    vm.runInContext(extract('onModelLoaded', 'onModelUnloaded'), p.context);
    const dict = vm.runInContext('I18N', p.context)[lang];
    assert.ok(dict.err_model_task_mismatch, `${lang} err_model_task_mismatch`);
    if (lang !== 'en') assert.notEqual(dict.err_model_task_mismatch, en, `${lang} is translated`);
    const panel = p.context.renderModelLoadPanel(activity({ state: 'failed', stage: 'preparing_memory',
      cancellable: false, code: 'model_task_mismatch',
      message: 'Embedding models do not accept audio, decision or generation settings.' }));
    assert.match(panel, new RegExp(`<div class="model-load-error">${p.context.escapeAttr(dict.err_model_task_mismatch)}</div>`), lang);
    assert.doesNotMatch(panel, /model-load-error-detail/, lang);
  }
});
