const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const html = fs.readFileSync(path.join(__dirname,
  '../../ironmlx-app/Sources/IronMLXAppCore/Resources/dashboard2.html'), 'utf8');
const blockStart = html.indexOf('  // ── Models page ──');
const blockEnd = html.indexOf('  // ── End models page ──');
const block = blockStart >= 0 && blockEnd > blockStart ? html.slice(blockStart, blockEnd) : '';
const i18nStart = html.indexOf('  const I18N =');
const translations = html.slice(i18nStart, html.indexOf('\n  };', i18nStart) + 5);

function slice(start, end) {
  const from = html.indexOf(start);
  assert.ok(from >= 0, `missing ${start}`);
  return html.slice(from, html.indexOf(end, from));
}

function page(extra = {}) {
  const context = Object.assign({
    currentLang: 'en',
    console,
    window: { __MODEL_INTEGRITY_STATUS__: {}, __LOCAL_MODELS__: {}, addEventListener() {} },
    document: { getElementById: () => null, querySelectorAll: () => [], querySelector: () => null, addEventListener() {} },
    escapeAttr: value => String(value || ''),
    setInterval() { return 1; }, clearInterval() {}, setTimeout() { return 1; }, clearTimeout() {},
  }, extra);
  vm.createContext(context);
  vm.runInContext(translations
    + slice('  function t(key, fallback)', '  function setLanguage(')
    + slice('  // ── Model load progress', "  window.__CURRENT_PARAM_MODEL__ = '';")
    + slice('  function modelReadiness(', '  function updateModelIntegrityRow(')
    + slice('  const STATUS_DASH', '  function formatStatusClock(')
    + slice('  function idleUnloadText(', '  // Shares of the limit')
    + slice('  function localizeErrorResult(', '  // ── Model load progress')
    + block, context);
  return context;
}

function fakeButton(value, left, width, role = 'tab') {
  const attrs = { role };
  return {
    dataset: { value },
    offsetLeft: left,
    offsetWidth: width,
    focused: false,
    clicked: 0,
    getAttribute: name => (name in attrs ? attrs[name] : null),
    setAttribute: (name, v) => { attrs[name] = String(v); },
    focus() { this.focused = true; },
    click() { this.clicked += 1; },
    attrs,
  };
}

function fakeGroup(buttons) {
  const thumb = { style: {}, classList: { add() {}, remove() {}, contains: () => false } };
  return {
    thumb,
    querySelector: sel => (sel === '.ds-seg-thumb' ? thumb : null),
    querySelectorAll: sel => (sel === '[data-value]' ? buttons : []),
  };
}

test('download_badge_counts_active_and_queued', () => {
  const p = page();
  assert.equal(p.modelsDownloadBadge({ active_count: 1, queued_count: 2 }), 3);
  assert.equal(p.modelsDownloadBadge({}), 0);
  assert.equal(p.modelsDownloadBadge(null), 0);
});

test('runtime_model_view_reads_shared_snapshot', () => {
  const p = page();
  const state = vm.runInContext('runtimeState', p);
  assert.equal(p.runtimeModelView('a'), null);
  state.snapshot = { models: [{ id: 'a', state: 'loaded' }] };
  assert.equal(p.runtimeModelView('a').state, 'loaded');
  assert.equal(p.runtimeModelView('b'), null);
});

test('runtime_listeners_each_isolated_from_failures', () => {
  const p = page();
  const seen = [];
  p.onRuntimeSnapshot(() => { throw new Error('boom'); });
  p.onRuntimeSnapshot(data => seen.push(data));
  p.runtimeSnapshotPublish({ models: [] });
  assert.equal(seen.length, 1);
  assert.deepEqual(vm.runInContext('runtimeState.snapshot', p), { models: [] });
});

test('seg_select_marks_one_tab', () => {
  const p = page();
  const buttons = [fakeButton('my', 2, 80), fakeButton('download', 84, 90), fakeButton('tuning', 176, 100)];
  const group = fakeGroup(buttons);
  p.dsSegSelect(group, 'download');
  assert.deepEqual(buttons.map(b => b.attrs['aria-selected']), ['false', 'true', 'false']);
  assert.match(group.thumb.style.transform, /84/);
  assert.equal(group.thumb.style.width, '90px');
});

test('seg_select_uses_aria_pressed_for_plain_toggles', () => {
  const p = page();
  const buttons = [fakeButton('a', 0, 40, null), fakeButton('b', 42, 40, null)];
  const group = fakeGroup(buttons);
  p.dsSegSelect(group, 'b');
  assert.deepEqual(buttons.map(b => b.attrs['aria-pressed']), ['false', 'true']);
});

test('seg_keydown_moves_between_tabs', () => {
  const p = page();
  const buttons = [fakeButton('my', 0, 80), fakeButton('download', 80, 80), fakeButton('tuning', 160, 80)];
  const group = fakeGroup(buttons);
  group.contains = () => true;
  let prevented = 0;
  // The target must be one of the group's buttons.
  p.dsSegKeydown({ key: 'ArrowRight', target: Object.assign(buttons[0], { closest: () => group }), preventDefault: () => { prevented += 1; } });
  assert.equal(buttons[1].focused, true);
  assert.equal(buttons[1].clicked, 1);
  p.dsSegKeydown({ key: 'ArrowLeft', target: Object.assign(buttons[0], { closest: () => group }), preventDefault: () => { prevented += 1; } });
  assert.equal(buttons[2].clicked, 1, 'left from the first wraps to the last');
  assert.equal(prevented, 2);
  p.dsSegKeydown({ key: 'a', target: Object.assign(buttons[0], { closest: () => group }), preventDefault: () => { prevented += 1; } });
  assert.equal(prevented, 2);
});

test('tab_labels_exist_in_five_languages', () => {
  const p = page();
  const I18N = vm.runInContext('I18N', p);
  const english = { models_tab_my: 'My models', models_tab_download: 'Download', models_tab_tuning: 'Tuning' };
  for (const [key, value] of Object.entries(english)) {
    assert.equal(I18N.en[key], value);
    for (const lang of ['zh-Hans', 'zh-Hant', 'ja', 'ko']) {
      assert.ok(I18N[lang][key], `${lang}.${key}`);
      assert.notEqual(I18N[lang][key], I18N.en[key], `${lang}.${key} must be translated`);
    }
  }
  for (const key of ['models_summary', 'model_download_primary', 'models_search_placeholder']) {
    for (const lang of ['en', 'zh-Hans', 'zh-Hant', 'ja', 'ko']) assert.ok(I18N[lang][key], `${lang}.${key}`);
  }
});

test('page_markup_has_tablist_in_order_with_download_badge', () => {
  const tablist = html.indexOf('class="ds-seg ds-seg-lg" role="tablist"');
  assert.ok(tablist > 0);
  const manage = html.indexOf('data-tab="models-manage"', tablist);
  const download = html.indexOf('data-tab="models-download"', tablist);
  const tuning = html.indexOf('data-tab="models-scheduler"', tablist);
  assert.ok(manage > tablist && download > manage && tuning > download);
  for (const id of ['models-page-summary', 'models-search', 'models-download-badge', 'model-import-button', 'model-import-status']) {
    assert.ok(html.includes(`id="${id}"`), id);
  }
});

test('search_field_only_visible_on_my_models', () => {
  const field = { hidden: false };
  const p = page({ document: { getElementById: id => (id === 'models-search-field' ? field : null), addEventListener() {}, querySelectorAll: () => [] } });
  p.modelsSearchVisibility('models-download');
  assert.equal(field.hidden, true);
  p.modelsSearchVisibility('models-scheduler');
  assert.equal(field.hidden, true);
  p.modelsSearchVisibility('models-manage');
  assert.equal(field.hidden, false);
  // display:flex would override the UA [hidden] rule, so the CSS must restore it.
  assert.match(html, /\.models-search-field\[hidden\]\s*\{\s*display:\s*none;/);
  assert.match(html, /switchToTab[\s\S]{0,900}modelsSearchVisibility\(tabName\)/);
});

test('download_sources_sit_in_the_tab_row_on_the_download_tab_only', () => {
  const field = { hidden: false }, sources = { hidden: true };
  const byId = { 'models-search-field': field, 'dl-source-row': sources };
  const p = page({ document: { getElementById: id => byId[id] || null, addEventListener() {}, querySelectorAll: () => [] } });
  p.modelsSearchVisibility('models-download');
  assert.deepEqual([field.hidden, sources.hidden], [true, false]);
  p.modelsSearchVisibility('models-scheduler');
  assert.deepEqual([field.hidden, sources.hidden], [true, true]);
  p.modelsSearchVisibility('models-manage');
  assert.deepEqual([field.hidden, sources.hidden], [false, true]);
  // Same row as the page tabs, right after the search box; hidden until the Download tab opens.
  const row = html.slice(html.indexOf('<div class="models-page-tabs">'), html.indexOf('<div class="models-page-scroll'));
  assert.match(row, /id="models-search-field"[\s\S]*<div class="dl-source-row" id="dl-source-row" hidden>[\s\S]*id="dl-tab-catalog"/);
  assert.match(html, /\.dl-source-row\[hidden\]\s*\{\s*display:none;/);
  assert.match(html, /\.catalog-filters \{[^}]*margin-bottom:24px/);
});

test('filter_chip_component_leaves_status_pill_alone', () => {
  assert.match(html, /\.ds-chip\.is-on\s*\{/);
  assert.doesNotMatch(html, /\.ds-pill\.is-on/);
});

// ---- presentation rules ----
const GB = 1024 ** 3;
const ctxOf = (activity = null, view = null) => ({ activity, view });
const act = (o = {}) => Object.assign({ model: 'm', variant: 'plain', kind: 'load', state: 'loading',
  stage: 'reading_weights', completedStages: [], bytesDone: GB, bytesTotal: 4 * GB,
  startedAt: Date.now(), cancellable: true }, o);
const mdl = (o = {}) => Object.assign({ id: 'm', display_name: 'M', type: 'llm', loaded: false,
  readiness: { status: 'ready' } }, o);

test('category_mapping', () => {
  const p = page();
  const want = { llm: 'text', vlm: 'vision', block_diffusion_vlm: 'vision', image_generation: 'image',
    asr: 'audio', tts: 'audio', alignment: 'audio', diarization: 'audio', embedding: 'embedding',
    reranker: 'embedding', decision: 'decision', mtp: 'other', weird: 'other' };
  for (const [type, cat] of Object.entries(want)) assert.equal(p.modelCategory(type), cat, type);
});

test('filter_and_search_combine', () => {
  const p = page();
  const m = { id: 'org/Qwen3-4B', display_name: 'Qwen3-4B', type: 'llm' };
  assert.equal(p.modelsFilterMatch(m, 'text', 'qwen'), true);
  assert.equal(p.modelsFilterMatch(m, 'vision', 'qwen'), false);
  assert.equal(p.modelsFilterMatch(m, 'all', '  QWEN3 '), true);
  assert.equal(p.modelsFilterMatch(m, 'all', 'gemma'), false);
  assert.equal(p.modelsFilterMatch(m, 'all', 'org/'), true);
});

test('groups_place_loading_models_in_loaded', () => {
  const p = page();
  const acts = {
    a: act({ model: 'a', state: 'loading' }),
    f: act({ model: 'f', state: 'failed' }),
    r: act({ model: 'r', kind: 'reload', state: 'failed' }),
  };
  const models = [mdl({ id: 'Beta', display_name: 'Beta', loaded: true }), mdl({ id: 'alpha', display_name: 'alpha', loaded: true }),
    mdl({ id: 'a', display_name: 'a-loading' }), mdl({ id: 'f', display_name: 'f-failed' }),
    mdl({ id: 'r', display_name: 'r-reload', loaded: true }), mdl({ id: 'x', type: 'mtp', loaded: true })];
  const g = p.modelsPageGroups(models, m => ctxOf(acts[m.id] || null));
  assert.deepEqual(Array.from(g.loaded, m => m.id), ['a', 'alpha', 'Beta', 'r']);
  assert.deepEqual(Array.from(g.unloaded, m => m.id), ['f']);
  assert.equal(p.modelsPageGroup(mdl(), ctxOf(act({ state: 'cancelling' }))), 'loaded');
  assert.equal(p.modelsPageGroup(mdl(), ctxOf(act({ state: 'queued' }))), 'loaded');
});

test('groups_sort_numbers_naturally', () => {
  const p = page();
  const models = ['m-10', 'm-2', 'Qwen3.5-27B', 'Qwen3.5-2B'].map(id => mdl({ id, display_name: id }));
  const g = p.modelsPageGroups(models, () => ctxOf(null));
  assert.deepEqual(Array.from(g.unloaded, m => m.id), ['m-2', 'm-10', 'Qwen3.5-2B', 'Qwen3.5-27B']);
});

test('failure_union', () => {
  const p = page();
  const snap = { state: 'failed', mode: 'dflash2', last_error: 'x' };
  const f = p.modelFailure(mdl(), ctxOf(null, snap));
  assert.deepEqual({ ...f }, { kind: 'load', source: 'snapshot', variant: 'dflash2', text: f.text, rawDetail: 'x' });
  assert.equal(p.modelFailure(mdl(), ctxOf(null, { state: 'failed', mode: 'normal' })).variant, 'plain');
  assert.equal(p.modelFailure(mdl(), ctxOf(null, { state: 'failed', mode: 'mtp' })).variant, 'mtp');
  const both = p.modelFailure(mdl(), ctxOf(act({ state: 'failed', variant: 'mtp', message: 'boom' }), snap));
  assert.equal(both.source, 'app');
  assert.equal(both.variant, 'mtp');
  assert.equal(both.kind, 'load');
  assert.equal(p.modelFailure(mdl(), ctxOf(null, { state: 'loaded' })), null);
  assert.equal(p.modelFailure(mdl(), ctxOf()), null);
});

test('row_verifying', () => {
  const p = page();
  p.window.__MODEL_INTEGRITY_STATUS__.m = { state: 'verifying', progress_pct: 50 };
  const r = p.modelRowPresentation(mdl(), ctxOf(), 'en', 0);
  assert.equal(r.dot, 'blue');
  assert.match(r.detail, /50%/);
});

test('row_loading_activity', () => {
  const p = page();
  const r = p.modelRowPresentation(mdl(), ctxOf(act()), 'en', 0);
  assert.equal(r.dot, 'blue');
  assert.equal(r.progress.pct, 25);
  assert.equal(r.primary.kind, 'cancel');
});

test('row_cancelling', () => {
  const p = page();
  const r = p.modelRowPresentation(mdl(), ctxOf(act({ state: 'cancelling' })), 'en', 0);
  assert.equal(r.dot, 'blue');
  assert.equal(r.primary.kind, 'cancelling');
});

test('row_load_failed_app', () => {
  const p = page();
  const r = p.modelRowPresentation(mdl(), ctxOf(act({ state: 'failed', message: 'boom' })), 'en', 0);
  assert.equal(r.dot, 'red');
  assert.equal(r.detailTone, 'danger');
  assert.equal(r.primary.kind, 'retry');
  assert.match(r.detail, /^Load failed:/);
  assert.equal(r.failure.source, 'app');
});

test('row_load_failed_snapshot_only', () => {
  const p = page();
  const r = p.modelRowPresentation(mdl(), ctxOf(null, { state: 'failed', mode: 'normal', last_error: 'oom' }), 'en', 0);
  assert.equal(r.dot, 'red');
  assert.equal(r.primary.kind, 'retry');
  assert.doesNotMatch(r.detail, /oom/);
  assert.match(r.detail, /^Load failed:/);
  assert.equal(r.failure.rawDetail, 'oom');
  assert.equal(r.failure.source, 'snapshot');
});

test('row_reload_failed', () => {
  const p = page();
  const r = p.modelRowPresentation(mdl({ loaded: true }), ctxOf(act({ kind: 'reload', state: 'failed', message: 'bad' })), 'en', 0);
  assert.equal(r.dot, 'amber');
  assert.equal(r.detailTone, 'warning');
  assert.equal(r.primary.kind, 'unload');
  assert.match(r.detail, /still using|previous|original/i);
});

test('row_reload_failure_on_unloaded_model_is_a_load_failure', () => {
  const p = page();
  const r = p.modelRowPresentation(mdl({ loaded: false }), ctxOf(act({ kind: 'reload', state: 'failed', message: 'bad' })), 'en', 0);
  assert.equal(r.dot, 'red');
  assert.equal(r.detailTone, 'danger');
  assert.equal(r.primary.kind, 'retry');
  assert.equal(r.failure.kind, 'load');
  assert.doesNotMatch(r.detail, /still using|previous|original/i);
});

test('row_corrupt_incomplete_download_only', () => {
  const p = page();
  for (const [status, extra, dot] of [['corrupt', {}, 'red'], ['incomplete', {}, 'amber'],
    ['unsupported', { reason_code: 'download_only_model' }, 'amber']]) {
    const m = mdl({ readiness: Object.assign({ status, message: 'MSG-' + status }, extra) });
    const r = p.modelRowPresentation(m, ctxOf(), 'en', 0);
    assert.equal(r.dot, dot, status);
    assert.ok(r.detail.length > 0, status);
    assert.ok(!r.detail.includes('MSG-' + status), status + ': raw reason stays out of the visible line');
    if (extra.reason_code !== 'download_only_model') assert.equal(r.detailTitle, 'MSG-' + status, status);
    assert.equal(r.primary.kind, 'unavailable', status);
  }
});

test('row_tts_missing_resources', () => {
  const p = page();
  const m = mdl({ type: 'tts', readiness: { status: 'incomplete', reason_code: 'audio_resources_missing', message: 'need audio' } });
  const r = p.modelRowPresentation(m, ctxOf(), 'en', 0);
  assert.equal(r.dot, 'amber');
  assert.equal(r.primary.kind, 'prepare');
});

test('row_generating', () => {
  const p = page();
  const view = { state: 'loaded', active_requests: 1, queued_requests: 2, live_decode_tokens_per_second: 142.4,
    weight_bytes: 19.8 * GB, kv_cache_bytes: 2.1 * GB };
  const r = p.modelRowPresentation(mdl({ loaded: true }), ctxOf(null, view), 'en', 0);
  assert.equal(r.dot, 'green');
  assert.match(r.detail, /\b142 tok\/s\b/);
  assert.match(r.detail, /\b2 queued\b/);
  assert.match(r.detail, /Weights 19\.8 GB/);
  assert.match(r.detail, /KV 2\.1 GB/);
  assert.equal(r.primary.kind, 'unload');
  assert.equal(r.pinned, false);
});

test('row_idle_with_unload', () => {
  const p = page();
  const view = { state: 'loaded', active_requests: 0, idle_unload_unix_ms: 12 * 60000, weight_bytes: 7 * GB, kv_cache_bytes: GB };
  const r = p.modelRowPresentation(mdl({ loaded: true }), ctxOf(null, view), 'en', 0);
  assert.equal(r.dot, 'gray');
  assert.ok(r.detail.startsWith(p.idleUnloadText(view, 0, 'en')));
});

test('row_idle_pinned', () => {
  const p = page();
  const view = { state: 'loaded', active_requests: 0, weight_bytes: 7 * GB, kv_cache_bytes: GB };
  const r = p.modelRowPresentation(mdl({ loaded: true, pinned: true }), ctxOf(null, view), 'en', 0);
  assert.equal(r.dot, 'gray');
  assert.equal(r.pinned, true);
  assert.ok(r.detail.includes(p.t('model_row_pinned_no_unload')));
  const plain = p.modelRowPresentation(mdl({ loaded: true }), ctxOf(null, view), 'en', 0);
  assert.ok(plain.detail.startsWith(p.t('model_row_idle')));
});

test('row_loaded_without_snapshot', () => {
  const p = page();
  const r = p.modelRowPresentation(mdl({ loaded: true }), ctxOf(), 'en', 0);
  assert.equal(r.dot, 'green');
  assert.equal(r.detail, p.t('model_row_loaded'));
});

test('row_not_loaded', () => {
  const p = page();
  const r = p.modelRowPresentation(mdl(), ctxOf(), 'en', 0);
  assert.equal(r.dot, 'hollow');
  assert.equal(r.detail, '');
  assert.equal(r.primary.kind, 'load');
  const split = mdl({ dflash2: { status: 'available', candidates: [{}] } });
  assert.equal(p.modelRowPresentation(split, ctxOf(), 'en', 0).primary.kind, 'load-split');
  const mtp = mdl({ mtp: { status: 'available', candidates: [{}] } });
  assert.equal(p.modelRowPresentation(mtp, ctxOf(), 'en', 0).primary.kind, 'load-split');
});

test('row_dflash2_target_offers_unload', () => {
  const p = page();
  const m = mdl({ loaded: true, dflash2: { enabled: true, status: 'available', candidates: [{}] } });
  const r = p.modelRowPresentation(m, ctxOf(), 'en', 0);
  assert.equal(r.primary.kind, 'unload');
  const failing = p.modelRowPresentation(m, ctxOf(null, { state: 'failed', mode: 'dflash2', last_error: 'x' }), 'en', 0);
  assert.equal(failing.primary.kind, 'unload');
});

test('default_menu_state', () => {
  const p = page();
  p.window.__DEFAULT_MODEL__ = 'm';
  assert.equal(p.modelsDefaultMenuState(mdl()), 'current');
  assert.equal(p.modelsDefaultMenuState(mdl({ id: 'e', type: 'embedding' })), 'hidden');
  assert.equal(p.modelsDefaultMenuState(mdl({ id: 'c', readiness: { status: 'corrupt' } })), 'hidden');
  assert.equal(p.modelsDefaultMenuState(mdl({ id: 'l' })), 'available');
});

test('row_unverified', () => {
  const p = page();
  const m = mdl({ readiness: { status: 'unverified', message: 'Not verified yet' } });
  const r = p.modelRowPresentation(m, ctxOf(), 'en', 0);
  assert.equal(r.dot, 'amber');
  assert.ok(r.detail.length > 0);
  assert.ok(!r.detail.includes('Not verified yet'));
  assert.equal(r.detailTitle, 'Not verified yet');
  assert.equal(r.primary.kind, 'unavailable');
});

test('row_loaded_with_failed_snapshot_is_reload_failure', () => {
  const p = page();
  const r = p.modelRowPresentation(mdl({ loaded: true }), ctxOf(null, { state: 'failed', mode: 'normal', last_error: 'x' }), 'en', 0);
  assert.equal(r.dot, 'amber');
  assert.equal(r.primary.kind, 'unload');
  assert.match(r.detail, /^Reload failed/);
});

test('row_queued_has_no_progress', () => {
  const p = page();
  const r = p.modelRowPresentation(mdl(), ctxOf(act({ state: 'queued' })), 'en', 0);
  assert.equal(r.progress, null);
});

test('row_queued_requests_only_is_busy', () => {
  const p = page();
  const view = { state: 'loaded', active_requests: 0, queued_requests: 2, weight_bytes: GB, kv_cache_bytes: 0 };
  const r = p.modelRowPresentation(mdl({ loaded: true }), ctxOf(null, view), 'en', 0);
  assert.equal(r.dot, 'green');
  assert.match(r.detail, /2 queued/);
});

test('snapshot_failure_does_not_log_per_call', () => {
  const logged = [];
  const p = page({ console: { error: m => logged.push(m), warn() {}, log() {} } });
  for (let i = 0; i < 3; i++) p.modelRowPresentation(mdl(), ctxOf(null, { state: 'failed', mode: 'normal', last_error: 'weird' }), 'en', 0);
  assert.equal(logged.length, 0);
});

// ---- rendering "My models" (Task 4) ----
// A small DOM: parses the markup the page writes, so tests hold element
// nodes, click real buttons and see which nodes a patch replaced.
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
  set className(value) { this.setAttribute('class', value); }
  get className() { return this.getAttribute('class'); }
  set type(value) { this.attributes.set('type', String(value)); }
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
  hasAttribute(name) { return this.getAttribute(name) !== null; }
  removeAttribute(name) { this.attributes.delete(name); if (name === 'disabled') this.disabled = false; }
  addEventListener(type, fn) { (this.listeners ||= {})[type] = fn; }
  focus() { this.doc.activeElement = this; }
  scrollIntoView() { this.scrolled = (this.scrolled || 0) + 1; }
  getBoundingClientRect() { return { left: 0, right: 10, top: 0, bottom: 10, width: 10, height: 10 }; }
  get offsetWidth() { return 1; }
  get isConnected() { return !!this.parentNode; }
  appendChild(node) { node.parentNode = this; this.childNodes.push(node); return node; }
  remove() {
    if (!this.parentNode) return;
    this.parentNode.childNodes = this.parentNode.childNodes.filter(node => node !== this);
    this.parentNode = null;
  }
  get outerHTML() {
    const tag = this.tagName.toLowerCase();
    const attrs = [...this.attributes].map(([name, value]) => name === 'class'
      ? ` class="${this.getAttribute('class')}"` : (value === '' ? ` ${name}` : ` ${name}="${value}"`)).join('');
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
  closest(selector) {
    for (let at = this; at && at.nodeType === 1; at = at.parentNode) if (at.matches(selector)) return at;
    return null;
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
    if (this.disabled) return;
    if (this.listeners && this.listeners.click) { this.listeners.click({ stopPropagation() {}, target: this }); return; }
    const code = this.getAttribute('onclick');
    if (code) vm.runInContext('Function', this.doc.context)('event', code).call(this, { target: this, stopPropagation() {} });
  }
}

function domPage(options = {}) {
  const doc = { activeElement: null };
  const elements = new Map();
  const posted = [];
  const timers = [];
  function element(id) {
    if (!elements.has(id)) {
      const el = new FakeElement(doc, 'div');
      el.setAttribute('id', id);
      elements.set(id, el);
    }
    return elements.get(id);
  }
  const body = new FakeElement(doc, 'body');
  const handler = type => ({ postMessage(body) { posted.push({ type, body: typeof body === 'string' && body.startsWith('{') ? JSON.parse(body) : body }); } });
  const handlers = {};
  for (const type of ['loadModel', 'acknowledgeModelLoadFailure', 'openModelLoadPanel', 'cancelModelLoad',
    'setDefaultModel', 'scanLocalModels', 'unloadModel']) handlers[type] = handler(type);
  const context = {
    currentLang: options.lang || 'en',
    console,
    window: {
      __MODEL_INTEGRITY_STATUS__: {}, __LOCAL_MODELS__: {}, __MODEL_PARAMS__: {}, __DEFAULT_MODEL__: '',
      innerWidth: 1200, innerHeight: 800, addEventListener() {},
      webkit: { messageHandlers: handlers },
    },
    document: {
      getElementById: element,
      querySelector: () => null,
      querySelectorAll: selector => body.querySelectorAll(selector),
      addEventListener() {},
      createElement: tag => new FakeElement(doc, tag),
      createElementNS: (_ns, tag) => new FakeElement(doc, tag),
      body,
      get activeElement() { return doc.activeElement; },
    },
    setInterval() { return 1; }, clearInterval() {},
    setTimeout(fn, ms) { timers.push([fn, ms]); return timers.length; }, clearTimeout() {},
    showToast() {}, syncModelList() {}, refreshSupportedModelActions() {}, renderProfileModelOptions() {},
    syncBenchmarkModels() {}, openParamsModal() {}, startDownload() {}, apiPost() {}, copyToClipboard() { return true; },
    verifyModelIntegrity() {}, openModelVersions() {}, deleteSingleModel() {}, navigateTo() {},
    showSidebarTooltip() {}, hideSidebarTooltip() {},
  };
  vm.createContext(context);
  doc.context = context;
  vm.runInContext(translations
    + slice('  function t(key, fallback)', '  function setLanguage(')
    + slice('  function setDefaultModel(', '  // ── Model Selection & Deletion ──')
    + slice('  function escapeAttr(', '  let _toastTimer')
    + slice('  function toggleModelLoad(', '  function unloadModel(')
    + slice('  // ── Model load progress', "  window.__CURRENT_PARAM_MODEL__ = '';")
    + slice('  function formatSize(', '  function modelReadiness(')
    + slice('  function modelReadiness(', '  function updateModelIntegrityRow(')
    + slice('  let activeModelMoreMenu = null;', '  // Called from Rust with local model scan results')
    + slice('  function onLocalModelsScanned(', '  let benchLocalModels')
    + slice('  const STATUS_DASH', '  function formatStatusClock(')
    + slice('  function idleUnloadText(', '  // Shares of the limit')
    + slice('  function localizeErrorResult(', '  // ── Model load progress')
    + block, context);
  for (const id of ['models-list', 'models-filter', 'model-import-empty', 'models-page-summary']) body.appendChild(element(id));
  // The page tells the App at start that no load panel is open; tests look at later messages.
  posted.length = 0;
  return {
    context, element, posted, timers, doc, body,
    list: element('models-list'),
    scan(models) { context.onLocalModelsScanned(JSON.stringify(models)); },
    push(...loads) { context.onModelLoadsChanged(JSON.stringify({ loads })); },
    snapshot(models) { context.runtimeSnapshotPublish({ models }); },
    rows() { return element('models-list').querySelectorAll('.ds-model-row'); },
    row(id) { return element('models-list').querySelectorAll('.ds-model-row').find(r => r.dataset.modelId === id) || null; },
    groups() { return element('models-list').querySelectorAll('.ds-model-group').map(g => g.dataset.group); },
    groupOf(id) {
      const group = element('models-list').querySelectorAll('.ds-model-group')
        .find(g => g.querySelectorAll('.ds-model-row').some(r => r.dataset.modelId === id));
      return group ? group.dataset.group : null;
    },
  };
}

const local = (o = {}) => Object.assign({ id: 'org/m', display_name: 'm', type: 'llm', loaded: false,
  size_mb: 1024, source: 'hf', readiness: { status: 'ready' } }, o);
const loadAct = (o = {}) => Object.assign({ model: 'org/m', variant: 'plain', kind: 'load', origin: 'dashboard',
  state: 'loading', stage: 'reading_weights', completedStages: [], bytesDone: GB, bytesTotal: 4 * GB,
  startedAt: 1000, cancellable: true, autoOpenPanel: false }, o);

test('render_builds_two_groups_and_summary', () => {
  const p = domPage();
  p.scan([local({ id: 'org/a', display_name: 'A', loaded: true, size_mb: 2048 }),
    local({ id: 'org/b', display_name: 'B', loaded: true, size_mb: 1024 }),
    local({ id: 'org/c', display_name: 'C', size_mb: 1024 }), local({ id: 'org/x', type: 'mtp' })]);
  assert.deepEqual(p.groups(), ['loaded', 'unloaded']);
  assert.deepEqual(p.rows().map(r => r.dataset.modelId), ['org/a', 'org/b', 'org/c']);
  const titles = p.list.querySelectorAll('.ds-model-group-title').map(el => el.textContent);
  assert.deepEqual(titles, [p.context.t('models_group_loaded'), p.context.t('models_group_unloaded')]);
  const summary = p.element('models-page-summary').textContent;
  assert.match(summary, /^3 models · 2 loaded · 4\.0 GB/);
  const row = p.row('org/a');
  for (const sel of ['.ds-model-dot', '.ds-model-name', '.ds-model-tags', '.ds-model-actions', '.ds-model-detail',
    '[data-model-pin]', '[data-model-params]', '[data-model-primary]', '.model-more']) assert.ok(row.querySelector(sel), sel);
  assert.equal(row.querySelector('.ds-model-name').getAttribute('title'), 'org/a');
  assert.equal(row.querySelector('[data-model-primary]').dataset.kind, 'unload');
  const chips = p.element('models-filter').querySelectorAll('.ds-chip');
  assert.deepEqual(chips.map(c => c.dataset.category), ['all', 'text']);
  assert.ok(chips[0].classList.contains('is-on'));
  assert.match(chips[0].textContent, /3/);
});

test('empty_group_is_omitted', () => {
  const p = domPage();
  p.scan([local({ id: 'org/a' }), local({ id: 'org/b' })]);
  assert.deepEqual(p.groups(), ['unloaded']);
  p.scan([local({ id: 'org/a', loaded: true })]);
  assert.deepEqual(p.groups(), ['loaded']);
});

test('no_models_shows_empty_state', () => {
  const p = domPage();
  p.scan([local()]);
  assert.equal(p.element('model-import-empty').style.display, 'none');
  p.scan([]);
  assert.equal(p.element('model-import-empty').style.display, '');
  assert.equal(p.rows().length, 0);
  assert.equal(p.element('models-filter').hidden, true);
  assert.equal(p.element('models-page-summary').textContent, '');
});

test('search_without_match_shows_hint', () => {
  const p = domPage();
  p.scan([local({ id: 'org/Qwen3', display_name: 'Qwen3' }), local({ id: 'org/gemma', display_name: 'gemma' })]);
  p.context.modelsPageSetQuery('QWEN');
  assert.deepEqual(p.rows().map(r => r.dataset.modelId), ['org/Qwen3']);
  p.context.modelsPageSetQuery('llama');
  assert.equal(p.rows().length, 0);
  assert.equal(p.list.querySelector('.ds-models-empty').textContent, p.context.t('models_no_match'));
  assert.equal(p.element('model-import-empty').style.display, 'none', 'the library is not empty');
});

test('patch_updates_detail_without_replacing_row', () => {
  const p = domPage();
  p.scan([local({ loaded: true })]);
  p.snapshot([{ id: 'org/m', state: 'loaded', active_requests: 1, live_decode_tokens_per_second: 10, weight_bytes: GB, kv_cache_bytes: 0 }]);
  const row = p.row('org/m');
  const actions = row.querySelector('.ds-model-actions');
  assert.match(row.querySelector('.ds-model-detail').textContent, /10 tok\/s/);
  p.snapshot([{ id: 'org/m', state: 'loaded', active_requests: 1, live_decode_tokens_per_second: 20, weight_bytes: GB, kv_cache_bytes: 0 }]);
  assert.equal(p.row('org/m'), row, 'the row element stays');
  assert.equal(row.querySelector('.ds-model-actions'), actions);
  assert.match(row.querySelector('.ds-model-detail').textContent, /20 tok\/s/);
  assert.ok(row.querySelector('.ds-model-dot').classList.contains('is-green'));
});

test('patch_keeps_actions_while_menu_open', () => {
  const p = domPage();
  p.scan([local({ loaded: true })]);
  const row = p.row('org/m');
  const actions = row.querySelector('.ds-model-actions');
  const primary = row.querySelector('[data-model-primary]');
  row.querySelector('.model-more').setAttribute('aria-expanded', 'true');
  // A reload starts: the primary would become "Cancel".
  p.push(loadAct({ kind: 'reload' }));
  assert.equal(row.querySelector('.ds-model-actions'), actions);
  assert.equal(row.querySelector('[data-model-primary]'), primary, 'not replaced while the menu is open');
  assert.match(row.querySelector('.ds-model-detail').textContent, /Reloading/, 'the detail still updates');
  row.querySelector('.model-more').setAttribute('aria-expanded', 'false');
  p.context.patchModelRows();
  assert.equal(row.querySelector('[data-model-primary]').dataset.kind, 'cancel', 'applied once the menu closes');
  assert.equal(row.querySelector('.ds-model-actions'), actions);
});

test('rebuild_then_patch_only_touches_new_rows', () => {
  const p = domPage();
  const models = [local({ id: 'org/a', loaded: true }), local({ id: 'org/b', loaded: true })];
  p.scan(models);
  const old = p.rows();
  p.scan(models);
  p.snapshot([{ id: 'org/a', state: 'loaded', active_requests: 1, live_decode_tokens_per_second: 33, weight_bytes: GB }]);
  assert.equal(p.rows().length, 2, 'no duplicate rows');
  assert.ok(p.rows().every(r => !old.includes(r)));
  assert.match(p.row('org/a').querySelector('.ds-model-detail').textContent, /33 tok\/s/);
  assert.doesNotMatch(old[0].querySelector('.ds-model-detail').textContent, /33/, 'detached rows are left alone');
});

test('loading_start_moves_row_to_loaded_group', () => {
  const p = domPage();
  p.scan([local(), local({ id: 'org/z', display_name: 'z' })]);
  assert.equal(p.groupOf('org/m'), 'unloaded');
  p.push(loadAct());
  assert.equal(p.groupOf('org/m'), 'loaded', 'without a rescan');
  const row = p.row('org/m');
  assert.equal(row.querySelector('[data-model-primary]').dataset.kind, 'cancel');
  assert.match(row.querySelector('.ds-progress-fill').getAttribute('style'), /width:25%/);
  // Progress moves in place.
  p.push(loadAct({ bytesDone: 2 * GB }));
  assert.equal(p.row('org/m'), row);
  assert.match(row.querySelector('.ds-progress-fill').getAttribute('style'), /width:50%/);
  assert.equal(row.querySelector('.ds-progress-pct').textContent, '50%');
  // Once loaded, the row offers unload and the pin without waiting for a rescan.
  p.push(loadAct({ state: 'loaded', bytesDone: 4 * GB, cancellable: false }));
  assert.equal(p.groupOf('org/m'), 'loaded');
  assert.equal(p.row('org/m').querySelector('[data-model-primary]').dataset.kind, 'unload');
  assert.ok(p.row('org/m').querySelector('[data-model-pin]'));
  assert.equal(p.row('org/m').querySelector('[data-model-primary]').disabled, true, 'settling after the load');
});

test('retry_acknowledges_then_loads_with_variant', () => {
  const p = domPage();
  p.scan([local({ dflash2: { status: 'available', candidates: [{ id: 'z/draft' }] } })]);
  p.push(loadAct({ variant: 'dflash2', state: 'failed', message: 'boom', cancellable: false }));
  const retry = p.row('org/m').querySelector('[data-model-primary]');
  assert.equal(retry.dataset.kind, 'retry');
  retry.click();
  const types = p.posted.map(m => m.type);
  assert.deepEqual(types, ['acknowledgeModelLoadFailure', 'loadModel']);
  assert.equal(p.posted[0].body, 'org/m');
  assert.equal(p.posted[1].body.use_dflash2, true);
  assert.equal(p.posted[1].body.dflash2_model_id, 'z/draft');

  // A failure only the snapshot reports: no acknowledgement, mode from the snapshot.
  const q = domPage();
  q.scan([local({ mtp: { status: 'available', candidates: [{ id: 'm/mtp' }] } })]);
  q.snapshot([{ id: 'org/m', state: 'failed', mode: 'mtp', last_error: 'x' }]);
  q.row('org/m').querySelector('[data-model-primary]').click();
  assert.deepEqual(q.posted.map(m => m.type), ['loadModel']);
  assert.equal(q.posted[0].body.use_mtp, true);
  assert.equal(q.posted[0].body.mtp_model_id, 'm/mtp');
});

test('retry_does_not_flash_unloaded', () => {
  const p = domPage();
  p.scan([local()]);
  p.snapshot([{ id: 'org/m', state: 'failed', mode: 'normal', last_error: 'x' }]);
  p.push(loadAct({ state: 'failed', message: 'boom', cancellable: false }));
  p.row('org/m').querySelector('[data-model-primary]').click();
  // Acknowledged; the snapshot still says failed: still a failure, never a plain unloaded row.
  const after = p.row('org/m');
  assert.notEqual(after.querySelector('.ds-model-dot').getAttribute('class'), 'ds-model-dot is-hollow');
  // The App lists the new load as queued while the snapshot is still failed.
  p.push(loadAct({ state: 'queued', stage: 'queued', startedAt: 2000 }));
  assert.equal(p.groupOf('org/m'), 'loaded');
  assert.match(p.row('org/m').querySelector('.ds-model-detail').textContent, /Queued/);
  assert.equal(p.row('org/m').querySelector('[data-model-primary]').dataset.kind, 'cancel');
});

test('split_button_menu_is_keyboard_usable', () => {
  const p = domPage();
  p.scan([local({ mtp: { status: 'available', candidates: [{ id: 'm/mtp' }] },
    dflash2: { status: 'available', candidates: [{ id: 'd/draft' }] } })]);
  const toggle = p.row('org/m').querySelector('.ds-split-toggle');
  toggle.click();
  const menu = p.body.querySelector('.model-more-menu');
  const items = menu.querySelectorAll('button');
  assert.equal(items.length, 3);
  assert.equal(p.context.document.activeElement, items[0], 'opening focuses the first choice');
  const key = name => {
    let prevented = false;
    menu.listeners.keydown({ key: name, preventDefault() { prevented = true; }, stopPropagation() {} });
    return prevented;
  };
  assert.ok(key('ArrowDown'));
  assert.equal(p.context.document.activeElement, items[1]);
  key('ArrowDown');
  key('ArrowDown');
  assert.equal(p.context.document.activeElement, items[0], 'wraps around');
  key('ArrowUp');
  assert.equal(p.context.document.activeElement, items[2], 'wraps backwards');
  key('Home');
  assert.equal(p.context.document.activeElement, items[0]);
  key('End');
  assert.equal(p.context.document.activeElement, items[2]);
  key('Escape');
  assert.equal(p.body.querySelector('.model-more-menu'), null, 'Escape closes the list');
  assert.equal(p.context.document.activeElement, toggle, 'Escape returns focus to the arrow button');
  assert.equal(toggle.getAttribute('aria-expanded'), 'false');
  // Tab leaves the list and closes it.
  toggle.click();
  const second = p.body.querySelector('.model-more-menu');
  second.listeners.keydown({ key: 'Tab', preventDefault() {}, stopPropagation() {} });
  assert.equal(p.body.querySelector('.model-more-menu'), null);
});

test('menu_set_default_follows_restrictions', () => {
  const p = domPage();
  p.context.window.__DEFAULT_MODEL__ = 'org/cur';
  p.scan([local({ id: 'org/cur' }), local({ id: 'org/llm' }), local({ id: 'org/emb', type: 'embedding' })]);
  const menuFor = id => {
    p.row(id).querySelector('.model-more').click();
    const menu = p.body.querySelector('.model-more-menu');
    const items = menu.querySelectorAll('button');
    return { menu, items, defaultItem: menu.querySelector('.model-more-default') };
  };
  const llm = menuFor('org/llm');
  assert.equal(llm.items[0], llm.defaultItem, 'set as default comes first');
  assert.equal(llm.defaultItem.textContent, p.context.t('model_menu_set_default'));
  llm.defaultItem.click();
  assert.deepEqual(p.posted.filter(m => m.type === 'setDefaultModel').map(m => m.body), ['org/llm']);
  assert.equal(p.body.querySelector('.model-more-menu'), null, 'the menu closes');
  assert.ok(p.row('org/llm').querySelector('.ds-model-tag.is-default'), 'the default tag moves');
  assert.equal(p.row('org/cur').querySelector('.ds-model-tag.is-default'), null);

  const current = menuFor('org/llm');
  assert.equal(current.defaultItem.disabled, true);
  assert.equal(current.defaultItem.getAttribute('aria-checked'), 'true');
  p.context.closeModelMoreActions();
  const emb = menuFor('org/emb');
  assert.equal(emb.defaultItem, null);
  assert.ok(emb.items.length >= 3, 'the other items stay');
});

test('split_button_options', () => {
  const p = domPage();
  const model = local({ mtp: { status: 'available', candidates: [{ id: 'm/mtp' }] },
    dflash2: { status: 'available', candidates: [{ id: 'd/draft' }] } });
  for (const [index, check] of [[0, b => b.use_mtp === false && b.use_dflash2 === false],
    [1, b => b.use_mtp === true && b.mtp_model_id === 'm/mtp'],
    [2, b => b.use_dflash2 === true && b.dflash2_model_id === 'd/draft']]) {
    p.scan([model]);
    p.posted.length = 0;
    const row = p.row('org/m');
    assert.equal(row.querySelector('[data-model-primary]').dataset.kind, 'load-split');
    const toggle = row.querySelector('.ds-split-toggle');
    toggle.click();
    assert.equal(toggle.getAttribute('aria-expanded'), 'true');
    const items = p.body.querySelector('.model-more-menu').querySelectorAll('button');
    assert.equal(items.length, 3);
    items[index].click();
    const sent = p.posted.filter(m => m.type === 'loadModel');
    assert.equal(sent.length, 1, `option ${index}`);
    assert.ok(check(sent[0].body), `option ${index}: ${JSON.stringify(sent[0].body)}`);
    assert.equal(p.body.querySelector('.model-more-menu'), null);
  }
  // The main part loads as usual.
  p.scan([model]);
  p.posted.length = 0;
  p.row('org/m').querySelector('[data-model-primary]').click();
  assert.equal(p.posted[0].body.use_mtp, false);
  assert.equal(p.posted[0].body.use_dflash2, false);
});

test('filter_survives_rescan', () => {
  const p = domPage();
  const models = [local({ id: 'org/qwen-vl', display_name: 'qwen-vl', type: 'vlm' }),
    local({ id: 'org/qwen', display_name: 'qwen' }), local({ id: 'org/gemma-vl', display_name: 'gemma-vl', type: 'vlm' })];
  p.scan(models);
  p.element('models-filter').querySelectorAll('.ds-chip').find(c => c.dataset.category === 'vision').click();
  p.context.modelsPageSetQuery('qwen');
  assert.deepEqual(p.rows().map(r => r.dataset.modelId), ['org/qwen-vl']);
  p.scan(models);
  assert.deepEqual(p.rows().map(r => r.dataset.modelId), ['org/qwen-vl']);
  const on = p.element('models-filter').querySelectorAll('.ds-chip.is-on').map(c => c.dataset.category);
  assert.deepEqual(on, ['vision']);
});

test('pin_only_on_loaded_rows', () => {
  const p = domPage();
  p.scan([local({ id: 'org/a', loaded: true, pinned: true }), local({ id: 'org/b' })]);
  const pin = p.row('org/a').querySelector('[data-model-pin]');
  assert.ok(pin.classList.contains('pinned'));
  assert.equal(pin.dataset.model, 'org/a');
  assert.ok(pin.dataset.modelTooltip);
  assert.equal(p.row('org/b').querySelector('[data-model-pin]'), null);
});

test('dflash2_target_row_has_unload_with_hint', () => {
  const p = domPage();
  p.scan([local({ loaded: true, dflash2: { enabled: true, status: 'available', candidates: [{}] } })]);
  const row = p.row('org/m');
  assert.equal(p.groupOf('org/m'), 'loaded');
  const primary = row.querySelector('[data-model-primary]');
  assert.equal(primary.dataset.kind, 'unload');
  assert.equal(primary.getAttribute('title'), p.context.t('model_dflash2_unload_hint'));
  assert.ok(row.querySelector('.ds-model-tag.is-accel'));
  assert.equal(row.querySelector('[data-model-pin]').dataset.dflash2, '1');
  // A failed entry in the snapshot does not offer retry.
  p.snapshot([{ id: 'org/m', state: 'failed', mode: 'dflash2', last_error: 'x' }]);
  assert.equal(p.row('org/m').querySelector('[data-model-primary]').dataset.kind, 'unload');
});

test('status_retry_focuses_new_row', () => {
  const p = domPage();
  p.scan([local({ id: 'org/a' }), local({ id: 'org/b', type: 'vlm' })]);
  p.context.modelsPageSetFilter('vision');
  assert.equal(p.context.statusFocusModelLoad, undefined, 'the status block is not loaded here');
  assert.equal(p.context.modelsPageFocusRow('org/a'), true, 'a filtered-out row is shown again');
  const row = p.row('org/a');
  assert.ok(row.classList.contains('model-load-focus'));
  assert.equal(p.context.document.activeElement, row.querySelector('[data-model-primary]'));
  assert.equal(p.context.modelsPageFocusRow('org/none'), false);
});

test('backend_text_is_escaped_and_raw_errors_stay_in_tooltips', () => {
  const p = domPage();
  p.scan([local({ id: 'org/<b>', display_name: '<img src=x onerror=alert(1)>' })]);
  p.snapshot([{ id: 'org/<b>', state: 'failed', mode: 'normal', last_error: '<script>raw backend</script>' }]);
  const row = p.row('org/<b>');
  assert.equal(row.querySelector('img'), null);
  assert.equal(row.querySelector('script'), null);
  const detail = row.querySelector('.ds-model-detail');
  assert.doesNotMatch(detail.textContent, /raw backend/);
  assert.equal(detail.getAttribute('title'), '&lt;script&gt;raw backend&lt;/script&gt;');
});

test('unsupported_reason_stays_in_tooltip', () => {
  const p = domPage();
  const reason = "Error: unsupported model_type: qwen4_exp (expected 'qwen3_5')";
  p.scan([local({ id: 'org/u', readiness: { status: 'unsupported', reason_code: 'unsupported_model_type', message: reason } })]);
  const detail = p.row('org/u').querySelector('.ds-model-detail');
  assert.doesNotMatch(detail.textContent, /qwen4_exp|Error:/);
  assert.ok(detail.textContent.trim().length > 0);
  assert.match(detail.getAttribute('title'), /qwen4_exp/);
});

test('unloadable_row_shows_disabled_load_button', () => {
  const p = domPage();
  p.scan([local({ id: 'org/u', readiness: { status: 'unsupported', reason_code: 'download_only_model' } })]);
  const primary = p.row('org/u').querySelector('[data-model-primary]');
  assert.ok(primary, 'the slot keeps a button so the row lines up with the others');
  assert.equal(primary.dataset.kind, 'unavailable');
  assert.equal(primary.getAttribute('aria-disabled'), 'true');
  assert.equal(primary.getAttribute('onclick'), null);
  assert.equal(primary.textContent, p.context.t('action_load'));
  assert.ok(primary.getAttribute('title'));
});

test('params_button_uses_sliders_icon_and_primary_slot_has_fixed_width', () => {
  assert.doesNotMatch(html, /MODELS_GEAR_ICON/);
  assert.match(html, /const MODELS_PARAMS_ICON = /);
  assert.match(html, /\.ds-model-primary \{[^}]*min-width: 102px/);
  assert.doesNotMatch(html, /\.ds-model-primary:empty \{ display: none; \}/);
});

test('group_memory_follows_the_snapshot', () => {
  const p = domPage();
  p.scan([local({ id: 'org/a', loaded: true }), local({ id: 'org/b', loaded: true })]);
  const memory = p.list.querySelector('.ds-model-group-memory');
  assert.ok(memory.hasAttribute('hidden'), 'hidden without a snapshot');
  p.snapshot([{ id: 'org/a', state: 'loaded', weight_bytes: 2 * GB, kv_cache_bytes: GB },
    { id: 'org/b', state: 'loaded', weight_bytes: GB, kv_cache_bytes: 0 }]);
  assert.equal(memory.hasAttribute('hidden'), false);
  assert.match(memory.textContent, /4\.0/);
});

// The loaded total reports everything that holds memory, whatever the filter shows.
test('group_memory_counts_every_loaded_model_under_a_filter', () => {
  const p = domPage();
  p.scan([local({ id: 'org/text', loaded: true }), local({ id: 'org/vision', type: 'vlm', loaded: true })]);
  p.snapshot([{ id: 'org/text', state: 'loaded', weight_bytes: GB, kv_cache_bytes: 0 },
    { id: 'org/vision', state: 'loaded', weight_bytes: 2 * GB, kv_cache_bytes: 0 }]);
  p.element('models-filter').querySelectorAll('.ds-chip').find(c => c.dataset.category === 'text').click();
  assert.equal(p.list.querySelectorAll('.ds-model-row').length, 1);
  const memory = p.list.querySelector('.ds-model-group-memory');
  assert.match(memory.textContent, /3\.0/);
  p.snapshot([{ id: 'org/text', state: 'loaded', weight_bytes: GB, kv_cache_bytes: GB },
    { id: 'org/vision', state: 'loaded', weight_bytes: 2 * GB, kv_cache_bytes: 0 }]);
  assert.match(p.list.querySelector('.ds-model-group-memory').textContent, /4\.0/);
});

test('app_failure_detail_opens_the_load_panel', () => {
  const p = domPage();
  p.scan([local()]);
  p.push(loadAct({ state: 'failed', message: 'boom', cancellable: false }));
  const detail = p.row('org/m').querySelector('.ds-model-detail');
  assert.ok(detail.classList.contains('is-link'));
  detail.click();
  assert.ok(p.posted.some(m => m.type === 'openModelLoadPanel' && m.body === 'org/m'));
});

test('new_model_page_strings_exist_in_five_languages', () => {
  const p = page();
  const I18N = vm.runInContext('I18N', p);
  for (const key of ['models_filter_all', 'models_search_placeholder', 'models_no_match', 'models_group_loaded',
    'models_group_unloaded', 'models_group_memory', 'model_menu_set_default', 'model_dflash2_unload_hint',
    'model_load_split_more']) {
    assert.ok(I18N.en[key], `en.${key}`);
    for (const lang of ['zh-Hans', 'zh-Hant', 'ja', 'ko']) {
      assert.ok(I18N[lang][key], `${lang}.${key}`);
      assert.notEqual(I18N[lang][key], I18N.en[key], `${lang}.${key} must be translated`);
    }
  }
});

test('row_css_keeps_names_on_one_line', () => {
  const rule = selector => {
    const match = html.match(new RegExp(`\\n\\s*${selector.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\s*\\{([^}]*)\\}`));
    assert.ok(match, `missing ${selector}`);
    return match[1];
  };
  assert.match(rule('.ds-model-name'), /min-width:\s*0/);
  assert.match(rule('.ds-model-name'), /text-overflow:\s*ellipsis/);
  assert.match(rule('.ds-model-actions'), /white-space:\s*nowrap/);
  assert.match(rule('.ds-model-tags'), /flex-wrap:\s*wrap/);
  assert.match(rule('.ds-model-title'), /display:\s*flex/);
  assert.doesNotMatch(rule('.ds-model-title'), /flex-wrap:\s*wrap/);
  assert.match(rule('.ds-model-name'), /white-space:\s*nowrap/);
  assert.match(rule('.ds-model-source'), /flex:\s*none/);
  assert.match(html, /\.model-more-menu button:focus-visible\s*\{/);
  assert.match(html, /\.model-more-menu button:focus,/);
  assert.match(html, /<div class="ds-models-list" id="models-list"><\/div>/);
  assert.doesNotMatch(html, /id="model-table"/);
});
