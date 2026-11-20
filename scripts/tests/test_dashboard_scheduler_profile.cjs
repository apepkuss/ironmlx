const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const html = fs.readFileSync(path.join(__dirname,
  '../../ironmlx-app/Sources/IronMLXAppCore/Resources/dashboard2.html'), 'utf8');
const extract = (start, end) => html.slice(html.indexOf(`  function ${start}(`),
  html.indexOf(`  function ${end}(`));
const LANGUAGES = ['en', 'zh-Hans', 'zh-Hant', 'ja', 'ko'];

function translations() {
  const start = html.indexOf('  const I18N =');
  return vm.runInNewContext(`${html.slice(start, html.indexOf('\n  };', start) + 5)}\nI18N`, {});
}

function escapeAttr(value) {
  return String(value).replace(/&/g, '&amp;').replace(/"/g, '&quot;')
    .replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

test('every language names the scheduler configuration and the unusable-profile notice', () => {
  const I18N = translations();
  for (const language of LANGUAGES) {
    const dict = I18N[language];
    for (const key of ['scheduler_config_label', 'scheduler_config_default',
      'scheduler_config_default_settings_mismatch', 'scheduler_config_profile',
      'warn_scheduler_profile_hardware_mismatch', 'warn_scheduler_profile_unusable']) {
      assert.equal(typeof dict[key], 'string', `${language}.${key}`);
      assert.ok(dict[key].trim(), `${language}.${key} is not empty`);
    }
    assert.ok(dict.warn_scheduler_profile_unusable.includes(dict.nav_models)
      && dict.warn_scheduler_profile_unusable.includes(dict.models_tab_tuning),
      `${language} names the Models page and the tab that re-tunes the model`);
    for (const removed of ['warn_default_scheduler_profile_used', 'warn_scheduler_profile_unmatched']) {
      assert.equal(dict[removed], undefined, `${language} no longer has ${removed}`);
    }
  }
  const zh = I18N['zh-Hans'];
  assert.equal(zh.scheduler_config_label, '调度配置');
  assert.equal(zh.scheduler_config_default, '默认');
  assert.equal(zh.scheduler_config_default_settings_mismatch, '默认（与当前设置不匹配）');
  assert.equal(zh.scheduler_config_profile, '专用 profile');
  assert.equal(zh.warn_scheduler_profile_hardware_mismatch,
    '该模型的 scheduler profile 是在其他硬件上生成的，本次使用默认调度配置。在本机重新调优即可获得适合本机的调度。');
  assert.equal(zh.warn_scheduler_profile_unusable,
    '该模型的 scheduler profile 无法使用（文件损坏或已失效），本次使用默认调度配置。请在「模型 → 性能调优」中重新调优。');
  assert.ok(!html.includes('scheduler_profile_unmatched'), 'no dead warning path');
});

function loadPage(language) {
  const toasts = [];
  const context = {
    I18N: translations(), currentLang: language, toasts,
    showToast(message, kind) { toasts.push({ message, kind }); },
    clearServerBanner() {}, syncModelList() {},
    console: { warn() {}, error() {} },
  };
  vm.createContext(context);
  vm.runInContext(extract('onModelLoaded', 'localizeErrorResult')
    + extract('localizeWarningResult', 'localizeBackendErrorMessage'), context);
  return context;
}

test('default, settings-mismatch and legacy default-profile loads show no toast', () => {
  for (const language of LANGUAGES) {
    const page = loadPage(language);
    page.onModelLoaded(JSON.stringify({ success: true, status: 'loaded', model: 'm' }));
    // A settings mismatch carries no warning; it is only shown passively.
    page.onModelLoaded(JSON.stringify({ success: true, status: 'loaded', model: 'm',
      loaded_models: [{ id: 'm', scheduler_profile: 'default_settings_mismatch' }] }));
    // An older backend still sends this code for every default load.
    page.onModelLoaded(JSON.stringify({ success: true, status: 'loaded', model: 'm',
      warning_code: 'default_scheduler_profile_used', warning: 'No matching scheduler profile.' }));
    assert.deepEqual(page.toasts, [], language);
  }
});

test('other-hardware and broken profiles each show one localized toast', () => {
  for (const code of ['scheduler_profile_hardware_mismatch', 'scheduler_profile_unusable']) {
    for (const language of LANGUAGES) {
      const page = loadPage(language);
      page.onModelLoaded(JSON.stringify({
        success: true, status: 'loaded', model: 'm',
        warning_code: code, warning: 'English backend message',
      }));
      assert.equal(page.toasts.length, 1, `${language} ${code}`);
      assert.equal(page.toasts[0].message, page.I18N[language][`warn_${code}`]);
      assert.equal(page.toasts[0].kind, 'info');
    }
  }
});

function statusCard(language) {
  const card = { style: {} };
  const container = { innerHTML: '' };
  const context = {
    I18N: translations(), currentLang: language, window: {}, Date,
    DECISION_RECENT_ACTIVITY_MS: 2500,
    runtimeActiveKvOpenModelIds: new Set(),
    captureRuntimeActiveKvOpenState() {}, renderActiveKvIncident() {},
    formatVersionBytes: String, escapeAttr,
    document: { getElementById(id) { return id === 'runtime-model-health-card' ? card : container; } },
  };
  vm.createContext(context);
  vm.runInContext(extract('runtimeOperationalState', 'renderRuntimeModels')
    + extract('renderRuntimeModels', 'onApiFetchResult'), context);
  return { context, container };
}

const causal = scheduler_profile => ({
  id: 'mlx-community/Qwen3.6-27B-4bit', runtime_kind: 'causal', queue_capacity: 8,
  usage: { cumulative_tokens: 12 }, scheduler_profile,
});

test('the runtime card shows which scheduler configuration a model runs with', () => {
  for (const language of LANGUAGES) {
    const { context, container } = statusCard(language);
    const dict = context.I18N[language];
    context.renderRuntimeModels([causal('default')]);
    assert.ok(container.innerHTML.includes(escapeAttr(dict.scheduler_config_label)), `${language} label`);
    assert.ok(container.innerHTML.includes('<strong>' + escapeAttr(dict.scheduler_config_default) + '</strong>'),
      `${language} default`);
    context.renderRuntimeModels([causal('profile')]);
    assert.ok(container.innerHTML.includes('<strong>' + escapeAttr(dict.scheduler_config_profile) + '</strong>'),
      `${language} profile`);
    context.renderRuntimeModels([causal('default_settings_mismatch')]);
    assert.ok(container.innerHTML.includes('<strong>'
      + escapeAttr(dict.scheduler_config_default_settings_mismatch) + '</strong>'), `${language} settings mismatch`);
  }
  const { context, container } = statusCard('zh-Hans');
  context.renderRuntimeModels([causal(undefined)]);
  assert.ok(!container.innerHTML.includes('调度配置'), 'no line without the field');
  context.renderRuntimeModels([causal('<img src=x onerror=alert(1)>')]);
  assert.ok(!container.innerHTML.includes('<img'), 'unknown values are not rendered');
  assert.ok(!container.innerHTML.includes('调度配置'));
  context.renderRuntimeModels([causal('default')]);
  assert.match(container.innerHTML, /调度配置<strong>默认<\/strong>/);
});

test('the parameter dialog of a loaded model shows its scheduler configuration', () => {
  const header = html.slice(html.indexOf('<div class="modal-overlay" id="model-params-modal">'),
    html.indexOf('<div class="modal-body"', html.indexOf('<div class="modal-overlay" id="model-params-modal">')));
  assert.match(header, /<div class="model-params-scheduler-profile" id="modal-scheduler-profile" hidden><\/div>/);
  assert.match(extract('openParamsModal', 'renderParamsSchedulerProfile'), /renderParamsSchedulerProfile\(name\);/);
  const healthBranch = html.slice(html.indexOf("} else if (path === '/health') {"),
    html.indexOf("settingsBackendRunning = true;", html.indexOf("} else if (path === '/health') {")));
  assert.match(healthBranch, /refreshOpenParamsSchedulerProfile\(\);/,
    'an open dialog follows the backend state');
  assert.match(extract('setLanguage', 'selectInterfaceLanguage'), /refreshOpenParamsSchedulerProfile\(\);/,
    'an open dialog follows a language switch');

  const line = { hidden: true, innerHTML: '' };
  const overlay = { classList: { open: false, contains(name) { return name === 'open' && this.open; } } };
  const context = {
    I18N: translations(), currentLang: 'zh-Hans', escapeAttr,
    window: { __CURRENT_PARAM_MODEL__: 'plain', __RUNTIME_MODELS__: [
      { id: 'tuned', scheduler_profile: 'profile' },
      { id: 'plain', scheduler_profile: 'default' },
      { id: 'changed', scheduler_profile: 'default_settings_mismatch' },
      { id: 'audio' },
    ] },
    document: { getElementById(id) {
      return id === 'modal-scheduler-profile' ? line : id === 'model-params-modal' ? overlay : null;
    } },
  };
  vm.createContext(context);
  vm.runInContext(extract('schedulerProfileText', 'onApiFetchResult')
    + extract('renderParamsSchedulerProfile', 'renderDFlash2Params'), context);
  context.refreshOpenParamsSchedulerProfile();
  assert.equal(line.innerHTML, '', 'a closed dialog is left alone');
  overlay.classList.open = true;
  context.refreshOpenParamsSchedulerProfile();
  assert.equal(line.innerHTML, '调度配置<strong>默认</strong>');
  context.currentLang = 'en';
  context.refreshOpenParamsSchedulerProfile();
  assert.equal(line.innerHTML, 'Scheduling<strong>Default</strong>', 'the open dialog follows the language');
  context.currentLang = 'zh-Hans';
  context.renderParamsSchedulerProfile('changed');
  assert.equal(line.innerHTML, '调度配置<strong>默认（与当前设置不匹配）</strong>');
  context.renderParamsSchedulerProfile('tuned');
  assert.equal(line.hidden, false);
  assert.equal(line.innerHTML, '调度配置<strong>专用 profile</strong>');
  context.renderParamsSchedulerProfile('plain');
  assert.equal(line.innerHTML, '调度配置<strong>默认</strong>');
  for (const id of ['audio', 'not-loaded']) {
    context.renderParamsSchedulerProfile(id);
    assert.equal(line.hidden, true, id);
    assert.equal(line.innerHTML, '', id);
  }
  context.currentLang = 'ja';
  context.renderParamsSchedulerProfile('plain');
  assert.equal(line.innerHTML, escapeAttr(context.I18N.ja.scheduler_config_label)
    + '<strong>' + escapeAttr(context.I18N.ja.scheduler_config_default) + '</strong>');
});

test('a busy model gets its own localized generation failure in every language', () => {
  const I18N = translations();
  const context = { I18N };
  vm.createContext(context);
  vm.runInContext(extract('schedulerProfileFailureDetail', 'onSchedulerProfileStatus'), context);
  const details = new Set();
  for (const language of LANGUAGES) {
    const dict = I18N[language];
    assert.equal(typeof dict.scheduler_profile_model_busy, 'string', `${language} busy message`);
    assert.ok(dict.scheduler_profile_model_busy.trim(), `${language} busy message is not empty`);
    assert.notEqual(dict.scheduler_profile_model_busy, dict.unknown_error, language);
    details.add(dict.scheduler_profile_model_busy);
    assert.equal(context.schedulerProfileFailureDetail({ state: 'failed', code: 'model_busy' }, dict),
      dict.scheduler_profile_model_busy, language);
    assert.equal(context.schedulerProfileFailureDetail({ state: 'failed', error: 'exit 1' }, dict),
      dict.unknown_error, language);
  }
  assert.equal(details.size, LANGUAGES.length, 'each language has its own text');
  assert.equal(I18N['zh-Hans'].scheduler_profile_model_busy, '该模型正被其他操作使用，请稍后再试。');
  const status = extract('onSchedulerProfileStatus', 'closeSchedulerProfileResult');
  assert.equal((status.match(/schedulerProfileFailureDetail\((?:status|shown), /g) || []).length, 2,
    'the toast and the status line both use it');
});

test('the scheduling model menu offers only loadable models', () => {
  const options = [];
  const select = {
    value: '',
    set innerHTML(value) { options.length = 0; },
    appendChild(option) { options.push(option); },
    get options() { return options; },
  };
  let previews = 0;
  const context = {
    I18N: translations(), currentLang: 'en',
    requestSchedulerProfilePreview() { previews += 1; },
    document: {
      getElementById(id) { return id === 'profile-model-select' ? select : null; },
      createElement() { return {}; },
    },
  };
  vm.createContext(context);
  context.window = {};
  vm.runInContext(extract('modelReadiness', 'modelReadinessTitle')
    + extract('profileOptionValue', 'runSchedulerProfileGeneration')
    + extract('renderProfileModelOptions', 'syncBenchmarkModels'), context);
  context.renderProfileModelOptions([
    { id: 'ready', capabilities: { runtime_kind: 'causal' }, readiness: { status: 'ready' } },
    { id: 'legacy', capabilities: { runtime_kind: 'causal' } },
    { id: 'partial', capabilities: { runtime_kind: 'causal' }, readiness: { status: 'incomplete' } },
    { id: 'embedder', capabilities: { runtime_kind: 'embedding' }, readiness: { status: 'ready' } },
  ]);
  assert.deepEqual(options.map(option => option.value), ['ready', 'legacy']);
  assert.equal(previews, 1);
});

test('a scheduler profile preview failure is logged once as a warning until it changes', () => {
  const warnings = [];
  const context = {
    schedulerProfilePreviewFailures: {},
    console: { warn(...args) { warnings.push(args.join(' ')); }, error() { throw new Error('no error log'); } },
  };
  vm.createContext(context);
  vm.runInContext(extract('logSchedulerProfilePreviewOutcome', 'renderSchedulerProfilePreview'), context);
  const failure = { success: false, error: 'Model not found locally: m' };
  context.logSchedulerProfilePreviewOutcome('m', failure);
  context.logSchedulerProfilePreviewOutcome('m', failure);
  context.logSchedulerProfilePreviewOutcome('other', failure);
  assert.equal(warnings.length, 2);
  context.logSchedulerProfilePreviewOutcome('m', { success: false, error: 'different' });
  assert.equal(warnings.length, 3);
  context.logSchedulerProfilePreviewOutcome('m', { success: true });
  context.logSchedulerProfilePreviewOutcome('m', failure);
  assert.equal(warnings.length, 4, 'a failure after a success is logged again');
  assert.ok(!extract('renderSchedulerProfilePreview', 'formatProfileEstimateDuration').includes('console.'),
    'rendering the hint logs nothing');
});


// ── Tuning tab ──

function fakeElement(id) {
  const attrs = {};
  const classes = new Set();
  const element = {
    id, style: {}, value: '', disabled: false, textContent: '', innerHTML: '', scrollTop: 0, scrollHeight: 999,
    classList: {
      add(name) { classes.add(name); }, remove(name) { classes.delete(name); },
      contains(name) { return classes.has(name); },
      toggle(name, on) { if (on === undefined ? !classes.has(name) : on) classes.add(name); else classes.delete(name); },
    },
    setAttribute(name, value) { attrs[name] = String(value); },
    getAttribute(name) { return attrs[name] ?? null; },
    removeAttribute(name) { delete attrs[name]; },
  };
  if (id === 'profile-progress') {
    const bar = fakeElement('bar');
    element.querySelector = selector => selector === '.profile-progress-bar' ? bar : null;
    element.bar = bar;
  }
  return element;
}

// A page with lazily created elements, radio groups and recorded bridge messages.
function tuningPage(language = 'zh-Hans', overrides = {}) {
  const elements = {};
  const radios = {};
  const messages = [];
  const timers = { set: [], cleared: 0 };
  const handler = name => ({ postMessage(body) { messages.push({ name, body }); } });
  const context = {
    I18N: translations(), currentLang: language, elements, radios, messages, timers,
    schedulerProfilePollTimer: null, schedulerProfilePreviewToken: 0, schedulerProfilePreview: null,
    schedulerProfilePreviewFailures: {}, schedulerProfileObservedActive: false, schedulerProfileDismissed: '',
    toasts: [], confirmations: [], confirmAnswer: true,
    Intl, Date, JSON, Object, Array, Number, String, Math,
    console: { warn() {}, error() {} },
    showToast(message, kind) { context.toasts.push({ message, kind }); },
    showConfirmDialog(message, options) { context.confirmations.push({ message, options }); return Promise.resolve(context.confirmAnswer); },
    escapeAttr,
    setInterval(fn, ms) { timers.set.push(ms); return timers.set.length; },
    clearInterval() { timers.cleared += 1; },
    window: {
      __LOCAL_MODELS__: {}, __TUNED_MODELS__: {},
      webkit: { messageHandlers: new Proxy({}, { get: (_, name) => handler(String(name)) }) },
    },
    document: {
      getElementById(id) { return elements[id] || (elements[id] = fakeElement(id)); },
      querySelector(selector) {
        const match = /name="([^"]+)"\]:checked/.exec(selector);
        return match && radios[match[1]] ? { value: radios[match[1]] } : null;
      },
      createElement() { return {}; },
    },
    ...overrides,
  };
  vm.createContext(context);
  vm.runInContext(
    extract('modelReadiness', 'modelReadinessTitle')
    + extract('profileOptionValue', 'requestSchedulerProfilePreview')
    + extract('requestSchedulerProfilePreview', 'onSchedulerProfilePreview')
    + extract('renderSchedulerProfilePreview', 'formatProfileEstimateDuration')
    + extract('formatProfileEstimateDuration', 'refreshSchedulerProfileStatus')
    + extract('refreshSchedulerProfileStatus', 'renderSchedulerProfileStatus')
    + extract('renderSchedulerProfileStatus', 'closeSchedulerProfileResult')
    + extract('closeSchedulerProfileResult', 'copySchedulerProfilePath')
    + extract('renderProfileModelOptions', 'syncBenchmarkModels'), context);
  return context;
}

test('option_value_defaults: radio groups fall back to their documented defaults', () => {
  const page = tuningPage();
  assert.equal(page.profileOptionValue('profile-level'), 'standard');
  assert.equal(page.profileOptionValue('profile-selection'), 'agent-long-prompt');
  page.radios['profile-level'] = 'quick';
  page.radios['profile-selection'] = 'balanced';
  assert.equal(page.profileOptionValue('profile-level'), 'quick');
  assert.equal(page.profileOptionValue('profile-selection'), 'balanced');
});

test('preview_and_run_read_cards: both requests use the radio cards', async () => {
  const page = tuningPage();
  page.document.getElementById('profile-model-select').value = 'org/Model-4bit';
  page.radios['profile-level'] = 'full';
  page.radios['profile-selection'] = 'balanced';
  page.requestSchedulerProfilePreview();
  const preview = page.messages.find(message => message.name === 'previewSchedulerProfileGeneration');
  assert.deepEqual(JSON.parse(preview.body), {
    model: 'org/Model-4bit', selection_profile: 'balanced', calibration_level: 'full', request_token: 1,
  });
  await page.runSchedulerProfileGeneration();
  const run = page.messages.find(message => message.name === 'generateSchedulerProfile');
  assert.deepEqual(JSON.parse(run.body), {
    model: 'org/Model-4bit', selection_profile: 'balanced', calibration_level: 'full',
  });
  assert.deepEqual(page.confirmations, [], 'nothing is loaded, so there is nothing to confirm');
});

test('tuned_suffix_priority: a changed setting beats the tuned date', () => {
  const page = tuningPage('zh-Hans');
  const tunedAt = Date.UTC(2026, 10, 12, 8);
  const sameYear = new Date(tunedAt);
  const suffix = page.tunedModelSuffix('m', tunedAt, [], 'zh-Hans');
  assert.ok(suffix.startsWith('已调优 · '), suffix);
  assert.ok(/11月12日|11\/12/.test(suffix) || suffix.includes('11'), suffix);
  assert.equal(page.tunedModelSuffix('m', tunedAt, [{ id: 'm', scheduler_profile: 'profile' }], 'zh-Hans'), suffix);
  assert.equal(page.tunedModelSuffix('m', tunedAt, [{ id: 'm', scheduler_profile: 'default_settings_mismatch' }], 'zh-Hans'),
    '设置已变，建议重新调优');
  assert.equal(page.tunedModelSuffix('m', undefined, [{ id: 'm', scheduler_profile: 'default_settings_mismatch' }], 'zh-Hans'),
    '设置已变，建议重新调优');
  assert.equal(page.tunedModelSuffix('m', undefined, [{ id: 'm', scheduler_profile: 'default' }], 'zh-Hans'), '');
  assert.equal(page.tunedModelSuffix('other', undefined, [], 'zh-Hans'), '');
  assert.equal(page.tunedModelSuffix('m', 0, null, 'zh-Hans'), '', 'a zero timestamp is not a date');
  // The year is added only when it is not the current one.
  const old = Date.UTC(2020, 2, 3, 12);
  assert.ok(page.tunedModelSuffix('m', old, [], 'en').includes('2020'));
  assert.ok(!page.tunedModelSuffix('m', Date.now(), [], 'en').includes(String(new Date().getFullYear())));
  assert.ok(sameYear instanceof Date);
});

test('option_text_includes_suffix: the option shows the status, its value stays the model id', () => {
  const options = [];
  const select = {
    value: '', set innerHTML(value) { options.length = 0; }, appendChild(option) { options.push(option); },
    get options() { return options; },
  };
  const page = tuningPage('en', {});
  page.document.getElementById = id => id === 'profile-model-select' ? select : fakeElement(id);
  page.window.__TUNED_MODELS__ = { tuned: Date.UTC(2020, 2, 3, 12) };
  page.runtimeState = { snapshot: { models: [{ id: 'changed', scheduler_profile: 'default_settings_mismatch' }] } };
  page.renderProfileModelOptions([
    { id: 'tuned', capabilities: { runtime_kind: 'causal' }, readiness: { status: 'ready' } },
    { id: 'changed', capabilities: { runtime_kind: 'causal' }, readiness: { status: 'ready' } },
    { id: 'plain', capabilities: { runtime_kind: 'causal' }, readiness: { status: 'ready' } },
  ]);
  assert.deepEqual(options.map(option => option.value), ['tuned', 'changed', 'plain']);
  assert.ok(options[0].textContent.startsWith('tuned'));
  assert.ok(options[0].textContent.includes('Tuned'), options[0].textContent);
  assert.ok(options[1].textContent.includes('Settings changed'), options[1].textContent);
  assert.equal(options[2].textContent, 'plain');
  // A new snapshot rewrites the text of the existing options without rebuilding the list.
  page.runtimeState = { snapshot: { models: [] } };
  const before = options.slice();
  page.refreshProfileModelSuffixes();
  assert.ok(options.every((option, index) => option === before[index]));
  assert.ok(!options[1].textContent.includes('Settings changed'));
});

test('tuned models arrive as a JSON string, an object or an error', () => {
  const page = tuningPage('en');
  page.window.__TUNED_MODELS__ = { stale: 1 };
  page.onTunedModels(JSON.stringify({ models: { a: 1700000000000 } }));
  assert.deepEqual(page.window.__TUNED_MODELS__, { a: 1700000000000 });
  page.onTunedModels({ models: { b: 2 } });
  assert.deepEqual(page.window.__TUNED_MODELS__, { b: 2 });
  const warnings = [];
  page.console.warn = (...args) => warnings.push(args.join(' '));
  page.onTunedModels(JSON.stringify({ models: {}, error: 'scan exploded' }));
  assert.deepEqual(page.window.__TUNED_MODELS__, {}, 'the latest reply wins');
  assert.deepEqual(page.toasts, [], 'an error is only logged');
  assert.equal(warnings.length, 1);
  page.onTunedModels('not json');
  assert.deepEqual(page.toasts, []);
});

test('confirm_before_tuning_when_models_loaded: loaded models or running requests need a confirmation', async () => {
  const page = tuningPage('zh-Hans');
  assert.equal(page.tuningNeedsConfirm({}, null), false);
  assert.equal(page.tuningNeedsConfirm({ a: { id: 'a', loaded: false } }, { activity: { active_requests: 0, queued_requests: 0 } }), false);
  assert.equal(page.tuningNeedsConfirm({ a: { id: 'a', loaded: true } }, null), true);
  assert.equal(page.tuningNeedsConfirm([{ id: 'a', loaded: true }], null), true);
  assert.equal(page.tuningNeedsConfirm({}, { activity: { active_requests: 1 } }), true);
  assert.equal(page.tuningNeedsConfirm({}, { activity: { queued_requests: 2 } }), true);

  page.document.getElementById('profile-model-select').value = 'm';
  page.window.__LOCAL_MODELS__ = { m: { id: 'm', loaded: true } };
  page.confirmAnswer = false;
  await page.runSchedulerProfileGeneration();
  assert.equal(page.confirmations.length, 1);
  assert.equal(page.confirmations[0].message, page.I18N['zh-Hans'].tuning_confirm);
  assert.ok(page.I18N['zh-Hans'].tuning_confirm.includes('后端'));
  assert.ok(!page.messages.some(message => message.name === 'generateSchedulerProfile'), 'cancel sends nothing');
  assert.equal(page.document.getElementById('profile-generate-btn').disabled, false);

  page.confirmAnswer = true;
  await page.runSchedulerProfileGeneration();
  assert.equal(page.messages.filter(message => message.name === 'generateSchedulerProfile').length, 1);
});

test('reopen_requests_status: entering the tab asks for the status and the tuned list; a running status shows progress', () => {
  const tab = source => {
    const start = html.indexOf(source);
    return html.slice(start, html.indexOf('\n  }\n', start));
  };
  assert.match(tab('  function switchToTab('), /models-scheduler[\s\S]*enterTuningTab\(\)/);
  assert.match(tab('  function enterTuningTab('), /refreshSchedulerProfileStatus\(\);[\s\S]*requestTunedModels\(\);/);
  const init = html.slice(html.indexOf('(function initDashboardAppShell()'));
  assert.match(init, /refreshSchedulerProfileStatus\(\);/, 'opening the Dashboard asks once');

  const page = tuningPage('zh-Hans');
  vm.runInContext(extract('enterTuningTab', 'switchToTab') + extract('requestTunedModels', 'onTunedModels'), page);
  page.enterTuningTab();
  assert.deepEqual(page.messages.map(message => message.name), ['refreshSchedulerProfileStatus', 'listTunedModels']);

  vm.runInContext(extract('onSchedulerProfileStatus', 'renderSchedulerProfileStatus'), page);
  page.onSchedulerProfileStatus(JSON.stringify({
    state: 'running', model: 'org/Model', completed_jobs: 21, total_jobs: 44, elapsed_seconds: 1140,
    estimated_remaining_seconds: 1440, current_job: 'job-22 prefill', log_tail: 'a',
  }));
  const el = id => page.document.getElementById(id);
  assert.equal(el('profile-run-card').style.display, '');
  assert.equal(el('profile-form').disabled, true, 'the form is locked while tuning runs');
  assert.equal(el('profile-progress').style.display, '');
  assert.equal(el('profile-progress').bar.style.width.slice(0, 4), '47.7');
  assert.ok(el('profile-progress-details').textContent.includes('21 / 44'));
  assert.ok(el('profile-progress-details').textContent.includes('job-22 prefill'));
  assert.equal(el('profile-run-title').textContent, '正在调优 org/Model');
  assert.equal(el('profile-cancel-btn').style.display, '');
  assert.equal(el('profile-cancel-btn').textContent, '停止调优');
  assert.equal(page.timers.set.at(-1), 2000, 'a status found on reopen keeps being polled');
  page.onSchedulerProfileStatus(JSON.stringify({ state: 'cancelling', model: 'org/Model' }));
  assert.equal(el('profile-cancel-btn').disabled, true);
  assert.equal(el('profile-cancel-btn').textContent, '正在停止…');
});

test('a language switch re-renders the tuning card and the stop button text is owned by the card', () => {
  assert.match(extract('setLanguage', 'selectInterfaceLanguage'), /renderSchedulerProfileStatus\(schedulerProfileLastStatus\);/);
  assert.doesNotMatch(html.match(/<button[^>]*id="profile-cancel-btn"[^>]*>/)[0], /data-i18n/);
});

test('the tuned-date formatter is cached and unchanged menu text is not rewritten', () => {
  const refresh = extract('refreshProfileModelSuffixes', 'tuningNeedsConfirm');
  assert.match(refresh, /option\.textContent !== text/);
  assert.match(extract('tunedModelSuffix', 'profileOptionText'), /tunedDateFormatter\(/);
  assert.equal((html.match(/new Intl\.DateTimeFormat/g) || []).length >= 1, true);
});

test('a finished tuning shows the result once, refreshes the tuned list and does not toast again on reopen', () => {
  const page = tuningPage('en');
  vm.runInContext(extract('onSchedulerProfileStatus', 'renderSchedulerProfileStatus')
    + extract('closeSchedulerProfileResult', 'copySchedulerProfilePath'), page);
  const el = id => page.document.getElementById(id);
  const done = { state: 'succeeded', model: 'org/Model', stored_runtime_profile: '/s/p.json', runtime_profile: '/r/p.json' };
  page.onSchedulerProfileStatus(JSON.stringify(done));
  assert.equal(page.toasts.length, 0, 'a status nobody was waiting for is not announced');
  assert.equal(el('profile-run-title').textContent, 'Tuning finished for org/Model');
  assert.equal(el('profile-result').style.display, '');
  assert.equal(el('profile-result-stored').textContent, '/s/p.json');
  assert.equal(el('profile-form').disabled, false);
  assert.equal(page.messages.filter(message => message.name === 'listTunedModels').length, 0);

  page.onSchedulerProfileStatus(JSON.stringify({ state: 'running', model: 'org/Model' }));
  page.onSchedulerProfileStatus(JSON.stringify(done));
  assert.equal(page.toasts.length, 1);
  assert.equal(page.toasts[0].kind, 'success');
  assert.equal(page.messages.filter(message => message.name === 'listTunedModels').length, 1);
  page.onSchedulerProfileStatus(JSON.stringify(done));
  assert.equal(page.toasts.length, 1);

  page.closeSchedulerProfileResult();
  assert.equal(el('profile-run-card').style.display, 'none');
  page.onSchedulerProfileStatus(JSON.stringify(done));
  assert.equal(el('profile-run-card').style.display, 'none', 'a closed result stays closed');
  page.onSchedulerProfileStatus(JSON.stringify({ state: 'idle' }));
  assert.equal(el('profile-run-card').style.display, 'none');
});

test('log_collapsed_by_default: the log sits in a closed details and keeps the last 18 lines', () => {
  assert.match(html, /<details[^>]*id="profile-log-details"(?![^>]*\sopen)[^>]*>\s*<summary[^>]*data-i18n="tuning_show_log"[\s\S]*?<pre id="profile-log-tail"/);
  const page = tuningPage('en');
  const lines = Array.from({ length: 30 }, (_, index) => 'line ' + (index + 1));
  page.renderSchedulerProfileStatus({ state: 'running', model: 'm', log_tail: lines.join('\n') });
  const log = page.document.getElementById('profile-log-tail');
  assert.equal(log.textContent.split('\n').length, 18);
  assert.ok(log.textContent.startsWith('line 13') && log.textContent.endsWith('line 30'));
  assert.equal(log.scrollTop, log.scrollHeight, 'the log follows the newest line');
  assert.equal(page.document.getElementById('profile-log-details').style.display, '');
  page.renderSchedulerProfileStatus({ state: 'failed', model: 'm', error: 'raw backend text' });
  assert.equal(page.document.getElementById('profile-log-details').style.display, 'none');
  assert.ok(!page.document.getElementById('profile-status').textContent.includes('raw backend text'));
});

test('the tab uses radio cards instead of selects', () => {
  const panel = html.slice(html.indexOf('id="tab-models-scheduler"'), html.indexOf('id="tab-models-download"'));
  assert.ok(!panel.includes('profile-selection-select') && !panel.includes('profile-level-select'));
  const group = id => panel.slice(panel.indexOf(`id="${id}"`), panel.indexOf('</div>', panel.indexOf(`id="${id}"`)));
  const selection = group('profile-selection-options');
  assert.match(selection, /role="radiogroup"/);
  assert.deepEqual([...selection.matchAll(/name="profile-selection" value="([^"]+)"/g)].map(m => m[1]),
    ['agent-long-prompt', 'balanced']);
  assert.match(selection, /value="agent-long-prompt"[^>]*\schecked/);
  const level = group('profile-level-options');
  assert.deepEqual([...level.matchAll(/name="profile-level" value="([^"]+)"/g)].map(m => m[1]),
    ['quick', 'standard', 'full']);
  assert.match(level, /value="standard"[^>]*\schecked/);
  assert.ok(level.includes('tuning_recommended'));
  for (const id of ['profile-model-select', 'profile-generate-btn', 'profile-cancel-btn', 'profile-level-hint',
    'profile-status', 'profile-progress', 'profile-progress-details', 'profile-result', 'profile-result-title',
    'profile-result-stored', 'profile-result-runtime', 'profile-log-tail', 'profile-generation-card']) {
    assert.ok(panel.includes(`id="${id}"`), id);
  }
  assert.ok(panel.includes('class="profile-progress-bar"'));
  assert.ok(panel.includes('data-i18n="tuning_backend_stops"'), 'the warning is always visible');
  assert.ok(!/\sstyle="[^"]*(?<![-\w])var\(--(?!ds-)/.test(panel), 'only --ds-* variables');
});

test('tuning_copy_in_five_languages: new keys exist, are translated and say tuning', () => {
  const I18N = translations();
  const keys = ['tuning_title', 'tuning_backend_stops', 'tuning_goal', 'tuning_goal_agent_desc',
    'tuning_goal_balanced_desc', 'tuning_recommended', 'tuning_start', 'tuning_stop', 'tuning_stopping',
    'tuning_running_title', 'tuning_done_title', 'tuning_tuned_on', 'tuning_settings_changed', 'tuning_confirm',
    'tuning_show_log', 'tuning_close'];
  for (const language of LANGUAGES) {
    for (const key of keys) {
      assert.equal(typeof I18N[language][key], 'string', `${language}.${key}`);
      assert.ok(I18N[language][key].trim(), `${language}.${key}`);
      if (language !== 'en') assert.notEqual(I18N[language][key], I18N.en[key], `${language}.${key} is translated`);
    }
    assert.ok(I18N[language].tuning_running_title.includes('{name}'), language);
    assert.ok(I18N[language].tuning_done_title.includes('{name}'), language);
    assert.ok(I18N[language].tuning_tuned_on.includes('{date}'), language);
  }
  assert.equal(I18N.en.tuning_start, 'Start tuning');
  assert.equal(I18N.en.tuning_stop, 'Stop tuning');
  assert.equal(I18N['zh-Hans'].tuning_start, '开始调优');
  assert.equal(I18N['zh-Hans'].tuning_stop, '停止调优');
  assert.equal(I18N['zh-Hans'].tuning_goal_agent_desc, '更看重长输入（4K token 以上）的首字延迟');
  assert.equal(I18N['zh-Hans'].tuning_goal_balanced_desc, '短对话和长输入同等对待');
  assert.ok(I18N['zh-Hans'].tuning_backend_stops.includes('调优期间后端会停止，所有模型暂时不可用'));
  // The words "profile" and "生成" leave the tab copy; only the stored-path note keeps "profile".
  const used = ['scheduler_profile_help_title', 'scheduler_profile_help_body', 'scheduler_profile_help_flow',
    'scheduler_profile_agent_long_prompt', 'scheduler_profile_balanced', 'scheduler_profile_level',
    'profile_running_hint', 'profile_cancelling', 'profile_cancelled',
    'profile_succeeded', 'profile_result_title', 'profile_failed', 'profile_report_path'];
  for (const language of ['zh-Hans', 'zh-Hant']) {
    for (const key of used) {
      assert.ok(!/profile|生成|生成|產生/i.test(I18N[language][key]), `${language}.${key}: ${I18N[language][key]}`);
    }
  }
  for (const key of used.filter(key => key !== 'scheduler_profile_help_levels')) {
    assert.ok(!/profile/i.test(I18N.en[key]) || key === 'x', `en.${key}: ${I18N.en[key]}`);
  }
  assert.ok(/profile/i.test(I18N['zh-Hans'].profile_store_path), 'the storage path note keeps "profile"');
});

test('tuning_help_uses_fixed_tooltip_and_intro_is_not_repeated', () => {
  // The help sits near the card's left edge; a fixed tooltip is placed inside the window instead of being clipped.
  assert.match(html, /<span class="profile-help profile-help-fixed">\s*<button type="button" class="profile-help-trigger" id="profile-help-trigger"/);
  assert.match(html, /class="profile-help-tooltip profile-help-tooltip-fixed" id="profile-help-tooltip"/);
  // The help already explains what tuning does; only the backend warning stays on the page.
  assert.doesNotMatch(html, /tuning_intro|ds-tuning-intro/);
  assert.match(html, /id="profile-backend-stops" data-i18n="tuning_backend_stops"/);
});
