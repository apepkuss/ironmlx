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
      && dict.warn_scheduler_profile_unusable.includes(dict.scheduler_profile_tab),
      `${language} names the Models page and the tab that regenerates the profile`);
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
    '该模型的 scheduler profile 是在其他硬件上生成的，本次使用默认调度配置。在本机重新生成即可获得适合本机的调度。');
  assert.equal(zh.warn_scheduler_profile_unusable,
    '该模型的 scheduler profile 无法使用（文件损坏或已失效），本次使用默认调度配置。请在「模型 → 调度配置」中重新生成。');
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
  const status = html.slice(html.indexOf('  function onSchedulerProfileStatus('),
    html.indexOf('  function formatProfileDuration(', html.indexOf('  function onSchedulerProfileStatus(')));
  assert.equal((status.match(/schedulerProfileFailureDetail\(status, /g) || []).length, 2,
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
  vm.runInContext(extract('modelReadiness', 'modelReadinessTitle')
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
