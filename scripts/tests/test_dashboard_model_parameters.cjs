const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const html = fs.readFileSync(path.join(__dirname,
  '../../ironmlx-app/Sources/IronMLXAppCore/Resources/dashboard2.html'), 'utf8');
const extract = (start, end) => html.slice(html.indexOf(`  function ${start}(`),
  html.indexOf(`  function ${end}(`));

function dashboard() {
  let payload;
  const modelID = 'mlx-community/Qwen3.5-4B-4bit';
  const context = {
    closeParamsModal() {},
    window: {
      __CURRENT_PARAM_MODEL__: modelID,
      __LOCAL_MODELS__: { [modelID]: { capabilities: { supports_mtp: false } } },
      webkit: { messageHandlers: {
        saveModelParams: { postMessage(value) { payload = JSON.parse(value); } },
      } },
    },
    document: { getElementById(id) {
      return { value: id === 'modal-max-tokens' ? '4096'
        : id === 'modal-max-output-tokens' ? '1024'
        : id === 'modal-context-size' ? '262144' : '', checked: false };
    } },
    I18N: { en: {
      err_settings_persist_failed: html.match(/err_settings_persist_failed: "(IronMLX 无法保存[^"\n]+)"/)[1],
      err_model_parameters_invalid: html.match(/err_model_parameters_invalid: "(模型参数无效[^"\n]+)"/)[1],
    } },
    currentLang: 'en',
    parsePositiveInteger: value => /^\d+$/.test(value) && Number(value) > 0 ? Number(value) : null,
    showToast(message) { throw Error(message); },
    localizeBackendErrorMessage: value => value,
  };
  vm.createContext(context);
  vm.runInContext(extract('saveModelParams', 'onModelParamsSaved'), context);
  vm.runInContext(extract('localizeErrorResult', 'localizeWarningResult'), context);
  return { context, get payload() { return payload; } };
}

test('saving an ordinary model keeps output budget separate from the context limit', () => {
  const page = dashboard();
  vm.runInContext('saveModelParams()', page.context);
  assert.equal(page.payload.mtp_model_id, null);
  assert.equal(page.payload.dflash2_model_id, null);
  assert.equal(page.payload.max_tokens, '4096');
  assert.equal(page.payload.max_output_tokens, '1024');
  assert.ok(!Object.hasOwn(page.payload, 'model_type'), 'readonly display must not save a type override');
});

test('model parameter help explains the three capacity fields in each language', () => {
  const start = html.indexOf('  const I18N =');
  const dictionary = html.slice(start, html.indexOf('\n  };', start) + 5);
  const translations = vm.runInNewContext(`${dictionary}\nI18N`, {});
  const fields = ['context_size', 'max_context_tokens', 'default_max_output_tokens'];
  for (const field of fields) {
    assert.ok(html.includes(`data-i18n-aria-label="${field}_help_label"`));
    assert.ok(html.includes(`data-i18n="${field}_help"`));
    for (const language of ['en', 'zh-Hans', 'zh-Hant', 'ja', 'ko']) {
      assert.ok(translations[language][`${field}_help_label`], `${language} ${field} label`);
      assert.ok(translations[language][`${field}_help`], `${language} ${field} help`);
    }
  }
  assert.match(translations.en.context_size_help, /Read-only/i);
  for (const language of ['en', 'zh-Hans', 'zh-Hant', 'ja', 'ko']) {
    assert.equal(translations[language].default_max_output_tokens, 'MAX OUTPUT TOKENS');
    for (const detail of ['budget', 'priority', 'scope']) {
      assert.ok(translations[language][`default_max_output_tokens_help_${detail}`],
        `${language} output budget ${detail}`);
    }
  }
});

test('model parameters form three base rows plus a hidden reasoning metadata row', () => {
  const rows = [...html.matchAll(/<div class="modal-row model-params-setting-row[^>]*>/g)]
    .filter(row => row.index > html.indexOf('<!-- Model Params Modal -->'));
  const expected = [
    ['modal-alias-input', 'modal-model-type', 'modal-context-size'],
    ['modal-reasoning-capability', 'modal-reasoning-efforts', 'modal-reasoning-default'],
    ['modal-max-tokens', 'modal-max-output-tokens', 'modal-temperature'],
    ['modal-top-p', 'modal-top-k', 'modal-repeat-penalty'],
  ];
  assert.equal(rows.length, expected.length);
  assert.match(rows[1][0], /id="modal-reasoning-metadata-row" hidden/);
  const outputSection = html.indexOf('<div class="model-params-output-section">', rows[0].index);
  assert.ok(outputSection > rows[1].index && outputSection < rows[2].index);
  assert.ok(html.includes('.model-params-output-section {\n    border-top: 0.5px solid var(--border);'));
  for (let index = 0; index < rows.length; index++) {
    const end = index + 1 < rows.length
      ? rows[index + 1].index : html.indexOf('<div id="modal-mtp-section"', rows[index].index);
    const row = html.slice(rows[index].index, end);
    const positions = expected[index].map(id => row.indexOf(`id="${id}"`));
    assert.equal((row.match(/class="modal-field"/g) || []).length, 3);
    assert.ok(positions.every(position => position >= 0));
    assert.ok(positions[0] < positions[1] && positions[1] < positions[2]);
    assert.equal((row.match(/class="profile-help-trigger"/g) || []).length, [1, 1, 2, 0][index]);
  }
});

test('parameter validation identifies the field without suggesting filesystem repair', () => {
  const { context } = dashboard();
  assert.equal(context.localizeErrorResult({ code: 'model_parameters_invalid', field: 'max_tokens' }),
    '模型参数无效：max_tokens，未保存更改。');
  assert.equal(context.localizeErrorResult({ code: 'settings_persist_failed' }),
    'IronMLX 无法保存应用设置，请检查文件权限和可用磁盘空间。');
});

test('decision runtime status uses persistent metrics without increasing poll frequency', () => {
  const start = html.indexOf('  const I18N =');
  const dictionary = html.slice(start, html.indexOf('\n  };', start) + 5);
  const translations = vm.runInNewContext(`${dictionary}\nI18N`, {});
  const keys = [
    'runtime_state_recent',
    'runtime_decision_performance',
    'runtime_decision_completed',
    'runtime_decision_errors',
    'runtime_decision_latency',
    'runtime_decision_input_rate',
    'runtime_decision_question_rate',
    'runtime_decision_performance_window',
    'runtime_decision_performance_empty',
  ];
  for (const language of ['en', 'zh-Hans', 'zh-Hant', 'ja', 'ko']) {
    for (const key of keys) {
      assert.ok(translations[language][key], `${language} ${key}`);
    }
  }

  const stateFunction = extract('runtimeOperationalState', 'renderRuntimeModels');
  const stateContext = { Date };
  vm.createContext(stateContext);
  vm.runInContext(
    `const DECISION_RECENT_ACTIVITY_MS = 2500;\n${stateFunction}`,
    stateContext
  );
  assert.equal(stateContext.runtimeOperationalState({
    runtime_kind: 'decision',
    active_requests: 1,
    decision_metrics: { last_request_unix_ms: 999_900 },
  }, 1_000_000), 'busy');
  assert.equal(stateContext.runtimeOperationalState({
    runtime_kind: 'decision',
    decision_metrics: { last_request_unix_ms: 999_000 },
  }, 1_000_000), 'recent');
  assert.equal(stateContext.runtimeOperationalState({
    runtime_kind: 'decision',
    decision_metrics: { last_request_unix_ms: 997_000 },
  }, 1_000_000), 'idle');
  assert.equal(stateContext.runtimeOperationalState({
    runtime_kind: 'causal',
    decision_metrics: { last_request_unix_ms: 999_000 },
  }, 1_000_000), 'idle');

  const render = extract('renderRuntimeModels', 'onApiFetchResult');
  assert.match(render, /model\.runtime_kind === 'decision'/);
  assert.match(render, /model\.decision_metrics/);
  assert.match(stateFunction, /last_request_unix_ms/);
  assert.match(render, /runtime-state-recent/);
  assert.match(render, /recent_completed_requests/);
  assert.match(render, /input_tokens_per_second/);
  assert.match(render, /questions_per_second/);
  assert.match(html, /const GPU_ACTIVE_POLL_MS = 500;/);
  assert.match(html, /const GPU_IDLE_POLL_MS = 1000;/);
});
