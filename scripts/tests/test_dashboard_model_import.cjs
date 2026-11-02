const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const html = fs.readFileSync(path.join(__dirname,
  '../../ironmlx-app/Sources/IronMLXAppCore/Resources/dashboard2.html'), 'utf8');
function slice(start, end) {
  return html.slice(html.indexOf(start), html.indexOf(end, html.indexOf(start)));
}

function page(language) {
  const elements = new Map();
  const confirmations = [];
  const recoveries = [];
  const toasts = [];
  const element = id => {
    if (!elements.has(id)) {
      const classes = new Set();
      elements.set(id, { textContent: '', hidden: false, disabled: false,
        attributes: {}, classList: { add: name => classes.add(name), delete: name => classes.delete(name),
          remove: name => classes.delete(name), contains: name => classes.has(name) },
        setAttribute(name, value) { this.attributes[name] = value; },
        removeAttribute(name) { delete this.attributes[name]; } });
    }
    return elements.get(id);
  };
  const context = { currentLang: language, document: { getElementById: element },
    window: { webkit: { messageHandlers: {
      confirmModelImport: { postMessage: value => confirmations.push(value) },
      cancelModelImport: { postMessage() {} },
      recoverModelImports: { postMessage: value => recoveries.push(value) },
    } } }, formatVersionBytes: String, showToast: (...args) => toasts.push(args) };
  vm.createContext(context);
  vm.runInContext(slice('  const I18N =', '  const LANG_MAP =')
    + slice('  function t(key, fallback)', '  function setLanguage(')
    + slice('  let modelImportBusy =', '  function deleteSingleModel('), context);
  const preview = downloadOnly => context.onModelImportPreview(JSON.stringify({
    name: 'Qwen3.8-Flash-Next-4bit', source: '/external/model', destination: '/models/standalone/model',
    modelType: 'qwen4_exp', totalBytes: 112000000000, fileCount: 30, downloadOnly,
  }));
  return { context, element, confirmations, preview, recoveries, toasts };
}

for (const language of ['en', 'zh-Hans', 'zh-Hant', 'ja', 'ko']) {
  test(`file-only import requires a visible explicit choice in ${language}`, () => {
    const p = page(language);
    p.preview(true);
    assert.equal(p.element('model-import-file-only-note').hidden, false);
    assert.equal(p.element('model-import-confirm').textContent, p.context.t('model_import_anyway'));
    assert.equal(p.element('model-import-confirm').attributes['data-i18n'], 'model_import_anyway');
    assert.ok(p.context.t('model_import_file_only_note'));
    assert.ok(p.context.t('model_file_only_status'));
    assert.equal(p.confirmations.length, 0);
    p.context.confirmModelImport();
    p.context.confirmModelImport();
    assert.deepEqual(p.confirmations, [true]);
    assert.equal(p.element('model-import-button').disabled, true);
  });
}

test('a supported preview clears the previous file-only choice and uses the ordinary import action', () => {
  const p = page('zh-Hans');
  p.preview(true);
  p.context.closeModelImport();
  p.preview(false);
  assert.equal(p.element('model-import-file-only-note').hidden, true);
  assert.equal(p.element('model-import-confirm').textContent, p.context.t('model_import_copy'));
  assert.equal(p.element('model-import-confirm').attributes['data-i18n'], 'model_import_copy');
  p.context.confirmModelImport();
  assert.deepEqual(p.confirmations, [false]);
});

test('cancelling a file-only preview does not confirm an import', () => {
  const p = page('zh-Hans');
  p.preview(true);
  p.context.closeModelImport();
  assert.deepEqual(p.confirmations, []);
  assert.equal(p.element('model-import-modal').classList.contains('open'), false);
});

test('interrupted import recovery shows verification progress and completion', () => {
  const p = page('zh-Hans');
  p.context.onModelImportRecoveryStarted();
  assert.equal(p.element('model-import-button').disabled, true);
  assert.match(p.element('model-import-status-text').textContent, /恢复中断/);
  p.context.onModelImportProgress(JSON.stringify({ phase: 'verifying', completedBytes: 60,
    totalBytes: 100, currentFile: 'model.safetensors' }));
  assert.match(p.element('model-import-status-text').textContent, /校验.*60%.*model.safetensors/);
  p.context.onModelImportRecoveryFinished(JSON.stringify({ completed: ['standalone/model'], errors: [] }));
  assert.equal(p.element('model-import-button').disabled, false);
  assert.equal(p.element('model-import-status').classList.contains('visible'), false);
  assert.deepEqual(p.toasts, [['已恢复 1 个模型的导入。', 'success']]);
});

test('a recovery failure keeps an actionable visible error and retries once', () => {
  const p = page('zh-Hans');
  p.context.onModelImportRecoveryStarted();
  p.context.onModelImportRecoveryFinished(JSON.stringify({ completed: [], errors: ['Checksum mismatch: model.safetensors'] }));
  assert.equal(p.element('model-import-status').classList.contains('visible'), true);
  assert.match(p.element('model-import-status-text').textContent, /文件已保留.*Checksum mismatch/);
  assert.equal(p.element('model-import-retry').hidden, false);
  assert.equal(p.element('model-import-progress').hidden, true);
  assert.equal(p.element('model-import-cancel').hidden, true);
  p.context.retryModelImportRecovery();
  p.context.retryModelImportRecovery();
  assert.equal(p.recoveries.length, 1);
  assert.equal(p.element('model-import-retry').hidden, true);
  assert.equal(p.element('model-import-progress').hidden, false);
  assert.equal(p.element('model-import-cancel').hidden, false);
});
