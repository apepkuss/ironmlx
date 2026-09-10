const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const html = fs.readFileSync(path.join(__dirname,
  '../../ironmlx-app/Sources/IronMLXAppCore/Resources/dashboard2.html'), 'utf8');
const source = html.slice(html.indexOf('  function exportCurrentLog()'), html.indexOf('  let activeLogTab'));
function page(file, lines, native = true) {
  const messages = [], toasts = [];
  const context = { logFileSelect: { value: file }, logRawLines: lines,
    logOutput: { textContent: 'fallback' }, I18N: { en: {
      log_exported: 'saved', log_export_failed: 'failed',
    } }, currentLang: 'en', showToast: (...args) => toasts.push(args),
    window: native ? { webkit: { messageHandlers: { exportRuntimeLog: {
      postMessage: value => messages.push(JSON.parse(value)),
    } } } } : {},
  };
  vm.createContext(context); vm.runInContext(source, context);
  return { context, messages, toasts };
}
for (const file of ['app', 'server']) {
  test(`${file} export snapshots fetched logs and prevents link navigation`, () => {
    const p = page(file, ['中文', 'line 2']); let prevented = false;
    p.context.exportCurrentLog({ preventDefault() { prevented = true; } });
    assert.ok(prevented);
    assert.deepEqual(p.messages, [{ source: `ironmlx-${file}`, content: '中文\nline 2\n' }]);
  });
}
test('export reports completion and failure but not cancellation', () => {
  const p = page('app', []);
  p.context.onRuntimeLogExported('cancelled');
  p.context.onRuntimeLogExported('busy');
  assert.equal(p.toasts.length, 0);
  p.context.onRuntimeLogExported('exported');
  p.context.onRuntimeLogExported('failed');
  assert.deepEqual(p.toasts, [['saved', 'success'], ['failed', 'warn']]);
  const missing = page('app', [], false);
  missing.context.exportCurrentLog();
  assert.deepEqual(missing.toasts, [['failed', 'warn']]);
});
