import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

const html = fs.readFileSync(new URL('../../ironmlx-app/Sources/IronMLXAppCore/Resources/dashboard2.html', import.meta.url), 'utf8');
for (const match of html.matchAll(/<script\b[^>]*>([\s\S]*?)<\/script>/g)) new vm.Script(match[1]);
const i18n = html.slice(html.indexOf('  const I18N ='), html.indexOf('  const HTML_LANG_MAP'));
// Only the dictionary declaration is needed; avoid running page initialization.
const dictionary = i18n.slice(0, i18n.indexOf('\n  };') + 5);
const context = vm.createContext({ console });
vm.runInContext(dictionary, context);
const source = name => {
  const match = html.match(new RegExp('  function ' + name + '\\([^]*?\\n  }'));
  assert.ok(match, name);
  return match[0];
};
for (const name of ['modelReadiness', 'isModelLoadable', 'modelReadinessTitle', 'modelIntegrityStatus',
  'formatIntegrityTime', 'modelStatusPresentation', 'onModelIntegrityStatus', 'formatVersionBytes',
  'openTTSDetails', 'closeTTSDetails', 'openParamsModal']) vm.runInContext(source(name), context);
let scans = 0;
const elements = new Map();
function element() {
  const classes = new Set();
  return { focus() {}, textContent: '', children: [], classList: { add: x => classes.add(x), remove: x => classes.delete(x), contains: x => classes.has(x) },
    replaceChildren() { this.children = []; }, appendChild(child) { this.children.push(child); } };
}
context.document = { getElementById(id) { if (!elements.has(id)) elements.set(id, element()); return elements.get(id); }, createElement: element };
context.window = { __LOCAL_MODELS__: {}, __MODEL_INTEGRITY_STATUS__: {}, webkit: { messageHandlers: { scanLocalModels: { postMessage() { scans++; } } } } };
context.updateModelIntegrityRow = () => {};
context.showToast = () => {};
context.applyI18n = () => {};
context.currentLang = 'zh-Hans';
const tts = { id: 'org/tts', type: 'tts', readiness: { status: 'unsupported', reason_code: 'unsupported_model_type' },
  integrity: { state: 'verified', verified_at: '2026-09-09T00:00:00Z' }, download_info: {
    files: [{ path: '<img src=x onerror=alert(1)>.safetensors', size: 1024 }], external_resources: ['funasr/campplus'] } };
assert.equal(context.isModelLoadable(tts), false);
assert.equal(context.isModelLoadable({ type: 'tts', readiness: { status: 'ready' } }), false);
assert.equal(context.modelStatusPresentation(tts, false).text, '文件已验证 · 暂不支持 TTS 推理');
for (const language of ['en', 'zh-Hans', 'zh-Hant', 'ja', 'ko']) {
  context.currentLang = language;
  assert.ok(context.modelStatusPresentation(tts, false).text);
}
context.currentLang = 'zh-Hans';
assert.equal(context.modelStatusPresentation({ ...tts, integrity: { state: 'corrupt' } }, false).dot, 'model-corrupt');
assert.equal(context.modelStatusPresentation({ ...tts, integrity: { state: 'verifying' } }, false).dot, 'model-busy');
assert.equal(context.modelStatusPresentation({ ...tts, type: 'embedding' }, false).dot, 'model-unsupported');
const llm = { id: 'org/llm', type: 'llm', readiness: { status: 'ready' }, integrity: tts.integrity };
assert.equal(context.isModelLoadable(llm), true);
assert.equal(context.modelStatusPresentation(llm, false).dot, 'model-ready');
context.window.__LOCAL_MODELS__[tts.id] = tts;
context.onModelIntegrityStatus(JSON.stringify({ repo_id: tts.id, state: 'corrupt' }));
context.onModelIntegrityStatus(JSON.stringify({ repo_id: tts.id, state: 'verified' }));
assert.equal(context.isModelLoadable(tts), false);
assert.equal(tts.readiness.status, 'unverified');
assert.equal(scans, 1);
context.openParamsModal(tts.id);
assert.ok(elements.get('tts-details-modal').classList.contains('open'));
assert.ok(!elements.has('modal-max-tokens'), 'TTS must not enter the LLM parameter form');
assert.equal(elements.get('tts-details-files').children[0].textContent, '<img src=x onerror=alert(1)>.safetensors · 1.00 KiB');
assert.equal(elements.get('tts-details-dependencies').children[0].textContent, 'funasr/campplus');
context.closeTTSDetails();
assert.ok(!elements.get('tts-details-modal').classList.contains('open'));
context.openTTSDetails({ id: 'empty', type: 'tts' });
assert.equal(elements.get('tts-details-files').children[0].textContent, '暂无文件信息。');
console.log('Dashboard TTS behavior and full script syntax checks passed.');
