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
for (const name of ['refreshVoices', 'renderVoiceProfiles', 'openVoiceEditor', 'saveVoiceProfile',
  'toggleVoice', 'deleteVoice', 'apiPatch', 'openVoiceReferencePicker', 'normalizeVoiceLanguage',
  'localizeVoiceFileStatus', 'readVoiceReference', 'voiceLanguageLabel', 'formatVoicePlayerTime',
  'voicePlayerIcon', 'releaseVoicePlayers', 'toggleVoicePreview', 'applyVoicePreviewResponse',
  'syncVoicePreview', 'syncVoicePreviewState', 'seekVoicePreview', 'toggleVoicePreviewMute',
  'positionVoiceEditor', 'closeVoiceEditor']) assert.ok(source(name));
assert.match(html, /data-page="voices"/);
assert.match(html, /<select id="voice-language">/);
for (const value of ['', 'zh', 'en', 'ja', 'es', 'ar']) {
  assert.match(html, new RegExp('<option value="' + value + '"'));
}
assert.doesNotMatch(html, /<input id="voice-language"/);
assert.match(html, /onclick="openVoiceReferencePicker\(\)" data-i18n="voice_choose_file"/);
assert.match(html, /id="voice-file-status"[^>]+data-i18n="voice_no_file_selected"/);
assert.match(source('refreshVoices'), /\/admin\/api\/audio\/voices/);
assert.match(source('saveVoiceProfile'), /\/v1\/audio\/voices/);
assert.match(source('toggleVoicePreview'), /\/preview/);
assert.match(source('positionVoiceEditor'), /card\.after\(editor\)/);
assert.match(source('positionVoiceEditor'), /list\.before\(editor\)/);
assert.match(source('closeVoiceEditor'), /list\.after\(editor\)/);
assert.doesNotMatch(html, /id="voice-preview-player"/);
for (const name of ['modelReadiness', 'isModelLoadable', 'modelReadinessTitle', 'modelIntegrityStatus',
  'formatIntegrityTime', 'modelStatusPresentation', 'onModelIntegrityStatus', 'formatVersionBytes', 'quantDisplayLabel', 'renderQuantBadge',
  'openTTSDetails', 'closeTTSDetails', 'openParamsModal', 'renderModelLoadActions', 'escapeAttr',
  'openVoiceReferencePicker', 'normalizeVoiceLanguage', 'localizeVoiceFileStatus', 'readVoiceReference',
  'voiceLanguageLabel', 'formatVoicePlayerTime', 'voicePlayerIcon', 'releaseVoicePlayers',
  'renderVoiceProfiles', 'deleteVoice']) vm.runInContext(source(name), context);
let scans = 0;
const elements = new Map();
function element() {
  const classes = new Set();
  return { focus() {}, click() { this.clicked = true; }, clicked: false, textContent: '', dataset: {}, children: [], classList: { add: x => classes.add(x), remove: x => classes.delete(x), contains: x => classes.has(x) },
    replaceChildren() { this.children = []; }, appendChild(child) { this.children.push(child); } };
}
context.document = { getElementById(id) { if (!elements.has(id)) elements.set(id, element()); return elements.get(id); }, createElement: element, querySelectorAll() { return []; } };
context.window = { __LOCAL_MODELS__: {}, __MODEL_INTEGRITY_STATUS__: {}, webkit: { messageHandlers: { scanLocalModels: { postMessage() { scans++; } } } } };
context.HTML_LANG_MAP = { en: 'en', 'zh-Hans': 'zh-CN', 'zh-Hant': 'zh-Hant', ja: 'ja', ko: 'ko' };
context.pendingVoicePreviewIds = new Set();
context.editingVoiceId = null;
context.updateModelIntegrityRow = () => {};
context.showToast = () => {};
context.applyI18n = () => {};
context.currentLang = 'zh-Hans';
for (const language of ['en', 'zh-Hans', 'zh-Hant', 'ja', 'ko']) {
  for (const key of ['voice_id_help_label', 'voice_id_help_title', 'voice_id_help',
    'voice_name_help_label', 'voice_name_help_title', 'voice_name_help',
    'voice_language_help_label', 'voice_language_help_title', 'voice_language_help',
    'voice_reference_help_label', 'voice_reference_help_title', 'voice_reference_help',
    'voice_save_only', 'voice_save_and_enable',
    'voice_choose_file', 'voice_no_file_selected', 'voice_player_label', 'voice_play',
    'voice_pause', 'voice_seek', 'voice_mute', 'voice_unmute']) {
    assert.ok(context.I18N?.[language]?.[key] || vm.runInContext('I18N[' + JSON.stringify(language) + '][' + JSON.stringify(key) + ']', context));
  }
}
const zhVoiceStrings = vm.runInContext('I18N["zh-Hans"]', context);
assert.equal(zhVoiceStrings.voice_name, '声音名称');
assert.doesNotMatch(zhVoiceStrings.voice_name_help, /API/);
assert.match(zhVoiceStrings.voice_id_help, /API/);
assert.match(html, /id="voice-id-help-tooltip" role="tooltip"/);
assert.match(html, /id="voice-name-help-tooltip" role="tooltip"/);
assert.doesNotMatch(html, /id="voice-enabled"/);
assert.match(html, /id="voice-save-only"[^>]+saveVoiceProfile\(false\)/);
assert.match(html, /id="voice-save-enable"[^>]+saveVoiceProfile\(true\)/);
assert.match(html, /\.voice-editor-actions\s*>\s*\[hidden\]\s*\{\s*display\s*:\s*none/);
assert.doesNotMatch(source('saveVoiceProfile'), /getElementById\('voice-enabled'\)/);
assert.match(source('saveVoiceProfile'), /body\.enabled = enableAfterSave === true/);
assert.doesNotMatch(source('deleteVoice'), /\bconfirm\s*\(/);
assert.match(source('deleteVoice'), /showConfirmDialog\(/);
assert.equal(context.normalizeVoiceLanguage('zh-CN'), 'zh');
assert.equal(context.normalizeVoiceLanguage('EN_us'), 'en');
assert.equal(context.normalizeVoiceLanguage('unsupported'), '');
context.openVoiceReferencePicker();
assert.equal(elements.get('voice-file').clicked, true);
context.FileReader = class {
  readAsDataURL() {
    this.result = 'data:audio/wav;base64,UklGRg==';
    this.onload();
  }
};
context.readVoiceReference({ files: [{ name: 'reference.wav', size: 8 }], value: 'reference.wav' });
assert.equal(elements.get('voice-file-status').dataset.state, 'selected');
assert.equal(elements.get('voice-file-status').textContent, 'reference.wav');
const deletedPaths = [];
let confirmOptions = null;
context.pendingVoiceDeleteId = null;
context.t = (_key, fallback) => fallback;
context.apiDelete = path => deletedPaths.push(path);
context.showConfirmDialog = (_message, options) => {
  confirmOptions = options;
  return Promise.resolve(false);
};
context.deleteVoice('speaker a');
await Promise.resolve();
assert.deepEqual(deletedPaths, []);
assert.equal(context.pendingVoiceDeleteId, null);
assert.equal(confirmOptions.danger, true);
assert.equal(confirmOptions.confirmText, '删除');
context.showConfirmDialog = () => Promise.resolve(true);
context.deleteVoice('speaker a');
await Promise.resolve();
assert.deepEqual(deletedPaths, ['/v1/audio/voices/speaker%20a']);
assert.equal(context.pendingVoiceDeleteId, 'speaker a');
assert.equal(context.formatVoicePlayerTime(2.1, true), '0:02');
assert.equal(context.formatVoicePlayerTime(65.9, false), '1:05');
context.renderVoiceProfiles({ data: [{ id: 'speaker_a', name: '测试声音', language: 'zh', enabled: true,
  reference_format: 'wav', duration_ms: 2100, sample_rate: 48000 }] });
const voiceMarkup = elements.get('voices-list').innerHTML;
assert.match(voiceMarkup, /class="voice-player"/);
assert.match(voiceMarkup, /aria-label="测试声音 的参考音频播放器"/);
assert.match(voiceMarkup, /测试声音<span class="voice-card-title-id">\(ID: speaker_a\)<\/span>/);
assert.doesNotMatch(voiceMarkup, /class="badge">启用/);
assert.match(voiceMarkup, /voice-player-current">0:00<\/span> \/ <span class="voice-player-duration">0:02/);
assert.match(voiceMarkup, /中文 · WAV · 2\.1s · 48,000 Hz/);
assert.doesNotMatch(voiceMarkup, /试听参考音频/);
context.renderVoiceProfiles({ data: [{ id: 'speaker_b', name: '停用声音', language: 'zh', enabled: false,
  reference_format: 'wav', duration_ms: 2100, sample_rate: 48000 }] });
const disabledVoiceMarkup = elements.get('voices-list').innerHTML;
assert.match(disabledVoiceMarkup, /class="voice-card"/);
assert.doesNotMatch(disabledVoiceMarkup, /is-disabled/);
assert.match(disabledVoiceMarkup, />启用<\/button>/);
assert.doesNotMatch(html, /\.voice-card\.is-disabled/);
const fp16Badge = context.renderQuantBadge({ quantization: { kind: 'dense', label: 'FP16', dtype: 'float16' }, readiness: { status: 'ready' } });
assert.match(fp16Badge, />FP16<\/span>/);
assert.match(fp16Badge, /title="未量化; dtype=float16"/);
const tts = { id: 'org/tts', type: 'tts', readiness: { status: 'unsupported', reason_code: 'unsupported_model_type' },
  integrity: { state: 'verified', verified_at: '2026-09-09T00:00:00Z' }, download_info: {
    files: [{ path: '<img src=x onerror=alert(1)>.safetensors', size: 1024 }], external_resources: ['funasr/campplus'] } };
assert.equal(context.isModelLoadable(tts), false);
assert.equal(context.isModelLoadable({ type: 'tts', readiness: { status: 'ready' } }), true);
assert.equal(context.isModelLoadable({ type: 'tts' }), false);
assert.equal(context.modelStatusPresentation(tts, false).text, '不支持');
const ready = { ...tts, readiness: { status: 'ready' } };
assert.equal(context.modelStatusPresentation(ready, false).dot, 'model-ready');
assert.match(context.renderModelLoadActions(ready, 'org/tts', 'org/tts', '', 'Load', false), /toggleModelLoad/);
assert.doesNotMatch(context.renderModelLoadActions(ready, 'org/tts', 'org/tts', '', 'Load', false), /disabled/);
const missing = { ...tts, readiness: { status: 'incomplete', reason_code: 'audio_resources_missing' } };
assert.match(context.renderModelLoadActions(missing, 'org/tts', 'org/tts', '', 'Load', false), /startDownload/);
assert.match(context.renderModelLoadActions(tts, 'org/tts', 'org/tts', '', 'Load', false), /disabled/);
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
