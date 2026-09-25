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
  'formatIntegrityTime', 'modelStatusPresentation', 'onModelIntegrityStatus', 'formatVersionBytes',
  'normalizedWeightDtype', 'weightDtypeLabel', 'quantDisplayLabel', 'hideQuantizationTooltip', 'showQuantizationTooltip', 'renderQuantBadge',
  'modelTypeLabel', 'renderModelType', 'openTTSDetails', 'closeTTSDetails',
  'ttsExecutionControls', 'renderTTSExecutionSettings',
  'openDecisionDetails', 'saveDecisionParams', 'closeDecisionDetails', 'openParamsModal', 'renderModelLoadActions', 'escapeAttr',
  'openVoiceReferencePicker', 'normalizeVoiceLanguage', 'localizeVoiceFileStatus', 'readVoiceReference',
  'voiceLanguageLabel', 'formatVoicePlayerTime', 'voicePlayerIcon', 'releaseVoicePlayers',
  'renderVoiceProfiles', 'deleteVoice']) vm.runInContext(source(name), context);
let scans = 0;
const elements = new Map();
function element() {
  const classes = new Set();
  const attributes = new Map();
  return { focus() {}, click() { this.clicked = true; }, closest() { return element(); }, clicked: false, hidden: true, textContent: '', dataset: {}, style: {}, children: [], classList: { add: x => classes.add(x), remove: x => classes.delete(x), contains: x => classes.has(x) },
    setAttribute(name, value) { attributes.set(name, String(value)); }, removeAttribute(name) { attributes.delete(name); }, getAttribute(name) { return attributes.get(name); },
    getBoundingClientRect() { return { left: 100, right: 160, top: 100, bottom: 124, width: 60, height: 24 }; },
    replaceChildren() { this.children = []; }, appendChild(child) { this.children.push(child); } };
}
context.document = { getElementById(id) { if (!elements.has(id)) { const value = element(); value.id = id; elements.set(id, value); } return elements.get(id); }, createElement: element, querySelector(selector) { if (!elements.has(selector)) elements.set(selector, element()); return elements.get(selector); }, querySelectorAll() { return []; } };
context.window = { innerWidth: 1000, innerHeight: 800, __LOCAL_MODELS__: {}, __MODEL_INTEGRITY_STATUS__: {}, webkit: { messageHandlers: { scanLocalModels: { postMessage() { scans++; } } } } };
context.HTML_LANG_MAP = { en: 'en', 'zh-Hans': 'zh-CN', 'zh-Hant': 'zh-Hant', ja: 'ja', ko: 'ko' };
context.pendingVoicePreviewIds = new Set();
context.editingVoiceId = null;
context.updateModelIntegrityRow = () => {};
context.showToast = () => {};
context.applyI18n = () => {};
context.apiFetch = () => {};
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
assert.match(fp16Badge, /data-quant-tooltip="未量化；dtype=fp16"/);
assert.match(fp16Badge, /tabindex="0"/);
assert.match(fp16Badge, /onmouseenter="showQuantizationTooltip\(this\)"/);
assert.doesNotMatch(fp16Badge, /\stitle=/);
assert.equal(context.quantDisplayLabel({ kind: 'dense', label: 'Dense', dtype: 'bfloat16' }), 'BF16');
assert.equal(context.quantDisplayLabel({ kind: 'dense', label: 'Dense' }), '未知');
assert.equal(context.quantDisplayLabel({ kind: 'affine', label: 'affine 4-bit', bits: 4 }), '4-bit');
const quantTarget = element();
quantTarget.hidden = false;
quantTarget.dataset.quantTooltip = '未量化；dtype=fp16';
context.showQuantizationTooltip(quantTarget);
assert.equal(elements.get('quantization-tooltip').hidden, false);
assert.equal(elements.get('quantization-tooltip').textContent, '未量化；dtype=fp16');
assert.equal(quantTarget.getAttribute('aria-describedby'), 'quantization-tooltip');
context.hideQuantizationTooltip();
assert.equal(elements.get('quantization-tooltip').hidden, true);
assert.equal(quantTarget.getAttribute('aria-describedby'), undefined);
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
assert.equal(elements.get('tts-alias').value, tts.id);
assert.equal(elements.get('tts-model-type').value, 'TTS');
assert.ok(!elements.has('tts-details-files'));
assert.ok(!elements.has('tts-details-dependencies'));
context.window.__MODEL_PARAMS__ = { [tts.id]: { alias: 'My voice', model_type: 'llm' } };
context.openTTSDetails(tts);
assert.equal(elements.get('tts-alias').value, 'My voice');
assert.equal(elements.get('tts-model-type').value, 'TTS');
context.closeTTSDetails();
assert.ok(!elements.get('tts-details-modal').classList.contains('open'));
context.openTTSDetails({ id: 'empty', type: 'tts' });
assert.equal(elements.get('tts-alias').value, 'empty');
console.log('Dashboard TTS behavior and full script syntax checks passed.');

context.agentEndpoint = () => 'http://127.0.0.1:19068/v1';
context.refreshEndpoints = () => {};
const decision = { id: 'aac6fef/laya-multilingual-mlx', type: 'decision', max_position_embeddings: 1024, readiness: { status: 'ready' } };
context.window.__LOCAL_MODELS__[decision.id] = decision;
assert.equal(context.isModelLoadable(decision), true);
context.openParamsModal(decision.id);
assert.ok(elements.get('decision-details-modal').classList.contains('open'));
assert.ok(!elements.has('modal-max-tokens'), 'Decision model must not enter the generation parameter form');
assert.equal(elements.get('decision-alias').value, decision.id);
assert.equal(elements.get('decision-model-type').value, 'DECISION');
assert.equal(elements.get('decision-context-size').value, '1,024');
context.window.__MODEL_PARAMS__ = { [decision.id]: { alias: 'My decision model', context_size: '8192' } };
context.openDecisionDetails(decision);
assert.equal(elements.get('decision-alias').value, 'My decision model');
assert.equal(elements.get('decision-context-size').value, '1,024');
context.openDecisionDetails({ id: 'unknown', type: 'decision' });
assert.equal(elements.get('decision-context-size').value, '--');
context.closeDecisionDetails();
assert.ok(!elements.get('decision-details-modal').classList.contains('open'));
for (const language of ['en', 'zh-Hans', 'zh-Hant', 'ja', 'ko']) {
  for (const key of ['decision_close', 'decision_context_help', 'err_dflash2_unload_required']) {
    assert.ok(vm.runInContext(`I18N[${JSON.stringify(language)}][${JSON.stringify(key)}]`, context), `${language}: ${key}`);
  }
}
console.log('Dashboard decision model metadata and localization checks passed.');

context.openDecisionDetails(decision);
assert.equal(elements.get('decision-dtype').value, 'float16');
assert.equal(elements.get('decision-batch-size').value, 16);
assert.equal(elements.get('decision-cache-prompts').checked, false);
let savedDecision;
context.window.webkit.messageHandlers.saveModelParams = { postMessage(payload) { savedDecision = JSON.parse(payload); } };
for (const invalid of ['', '0', '-1', '1.5', '257', 'NaN']) {
  elements.get('decision-batch-size').value = invalid;
  context.saveDecisionParams();
  assert.equal(savedDecision, undefined);
  assert.ok(elements.get('decision-details-modal').classList.contains('open'));
}
elements.get('decision-batch-size').value = '2';
elements.get('decision-dtype').value = 'float32';
elements.get('decision-cache-prompts').checked = true;
context.saveDecisionParams();
assert.deepEqual(savedDecision.decision, { dtype: 'float32', batch_size: 2, cache_prompts: true, device: 'auto', compile: false, pad_to_multiple: null });
assert.equal(savedDecision.alias, 'My decision model');
context.window.__MODEL_PARAMS__[decision.id] = savedDecision;
context.openDecisionDetails(decision);
assert.equal(elements.get('decision-dtype').value, 'float32');
assert.equal(elements.get('decision-batch-size').value, 2);
assert.equal(elements.get('decision-cache-prompts').checked, true);
for (const language of ['en', 'zh-Hans', 'zh-Hant', 'ja', 'ko']) {
  for (const key of ['decision_precision', 'decision_batch', 'decision_cache', 'decision_precision_help', 'decision_batch_help', 'decision_cache_help', 'decision_batch_invalid']) {
    assert.ok(vm.runInContext(`I18N[${JSON.stringify(language)}][${JSON.stringify(key)}]`, context));
  }
}
console.log('Decision settings defaults, validation, persistence and localization passed.');

for (const [type, label] of Object.entries({llm:'LLM', vlm:'LLM/VLM', block_diffusion_vlm:'Block Diffusion VLM', embedding:'Embedding', reranker:'Reranker', asr:'ASR', tts:'TTS', decision:'DECISION'})) {
  assert.equal(context.modelTypeLabel({type}), label);
  assert.ok(context.renderModelType({type}).includes('>' + label + '</span>'));
}
for (const id of ['modal-model-type', 'decision-model-type', 'tts-model-type']) {
  assert.match(html, new RegExp('<input[^>]+id="' + id + '"[^>]*readonly'));
}
assert.ok(!html.includes("saved.model_type || 'auto'"));
console.log('All model types use the same detected label and readonly control.');

assert.equal(elements.get('decision-device').value, 'auto');
assert.equal(elements.get('decision-advanced').open, false);
assert.equal(elements.get('decision-head-budget').value, '--');
assert.equal(elements.get('decision-calibration').value, '--');
decision.decision_metadata = {head_max_len: 192, temperature: [0.5, 1.2, 5], temperature_by_options: {'choice:2': 1.3}};
context.openDecisionDetails(decision);
assert.equal(elements.get('decision-head-budget').value, 192);
assert.equal(elements.get('decision-calibration').value, 'choice: 0.5 · score: 1.2 · noul: 5.0 · choice:2: 1.3');
elements.get('decision-batch-size').value = '16';
elements.get('decision-padding').checked = true;
savedDecision = undefined;
for (const invalid of ['', '0', '1.5', '1025', 'no']) {
  elements.get('decision-pad-multiple').value = invalid;
  context.saveDecisionParams();
  assert.equal(savedDecision, undefined);
}
elements.get('decision-pad-multiple').value = '32';
elements.get('decision-device').value = 'cpu';
elements.get('decision-compile').checked = true;
context.saveDecisionParams();
assert.equal(savedDecision.decision.device, 'cpu');
assert.equal(savedDecision.decision.compile, true);
assert.equal(savedDecision.decision.pad_to_multiple, 32);
assert.ok(!('head_max_len' in savedDecision.decision));
context.window.__MODEL_PARAMS__[decision.id] = savedDecision;
context.openDecisionDetails(decision);
assert.equal(elements.get('decision-device').value, 'cpu');
assert.equal(elements.get('decision-compile').checked, true);
assert.equal(elements.get('decision-padding').checked, true);
assert.equal(elements.get('decision-pad-multiple').value, 32);
assert.equal(elements.get('decision-pad-multiple').disabled, false);
for (const language of ['en', 'zh-Hans', 'zh-Hant', 'ja', 'ko']) {
  for (const key of ['advanced','device','device_auto','device_help','compile','compile_help','padding','padding_help','padding_invalid','head_budget','head_budget_help','calibration','calibration_help']) {
    assert.ok(vm.runInContext(`I18N[${JSON.stringify(language)}][${JSON.stringify('decision_' + key)}]`, context));
  }
}
console.log('Decision advanced settings validation, metadata and persistence passed.');
