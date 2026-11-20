const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const html = fs.readFileSync(path.join(__dirname,
  '../../ironmlx-app/Sources/IronMLXAppCore/Resources/dashboard2.html'), 'utf8');
const LANGS = ['en', 'zh-Hans', 'zh-Hant', 'ja', 'ko'];
const STATUS_DISABLED = { en: 'Disabled', 'zh-Hans': '已停用', 'zh-Hant': '已停用', ja: '無効', ko: '비활성' };

function i18nSource() {
  const start = html.indexOf('  const I18N =');
  return html.slice(start, html.indexOf('\n  };', start) + 5);
}

function fnSource(name) {
  const match = html.match(new RegExp('  function ' + name + '\\([^]*?\\n  }'));
  assert.ok(match, name);
  return match[0];
}

// One-line CSS rule body for an exact selector.
function cssRule(selector) {
  const at = html.indexOf('\n  ' + selector + ' {');
  assert.ok(at >= 0, 'missing rule ' + selector);
  const open = html.indexOf('{', at);
  return html.slice(open + 1, html.indexOf('}', open));
}

const VOICES = [
  { id: 'a', name: '甲', language: 'zh', enabled: true, reference_format: 'wav', duration_ms: 8400, sample_rate: 48000 },
  { id: 'b', name: '乙', language: 'en', enabled: false, reference_format: 'mp3', duration_ms: 2100, sample_rate: 24000 },
];

// Render the list in a minimal page and return the list markup.
function render(voices, lang = 'zh-Hans') {
  const elements = new Map();
  const element = id => {
    if (!elements.has(id)) {
      elements.set(id, { id, innerHTML: '', hidden: true, dataset: {}, parentElement: null,
        after() {}, before() {}, querySelectorAll() { return []; } });
    }
    return elements.get(id);
  };
  const context = {
    currentLang: lang,
    HTML_LANG_MAP: { en: 'en', 'zh-Hans': 'zh-CN', 'zh-Hant': 'zh-Hant', ja: 'ja', ko: 'ko' },
    pendingVoicePreviewIds: new Set(),
    voiceProfiles: [],
    editingVoiceId: null,
    document: { getElementById: element, querySelectorAll() { return []; } },
  };
  vm.createContext(context);
  vm.runInContext(i18nSource(), context);
  for (const name of ['escapeAttr', 'normalizeVoiceLanguage', 'voiceLanguageLabel', 'formatVoicePlayerTime',
    'voicePlayerIcon', 'releaseVoicePlayers', 'positionVoiceEditor', 'renderVoiceProfiles']) {
    vm.runInContext(fnSource(name), context);
  }
  vm.runInContext('renderVoiceProfiles(' + JSON.stringify(Array.isArray(voices) ? { data: voices } : voices) + ')', context);
  render.addButton = element('voice-add-btn');
  return element('voices-list').innerHTML;
}

function realI18N() {
  const context = {};
  vm.createContext(context);
  vm.runInContext(i18nSource() + '\nthis.I18N = I18N;', context);
  return context.I18N;
}

test('disabled_row_is_marked', () => {
  const out = render(VOICES);
  assert.ok(out.includes('data-voice-card-id="a" data-enabled="true"'));
  assert.ok(out.includes('data-voice-card-id="b" data-enabled="false"'));
  assert.equal(out.split('>已停用<').length - 1, 1);
  assert.ok(out.indexOf('>已停用<') > out.indexOf('data-voice-card-id="b"'));
  assert.ok(!out.includes('is-disabled'));
  assert.ok(!html.includes('is-disabled'));
});

test('disabled_pill_is_localized', () => {
  assert.ok(render(VOICES, 'en').includes('<span class="voice-status-pill">Disabled</span>'));
});

test('id_shown_without_label', () => {
  const out = render(VOICES);
  assert.ok(out.includes('<span class="voice-card-title-id" title="a">a</span>'));
  assert.ok(!out.includes('(ID:'));
});

test('voice_name_title_is_escaped', () => {
  const out = render([{ ...VOICES[0], name: 'A"<b>' }]);
  assert.ok(out.includes('<span class="voice-card-name" title="A&quot;&lt;b&gt;">A&quot;&lt;b&gt;</span>'));
  assert.ok(!out.includes('<b>'));
});

test('delete_is_text_button', () => {
  const out = render(VOICES);
  assert.ok(out.includes('<button class="voice-delete-btn" data-voice-id="a" onclick="deleteVoice(this.dataset.voiceId)">'));
  assert.ok(!out.includes('btn-danger'));
});

test('status_key_in_five_languages', () => {
  const I18N = realI18N();
  for (const lang of LANGS) assert.equal(I18N[lang].voice_status_disabled, STATUS_DISABLED[lang], lang);
});

test('list_is_one_card', () => {
  const rule = cssRule('.voices-list');
  for (const token of ['var(--ds-surface-1)', 'var(--ds-border)', 'var(--ds-radius-card)']) {
    assert.ok(rule.includes(token), token);
  }
  assert.ok(!rule.includes('box-shadow'));
});

// Help tooltips inside an inline editor hang below the list, so the list card must not clip its children.
test('list_does_not_clip_inline_editor_tooltips', () => {
  assert.ok(!cssRule('.voices-list').includes('overflow: hidden'));
  assert.ok(cssRule('.voice-editor[data-position="inline"]:last-child').includes('border-radius: 0 0 var(--ds-radius-card) var(--ds-radius-card)'));
});

// Only the new narrow-window rules may lay out a voice row; older breakpoints used to override them.
test('one_narrow_layout_for_voice_rows', () => {
  const blocks = html.match(/@media \(max-width:\s*\d+px\) \{[^@]*?\.voice-card-header[^@]*?\}\s*\}/g) || [];
  assert.equal(blocks.length, 1, blocks.join('\n'));
  assert.ok(blocks[0].startsWith('@media (max-width: 760px)'));
  assert.ok(blocks[0].includes('.voice-editor-grid { grid-template-columns: 1fr; }'));
  assert.ok(!/max-width:\s*(900|560)px\)\s*\{[^}]*\.voice/.test(html));
});

// Voice IDs are up to 64 characters with no spaces; they must shorten instead of running into the player.
test('long_voice_id_is_truncated', () => {
  const rule = cssRule('.voice-card-title-id');
  for (const token of ['min-width: 0', 'overflow: hidden', 'text-overflow: ellipsis', 'white-space: nowrap']) {
    assert.ok(rule.includes(token), token);
  }
  assert.ok(render(VOICES).includes('<span class="voice-card-title-id" title="a">a</span>'));
});

test('editor_buttons_keep_hover_feedback', () => {
  assert.ok(cssRule('#page-voices .voice-file-picker .btn-secondary:hover,\n  #page-voices .voice-editor-actions .btn-secondary:hover').includes('var(--ds-surface-2)'));
});

test('rows_are_aligned', () => {
  assert.ok(cssRule('.voice-card-header').includes('grid-template-columns: minmax(0, 1fr) 300px auto'));
  assert.ok(cssRule('.voice-card + .voice-card').includes('border-top: 0.5px solid var(--ds-border)'));
});

test('disabled_info_is_dimmed', () => {
  // Dim the name, ID and details, but keep the "Disabled" pill at full strength so it stays readable.
  const rule = '.voice-card[data-enabled="false"] .voice-card-name,\n  .voice-card[data-enabled="false"] .voice-card-title-id,\n  .voice-card[data-enabled="false"] .voice-card-meta';
  assert.ok(cssRule(rule).includes('opacity: 0.55'));
  assert.ok(!html.includes('.voice-card[data-enabled="false"] .voice-card-main {'));
  assert.ok(cssRule('.voice-status-pill').includes('var(--ds-surface-2)'));
});

test('delete_is_red_text', () => {
  const rule = cssRule('.voice-delete-btn');
  assert.ok(rule.includes('var(--ds-red)'));
  assert.ok(rule.includes('background: transparent'));
});

test('voice_css_uses_ds_tokens_only', () => {
  const re = /\n\s+((?:\.voice|#page-voices)[^{}]*)\{([^}]*)\}/g;
  let match;
  let count = 0;
  while ((match = re.exec(html))) {
    count += 1;
    const [, selector, body] = match;
    assert.ok(!/#[0-9a-fA-F]{3,8}\b/.test(body), selector + ' uses a hex color');
    assert.ok(!/\brgba?\(/.test(body), selector + ' uses an rgb color');
    assert.ok(!/var\(--(?!ds-)/.test(body), selector + ' uses a non-ds variable');
  }
  assert.ok(count > 20);
});

// Restyling this page must not touch rules shared with other pages.
test('shared_rules_untouched', () => {
  assert.ok(html.includes('  .btn-accent {\n    background: var(--accent);\n    color: white;\n    border: none;\n    border-radius: 6px;\n    padding: 7px 18px;\n    font-size: 12px;\n    font-weight: 600;'));
  assert.ok(html.includes('  .btn-secondary {\n    padding: 8px 16px;\n    border-radius: 8px;\n    border: 1px solid var(--border);\n    background: var(--card-bg);\n    color: var(--text);\n    font-size: 13px;'));
  assert.ok(html.includes('  .btn-danger {\n    background: #ff3b30;\n    color: white;\n    border: none;\n    border-radius: 8px;\n    padding: 8px 20px;'));
  assert.ok(html.includes('  .card-settings {\n    background: var(--card-bg);\n    border: 0.5px solid var(--border);\n    border-radius: var(--radius);\n    margin-bottom: 20px;\n    box-shadow: 0 1px 3px rgba(0,0,0,0.04);'));
});

test('editor_uses_new_card', () => {
  assert.ok(html.includes('<div id="voice-editor" class="voice-editor" hidden>'));
  assert.ok(html.includes('<div class="voice-editor-title" id="voice-editor-title" data-i18n="voice_add">'));
  assert.ok(!html.includes('<div class="card-header"><span id="voice-editor-title"'));
});

// Editing a row opens the form inside the list card; it must read as that row expanding, not a card in a card.
test('inline_editor_has_no_own_card_border', () => {
  const inline = cssRule('.voice-editor[data-position="inline"]');
  for (const token of ['border: 0', 'border-radius: 0', 'border-top: 0.5px solid var(--ds-border)', 'var(--ds-surface-2)']) {
    assert.ok(inline.includes(token), token);
  }
  const card = cssRule('.voice-editor');
  assert.ok(card.includes('var(--ds-surface-1)'));
  assert.ok(card.includes('var(--ds-radius-card)'));
});

test('editor_inputs_match_ds_select', () => {
  const rule = cssRule('.voice-editor input[type="text"], .voice-editor select');
  for (const token of ['var(--ds-border-strong)', 'var(--ds-radius-control)', 'padding: 6px 10px']) {
    assert.ok(rule.includes(token), token);
  }
  // Only the select needs room for its arrow.
  assert.ok(cssRule('.voice-editor select').includes('padding-right: 28px'));
});

// The row after an open inline editor keeps its divider.
test('row_after_inline_editor_has_divider', () => {
  assert.ok(cssRule('.voice-editor[data-position="inline"] + .voice-card').includes('border-top: 0.5px solid var(--ds-border)'));
  assert.ok(!html.includes('.voice-editor[data-position="create"] { margin: 0 0 10px; }'));
});

test('editor_hidden_actions_rule_kept', () => {
  assert.match(html, /\.voice-editor-actions\s*>\s*\[hidden\]\s*\{\s*display\s*:\s*none/);
});

// A broken profile file must read as a problem, not as "no voices yet", and adding must be blocked.
test('unavailable_store_is_explained', () => {
  const out = render({ data: [], unavailable: 'parsing profiles: key <must> be a string' });
  assert.ok(out.includes('class="voice-store-unavailable"'));
  assert.ok(out.includes('声音配置文件无法读取'));
  assert.ok(out.includes('parsing profiles: key &lt;must&gt; be a string'));
  assert.ok(!out.includes('尚未配置声音'));
  assert.equal(render.addButton.disabled, true);
  render(VOICES);
  assert.equal(render.addButton.disabled, false);
});

test('add_button_has_stable_id', () => {
  assert.ok(html.includes('<button id="voice-add-btn" class="btn-accent" type="button" onclick="openVoiceEditor()" data-i18n="voice_add">'));
});

test('unavailable_message_in_five_languages', () => {
  const I18N = realI18N();
  for (const lang of LANGS) {
    assert.equal(typeof I18N[lang].voice_store_unavailable, 'string', lang);
    assert.ok(I18N[lang].voice_store_unavailable.length > 0, lang);
  }
});

module.exports = { cssRule };
