const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const html = fs.readFileSync(path.join(__dirname,
  '../../ironmlx-app/Sources/IronMLXAppCore/Resources/dashboard2.html'), 'utf8');
const pageStart = html.indexOf('<div class="page" id="page-chat">');
const pageEnd = html.indexOf('<div class="page" id="page-voices">', pageStart);
const agentPage = pageStart >= 0 && pageEnd > pageStart ? html.slice(pageStart, pageEnd) : '';
const LANGS = ['en', 'zh-Hans', 'zh-Hant', 'ja', 'ko'];
const MANUAL_PATHS = {
  'hermes-manual': '~/.hermes/profiles/ironmlx/config.yaml',
  'omp-manual': '~/.omp/agent/models.yml',
  'opencode-manual': '~/.config/opencode/opencode.jsonc',
};
const DSH_MANUALS = ['dsh-harness-cli-manual', 'dsh-harness-desktop-manual', 'dsh-desktop-manual'];

function realI18N() {
  const start = html.indexOf('  const I18N =');
  const src = html.slice(start, html.indexOf('\n  };', start) + 5);
  const context = {};
  vm.createContext(context);
  vm.runInContext(src + '\nthis.I18N = I18N;', context);
  return context.I18N;
}

// The <summary> of one manual-configuration <details>.
function manualSummary(id) {
  const open = agentPage.indexOf('id="' + id + '"');
  assert.ok(open >= 0, 'missing #' + id);
  const from = agentPage.indexOf('<summary', open);
  return agentPage.slice(from, agentPage.indexOf('</summary>', from) + '</summary>'.length);
}

// CSS rule body for an exact selector (first occurrence).
function cssRule(selector) {
  const at = html.indexOf('\n  ' + selector + ' {');
  assert.ok(at >= 0, 'missing rule ' + selector);
  const open = html.indexOf('{', at);
  return html.slice(open + 1, html.indexOf('}', open));
}

test('eyebrow_is_removed', () => {
  assert.ok(agentPage.length > 0);
  assert.ok(!agentPage.includes('agent-guide-eyebrow'));
  assert.ok(!html.includes('agent_provider_setup'));
});

test('manual_summaries_keep_label_in_inner_span', () => {
  for (const id of [...Object.keys(MANUAL_PATHS), ...DSH_MANUALS]) {
    const summary = manualSummary(id);
    assert.match(summary,
      /^<summary class="agent-guide-section-title agent-manual-summary"><span data-i18n="dsh_manual_configuration">/,
      id);
  }
  assert.ok(!/<summary[^>]*data-i18n=/.test(agentPage), 'no summary carries data-i18n itself');
});

test('manual_paths_for_three_agents', () => {
  for (const [id, filePath] of Object.entries(MANUAL_PATHS)) {
    const summary = manualSummary(id);
    assert.ok(summary.includes('class="agent-manual-path"'), id);
    assert.ok(summary.includes('data-i18n="agent_edit_file"'), id);
    assert.ok(summary.includes('<code>' + filePath + '</code>'), id);
  }
  for (const id of DSH_MANUALS) assert.ok(!manualSummary(id).includes('agent-manual-path'), id);
});

// setLanguage rewrites textContent of every [data-i18n] element, so the path must sit outside them.
test('language_switch_keeps_manual_paths', () => {
  const re = /<(span|summary|div|p|h3|label)\b[^>]*data-i18n="[^"]+"[^>]*>/g;
  let match;
  while ((match = re.exec(agentPage))) {
    const close = agentPage.indexOf('</' + match[1] + '>', match.index);
    assert.ok(!agentPage.slice(match.index, close).includes('agent-manual-path'), match[0]);
  }
});

test('agent_edit_file_exists_in_five_languages', () => {
  const I18N = realI18N();
  for (const lang of LANGS) {
    assert.equal(typeof I18N[lang].agent_edit_file, 'string', lang);
    assert.ok(I18N[lang].agent_edit_file.length > 0, lang);
  }
});

// In light mode --ds-control-selected is white, the same as the card, so it cannot mark the selection.
test('selected_nav_item_is_not_white_in_light_theme', () => {
  const rule = cssRule('.agent-guide-nav-item.active');
  assert.ok(rule.includes('var(--ds-surface-2)'));
  assert.ok(!rule.includes('--ds-control-selected'));
  assert.ok(!rule.includes('--accent'));
});

test('cards_use_ds_tokens', () => {
  for (const selector of ['.agent-guide-nav', '.agent-guide-hero', '.agent-guide-section', '.agent-guide-compact-bar']) {
    const rule = cssRule(selector);
    assert.ok(rule.includes('var(--ds-surface-1)'), selector);
    assert.ok(rule.includes('var(--ds-border)'), selector);
    assert.ok(rule.includes('var(--ds-radius-card)'), selector);
    assert.ok(!rule.includes('box-shadow'), selector);
  }
});

test('version_badge_is_neutral_pill', () => {
  const rule = cssRule('.agent-guide-version-badge');
  assert.ok(rule.includes('var(--ds-surface-2)'));
  assert.ok(rule.includes('border-radius: 999px'));
  assert.ok(!rule.includes('--accent'));
});

test('manual_path_is_secondary', () => {
  assert.ok(cssRule('.agent-manual-path').includes('var(--ds-text-2)'));
});

test('connection_tiles_are_gray_and_endpoint_is_mono', () => {
  assert.ok(cssRule('.agent-runtime-item').includes('var(--ds-surface-2)'));
  assert.ok(cssRule('.agent-runtime-value').includes('var(--ds-font-mono)'));
});

// Every .agent-* / .dsh-* rule body (logo tiles keep their fixed brand backgrounds).
function agentRules() {
  const out = [];
  const re = /\n  ((?:\.agent-|\.dsh-)[^{}]*)\{([^}]*)\}/g;
  let match;
  while ((match = re.exec(html))) {
    if (/\.agent-guide-mark/.test(match[1])) continue;
    out.push({ selector: match[1].trim(), body: match[2] });
  }
  return out;
}

test('agent_css_has_no_hardcoded_colors', () => {
  const rules = agentRules();
  assert.ok(rules.length > 50);
  for (const { selector, body } of rules) {
    assert.ok(!/#[0-9a-fA-F]{3,8}\b/.test(body), selector + ' uses a hex color');
    assert.ok(!/\brgba?\(/.test(body), selector + ' uses an rgb color');
    assert.ok(!/var\(--(accent|card-bg|bg|text|text-secondary|text-primary|border|green|orange|settings-[a-z-]+)[,)]/.test(body),
      selector + ' uses a pre-design-system variable');
  }
});

test('tones_use_color_mix', () => {
  assert.ok(cssRule('.agent-setup-feedback[data-tone="warning"]').includes('color-mix(in srgb, var(--ds-amber)'));
  assert.ok(cssRule('.agent-setup-feedback[data-tone="error"]').includes('color-mix(in srgb, var(--ds-red)'));
  assert.ok(cssRule('.agent-setup-feedback[data-tone="success"]').includes('color-mix(in srgb, var(--ds-green)'));
  assert.ok(cssRule('.agent-profile-option.active').includes('color-mix(in srgb, var(--ds-blue)'));
  assert.ok(cssRule('.agent-guide-note.warning').includes('color-mix(in srgb, var(--ds-amber)'));
});

test('code_block_is_themed', () => {
  const code = cssRule('.agent-code');
  assert.ok(code.includes('var(--ds-font-mono)'));
  assert.ok(code.includes('var(--ds-text-1)'));
  assert.ok(cssRule('.agent-code-toolbar').includes('var(--ds-surface-2)'));
  assert.ok(cssRule('.agent-code-wrap').includes('var(--ds-radius-control)'));
});

test('model_selects_match_ds_select', () => {
  const rule = cssRule('.agent-guide-field select,\n  .agent-guide-field input');
  assert.ok(rule.includes('var(--ds-border-strong)'));
  assert.ok(rule.includes('var(--ds-radius-control)'));
  assert.ok(rule.includes('var(--ds-surface-1)'));
});

// The global select draws its chevron at the right edge; long model names must not run under it.
test('model_select_leaves_room_for_chevron', () => {
  const rule = cssRule('.agent-guide-field select,\n  .agent-guide-field input');
  assert.match(rule, /padding: 6px 28px 6px 10px;/);
});

// JS disables the one-click buttons while setup runs; that state has to look different.
test('one_click_button_shows_disabled_state', () => {
  const rule = cssRule('.agent-guide-section .btn-accent:disabled');
  assert.ok(rule.includes('opacity: 0.5'));
  assert.ok(rule.includes('cursor: default'));
});

// Restyling this page must not touch rules shared with other pages.
test('shared_rules_untouched', () => {
  assert.ok(html.includes('  .page-title {\n    font-size: 24px;\n    font-weight: 700;\n    margin-bottom: 6px;\n  }'));
  assert.ok(html.includes('  .page-subtitle {\n    font-size: 13px;\n    color: var(--text-secondary);\n    margin-bottom: 24px;\n  }'));
  assert.ok(html.includes('  .btn-accent {\n    background: var(--accent);\n    color: white;\n    border: none;\n    border-radius: 6px;\n    padding: 7px 18px;'));
  assert.ok(html.includes('  .ds-tuning-select { box-sizing: border-box; width: 100%; min-width: 0; padding: 6px 10px; border: 0.5px solid var(--ds-border-strong); border-radius: var(--ds-radius-control); background-color: var(--ds-surface-1); color: var(--ds-text-1); font: inherit; font-size: 13px; }'));
});

module.exports = { cssRule };
