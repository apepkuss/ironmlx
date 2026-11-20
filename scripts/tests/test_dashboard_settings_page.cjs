const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const html = fs.readFileSync(path.join(__dirname,
  '../../ironmlx-app/Sources/IronMLXAppCore/Resources/dashboard2.html'), 'utf8');
const pageStart = html.indexOf('<div class="page" id="page-settings">');
const pageEnd = html.indexOf('<!-- Cluster Page -->', pageStart);
const settingsPage = pageStart >= 0 && pageEnd > pageStart ? html.slice(pageStart, pageEnd) : '';
const i18nStart = html.indexOf('  const I18N =');
const translations = html.slice(i18nStart, html.indexOf('\n  };', i18nStart) + 5);
const LANGS = ['en', 'zh-Hans', 'zh-Hant', 'ja', 'ko'];

const GROUPS = ['general', 'server', 'models', 'cache', 'acceleration'];
const ROWS = {
  general: ['lang-select', 'theme-select'],
  server: ['cfg-network-mode', 'cfg-lan-host', 'cfg-lan-security-row', 'cfg-port', 'cfg-log-level'],
  models: ['cfg-max-models', 'cfg-model-ttl', 'cfg-max-sequences', 'cfg-mem-total', 'cfg-mem-model',
    'cfg-verify-model-on-load'],
  cache: ['cfg-cache-enable', 'cfg-hot-cache', 'cfg-cache-dir', 'cfg-cold-cache', 'cfg-kv-quant',
    'cfg-active-kv-offload'],
  acceleration: ['cfg-m5-dflash2-profile', 'cfg-ane'],
};

function slice(start, end) {
  const from = html.indexOf(start);
  assert.ok(from >= 0, `missing ${start}`);
  const to = html.indexOf(end, from);
  assert.ok(to > from, `missing ${end}`);
  return html.slice(from, to);
}

// Each group's markup, from its opening tag to the next group (or the end of the scroller).
function groupSections() {
  const re = /<section class="([^"]*)" data-settings-group="([a-z]+)">/g;
  const marks = [];
  let match;
  while ((match = re.exec(settingsPage))) marks.push({ index: match.index, classes: match[1], group: match[2] });
  return marks.map((mark, i) => {
    const end = i + 1 < marks.length ? marks[i + 1].index : settingsPage.indexOf('id="settings-save-bar"');
    const body = settingsPage.slice(mark.index, end);
    return { ...mark, body: body.slice(0, body.lastIndexOf('</section>') + '</section>'.length) };
  });
}

// Depth of every element relative to the section, so rows can be checked as direct children.
function childTags(sectionHtml) {
  const tokens = /<(\/?)(div|section|span|label|select|button|svg|input|option|path|rect|circle)\b([^>]*)>/g;
  const voids = new Set(['input', 'path', 'rect', 'circle']);
  const out = [];
  let depth = 0;
  let token;
  while ((token = tokens.exec(sectionHtml))) {
    const [, closing, tag, attrs] = token;
    if (closing) { depth -= 1; continue; }
    if (depth === 1) out.push({ tag, attrs });
    if (!voids.has(tag) && !attrs.trim().endsWith('/')) depth += 1;
  }
  return out;
}

function loadI18N() {
  const context = {};
  vm.createContext(context);
  vm.runInContext(translations + '\nthis.I18N = I18N;', context);
  return context.I18N;
}

test('groups_in_order', () => {
  const sections = groupSections();
  assert.deepEqual(sections.map(s => s.group), GROUPS);
  for (const section of sections) {
    assert.ok(section.classes.split(' ').includes('card-settings'), `${section.group} keeps .card-settings`);
    assert.ok(section.classes.split(' ').includes('ds-settings-card'), `${section.group} uses .ds-settings-card`);
  }
});

test('group_header_first_then_rows_as_direct_children', () => {
  const sections = groupSections();
  assert.equal(sections.length, GROUPS.length);
  for (const section of sections) {
    const children = childTags(section.body);
    assert.ok(children.length > 1, `${section.group} has children`);
    assert.match(children[0].attrs, /class="card-header"/, `${section.group} starts with .card-header`);
    const header = section.body.slice(section.body.indexOf('class="card-header"'));
    assert.match(header.slice(0, header.indexOf('</div>')),
      new RegExp(`data-i18n="settings_group_${section.group}"`), `${section.group} header key`);
    for (const child of children.slice(1)) {
      assert.match(child.attrs, /class="settings-row(?: ds-sub-row)?"/, `${section.group}: ${child.tag}${child.attrs} must be a .settings-row`);
    }
  }
});

test('rows_per_group', () => {
  const known = new Set(Object.values(ROWS).flat());
  const sections = groupSections();
  assert.equal(sections.length, GROUPS.length);
  for (const section of sections) {
    const ids = [...section.body.matchAll(/\bid="([^"]+)"/g)].map(m => m[1]).filter(id => known.has(id));
    assert.deepEqual(ids, ROWS[section.group], `${section.group} rows`);
  }
});

test('all_settings_ids_still_present', () => {
  const ids = [
    // Read without null checks when the page script loads.
    'cfg-hot-cache', 'cfg-hot-cache-auto', 'cfg-hot-cache-auto-hint', 'cfg-mem-total', 'cfg-mem-total-auto',
    'cfg-mem-model', 'cfg-mem-model-auto', 'cfg-cold-cache', 'cfg-cache-enable', 'cfg-cache-dir',
    'cfg-m5-dflash2-profile', 'cfg-ane', 'ane-requires-m5-profile', 'lang-select', 'onboarding-lang-select',
    // Every other settings control, row and save-bar element.
    'theme-select', 'cfg-network-mode', 'cfg-lan-host-row', 'cfg-lan-host', 'cfg-lan-security-row', 'cfg-port',
    'cfg-log-level', 'log-level-status', 'cfg-max-models', 'cfg-model-ttl', 'cfg-max-sequences',
    'cfg-verify-model-on-load', 'cfg-kv-quant', 'cfg-active-kv-offload', 'settings-save-bar',
    'settings-save-status', 'settings-save-button', 'settings-cancel-button',
  ];
  for (const id of ids) {
    const count = html.split(`id="${id}"`).length - 1;
    assert.equal(count, 1, `id="${id}" appears ${count} times`);
  }
});

test('no_shared_badge_restart', () => {
  assert.ok(settingsPage.length > 0);
  assert.ok(!settingsPage.includes('badge-restart'));
});

test('theme_uses_ds_seg', () => {
  const open = settingsPage.match(/<div class="([^"]*)" id="theme-select"[^>]*>/);
  assert.ok(open, 'theme-select element');
  const classes = open[1].split(' ');
  assert.ok(classes.includes('ds-seg'), 'ds-seg class');
  assert.ok(classes.includes('seg-group'), 'keeps seg-group class');
  const from = settingsPage.indexOf(open[0]);
  const body = settingsPage.slice(from, settingsPage.indexOf('</div>', from));
  assert.match(body, /class="ds-seg-thumb"/);
  const buttons = [...body.matchAll(/<button[^>]*class="seg-btn[^"]*"[^>]*data-value="([a-z]+)"/g)].map(m => m[1]);
  assert.deepEqual(buttons, ['light', 'dark', 'system']);
});

test('theme_selection_calls_dsSegSelect', () => {
  const buttons = ['light', 'dark', 'system'].map(value => {
    const classes = new Set(value === 'system' ? ['active'] : []);
    return { dataset: { value }, classList: { toggle: (name, on) => (on ? classes.add(name) : classes.delete(name)), has: n => classes.has(n) } };
  });
  const group = { id: 'theme-select' };
  const calls = [];
  const posted = [];
  const context = {
    document: {
      querySelectorAll: sel => (sel === '#theme-select .seg-btn' ? buttons : []),
      getElementById: id => (id === 'theme-select' ? group : null),
    },
    window: { webkit: { messageHandlers: { setTheme: { postMessage: v => posted.push(v) } } } },
    dsSegSelect: (g, value) => calls.push([g, value]),
  };
  vm.createContext(context);
  vm.runInContext(
    slice('  function setTheme(value)', "  document.querySelectorAll('#theme-select .seg-btn').forEach(btn => {")
    + slice('  function applyThemeInput(', '  function refreshSettingsDerivedUI('), context);

  context.setTheme('dark');
  assert.deepEqual(posted, ['dark'], 'setTheme still posts the bridge message');
  assert.deepEqual(buttons.map(b => b.classList.has('active')), [false, true, false]);
  assert.deepEqual(calls, [[group, 'dark']]);

  context.applyThemeInput('');
  assert.deepEqual(buttons.map(b => b.classList.has('active')), [false, false, true]);
  assert.deepEqual(calls[1], [group, 'system']);
  assert.deepEqual(posted, ['dark'], 'applyThemeInput does not post');
});

test('header_uses_settings_header_hint', () => {
  const header = settingsPage.slice(0, settingsPage.indexOf('settings-page-scroll'));
  assert.match(header, /data-i18n="settings_title"/);
  assert.match(header, /id="settings-subtitle"/);
  assert.ok(!html.includes('settings_subtitle_html'), 'old subtitle key removed');
});

test('group_titles_in_five_languages', () => {
  const I18N = loadI18N();
  const keys = GROUPS.map(g => `settings_group_${g}`).concat(['settings_header_hint']);
  for (const key of keys) {
    for (const lang of LANGS) assert.ok(I18N[lang][key], `${lang}.${key}`);
    for (const lang of LANGS.slice(1)) {
      assert.notEqual(I18N[lang][key], I18N.en[key], `${lang}.${key} must be translated`);
    }
  }
  for (const lang of LANGS) {
    assert.ok(I18N[lang].settings_header_hint.includes('{tag}'), `${lang}.settings_header_hint has {tag}`);
  }
  assert.equal(I18N.en.settings_group_general, 'General');
  assert.equal(I18N.en.settings_group_server, 'Server');
  assert.equal(I18N.en.settings_group_models, 'Models and memory');
  assert.equal(I18N.en.settings_group_cache, 'Cache');
  assert.equal(I18N.en.settings_group_acceleration, 'Acceleration');
  assert.equal(I18N['zh-Hans'].settings_group_general, '通用');
  assert.equal(I18N['zh-Hans'].settings_group_server, '服务器');
  assert.equal(I18N['zh-Hans'].settings_group_models, '模型与内存');
  assert.equal(I18N['zh-Hans'].settings_group_cache, '缓存');
  assert.equal(I18N['zh-Hans'].settings_group_acceleration, '加速');
  assert.equal(I18N['zh-Hans'].settings_header_hint, '标有 {tag} 的设置保存后会重启后端');
  assert.equal(I18N.en.settings_title, 'Settings');
  assert.equal(I18N['zh-Hans'].settings_title, '设置');
  assert.equal(I18N['zh-Hans'].theme_mode, '外观');
  for (const lang of LANGS.slice(1)) {
    assert.notEqual(I18N[lang].settings_title, I18N.en.settings_title, `${lang}.settings_title`);
  }
});

function restartFields() {
  const text = slice('  const RESTART_FIELDS = [', '];');
  const context = {};
  vm.createContext(context);
  vm.runInContext(text + '];\nthis.fields = RESTART_FIELDS;', context);
  return context.fields;
}

function settingsRows() {
  return settingsPage.split(/<div class="settings-row(?: ds-sub-row)?"/).slice(1).map(chunk => {
    const row = chunk.split('</section>')[0];
    return { html: row, ids: [...row.matchAll(/\bid="([^"]+)"/g)].map(m => m[1]) };
  });
}

test('restart_tags_match_restart_fields', () => {
  const fields = new Set(restartFields());
  const rows = settingsRows();
  assert.equal(rows.length, 21);
  for (const row of rows) {
    const expected = row.ids.some(id => fields.has(id));
    const tags = (row.html.match(/class="ds-restart-tag"/g) || []).length;
    assert.equal(tags, expected ? 1 : 0, `row ${row.ids.join(',') || row.html.slice(0, 80)}`);
  }
  for (const id of ['cfg-lan-security-row', 'cfg-verify-model-on-load', 'cfg-log-level', 'lang-select', 'theme-select']) {
    const row = rows.find(r => r.ids.includes(id));
    assert.ok(row, id);
    assert.ok(!row.html.includes('ds-restart-tag'), `${id} row has no restart tag`);
  }
  assert.ok(!settingsPage.includes('badge-restart'), 'settings page never uses .badge-restart');
  assert.ok(!settingsPage.includes('badge-live'), 'badge-live is gone from the settings page');
});

test('restart_tag_sits_in_the_label_and_uses_the_i18n_key', () => {
  for (const row of settingsRows()) {
    if (!row.html.includes('ds-restart-tag')) continue;
    const label = row.html.slice(row.html.indexOf('settings-row-label'), row.html.indexOf('settings-row-right'));
    assert.match(label, /<span class="ds-restart-tag" data-i18n="settings_restart_tag">Restart required<\/span>/);
  }
});

test('experimental_tag_only_on_active_kv_offload', () => {
  const rows = settingsRows();
  const tagged = rows.filter(r => r.html.includes('ds-experimental-tag'));
  assert.equal(tagged.length, 1);
  assert.ok(tagged[0].ids.includes('cfg-active-kv-offload'));
  assert.ok((tagged[0].html.match(/class="ds-experimental-tag"/g) || []).length === 1);
  assert.ok(tagged[0].html.indexOf('ds-experimental-tag') < tagged[0].html.indexOf('ds-restart-tag'),
    'experimental tag comes before the restart tag');
  assert.match(tagged[0].html, /data-i18n="settings_experimental_tag"/);
});

test('log_level_row_applies_immediately', () => {
  const row = settingsRows().find(r => r.ids.includes('cfg-log-level'));
  assert.match(row.html, /id="log-level-status"[^>]*data-i18n="settings_applies_now"/);
  assert.ok(!html.includes("logLevelStatusKey = 'log_level_live_desc'"), 'idle status uses settings_applies_now');
});

test('tag_copy_in_five_languages', () => {
  const I18N = loadI18N();
  const keys = ['settings_restart_tag', 'settings_experimental_tag', 'settings_applies_now'];
  for (const key of keys) {
    for (const lang of LANGS) assert.ok(I18N[lang][key], `${lang}.${key}`);
    for (const lang of LANGS.slice(1)) {
      assert.notEqual(I18N[lang][key], I18N.en[key], `${lang}.${key} must be translated`);
    }
  }
  assert.equal(I18N.en.settings_restart_tag, 'Restart required');
  assert.equal(I18N.en.settings_experimental_tag, 'Experimental');
  assert.equal(I18N.en.settings_applies_now, 'Applies immediately');
  assert.equal(I18N['zh-Hans'].settings_restart_tag, '需重启');
  assert.equal(I18N['zh-Hans'].settings_experimental_tag, '实验');
  assert.equal(I18N['zh-Hans'].settings_applies_now, '立即生效');
});

test('header_hint_renders_one_restart_tag_per_language', () => {
  const I18N = loadI18N();
  for (const lang of LANGS) {
    const created = [];
    const makeEl = tag => ({ tag, className: '', textContent: '', children: [] });
    const hint = { children: [], replaceChildren(...nodes) { this.children = nodes; } };
    const context = {
      document: {
        getElementById: id => (id === 'settings-subtitle' ? hint : null),
        createElement: tag => { const el = makeEl(tag); created.push(el); return el; },
        createTextNode: text => ({ text }),
      },
      I18N,
    };
    vm.createContext(context);
    vm.runInContext(slice('  function renderSettingsHeaderHint(', '  function t(key, fallback)'), context);
    context.renderSettingsHeaderHint(I18N[lang]);
    const tags = hint.children.filter(n => n.className === 'ds-restart-tag');
    assert.equal(tags.length, 1, `${lang}: exactly one tag`);
    assert.equal(tags[0].textContent, I18N[lang].settings_restart_tag, `${lang}: tag text`);
    const text = hint.children.map(n => (n.className ? '\u0000' : n.text)).join('');
    assert.ok(!text.includes('{tag}'), `${lang}: no literal {tag}`);
    assert.ok(text.includes('\u0000'), `${lang}: tag placed`);
  }
});

function itemCounter() {
  const text = slice('  const RESTART_FIELDS = [', '  let settingsBaseline = null;');
  const context = {};
  vm.createContext(context);
  vm.runInContext(text + '\nthis.count = settingsChangedItems;', context);
  return ids => JSON.parse(JSON.stringify(context.count(ids)));
}

test('items_dedupe_auto_and_slider', () => {
  assert.deepEqual(itemCounter()(['cfg-mem-total-auto', 'cfg-mem-total']),
    { count: 1, clusterCount: 0, restart: true });
  assert.deepEqual(itemCounter()(['cfg-network-mode', 'cfg-lan-host', 'cfg-port']),
    { count: 2, clusterCount: 0, restart: true });
});

test('items_count_cluster_separately', () => {
  assert.deepEqual(itemCounter()(['cfg-distributed-backend', 'cfg-port']),
    { count: 2, clusterCount: 1, restart: true });
  assert.deepEqual(itemCounter()(['cfg-distributed-backend', 'cfg-parallel-mode']),
    { count: 2, clusterCount: 2, restart: true });
});

test('items_non_restart_only', () => {
  assert.deepEqual(itemCounter()(['cfg-verify-model-on-load']), { count: 1, clusterCount: 0, restart: false });
  assert.deepEqual(itemCounter()([]), { count: 0, clusterCount: 0, restart: false });
});

test('clean_bar_is_hidden_by_css', () => {
  assert.match(html, /#settings-save-bar\[data-state="clean"\]\s*\{\s*display:\s*none;/);
  assert.match(html, /#settings-save-bar\[data-restart="true"\]/);
});

test('save_bar_copy_in_five_languages', () => {
  const context = {};
  vm.createContext(context);
  vm.runInContext(translations + '\nthis.I18N = I18N;', context);
  const I18N = context.I18N;
  const keys = ['settings_changes_count', 'settings_changes_one', 'settings_changes_cluster', 'settings_restart_note'];
  for (const lang of LANGS) {
    for (const key of keys) assert.ok(I18N[lang][key], `${lang}.${key}`);
    assert.ok(I18N[lang].settings_changes_count.includes('{n}'), `${lang} many has {n}`);
    assert.ok(I18N[lang].settings_changes_cluster.includes('{m}'), `${lang} cluster has {m}`);
  }
  assert.equal(I18N.en.settings_changes_one, '1 change');
  assert.equal(I18N.en.settings_changes_count, '{n} changes');
  assert.equal(I18N['zh-Hans'].settings_changes_count, '{n} 项改动');
  assert.equal(I18N['zh-Hans'].settings_restart_note, '保存后会重启后端，进行中的请求会中断');
  for (const lang of LANGS.slice(1)) {
    for (const key of ['settings_changes_count', 'settings_changes_cluster', 'settings_restart_note']) {
      assert.notEqual(I18N[lang][key], I18N.en[key], `${lang}.${key} translated`);
    }
  }
});

test('renamed_labels', () => {
  const I18N = loadI18N();
  const expected = {
    en: { max_sequences: 'Concurrent requests', max_models: 'Models loaded at once', model_ttl: 'Unload when idle',
      enable_cache: 'Prefix cache', hot_cache: 'Memory cache limit', cold_cache: 'SSD cache limit',
      m5_dflash2_profile: 'M5 DFlash2 optimization', settings_title: 'Settings', theme_mode: 'Appearance' },
    'zh-Hans': { max_sequences: '并发请求数', max_models: '同时加载的模型数', model_ttl: '空闲自动卸载',
      enable_cache: '前缀缓存', hot_cache: '内存缓存上限', cold_cache: 'SSD 缓存上限',
      m5_dflash2_profile: 'M5 DFlash2 优化', settings_title: '设置', theme_mode: '外观' },
  };
  for (const [lang, values] of Object.entries(expected)) {
    for (const [key, value] of Object.entries(values)) assert.equal(I18N[lang][key], value, `${lang}.${key}`);
  }
  for (const lang of LANGS) {
    for (const key of Object.keys(expected.en)) {
      assert.ok(I18N[lang][key], `${lang}.${key} exists`);
      if (lang !== 'en' && key !== 'settings_title' && key !== 'theme_mode') {
        assert.notEqual(I18N[lang][key], I18N.en[key], `${lang}.${key} is translated`);
      }
    }
    // The M5 help copy uses the new name too.
    for (const key of ['m5_dflash2_profile_help_title', 'm5_dflash2_profile_help_label']) {
      assert.ok(I18N[lang][key].includes(I18N[lang].m5_dflash2_profile),
        `${lang}.${key} follows the renamed feature`);
    }
  }
  assert.ok(I18N.en.m5_dflash2_profile_help_title.includes('M5 DFlash2 optimization'));
  assert.ok(I18N.en.m5_dflash2_profile_help_label.includes('M5 DFlash2 optimization'));
});

test('unused_i18n_keys_removed', () => {
  const I18N = loadI18N();
  for (const lang of LANGS) {
    for (const key of ['model_runtime', 'resource_mgmt', 'appearance', 'log_level_live_desc', 'settings_dirty',
      'settings_restart_pending']) {
      assert.ok(!(key in I18N[lang]), `${lang}.${key} removed`);
    }
  }
});

test('legacy_colour_variables_and_badge_live_are_gone_from_settings', () => {
  assert.ok(!html.includes('badge-live'), 'badge-live CSS removed');
  assert.ok(!settingsPage.includes('var(--accent)'), 'no --accent in settings');
  assert.ok(!settingsPage.includes('var(--text-secondary)'), 'no --text-secondary in settings');
  assert.ok(!settingsPage.includes('var(--orange'), 'no --orange in settings');
});

test('hot_cache_hint_is_not_overwritten_by_applyI18n', () => {
  assert.ok(!/id="cfg-hot-cache-auto-hint"[^>]*data-i18n=/.test(settingsPage), 'hint is rendered by updateAutoHint only');
});

test('settings_heading_can_take_focus', () => {
  assert.match(settingsPage, /<div class="page-title" id="settings-page-title" tabindex="-1"/);
});

test('renamed_copy_reads_correctly', () => {
  const I18N = loadI18N();
  for (const lang of LANGS) {
    assert.ok(!('badge_live' in I18N[lang]), `${lang}.badge_live removed`);
    for (const [key, value] of Object.entries(I18N[lang])) {
      assert.ok(!/最大序列|最大同時載入模型數|最大同时加载模型数|Max Sequences|Hot Cache|Enable Cache/.test(value), `${lang}.${key} has no old name`);
    }
  }
  assert.ok(!/최적화이|최적화은|최적화을/.test(JSON.stringify(I18N.ko)));
  assert.ok(!/取 并发请求数|取 並行請求數|は 同時リクエスト数/.test(JSON.stringify(I18N)));
  assert.ok(!/在前缀缓存的同时|在前綴快取的同時/.test(JSON.stringify(I18N)));
  assert.ok(!I18N.ja.hot_cache_help_body.includes('キャッシュ有効化'));
  assert.ok(!/hot cache/i.test(I18N.en.hot_cache_help_title));
  for (const lang of LANGS) assert.ok(!/最大同時|最大同时|Maximum concurrent|최대 동시/.test(I18N[lang].err_max_loaded_models_reached), lang);
  assert.ok(!/\n\n\n  \/\* ── Toggle switch/.test(html));
});

test('remaining_setting_labels_use_the_unified_wording', () => {
  // English labels use sentence case like the renamed ones; zh names the memory limits as "上限".
  for (const label of ['Network mode', 'LAN interface', 'LAN access credentials', 'Log level', 'Total memory limit',
    'Model memory limit', 'SSD cache directory', 'KV cache compression', 'Active KV cache offload', 'Neural Engine (ANE) acceleration']) {
    assert.ok(html.includes('"' + label + '"'), label);
  }
  for (const old of ['Network Mode', 'Log Level', 'Memory Limit (Total)', 'Memory Limit (Model Only)', 'SSD Cache Directory',
    'KV Cache Compression', 'Active KV Cache Offload', '内存限制（总量）', '内存限制（仅模型）', '記憶體限制（總量）', 'メモリ制限（合計）', '메모리 제한 (전체)']) {
    assert.ok(!html.includes(old), old);
  }
  assert.ok(html.includes('mem_limit_total: "总内存上限"') && html.includes('mem_limit_model: "模型内存上限"'));
  assert.ok(html.includes("'모델 메모리 상한'을") && html.includes("'전체 메모리 상한'을"));
});

test('ane_note_shares_the_description_line', () => {
  // The M5 requirement note is an inline part of the ANE description, still toggled by its own id.
  assert.match(html, /<div class="settings-row-desc"><span id="ane-acceleration-desc" data-i18n="ane_acceleration_desc">[^<]*<\/span><span class="ds-inline-note" id="ane-requires-m5-profile" data-i18n="ane_requires_m5_profile" hidden>/);
  assert.match(html, /#page-settings \.ds-inline-note \{ margin-left: 4px; \}/);
  assert.equal((html.match(/id="ane-requires-m5-profile"/g) || []).length, 1);
});

test('ane_is_a_sub_row_of_m5', () => {
  // ANE depends on the M5 profile: it follows M5 as an indented sub-row without a divider.
  assert.match(html, /<div class="settings-row ds-sub-row" id="ane-row">[\s\S]*?id="cfg-ane"/);
  assert.match(html, /#page-settings \.settings-row\.ds-sub-row \{ position: relative; border-top: none;/);
  assert.match(html, /#page-settings \.toggle\.is-forced-off input:checked \+ \.toggle-slider::before \{ transform: none; \}/);
  assert.match(html, /id="m5-unsupported-note" data-i18n="m5_unsupported_chip" hidden>/);
});

test('prefix_cache_and_lan_children_are_sub_rows', () => {
  // Rows that only apply under a parent setting are indented sub-rows of it.
  for (const id of ['hot-cache-row', 'cache-dir-row', 'cold-cache-row']) {
    assert.match(html, new RegExp('<div class="settings-row ds-sub-row" id="' + id + '">'), id);
  }
  for (const id of ['cfg-lan-host-row', 'cfg-lan-security-row']) {
    assert.match(html, new RegExp('<div class="settings-row ds-sub-row" id="' + id + '" style="display:none;">'), id);
  }
  assert.match(html, /id="cache-requires-prefix" data-i18n="cache_requires_prefix" hidden>/);
  assert.match(html, /#page-settings \.settings-row\.ds-sub-row::before \{/);
  const dict = vm.runInNewContext(html.slice(i18nStart, html.indexOf('  const LANG_MAP =')) + '\nI18N');
  for (const lang of ['en', 'zh-Hans', 'zh-Hant', 'ja', 'ko']) assert.ok(dict[lang].cache_requires_prefix, lang);
  assert.equal(dict['zh-Hans'].cache_requires_prefix, '需先开启前缀缓存。');
});

test('prefix_cache_has_a_short_description', () => {
  assert.match(html, /<span data-i18n="enable_cache">[^<]*<\/span> <span class="ds-restart-tag"[^>]*>[^<]*<\/span><\/div><div class="settings-row-desc" data-i18n="enable_cache_desc">/);
  const dict = vm.runInNewContext(html.slice(i18nStart, html.indexOf('  const LANG_MAP =')) + '\nI18N');
  for (const lang of ['en', 'zh-Hans', 'zh-Hant', 'ja', 'ko']) assert.ok(dict[lang].enable_cache_desc, lang);
  assert.notEqual(dict['zh-Hans'].enable_cache_desc, dict.en.enable_cache_desc);
});

test('ssd_cache_follows_the_directory', () => {
  // A blank SSD cache directory keeps the prefix cache memory-only; the SSD limit then says why it is off.
  assert.match(html, /id="ssd-requires-dir" data-i18n="ssd_requires_dir" hidden>/);
  assert.match(html, /data-i18n="ssd_cache_dir_desc">/);
  assert.match(html, /const ssdOn = enabled && cacheDirInput\.value\.trim\(\) !== '';/);
  assert.match(html, /if \(!cacheDirInput \|\| !cacheDirInput\.value\.trim\(\)\) return;/);
  const dict = vm.runInNewContext(html.slice(i18nStart, html.indexOf('  const LANG_MAP =')) + '\nI18N');
  for (const lang of ['en', 'zh-Hans', 'zh-Hant', 'ja', 'ko']) {
    assert.ok(dict[lang].ssd_cache_dir_desc && dict[lang].ssd_requires_dir, lang);
    assert.doesNotMatch(dict[lang].cold_cache_help_body, /turn off Prefix cache to disable it|如需禁用，请关闭/);
  }
});

test('memory_cache_help_is_short', () => {
  // The help keeps only what Auto does; the parent switch and restart are already shown on the page.
  const dict = vm.runInNewContext(html.slice(i18nStart, html.indexOf('  const LANG_MAP =')) + '\nI18N');
  for (const lang of ['en', 'zh-Hans', 'zh-Hant', 'ja', 'ko']) {
    const body = dict[lang].hot_cache_help_body;
    assert.match(body, /8 ?GB/, lang);
    assert.doesNotMatch(body, /restart|重启|重啟|再起動|재시작|Prefix cache switch|总开关|總開關|スイッチ|스위치/i, lang);
    assert.ok(body.length < (lang === 'en' ? 160 : 100), lang + ' length ' + body.length);
  }
});

test('ssd_cache_help_is_short', () => {
  // Like the memory cache help: what it does and how the maximum is set; the page already shows restart and on/off rules.
  const dict = vm.runInNewContext(html.slice(i18nStart, html.indexOf('  const LANG_MAP =')) + '\nI18N');
  for (const lang of ['en', 'zh-Hans', 'zh-Hant', 'ja', 'ko']) {
    const body = dict[lang].cold_cache_help_body;
    assert.match(body, /SSD/, lang);
    assert.doesNotMatch(body, /restart|重启|重啟|再起動|재시작/i, lang);
    assert.ok(body.length < (lang === 'en' ? 160 : 100), lang + ' length ' + body.length);
  }
  assert.match(html, /data-i18n="cold_cache_help_body">Keeps prefix cache data on local SSD/);
});

test('inactive_sub_rows_keep_help_tooltips_on_top', () => {
  // Opacity on the whole left column makes a stacking context, and the "?" tooltip then draws under the next row.
  assert.doesNotMatch(html, /\.ds-sub-row\.is-inactive > \.settings-row-left\s*\{[^}]*opacity/);
  assert.match(html, /\.is-inactive > \.settings-row-left > \.settings-row-label > :not\(\.profile-help\)/);
});

test('active_kv_offload_help_is_short_and_names_dflash2', () => {
  const dict = vm.runInNewContext(html.slice(i18nStart, html.indexOf('  const LANG_MAP =')) + '\nI18N');
  for (const lang of ['en', 'zh-Hans', 'zh-Hant', 'ja', 'ko']) {
    const d = dict[lang];
    assert.match(d.active_kv_offload_help_not_applicable, /DFlash2/, lang);
    assert.match(d.active_kv_offload_help_body, /SSD/, lang);
    assert.ok(d.active_kv_offload_help_body.length < (lang === 'en' ? 260 : 160), lang + ' length ' + d.active_kv_offload_help_body.length);
    assert.equal(d.active_kv_offload_help_trigger, undefined, lang);
    assert.equal(d.active_kv_offload_help_applicable, undefined, lang);
  }
});

test('active_kv_offload_description_is_short', () => {
  // The Experimental tag already says it is experimental; the "?" carries the details.
  const dict = vm.runInNewContext(html.slice(i18nStart, html.indexOf('  const LANG_MAP =')) + '\nI18N');
  for (const lang of ['en', 'zh-Hans', 'zh-Hant', 'ja', 'ko']) {
    const desc = dict[lang].active_kv_offload_desc;
    assert.match(desc, /SSD/, lang);
    assert.doesNotMatch(desc, /Experimental|实验|實驗|実験|실험|MAX CONTEXT TOKENS/, lang);
    assert.ok(desc.length < (lang === 'en' ? 90 : 50), lang + ' length ' + desc.length);
  }
});
