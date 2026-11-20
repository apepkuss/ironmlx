const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const html = fs.readFileSync(path.join(__dirname,
  '../../ironmlx-app/Sources/IronMLXAppCore/Resources/dashboard2.html'), 'utf8');
const blockStart = html.indexOf('  // ── Runtime log parsing ──');
const blockEnd = html.indexOf('  // ── End runtime log parsing ──');
const block = blockStart >= 0 && blockEnd > blockStart ? html.slice(blockStart, blockEnd) : '';

function escapeHTML(value) {
  return value.replace(/[&<>"']/g, c => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  })[c]);
}

function page(extra = {}) {
  const context = { escapeHTML, ...extra };
  vm.createContext(context);
  vm.runInContext(block, context);
  return context;
}

const INFO_LINE = '2026-11-17T10:18:27.505013Z  INFO ironmlx::cli::serve: ready';
const ERROR_LINE = '2026-11-17T10:18:28.100000Z ERROR ironmlx::cli::serve: boom';
const pad = n => String(n).padStart(2, '0');

test('strips_ansi_from_backend_line', () => {
  const p = page();
  const raw = '\x1b[2m2026-11-17T10:18:27.505013Z\x1b[0m \x1b[32m INFO\x1b[0m \x1b[2mironmlx::cli::serve\x1b[0m\x1b[2m:\x1b[0m ready';
  assert.equal(p.stripLogAnsi(raw), INFO_LINE);
});

test('parses_backend_and_app_lines', () => {
  const p = page();
  const [a, b, c, d] = p.parseLogRecords([
    INFO_LINE,
    '2026-11-17T10:18:17.504Z INFO ironmlx-app: Application did finish launching',
    '2026-11-17T10:18:17.504Z  WARN ironmlx-app: slow',
    '2026-11-17T10:18:17.504Z WARNING ironmlx-app: slower',
  ]);
  assert.equal(a.level, 'INFO');
  assert.equal(a.module, 'ironmlx::cli::serve');
  assert.equal(a.message, 'ready');
  assert.equal(a.time, '2026-11-17T10:18:27.505013Z');
  assert.equal(b.module, 'ironmlx-app');
  assert.equal(b.message, 'Application did finish launching');
  assert.equal(c.level, 'WARN');
  assert.equal(d.level, 'WARN');
});

test('continuation_lines_join_previous_record', () => {
  const p = page();
  const records = p.parseLogRecords([ERROR_LINE, 'stack line 1', 'stack line 2', INFO_LINE]);
  assert.equal(records.length, 2);
  assert.equal(records[0].lines.length, 3);
  assert.equal(records[0].level, 'ERROR');
  assert.equal(records[1].lines.length, 1);
});

test('first_unmatched_line_becomes_plain_record', () => {
  const p = page();
  const records = p.parseLogRecords(['partial tail', '', 'more tail', INFO_LINE]);
  assert.equal(records.length, 2);
  assert.equal(records[0].time, null);
  assert.equal(records[0].level, 'PLAIN');
  assert.equal(records[0].message, 'partial tail');
  assert.deepEqual(Array.from(records[0].lines), ['partial tail', 'more tail']);
  const sep = p.parseLogRecords(['===== IronMLX App started at 2026-11-17T10:18:17.503Z =====']);
  assert.equal(sep.length, 1);
});

test('banner_lines_start_their_own_plain_row', () => {
  const p = page();
  const records = p.parseLogRecords([
    ERROR_LINE, 'stack line 1',
    '===== IronMLX Backend started at 2026-11-20T04:17:17.028Z =====',
    'command: /Applications/IronMLX.app/Contents/Helpers/ironmlx serve',
    INFO_LINE,
  ]);
  assert.equal(records.length, 3);
  assert.equal(records[0].level, 'ERROR');
  assert.deepEqual(Array.from(records[0].lines), [ERROR_LINE.replace(/\x1b\[[0-9;]*[A-Za-z]/g, ''), 'stack line 1']);
  assert.equal(records[1].level, 'PLAIN');
  assert.equal(records[1].time, null);
  assert.equal(records[1].lines.length, 2, 'the command line stays with its banner');
  assert.equal(p.filterLogRecords(records, 'error', '').length, 1);
  assert.equal(p.filterLogRecords(records, 'warn', '').length, 1);
  const out = p.renderLogRecordsHTML(records.slice(1, 2));
  assert.ok(out.includes('ds-log-line--plain'));
  assert.ok(!out.includes('ds-log-line--error'));
  assert.ok(out.includes('<span class="ds-log-level"></span>'));
});

test('filters_by_level_and_search_per_record', () => {
  const p = page();
  const records = p.parseLogRecords([
    INFO_LINE,
    '2026-11-17T10:18:27.600000Z  WARN a::b: careful',
    ERROR_LINE, 'stack line 1', 'stack line 2',
  ]);
  assert.deepEqual(Array.from(p.filterLogRecords(records, 'warn', ''), r => r.level), ['WARN', 'ERROR']);
  assert.deepEqual(Array.from(p.filterLogRecords(records, 'error', ''), r => r.level), ['ERROR']);
  assert.equal(p.filterLogRecords(records, 'all', '').length, 3);
  const hit = p.filterLogRecords(records, 'all', '  STACK LINE 2 ');
  assert.equal(hit.length, 1);
  assert.equal(hit[0].lines.length, 3);
});

test('module_short_and_clock', () => {
  const p = page();
  assert.equal(p.logModuleShort('ironmlx_runtime::core::dflash2_actor'), 'dflash2_actor');
  assert.equal(p.logModuleShort('ironmlx-app'), 'ironmlx-app');
  const iso = '2026-11-17T10:18:27.505013Z';
  const d = new Date(iso);
  const clock = `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}.505`;
  assert.equal(p.formatLogClock(iso), clock);
  assert.equal(p.formatLogDateTime(iso),
    `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${clock}`);
  assert.equal(p.formatLogClock('bad'), 'bad');
});

test('render_escapes_and_marks_levels', () => {
  const p = page();
  const records = p.parseLogRecords([
    '2026-11-17T10:18:27.505013Z  INFO a::b: <b>bold</b> & "q"',
    ERROR_LINE, 'stack <i>line</i>',
  ]);
  const out = p.renderLogRecordsHTML(records);
  assert.ok(!out.includes('<b>'));
  assert.ok(!out.includes('<i>'));
  assert.ok(out.includes('&lt;b&gt;bold&lt;/b&gt; &amp; &quot;q&quot;'));
  assert.ok(out.includes('ds-log-line--info'));
  assert.ok(out.includes('ds-log-line--error'));
  assert.ok(out.includes('ds-log-line ds-log-line--error ds-log-cont'));
  assert.ok(out.includes('stack &lt;i&gt;line&lt;/i&gt;'));
  assert.ok(out.includes('title="ironmlx::cli::serve"'));
  assert.ok(out.includes('>serve<'));
});

test('render_leaves_time_empty_for_unmatched_first_line', () => {
  const p = page();
  const out = p.renderLogRecordsHTML(p.parseLogRecords(['partial tail']));
  assert.ok(out.includes('<div class="ds-log-line ds-log-line--plain">'));
  assert.ok(!out.includes('title='));
});

function filterPage() {
  const writes = [];
  const logOutput = {
    scrollHeight: 1000, clientHeight: 400, scrollTop: 0,
    set innerHTML(v) { writes.push(v); }, get innerHTML() { return writes[writes.length - 1] || ''; },
  };
  const logCount = { textContent: '' };
  const logTimestamp = { textContent: '' };
  const logSearchInput = { value: '' };
  const logViewState = { source: 'server', level: 'all', paused: false, lastRenderKey: null };
  const context = {
    logOutput, logCount, logTimestamp, logSearchInput, logViewState, logRawLines: [INFO_LINE, ERROR_LINE],
    I18N: { en: { log_count: '{shown}/{total}', log_no_entries: 'none', log_no_app_logs: 'no app logs', log_server_offline: 'offline', log_start_hint: 'start it', logs_updated_at: '{time}' } }, currentLang: 'en',
  };
  const p = page(context);
  vm.runInContext(html.slice(html.indexOf('  function formatLogTimestamp('),
    html.indexOf('  function refreshLogs()')), p);
  let text = '';
  Object.defineProperty(logOutput, 'textContent', { set(v) { text = v; }, get() { return text; } });
  return { p, writes, logOutput, logCount, logSearchInput, logViewState };
}

test('unchanged_text_does_not_rerender', () => {
  const f = filterPage();
  f.p.applyLogFilters();
  f.p.applyLogFilters();
  assert.equal(f.writes.length, 1);
  assert.equal(f.logCount.textContent, '2/2');
  f.logViewState.level = 'error';
  f.p.applyLogFilters();
  assert.equal(f.writes.length, 2);
  assert.equal(f.logCount.textContent, '1/2');
});

test('follows_bottom_only_when_near_bottom', () => {
  const f = filterPage();
  f.p.applyLogFilters();
  assert.equal(f.logOutput.scrollTop, 1000);
  f.logOutput.scrollTop = 600;
  f.logSearchInput.value = 'ready';
  f.p.applyLogFilters();
  assert.equal(f.logOutput.scrollTop, 1000);
  f.logOutput.scrollTop = 100;
  f.logSearchInput.value = 'boom';
  f.p.applyLogFilters();
  assert.equal(f.logOutput.scrollTop, 100);
});

test('refresh_timer_never_duplicates', () => {
  let active = 0, nextId = 1; const ids = new Set(); const intervals = [];
  const context = {
    logRefreshTimer: null, refreshLogs() {}, logViewState: { paused: false },
    setInterval(fn, ms) { intervals.push(ms); ids.add(nextId); active++; return nextId++; },
    clearInterval(id) { if (ids.delete(id)) active--; },
  };
  vm.createContext(context);
  vm.runInContext(html.slice(html.indexOf('  function startLogAutoRefresh('),
    html.indexOf('  function exportCurrentLog(')), context);
  context.startLogAutoRefresh(); context.startLogAutoRefresh(); context.startLogAutoRefresh();
  assert.equal(active, 1);
  assert.deepEqual(intervals, [5000, 5000, 5000]);
  context.stopLogAutoRefresh();
  assert.equal(active, 0);
  context.logViewState.paused = true;
  context.startLogAutoRefresh();
  assert.equal(active, 0);
});

test('range_params_for_presets', () => {
  const p = incidentPage();
  const now = Date.UTC(2026, 10, 20, 12, 0, 0);
  assert.deepEqual({ ...p.incidentRangeParams('all', now, null) }, {});
  assert.deepEqual({ ...p.incidentRangeParams('24h', now, null) },
    { from: new Date(now - 24 * 3600000).toISOString() });
  assert.deepEqual({ ...p.incidentRangeParams('7d', now, null) },
    { from: new Date(now - 7 * 24 * 3600000).toISOString() });
  assert.deepEqual({ ...p.incidentRangeParams('30d', now, null) },
    { from: new Date(now - 30 * 24 * 3600000).toISOString() });
  const custom = { from: '2026-11-01T00:00:00.000Z', to: '2026-11-02T00:00:00.000Z' };
  assert.deepEqual({ ...p.incidentRangeParams('custom', now, custom) }, custom);
});

test('invalid_custom_range_keeps_applied_export_path', () => {
  const start = html.indexOf('  function incidentFilterPath(');
  const end = html.indexOf('  function incidentStatusLabel(');
  assert.ok(start >= 0 && end > start);
  const helpers = incidentPage();
  const fetched = [];
  const context = {
    incidentRangeParams: helpers.incidentRangeParams,
    incidentTimeRange: 'custom',
    incidentFilterDateTimeValue: () => null,
    appliedIncidentExportPath: '/admin/api/incidents/export?status=recovered',
    appliedIncidentFilterIsAll: true,
    document: { getElementById: () => ({ value: '' }) },
    apiFetch: path => fetched.push(path),
    URLSearchParams,
    Date,
  };
  vm.createContext(context);
  vm.runInContext(html.slice(start, end), context);
  context.refreshIncidentHistory(true);
  assert.deepEqual(fetched, []);
  assert.equal(context.appliedIncidentExportPath, '/admin/api/incidents/export?status=recovered');
  context.incidentTimeRange = '7d';
  context.refreshIncidentHistory(true);
  assert.equal(fetched.length, 2);
  assert.match(fetched[0], /^\/admin\/api\/incidents\?from=/);
  assert.ok(!fetched[0].includes('model='));
});

test('applied_export_state_derives_from_response_path', () => {
  const start = html.indexOf('  function incidentFilterPath(');
  const end = html.indexOf('  function incidentStatusLabel(');
  const context = {
    appliedIncidentExportPath: '/admin/api/incidents/export',
    appliedIncidentFilterIsAll: true,
    URLSearchParams,
  };
  vm.createContext(context);
  vm.runInContext(html.slice(start, end), context);
  const p1 = '/admin/api/incidents?limit=20';
  const p2 = '/admin/api/incidents?status=recovered&limit=20';
  // P1, P2, P1 sent; responses arrive P1, P2, P1: each applies its own path.
  context.applyIncidentQueryPath(p1);
  assert.equal(context.appliedIncidentExportPath, '/admin/api/incidents/export');
  assert.equal(context.appliedIncidentFilterIsAll, true);
  context.applyIncidentQueryPath(p2);
  assert.equal(context.appliedIncidentExportPath, '/admin/api/incidents/export?status=recovered');
  assert.equal(context.appliedIncidentFilterIsAll, false);
  context.applyIncidentQueryPath(p1);
  assert.equal(context.appliedIncidentExportPath, '/admin/api/incidents/export');
  assert.equal(context.appliedIncidentFilterIsAll, true);
  context.applyIncidentQueryPath('/admin/api/incidents?from=2026-11-01T00%3A00%3A00.000Z&limit=20');
  assert.equal(context.appliedIncidentFilterIsAll, false);
});

test('no_top_level_function_declared_twice', () => {
  // The later declaration silently wins, so none are allowed (the file has no pre-existing duplicates).
  const allowed = new Set();
  const seen = new Map();
  for (const m of html.matchAll(/^ {2}(?:async )?function\s+([A-Za-z0-9_$]+)\s*\(/gm)) {
    seen.set(m[1], (seen.get(m[1]) || 0) + 1);
  }
  const dupes = [...seen].filter(([name, n]) => n > 1 && !allowed.has(name)).map(([name]) => name);
  assert.deepEqual(dupes, []);
});

function incidentPage(extra = {}) {
  const helpersStart = html.indexOf('  // ── Incident filter helpers ──');
  const helpersEnd = html.indexOf('  // ── End incident filter helpers ──');
  const rStart = html.indexOf('function renderIncidentHistory(');
  const rEnd = html.indexOf('function updateIncidentUnreadBadge(');
  assert.ok(helpersStart >= 0 && helpersEnd > helpersStart && rStart >= 0 && rEnd > rStart);
  const listEl = { innerHTML: '', classList: { toggle() {} } };
  const emptyEl = { style: {} };
  const context = {
    escapeHTML,
    currentLang: 'en',
    I18N: { en: { incident_status_attempts: '{status} · {count} times', incident_status_attempts_one: '{status} · {count} time', incident_type_backend_exit: 'Unexpected backend exit', incident_no_models: 'No models' } },
    document: { getElementById: id => (id === 'incident-list' ? listEl : emptyEl) },
    initializeIncidentDateDefaults() {},
    incidentStatusLabel: s => `S:${s}`,
    incidentReasonLabel: r => `R:${r}`,
    incidentDate: v => `D:${v}`,
    incidentSummaries: [],
    listEl,
    ...extra,
  };
  vm.createContext(context);
  vm.runInContext(html.slice(helpersStart, helpersEnd), context);
  vm.runInContext(html.slice(rStart, rEnd), context);
  return context;
}

test('status_tone_mapping', () => {
  const p = incidentPage();
  assert.equal(p.incidentStatusTone('recovered'), 'green');
  assert.equal(p.incidentStatusTone('recovering'), 'blue');
  assert.equal(p.incidentStatusTone('recovery_pending'), 'blue');
  assert.equal(p.incidentStatusTone('partially_recovered'), 'amber');
  assert.equal(p.incidentStatusTone('recovery_failed'), 'red');
  assert.equal(p.incidentStatusTone('automatic_recovery_stopped'), 'red');
  assert.equal(p.incidentStatusTone('whatever'), 'gray');
  assert.equal(p.incidentStatusTone(undefined), 'gray');
});

test('incident_rows_are_keyboard_expandable', () => {
  const p = incidentPage();
  p.renderIncidentHistory({ incidents: [{
    id: 'inc-1"x', occurred_at: '2026-11-19T22:41:07', affected_models: ['A<b>'],
    recovery_status: 'recovery_failed', recovery_attempt: 3, primary_failure_reason: 'memory_insufficient',
  }] });
  const out = p.listEl.innerHTML;
  assert.match(out, /role="button"/);
  assert.match(out, /tabindex="0"/);
  assert.match(out, /aria-expanded="false"/);
  assert.match(out, /ds-pill--red/);
  assert.ok(!out.includes('ds-dot'), 'status shows once, in the pill');
  assert.match(out, /id="incident-detail-inc-1&quot;x"/);
  assert.match(out, /data-incident-id="inc-1&quot;x"/);
  assert.match(out, /S:recovery_failed · 3 times/);
  assert.ok(!out.includes('A<b>'));
  assert.match(out, /A&lt;b&gt;/);
  assert.match(out, /11-19 22:41:07/);
});

test('short_time_format', () => {
  const p = incidentPage();
  assert.equal(p.incidentShortTime('2026-11-19T22:41:07'), '11-19 22:41:07');
  assert.equal(p.incidentShortTime(''), '—');
});

function detailPage(lang, dict) {
  const logBlock = block;
  const hs = html.indexOf('  // ── Incident filter helpers ──');
  const he = html.indexOf('  // ── End incident filter helpers ──');
  const ds = html.indexOf('function incidentDetailField(');
  const de = html.indexOf('function toggleIncidentFromEvent(');
  const el = { innerHTML: '', dataset: {} };
  const context = {
    escapeHTML, currentLang: lang, I18N: { [lang]: dict },
    document: { getElementById: () => el },
    incidentStatusLabel: s => s, incidentReasonLabel: r => `R:${r}`, incidentActionLabel: a => `A:${a}`,
    incidentStepLabel: a => `S:${a}`, incidentDate: v => v, el,
  };
  vm.createContext(context);
  vm.runInContext(logBlock, context);
  vm.runInContext(html.slice(hs, he), context);
  vm.runInContext(html.slice(ds, de), context);
  return context;
}

test('incident_detail_separator_is_localized', () => {
  const inc = { id: 'a', failures: [{ model: 'M', reason: 'x', action: 'retry' }], recovery_steps: [{ action: 'go', occurred_at: '2026-11-19T22:41:09' }] };
  const en = detailPage('en', { incident_label_value: '{label}: {value}', incident_reason: 'Reason', incident_suggestion: 'Suggestion', incident_recovery_steps: 'Steps' });
  en.renderIncidentDetail(inc);
  assert.match(en.el.innerHTML, /Reason: R:x<\/div>/);
  assert.match(en.el.innerHTML, /M · R:x · Suggestion: A:retry/);
  assert.match(en.el.innerHTML, /Steps: S:go 22:41:09/);
  assert.ok(!en.el.innerHTML.includes('：'));
  const zh = detailPage('zh', { incident_label_value: '{label}：{value}', incident_reason: '原因', incident_suggestion: '建议', incident_recovery_steps: '恢复过程' });
  zh.renderIncidentDetail(inc);
  assert.match(zh.el.innerHTML, /原因：R:x<\/div>/);
  assert.match(zh.el.innerHTML, /M · R:x · 建议：A:retry/);
});

test('incident_detail_shows_reason_without_failures', () => {
  const p = detailPage('en', { incident_label_value: '{label}: {value}', incident_reason: 'Reason' });
  p.renderIncidentDetail({ id: 'a', failures: [], primary_failure_reason: 'memory_insufficient' });
  assert.match(p.el.innerHTML, /Reason: R:memory_insufficient/);
  assert.ok(!p.el.innerHTML.includes('Suggestion'));
});

test('narrow_media_query_keeps_status_pill', () => {
  const m = html.match(/@media \(max-width: 900px\) \{\s*\.ds-incident-row[^}]*\}[^}]*\}/);
  assert.ok(m, 'narrow incident rule present');
  assert.ok(!/ds-pill[^}]*display:\s*none/.test(m[0]));
});

test('incident_i18n_keys_exist_once_per_language_with_real_tables', () => {
  const start = html.indexOf('  const I18N =');
  const src = html.slice(start, html.indexOf('\n  };', start) + 5);
  const ctx = {};
  vm.createContext(ctx);
  vm.runInContext(src + '\nthis.I18N = I18N;', ctx);
  const separators = { en: ': ', 'zh-Hans': '：', 'zh-Hant': '：', ja: '：', ko: ': ' };
  const keys = ['incident_label_value', 'incident_status_attempts', 'incident_process_section',
    'incident_suggestion', 'incident_recovery_steps', 'incident_more', 'incident_log_tail'];
  const langKeys = { en: 'en', 'zh-Hans': '"zh-Hans"', 'zh-Hant': '"zh-Hant"', ja: 'ja', ko: 'ko' };
  for (const lang of Object.keys(separators)) {
    const dict = ctx.I18N[lang];
    assert.ok(dict, lang);
    for (const k of keys) assert.equal(typeof dict[k], 'string', `${lang}.${k}`);
    assert.equal(dict.incident_label_value, `{label}${separators[lang]}{value}`);
  }
  // Each key must be declared once per language dict (duplicates silently override).
  for (const k of keys) {
    const n = (src.match(new RegExp(`\\b${k}: "`, 'g')) || []).length;
    assert.equal(n, 5, `${k} declared ${n} times`);
  }
  const zh = detailPage('zh-Hans', ctx.I18N['zh-Hans']);
  zh.renderIncidentDetail({ id: 'a', failures: [], primary_failure_reason: 'm' });
  assert.match(zh.el.innerHTML, /：R:m/);
});

function realI18N() {
  const start = html.indexOf('  const I18N =');
  const src = html.slice(start, html.indexOf('\n  };', start) + 5);
  const ctx = {};
  vm.createContext(ctx);
  vm.runInContext(src + '\nthis.I18N = I18N;', ctx);
  return { src, I18N: ctx.I18N };
}

test('logs_page_strings_exist_in_every_language', () => {
  const { src, I18N } = realI18N();
  const keys = ['logs_source_backend', 'logs_source_app', 'logs_level_all', 'logs_level_warn',
    'logs_level_error', 'logs_search_placeholder', 'logs_refreshing', 'logs_paused', 'logs_pause',
    'logs_resume', 'logs_export', 'log_count', 'logs_updated_at',
    'incident_range_all', 'incident_range_24h', 'incident_range_7d', 'incident_range_30d',
    'incident_range_custom', 'incident_status_all', 'incident_reason_all', 'incident_export_short',
    'incident_more', 'incident_label_value', 'incident_status_attempts', 'incident_process_section',
    'incident_suggestion', 'incident_recovery_steps', 'incident_status_attempts_one'];
  for (const lang of ['en', 'zh-Hans', 'zh-Hant', 'ja', 'ko']) {
    for (const k of keys) {
      assert.equal(typeof I18N[lang][k], 'string', `${lang}.${k}`);
      assert.notEqual(I18N[lang][k], '', `${lang}.${k} empty`);
    }
  }
  for (const k of keys) {
    const n = (src.match(new RegExp(`\\b${k}: "`, 'g')) || []).length;
    assert.equal(n, 5, `${k} declared ${n} times`);
  }
});

test('obsolete_log_strings_are_removed', () => {
  const obsolete = ['log_lines', 'log_interval', 'log_interval_hint', 'auto_scroll', 'auto_refresh',
    'log_viewer', 'incident_query', 'incident_model', 'incident_model_placeholder',
    'incident_filters', 'incident_export_unapplied_message', 'incident_export_current_title',
    'incident_export_current_message', 'incident_export_current_confirm', 'refresh'];
  for (const k of obsolete) {
    const pattern = k === 'refresh'
      ? /\brefresh: "|data-i18n(?:-[a-z-]+)?="refresh"|\bt\('refresh'|dict\.refresh/
      : new RegExp(`\\b${k}\\b`);
    assert.ok(!pattern.test(html), `${k} still present`);
  }
});

test('app_logs_empty_shows_no_app_logs_not_offline', () => {
  const f = filterPage();
  f.logViewState.source = 'app';
  f.p.receiveAppLogs('');
  assert.equal(f.logOutput.textContent, 'no app logs');
  assert.equal(f.logCount.textContent, '0/0');
  assert.equal(f.logViewState.placeholder, 'empty-app');
});

test('server_logs_empty_shows_offline_hint', () => {
  const f = filterPage();
  f.p.receiveLogText('', 'server');
  assert.equal(f.logOutput.textContent, 'offline\nstart it');
  f.p.receiveLogText('null', 'server');
  assert.equal(f.logOutput.textContent, 'offline\nstart it');
});

test('filters_keep_placeholder_until_real_data', () => {
  const f = filterPage();
  f.p.receiveLogText('', 'server');
  const before = f.writes.length;
  f.logViewState.level = 'error';
  f.logSearchInput.value = 'x';
  f.p.applyLogFilters();
  assert.equal(f.writes.length, before);
  assert.equal(f.logOutput.textContent, 'offline\nstart it');
  f.p.receiveLogText(INFO_LINE + '\n' + ERROR_LINE, 'server');
  assert.equal(f.logViewState.placeholder, null);
});

test('english_attempts_singular_and_plural', () => {
  const p = incidentPage();
  const item = n => ({ id: 'a', occurred_at: '2026-11-19T22:41:07', affected_models: [], recovery_status: 'recovered', recovery_attempt: n });
  p.renderIncidentHistory({ incidents: [item(1)] });
  assert.match(p.listEl.innerHTML, /S:recovered · 1 time</);
  p.renderIncidentHistory({ incidents: [item(2)] });
  assert.match(p.listEl.innerHTML, /S:recovered · 2 times</);
  p.renderIncidentHistory({ incidents: [item(0)] });
  assert.match(p.listEl.innerHTML, /S:recovered · 0 times</);
});

test('incident_detail_shows_code_and_message_line', () => {
  const dict = { incident_label_value: '{label}: {value}', incident_reason: 'Reason', incident_suggestion: 'Suggestion' };
  const p = detailPage('en', dict);
  p.renderIncidentDetail({ id: 'a', primary_failure_reason: 'm', failures: [
    { model: 'M', reason: 'm', action: 'retry', code: 'E<1>', message: 'bad & worse' },
    { model: 'N', reason: 'm', action: 'retry', message: 'only msg' },
  ] });
  const out = p.el.innerHTML;
  assert.match(out, /M · R:m · Suggestion: A:retry<\/div><div class="ds-incident-line ds-incident-line--sub">E&lt;1&gt; · bad &amp; worse<\/div>/);
  assert.match(out, /ds-incident-line--sub">only msg</);
  assert.match(out, /<div class="ds-incident-line">Reason: R:m<\/div>/);
  assert.ok(!/Reason: R:m · Suggestion/.test(out));
});

test('incident_log_tail_strips_ansi', () => {
  const p = detailPage('en', {});
  p.renderIncidentDetail({ id: 'a', failures: [], log_tail: '\u001b[2m2026\u001b[0m \u001b[32m INFO\u001b[0m x' });
  assert.ok(p.el.innerHTML.includes('2026  INFO x'));
  assert.ok(!p.el.innerHTML.includes('\u001b'));
});

function customRangeHarness(oldest) {
  const start = html.indexOf('  function setIncidentTimeValidity(');
  const end = html.indexOf('  function incidentFilterDateTimeValue(');
  assert.ok(start >= 0 && end > start);
  const els = {};
  const mk = (id, value) => (els[id] = {
    value, dataset: {}, setCustomValidity() {}, addEventListener() {},
    closest: () => ({ classList: { toggle() {} } }),
  });
  mk('incident-filter-from-date', ''); mk('incident-filter-to-date', '');
  mk('incident-filter-from-time', '00:00'); mk('incident-filter-to-time', '23:59');
  const otherEl = { hidden: true };
  const refreshes = [];
  const context = {
    t: (k, d) => d, refreshes,
    document: { getElementById: id => els[id] || otherEl, querySelectorAll: () => [] },
  };
  vm.createContext(context);
  vm.runInContext("let incidentTimeRange = 'all'; let incidentOldestRetainedAt = null;"
    + "const incidentDateFilterTouched = { from: false, to: false };"
    + "function refreshIncidentHistory() { refreshes.push(incidentTimeRange); }\n"
    + html.slice(start, end), context);
  context.initializeIncidentDateDefaults({ oldest_retained_occurred_at: oldest });
  return { context, els, refreshes };
}

test('custom_range_defaults_computed_on_switch_without_duplicate_blur_query', () => {
  const { context, els, refreshes } = customRangeHarness('2026-11-10T08:00:00');
  assert.equal(els['incident-filter-from-date'].value, '');
  context.selectIncidentRange('custom');
  assert.equal(els['incident-filter-from-date'].value, '2026-11-10');
  assert.equal(els['incident-filter-to-date'].value, context.incidentDateInputValue(null));
  assert.equal(els['incident-filter-from-time'].value, '00:00');
  assert.equal(els['incident-filter-to-time'].value, '23:59');
  assert.equal(refreshes.length, 1);
  context.commitIncidentDateInput(els['incident-filter-from-date']);
  context.commitIncidentDateInput(els['incident-filter-to-date']);
  context.commitIncidentTimeInput(els['incident-filter-from-time']);
  context.commitIncidentTimeInput(els['incident-filter-to-time']);
  assert.equal(refreshes.length, 1);
  els['incident-filter-from-date'].value = '2026-11-12';
  context.commitIncidentDateInput(els['incident-filter-from-date']);
  assert.equal(refreshes.length, 2);
});

test('custom_range_without_retained_incidents_starts_today', () => {
  const { context, els } = customRangeHarness(null);
  context.selectIncidentRange('custom');
  assert.equal(els['incident-filter-from-date'].value, context.incidentDateInputValue(null));
});

test('stale_incident_list_response_is_ignored', () => {
  assert.match(html, /\} else if \(path === lastIncidentListPath\) \{\s*renderIncidentHistory\(result\);/);
  const start = html.indexOf('  function refreshIncidentHistory(');
  const end = html.indexOf('  function incidentStatusLabel(');
  const fetched = [];
  const context = {
    incidentFilterPath: e => e + '?limit=20&n=' + (fetched.length + 1), apiFetch: p => fetched.push(p),
  };
  vm.createContext(context);
  vm.runInContext('let lastIncidentListPath = null;\n' + html.slice(start, end)
    + '\nthis.last = () => lastIncidentListPath;', context);
  context.refreshIncidentHistory(true);
  const first = context.last();
  context.refreshIncidentHistory(true);
  assert.notEqual(context.last(), first);
  assert.equal(context.last(), fetched[2]);
  assert.ok(fetched[3].includes('unread=1'));
});
