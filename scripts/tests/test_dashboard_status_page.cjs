const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const html = fs.readFileSync(path.join(__dirname,
  '../../ironmlx-app/Sources/IronMLXAppCore/Resources/dashboard2.html'), 'utf8');
const blockStart = html.indexOf('  // ── Runtime status page ──');
const blockEnd = html.indexOf('  // ── End runtime status page ──');
const block = blockStart >= 0 && blockEnd > blockStart ? html.slice(blockStart, blockEnd) : '';
const i18nStart = html.indexOf('  const I18N =');
const translations = html.slice(i18nStart, html.indexOf('\n  };', i18nStart) + 5);

function page(lang = 'en', extra = {}) {
  const context = Object.assign({ currentLang: lang }, extra);
  vm.createContext(context);
  vm.runInContext(translations + '\n' + block, context);
  return context;
}

// The page's `const` state is not a property of the context; read it through the script scope.
function pageState(p) {
  return vm.runInContext('statusPageState', p);
}

function fakeElement(extra = {}) {
  const classes = new Set(extra.classes || []);
  return Object.assign({
    dataset: {},
    hidden: false,
    textContent: '',
    innerHTML: '',
    classList: {
      add: name => classes.add(name),
      remove: name => classes.delete(name),
      contains: name => classes.has(name),
    },
    classes,
  }, extra);
}

function snapshot(overrides = {}) {
  return Object.assign({
    schema_version: 1,
    unix_ms: 1763280000000,
    backend: { version: '0.5.0', started_unix_ms: 1763268480000, device_name: 'Apple M5 Max' },
    health: { status: 'healthy', degraded_reasons: [] },
    activity: { state: 'idle', active_requests: 0, prefilling_requests: 0, queued_requests: 0 },
    performance: {
      live_decode_tokens_per_second: null, prefill_tokens_per_second: null,
      decode_tokens_per_second: null, ttft_ms: null, window_seconds: 60,
    },
    memory: {
      pressure_level: 'normal', process_bytes: 0, ceiling_bytes: 0,
      kv_cache_bytes: 0, mlx_cache_bytes: 0, telemetry_degraded: false,
    },
    prefix_cache: { hit_tokens: 0, eligible_tokens: 0 },
    models: [],
  }, overrides);
}

function point(t, outputTokens = 0, extra = {}) {
  return Object.assign({
    t, decode_tps: outputTokens / 5, prefill_tps: null, ttft_ms: null,
    output_tokens: outputTokens, first_tokens: 0, requests_started: 0, requests_finished: 0,
    active_max: 0, queued_max: 0, process_bytes: 100, kv_cache_bytes: 10,
    mlx_cache_bytes: 5, ceiling_bytes: 1000,
  }, extra);
}

test('the status page block is marked for slicing', () => {
  assert.ok(block.length > 0, 'runtime status page block markers are present');
});

test('status indicator stays blue while starting', () => {
  const p = page();
  const indicator = p.statusIndicator({ state: 'starting', connected: false }, null);
  assert.equal(indicator.tone, 'blue');
  assert.equal(indicator.labelKey, 'status_starting');
  assert.equal(p.statusIndicator({ state: 'recovering', connected: false }, null).tone, 'blue');
});

test('reconnecting keeps last data', () => {
  const p = page();
  const snap = snapshot({ activity: { state: 'generating', active_requests: 1, queued_requests: 0 } });
  const indicator = p.statusIndicator({ state: 'running', connected: false }, snap);
  assert.equal(indicator.tone, 'gray');
  assert.equal(indicator.labelKey, 'status_reconnecting');
  // The cards keep the last snapshot while the stream reconnects, and drop it once stopped.
  p.onRuntimeBackendState({ state: 'running', connected: true });
  p.onRuntimeMetrics(snap);
  p.onRuntimeBackendState({ state: 'running', connected: false });
  assert.equal(p.statusLiveSnapshot(), null);
  assert.equal(p.statusDisplaySnapshot(), snap);
  p.onRuntimeBackendState({ state: 'stopped', connected: false });
  assert.equal(p.statusDisplaySnapshot(), null);
});

test('stopped is red', () => {
  const p = page();
  assert.equal(p.statusIndicator({ state: 'stopped', connected: false }, null).tone, 'red');
  assert.equal(p.statusIndicator({ state: 'stopped', connected: false }, null).labelKey, 'status_stopped');
  assert.equal(p.statusIndicator({ state: 'failed', connected: false }, snapshot()).tone, 'red');
  assert.equal(p.statusIndicator({ state: 'failed', connected: false }, null).labelKey, 'status_failed');
});

test('model_failed is red', () => {
  const p = page();
  const snap = snapshot({ health: { status: 'degraded', degraded_reasons: ['model_failed'] } });
  assert.equal(p.statusIndicator({ state: 'running', connected: true }, snap).tone, 'red');
  const emergency = snapshot({ health: { status: 'degraded', degraded_reasons: ['process_memory_emergency'] } });
  assert.equal(p.statusIndicator({ state: 'running', connected: true }, emergency).tone, 'red');
  const down = snapshot({ health: { status: 'down', degraded_reasons: [] } });
  assert.equal(p.statusIndicator({ state: 'running', connected: true }, down).tone, 'red');
});

test('generating is green', () => {
  const p = page();
  const generating = snapshot({ activity: { state: 'generating', active_requests: 1, queued_requests: 0 } });
  assert.deepEqual({ ...p.statusIndicator({ state: 'running', connected: true }, generating) },
    { tone: 'green', labelKey: 'status_generating' });
  const prefilling = snapshot({ activity: { state: 'prefilling', active_requests: 1, queued_requests: 0 } });
  assert.equal(p.statusIndicator({ state: 'running', connected: true }, prefilling).labelKey, 'status_prefilling');
  const loading = snapshot({ activity: { state: 'loading', active_requests: 0, queued_requests: 0 } });
  assert.equal(p.statusIndicator({ state: 'running', connected: true }, loading).tone, 'blue');
  assert.equal(p.statusIndicator({ state: 'running', connected: true }, snapshot()).tone, 'gray');
  assert.equal(p.statusIndicator({ state: 'running', connected: true }, snapshot()).labelKey, 'status_idle');
});

test('degraded is amber', () => {
  const p = page();
  assert.equal(p.statusIndicator({ state: 'degraded', connected: true }, snapshot()).tone, 'amber');
  const snap = snapshot({ health: { status: 'degraded', degraded_reasons: ['scheduler_queue_high'] } });
  const indicator = p.statusIndicator({ state: 'running', connected: true }, snap);
  assert.equal(indicator.tone, 'amber');
  assert.equal(indicator.labelKey, 'status_degraded');
  assert.equal(p.statusIndicator({ state: 'stopping', connected: false }, null).tone, 'gray');
});

test('tiles use live decode while generating', () => {
  const p = page();
  const generating = p.statusTiles(snapshot({
    activity: { state: 'generating', active_requests: 2, queued_requests: 1 },
    performance: {
      live_decode_tokens_per_second: 42.84, prefill_tokens_per_second: 1840.4,
      decode_tokens_per_second: 44.1, ttft_ms: 312.4, window_seconds: 60,
    },
  }), 'en');
  assert.equal(generating.decode, '42.8');
  assert.equal(generating.prefill, '1,840');
  assert.equal(generating.ttft, '312');
  assert.equal(generating.requests, '2 active · 1 queued');
  const idle = p.statusTiles(snapshot({
    performance: {
      live_decode_tokens_per_second: 3.2, prefill_tokens_per_second: null,
      decode_tokens_per_second: 44.06, ttft_ms: null, window_seconds: 60,
    },
  }), 'en');
  assert.equal(idle.decode, '44.1');
  assert.equal(idle.prefill, '—');
  assert.equal(idle.ttft, '—');
  const empty = p.statusTiles(null, 'zh-Hans');
  assert.deepEqual([empty.decode, empty.prefill, empty.ttft, empty.requests], ['—', '—', '—', '—']);
  const zh = p.statusTiles(snapshot({ activity: { state: 'idle', active_requests: 0, queued_requests: 3 } }), 'zh-Hans');
  assert.equal(zh.requests, '0 进行中 · 3 排队');
});

test('gap over two buckets breaks the line', () => {
  const p = page();
  const series = p.trendSeries([point(0, 50), point(5000, 50), point(20000, 50)], 'throughput', 5000);
  assert.equal(series.decode.length, 4);
  assert.equal(series.decode[2], null);
  assert.deepEqual({ ...series.decode[1] }, { t: 5000, v: 10 });
  assert.deepEqual({ ...series.decode[3] }, { t: 20000, v: 10 });
  assert.equal(series.prefill[2], null);
  const continuous = p.trendSeries([point(0), point(5000), point(15000)], 'throughput', 5000);
  assert.ok(continuous.decode.every(entry => entry !== null), 'a gap of exactly two buckets stays joined');
  const memory = p.trendSeries([point(0), point(30000)], 'memory', 5000);
  assert.deepEqual(Object.keys(memory).sort(), ['cache', 'ceiling', 'kv', 'process']);
  assert.equal(memory.process[1], null);
  assert.deepEqual({ ...memory.kv[0] }, { t: 0, v: 10 });
  assert.deepEqual({ ...memory.ceiling[2] }, { t: 30000, v: 1000 });
});

test('history gap triggers refetch', () => {
  const p = page();
  const state = p.createHistoryState();
  assert.equal(p.mergeHistoryPoint(state, point(0)).refetch, false);
  assert.equal(p.mergeHistoryPoint(state, point(5000)).refetch, false);
  assert.equal(p.mergeHistoryPoint(state, point(15000)).refetch, true);
  const next = p.createHistoryState();
  p.mergeHistoryPoint(next, point(0));
  p.mergeHistoryPoint(next, point(5000));
  assert.equal(p.mergeHistoryPoint(next, point(10000)).refetch, false);
  assert.deepEqual(Array.from(next.points['5s'], item => item.t), [0, 5000, 10000]);
  assert.equal(p.mergeHistoryPoint(next, point(10000)).refetch, false, 'a repeated point is ignored');
  assert.equal(next.points['5s'].length, 3);
});

test('local rollup matches backend rule', () => {
  const p = page();
  const state = p.createHistoryState();
  for (let i = 0; i < 6; i++) {
    p.mergeHistoryPoint(state, point(i * 5000, 50, i === 2
      ? { first_tokens: 2, ttft_ms: 250, prefill_tps: 2000, process_bytes: 400 }
      : {}));
  }
  assert.equal(state.points['30s'].length, 1);
  const coarse = state.points['30s'][0];
  assert.equal(coarse.t, 0);
  assert.equal(coarse.decode_tps, 10);
  assert.equal(coarse.output_tokens, 300);
  assert.equal(coarse.first_tokens, 2);
  assert.equal(coarse.ttft_ms, 250);
  assert.equal(coarse.prefill_tps, 2000);
  assert.equal(coarse.process_bytes, 400);
  // A new window closes the open one even when slots are missing.
  const sparse = p.createHistoryState();
  p.mergeHistoryPoint(sparse, point(5000, 60));
  p.mergeHistoryPoint(sparse, point(30000, 10));
  assert.equal(sparse.points['30s'].length, 1);
  assert.equal(sparse.points['30s'][0].decode_tps, 2);
  // A fetched history resumes the open window from its fine points.
  const fetched = p.historyStateFromResponse({
    schema_version: 1, bucket_ms: { '5s': 5000, '30s': 30000, '5m': 300000 },
    points: { '5s': [point(30000, 30), point(35000, 30)], '30s': [], '5m': [] },
  });
  for (let t = 40000; t < 60000; t += 5000) p.mergeHistoryPoint(fetched, point(t, 30));
  assert.equal(fetched.points['30s'].length, 1);
  assert.equal(fetched.points['30s'][0].output_tokens, 180);
});

test('recent request merge and gap', () => {
  const p = page();
  const list = [];
  assert.equal(p.mergeRecentRequest(list, { id: 5 }).refetch, false);
  assert.equal(p.mergeRecentRequest(list, { id: 6 }).refetch, false);
  assert.deepEqual(Array.from(list, item => item.id), [6, 5]);
  assert.equal(p.mergeRecentRequest(list, { id: 6 }).refetch, false, 'a repeated request is ignored');
  assert.equal(list.length, 2);
  assert.equal(p.mergeRecentRequest(list, { id: 8 }).refetch, true);
  assert.equal(list[0].id, 8);
  const full = [];
  for (let id = 1; id <= 105; id++) p.mergeRecentRequest(full, { id });
  assert.equal(full.length, 100);
  assert.equal(full[0].id, 105);
  assert.equal(full[99].id, 6);
});

test('aborted row is marked', () => {
  const finished = new Date(2026, 10, 17, 10, 42, 7).getTime();
  const request = {
    id: 1, finished_unix_ms: finished, model: 'mlx-community/Qwen3.6-35B-A3B-4bit',
    endpoint: '/v1/chat/completions', input_tokens: 1832, output_tokens: 412,
    ttft_ms: 298.4, decode_tps: 44.04, duration_ms: 9500, outcome: 'completed',
  };
  const en = page().recentRequestRow(request, 'en');
  assert.equal(en.time, '10:42:07');
  assert.equal(en.endpoint, '/v1/chat/completions');
  assert.equal(en.model, 'Qwen3.6-35B-A3B-4bit');
  assert.equal(en.tokens, '1,832→412');
  assert.equal(en.ttft, '298 ms');
  assert.equal(en.speed, '44.0 tok/s');
  assert.equal(en.aborted, false);
  const aborted = page().recentRequestRow(Object.assign({}, request,
    { outcome: 'aborted', ttft_ms: null, decode_tps: null, endpoint: null }), 'zh-Hans');
  assert.equal(aborted.aborted, true);
  assert.equal(aborted.outcomeLabel, '中断');
  assert.equal(aborted.ttft, '—');
  assert.equal(aborted.speed, '—');
  assert.equal(aborted.endpoint, '—');
});

test('idle unload text rounds up and says soon', () => {
  const p = page();
  const now = 1763280000000;
  assert.equal(p.idleUnloadText({ idle_unload_unix_ms: null }, now, 'zh-Hans'), null);
  assert.equal(p.idleUnloadText({}, now, 'en'), null);
  assert.equal(p.idleUnloadText({ idle_unload_unix_ms: now + 23 * 60000 + 1000 }, now, 'zh-Hans'),
    '空闲，24 分钟后卸载');
  assert.equal(p.idleUnloadText({ idle_unload_unix_ms: now + 60000 }, now, 'en'), 'Idle, unloads in 1 min');
  assert.equal(p.idleUnloadText({ idle_unload_unix_ms: now }, now, 'zh-Hans'), '即将卸载');
  assert.equal(p.idleUnloadText({ idle_unload_unix_ms: now - 5000 }, now, 'en'), 'Unloading soon');
});

test('memory segments clamp at zero', () => {
  const p = page();
  const clamped = p.memorySegments({ process_bytes: 10, kv_cache_bytes: 6, mlx_cache_bytes: 6, ceiling_bytes: 100 });
  assert.equal(clamped.model, 0);
  assert.equal(clamped.kv, 0.06);
  assert.equal(clamped.cache, 0.06);
  const normal = p.memorySegments({ process_bytes: 58, kv_cache_bytes: 10, mlx_cache_bytes: 8, ceiling_bytes: 100 });
  assert.equal(normal.model, 0.4);
  assert.equal(p.memorySegments(null), null);
  const over = p.memorySegments({ process_bytes: 300, kv_cache_bytes: 100, mlx_cache_bytes: 100, ceiling_bytes: 100 });
  assert.ok(over.model + over.kv + over.cache <= 1 + 1e-9, 'segments never overflow the bar');
});

test('prefix hit rate is null without eligible tokens', () => {
  const p = page();
  assert.equal(p.prefixHitRate({ hit_tokens: 0, eligible_tokens: 0 }), null);
  assert.equal(p.prefixHitRate(null), null);
  assert.equal(p.prefixHitRate({ hit_tokens: 120345, eligible_tokens: 138200 }), 87);
});

test('all five languages have every status key', () => {
  const I18N = vm.runInNewContext(`${translations}\nI18N`, {});
  const keys = [
    'status_generating', 'status_prefilling', 'status_idle', 'status_loading', 'status_starting',
    'status_degraded', 'status_stopped', 'status_failed', 'status_reconnecting',
    'status_tile_decode', 'status_tile_prefill', 'status_tile_ttft', 'status_tile_requests',
    'status_requests_value', 'status_trend', 'status_trend_throughput', 'status_trend_memory',
    'status_models', 'status_memory', 'status_memory_model', 'status_memory_kv', 'status_memory_cache',
    'status_pressure', 'status_prefix_hit', 'status_recent', 'status_view_logs', 'status_aborted',
    'status_idle_unload', 'status_unload_soon', 'status_retry', 'status_unload', 'status_no_models',
    'status_no_requests', 'status_last_updated', 'status_runtime_details', 'status_backend_stopped',
  ];
  for (const lang of ['en', 'zh-Hans', 'zh-Hant', 'ja', 'ko']) {
    for (const key of keys) {
      assert.equal(typeof I18N[lang][key], 'string', `${lang}.${key}`);
      assert.ok(I18N[lang][key].trim().length > 0, `${lang}.${key} is empty`);
    }
  }
  assert.equal(I18N['zh-Hans'].status_tile_ttft, '首字延迟');
  assert.equal(I18N.en.status_requests_value, '{active} active · {queued} queued');
});

test('idle countdown advances on the status poll tick', () => {
  const statusPage = fakeElement({ classes: ['page', 'active'] });
  const p = page('zh-Hans', { document: { getElementById: id => (id === 'page-status' ? statusPage : null) } });
  let renders = 0;
  p.renderStatusModels = () => { renders += 1; };
  p.statusPageTick();
  assert.equal(renders, 1, 'the visible status page re-renders its model rows');
  statusPage.classList.remove('active');
  p.statusPageTick();
  assert.equal(renders, 1, 'a hidden status page is left alone');
  p.renderStatusModels = () => { throw new Error('boom'); };
  statusPage.classList.add('active');
  assert.doesNotThrow(() => p.statusPageTick());
  // Only the text moves with the clock, so the row is updated in place.
  const now = 1763280000000;
  const model = { id: 'org/Model', state: 'loaded', weight_bytes: 1073741824, idle_unload_unix_ms: now + 5 * 60000 };
  const before = p.statusModelView(model, 'zh-Hans', now);
  const later = p.statusModelView(model, 'zh-Hans', now + 2 * 60000);
  assert.match(before.detail, /空闲，5 分钟后卸载/);
  assert.match(later.detail, /空闲，3 分钟后卸载/);
  assert.equal(p.statusModelStructure(later), p.statusModelStructure(before));
});

test('metrics events redraw the trend only when it goes live or offline', () => {
  const p = page();
  let trend = 0;
  p.renderStatusTrend = () => {
    trend += 1;
    pageState(p).trendLive = !!p.statusLiveSnapshot();
  };
  ['renderStatusHeader', 'renderStatusTiles', 'renderStatusModels', 'renderStatusMemory', 'renderStatusRecent']
    .forEach(name => { p[name] = () => {}; });
  p.onRuntimeBackendState({ state: 'running', connected: true });
  trend = 0;
  p.onRuntimeMetrics(snapshot());
  assert.equal(trend, 1, 'the first live snapshot hides the last-updated note');
  p.onRuntimeMetrics(snapshot({ unix_ms: 1763280001000 }));
  p.onRuntimeMetrics(snapshot({ unix_ms: 1763280002000 }));
  assert.equal(trend, 1, 'per-second snapshots leave the trend alone');
  p.onRuntimeMetrics(null);
  assert.equal(trend, 2, 'going offline shows the last-updated note');
  p.onRuntimeHistoryPoint(point(5000));
  assert.equal(trend, 3, 'a history point still redraws');
});

test('models card says the backend is starting', () => {
  const container = fakeElement();
  const p = page('zh-Hans', { document: { getElementById: id => (id === 'status-models' ? container : null) } });
  p.onRuntimeBackendState({ state: 'running', connected: true });
  p.onRuntimeMetrics(snapshot());
  p.onRuntimeBackendState({ state: 'starting', connected: false });
  assert.match(container.innerHTML, /后端正在启动…/);
  assert.doesNotMatch(container.innerHTML, /还没有加载模型/);
  p.onRuntimeBackendState({ state: 'recovering', connected: false });
  assert.match(container.innerHTML, /后端正在启动…/);
  p.onRuntimeBackendState({ state: 'stopped', connected: false });
  assert.match(container.innerHTML, /后端未运行。/);
  p.onRuntimeBackendState({ state: 'running', connected: true });
  p.onRuntimeMetrics(snapshot());
  assert.match(container.innerHTML, /还没有加载模型。/);
});

test('retry opens the model page load action for the failed model', () => {
  const failed = { id: 'org/Broken', state: 'failed', last_error: 'bad weights' };
  const views = page('en');
  assert.equal(views.statusModelView(failed, 'en', 0).action.kind, 'retry');
  assert.equal(views.statusModelView(Object.assign({}, failed, { mode: 'dflash2' }), 'en', 0).action.kind, 'logs');

  const control = { focused: 0, focus() { this.focused += 1; } };
  const row = fakeElement({
    dataset: { modelId: 'org/Broken' },
    offsetWidth: 1,
    scrollIntoView() { this.scrolled = true; },
    querySelector: selector => (selector.includes('.action-load') ? control : null),
  });
  const other = fakeElement({ dataset: { modelId: 'org/Other' }, querySelector: () => null });
  const calls = [];
  const timers = [];
  const p = page('en', {
    document: {
      getElementById: () => null,
      querySelectorAll: selector => (selector === '#tab-models-manage tbody tr' ? [other, row] : []),
    },
    setTimeout: (fn, ms) => { timers.push([fn, ms]); return timers.length; },
  });
  p.navigateTo = target => calls.push(['navigateTo', target]);
  p.switchToTab = tab => calls.push(['switchToTab', tab]);
  p.openModelLoadPanel = id => calls.push(['openModelLoadPanel', id]);
  p.loadModel = id => calls.push(['loadModel', id]);
  p.statusRetryModel({ dataset: { modelId: 'org/Broken' } });
  assert.deepEqual(calls, [['navigateTo', 'models'], ['switchToTab', 'models-manage']],
    'retry goes through the model page, never straight to a load or the progress panel');
  assert.ok(row.classes.has('model-load-focus'));
  assert.equal(row.scrolled, true);
  assert.equal(control.focused, 1);
  assert.equal(other.classes.has('model-load-focus'), false);
  // The model list re-syncs when the page opens; the delayed pass points at the rebuilt row.
  timers.filter(([, ms]) => ms === 500).forEach(([fn]) => fn());
  assert.equal(control.focused, 2);
  assert.doesNotThrow(() => p.statusRetryModel(null));
});

test('reopening the window fetches history and recent requests again', () => {
  const fetched = [];
  const p = page('en', { apiFetch: path => fetched.push(path) });
  p.statusRefreshData(true);
  fetched.length = 0;
  p.statusWindowReopened();
  assert.deepEqual(fetched, ['/admin/api/metrics/history', '/admin/api/requests/recent'],
    'a reopen fetches even right after the last fetch');
});
