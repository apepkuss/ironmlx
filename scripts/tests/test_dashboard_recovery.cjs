const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const html = fs.readFileSync(path.join(__dirname,
  '../../ironmlx-app/Sources/IronMLXAppCore/Resources/dashboard2.html'), 'utf8');
const source = html.slice(html.indexOf('  let serverBannerDismissTimer = null;'),
  html.indexOf('  function normalizeKVQuantValue('));

function dashboard() {
  let banner = null;
  let nextID = 0;
  const timers = new Map();
  const context = {
    I18N: { en: {} }, currentLang: 'en', activeLogTab: 'runtime',
    updateServerStatusVisual() {}, refreshIncidentHistory() {},
    escapeAttr: String,
    setTimeout(fn, delay) { const id = ++nextID; timers.set(id, { fn, delay }); return id; },
    clearTimeout(id) { timers.delete(id); },
    document: {
      getElementById() { return banner; },
      createElement() { return { style: {}, remove() { banner = null; } }; },
      querySelector() { return { firstChild: null, insertBefore(node) { banner = node; } }; },
    },
  };
  vm.createContext(context);
  vm.runInContext(source, context);
  return {
    crash(phase) { context.onServerCrash(phase, { can_retry: true }); },
    clear() { context.clearServerBanner(); },
    get banner() { return banner; },
    get timers() { return timers; },
    expire() { for (const { fn } of [...timers.values()]) fn(); },
  };
}

test('recovery success dismisses its own banner after ten seconds', () => {
  const page = dashboard();
  page.crash('recovered');
  assert.equal([...page.timers.values()][0].delay, 10000);
  page.expire();
  assert.equal(page.banner, null);
  assert.equal(page.timers.size, 0);
});

for (const phase of ['breaker', 'recovering']) {
  test(`a later ${phase} banner survives the earlier recovery timeout`, () => {
    const page = dashboard();
    page.crash('recovered');
    page.crash(phase);
    assert.equal(page.timers.size, 0);
    page.expire();
    assert.ok(page.banner);
    if (phase === 'breaker') assert.match(page.banner.innerHTML, /retryBackendRecovery/);
  });
}

test('clearing a banner cancels its pending dismissal', () => {
  const page = dashboard();
  page.crash('recovered');
  page.clear();
  assert.equal(page.banner, null);
  assert.equal(page.timers.size, 0);
});
