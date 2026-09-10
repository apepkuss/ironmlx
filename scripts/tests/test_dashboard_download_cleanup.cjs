const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const html = fs.readFileSync(path.join(__dirname,
  '../../ironmlx-app/Sources/IronMLXAppCore/Resources/dashboard2.html'), 'utf8');
const confirmation = html.slice(html.indexOf('  function showConfirmDialog('), html.indexOf('  function toggleModelLoad('));
const cleanup = html.slice(html.indexOf('  let downloadCleanupPending'), html.indexOf('  function dismissDownloadReminder('));
const callback = html.slice(html.indexOf('  function onApiFetchResult('), html.indexOf('  function initLogs('));
const translations = html.slice(html.indexOf('  const I18N ='), html.indexOf('\n  };', html.indexOf('  const I18N =')) + 5);

function page(lang = 'en') {
  const elements = new Map(), events = new Map(), requests = [], toasts = [];
  let refreshes = 0;
  function element(id) {
    if (!elements.has(id)) {
      const classes = new Set();
      elements.set(id, { style: {}, focus() {}, classList: {
        add: value => classes.add(value), remove: value => classes.delete(value),
        contains: value => classes.has(value),
      } });
    }
    return elements.get(id);
  }
  const context = {
    currentLang: lang,
    document: { getElementById: element,
      addEventListener: (name, fn) => events.set(name, fn),
      removeEventListener: name => events.delete(name),
    },
    setTimeout: fn => fn(), apiPost: (...args) => requests.push(args),
    showToast: (...args) => toasts.push(args), refreshDownloadQueue: () => refreshes++,
  };
  vm.createContext(context);
  vm.runInContext(translations + '\n' + confirmation + '\n' + cleanup + '\n' + callback, context);
  return { context, element, events, requests, toasts, refreshes: () => refreshes };
}

test('cleanup requires confirmation and posts only once despite repeated clicks', async () => {
  const p = page();
  const pending = p.context.clearFinishedDownloadTasks();
  await p.context.clearFinishedDownloadTasks();
  assert.equal(p.requests.length, 0);
  assert.ok(p.element('confirm-modal').classList.contains('open'));
  assert.match(p.element('confirm-modal-message').textContent, /Installed models.*active or queued/);
  assert.match(p.element('confirm-modal-message').textContent, /cannot be used to resume/);
  p.element('confirm-modal-confirm').onclick();
  await pending;
  assert.equal(p.requests.length, 1);
  assert.equal(p.requests[0][0], '/admin/api/models/downloads/clear-finished');
  await p.context.clearFinishedDownloadTasks();
  assert.equal(p.requests.length, 1);
  p.context.onApiFetchResult('/admin/api/models/downloads/clear-finished', JSON.stringify({ success: true, cleared_count: 2 }));
  assert.equal(p.refreshes(), 1);
  assert.equal(p.toasts[0][1], 'success');
});

for (const cancel of ['button', 'Escape', 'backdrop']) {
  test(`${cancel} cancels cleanup without submitting and allows retry`, async () => {
    const p = page();
    const pending = p.context.clearFinishedDownloadTasks();
    if (cancel === 'button') p.element('confirm-modal-cancel').onclick();
    if (cancel === 'Escape') p.events.get('keydown')({ key: 'Escape' });
    if (cancel === 'backdrop') p.element('confirm-modal').onclick({ target: p.element('confirm-modal') });
    await pending;
    assert.equal(p.requests.length, 0);
    assert.equal(p.events.size, 0);
    const retry = p.context.clearFinishedDownloadTasks();
    p.element('confirm-modal-confirm').onclick();
    await retry;
    assert.equal(p.requests.length, 1);
  });
}

test('partial failures and finishing tasks report retained records and permit retry', async () => {
  const p = page();
  p.context.onDownloadCleanupResult(JSON.stringify({ success: false, cleared_count: 2,
    skipped_count: 1, failures: [{ repo_id: 'org/model', error: 'Repository busy' }] }));
  assert.match(p.toasts[0][0], /2 download records/);
  assert.match(p.toasts[0][0], /records were kept/);
  assert.match(p.toasts[0][0], /org\/model: Repository busy/);
  assert.match(p.toasts[0][0], /still finishing/);
  assert.equal(p.toasts[0][1], 'warn');
  const retry = p.context.clearFinishedDownloadTasks();
  p.element('confirm-modal-cancel').onclick();
  await retry;
  p.context.onDownloadCleanupResult('invalid JSON');
  assert.equal(p.toasts[1][1], 'warn');
  assert.equal(p.refreshes(), 2);
});

for (const lang of ['en', 'zh-Hans', 'zh-Hant', 'ja', 'ko']) {
  test(`cleanup confirmation and feedback are translated in ${lang}`, async () => {
    const p = page(lang);
    const pending = p.context.clearFinishedDownloadTasks();
    assert.ok(p.element('confirm-modal-message').textContent?.length > 30);
    assert.ok(p.element('confirm-modal-confirm').textContent?.length > 0);
    if (lang === 'zh-Hans') assert.match(p.element('confirm-modal-message').textContent, /无法用于续传/);
    if (lang === 'zh-Hant') assert.match(p.element('confirm-modal-message').textContent, /無法用於續傳/);
    p.element('confirm-modal-cancel').onclick();
    await pending;
    p.context.onDownloadCleanupResult('{"success":true,"cleared_count":3}');
    assert.match(p.toasts[0][0], /3/);
  });
}
