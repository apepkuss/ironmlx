const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const html = fs.readFileSync(
  path.join(__dirname, '../../ironmlx-app/Sources/IronMLXAppCore/Resources/dashboard2.html'), 'utf8');
function slice(start, end) { return html.slice(html.indexOf(start), html.indexOf(end, html.indexOf(start))); }
const page = slice('<div class="page" id="page-onboarding">', '<div class="page active" id="page-status">');
const I18N = vm.runInNewContext(slice('  const I18N =', '  const LANG_MAP =') + '; I18N');
const languages = ['en', 'zh-Hans', 'zh-Hant', 'ja', 'ko'];

function steps() {
  return [...page.matchAll(/<button type="button" class="onboarding-step"([^>]*)>([\s\S]*?)<\/button>/g)]
    .map(match => ({ attributes: match[1], body: match[2] }));
}

// Body of every CSS rule whose selector belongs to this page.
function pageRules() {
  const out = [];
  const re = /\n  ((?:#page-onboarding|\.onboarding|\.step-)[^{}]*)\{([^}]*)\}/g;
  let match;
  while ((match = re.exec(html))) out.push({ selector: match[1].trim(), body: match[2] });
  return out;
}

function rule(selector) {
  const found = pageRules().find(item => item.selector === selector);
  assert.ok(found, 'missing rule ' + selector);
  return found.body;
}

function click(step) {
  const onclick = /onclick="([^"]*)"/.exec(step.attributes);
  assert.ok(onclick, 'step has an onclick handler');
  const messages = [];
  const handler = name => ({ postMessage: body => messages.push([name, body]) });
  const window = { webkit: { messageHandlers: {
    startStarterDownload: handler('startStarterDownload'),
    openChat: handler('openChat'),
  } } };
  vm.runInNewContext(onclick[1], { window });
  return messages.map(([name]) => name);
}

test('onboarding has exactly two steps: one-click download, then chat', () => {
  const found = steps();
  assert.equal(found.length, 2);
  assert.deepEqual(click(found[0]), ['startStarterDownload']);
  assert.deepEqual(click(found[1]), ['openChat']);
  assert.doesNotMatch(page, /setDefaultModel|models-download|navigateTo/);
});

test('onboarding buttons do nothing outside the App web view', () => {
  for (const step of steps()) {
    const onclick = /onclick="([^"]*)"/.exec(step.attributes)[1];
    assert.doesNotThrow(() => vm.runInNewContext(onclick, { window: {} }));
  }
});

test('every onboarding string exists in all five languages', () => {
  const keys = [...page.matchAll(/data-i18n="([^"]+)"/g)].map(match => match[1]);
  for (const key of ['onboarding_download_title', 'onboarding_download_desc', 'onboarding_chat_title', 'onboarding_chat_desc']) {
    assert.ok(keys.includes(key), `${key} is used on the page`);
  }
  for (const lang of languages) {
    for (const key of keys) {
      assert.equal(typeof I18N[lang][key], 'string', `${lang}.${key}`);
      assert.ok(I18N[lang][key].trim().length > 0, `${lang}.${key} is not empty`);
    }
    for (const removed of ['step1_title', 'step1_desc', 'step2_title', 'step2_desc', 'step3_title', 'step3_desc']) {
      assert.equal(I18N[lang][removed], undefined, `${lang}.${removed} was removed`);
    }
  }
  assert.equal(I18N['zh-Hans'].onboarding_chat_title, '开始对话');
});

test('chat step promises chat only while the download window stays open', () => {
  // With the window closed the App notifies instead of opening chat.
  assert.match(I18N['zh-Hans'].onboarding_chat_desc, /窗口开着.*直接进入对话.*通知你/);
  assert.match(I18N.en.onboarding_chat_desc, /stays open.*notified/);
});

// Steps are real buttons so the keyboard and screen readers can reach them.
test('onboarding steps are buttons, not clickable divs', () => {
  assert.equal(steps().length, 2);
  assert.doesNotMatch(page, /<div class="onboarding-step"/);
  for (const step of steps()) assert.match(step.body, /class="step-title"/);
  assert.ok(rule('.onboarding-step:focus-visible').includes('var(--ds-blue)'));
});

test('onboarding styles use design-system tokens only', () => {
  const rules = pageRules();
  assert.ok(rules.length >= 10);
  for (const { selector, body } of rules) {
    if (selector === '.onboarding-logo') continue; // white "Fe" on the brand tile
    assert.doesNotMatch(body, /#[0-9a-fA-F]{3,8}\b|\brgba?\(|box-shadow/, selector);
    assert.doesNotMatch(body, /var\(--(?!ds-)/, selector);
  }
  const step = rule('.onboarding-step');
  for (const token of ['var(--ds-surface-1)', 'var(--ds-border)', 'var(--ds-radius-card)']) assert.ok(step.includes(token), token);
  const number = rule('.step-num');
  assert.ok(number.includes('var(--ds-surface-2)'));
  assert.doesNotMatch(number, /--ds-blue/);
});
