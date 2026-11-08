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
  return [...page.matchAll(/<div class="onboarding-step"([^>]*)>([\s\S]*?)<\/div>\s*<\/div>\s*<\/div>/g)]
    .map(match => ({ attributes: match[1], body: match[2] }));
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
