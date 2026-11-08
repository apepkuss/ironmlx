const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { loadChat } = require('./chat_page_harness.cjs');

const visible = el => Boolean(el) && !el.hidden;
const at = (y, m, d, h, min) => new Date(y, m - 1, d, h, min).getTime();

test('groups history by today yesterday earlier', () => {
  const page = loadChat({ now: at(2026, 11, 8, 14, 5) });
  page.state();
  page.receive({ type: 'history', items: [
    { id: 'a', title: '写一个 Python 函数，统计词频', updatedAt: at(2026, 11, 8, 14, 2), current: true },
    { id: 'b', title: '', updatedAt: at(2026, 11, 8, 13, 47), current: false },
    { id: 'c', title: '用三段话解释黑洞是怎么形成的', updatedAt: at(2026, 11, 7, 21, 15), current: false },
    { id: 'd', title: '给我排一个杭州周末两日游', updatedAt: at(2026, 11, 1, 9, 0), current: false },
  ] });
  page.$('#history-btn').click();
  assert.ok(visible(page.$('#history')));
  assert.deepEqual(page.$$('#history .grp').map(group => group.textContent), ['今天', '昨天', '更早']);
  const items = page.$$('#history .it');
  assert.deepEqual(items.map(item => item.querySelector('.it-title').textContent),
    ['写一个 Python 函数，统计词频', '图片', '用三段话解释黑洞是怎么形成的', '给我排一个杭州周末两日游']);
  assert.deepEqual(items.map(item => item.querySelector('.it-time').textContent), ['14:02', '13:47', '21:15', '11月1日']);
  assert.equal(items[0].classList.contains('on'), true);
  items[2].click();
  assert.deepEqual(page.last(), { type: 'openConversation', id: 'c' });
  assert.equal(visible(page.$('#history')), false);
});

test('history footer links to agents', () => {
  const page = loadChat();
  page.state();
  page.receive({ type: 'history', items: [] });
  page.$('#history-btn').click();
  assert.match(page.$('#history .foot').textContent, /想让本地模型帮你写代码、操作文件？/);
  const link = page.$('#history .foot .agents');
  assert.equal(link.textContent, '接入 DeepSeek Harness 等外部 Agent →');
  link.click();
  assert.deepEqual(page.last(), { type: 'openAgents' });
});

test('empty state examples per language', () => {
  for (const language of ['en', 'zh-Hans', 'zh-Hant', 'ja', 'ko']) {
    const page = loadChat();
    const strings = page.i18n[language];
    assert.equal(strings.examples.length, 4, language);
    page.state({ language, supportsVision: true });
    let cards = page.$$('#empty .sug');
    assert.equal(cards.length, 4, language);
    assert.equal(cards[3].querySelector('.sug-text').textContent, strings.examples[3].text);
    assert.equal(strings.examples[3].vision, true);
    assert.equal(page.$('#input').getAttribute('placeholder'), strings.placeholder);

    page.state({ language, supportsVision: false });
    cards = page.$$('#empty .sug');
    assert.equal(cards.length, 4, language);
    assert.notEqual(cards[3].querySelector('.sug-text').textContent, strings.examples[3].text, language);
    cards[0].click();
    page.flush();
    assert.deepEqual(page.last('send'), { type: 'send', text: strings.examples[0].text, images: [] }, language);
    assert.equal(visible(page.$('#empty')), false);
  }
  const zh = loadChat();
  zh.state();
  assert.match(zh.$('#empty').textContent, /今天想聊点什么？/);
  assert.match(zh.$('#empty').textContent, /Qwen3.5 4B 正在你的 Mac 上本地运行/);
});

test('upgrade banner', () => {
  const page = loadChat();
  page.state();
  page.receive({ type: 'upgradeOffer', name: 'Qwen3.6 35B-A3B', sizeText: '20.4 GB' });
  page.flush();
  const banner = page.$('#banner .upgrade');
  assert.ok(visible(banner));
  assert.match(banner.textContent, /你的 Mac 能运行更强的 Qwen3.6 35B-A3B（20.4 GB）/);
  const buttons = banner.querySelectorAll('button');
  assert.deepEqual(buttons.map(button => button.textContent), ['下载', '不再提示']);
  buttons[0].click();
  assert.deepEqual(page.last(), { type: 'downloadUpgrade' });
  page.receive({ type: 'upgradeStatus', phase: 'downloading', fraction: 0.42 });
  page.flush();
  assert.match(page.$('#banner .upgrade').textContent, /正在下载 Qwen3.6 35B-A3B · 42%/);
  page.receive({ type: 'upgradeStatus', phase: 'done' });
  page.flush();
  assert.equal(page.$('#banner .upgrade'), null);

  const other = loadChat();
  other.state();
  other.receive({ type: 'upgradeOffer', name: 'Qwen3.6 35B-A3B', sizeText: '20.4 GB' });
  other.flush();
  other.$$('#banner .upgrade button')[1].click();
  assert.deepEqual(other.last(), { type: 'dismissUpgrade' });
  assert.equal(other.$('#banner .upgrade'), null);
});

/// The fake DOM has no layout: the list's height shrinks by the banner's
/// height while the banner shows, as the dock grows in WebKit.
function layoutList(page, { content = 1000, view = 400, banner = 60 } = {}) {
  const scroll = page.$('#scroll');
  const bannerNode = page.$('#banner');
  const height = () => (bannerNode.hidden ? view : view - banner);
  let top = 0;
  Object.defineProperty(scroll, 'scrollHeight', { configurable: true, get: () => content });
  Object.defineProperty(scroll, 'clientHeight', { configurable: true, get: height });
  Object.defineProperty(scroll, 'scrollTop', {
    configurable: true,
    get: () => top,
    set: value => { top = Math.max(0, Math.min(value, content - height())); },
  });
  return scroll;
}

function cssRule(selector) {
  const css = fs.readFileSync(path.join(__dirname, '../../ironmlx-app/Sources/IronMLXAppCore/Resources/chat/chat.css'), 'utf8');
  const start = css.indexOf('\n' + selector + ' {');
  assert.ok(start >= 0, selector);
  return css.slice(start, css.indexOf('}', start));
}

test('the banner takes room below the list and never covers the last reply', () => {
  // The banner is in the dock's flow, above the input: it pushes the list up.
  assert.doesNotMatch(cssRule('.banner'), /position:\s*(absolute|fixed)/);
  assert.doesNotMatch(cssRule('.dock'), /position:\s*(absolute|fixed)/);
  assert.match(cssRule('.scroll'), /flex: 1;/);

  const page = loadChat();
  page.state();
  page.send('讲个笑话');
  page.receive({ type: 'turnFinished', text: '好的', reasoning: null, stopped: false, truncated: false,
    metrics: { firstTokenSeconds: 0.2, tokensPerSecond: 50, outputTokens: 3 } });
  page.flush();
  assert.equal(page.$('#banner').parentNode.parentNode.className, 'dock');
  const scroll = layoutList(page);
  scroll.scrollTop = 600;
  scroll.dispatch('scroll');
  const atBottom = () => scroll.scrollTop + scroll.clientHeight === scroll.scrollHeight;
  assert.ok(atBottom());

  page.receive({ type: 'upgradeOffer', name: 'Qwen3.6 35B-A3B', sizeText: '20.4 GB' });
  page.flush();
  assert.ok(page.$('#banner .upgrade'));
  assert.ok(atBottom(), 'the last reply and its speed pill stay in view');
  page.$$('#banner .upgrade button')[1].click();
  page.flush();
  assert.ok(atBottom());

  // Reading further up: what was at the bottom edge stays there.
  scroll.scrollTop = 300;
  scroll.dispatch('scroll');
  page.receive({ type: 'upgradeOffer', name: 'Qwen3.6 35B-A3B', sizeText: '20.4 GB' });
  page.flush();
  assert.equal(scroll.scrollTop + scroll.clientHeight, 700);
  page.receive({ type: 'upgradeStatus', phase: 'done' });
  page.flush();
  assert.equal(scroll.scrollTop + scroll.clientHeight, 700);
});

test('conversation messages rebuild the thread', () => {
  const page = loadChat();
  page.state();
  page.receive({ type: 'conversation', id: 'x', turns: [
    { role: 'user', text: '这是哪里？', images: ['data:image/jpeg;base64,AAAA'] },
    { role: 'assistant', text: '**西湖**', reasoning: '看到断桥', stopped: false, truncated: false,
      metrics: { firstTokenSeconds: 0.4, thinkingSeconds: 1.2, tokensPerSecond: 70, outputTokens: 30 } },
  ] });
  page.flush();
  assert.equal(visible(page.$('#empty')), false);
  assert.equal(page.$$('.user .uimgs img').length, 1);
  assert.match(page.$('.atext').innerHTML, /<strong>西湖<\/strong>/);
  assert.equal(page.$('.think-head').textContent, '已思考 1.2 秒 ›');
  assert.equal(page.$('.pill').textContent, '思考 1.2 s · 速度 70 tok/s · 30 tokens');
  page.$('#new-btn').click();
  assert.deepEqual(page.last(), { type: 'newConversation' });
  page.receive({ type: 'conversation', id: 'y', turns: [] });
  page.flush();
  assert.ok(visible(page.$('#empty')));
  assert.equal(page.$$('.user').length, 0);
});
