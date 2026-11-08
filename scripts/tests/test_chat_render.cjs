const test = require('node:test');
const assert = require('node:assert/strict');
const { loadChat } = require('./chat_page_harness.cjs');

function replyTo(page, text) {
  page.state();
  page.send(text);
}

test('sanitizes model html', () => {
  const page = loadChat();
  assert.equal(page.purifySupported, false, 'node has no DOM, so the harness stands in for DOMPurify');
  replyTo(page, '你好');
  const hostile = 'Hi <img src=x onerror=alert(1)> <script>alert(2)</script>\n\n'
    + '[a](javascript:alert(1)) [b](JAVASCRIPT:alert(1)) ![c](https://example.com/x.png)\n\n<div onclick="x()">block</div>';
  page.receive({ type: 'delta', reasoning: '', text: hostile });
  page.receive({ type: 'turnFinished', text: hostile, reasoning: null, stopped: false, truncated: false,
    metrics: { firstTokenSeconds: 0.2, tokensPerSecond: 50, outputTokens: 30 } });
  page.flush();
  const html = page.$('.atext').innerHTML;
  assert.match(html, /Hi/);
  // Raw HTML stays visible as text; nothing in it becomes a tag, a handler or a link.
  for (const bad of [/<[^>]*\bonerror/i, /<[^>]*\bonclick/i, /<script/i, /javascript:/i, /<img/i, /<div/i]) {
    assert.doesNotMatch(html, bad);
  }
  assert.match(html, /&lt;img src=x onerror=alert\(1\)&gt;/);
  assert.match(page.window.ironChat.renderMarkdown('a<br>b<sup>2</sup>', true), /a<br>b<sup>2<\/sup>/);
  // Every model html goes through DOMPurify before it reaches the page.
  assert.ok(page.purified.length > 0);
  assert.ok(page.purified.every(chunk => !/<[^>]*\bon\w+=|<script|javascript:/i.test(chunk)));
  // Allowed links stay, and open outside (handled natively by ChatBridge).
  assert.match(page.window.ironChat.renderMarkdown('[docs](https://example.com/a)', true), /href="https:\/\/example.com\/a"/);
});

test('renders markdown and highlights code', () => {
  const page = loadChat();
  const html = page.window.ironChat.renderMarkdown('**粗体**\n\n```python\ndef f(x):\n    return x\n```\n\n| a | b |\n|---|---|\n| 1 | 2 |', true);
  assert.match(html, /<strong>粗体<\/strong>/);
  assert.match(html, /<pre><code class="hljs language-python">/);
  assert.match(html, /<span class="hljs-keyword">def<\/span>/);
  assert.match(html, /<table>/);
  // An unknown language is escaped, never interpreted.
  assert.match(page.window.ironChat.renderMarkdown('```weird\n<b>x</b>\n```', false), /&lt;b&gt;x&lt;\/b&gt;/);
});

test('error shows retry', () => {
  const page = loadChat();
  replyTo(page, '讲个笑话');
  assert.deepEqual(page.last('send'), { type: 'send', text: '讲个笑话', images: [] });
  page.receive({ type: 'error', kind: 'connection', message: 'socket closed' });
  page.flush();
  const error = page.$('.asst .err');
  assert.ok(error, 'the error shows where the reply would be');
  assert.match(error.textContent, /连接断开/);
  assert.doesNotMatch(error.textContent, /socket closed/);
  assert.equal(page.alerts.length, 0, 'no dialog');
  assert.equal(page.$('#send').classList.contains('stop'), false);
  const retry = page.$('.asst .err button.retry');
  assert.equal(retry.textContent, '重试');
  retry.click();
  page.flush();
  assert.deepEqual(page.last(), { type: 'send', text: '讲个笑话', images: [], retry: true });
  assert.equal(page.$$('.user').length, 1, 'the question is not repeated');
  assert.equal(page.$('.asst .err'), null);
});

test('speed pill format', () => {
  const page = loadChat();
  replyTo(page, '写个函数');
  page.receive({ type: 'delta', reasoning: '', text: '可以用 Counter' });
  page.receive({ type: 'turnFinished', text: '可以用 Counter', reasoning: null, stopped: false, truncated: false,
    metrics: { firstTokenSeconds: 0.19, tokensPerSecond: 88.3, outputTokens: 142 } });
  page.flush();
  assert.equal(page.$('.pill').textContent, '首字 0.19 s · 速度 88 tok/s · 142 tokens');

  page.send('水池问题');
  page.receive({ type: 'delta', reasoning: '甲管', text: '' });
  page.receive({ type: 'delta', reasoning: '', text: '3.6 小时' });
  page.receive({ type: 'turnFinished', text: '3.6 小时', reasoning: '甲管', stopped: false, truncated: false,
    metrics: { firstTokenSeconds: 0.2, thinkingSeconds: 3.1, tokensPerSecond: 112, outputTokens: 204 } });
  page.flush();
  const pills = page.$$('.pill');
  assert.equal(pills[pills.length - 1].textContent, '思考 3.1 s · 速度 112 tok/s · 204 tokens');
  assert.equal(page.$$('.think-head')[0].textContent, '已思考 3.1 秒 ›');
});

test('thinking collapses when text starts', () => {
  const page = loadChat();
  replyTo(page, '水池问题');
  page.receive({ type: 'delta', reasoning: '甲管每小时注满 1/6', text: '' });
  page.flush();
  const think = page.$('.think');
  assert.ok(think);
  assert.equal(think.classList.contains('closed'), false);
  assert.match(page.$('.think-head').textContent, /^思考中/);
  assert.match(page.$('.think-body').textContent, /甲管每小时注满/);
  page.advance(3100);
  page.receive({ type: 'delta', reasoning: '', text: '两管一起开需要' });
  page.flush();
  assert.equal(think.classList.contains('closed'), true);
  assert.equal(page.$('.think-head').textContent, '已思考 3.1 秒 ›');
  page.$('.think-head').click();
  assert.equal(think.classList.contains('open'), true, 'clicking expands the thinking');
});

test('stopped and truncated replies are marked', () => {
  const page = loadChat();
  replyTo(page, '长文');
  page.receive({ type: 'delta', reasoning: '', text: '第一段' });
  page.receive({ type: 'turnFinished', text: '第一段', reasoning: null, stopped: false, truncated: true,
    metrics: { firstTokenSeconds: 0.3, tokensPerSecond: 60, outputTokens: 4096 } });
  page.flush();
  assert.match(page.$('.mark').textContent, /截断/);
  page.send('再来');
  page.receive({ type: 'turnFinished', text: '', reasoning: null, stopped: true, truncated: false, metrics: null });
  page.flush();
  const marks = page.$$('.mark');
  assert.equal(marks[marks.length - 1].textContent, '已停止');
});

test('trimmed history shows a light notice', () => {
  const page = loadChat();
  replyTo(page, '继续');
  page.receive({ type: 'trimmed', count: 4 });
  page.flush();
  assert.match(page.$('.sys').textContent, /较早的 4 条消息/);
});

test('model problems show actions instead of dialogs', () => {
  const page = loadChat();
  page.state({ model: { id: 'b', name: 'Big', status: 'noModel', failedModelID: 'b', failedName: 'Big',
    previousModelID: 'a', previousName: 'Small' } });
  assert.equal(page.$('#model-chip').textContent.includes('没有可用的模型'), true);
  assert.match(page.$('.notice').textContent, /没能切换到 Big，Small 也没能重新加载/);
  const reload = page.$('.notice button');
  assert.equal(reload.textContent, '重新加载 Small');
  reload.click();
  assert.deepEqual(page.last(), { type: 'reloadModel', id: 'a' });

  page.state();
  page.receive({ type: 'error', kind: 'modelRequiresDashboard', modelID: 'q', modelName: 'Qwen3.6 27B' });
  page.flush();
  assert.match(page.$('.notice').textContent, /Qwen3.6 27B 使用 DFlash2，请在 Dashboard 里切换/);
  page.$('.notice button').click();
  assert.deepEqual(page.last(), { type: 'openDashboardModels' });

  page.state({ model: { id: 'c', name: 'Gemma', status: 'failed', message: 'oom' } });
  assert.match(page.$('.notice').textContent, /Gemma 没能加载/);
  page.$('.notice button').click();
  assert.deepEqual(page.last(), { type: 'reloadModel', id: 'c' });
  assert.equal(page.alerts.length, 0);
});

test('a model unloaded elsewhere offers to load it again', () => {
  const page = loadChat();
  page.state({ model: { id: 'c', name: 'Gemma', status: 'failed', reason: 'unloaded', message: 'gone' } });
  assert.match(page.$('.notice').textContent, /Gemma 已不再加载/);
  assert.doesNotMatch(page.$('.notice').textContent, /内存不够/);
  const reload = page.$('.notice button');
  assert.equal(reload.textContent, '重新加载 Gemma');
  reload.click();
  assert.deepEqual(page.last(), { type: 'reloadModel', id: 'c' });
  for (const language of ['en', 'zh-Hans', 'zh-Hant', 'ja', 'ko']) {
    assert.equal(typeof page.i18n[language].modelUnloaded, 'string', language);
    assert.equal(typeof page.i18n[language].switchFailedPinned, 'string', language);
  }
});

test('a switch refused for memory points at unpinning the current model', () => {
  const page = loadChat();
  page.state();
  page.receive({ type: 'error', kind: 'modelLoadFailed', modelID: 'big', modelName: 'Qwen3.6 35B',
    message: 'oom', unpinHint: true, pinnedModelName: 'Qwen3.5 4B' });
  page.flush();
  const text = page.$('.notice').textContent;
  assert.match(text, /没能切换到 Qwen3.6 35B/);
  assert.match(text, /Qwen3.5 4B 已被固定/);
  assert.match(text, /Dashboard/);
  page.$('.notice button').click();
  assert.deepEqual(page.last(), { type: 'openDashboardModels' });

  // Without the hint the plain switch failure stays as it was.
  const plain = loadChat();
  plain.state();
  plain.receive({ type: 'error', kind: 'modelLoadFailed', modelID: 'big', modelName: 'Qwen3.6 35B', message: 'x' });
  plain.flush();
  assert.match(plain.$('.notice').textContent, /没能切换到 Qwen3.6 35B，仍在使用 Qwen3.5 4B。/);
  assert.doesNotMatch(plain.$('.notice').textContent, /固定/);
});
