const test = require('node:test');
const assert = require('node:assert/strict');
const { loadChat } = require('./chat_page_harness.cjs');

const visible = el => Boolean(el) && !el.hidden;

test('reasoning control by capability', () => {
  let page = loadChat();
  page.state({ reasoning: null });
  assert.equal(visible(page.$('#think')), false, 'no control when the model cannot think');

  page = loadChat();
  page.state({ reasoning: { native_efforts: [] } });
  const toggle = page.$('#think .tog');
  assert.ok(visible(page.$('#think')));
  assert.equal(toggle.textContent, '思考');
  assert.equal(toggle.classList.contains('on'), false, 'off by default');
  toggle.click();
  assert.deepEqual(page.last(), { type: 'setReasoning', choice: 'on' });
  assert.equal(toggle.classList.contains('on'), true);
  toggle.click();
  assert.deepEqual(page.last(), { type: 'setReasoning', choice: 'off' });

  page = loadChat();
  page.state({ reasoning: { native_efforts: ['low', 'medium', 'xhigh'] } });
  const levels = page.$('#think .tog');
  assert.equal(levels.textContent, '思考');
  assert.equal(levels.classList.contains('on'), false);
  levels.click();
  assert.deepEqual(page.last(), { type: 'setReasoning', choice: 'medium' }, 'turning it on starts at medium');
  assert.equal(levels.textContent, '思考 · 中 ▾');
  const items = page.$$('#think .mi');
  assert.deepEqual(items.map(item => item.querySelector('.mi-label').textContent), ['关闭', '低', '中', '高']);
  assert.ok(visible(page.$('#think .menu')));
  items[3].click();
  assert.deepEqual(page.last(), { type: 'setReasoning', choice: 'high' });
  assert.equal(levels.textContent, '思考 · 高 ▾');
  levels.click();
  page.$$('#think .mi')[0].click();
  assert.deepEqual(page.last(), { type: 'setReasoning', choice: 'off' });
  assert.equal(levels.textContent, '思考');

  page = loadChat();
  page.state({ reasoning: { native_efforts: ['low', 'medium', 'xhigh'] }, reasoningChoice: 'low' });
  assert.equal(page.$('#think .tog').textContent, '思考 · 低 ▾', 'the saved choice wins');
  assert.equal(page.sent('setReasoning').length, 0);
});

test('switching model asks for confirmation', () => {
  const page = loadChat();
  const models = [
    { id: 'small', name: 'Qwen3.5 4B', sizeBytes: 3e9, loaded: true, current: true, pinned: false },
    { id: 'big', name: 'Qwen3.6 35B-A3B', sizeBytes: 20.4e9, loaded: false, current: false, pinned: false },
  ];
  page.state({ models, model: { id: 'small', name: 'Qwen3.5 4B', status: 'ready' } });
  page.$('#model-chip').click();
  const rows = page.$$('#model-menu .mrow');
  assert.deepEqual(rows.map(row => row.querySelector('.mname').textContent), ['Qwen3.5 4B', 'Qwen3.6 35B-A3B']);
  rows[1].click();
  const dialog = page.$('#dialog');
  assert.ok(visible(dialog));
  assert.match(dialog.textContent, /切换到 Qwen3.6 35B-A3B？/);
  assert.match(dialog.textContent, /会替换当前的 Qwen3.5 4B/);
  assert.match(dialog.textContent, /大约需要 \d+ 秒/);
  assert.equal(page.sent('switchModel').length, 0, 'nothing is switched before the user agrees');
  page.$('#dialog .cancel').click();
  assert.equal(visible(dialog), false);
  assert.equal(page.sent('switchModel').length, 0);

  page.$('#model-chip').click();
  page.$$('#model-menu .mrow')[1].click();
  page.$('#dialog .confirm').click();
  assert.deepEqual(page.last(), { type: 'switchModel', id: 'big' });

  // A pinned default stays loaded, and the copy says so.
  page.state({ models: [{ ...models[0], pinned: true }, models[1]], model: { id: 'small', name: 'Qwen3.5 4B', status: 'ready' } });
  page.$('#model-chip').click();
  page.$$('#model-menu .mrow')[1].click();
  assert.match(page.$('#dialog').textContent, /替换 Qwen3.5 4B/);
  assert.match(page.$('#dialog').textContent, /已被固定，会继续保持加载/);

  // Choosing the current model does nothing.
  page.$('#dialog .cancel').click();
  page.$('#model-chip').click();
  page.$$('#model-menu .mrow')[0].click();
  assert.equal(visible(page.$('#dialog')), false);
});

test('rejects image drop for text-only model', () => {
  const page = loadChat();
  page.state({ supportsVision: false });
  const files = [{ type: 'image/png', name: 'cat.png', size: 2048 }];
  page.doc.dispatch('dragover', { dataTransfer: { types: ['Files'], files } });
  page.doc.dispatch('drop', { dataTransfer: { types: ['Files'], files } });
  page.flush();
  assert.equal(page.$$('#thumbs .thumb').length, 0);
  assert.match(page.$('.notice').textContent, /Qwen3.5 4B 不能看图/);
  assert.equal(visible(page.$('#add-btn')), false, 'no "+" for a text-only model');
  page.send('你好');
  assert.deepEqual(page.last('send').images, []);

  const vision = loadChat();
  vision.state({ supportsVision: true });
  vision.doc.dispatch('drop', { dataTransfer: { types: ['Files'], files } });
  vision.flush();
  assert.equal(vision.$$('#thumbs .thumb').length, 1);
  vision.send('这是什么');
  assert.deepEqual(vision.last('send').images, ['data:image/png;base64,' + Buffer.from('cat.png').toString('base64')]);
  assert.equal(vision.$$('#thumbs .thumb').length, 0);
  assert.equal(vision.$$('.user .uimgs img').length, 1, 'the thumbnail shows above the bubble');
});

test('send disabled while loading', () => {
  const page = loadChat();
  page.state({ model: { id: 'm', name: 'Qwen3.5 4B', status: 'loading' } });
  assert.match(page.$('#model-chip').textContent, /正在加载…/);
  page.type('你好');
  assert.equal(page.$('#send').disabled, true);
  page.$('#input').dispatch('keydown', { key: 'Enter', shiftKey: false, isComposing: false });
  assert.equal(page.sent('send').length, 0);
  page.state({ model: { id: 'm', name: 'Qwen3.5 4B', status: 'ready' } });
  assert.equal(page.$('#send').disabled, false, 'typed text survives the load');
  page.$('#input').dispatch('keydown', { key: 'Enter', shiftKey: false, isComposing: true });
  assert.equal(page.sent('send').length, 0, 'Enter while composing CJK text does not send');
  page.$('#input').dispatch('keydown', { key: 'Enter', shiftKey: false, isComposing: false });
  assert.equal(page.sent('send').length, 1);
});

test('stop replaces send while generating', () => {
  const page = loadChat();
  page.state();
  page.send('写首诗');
  const button = page.$('#send');
  assert.equal(button.classList.contains('stop'), true);
  assert.equal(button.getAttribute('aria-label'), '停止');
  assert.equal(button.disabled, false);
  button.click();
  assert.deepEqual(page.last(), { type: 'stop' });
  page.receive({ type: 'turnFinished', text: '床前', reasoning: null, stopped: true, truncated: false,
    metrics: { firstTokenSeconds: 0.2 } });
  page.flush();
  assert.equal(button.classList.contains('stop'), false);
  assert.equal(button.getAttribute('aria-label'), '发送');
});

test('send waits until every image is read', () => {
  const page = loadChat();
  page.state({ supportsVision: true });
  page.holdReads();
  const files = [{ type: 'image/png', name: 'big.png', size: 15 * 1024 * 1024 }];
  page.doc.dispatch('drop', { dataTransfer: { types: ['Files'], files } });
  page.type('这是什么');
  assert.equal(page.$('#send').disabled, true, 'an image still being read blocks sending');
  page.$('#input').dispatch('keydown', { key: 'Enter', shiftKey: false, isComposing: false });
  page.$('#send').click();
  assert.equal(page.sent('send').length, 0);
  assert.equal(page.$('#send').classList.contains('stop'), false);
  page.finishReads();
  assert.equal(page.$('#send').disabled, false);
  page.$('#send').click();
  const sent = page.last('send');
  assert.equal(sent.images.length, 1);
  assert.ok(sent.images.every(src => typeof src === 'string' && src.startsWith('data:image/')));
});

test('text over the length limit is refused with a message', () => {
  const page = loadChat();
  page.state();
  page.send('长'.repeat(100001));
  assert.equal(page.sent('send').length, 0);
  assert.match(page.$('.notice').textContent, /太长/);
  assert.equal(page.$('#send').classList.contains('stop'), false);
  assert.equal(page.$('#input').value.length, 100001, 'the text stays so it can be shortened');
});

test('a rejected or busy send returns to idle', () => {
  const page = loadChat();
  page.state();
  page.send('你好');
  page.receive({ type: 'error', kind: 'rejected' });
  page.flush();
  assert.equal(page.$('#send').classList.contains('stop'), false);
  assert.match(page.$('.asst .err').textContent, /没有发出去/);
  page.$('.asst .err button.retry').click();
  assert.deepEqual(page.last(), { type: 'send', text: '你好', images: [], retry: true });

  page.receive({ type: 'error', kind: 'busy' });
  page.flush();
  assert.equal(page.$('#send').classList.contains('stop'), false, 'busy resets Stop');
  assert.match(page.$('.asst .err').textContent, /上一条回复还没结束/);

  // Stop pressed while Swift was still preparing the images: nothing was sent.
  page.send('再试');
  page.$('#send').click();
  assert.deepEqual(page.last(), { type: 'stop' });
  page.receive({ type: 'error', kind: 'rejected' });
  page.flush();
  assert.equal(page.$('#send').classList.contains('stop'), false);
  const marks = page.$$('.mark');
  assert.equal(marks[marks.length - 1].textContent, '已停止');
});

test('a reloaded page follows a reply that is still running', () => {
  const page = loadChat();
  page.state({ generating: true });
  assert.equal(page.$('#send').classList.contains('stop'), true, 'Stop shows for the running reply');
  page.receive({ type: 'delta', reasoning: '', text: '还在输出' });
  page.receive({ type: 'turnFinished', text: '还在输出', reasoning: null, stopped: false, truncated: false, metrics: null });
  page.flush();
  assert.equal(page.$('#send').classList.contains('stop'), false);

  const other = loadChat();
  other.state();
  other.receive({ type: 'delta', reasoning: '', text: '意外的输出' });
  other.flush();
  assert.equal(other.$('#send').classList.contains('stop'), true, 'a reply nobody started here still gets Stop');
});
