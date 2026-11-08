const test = require('node:test');
const assert = require('node:assert/strict');
const { loadChat } = require('./chat_page_harness.cjs');

const KATEX = /<span class="katex">/;
const DISPLAY = /<span class="katex-display">/;

function render(text, final = true) {
  return loadChat().window.ironChat.renderMarkdown(text, final);
}

test('inline and display math render with KaTeX', () => {
  const inline = render('每小时注满 $\\frac{1}{6}$ 个水池');
  assert.match(inline, KATEX);
  assert.doesNotMatch(inline, DISPLAY);
  assert.doesNotMatch(inline, /\$/);
  assert.match(inline, /class="mfrac"/);
  assert.match(inline, /每小时注满 /);

  const block = render('合起来是\n\n$$\n\\frac{1}{6} + \\frac{1}{9} = \\frac{5}{18}\n$$\n\n所以要 3.6 小时');
  assert.match(block, DISPLAY);
  assert.doesNotMatch(block, /\$\$|<h1|<h2/, 'math lines are not Markdown');
  assert.match(block, /所以要 3.6 小时/);

  assert.match(render('总时间 $$t = \\frac{18}{5}$$ 小时'), DISPLAY);
  const paren = render('设 \\(x^2 + 1\\) 为');
  assert.match(paren, KATEX);
  assert.doesNotMatch(paren, DISPLAY);
  assert.doesNotMatch(paren, /\\\(/);
  assert.match(render('\\[\n\\sum_{i=1}^n i = \\frac{n(n+1)}{2}\n\\]'), DISPLAY);
  assert.match(render('即 \\[a^2 + b^2 = c^2\\] 成立'), DISPLAY);
});

test('currency, spaced dollars and code are not math', () => {
  for (const text of ['It costs $5 and $10 today.', 'between $5,$10', 'pay $ 5 $ now', 'a $x $ b', '$x$1 off']) {
    const html = render(text);
    assert.doesNotMatch(html, /katex/, text);
  }
  assert.match(render('It costs $5 and $10 today.'), /It costs \$5 and \$10 today\./);
  assert.match(render('Escaped \\$x\\$ stays'), /Escaped \$x\$ stays/);

  const span = render('Use `$x$` or `\\(y\\)` literally');
  assert.doesNotMatch(span, /katex/);
  assert.match(span, /<code>\$x\$<\/code>/);
  const fence = render('```latex\n$$\\frac{1}{2}$$ and $y$\n```');
  assert.doesNotMatch(fence, /katex/);
  assert.match(fence, /\$\$\\frac\{1\}\{2\}\$\$ and \$y\$/);
});

test('math cannot make links or run code', () => {
  for (const text of ['$\\href{javascript:alert(1)}{click}$', '$\\url{javascript:alert(1)}$',
    '$\\href{https://example.com}{x}$', '$\\htmlId{a}{x}\\includegraphics{https://example.com/a.png}$']) {
    const html = render(text);
    assert.doesNotMatch(html, /<a\b|href=|<img|\bid=/i, text);
  }
});

test('malformed TeX shows its source, never an error box', () => {
  const html = render('结果是 $\\frac{1$ 对吧');
  assert.doesNotMatch(html, /katex/);
  assert.match(html, /结果是 \$\\frac\{1\$ 对吧/);
  // Runaway macros stop at maxExpand and show as source too.
  const runaway = render('$\\def\\a{\\a\\a}\\a$');
  assert.doesNotMatch(runaway, /katex/);
  assert.match(runaway, /\$\\def\\a/);
});

test('every piece of math passes DOMPurify and raw HTML around it stays text', () => {
  const page = loadChat();
  const html = page.window.ironChat.renderMarkdown('<img src=x onerror=alert(1)> $x^2$ <script>alert(2)</script>', true);
  assert.match(html, KATEX);
  assert.match(html, /&lt;img src=x onerror=alert\(1\)&gt;/);
  assert.doesNotMatch(html, /<img|<script|<[^>]*onerror/);
  assert.ok(page.purified.some(chunk => /class="katex"/.test(chunk)), 'KaTeX markup is sanitized');
});

/// Like the real DOMPurify, the stand-in turns character references back
/// into characters, so a reply can smuggle any code point past Markdown.
function decodingSanitizer(page) {
  page.window.DOMPurify.sanitize = (html) => {
    page.purified.push(html);
    return html.replace(/&#(\d+);/g, (_, code) => String.fromCodePoint(Number(code)))
      .replace(/&#x([0-9a-f]+);/gi, (_, code) => String.fromCodePoint(parseInt(code, 16)));
  };
}

test('a reply cannot forge the placeholders the page uses for math', () => {
  const page = loadChat();
  decodingSanitizer(page);
  const render = text => page.window.ironChat.renderMarkdown(text, true);
  for (const forged of ['a \uE000' + '0\uE001 b $y$', 'a &#57344;0&#57345; b $y$', 'a &#xE000;0&#xE001; b $y$',
    '$y$ then &#57344;0&#57345; and &#57344;0&#57345;']) {
    const html = render(forged);
    assert.equal((html.match(/class="katex"/g) || []).length, 1, forged);
    assert.doesNotMatch(html, /[\uE000\uE001]/, forged);
  }
  // Two formulas still land in their own places.
  const two = render('$a$ and $b$');
  assert.equal((two.match(/class="katex"/g) || []).length, 2);
  assert.ok(two.indexOf('mathnormal">a<') < two.indexOf('mathnormal">b<'));
});

test('an unclosed $$ stays text instead of flashing inline math', () => {
  for (const text of ['$$\\frac{1}{6}$', '求 $$\\frac{1}{6}$', '$$\n\\frac{1}{6}$']) {
    assert.doesNotMatch(render(text, false), /katex/, text);
  }
  assert.match(render('$$\\frac{1}{6}$$', false), DISPLAY);
  assert.match(render('求 $$\\frac{1}{6}$$ 即可', false), DISPLAY);
  assert.match(render('$$x$$ and $y$'), KATEX);
});

test('unclosed \\( and \\[ add only bounded work to a long reply', () => {
  const page = loadChat();
  // marked alone is the yardstick: math scanning must not grow with the
  // length of the reply after each unclosed delimiter.
  const plain = new page.window.marked.Marked({ gfm: true, breaks: true, async: false });
  const best = fn => {
    let fastest = Infinity;
    for (let i = 0; i < 3; i++) {
      const started = process.hrtime.bigint();
      fn();
      fastest = Math.min(fastest, Number(process.hrtime.bigint() - started) / 1e6);
    }
    return fastest;
  };
  for (const open of ['\\(', '\\[']) {
    const text = (open + ' ' + 'xy '.repeat(20)).repeat(4000);
    let html = '';
    const page_ms = best(() => { html = page.window.ironChat.renderMarkdown(text, false); });
    const marked_ms = best(() => plain.parse(text));
    assert.doesNotMatch(html, /katex/);
    assert.ok(page_ms < marked_ms * 2 + 40, `${open}: page ${Math.round(page_ms)} ms, marked ${Math.round(marked_ms)} ms`);
  }
});

test('unfinished math streams as text until it closes', () => {
  const page = loadChat();
  page.state();
  page.send('水池问题');
  page.receive({ type: 'delta', reasoning: '', text: '每小时注满 $\\frac{1}{' });
  page.flush();
  const text = page.$('.atext').innerHTML;
  assert.doesNotMatch(text, /katex/);
  assert.match(text, /\$\\frac\{1\}\{/);
  page.receive({ type: 'delta', reasoning: '', text: '6}$ 个水池' });
  page.flush();
  assert.match(page.$('.atext').innerHTML, KATEX);
  page.receive({ type: 'turnFinished', text: '每小时注满 $\\frac{1}{6}$ 个水池', reasoning: null, stopped: false,
    truncated: false, metrics: { firstTokenSeconds: 0.2, tokensPerSecond: 50, outputTokens: 12 } });
  page.flush();
  assert.match(page.$('.atext').innerHTML, KATEX);
});
