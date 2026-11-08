// The menu bar chat page. It never talks to the backend: everything goes
// through the Swift bridge (`window.webkit.messageHandlers.chat`), and Swift
// calls `window.ironChat.receive({type, ...})`. Model output is rendered as
// Markdown and always passes DOMPurify before it is inserted.
(function () {
  'use strict';

  var w = window;
  var d = document;
  var STRINGS = w.CHAT_I18N || {};
  var MAX_IMAGES = 4;
  var MAX_IMAGE_BYTES = 20 * 1024 * 1024;
  // UTF-16 units; ChatBridge refuses more than 200 000 characters.
  var MAX_TEXT_LENGTH = 100000;
  var AUTO_LANGUAGES = ['python', 'javascript', 'typescript', 'rust', 'swift', 'bash', 'json', 'go', 'java', 'c', 'cpp'];
  var LEVELS = ['off', 'low', 'medium', 'high'];
  var LEVEL_KEYS = { off: 'thinkOff', low: 'thinkLow', medium: 'thinkMedium', high: 'thinkHigh' };

  var ICONS = {
    history: '<svg viewBox="0 0 20 20" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="10" cy="10" r="7.25"/><path d="M10 6v4.25l2.75 1.75"/></svg>',
    compose: '<svg viewBox="0 0 20 20" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M9.5 3.75H5.75a2 2 0 0 0-2 2v8.5a2 2 0 0 0 2 2h8.5a2 2 0 0 0 2-2V10.5"/><path d="M14.4 3.1a1.55 1.55 0 0 1 2.2 2.2l-6.15 6.15-2.95.75.75-2.95z"/></svg>',
    plus: '<svg viewBox="0 0 20 20" width="16" height="16" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"><path d="M10 4.5v11M4.5 10h11"/></svg>',
    send: '<svg viewBox="0 0 20 20" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10 15.5V4.75M5.25 9.5L10 4.75l4.75 4.75"/></svg>',
    stop: '<svg viewBox="0 0 20 20" width="11" height="11"><rect x="3" y="3" width="14" height="14" rx="2.5" fill="currentColor"/></svg>',
    think: '<svg viewBox="0 0 16 16" width="13" height="13" fill="none" stroke="currentColor" stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round"><path d="M6.1 12.4h3.8M6.6 14.4h2.8M8 1.6a4.3 4.3 0 0 0-2.65 7.7c.5.4.8.95.8 1.55v.1h3.7v-.1c0-.6.3-1.15.8-1.55A4.3 4.3 0 0 0 8 1.6z"/></svg>',
    close: '<svg viewBox="0 0 16 16" width="10" height="10" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><path d="M4 4l8 8M12 4l-8 8"/></svg>',
    check: '<svg viewBox="0 0 16 16" width="12" height="12" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M3.5 8.5l3 3 6-7"/></svg>',
    image: '<svg viewBox="0 0 24 24" width="30" height="30" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="4.5" width="18" height="15" rx="3"/><circle cx="9" cy="10" r="1.8"/><path d="M21 16l-5-5-8.5 8.5"/></svg>'
  };

  var S = {
    lang: normalizeLanguage(w.navigator && w.navigator.language),
    models: [],
    model: null,
    reasoning: null,
    choice: 'off',
    vision: false,
    generating: false,
    pending: null,
    stopRequested: false,
    lastSend: null,
    attachments: [],
    history: [],
    conversationId: null,
    errorNotice: null,
    upgrade: null,
    popover: null,
    pendingSwitch: null,
    stickToBottom: true,
    listHeight: null
  };

  // ── Helpers ──────────────────────────────────────────────────────────

  function normalizeLanguage(value) {
    var lang = String(value || '').trim();
    if (lang === 'zh' || lang === 'zh-Hans' || /^zh-(CN|SG)/i.test(lang)) return 'zh-Hans';
    if (lang === 'zh-Hant' || /^zh-(TW|HK|MO)/i.test(lang)) return 'zh-Hant';
    if (/^ja/i.test(lang)) return 'ja';
    if (/^ko/i.test(lang)) return 'ko';
    return 'en';
  }

  function strings() { return STRINGS[S.lang] || STRINGS.en || {}; }

  function t(key, vars) {
    var value = strings()[key];
    if (value === undefined && STRINGS.en) value = STRINGS.en[key];
    if (typeof value !== 'string') return key;
    return value.replace(/\{(\w+)\}/g, function (match, name) {
      return vars && vars[name] !== undefined && vars[name] !== null ? String(vars[name]) : match;
    });
  }

  function nowMs() {
    return typeof w.__IRONCHAT_NOW__ === 'function' ? w.__IRONCHAT_NOW__() : Date.now();
  }

  function el(tag, className, text) {
    var node = d.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = text;
    return node;
  }

  function icon(name, className) {
    var span = el('span', className || 'ico');
    span.innerHTML = ICONS[name];
    span.setAttribute('aria-hidden', 'true');
    return span;
  }

  function button(className, label) {
    var node = el('button', className, label);
    node.type = 'button';
    return node;
  }

  function on(node, type, handler) { node.addEventListener(type, handler); }

  function show(node, visible) { node.hidden = !visible; }

  function escapeHtml(value) {
    return String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }

  function post(message) {
    try {
      w.webkit.messageHandlers.chat.postMessage(message);
    } catch (error) {
      // Outside the App there is nobody to talk to.
    }
  }

  function ready() { return Boolean(S.model && S.model.status === 'ready'); }

  function modelName() { return (S.model && S.model.name) || ''; }

  // ── Markdown ─────────────────────────────────────────────────────────

  var markdownParser = null;
  var finalRender = false;
  var SIMPLE_TAG = /^<(br ?\/?|\/?sup|\/?sub)>$/i;

  function safeUrl(href) {
    if (!href) return null;
    var value = String(href).replace(/[\u0000- \u007F]+/g, '');
    return /^(https?:\/\/|mailto:)/i.test(value) ? value : null;
  }

  function highlight(code, lang) {
    var hljs = w.hljs;
    if (hljs && lang && hljs.getLanguage(lang)) {
      return { html: hljs.highlight(code, { language: lang, ignoreIllegals: true }).value, lang: lang };
    }
    // Detection is costly, so it waits for the finished reply.
    if (hljs && finalRender && !lang && code.length < 20000) {
      var auto = hljs.highlightAuto(code, AUTO_LANGUAGES);
      return { html: auto.value, lang: auto.language || '' };
    }
    return { html: escapeHtml(code), lang: lang };
  }

  // ── Math ─────────────────────────────────────────────────────────────
  // TeX in a reply becomes KaTeX markup. While Markdown renders and
  // DOMPurify runs, each formula is an opaque placeholder; afterwards the
  // placeholder becomes KaTeX's output, which DOMPurify sanitizes on its own.
  // Code spans and blocks never reach these tokenizers.

  var MATH_OPEN = '\uE000';
  var MATH_CLOSE = '\uE001';
  var MATH_MARKS = /[\uE000\uE001]/g;
  // A tag, one of this render's placeholders, or a stray mark: a reply can
  // still spell the marks as character references, but never the nonce.
  var MATH_SLOT = /<[^>]*>|\uE000([0-9a-f]+)-(\d+)\uE001|[\uE000\uE001]/g;
  var KATEX_OPTIONS = { throwOnError: true, trust: false, output: 'html', maxExpand: 1000, maxSize: 20, strict: 'ignore' };
  var KATEX_PURIFY = {
    ALLOWED_TAGS: ['span', 'svg', 'path', 'line'],
    ALLOWED_ATTR: ['class', 'style', 'aria-hidden', 'width', 'height', 'viewbox', 'preserveaspectratio',
      'd', 'x1', 'x2', 'y1', 'y2', 'stroke-width']
  };
  // Every scan is bounded, so a long reply full of unclosed delimiters still
  // renders in linear time.
  var MATH_BLOCK = [/^ {0,3}\$\$((?:[^\n]|\n(?!\s*\n)){1,8000}?)\$\$[ \t]*(?:\n+|$)/,
    /^ {0,3}\\\[((?:[^\n]|\n(?!\s*\n)){1,8000}?)\\\][ \t]*(?:\n+|$)/];
  var mathSlots = null;
  var mathNonce = '';
  var mathCache = new Map();

  /// `$...$` follows pandoc: no space just inside the dollars and no digit
  /// right after, so "$5 and $10" stays text.
  function inlineMath(src) {
    var m = /^\$\$((?:\\[\s\S]|[^\\$]){1,8000}?)\$\$/.exec(src);
    if (m) return { raw: m[0], tex: m[1], display: true };
    // An unclosed `$$` (still streaming) is text, so its second `$` cannot
    // open inline math for a frame.
    if (src.charAt(0) === '$' && src.charAt(1) === '$') return { raw: '$$', literal: true };
    m = /^\\\(((?:\\[^\n]|[^\\\n]|\n(?![ \t]*\n)){1,2000}?)\\\)/.exec(src);
    if (m) return { raw: m[0], tex: m[1], display: false };
    m = /^\\\[((?:\\[\s\S]|[^\\]){1,4000}?)\\\]/.exec(src);
    if (m) return { raw: m[0], tex: m[1], display: true };
    m = /^\$(?!\$)((?:\\[^\n]|[^\\$\n]){1,2000}?)\$(?!\d)/.exec(src);
    if (m && !/^\s|\s$/.test(m[1])) return { raw: m[0], tex: m[1], display: false };
    return null;
  }

  function blockMath(src) {
    for (var i = 0; i < MATH_BLOCK.length; i++) {
      var m = MATH_BLOCK[i].exec(src);
      if (m) return { raw: m[0], tex: m[1], display: true };
    }
    return null;
  }

  function mathToken(type, found) {
    if (found && found.literal) return { type: type, raw: found.raw, text: found.raw, literal: true };
    return found && found.tex.trim() ? { type: type, raw: found.raw, text: found.raw.trim(), tex: found.tex, display: found.display } : undefined;
  }

  /// KaTeX's markup for one formula, or null when it does not parse.
  function texHtml(tex, display) {
    var key = (display ? 'D' : 'I') + tex;
    if (mathCache.has(key)) return mathCache.get(key);
    var html = null;
    try {
      html = String(w.DOMPurify.sanitize(w.katex.renderToString(tex, Object.assign({ displayMode: display }, KATEX_OPTIONS)), KATEX_PURIFY));
    } catch (error) {
      html = null;
    }
    if (mathCache.size > 300) mathCache.clear();
    mathCache.set(key, html);
    return html;
  }

  /// A placeholder that survives Markdown and DOMPurify; `fillMath` swaps it.
  function mathSlot(token) {
    var html = w.katex ? texHtml(token.tex, token.display) : null;
    // A formula that does not parse shows its source as typed.
    if (html === null) html = escapeHtml(token.text).replace(/\n/g, '<br>');
    mathSlots.push(html);
    return MATH_OPEN + mathNonce + '-' + (mathSlots.length - 1) + MATH_CLOSE;
  }

  function newNonce() {
    var words = new Uint32Array(2);
    if (w.crypto && w.crypto.getRandomValues) w.crypto.getRandomValues(words);
    else for (var i = 0; i < words.length; i++) words[i] = Math.floor(Math.random() * 0x100000000);
    return words[0].toString(16) + words[1].toString(16);
  }

  /// Each placeholder of this render becomes its formula once; anything
  /// else that looks like one is dropped.
  function fillMath(html, slots, nonce) {
    if (html.indexOf(MATH_OPEN) < 0 && html.indexOf(MATH_CLOSE) < 0) return html;
    var used = {};
    return html.replace(MATH_SLOT, function (match, key, index) {
      if (match.charAt(0) === '<') return match;
      if (key !== nonce || used[index] || slots[Number(index)] === undefined) return '';
      used[index] = true;
      return slots[Number(index)];
    });
  }

  var MATH_EXTENSIONS = [{
    name: 'mathBlock',
    level: 'block',
    start: function (src) {
      var lineStart = /(?:^|\n) {0,3}(?:\$\$|\\\[)/g;
      var m;
      while ((m = lineStart.exec(src))) {
        var at = m[0].charAt(0) === '\n' ? m.index + 1 : m.index;
        if (blockMath(src.slice(at))) return at;
        lineStart.lastIndex = m.index + 1;
      }
      return undefined;
    },
    tokenizer: function (src) { return mathToken('mathBlock', blockMath(src)); },
    renderer: function (token) { return '<p>' + mathSlot(token) + '</p>\n'; }
  }, {
    name: 'mathInline',
    level: 'inline',
    start: function (src) {
      var m = /\$|\\[(\[]/.exec(src);
      return m ? m.index : undefined;
    },
    tokenizer: function (src) { return mathToken('mathInline', inlineMath(src)); },
    renderer: function (token) { return token.literal ? escapeHtml(token.text) : mathSlot(token); }
  }];

  function parser() {
    if (markdownParser || !w.marked) return markdownParser;
    markdownParser = new w.marked.Marked({ gfm: true, breaks: true, async: false });
    markdownParser.use({
      extensions: MATH_EXTENSIONS,
      renderer: {
        // Raw HTML from the model is shown as text, never interpreted,
        // except bare line-break and super/subscript tags.
        html: function (html) {
          return SIMPLE_TAG.test(html) ? html.toLowerCase().replace('/>', '>').replace(' ', '') : escapeHtml(html);
        },
        link: function (href, title, text) {
          var url = safeUrl(href);
          if (!url) return text;
          return '<a href="' + escapeHtml(url) + '"' + (title ? ' title="' + escapeHtml(title) + '"' : '') + '>' + text + '</a>';
        },
        // Remote images are never loaded; they become links.
        image: function (href, title, text) {
          var url = safeUrl(href);
          var label = escapeHtml(text || href || '');
          return url ? '<a href="' + escapeHtml(url) + '">' + label + '</a>' : label;
        },
        code: function (code, info) {
          var lang = String(info || '').trim().split(/\s+/)[0].toLowerCase().replace(/[^a-z0-9+#_-]/g, '');
          var result = highlight(code, lang);
          var cls = 'hljs' + (result.lang ? ' language-' + result.lang : '');
          return '<pre><code class="' + cls + '">' + result.html + '</code></pre>\n';
        }
      }
    });
    return markdownParser;
  }

  var PURIFY_CONFIG = {
    ALLOWED_TAGS: ['p', 'br', 'strong', 'b', 'em', 'i', 'del', 's', 'code', 'pre', 'span', 'a', 'ul', 'ol', 'li',
      'blockquote', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'hr', 'table', 'thead', 'tbody', 'tr', 'th', 'td', 'sup', 'sub'],
    ALLOWED_ATTR: ['href', 'title', 'class', 'start', 'align'],
    ALLOWED_URI_REGEXP: /^(?:https?:|mailto:)/i
  };

  function plain(text) {
    return '<p>' + escapeHtml(text).replace(/\n/g, '<br>') + '</p>';
  }

  /// Markdown → HTML that is safe to insert. Without DOMPurify nothing
  /// is interpreted: the text is shown as is.
  function renderMarkdown(text, final) {
    // The placeholder characters are the page's own; the model cannot use them.
    var source = String(text || '').replace(MATH_MARKS, '');
    var purify = w.DOMPurify;
    if (!purify || typeof purify.sanitize !== 'function') return plain(source);
    var html;
    var slots = mathSlots = [];
    var nonce = mathNonce = newNonce();
    finalRender = Boolean(final);
    try {
      var md = parser();
      html = md ? md.parse(source) : plain(source);
    } catch (error) {
      html = plain(source);
      slots.length = 0;
    }
    finalRender = false;
    mathSlots = null;
    return fillMath(String(purify.sanitize(html, PURIFY_CONFIG)), slots, nonce);
  }

  // ── Shell ────────────────────────────────────────────────────────────

  var ui = {};

  function build() {
    var app = el('div', 'app');
    app.id = 'app';

    var bar = el('header', 'bar');
    ui.chip = button('chip');
    ui.chip.id = 'model-chip';
    ui.chipDot = el('span', 'dot');
    ui.chipName = el('span', 'chip-name');
    ui.chip.append(ui.chipDot, ui.chipName, el('span', 'caret', '▾'));
    on(ui.chip, 'click', function () { togglePopover('models'); });
    ui.historyBtn = button('icon');
    ui.historyBtn.id = 'history-btn';
    ui.historyBtn.appendChild(icon('history'));
    on(ui.historyBtn, 'click', function () { togglePopover('history'); });
    ui.newBtn = button('icon');
    ui.newBtn.id = 'new-btn';
    ui.newBtn.appendChild(icon('compose'));
    on(ui.newBtn, 'click', function () {
      closePopovers();
      post({ type: 'newConversation' });
    });
    var actions = el('div', 'bar-actions');
    actions.append(ui.historyBtn, ui.newBtn);
    var center = el('div', 'bar-center');
    center.appendChild(ui.chip);
    bar.append(el('div', 'bar-side'), center, actions);

    ui.scroll = el('main', 'scroll');
    ui.scroll.id = 'scroll';
    on(ui.scroll, 'scroll', function () {
      var gap = ui.scroll.scrollHeight - ui.scroll.scrollTop - ui.scroll.clientHeight;
      S.stickToBottom = gap < 80;
      S.listHeight = ui.scroll.clientHeight;
    });
    // The banner, a taller input or a resized window changes the list's height.
    if (w.ResizeObserver) new w.ResizeObserver(keepBottomEdge).observe(ui.scroll);
    ui.thread = el('div', 'col thread');
    ui.thread.id = 'thread';
    ui.empty = el('div', 'empty');
    ui.empty.id = 'empty';
    ui.scroll.append(ui.thread, ui.empty);

    var dock = el('div', 'dock');
    var col = el('div', 'col');
    ui.banner = el('div', 'banner');
    ui.banner.id = 'banner';
    ui.box = el('div', 'box');
    ui.thumbs = el('div', 'thumbs');
    ui.thumbs.id = 'thumbs';
    ui.input = el('textarea', 'input');
    ui.input.id = 'input';
    ui.input.rows = 1;
    on(ui.input, 'input', function () { autosize(); updateSend(); });
    on(ui.input, 'keydown', onKeyDown);
    on(ui.input, 'paste', onPaste);
    var tools = el('div', 'tools');
    ui.addBtn = button('plus');
    ui.addBtn.id = 'add-btn';
    ui.addBtn.appendChild(icon('plus'));
    ui.file = el('input', 'file');
    ui.file.id = 'file';
    ui.file.type = 'file';
    ui.file.setAttribute('accept', 'image/*');
    ui.file.multiple = true;
    ui.file.hidden = true;
    on(ui.addBtn, 'click', function () { ui.file.click(); });
    on(ui.file, 'change', function () {
      addFiles(ui.file.files);
      ui.file.value = '';
    });
    ui.think = el('div', 'think-ctl');
    ui.think.id = 'think';
    buildThink();
    ui.send = button('send');
    ui.send.id = 'send';
    on(ui.send, 'click', function () {
      if (S.generating) {
        S.stopRequested = true;
        post({ type: 'stop' });
      }
      else sendCurrent();
    });
    tools.append(ui.addBtn, ui.file, ui.think, el('div', 'grow'), ui.send);
    ui.box.append(ui.thumbs, ui.input, tools);
    ui.hint = el('div', 'hint');
    col.append(ui.banner, ui.box, ui.hint);
    dock.appendChild(col);

    ui.scrim = el('div', 'scrim');
    ui.scrim.hidden = true;
    on(ui.scrim, 'click', closePopovers);
    ui.history = el('div', 'pop history');
    ui.history.id = 'history';
    ui.history.hidden = true;
    ui.models = el('div', 'pop models');
    ui.models.id = 'model-menu';
    ui.models.hidden = true;
    ui.dialog = el('div', 'dialog-wrap');
    ui.dialog.id = 'dialog';
    ui.dialog.hidden = true;
    ui.drop = el('div', 'drop');
    ui.drop.id = 'drop';
    ui.drop.hidden = true;
    ui.dropLabel = el('div', 'drop-label');
    var dropCard = el('div', 'drop-card');
    dropCard.append(icon('image', 'drop-ico'), ui.dropLabel);
    ui.drop.appendChild(dropCard);

    app.append(bar, ui.scroll, dock, ui.scrim, ui.history, ui.models, ui.dialog, ui.drop);
    d.body.appendChild(app);

    on(d, 'dragenter', onDragOver);
    on(d, 'dragover', onDragOver);
    on(d, 'dragleave', function (event) {
      if (!event.relatedTarget) show(ui.drop, false);
    });
    on(d, 'drop', onDrop);
    on(d, 'keydown', function (event) {
      if (event.key === 'Escape') {
        closePopovers();
        closeDialog();
      }
    });
  }

  function applyLanguage() {
    d.documentElement.setAttribute('lang', S.lang);
    ui.input.setAttribute('placeholder', t('placeholder'));
    ui.hint.textContent = t('hint');
    ui.historyBtn.title = t('history');
    ui.historyBtn.setAttribute('aria-label', t('history'));
    ui.newBtn.title = t('newChat');
    ui.newBtn.setAttribute('aria-label', t('newChat'));
    ui.addBtn.title = t('addImage');
    ui.addBtn.setAttribute('aria-label', t('addImage'));
    ui.dropLabel.textContent = t('dropHint');
    if (S.popover === 'history') renderHistory();
    if (S.popover === 'models') renderModelMenu();
  }

  function autosize() {
    ui.input.style.height = 'auto';
    var height = Math.min(Math.max(ui.input.scrollHeight || 0, 22), 180);
    ui.input.style.height = height + 'px';
  }

  // ── Top bar and model menu ───────────────────────────────────────────

  function renderChip() {
    var status = S.model ? S.model.status : 'idle';
    ui.chipDot.className = 'dot ' + (status === 'ready' ? 'ready' : status === 'loading' || status === 'idle' ? 'loading' : 'bad');
    if (status === 'ready' || status === 'failed') ui.chipName.textContent = modelName();
    else if (status === 'noModel') ui.chipName.textContent = t('noModelChip');
    else ui.chipName.textContent = t('loadingChip');
  }

  function togglePopover(name) {
    if (S.popover === name) {
      closePopovers();
      return;
    }
    closePopovers();
    S.popover = name;
    if (name === 'history') {
      renderHistory();
      show(ui.history, true);
      ui.historyBtn.classList.add('on');
    } else if (name === 'models') {
      renderModelMenu();
      show(ui.models, true);
      ui.chip.classList.add('on');
    } else if (name === 'think') {
      renderThink();
    }
    show(ui.scrim, true);
  }

  function closePopovers() {
    var wasThink = S.popover === 'think';
    S.popover = null;
    show(ui.history, false);
    show(ui.models, false);
    show(ui.scrim, false);
    ui.historyBtn.classList.remove('on');
    ui.chip.classList.remove('on');
    if (wasThink) renderThink();
  }

  function sizeText(bytes) {
    if (!bytes) return '';
    return (bytes / 1e9).toFixed(1) + ' GB';
  }

  function renderModelMenu() {
    ui.models.replaceChildren();
    var current = S.model && S.model.id;
    S.models.forEach(function (model) {
      var row = button('mrow' + (model.id === current ? ' current' : ''));
      var dot = el('span', 'mdot' + (model.id === current && ready() ? ' ready' : model.loaded ? ' loaded' : ''));
      var text = el('span', 'mtext');
      text.append(el('span', 'mname', model.name || model.id));
      var sub = [sizeText(model.sizeBytes)];
      if (model.loaded && model.id !== current) sub.push(t('loadedTag'));
      text.append(el('span', 'msub', sub.filter(Boolean).join(' · ')));
      row.append(dot, text);
      if (model.id === current) row.appendChild(icon('check', 'mcheck'));
      on(row, 'click', function () {
        closePopovers();
        if (model.id !== current) confirmSwitch(model);
      });
      ui.models.appendChild(row);
    });
    var manage = button('mfoot', t('manageModels'));
    on(manage, 'click', function () {
      closePopovers();
      post({ type: 'openDashboardModels' });
    });
    ui.models.appendChild(manage);
  }

  function loadSeconds(bytes) {
    if (!bytes) return 10;
    return Math.max(5, Math.round(3 + bytes / 1e9));
  }

  function confirmSwitch(model) {
    var currentModel = null;
    S.models.forEach(function (item) { if (S.model && item.id === S.model.id) currentModel = item; });
    var vars = { model: model.name || model.id, current: modelName(), s: loadSeconds(model.sizeBytes) };
    var body;
    if (!ready()) body = t('switchBodyNoCurrent', vars);
    else if (currentModel && currentModel.pinned) body = t('switchBodyPinned', vars);
    else body = t('switchBody', vars);
    openDialog(t('switchTitle', vars), body, t('switchConfirm'), function () {
      post({ type: 'switchModel', id: model.id });
    });
  }

  function openDialog(title, body, confirmLabel, onConfirm) {
    ui.dialog.replaceChildren();
    var card = el('div', 'dialog');
    card.setAttribute('role', 'dialog');
    var buttons = el('div', 'dbtns');
    var cancel = button('cancel', t('cancel'));
    var confirm = button('confirm', confirmLabel);
    on(cancel, 'click', closeDialog);
    on(confirm, 'click', function () {
      closeDialog();
      onConfirm();
    });
    buttons.append(cancel, confirm);
    card.append(el('h3', '', title), el('p', '', body), buttons);
    ui.dialog.appendChild(card);
    show(ui.dialog, true);
    confirm.focus();
  }

  function closeDialog() {
    show(ui.dialog, false);
    ui.dialog.replaceChildren();
  }

  // ── History ──────────────────────────────────────────────────────────

  function pad(value) { return (value < 10 ? '0' : '') + value; }

  function startOfDay(ms, offsetDays) {
    var date = new Date(ms);
    return new Date(date.getFullYear(), date.getMonth(), date.getDate() + (offsetDays || 0)).getTime();
  }

  function renderHistory() {
    ui.history.replaceChildren();
    var now = nowMs();
    var today = startOfDay(now, 0);
    var yesterday = startOfDay(now, -1);
    var groups = [['today', []], ['yesterday', []], ['earlier', []]];
    S.history.forEach(function (item) {
      var time = Number(item.updatedAt) || 0;
      groups[time >= today ? 0 : time >= yesterday ? 1 : 2][1].push(item);
    });
    var list = el('div', 'list');
    if (!S.history.length) list.appendChild(el('div', 'none', t('historyEmpty')));
    groups.forEach(function (group) {
      if (!group[1].length) return;
      list.appendChild(el('div', 'grp', t(group[0])));
      group[1].forEach(function (item) {
        var date = new Date(Number(item.updatedAt) || 0);
        var when = group[0] === 'earlier'
          ? (strings().dateShort ? strings().dateShort(date.getMonth() + 1, date.getDate()) : (date.getMonth() + 1) + '/' + date.getDate())
          : pad(date.getHours()) + ':' + pad(date.getMinutes());
        var row = button('it' + (item.current || item.id === S.conversationId ? ' on' : ''));
        row.append(el('span', 'it-title', item.title || t('imageTitle')), el('span', 'it-time', when));
        on(row, 'click', function () {
          closePopovers();
          post({ type: 'openConversation', id: item.id });
        });
        list.appendChild(row);
      });
    });
    var foot = el('div', 'foot');
    var link = button('agents', t('agentsLink'));
    on(link, 'click', function () {
      closePopovers();
      post({ type: 'openAgents' });
    });
    foot.append(el('div', 'foot-text', t('agentsText')), link);
    ui.history.append(list, foot);
  }

  // ── Thinking control ─────────────────────────────────────────────────

  function hasLevels() {
    return Boolean(S.reasoning && S.reasoning.native_efforts && S.reasoning.native_efforts.length);
  }

  function normalizeChoice(choice) {
    if (!S.reasoning) return 'off';
    if (!choice || choice === 'off') return 'off';
    if (hasLevels()) return choice === 'on' ? 'medium' : (LEVELS.indexOf(choice) >= 0 ? choice : 'medium');
    return 'on';
  }

  function setChoice(choice) {
    if (choice === S.choice) return;
    S.choice = choice;
    post({ type: 'setReasoning', choice: choice });
  }

  function buildThink() {
    ui.thinkTog = button('tog');
    ui.thinkLabel = el('span', 'tog-label');
    ui.thinkTog.append(icon('think'), ui.thinkLabel);
    on(ui.thinkTog, 'click', function () {
      var active = S.choice !== 'off';
      if (!hasLevels()) {
        setChoice(active ? 'off' : 'on');
        renderThink();
      } else if (!active) {
        // Turning thinking on starts at the default level.
        setChoice('medium');
        if (S.popover !== 'think') togglePopover('think');
        else renderThink();
      } else {
        togglePopover('think');
      }
    });
    ui.thinkMenu = el('div', 'menu');
    ui.think.append(ui.thinkTog, ui.thinkMenu);
  }

  function renderThink() {
    show(ui.think, Boolean(S.reasoning));
    if (!S.reasoning) return;
    var active = S.choice !== 'off';
    var label = t('thinking');
    if (hasLevels() && active) label += ' · ' + t(LEVEL_KEYS[S.choice]) + ' ▾';
    ui.thinkLabel.textContent = label;
    ui.thinkTog.classList.toggle('on', active);
    ui.thinkTog.setAttribute('aria-pressed', active ? 'true' : 'false');
    ui.thinkMenu.replaceChildren();
    show(ui.thinkMenu, hasLevels() && S.popover === 'think');
    if (!hasLevels()) return;
    LEVELS.forEach(function (level, index) {
      var item = button('mi' + (S.choice === level ? ' sel' : ''));
      item.append(el('span', 'mi-label', t(LEVEL_KEYS[level])), el('small', '', t(LEVEL_KEYS[level] + 'Sub')));
      on(item, 'click', function () {
        setChoice(level);
        closePopovers();
        renderThink();
      });
      ui.thinkMenu.appendChild(item);
      if (index === 0) ui.thinkMenu.appendChild(el('div', 'sep'));
    });
  }

  // ── Composer ─────────────────────────────────────────────────────────

  function updateSend() {
    ui.send.replaceChildren();
    if (S.generating) {
      ui.send.classList.add('stop');
      ui.send.disabled = false;
      ui.send.appendChild(icon('stop'));
      ui.send.setAttribute('aria-label', t('stop'));
      ui.send.title = t('stop');
      return;
    }
    ui.send.classList.remove('stop');
    ui.send.appendChild(icon('send'));
    ui.send.setAttribute('aria-label', t('send'));
    ui.send.title = t('send');
    var hasContent = String(ui.input.value || '').trim().length > 0 || S.attachments.length > 0;
    ui.send.disabled = !(ready() && hasContent && attachmentsReady());
  }

  /// Every image has been read; until then nothing is sent.
  function attachmentsReady() {
    return S.attachments.every(function (item) { return Boolean(item.src); });
  }

  function onKeyDown(event) {
    if (event.key !== 'Enter' || event.shiftKey || event.isComposing || event.keyCode === 229) return;
    event.preventDefault();
    if (!S.generating) sendCurrent();
  }

  function sendCurrent() {
    if (S.generating || !ready() || !attachmentsReady()) return;
    var text = String(ui.input.value || '').trim();
    var images = S.attachments.map(function (item) { return item.src; }).filter(Boolean);
    if (!text && !images.length) return;
    if (text.length > MAX_TEXT_LENGTH) {
      setErrorNotice(t('tooLong'));
      return;
    }
    ui.input.value = '';
    S.attachments = [];
    renderThumbs();
    autosize();
    submit(text, images, false);
  }

  function submit(text, images, retry) {
    S.stopRequested = false;
    S.lastSend = { text: text, images: images.slice() };
    S.errorNotice = null;
    renderBanner();
    if (!retry) ui.thread.appendChild(userRow(text, images));
    startPending();
    var message = { type: 'send', text: text, images: images };
    if (retry) message.retry = true;
    post(message);
    setGenerating(true);
    updateEmpty();
    scrollToEnd(true);
  }

  function setGenerating(value) {
    S.generating = value;
    updateSend();
  }

  function renderThumbs() {
    ui.thumbs.replaceChildren();
    show(ui.thumbs, S.attachments.length > 0);
    S.attachments.forEach(function (item, index) {
      var thumb = el('div', 'thumb' + (item.src ? '' : ' reading'));
      var img = el('img');
      if (item.src) img.setAttribute('src', item.src);
      img.setAttribute('alt', '');
      var remove = button('rm');
      remove.appendChild(icon('close'));
      remove.setAttribute('aria-label', t('dismiss'));
      on(remove, 'click', function () {
        S.attachments.splice(index, 1);
        renderThumbs();
        updateSend();
      });
      thumb.append(img, remove);
      ui.thumbs.appendChild(thumb);
    });
  }

  function imageFiles(list) {
    var files = [];
    for (var i = 0; list && i < list.length; i += 1) {
      if (list[i] && /^image\//.test(list[i].type || '')) files.push(list[i]);
    }
    return files;
  }

  function addFiles(list) {
    var files = imageFiles(list);
    if (!files.length) return;
    if (!S.vision) {
      setErrorNotice(t('errImages', { model: modelName() }));
      return;
    }
    files.forEach(function (file) {
      if (S.attachments.length >= MAX_IMAGES) {
        setErrorNotice(t('tooManyImages'));
        return;
      }
      if (file.size > MAX_IMAGE_BYTES) {
        setErrorNotice(t('imageTooLarge'));
        return;
      }
      var entry = { src: null };
      S.attachments.push(entry);
      var reader = new FileReader();
      reader.onload = function () {
        entry.src = String(reader.result || '');
        renderThumbs();
        updateSend();
      };
      reader.onerror = function () {
        S.attachments.splice(S.attachments.indexOf(entry), 1);
        renderThumbs();
        setErrorNotice(t('errAttachment'));
      };
      reader.readAsDataURL(file);
    });
    S.attachments = S.attachments.filter(function (item) { return item.src !== ''; });
    renderThumbs();
    updateSend();
  }

  function hasFiles(event) {
    var types = event.dataTransfer && event.dataTransfer.types;
    if (!types) return false;
    for (var i = 0; i < types.length; i += 1) if (types[i] === 'Files') return true;
    return false;
  }

  function onDragOver(event) {
    if (!hasFiles(event)) return;
    // Without this WebKit would open the dropped file in place of the page.
    event.preventDefault();
    if (S.vision) show(ui.drop, true);
  }

  function onDrop(event) {
    event.preventDefault();
    show(ui.drop, false);
    if (event.dataTransfer) addFiles(event.dataTransfer.files);
  }

  function onPaste(event) {
    var files = imageFiles(event.clipboardData && event.clipboardData.files);
    if (!files.length) return;
    event.preventDefault();
    addFiles(files);
  }

  // ── Messages ─────────────────────────────────────────────────────────

  function userRow(text, images) {
    var row = el('div', 'user');
    if (images && images.length) {
      var imgs = el('div', 'uimgs');
      images.forEach(function (src) {
        if (!/^data:image\//.test(String(src))) return;
        var img = el('img');
        img.setAttribute('src', src);
        img.setAttribute('alt', '');
        imgs.appendChild(img);
      });
      row.appendChild(imgs);
    }
    if (text) row.appendChild(el('div', 'ububble', text));
    return row;
  }

  function assistantRow() {
    var row = el('div', 'asst');
    row.appendChild(el('div', 'av', 'Fe'));
    var body = el('div', 'abody');
    row.appendChild(body);
    return { row: row, body: body };
  }

  function seconds(value) { return (Math.round(value * 10) / 10).toFixed(1); }

  function thinkBlock(view) {
    if (view.think) return view.think;
    var think = el('div', 'think');
    var head = button('think-head');
    var body = el('div', 'think-body');
    on(head, 'click', function () {
      if (think.classList.contains('closed')) think.classList.toggle('open');
    });
    think.append(head, body);
    view.body.insertBefore(think, view.body.firstChild);
    view.think = think;
    view.thinkHead = head;
    view.thinkBody = body;
    return think;
  }

  function setThinkLive(view) {
    view.thinkHead.replaceChildren(el('span', 'spin'),
      el('span', '', t('thinkingLive', { s: seconds((nowMs() - view.thinkStart) / 1000) })));
  }

  function closeThink(view, value) {
    if (!view.think) return;
    view.think.classList.add('closed');
    view.thinkHead.replaceChildren();
    var label = value === null || value === undefined ? t('thinking') : t('thinkingDone', { s: seconds(value) });
    view.thinkHead.textContent = label + ' ›';
  }

  function pill(metrics) {
    if (!metrics) return null;
    var parts = [];
    if (metrics.thinkingSeconds !== undefined && metrics.thinkingSeconds !== null) {
      parts.push(['pillThink', seconds(metrics.thinkingSeconds) + ' s']);
    } else if (metrics.firstTokenSeconds !== undefined && metrics.firstTokenSeconds !== null) {
      parts.push(['pillFirst', Number(metrics.firstTokenSeconds).toFixed(2) + ' s']);
    }
    if (metrics.tokensPerSecond !== undefined && metrics.tokensPerSecond !== null) {
      parts.push(['pillSpeed', Math.round(metrics.tokensPerSecond) + ' tok/s']);
    }
    if (metrics.outputTokens !== undefined && metrics.outputTokens !== null) {
      parts.push(['pillTokens', String(metrics.outputTokens)]);
    }
    if (!parts.length) return null;
    var node = el('div', 'pill');
    parts.forEach(function (part, index) {
      if (index) node.append(' · ');
      var template = t(part[0]).split('{v}');
      if (template[0]) node.append(template[0]);
      node.appendChild(el('b', '', part[1]));
      if (template[1]) node.append(template[1]);
    });
    return node;
  }

  function finishView(view, turn) {
    if (view.dots) {
      view.dots.remove();
      view.dots = null;
    }
    var reasoning = turn.reasoning || '';
    var metrics = turn.metrics || null;
    if (reasoning) {
      thinkBlock(view);
      view.thinkBody.textContent = reasoning;
      var measured = view.thinkStart ? (view.thinkEnd || nowMs()) - view.thinkStart : null;
      var value = metrics && metrics.thinkingSeconds !== undefined && metrics.thinkingSeconds !== null
        ? metrics.thinkingSeconds : measured === null ? null : measured / 1000;
      closeThink(view, value);
    } else if (view.think) {
      view.think.remove();
      view.think = null;
    }
    var text = turn.text || '';
    if (text) {
      if (!view.text) {
        view.text = el('div', 'atext');
        view.body.appendChild(view.text);
      }
      view.text.innerHTML = renderMarkdown(text, true);
    } else if (view.text) {
      view.text.remove();
      view.text = null;
    }
    var meta = el('div', 'meta');
    var speed = pill(metrics);
    if (speed) meta.appendChild(speed);
    if (turn.stopped) meta.appendChild(el('span', 'mark', t('stopped')));
    if (turn.truncated) meta.appendChild(el('span', 'mark', t('truncated')));
    if (meta.firstChild) view.body.appendChild(meta);
  }

  function startPending() {
    var view = assistantRow();
    view.dots = el('div', 'dots');
    view.dots.append(el('i'), el('i'), el('i'));
    view.body.appendChild(view.dots);
    view.textValue = '';
    view.reasoningValue = '';
    ui.thread.appendChild(view.row);
    S.pending = view;
    return view;
  }

  function scheduleRender() {
    var view = S.pending;
    if (!view || view.queued) return;
    view.queued = true;
    // A long reply is parsed and sanitized again on every render, so it
    // renders less often; short ones follow every frame.
    var later = view.textValue.length > 4000
      ? function (fn) { w.setTimeout(fn, 80); }
      : function (fn) { w.requestAnimationFrame(fn); };
    later(function () {
      view.queued = false;
      if (S.pending !== view) return;
      if (view.reasoningValue && view.think && !view.think.classList.contains('closed')) {
        view.thinkBody.textContent = view.reasoningValue;
      }
      if (view.textValue) {
        if (!view.text) {
          view.text = el('div', 'atext');
          view.body.appendChild(view.text);
        }
        view.text.innerHTML = renderMarkdown(view.textValue, false);
      }
      scrollToEnd(false);
    });
  }

  function onDelta(message) {
    // A reply this page did not start (it reloaded meanwhile) still gets Stop.
    var view = S.pending || startPending();
    if (!S.generating) setGenerating(true);
    var reasoning = String(message.reasoning || '');
    var text = String(message.text || '');
    if (!reasoning && !text) return;
    if (view.dots) {
      view.dots.remove();
      view.dots = null;
    }
    if (reasoning) {
      if (!view.think) {
        view.thinkStart = nowMs();
        thinkBlock(view);
        setThinkLive(view);
        view.timer = w.setInterval(function () {
          if (S.pending === view && view.think && !view.think.classList.contains('closed')) setThinkLive(view);
        }, 200);
      }
      view.reasoningValue += reasoning;
    }
    if (text) {
      if (view.think && !view.think.classList.contains('closed')) {
        // The answer started: the thinking folds away.
        view.thinkEnd = nowMs();
        view.thinkBody.textContent = view.reasoningValue;
        closeThink(view, (view.thinkEnd - view.thinkStart) / 1000);
      }
      view.textValue += text;
    }
    scheduleRender();
  }

  function onTurnFinished(message) {
    S.stopRequested = false;
    var view = S.pending || startPending();
    if (view.timer) w.clearInterval(view.timer);
    S.pending = null;
    finishView(view, {
      text: message.text !== undefined && message.text !== null ? message.text : view.textValue,
      reasoning: message.reasoning !== undefined && message.reasoning !== null ? message.reasoning : view.reasoningValue,
      metrics: message.metrics,
      stopped: message.stopped,
      truncated: message.truncated
    });
    setGenerating(false);
    scrollToEnd(false);
  }

  function showReplyError(kind, text) {
    var view = S.pending;
    S.stopRequested = false;
    if (view.timer) w.clearInterval(view.timer);
    S.pending = null;
    var box = el('div', 'err');
    box.appendChild(el('span', 'err-text', text));
    if (kind === 'contextOverflow') {
      var fresh = button('retry', t('newChat'));
      on(fresh, 'click', function () { post({ type: 'newConversation' }); });
      box.appendChild(fresh);
    } else {
      var retry = button('retry', t('retry'));
      on(retry, 'click', function () {
        var last = S.lastSend;
        if (!last || S.generating || !ready()) return;
        view.row.remove();
        submit(last.text, last.images, true);
      });
      box.appendChild(retry);
    }
    view.body.replaceChildren(box);
    view.think = null;
    view.text = null;
    setGenerating(false);
    scrollToEnd(false);
  }

  var REPLY_ERRORS = {
    connection: 'errConnection', server: 'errServer', contextOverflow: 'errContext',
    busy: 'errBusy', rejected: 'errRejected',
    modelNotReady: 'errModelNotReady', imagesNotSupported: 'errImages', attachment: 'errAttachment'
  };

  function onError(message) {
    var kind = message.kind;
    if (kind === 'rejected' && S.pending && S.stopRequested) {
      // Stopped before anything was sent: the reply simply ends.
      onTurnFinished({ text: '', reasoning: null, stopped: true, truncated: false, metrics: null });
      return;
    }
    if (REPLY_ERRORS[kind]) {
      showReplyErrorOrNotice(kind, t(REPLY_ERRORS[kind], { model: modelName() }));
      return;
    }
    var name = message.modelName || message.modelID || '';
    if (kind === 'modelRequiresDashboard') {
      setErrorNotice(t('dflash', { model: name }), { label: t('openDashboard'), message: { type: 'openDashboardModels' } });
    } else if (kind === 'modelLoadFailed') {
      // A failed load that left no model shows through the model state.
      if (ready() && S.model.id !== message.modelID) {
        if (message.unpinHint) {
          // Memory refused the new model while the pinned current one stays loaded.
          setErrorNotice(t('switchFailedPinned', { model: name, current: message.pinnedModelName || modelName() }),
            { label: t('openDashboard'), message: { type: 'openDashboardModels' } });
        } else {
          setErrorNotice(t('switchFailed', { model: name, current: modelName() }));
        }
      }
    }
    if (S.pending && !ready()) showReplyError('modelNotReady', t('errModelNotReady'));
  }

  /// Whatever ended a reply, the page goes back to idle.
  function showReplyErrorOrNotice(kind, text) {
    if (S.pending) showReplyError(kind, text);
    else {
      setGenerating(false);
      setErrorNotice(text);
    }
  }

  function scrollToEnd(force) {
    if (!force && !S.stickToBottom) return;
    S.stickToBottom = true;
    ui.scroll.scrollTop = ui.scroll.scrollHeight;
  }

  /// When the list gets shorter or taller, its bottom edge stays on the same
  /// content, so the banner never covers the last reply and its speed pill.
  function keepBottomEdge() {
    var height = ui.scroll.clientHeight;
    var change = S.listHeight === null ? 0 : S.listHeight - height;
    S.listHeight = height;
    if (!change) return;
    if (S.stickToBottom) scrollToEnd(false);
    else ui.scroll.scrollTop = Math.max(0, ui.scroll.scrollTop + change);
  }

  function renderConversation(message) {
    S.conversationId = message.id || null;
    if (S.pending && S.pending.timer) w.clearInterval(S.pending.timer);
    S.pending = null;
    S.lastSend = null;
    ui.thread.replaceChildren();
    (message.turns || []).forEach(function (turn) {
      if (turn.role === 'user') {
        ui.thread.appendChild(userRow(turn.text || '', turn.images || []));
      } else {
        var view = assistantRow();
        ui.thread.appendChild(view.row);
        finishView(view, turn);
      }
    });
    setGenerating(false);
    updateEmpty();
    scrollToEnd(true);
    ui.input.focus();
  }

  // ── Empty state ──────────────────────────────────────────────────────

  function updateEmpty() {
    var empty = !ui.thread.firstChild && !S.pending;
    show(ui.empty, empty);
    ui.scroll.classList.toggle('is-empty', empty);
    if (!empty) return;
    ui.empty.replaceChildren();
    var status = S.model ? S.model.status : 'idle';
    var sub = el('div', 'sub');
    if (status === 'ready') {
      sub.append(el('span', 'dot ready'), el('span', '', t('emptySub', { model: modelName() })));
    } else if (status === 'loading' || status === 'idle') {
      sub.append(el('span', 'dot loading'), el('span', '', modelName() ? t('emptySubLoading', { model: modelName() }) : t('loadingChip')));
    }
    var grid = el('div', 'grid');
    var examples = (strings().examples || []).slice(0, 4);
    if (!S.vision && examples[3] && examples[3].vision && strings().exampleText) examples[3] = strings().exampleText;
    examples.forEach(function (example) {
      var card = button('sug');
      if (example.vision) card.setAttribute('data-vision', 'true');
      card.append(el('span', 'sug-text', example.text), el('small', 'sug-tag', example.tag));
      on(card, 'click', function () {
        if (example.vision) {
          ui.file.click();
          return;
        }
        if (ready() && !S.generating) submit(example.text, [], false);
        else {
          ui.input.value = example.text;
          autosize();
          updateSend();
        }
      });
      grid.appendChild(card);
    });
    ui.empty.append(el('div', 'logo', 'Fe'), el('h2', '', t('emptyTitle')), sub, grid);
  }

  // ── Banner: notices and the upgrade offer ────────────────────────────

  function setErrorNotice(text, action) {
    S.errorNotice = { text: text, action: action || null };
    renderBanner();
  }

  function stateNotice() {
    var model = S.model;
    if (!model) return null;
    if (model.status === 'noModel') {
      var previous = model.previousName || model.previousModelID;
      return {
        text: t('noModel', { failed: model.failedName || model.failedModelID, previous: previous }),
        action: { label: t('reloadModel', { model: previous }), message: { type: 'reloadModel', id: model.previousModelID } },
        fixed: true
      };
    }
    if (model.status === 'failed') {
      var failedName = model.name || model.id;
      // `unloaded`: the model was ready, then unloaded elsewhere.
      var unloaded = model.reason === 'unloaded';
      return {
        text: t(unloaded ? 'modelUnloaded' : 'loadFailed', { model: failedName }),
        action: {
          label: unloaded ? t('reloadModel', { model: failedName }) : t('retry'),
          message: { type: 'reloadModel', id: model.id }
        },
        fixed: true
      };
    }
    return null;
  }

  function renderBanner() {
    ui.banner.replaceChildren();
    var notice = stateNotice() || S.errorNotice;
    if (notice) {
      var box = el('div', 'notice');
      box.appendChild(el('span', 'notice-text', notice.text));
      if (notice.action) {
        var act = button('act', notice.action.label);
        on(act, 'click', function () { post(notice.action.message); });
        box.appendChild(act);
      }
      if (!notice.fixed) {
        var close = button('x');
        close.appendChild(icon('close'));
        close.setAttribute('aria-label', t('dismiss'));
        on(close, 'click', function () {
          S.errorNotice = null;
          renderBanner();
        });
        box.appendChild(close);
      }
      ui.banner.appendChild(box);
    }
    var upgrade = S.upgrade;
    if (upgrade) {
      var card = el('div', 'upgrade');
      var vars = { model: upgrade.name, size: upgrade.size, pct: Math.round((upgrade.fraction || 0) * 100) };
      var copy = {
        offer: 'upgradeOffer', preparing: 'upgradePreparing', downloading: 'upgradeDownloading',
        paused: 'upgradePaused', switching: 'upgradeSwitching', failed: 'upgradeFailed'
      }[upgrade.phase] || 'upgradeOffer';
      card.appendChild(el('span', 'upgrade-text', t(copy, vars)));
      if (upgrade.phase === 'downloading') {
        var bar = el('div', 'progress');
        var fill = el('i');
        fill.style.width = vars.pct + '%';
        bar.appendChild(fill);
        card.appendChild(bar);
      }
      if (upgrade.phase === 'offer' || upgrade.phase === 'failed') {
        var download = button('act primary', upgrade.phase === 'failed' ? t('retry') : t('upgradeDownload'));
        on(download, 'click', function () {
          S.upgrade.phase = 'preparing';
          post({ type: 'downloadUpgrade' });
          renderBanner();
        });
        var dismiss = button('act', t('upgradeDismiss'));
        on(dismiss, 'click', function () {
          S.upgrade = null;
          post({ type: 'dismissUpgrade' });
          renderBanner();
        });
        card.append(download, dismiss);
      }
      ui.banner.appendChild(card);
    }
    show(ui.banner, Boolean(ui.banner.firstChild));
    keepBottomEdge();
  }

  // ── Messages from Swift ──────────────────────────────────────────────

  function onState(message) {
    var language = normalizeLanguage(message.language);
    var languageChanged = language !== S.lang || !S.started;
    S.started = true;
    S.lang = language;
    var previousModel = S.model && S.model.id;
    S.models = Array.isArray(message.models) ? message.models : [];
    S.model = message.model || null;
    if (S.model && previousModel && S.model.id !== previousModel && S.errorNotice && !S.errorNotice.sticky) {
      S.errorNotice = null;
    }
    S.reasoning = message.reasoning || null;
    S.choice = normalizeChoice(message.reasoningChoice);
    S.vision = Boolean(message.supportsVision);
    if (message.generating === true && !S.generating) {
      // After a reload the reply that is still running keeps its Stop button.
      if (!S.pending) startPending();
      S.generating = true;
    } else if (message.generating === false && S.generating && !S.pending) {
      S.generating = false;
    }
    if (languageChanged) applyLanguage();
    renderChip();
    renderThink();
    show(ui.addBtn, S.vision);
    if (!S.vision && S.attachments.length) {
      S.attachments = [];
      renderThumbs();
    }
    if (S.popover === 'models') renderModelMenu();
    renderBanner();
    updateEmpty();
    updateSend();
  }

  function receive(message) {
    if (!message || typeof message !== 'object') return;
    switch (message.type) {
      case 'state': onState(message); break;
      case 'history':
        S.history = Array.isArray(message.items) ? message.items : [];
        if (S.popover === 'history') renderHistory();
        break;
      case 'conversation': renderConversation(message); break;
      case 'delta': onDelta(message); break;
      case 'turnFinished': onTurnFinished(message); break;
      case 'error': onError(message); break;
      case 'trimmed': {
        var note = el('div', 'sys', t('trimmed', { n: message.count }));
        if (S.pending) ui.thread.insertBefore(note, S.pending.row);
        else ui.thread.appendChild(note);
        break;
      }
      case 'upgradeOffer':
        S.upgrade = { name: message.name, size: message.sizeText, phase: 'offer', fraction: 0 };
        renderBanner();
        break;
      case 'upgradeStatus':
        if (!S.upgrade) break;
        if (message.phase === 'done') S.upgrade = null;
        else {
          S.upgrade.phase = message.phase;
          if (typeof message.fraction === 'number') S.upgrade.fraction = message.fraction;
        }
        renderBanner();
        break;
      default:
        break;
    }
  }

  build();
  applyLanguage();
  renderChip();
  renderThink();
  renderThumbs();
  renderBanner();
  updateEmpty();
  updateSend();
  w.ironChat = { receive: receive, renderMarkdown: renderMarkdown };
  post({ type: 'ready' });
  ui.input.focus();
})();
