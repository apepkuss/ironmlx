const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const resourceRoot = path.join(__dirname, '../../ironmlx-app/Sources/IronMLXAppCore/Resources');
const html = fs.readFileSync(path.join(resourceRoot, 'dashboard2.html'), 'utf8');
const catalog = JSON.parse(fs.readFileSync(path.join(resourceRoot, 'supported-models.json'), 'utf8'));
function slice(start, end) { return html.slice(html.indexOf(start), html.indexOf(end, html.indexOf(start))); }
const translations = slice('  const I18N =', '  const LANG_MAP =');
const sourceTabs = slice('  function switchDlSource(', '  const ACTIVE_DOWNLOAD_PHASES');
const queueHelpers = slice('  const ACTIVE_DOWNLOAD_PHASES', '  function renderDownloadQueue(');
const catalogScript = slice('  // ── Supported model catalogue (', '  function onDownloadProgress(');
const translate = slice('  function t(key, fallback)', '  function setLanguage(');

function page(lang = 'zh-Hans') {
  const elements = new Map(), requests = [], timers = [], toasts = [];
  function element(id) {
    if (!elements.has(id)) {
      let content = '';
      const el = { id, value: '', textContent: '', style: {}, dataset: {}, attributes: {}, events: {},
        setAttribute(key, value) { this.attributes[key] = value; },
        addEventListener(key, fn) { this.events[key] = fn; },
        focus() { this.focused = true; },
        querySelectorAll(selector) { return selector === '.catalog-action' ? this.buttons || []
          : selector === '.catalog-variant select' ? this.selects || [] : []; },
      };
      Object.defineProperty(el, 'innerHTML', { get() { return content; }, set(value) {
        content = value;
        el.buttons = [...value.matchAll(/<button class="btn-secondary catalog-action"[^>]*>/g)].map((match, index) => {
          const button = element('action-' + index);
          button.dataset = Object.fromEntries([...match[0].matchAll(/data-([a-z-]+)="([^"]*)"/g)]
            .map(m => [m[1].replace(/-([a-z])/g, (_, char) => char.toUpperCase()), m[2]]));
          return button;
        });
        el.selects = [...value.matchAll(/<select data-model-id="([^"]+)"[^>]*>(.*?)<\/select>/g)].map(match => {
          const select = element('variant-' + match[1]);
          select.dataset = { modelId: match[1] };
          select.options = [...match[2].matchAll(/<option value="([^"]*)"([^>]*)>(.*?)<\/option>/g)]
            .map(option => ({ value: option[1], selected: option[2].includes(' selected'), label: option[3] }));
          select.value = select.options.find(option => option.selected).value;
          return select;
        });
      } });
      elements.set(id, el);
    }
    return elements.get(id);
  }
  const tabs = ['catalog', 'hf', 'ms'].map(name => element('dl-tab-' + name));
  const context = { currentLang: lang,
    window: { __IRONMLX_SUPPORTED_MODELS__: structuredClone(catalog),
      __IRONMLX_CATALOG_MEMORY_HINTS__: { availableBytes: 14 * 1024 ** 3, estimatedBytes: Object.fromEntries(catalog.entries.map(entry => [entry.id, entry.weightBytes])) }, __LOCAL_MODELS__: {},
      confirm: () => true, webkit: { messageHandlers: {
        downloadModel: { postMessage: body => requests.push(['hf', JSON.parse(body)]) },
      } } },
    document: { getElementById: element, querySelectorAll: selector => selector === '.dl-source-tab'
      ? tabs : selector === '.catalog-action' ? element('catalog-model-list').buttons || [] : [] },
    apiPost: (...args) => requests.push(args), clearCompletedHuggingFaceSearchState() {},
    setTimeout: (fn, ms) => timers.push({ fn, ms }), refreshDownloadQueue() {},
    showToast: (...args) => toasts.push(args),
    formatVersionBytes: bytes => String(bytes) + ' bytes',
    escapeAttr: value => String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;')
      .replace(/>/g, '&gt;').replace(/"/g, '&quot;'),
  };
  vm.createContext(context);
  vm.runInContext(translations + translate + sourceTabs + queueHelpers + catalogScript, context);
  return { context, element, requests, timers, toasts, render: () => context.renderSupportedModels(),
    choose: (modelId, entryId) => {
      context.renderSupportedModels();
      const select = element('variant-' + modelId);
      select.value = entryId; select.events.change();
    },
    queue: tasks => { context.tasksForTest = tasks; vm.runInContext('downloadQueueSnapshot.tasks = tasksForTest;', context); } };
}

test('default tab is the catalogue and keyboard navigation selects one panel', () => {
  assert.match(html, /id="dl-tab-catalog"[^>]*aria-selected="true"/);
  assert.match(html, /id="dl-panel-hf"[^>]*display:none/);
  const p = page(); p.context.switchDlSource('catalog');
  assert.equal(p.element('dl-panel-catalog').style.display, '');
  p.element('dl-tab-catalog').events.keydown({ key: 'ArrowRight', preventDefault() {} });
  assert.equal(p.element('dl-tab-hf').attributes['aria-selected'], 'true');
  assert.equal(p.element('dl-panel-catalog').style.display, 'none');
  assert.equal(p.element('dl-tab-hf').tabIndex, 0);
  assert.equal(p.element('dl-tab-catalog').tabIndex, -1);
  p.element('dl-tab-hf').events.keydown({ key: 'End', preventDefault() {} });
  assert.equal(p.element('dl-tab-ms').focused, true);
});

test('catalogue lists all models without presenting assistants as chat models', () => {
  const p = page(); p.render();
  assert.equal(p.element('catalog-model-list').buttons.length, 28);
  assert.match(p.element('catalog-summary').textContent, /28 个模型 · 58 个版本/);
  p.choose('gemma4-e2b-4bit', 'gemma4-e2b-4bit');
  assert.match(p.element('catalog-model-list').innerHTML, /架构兼容 · 待运行验证/);
  p.choose('dflash2-qwen38', 'dflash2-qwen38');
  assert.match(p.element('catalog-model-list').innerHTML, /不能作为主模型加载/);
});

test('every model requires a quantization choice, including single FP16 and BF16 variants', () => {
  const p = page(); p.render();
  assert.equal((p.element('catalog-model-list').innerHTML.match(/class="catalog-row"/g) || []).length, 28);
  const qwen35 = p.element('variant-qwen35-2b');
  assert.deepEqual(qwen35.options.map(option => option.label), ['选择量化版本', 'Affine · 4 bit', 'Affine · 5 bit', 'Affine · 6 bit', 'OptiQ · 4 bit']);
  assert.equal(qwen35.value, '');
  const qwen38 = p.element('variant-qwen38-27b');
  assert.deepEqual(qwen38.options.map(option => option.label), ['选择量化版本', 'Affine · 4 bit', 'Affine · 8 bit']);
  assert.equal(qwen38.value, '');
  const buttons = p.element('catalog-model-list').buttons;
  assert.ok(buttons.every(button => button.disabled && !button.dataset.entryId));
  p.context.refreshSupportedModelActions();
  assert.ok(buttons.filter(button => !button.dataset.entryId).every(button => button.disabled));
  p.context.downloadSupportedModel('qwen35-2b-4bit', 'huggingface');
  p.context.downloadSupportedModel('qwen38-27b-8bit', 'modelscope');
  p.context.downloadSupportedModel('laya', 'huggingface');
  assert.equal(p.requests.length, 0);
  const firstRow = p.element('catalog-model-list').innerHTML.split('</article>')[0];
  assert.doesNotMatch(firstRow, /class="catalog-repo"|预计下载|权重预算|架构兼容/);
  assert.doesNotMatch(firstRow, /请先选择量化版本/);
  assert.equal(p.element('catalog-model-list').selects.length, 28);
  assert.deepEqual(p.element('variant-laya').options.map(option => option.label), ['选择量化版本', 'FP16']);
  assert.deepEqual(p.element('variant-dflash2-qwen38').options.map(option => option.label), ['选择量化版本', 'BF16']);
  p.choose('laya', 'laya');
  const layaButton = p.element('catalog-model-list').buttons.find(button => button.dataset.entryId === 'laya');
  assert.equal(layaButton.disabled, false);
  layaButton.events.click();
  assert.equal(p.requests[0][1].repo_id, 'aac6fef/laya-multilingual-mlx');
  const distinct = structuredClone(catalog.entries.find(entry => entry.id === 'qwen38-27b-4bit'));
  distinct.id = 'different-finetune'; distinct.modelId = 'different-finetune';
  p.context.window.__IRONMLX_SUPPORTED_MODELS__.entries.push(distinct); p.render();
  assert.equal((p.element('catalog-model-list').innerHTML.match(/class="catalog-row"/g) || []).length, 29);
});

test('variant changes update repository, bytes, memory, status and the HF download target', () => {
  const p = page();
  const repo4 = 'mlx-community/Qwen3.8-27B-4bit';
  const repo8 = 'mlx-community/Qwen3.8-27B-8bit';
  p.context.window.__LOCAL_MODELS__ = { row: { repo_id: repo4, source: 'hf', readiness: { status: 'ready' } } };
  // Distinct validation states ensure the status follows the selected variant too.
  p.context.window.__IRONMLX_SUPPORTED_MODELS__.entries.find(entry => entry.id === 'qwen38-27b-8bit').status = 'compatible';
  p.render();
  p.choose('qwen38-27b', 'qwen38-27b-4bit');
  assert.equal(p.element('catalog-model-list').buttons.find(button => button.dataset.repoId === repo4).textContent, '已下载 · 检查更新');
  const select = p.element('variant-qwen38-27b');
  select.value = 'qwen38-27b-8bit'; select.events.change();
  const entry8 = catalog.entries.find(entry => entry.id === 'qwen38-27b-8bit');
  const row = p.element('catalog-model-list').innerHTML;
  assert.match(row, /Qwen3\.8-27B-8bit/); assert.doesNotMatch(row, /mlx-community\/Qwen3\.8-27B-4bit/);
  assert.ok(row.includes(String(entry8.downloadBytes))); assert.ok(row.includes(String(entry8.weightBytes)));
  assert.match(row, /架构兼容 · 待运行验证/); assert.match(row, /超出本机内存预算/);
  const buttons = p.element('catalog-model-list').buttons.filter(button => button.dataset.repoId === repo8);
  assert.equal(buttons.length, 1);
  assert.equal(buttons[0].textContent, 'HuggingFace · 下载');
  assert.equal(select.focused, true);
  buttons[0].events.click();
  assert.equal(p.requests[0][1].repo_id, repo8);
  select.value = 'qwen38-27b-4bit'; select.events.change();
  assert.equal(p.element('catalog-model-list').buttons.find(button => button.dataset.repoId === repo4).disabled, false);
  assert.equal(p.element('catalog-model-list').buttons.find(button => button.dataset.repoId === repo4).textContent, '已下载 · 检查更新');
  select.value = 'qwen38-27b-8bit'; select.events.change();
  assert.equal(p.element('catalog-model-list').buttons.find(button => button.dataset.repoId === repo8).disabled, true);
});

test('same-bit formats stay distinct and selection survives rerenders, tabs and queue refreshes', () => {
  const p = page(); p.render();
  const select = p.element('variant-qwen35-2b'); select.value = 'qwen35-2b-optiq-4bit'; select.events.change();
  assert.match(p.element('catalog-model-list').innerHTML, /采用 4\/8-bit 混合精度/);
  p.render();
  assert.equal(p.element('variant-qwen35-2b').value, 'qwen35-2b-optiq-4bit');
  assert.match(p.element('catalog-summary').textContent, /28 个模型 · 58 个版本/);
  p.context.switchDlSource('hf'); p.context.switchDlSource('catalog');
  p.context.refreshSupportedModelActions();
  assert.equal(p.element('variant-qwen35-2b').value, 'qwen35-2b-optiq-4bit');
  p.element('catalog-model-list').buttons[0].events.click();
  assert.equal(p.requests[0][1].repo_id, 'mlx-community/Qwen3.5-2B-OptiQ-4bit');
});

test('public catalogue downloads use the existing queues without credentials', () => {
  const p = page();
  p.choose('qwen35-2b', 'qwen35-2b-4bit');
  p.context.downloadSupportedModel('qwen35-2b-4bit', 'huggingface');
  assert.deepEqual(p.requests[0], ['hf', { repo_id: 'mlx-community/Qwen3.5-2B-4bit', token: null }]);
  p.context.downloadSupportedModel('qwen35-2b-4bit', 'modelscope');
  assert.deepEqual(JSON.parse(JSON.stringify(p.requests[1])), ['/admin/api/models/ms/download', { repo_id: 'mlx-community/Qwen3.5-2B-4bit' }]);
  p.context.downloadSupportedModel('laya', 'modelscope');
  p.context.downloadSupportedModel('laya', 'unknown');
  p.context.downloadSupportedModel('unknown', 'huggingface');
  assert.equal(p.requests.length, 2);
});

test('public catalogue download does not reuse a hidden HF search token', () => {
  const p = page(); p.element('search-hf-token').value = 'hidden-token';
  p.choose('laya', 'laya');
  p.context.downloadSupportedModel('laya', 'huggingface');
  assert.equal(p.requests[0][1].token, null);
});

for (const provider of ['huggingface', 'modelscope']) {
  test(`${provider} preserves image-model licence confirmation and cancellation`, () => {
    const p = page(); let confirmations = 0;
    p.choose('qwen-image21', 'qwen-image21');
    p.context.window.confirm = () => { confirmations++; return false; };
    p.context.downloadSupportedModel('qwen-image21', provider);
    assert.equal(confirmations, 1); assert.equal(p.requests.length, 0);
    p.context.window.confirm = () => { confirmations++; return true; };
    p.context.downloadSupportedModel('qwen-image21', provider);
    assert.equal(confirmations, 2); assert.equal(p.requests.length, 1);
  });
}

test('pending submissions and active queues prevent duplicates across rerenders and tabs', () => {
  const p = page(); p.choose('laya', 'laya'); p.context.downloadSupportedModel('laya', 'huggingface');
  p.context.switchDlSource('hf'); p.context.switchDlSource('catalog');
  p.context.downloadSupportedModel('laya', 'huggingface');
  assert.equal(p.requests.length, 1);
  p.queue([{ provider: 'huggingface', repo_id: 'aac6fef/laya-multilingual-mlx', status: 'queued', queue_position: 1 }]);
  p.timers.find(timer => timer.ms === 5000).fn();
  p.context.downloadSupportedModel('laya', 'huggingface');
  assert.equal(p.requests.length, 1);
  p.render();
  const button = p.element('catalog-model-list').buttons.find(b => b.dataset.entryId === 'laya');
  assert.equal(button.disabled, true); assert.match(button.textContent, /排队/);
  p.queue([{ provider: 'huggingface', repo_id: 'aac6fef/laya-multilingual-mlx', status: 'failed' }]);
  p.context.refreshSupportedModelActions(); assert.equal(button.disabled, false);
});

test('installed models can check updates and incomplete snapshots can be repaired per provider', () => {
  const p = page(); const repo = 'mlx-community/Qwen3.5-2B-4bit';
  p.context.window.__LOCAL_MODELS__ = { row: { repo_id: repo, source: 'hf', readiness: { status: 'ready' } } };
  assert.match(p.context.catalogDownloadState('huggingface', repo).label, /检查更新/);
  assert.equal(p.context.catalogDownloadState('huggingface', repo).disabled, false);
  assert.equal(p.context.catalogDownloadState('huggingface', repo).label, '已下载 · 检查更新');
  p.choose('qwen35-2b', 'qwen35-2b-4bit');
  p.context.downloadSupportedModel('qwen35-2b-4bit', 'huggingface');
  assert.equal(p.requests[0][0], '/admin/api/models/update/check');
  assert.equal(p.context.catalogDownloadState('huggingface', repo).disabled, true);
  const result = { success: true, provider: 'huggingface', repo_id: repo,
    local_commit_sha: 'a'.repeat(40), remote_commit_sha: 'b'.repeat(40), update_available: true };
  p.context.onCatalogUpdateResult('check', JSON.stringify(result));
  assert.equal(p.requests.length, 1); // Checking does not start a download.
  assert.equal(p.context.catalogDownloadState('huggingface', repo).label, '下载更新');
  p.context.downloadSupportedModel('qwen35-2b-4bit', 'huggingface');
  assert.equal(p.requests[1][0], '/admin/api/models/update/download');
  assert.equal(p.requests[1][1].commit_sha, result.remote_commit_sha);
  // A no-update response keeps checking available without queueing a download.
  vm.runInContext("catalogPendingDownloads.clear();", p.context);
  p.context.onCatalogUpdateResult('check', JSON.stringify({ ...result, update_available: false }));
  assert.equal(p.toasts.at(-1)[0], '已是最新版本');
  assert.equal(p.context.catalogDownloadState('modelscope', repo).label, '下载');
  p.context.window.__LOCAL_MODELS__.row.readiness.status = 'incomplete';
  assert.match(p.context.catalogDownloadState('huggingface', repo).label, /修复/);
});

test('memory hints distinguish unknown and excessive weight budgets', () => {
  const p = page(); const entry = catalog.entries.find(e => e.id === 'qwen38-27b-8bit');
  assert.equal(p.context.catalogMemoryHint(entry).key, 'catalog_memory_large');
  p.context.window.__IRONMLX_CATALOG_MEMORY_HINTS__.availableBytes = 64 * 1024 ** 3;
  assert.equal(p.context.catalogMemoryHint(entry).key, 'catalog_memory_budget');
  p.context.window.__IRONMLX_CATALOG_MEMORY_HINTS__ = null;
  assert.equal(p.context.catalogMemoryHint(entry).key, 'catalog_memory_unknown');
});

for (const lang of ['en', 'zh-Hans', 'zh-Hant', 'ja', 'ko']) {
  test(`catalogue and its dynamic labels are translated in ${lang}`, () => {
    const p = page(lang); p.render();
    assert.ok(p.element('catalog-summary').textContent.includes(catalog.updatedAt));
    assert.doesNotMatch(p.element('catalog-model-list').innerHTML, /catalog_[a-z_]+/);
    assert.ok(p.element('catalog-model-list').buttons.every(button => button.textContent));
    if (lang !== 'en') assert.doesNotMatch(p.element('catalog-model-list').innerHTML, /Runtime verified|Approx\. download/);
  });
}

test('missing catalogue gives an actionable fallback and escapes catalogue content', () => {
  const p = page(); p.context.window.__IRONMLX_SUPPORTED_MODELS__ = null; p.render();
  assert.match(p.element('catalog-model-list').innerHTML, /HuggingFace.*ModelScope/);
  p.context.window.__IRONMLX_SUPPORTED_MODELS__ = structuredClone(catalog);
  p.context.window.__IRONMLX_SUPPORTED_MODELS__.entries[0].name = '<img src=x onerror=alert(1)>';
  p.render(); assert.doesNotMatch(p.element('catalog-model-list').innerHTML, /<img/);
  assert.match(p.element('catalog-model-list').innerHTML, /&lt;img/);
});

test('EmbeddingGemma 2 groups BF16 and affine4 under text, image and audio embedding capabilities', () => {
  const entries = catalog.entries.filter(entry => entry.modelId === 'embeddinggemma2');
  assert.equal(entries.length, 2);
  assert.deepEqual(entries.map(entry => entry.variantLabel), ['BF16', 'Affine · 4 bit']);
  entries.forEach(entry => {
    assert.equal(entry.category, 'embedding');
    assert.deepEqual(entry.capabilities, ['text', 'vision', 'audio_embedding', 'embedding']);
    assert.equal(entry.msRepo, null);
  });
  const p = page(); p.render(); p.choose('embeddinggemma2', 'embeddinggemma2-4bit');
  const row = p.element('catalog-model-list').innerHTML;
  assert.match(row, /\/v1\/embeddings/);
  assert.match(row, /embeddinggemma-2-4bit/);
});

for (const lang of ['en', 'zh-Hans', 'zh-Hant', 'ja', 'ko']) {
  test(`embedding catalogue labels describe vectors rather than generation in ${lang}`, () => {
    const p = page(lang);
    vm.runInContext("catalogSelectedType = 'embedding'", p.context);
    p.render();
    const row = p.element('catalog-model-list').innerHTML;
    for (const key of ['catalog_text_embedding', 'catalog_image_embedding', 'catalog_audio_embedding']) {
      assert.ok(row.includes(p.context.t(key)), `${lang} ${key}`);
    }
    assert.ok(!row.includes(p.context.t('catalog_text')));
  });
}
