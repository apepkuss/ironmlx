const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const html = fs.readFileSync(path.join(__dirname,
  '../../ironmlx-app/Sources/IronMLXAppCore/Resources/dashboard2.html'), 'utf8');
function slice(start, end) {
  return html.slice(html.indexOf(start), html.indexOf(end, html.indexOf(start)));
}

function page(lang = 'zh-Hans') {
  const elements = new Map(), toasts = [], navigation = [], requests = [];
  function element(id) {
    if (!elements.has(id)) {
      const el = { style: {}, textContent: '', value: '', buttons: [], focus() {},
        querySelectorAll() { return this.buttons; } };
      let content = '';
      Object.defineProperty(el, 'innerHTML', {
        get: () => content,
        set(value) {
          content = value;
          el.buttons = [...value.matchAll(/<button[^>]+data-download-action="([^"]+)"[^>]*>/g)]
            .map(match => ({ dataset: { downloadAction: match[1],
              provider: match[0].match(/data-provider="([^"]+)"/)[1],
              repoId: match[0].match(/data-repo-id="([^"]+)"/)[1] },
              addEventListener(_, callback) { this.click = callback; } }));
        }
      });
      elements.set(id, el);
    }
    return elements.get(id);
  }
  const context = { currentLang: lang, console: { error() {} },
    document: { getElementById: element }, window: { __MODEL_INTEGRITY_STATUS__: {} },
    apiPost: (...args) => requests.push(args),
    escapeAttr: value => String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;')
      .replace(/>/g, '&gt;').replace(/"/g, '&quot;'),
    formatVersionBytes: value => String(value),
    renderDownloadRecovery() {}, refreshSearchResultDownloadStates() {},
    refreshSupportedModelActions() {}, refreshDownloadQueue() {}, syncModelList() {},
    showToast: (...args) => toasts.push(args),
    switchDlSource: source => navigation.push(source),
  };
  vm.createContext(context);
  vm.runInContext(slice('  const I18N =', '  const LANG_MAP =')
    + slice('  function t(key, fallback)', '  function setLanguage(')
    + slice('  function localizeErrorResult(', '  // Called from Rust after unload completes')
    + slice('  const ACTIVE_DOWNLOAD_PHASES', '  function renderDownloadRecovery(')
    + slice('  function onDownloadComplete(', '  // ── HuggingFace Search')
    + slice('  function performDownloadTaskAction(', '  async function deleteDownloadTask(')
    + slice('  function onDownloadTaskActionResult(', '  let downloadCleanupPending')
    + slice('  function modelReadiness(', '  function updateModelIntegrityRow(')
    + slice('  function renderModelLoadActions(', '  // Called from Rust with local model scan results'), context);
  return { context, element, toasts, navigation, requests,
    render(tasks) { context.renderDownloadQueue({ tasks }); return element('download-task-list').innerHTML; } };
}

const rejection = {
  provider: 'huggingface', repo_id: 'mlx-community/embeddinggemma-2-bf16', status: 'rejected',
  error_code: 'unsupported_model_metadata', can_continue_download: true,
  error: "Error: unsupported model_type: embedding_gemma2 (expected 'qwen3_5' or 'gemma4')",
};

test('persisted architecture rejection explains the reason and offers a working catalogue action', () => {
  const p = page();
  const output = p.render([JSON.parse(JSON.stringify(rejection))]);
  assert.match(output, /模型不兼容/);
  assert.match(output, /当前版本的 IronMLX 不支持 embedding_gemma2 架构/);
  assert.doesNotMatch(output, /data-download-action="resume"|expected|Error:/);
  assert.match(output, /data-download-action="delete"/);
  p.element('download-task-list').buttons.find(button => button.dataset.downloadAction === 'catalog').click();
  assert.deepEqual(p.navigation, ['catalog']);
  assert.match(p.render([rejection]), /embedding_gemma2 架构/);
});

for (const lang of ['en', 'zh-Hans', 'zh-Hant', 'ja', 'ko']) {
  test(`completion toast and persistent reason agree in ${lang}`, () => {
    const p = page(lang);
    const output = p.render([rejection]);
    p.context.onDownloadComplete(JSON.stringify({ success: false, code: rejection.error_code,
      error: rejection.error, repo_id: rejection.repo_id }));
    assert.equal(p.toasts[0][1], 'warn');
    assert.ok(output.includes(p.toasts[0][0]));
    assert.match(p.toasts[0][0], /embedding_gemma2/);
    assert.doesNotMatch(p.toasts[0][0], /expected|Error:|\{model_type\}/);
  });
}

test('other metadata failures and disk rejection keep retry and use distinct reasons', () => {
  const p = page();
  const metadata = p.render([{ ...rejection, can_continue_download: false, error: 'Unsupported quantization mode: future_mode' }]);
  assert.match(metadata, /量化格式或配置/);
  assert.match(metadata, /data-download-action="resume"/);
  assert.doesNotMatch(metadata, /data-download-action="catalog"|不支持 embedding_gemma2/);
  const disk = p.render([{ ...rejection, can_continue_download: false, error_code: 'insufficient_disk', error: 'Insufficient disk space' }]);
  assert.match(disk, /已拒绝/);
  assert.doesNotMatch(disk, /模型不兼容|data-download-action="catalog"/);
  assert.match(disk, /data-download-action="resume"/);
});

test('unsupported DFlash2 context layout offers explicit download-only continuation', () => {
  const p = page();
  const output = p.render([{ ...rejection, repo_id: 'incoai/Qwen3.6-35B-A3B-DFlash2',
    error: 'Error: unsupported DFlash2 configuration: target_layer_ids count 8 differs from draft layer count 6' }]);
  assert.match(output, /仍然下载/);
  assert.match(output, /当前版本无法加载运行/);
  assert.doesNotMatch(output, /data-download-action="resume"/);
  p.element('download-task-list').buttons.find(button => button.dataset.downloadAction === 'continue').click();
  assert.equal(p.requests[0][1].download_only, true);
});

test('active downloads hide stale failure details and untrusted errors cannot become markup', () => {
  const p = page();
  const active = p.render([{ ...rejection, status: 'downloading' }]);
  assert.doesNotMatch(active, /download-task-error|模型不兼容|data-download-action="catalog"/);
  const unsafe = p.render([{ ...rejection, can_continue_download: false, repo_id: '<img src=x onerror=alert(1)>',
    error: 'unsupported model_type: <img src=x onerror=alert(1)>' }]);
  assert.doesNotMatch(unsafe, /<img|data-download-action="catalog"/);
  assert.match(unsafe, /&lt;img/);
});


test('download anyway sits between catalogue and delete, submits once and preserves the token on retry', () => {
  const p = page();
  p.render([rejection]);
  const buttons = p.element('download-task-list').buttons;
  assert.deepEqual(buttons.map(button => button.dataset.downloadAction), ['catalog', 'continue', 'delete']);
  buttons[1].click();
  buttons[1].click();
  assert.equal(p.requests.length, 1);
  assert.equal(p.requests[0][0], '/admin/api/models/download/resume');
  assert.equal(p.requests[0][1].download_only, true);
  p.context.onDownloadTaskActionResult('resume', JSON.stringify({ success: false,
    code: 'download_token_required', provider: rejection.provider, repo_id: rejection.repo_id }));
  p.element('search-hf-token').value = 'test-token';
  p.element('download-task-list').buttons.find(button => button.dataset.downloadAction === 'continue').click();
  assert.equal(p.requests.length, 2);
  assert.equal(p.requests[1][1].download_only, true);
  assert.equal(p.requests[1][1].token, 'test-token');
});

for (const lang of ['en', 'zh-Hans', 'zh-Hant', 'ja', 'ko']) {
  test(`download-only completion explains runtime support in ${lang} and disables loading`, () => {
    const p = page(lang);
    const output = p.render([{ ...rejection, status: 'completed', download_only: true }]);
    assert.doesNotMatch(output, /data-download-action="continue"/);
    p.context.onDownloadComplete(JSON.stringify({ success: true, download_only: true, repo_id: rejection.repo_id }));
    assert.equal(p.toasts[0][1], 'success');
    if (lang === 'zh-Hans') {
      assert.match(output, /已下载 · 当前不支持运行/);
      assert.match(p.toasts[0][0], /已下载.*不支持运行/);
    }
    const model = { id: rejection.repo_id, readiness: { status: 'unsupported', reason_code: 'download_only_model' } };
    assert.equal(p.context.isModelLoadable(model), false);
    assert.match(p.context.renderModelLoadActions(model, '', '', '', '', false), /disabled/);
    const status = p.context.modelStatusPresentation(model, false);
    assert.ok(output.includes(status.text));
  });
}

for (const lang of ['en', 'zh-Hans', 'zh-Hant', 'ja', 'ko']) {
  test(`download transport failures explain recovery in ${lang}`, () => {
    const p = page(lang);
    for (const code of ['download_tls_failed', 'download_network_timeout', 'download_network_failed']) {
      const output = p.render([{ ...rejection, status: 'interrupted', download_only: true,
        can_continue_download: false, error_code: code, error: 'tls handshake eof' }]);
      assert.doesNotMatch(output, /tls handshake eof|data-download-action="continue"/);
      assert.match(output, /data-download-action="resume"/);
      if (lang === 'zh-Hans') assert.match(output, /检查网络或代理设置后重试.*已下载的数据会保留/);
    }
    const legacy = p.render([{ ...rejection, status: 'interrupted', can_continue_download: false,
      error_code: 'download_failed', error: 'Error: hf-hub Range download failed\nCaused by: tls handshake eof' }]);
    if (lang === 'zh-Hans') assert.match(legacy, /安全连接/);
  });
}
