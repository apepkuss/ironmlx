import AppKit
import Foundation
import Testing
import WebKit

@testable import IronMLXAppCore

@MainActor
@Test func dashboardDownloadOnlyFlowUsesExplicitChoiceAndKeepsModelUnavailable() async throws {
    let configuration = WKWebViewConfiguration()
    configuration.websiteDataStore = .nonPersistent()
    configuration.userContentController.addUserScript(WKUserScript(
        source: try DashboardWindowController.bootstrapScript(config: AppConfig(), route: .status),
        injectionTime: .atDocumentStart, forMainFrameOnly: true
    ))
    let view = WKWebView(frame: NSRect(x: 0, y: 0, width: 1200, height: 800), configuration: configuration)
    let html = URL(fileURLWithPath: "Sources/IronMLXAppCore/Resources/dashboard2.html").standardizedFileURL
    view.loadFileURL(html, allowingReadAccessTo: html.deletingLastPathComponent())
    let deadline = Date().addingTimeInterval(15)
    while Date() < deadline {
        if (try? await view.evaluateJavaScript("typeof settingsBaseline !== 'undefined' && settingsBaseline !== null")) as? Bool == true { break }
        try await Task.sleep(for: .milliseconds(100))
    }
    let result = try await view.evaluateJavaScript("""
    (() => {
      const check = (value, message) => { if (!value) throw new Error(message); };
      const requests = [];
      apiPost = (path, body) => requests.push({path, body});
      setLanguage('zh-Hans'); navigateTo('models'); switchToTab('models-download');
      const task = {provider:'huggingface', repo_id:'mlx-community/embeddinggemma-2-bf16',
        status:'rejected', error_code:'unsupported_model_metadata', can_continue_download:true,
        error:'Error: unsupported model_type: embedding_gemma2 (expected llama)'};
      renderDownloadQueue({tasks:[task]});
      const buttons = [...document.querySelectorAll('.download-task-action')];
      check(buttons.map(button => button.textContent).join('|') === '查看支持模型|仍然下载|删除任务', 'action order');
      check(document.querySelector('.download-task').textContent.includes('当前版本无法加载运行'), 'file-only explanation');
      buttons[1].click();
      check(requests.length === 1 && requests[0].path.endsWith('/resume') && requests[0].body.download_only === true, 'explicit download-only request');
      buttons[1].click(); check(requests.length === 1, 'duplicate clicks are ignored');
      renderDownloadQueue({tasks:[{...task, status:'completed', download_only:true}]});
      check(document.querySelector('.download-task-status').textContent === '已下载 · 当前不支持运行', 'completion status');
      const model = {id:task.repo_id, repo_id:task.repo_id, source:'hf', type:'embedding', size_mb:1,
        readiness:{status:'unsupported', reason_code:'download_only_model'},
        integrity:{state:'verified'}};
      onLocalModelsScanned(JSON.stringify([model]));
      switchToTab('models-manage');
      const row = document.querySelector('#tab-models-manage tbody tr');
      check(row.querySelector('.action-load').disabled, 'loading stays disabled after integrity verification');
      check(row.querySelector('input[name="default-model"]').disabled, 'default-model selection stays disabled');
      check(row.querySelector('.model-status-text').textContent === '已下载 · 当前不支持运行', 'local model status');
      row.querySelector('.model-more').click();
      const files = [...document.querySelectorAll('.model-more-menu button')].find(button => button.textContent === '查看模型文件');
      check(!!files, 'file access remains available');
      files.click(); check(requests.at(-1).path === '/admin/api/models/reveal', 'reveal request');
      return true;
    })()
    """)
    #expect(result as? Bool == true)
}
