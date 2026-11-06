import Foundation
import Testing
import WebKit

@testable import IronMLXAppCore

@Test @MainActor func dashboardKeepsModelBusyStateAcrossModelListRefreshes() async throws {
    let url = try #require(IronMLXAppResourceResolver.url(forResource: "dashboard2", withExtension: "html"))
    let web = WKWebView()
    web.loadHTMLString(try String(contentsOf: url, encoding: .utf8), baseURL: url.deletingLastPathComponent())
    var ready = false
    for _ in 0..<100 {
        if (try? await web.evaluateJavaScript("typeof onLocalModelsScanned === 'function'")) as? Bool == true {
            ready = true
            break
        }
        try await Task.sleep(for: .milliseconds(50))
    }
    #expect(ready)
    let result = try #require(try await web.evaluateJavaScript("""
        (() => {
          currentLang = 'en';
          const row = loaded => JSON.stringify([{
            id: 'org/flash', type: 'llm', size_mb: 10, loaded,
            readiness: {status: 'ready'}, integrity: {state: 'verified'},
            dflash2: {status: 'available', candidates: [{id: 'draft'}]}
          }]);
          const snapshot = () => {
            const cell = document.querySelector('#tab-models-manage tbody .model-status-cell');
            const action = document.querySelector('#tab-models-manage tbody .action-btn, #tab-models-manage tbody select');
            return cell.textContent.trim() + '|' + action.textContent.trim() + '|' + action.disabled;
          };
          onLocalModelsScanned(row(false));
          const control = document.querySelector('#tab-models-manage tbody select');
          loadModel('org/flash', control, false, '', true, 'draft');
          onLocalModelsScanned(row(false));
          const loadingBeforeRestart = snapshot();
          onLocalModelsScanned(row(true));
          const loadingDuringRestart = snapshot();
          const otherModelCallback = (onModelLoaded('{"success":true}', 'org/other'), modelOperation('org/flash'));
          onModelLoaded('{"success":true}', 'org/flash');
          onLocalModelsScanned(row(true));
          const loaded = snapshot();
          unloadModel('org/flash', document.querySelector('#tab-models-manage tbody .action-btn'));
          onLocalModelsScanned(row(true));
          const unloading = snapshot();
          onModelUnloaded('{"success":true}', 'org/flash');
          onLocalModelsScanned(row(false));
          const unloaded = snapshot();
          loadModel('org/flash', document.querySelector('#tab-models-manage tbody select'), false);
          onModelLoaded('{"success":false,"error":"failed"}', 'org/flash');
          onLocalModelsScanned(row(false));
          return {
            loadingBeforeRestart, loadingDuringRestart, otherModelCallback,
            loaded, unloading, unloaded, afterFailure: snapshot()
          };
        })()
        """) as? [String: String])
    #expect(result["loadingBeforeRestart"] == "Loading...|Loading...|true")
    #expect(result["loadingDuringRestart"] == "Loading...|Loading...|true")
    #expect(result["otherModelCallback"] == "loading")
    #expect(result["loaded"]?.hasSuffix("|Unload|false") == true)
    #expect(result["unloading"] == "Unloading...|Unloading...|true")
    #expect(result["unloaded"]?.hasSuffix("|false") == true)
    #expect(result["unloaded"]?.hasPrefix("Not loaded|") == true)
    #expect(result["afterFailure"]?.hasPrefix("Not loaded|") == true)
    #expect(result["afterFailure"]?.hasSuffix("|false") == true)
}
