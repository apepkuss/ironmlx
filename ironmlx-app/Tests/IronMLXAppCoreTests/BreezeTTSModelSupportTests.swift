import Foundation
import AppKit
import WebKit
import Testing
@testable import IronMLXAppCore

private func breezeConfig(quantized: Bool = false) -> [String: Any] {
    var config: [String: Any] = ["model_type": "breeze_tts", "audio_num_codebooks": 16, "max_position_embeddings": 2048,
        "backbone_config": ["model_type": "qwen3", "num_hidden_layers": 28, "hidden_size": 2048],
        "depth_decoder_config": ["num_hidden_layers": 12],
        "text_encoder_config": ["model_type": "t5gemma2_text", "num_hidden_layers": 26]]
    if quantized { config["quantization"] = ["mode": "mxfp8", "bits": 8, "group_size": 32] }
    return config
}

private struct BreezePreflight: ModelMetadataPreflighting {
    var accepts = true
    func validate(metadataDirectory _: URL) async throws -> ModelMetadataPreflightResult {
        guard accepts else { throw ModelMetadataPreflightError.rejected("unsupported Breeze profile") }
        return ModelMetadataPreflightResult(modelType: "breeze_tts", artifactRole: "tts", quantization: nil)
    }
}

@Test func breezeDownloadProfileIncludesIndependentCodecAndRejectsMissingCodec() throws {
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    try JSONSerialization.data(withJSONObject: breezeConfig(quantized: true)).write(to: root.appendingPathComponent("config.json"))
    let paths = BreezeTTSModelSupport.requiredMetadata + ["model.safetensors", BreezeTTSModelSupport.codecWeight]
    let files = paths.map { RemoteModelFile(path: $0, size: 12, sha256: nil, blobID: nil, etag: nil) }
    let profile = try #require(try TTSModelDownloadProfile.inspect(directory: root, files: files))
    #expect(profile.weightPaths == ["model.safetensors", "audio_tokenizer/model.safetensors"])
    #expect(profile.externalResources.isEmpty)
    #expect(profile.compatibility.modelType == "breeze_tts")
    #expect(profile.compatibility.quantization?.mode == "mxfp8")
    #expect(throws: ModelMetadataPreflightError.self) {
        try TTSModelDownloadProfile.inspect(directory: root, files: files.filter { $0.path != BreezeTTSModelSupport.codecWeight })
    }
    var badConfig = breezeConfig(quantized: true)
    badConfig["quantization"] = ["mode": "affine", "bits": 8, "group_size": 64]
    #expect(!BreezeTTSModelSupport.isSupported(badConfig))
}

@Test func breezeSnapshotPromotionRequiresNativePreflightAndKeepsVerifiedFiles() async throws {
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let repo = "mlx-community/Breeze-TTS-2-mlx"
    var files = Dictionary(uniqueKeysWithValues: BreezeTTSModelSupport.requiredMetadata.map { ($0, Data("{}".utf8)) })
    files["config.json"] = try JSONSerialization.data(withJSONObject: breezeConfig())
    files["model.safetensors"] = Data("weights".utf8)
    files[BreezeTTSModelSupport.codecWeight] = Data("codec".utf8)
    let snapshot = try writeVerifiedTestSnapshot(root: root, repoID: repo, files: files)
    let verifier = ModelSnapshotVerifier()
    var manifest = try verifier.loadManifest(at: snapshot)
    manifest.compatibility.modelType = "breeze_tts"
    manifest.compatibility.artifactRole = "download_only"
    manifest.compatibility.downloadOnly = true
    try ModelDownloadStore(rootURL: root).writeManifest(manifest, to: snapshot)
    await #expect(throws: ModelMetadataPreflightError.self) {
        try await BreezeTTSModelSupport.upgradeDownloadOnlySnapshot(at: snapshot.path, preflight: BreezePreflight(accepts: false))
    }
    #expect(try verifier.loadManifest(at: snapshot).compatibility.downloadOnly == true)
    let scanner = LocalModelScanner(rootURL: root)
    let before = try #require(scanner.model(for: repo))
    #expect(before.type == "tts")
    #expect(before.readiness?.isLoadable == true)
    try await BreezeTTSModelSupport.upgradeDownloadOnlySnapshot(at: snapshot.path, preflight: BreezePreflight())
    let updated = try verifier.verify(snapshot: snapshot)
    #expect(updated.files == manifest.files)
    #expect(updated.compatibility.artifactRole == "tts")
    #expect(updated.compatibility.downloadOnly == nil)
    let model = try #require(scanner.model(for: repo))
    #expect(model.ttsMetadata?.outputSampleRateHz == 24000)
    #expect(model.ttsMetadata?.supportedLanguages == ["zh", "en"])
    #expect(try scanner.audioResources(for: repo) == nil)
    let customExecution = BackendAudioExecutionSettings(firstAudioTimeoutMS: 180000, maxOutputFrames: 24000 * 600)
    #expect(try scanner.audioResources(for: repo, execution: customExecution)?.execution == customExecution)
    try FileManager.default.removeItem(at: snapshot.appendingPathComponent(BreezeTTSModelSupport.codecWeight))
    #expect(scanner.model(for: repo)?.readiness?.isLoadable == false)
}

@MainActor @Test func dashboardBreezeSynthesisFormExposesDesignCloneAndDirection() async throws {
    let url = try #require(IronMLXAppResourceResolver.url(forResource: "dashboard2", withExtension: "html"))
    let web = WKWebView(frame: NSRect(x: 0, y: 0, width: 1200, height: 1000))
    web.loadHTMLString(try String(contentsOf: url, encoding: .utf8), baseURL: url.deletingLastPathComponent())
    for _ in 0..<100 {
        if (try? await web.evaluateJavaScript("typeof openTTSDetails === 'function'")) as? Bool == true { break }
        try await Task.sleep(for: .milliseconds(50))
    }
    let result = try await web.evaluateJavaScript("""
        (() => {
          const model = {id:'mlx-community/Breeze-TTS-2-mlx-8bit', architecture:'breeze-tts', type:'tts',
            tts_metadata:{output_sample_rate_hz:24000,supported_languages:['zh','en']}};
          window.__LOCAL_MODELS__[model.id] = model;
          openTTSDetails(model);
          document.getElementById('breeze-text').value = '你好，世界。';
          document.getElementById('breeze-instructions').value = '温柔清晰的年轻女声';
          let request = null;
          apiPost = (path, body) => {request = {path,body};};
          generateBreezeSpeech();
          return {visible:!document.getElementById('breeze-synthesis').hidden,request,
            rate:document.getElementById('tts-output-sample-rate').value,
            transcript:!!document.getElementById('voice-ref-text'),
            modeHelp:document.querySelector('[data-i18n="breeze_synthesis_help"]').textContent};
        })()
        """) as? [String: Any]
    #expect(result?["visible"] as? Bool == true)
    #expect(result?["rate"] as? String == "24,000 Hz")
    #expect(result?["transcript"] as? Bool == true)
    let request = try #require(result?["request"] as? [String: Any])
    #expect(request["path"] as? String == "/admin/api/audio/synthesis")
    let body = try #require(request["body"] as? [String: Any])
    #expect(body["instructions"] as? String == "温柔清晰的年轻女声")
    #expect(body["voice"] == nil)
    #expect(body["response_format"] as? String == "wav")
    if let path = ProcessInfo.processInfo.environment["IRONMLX_BREEZE_UI_SNAPSHOT"] {
        let image = try await web.takeSnapshot(configuration: nil)
        let bitmap = try #require(image.tiffRepresentation.flatMap(NSBitmapImageRep.init(data:)))
        let png = try #require(bitmap.representation(using: .png, properties: [:]))
        try png.write(to: URL(fileURLWithPath: path))
    }
}
