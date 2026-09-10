import Foundation
import Testing

@testable import IronMLXAppCore

@Test func emptyModelSelectionsSaveAndReloadContextLimit() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let url = root.appendingPathComponent("model_params.json")
    let model = "mlx-community/Qwen3.5-4B-4bit"
    let payload = Data(
        """
        {"model_id":"\(model)","max_tokens":"4096","mtp_enabled":false,
         "mtp_model_id":"","dflash2_enabled":false,"dflash2_model_id":"  "}
        """.utf8)
    let parameters = try JSONDecoder().decode(ModelParameters.self, from: payload)
    let store = ModelParameterStore(url: url)
    try store.save(parameters)
    let reloaded = try #require(ModelParameterStore(url: url).loadAll()[model])
    #expect(reloaded.maxCacheCap == 4096)
    #expect(reloaded.mtpModelID == nil)
    #expect(reloaded.dflash2ModelID == nil)
    let object = try #require(
        JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
    let models = try #require(object["models"] as? [String: [String: Any]])
    #expect(models[model]?["mtp_model_id"] == nil)
    #expect(models[model]?["dflash2_model_id"] == nil)
    try store.replaceParameters(for: model, with: parameters)
    #expect(try ModelParameterStore(url: url).loadAll()[model]?.maxCacheCap == 4096)
}

@Test(arguments: [0, 1]) func emptyModelSelectionsLoadExistingConfigurations(version: Int) throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let url = root.appendingPathComponent("model_params.json")
    let models: [String: Any] = [
        "model": [
            "model_id": "model", "max_tokens": "4096", "mtp_model_id": " \t",
            "dflash2_model_id": "",
        ]
    ]
    let object: [String: Any] = version == 0 ? models : ["schema_version": 1, "models": models]
    let original = try JSONSerialization.data(withJSONObject: object)
    try original.write(to: url)
    let store = ModelParameterStore(url: url)
    let loaded = try #require(store.loadAll()["model"])
    #expect(loaded.maxCacheCap == 4096)
    #expect(loaded.mtpModelID == nil && loaded.dflash2ModelID == nil)
    #expect(store.recoveryIssue == nil)
    if version == 0 {
        let layout = ConfigurationFileLayout(activeURL: url)
        let files = try FileManager.default.contentsOfDirectory(
            at: layout.recoveryDirectoryURL, includingPropertiesForKeys: nil)
        let evidence = try #require(
            files.first { $0.lastPathComponent.contains("pre-migration-v0-") })
        #expect(try Data(contentsOf: evidence) == original)
        #expect(try Data(contentsOf: layout.lkgURL) == Data(contentsOf: url))
    }
}

@Test @MainActor func modelParameterSaveErrorsDistinguishValidationAndIO() throws {
    let invalid = DashboardBridge.modelParameterSaveErrorJSON(
        ConfigurationPersistenceError.invalidValue("max_tokens"))
    let parsed = try #require(
        JSONSerialization.jsonObject(with: Data(invalid.utf8)) as? [String: Any])
    #expect(parsed["code"] as? String == "model_parameters_invalid")
    #expect(parsed["field"] as? String == "max_tokens")
    let io = DashboardBridge.modelParameterSaveErrorJSON(
        NSError(domain: NSCocoaErrorDomain, code: NSFileWriteNoPermissionError))
    let ioParsed = try #require(JSONSerialization.jsonObject(with: Data(io.utf8)) as? [String: Any])
    #expect(ioParsed["code"] as? String == "settings_persist_failed")
}

@Test func invalidParametersLeaveSavedConfigurationIntact() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let url = root.appendingPathComponent("model_params.json")
    let store = ModelParameterStore(url: url)
    let original = ModelParameters(modelID: "model", maxTokens: "4096", mtpModelID: "org/draft")
    try store.save(original)
    let before = try Data(contentsOf: url)
    var invalid = original
    invalid.maxTokens = "0"
    invalid.dflash2ModelID = ""
    #expect(throws: ConfigurationPersistenceError.invalidValue("max_tokens")) {
        try store.save(invalid)
    }
    #expect(try Data(contentsOf: url) == before)
    #expect(try ModelParameterStore(url: url).loadAll()["model"] == original)
}
