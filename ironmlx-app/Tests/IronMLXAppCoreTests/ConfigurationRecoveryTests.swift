import AppKit
import Foundation
import Testing

@testable import IronMLXAppCore

@Test(arguments: ["en", "zh-Hans", "zh-Hant", "ja", "ko"])
func configurationRecoveryTextLocalizesEverySupportedLanguage(language: String) {
    let text = ConfigurationRecoveryText(language: language)

    #expect(!text.recoveryTitle.isEmpty)
    #expect(!text.unsupportedTitle.isEmpty)
    #expect(!text.restoreLKG.isEmpty)
    #expect(!text.reset.isEmpty)
    #expect(!text.showFiles.isEmpty)
    #expect(!text.cancel.isEmpty)
    #expect(!text.quit.isEmpty)
    #expect(!text.displayName(for: .appConfig).isEmpty)
    #expect(!text.displayName(for: .modelParameters).isEmpty)
    #expect(text.migrationReason(from: 0, to: 1).contains("0"))
    #expect(text.versionReason(found: 2, supported: 1).contains("2"))
}

@Test func configurationIssueMapsPreciseStableDashboardCodes() {
    let root = URL(fileURLWithPath: "/tmp/config.json")
    let lkg = URL(fileURLWithPath: "/tmp/config.lkg.json")
    let base = { (reason: ConfigurationRecoveryIssue.Reason) in
        ConfigurationRecoveryIssue(
            kind: .appConfig,
            sourceURL: root,
            preservedURL: nil,
            lkgURL: lkg,
            lkgErrorDescription: "missing",
            reason: reason,
            errorDescription: "error",
            preservationErrorDescription: nil
        )
    }

    #expect(base(.corruption).dashboardErrorCode == "configuration_recovery_required")
    #expect(
        base(.migrationFailed(from: 0, to: 1)).dashboardErrorCode
            == "configuration_migration_failed"
    )
    #expect(
        base(.unsupportedVersion(found: 2, supported: 1)).dashboardErrorCode
            == "configuration_version_unsupported"
    )
}

@Test @MainActor
func configurationRecoveryManagerRestoresBothStoresFromValidatedLKG() throws {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("ironmlx-recovery-manager-lkg-\(UUID().uuidString)", isDirectory: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let configURL = root.appendingPathComponent("app_config.json")
    let parametersURL = root.appendingPathComponent("model_params.json")
    let configStore = AppConfigStore(url: configURL)
    let parameterStore = ModelParameterStore(url: parametersURL)
    let expectedConfig = AppConfig(language: "ja")
    let expectedParameters = ModelParameters(modelID: "mlx-community/Restore-4bit", maxTokens: "4096")
    #expect(configStore.save(expectedConfig))
    try parameterStore.save(expectedParameters)
    try Data("broken-config".utf8).write(to: configURL)
    try Data("broken-parameters".utf8).write(to: parametersURL)
    let manager = ConfigurationRecoveryManager(
        appConfigStore: configStore,
        modelParameterStore: parameterStore
    )
    manager.inspect()
    #expect(manager.issues.allSatisfy { $0.hasValidLKG })

    try manager.restoreAffectedConfigurationsFromLKG()

    #expect(!manager.hasIssues)
    #expect(configStore.load() == expectedConfig)
    #expect(try parameterStore.loadAll()[expectedParameters.modelID] == expectedParameters)
}

@Test @MainActor
func successfulAutomaticMigrationDoesNotCreateRecoveryIssue() throws {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("ironmlx-successful-config-migration-\(UUID().uuidString)", isDirectory: true)
    defer { try? FileManager.default.removeItem(at: root) }
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    let configURL = root.appendingPathComponent("app_config.json")
    let parametersURL = root.appendingPathComponent("model_params.json")
    try JSONEncoder().encode(AppConfig(language: "ko")).write(to: configURL)
    let parameters = ModelParameters(modelID: "model", maxTokens: "1024")
    try JSONEncoder().encode(["model": parameters]).write(to: parametersURL)
    let manager = ConfigurationRecoveryManager(
        appConfigStore: AppConfigStore(url: configURL),
        modelParameterStore: ModelParameterStore(url: parametersURL)
    )

    manager.inspect()

    #expect(!manager.hasIssues)
}

@Test @MainActor
func recoveryMessageDistinguishesMigrationUnsupportedVersionAndInvalidLKG() throws {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("ironmlx-recovery-message-\(UUID().uuidString)", isDirectory: true)
    let manager = ConfigurationRecoveryManager(
        appConfigStore: AppConfigStore(url: root.appendingPathComponent("app_config.json")),
        modelParameterStore: ModelParameterStore(url: root.appendingPathComponent("model_params.json"))
    )
    let issues = [
        ConfigurationRecoveryIssue(
            kind: .appConfig,
            sourceURL: root.appendingPathComponent("app_config.json"),
            preservedURL: root.appendingPathComponent("pre-v0.json"),
            lkgURL: root.appendingPathComponent("app.lkg.json"),
            lkgErrorDescription: "invalid LKG",
            reason: .migrationFailed(from: 0, to: 1),
            errorDescription: "migration failed",
            preservationErrorDescription: nil
        ),
        ConfigurationRecoveryIssue(
            kind: .modelParameters,
            sourceURL: root.appendingPathComponent("model_params.json"),
            preservedURL: nil,
            lkgURL: root.appendingPathComponent("models.lkg.json"),
            lkgErrorDescription: nil,
            reason: .unsupportedVersion(found: 2, supported: 1),
            errorDescription: "future version",
            preservationErrorDescription: nil
        ),
    ]

    let message = manager.recoveryMessage(
        for: issues,
        text: ConfigurationRecoveryText(language: "en")
    )

    #expect(message.contains("migration from v0 to v1 failed"))
    #expect(message.contains("schema v2"))
    #expect(message.contains("invalid LKG"))
}
