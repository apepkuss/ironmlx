import Foundation
import Testing

@testable import IronMLXAppCore

@Suite("Diagnostic export coordinator", .serialized)
@MainActor
struct DiagnosticExportCoordinatorTests {
    @Test func savePanelCancellationIsReportedWithoutCreatingFiles() async throws {
        let root = try temporaryDirectory()
        let coordinator = DiagnosticExportCoordinator(
            window: nil,
            service: service(root: root),
            destinationChooser: { nil }
        )

        let result = await withCheckedContinuation { continuation in
            coordinator.export(config: AppConfig(), backendRunning: false) {
                continuation.resume(returning: $0)
            }
        }

        #expect(result.status == .cancelled)
        #expect(try FileManager.default.contentsOfDirectory(atPath: root.path).allSatisfy {
            !$0.hasSuffix(".zip") && !$0.hasSuffix(".tmp")
        })
    }

    @Test func duplicateTriggerIsRejectedAndCancellationFinishesActiveExport() async throws {
        let root = try temporaryDirectory()
        let coordinator = DiagnosticExportCoordinator(
            window: nil,
            service: service(root: root),
            destinationChooser: {
                try? await Task.sleep(for: .seconds(10))
                return nil
            }
        )
        var statuses: [DiagnosticExportStatus] = []
        coordinator.export(config: AppConfig(), backendRunning: false) { statuses.append($0.status) }
        coordinator.export(config: AppConfig(), backendRunning: false) { statuses.append($0.status) }

        #expect(statuses == [.busy])
        coordinator.cancel()
        #expect(statuses == [.busy, .cancelled])
        #expect(!coordinator.isExporting)
    }

    @Test func zipCreationFailureIsStableAndDoesNotOpenSavePanel() async throws {
        let root = try temporaryDirectory()
        var chooserCalled = false
        let coordinator = DiagnosticExportCoordinator(
            window: nil,
            service: service(root: root, archiveMaker: { _, _ in
                throw DiagnosticBundleError.archiveUnavailable
            }),
            destinationChooser: {
                chooserCalled = true
                return root.appendingPathComponent("should-not-exist.zip")
            }
        )

        let result = await withCheckedContinuation { continuation in
            coordinator.export(config: AppConfig(), backendRunning: false) {
                continuation.resume(returning: $0)
            }
        }

        #expect(result.status == .failed)
        #expect(result.errorCode == "diagnostic_export_failed")
        #expect(!chooserCalled)
        #expect(!FileManager.default.fileExists(atPath: root.appendingPathComponent("should-not-exist.zip").path))
    }

    private func service(
        root: URL,
        archiveMaker: @escaping DiagnosticBundleService.ArchiveMaker = { _, _ in Data("test-zip".utf8) }
    ) -> DiagnosticBundleService {
        DiagnosticBundleService(
            scanner: LocalModelScanner(rootURL: root),
            versionService: ModelVersionManagementService(rootURL: root),
            incidentStore: BackendIncidentStore(url: root.appendingPathComponent("incidents.json")),
            logStore: IronMLXLogStore(rootURL: root.appendingPathComponent("logs")),
            archiveMaker: archiveMaker
        )
    }
}
