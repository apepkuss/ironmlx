import AppKit
import Foundation

struct RuntimeLogExportRequest: Decodable {
    let source: String
    let content: String

    var suggestedFilename: String {
        let timestamp = ISO8601DateFormatter().string(from: Date())
            .replacingOccurrences(of: ":", with: "-")
        return "\(source)-\(timestamp).log"
    }

    static func decode(_ json: String) throws -> Self {
        let request = try JSONDecoder().decode(Self.self, from: Data(json.utf8))
        guard ["ironmlx-app", "ironmlx-server"].contains(request.source) else {
            throw CocoaError(.fileWriteInvalidFileName)
        }
        return request
    }
}

@MainActor
final class RuntimeLogExporter {
    typealias DestinationChooser = (String) async -> URL?
    private let chooseDestination: DestinationChooser
    private var exporting = false

    init(window: NSWindow?) {
        chooseDestination = { [weak window] filename in
            let panel = NSSavePanel()
            panel.nameFieldStringValue = filename
            panel.canCreateDirectories = true
            return await withCheckedContinuation { continuation in
                let completion: (NSApplication.ModalResponse) -> Void = { response in
                    continuation.resume(returning: response == .OK ? panel.url : nil)
                }
                if let window {
                    panel.beginSheetModal(for: window, completionHandler: completion)
                } else {
                    panel.begin(completionHandler: completion)
                }
            }
        }
    }

    init(chooseDestination: @escaping DestinationChooser) {
        self.chooseDestination = chooseDestination
    }

    func export(_ request: RuntimeLogExportRequest) async -> String {
        guard !exporting else { return "busy" }
        exporting = true
        defer { exporting = false }
        guard let destination = await chooseDestination(request.suggestedFilename) else {
            return "cancelled"
        }
        do {
            try Data(request.content.utf8).write(to: destination, options: .atomic)
            return "exported"
        } catch {
            let error = error as NSError
            IronMLXAppLogger.error(
                "event=runtime_log_export_failed error_domain=\(error.domain) error_code=\(error.code)"
            )
            return "failed"
        }
    }
}
