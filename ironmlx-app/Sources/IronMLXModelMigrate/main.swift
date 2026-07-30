import Foundation
import IronMLXAppCore

@main
struct IronMLXModelMigrate {
    static func main() async {
        do {
            let root = try rootURL(arguments: CommandLine.arguments)
            let service = ModelCacheMigrationService(rootURL: root)
            let results = try await service.migrateAll { message in
                FileHandle.standardError.write(Data((message + "\n").utf8))
            }
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
            FileHandle.standardOutput.write(try encoder.encode(results))
            FileHandle.standardOutput.write(Data("\n".utf8))
        } catch {
            FileHandle.standardError.write(Data(("migration failed: \(error.localizedDescription)\n").utf8))
            Foundation.exit(EXIT_FAILURE)
        }
    }

    private static func rootURL(arguments: [String]) throws -> URL {
        guard let index = arguments.firstIndex(of: "--root") else {
            return FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent(".ironmlx", isDirectory: true)
        }
        let valueIndex = arguments.index(after: index)
        guard valueIndex < arguments.endIndex else {
            throw MigrationCLIError.missingRoot
        }
        return URL(
            fileURLWithPath: NSString(string: arguments[valueIndex]).expandingTildeInPath,
            isDirectory: true
        )
    }
}

private enum MigrationCLIError: LocalizedError {
    case missingRoot

    var errorDescription: String? {
        "--root requires a directory path."
    }
}
