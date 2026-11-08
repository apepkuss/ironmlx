import Foundation

/// Hosts the beginner download flow can fetch Hugging Face repositories from.
enum DownloadSource: String, Codable, Sendable {
    case huggingFace
    case hfMirror

    var endpoint: URL {
        switch self {
        case .huggingFace: HuggingFaceEndpointOverride.official
        case .hfMirror: URL(string: "https://hf-mirror.com")!
        }
    }

    var usesHfMirror: Bool { self == .hfMirror }
}

protocol ModelSourceProbing: Sendable {
    func isReachable(_ source: DownloadSource, repoID: String, revision: String) async -> Bool
}

/// Requests the pinned revision's metadata on a host; any failure or a reply
/// slower than `timeout` counts as unreachable.
struct DownloadSourceProbe: ModelSourceProbing {
    static let defaultTimeout: Duration = .seconds(3)

    private let httpClient: any ModelDownloadHTTPClient
    private let timeout: Duration

    init(
        httpClient: any ModelDownloadHTTPClient = URLSessionModelDownloadHTTPClient(),
        timeout: Duration = Self.defaultTimeout
    ) {
        self.httpClient = httpClient
        self.timeout = timeout
    }

    func isReachable(_ source: DownloadSource, repoID: String, revision: String) async -> Bool {
        let url = repoID.split(separator: "/").reduce(
            source.endpoint.appendingPathComponent("api").appendingPathComponent("models")
        ) { $0.appendingPathComponent(String($1)) }
            .appendingPathComponent("revision")
            .appendingPathComponent(revision)
        let request = URLRequest(
            url: url,
            timeoutInterval: Double(timeout.components.seconds) + Double(timeout.components.attoseconds) / 1e18
        )
        let client = httpClient
        let timeout = timeout
        return await withTaskGroup(of: Bool.self) { group in
            group.addTask {
                (try? await client.data(for: request)) != nil
            }
            group.addTask {
                try? await Task.sleep(for: timeout)
                return false
            }
            let first = await group.next() ?? false
            group.cancelAll()
            return first
        }
    }
}

/// Probes both hosts at once and returns the reachable ones, official first.
/// An empty result means neither host answered.
func chooseDownloadSource(
    probe: some ModelSourceProbing,
    repoID: String,
    revision: String
) async -> [DownloadSource] {
    async let official = probe.isReachable(.huggingFace, repoID: repoID, revision: revision)
    async let mirror = probe.isReachable(.hfMirror, repoID: repoID, revision: revision)
    let reachable = await (official, mirror)
    return [(DownloadSource.huggingFace, reachable.0), (.hfMirror, reachable.1)]
        .filter(\.1).map(\.0)
}
