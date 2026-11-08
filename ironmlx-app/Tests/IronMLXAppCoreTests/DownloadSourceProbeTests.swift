import Foundation
import Testing

@testable import IronMLXAppCore

private struct FakeSourceProbe: ModelSourceProbing {
    var reachable: Set<DownloadSource>
    var delay: Duration = .zero

    func isReachable(_ source: DownloadSource, repoID _: String, revision _: String) async -> Bool {
        if delay > .zero { try? await Task.sleep(for: delay) }
        return reachable.contains(source)
    }
}

private final class ProbeHTTPClient: ModelDownloadHTTPClient, @unchecked Sendable {
    private let lock = NSLock()
    private var urls: [String] = []
    let succeed: Set<String>
    let hang: Bool

    init(succeed: Set<String> = [], hang: Bool = false) {
        self.succeed = succeed
        self.hang = hang
    }

    var requestedURLs: [String] { lock.withLock { urls } }

    func data(for request: URLRequest) async throws -> (Data, HTTPURLResponse) {
        let url = request.url!.absoluteString
        lock.withLock { urls.append(url) }
        if hang { try await Task.sleep(for: .seconds(30)) }
        guard succeed.contains(url) else { throw ModelDownloadHTTPError(statusCode: 404) }
        return (Data("{}".utf8), HTTPURLResponse(url: request.url!, statusCode: 200, httpVersion: nil, headerFields: nil)!)
    }

    func stream(
        for _: URLRequest,
        onResponse _: @escaping @Sendable (HTTPURLResponse) async throws -> Void,
        onData _: @escaping @Sendable (Data) async throws -> Void
    ) async throws {
        throw URLError(.unsupportedURL)
    }
}

@Suite("Download source selection")
struct DownloadSourceProbeTests {
    private let repo = "mlx-community/Qwen3.5-2B-4bit"
    private let revision = String(repeating: "a", count: 40)

    @Test func officialFirstWhenBothReachable() async {
        let sources = await chooseDownloadSource(
            probe: FakeSourceProbe(reachable: [.huggingFace, .hfMirror]), repoID: repo, revision: revision
        )
        #expect(sources == [.huggingFace, .hfMirror])
    }

    @Test func mirrorOnlyWhenOfficialUnreachable() async {
        let sources = await chooseDownloadSource(
            probe: FakeSourceProbe(reachable: [.hfMirror]), repoID: repo, revision: revision
        )
        #expect(sources == [.hfMirror])
    }

    @Test func emptyWhenNothingReachable() async {
        let sources = await chooseDownloadSource(
            probe: FakeSourceProbe(reachable: []), repoID: repo, revision: revision
        )
        #expect(sources.isEmpty)
    }

    @Test func probesRunConcurrently() async {
        let timeout = Duration.milliseconds(400)
        let clock = ContinuousClock()
        let start = clock.now
        _ = await chooseDownloadSource(
            probe: FakeSourceProbe(reachable: [], delay: timeout), repoID: repo, revision: revision
        )
        #expect(clock.now - start < timeout * 1.5)
    }

    @Test func probeRequestsPinnedRevisionMetadataOnEachHost() async {
        let official = "https://huggingface.co/api/models/\(repo)/revision/\(revision)"
        let client = ProbeHTTPClient(succeed: [official])
        let probe = DownloadSourceProbe(httpClient: client)
        #expect(await probe.isReachable(.huggingFace, repoID: repo, revision: revision))
        #expect(await probe.isReachable(.hfMirror, repoID: repo, revision: revision) == false)
        #expect(client.requestedURLs == [
            official,
            "https://hf-mirror.com/api/models/\(repo)/revision/\(revision)",
        ])
    }

    @Test func probeGivesUpAfterItsTimeout() async {
        let probe = DownloadSourceProbe(httpClient: ProbeHTTPClient(hang: true), timeout: .milliseconds(200))
        let clock = ContinuousClock()
        let start = clock.now
        #expect(await probe.isReachable(.huggingFace, repoID: repo, revision: revision) == false)
        #expect(clock.now - start < .seconds(2))
    }

    @Test func defaultTimeoutIsThreeSeconds() {
        #expect(DownloadSourceProbe.defaultTimeout == .seconds(3))
    }
}
