import Foundation
import Testing

@testable import IronMLXAppCore

@MainActor
@Test func dashboardBridgeRegistersModelDownloadHandlers() {
    #expect(DashboardBridge.handlerNames.contains("downloadModel"))
    #expect(DashboardBridge.handlerNames.contains("searchHF"))
}

@Test func huggingFaceSearchUsesMlxFilterAndSort() async throws {
    let client = FakeModelDownloadHTTPClient()
    client.dataResponses["https://huggingface.co/api/models?search=qwen&sort=downloads&direction=-1&limit=20&filter=mlx"] = Data("""
    [{"id":"mlx-community/Qwen3-0.6B-4bit","downloads":123,"likes":4,"pipeline_tag":"text-generation"}]
    """.utf8)
    let service = ModelDownloadService(rootURL: try temporaryDirectory(), httpClient: client)

    let results = try await service.searchHuggingFace(query: "qwen", sort: "downloads")

    #expect(results.map(\.id) == ["mlx-community/Qwen3-0.6B-4bit"])
    #expect(results.first?.downloads == 123)
    #expect(client.dataRequests == [
        "https://huggingface.co/api/models?search=qwen&sort=downloads&direction=-1&limit=20&filter=mlx",
    ])
}

@Test func huggingFaceDownloadWritesUsableSnapshotAndReportsProgress() async throws {
    let root = try temporaryDirectory()
    let client = FakeModelDownloadHTTPClient()
    client.dataResponses["https://huggingface.co/api/models/mlx-community/Tiny-4bit"] = Data("""
    {
      "siblings": [
        {"rfilename":"config.json"},
        {"rfilename":"tokenizer.json"},
        {"rfilename":"tokenizer_config.json"},
        {"rfilename":"model.safetensors"}
      ]
    }
    """.utf8)
    client.dataResponses["https://huggingface.co/mlx-community/Tiny-4bit/resolve/main/config.json"] = Data("{}".utf8)
    client.downloadResponses["https://huggingface.co/mlx-community/Tiny-4bit/resolve/main/config.json"] = Data("{}".utf8)
    client.downloadResponses["https://huggingface.co/mlx-community/Tiny-4bit/resolve/main/tokenizer.json"] = Data("{}".utf8)
    client.downloadResponses["https://huggingface.co/mlx-community/Tiny-4bit/resolve/main/tokenizer_config.json"] = Data("{}".utf8)
    client.downloadResponses["https://huggingface.co/mlx-community/Tiny-4bit/resolve/main/model.safetensors"] = Data("weights".utf8)
    let service = ModelDownloadService(rootURL: root, httpClient: client)
    let progress = ProgressRecorder()

    let result = await service.downloadHuggingFace(
        repoID: "mlx-community/Tiny-4bit",
        token: nil,
        progress: { await progress.record($0) }
    )

    #expect(result.success)
    #expect(LocalModelScanner(rootURL: root).scan().map(\.repoID) == ["mlx-community/Tiny-4bit"])
    #expect(await progress.percentages().last == 100)
}

@Test func huggingFaceDownloadIncludesOptiqMetadataAndSidecars() async throws {
    let root = try temporaryDirectory()
    let client = FakeModelDownloadHTTPClient()
    client.dataResponses["https://huggingface.co/api/models/mlx-community/Gemma-OptiQ"] = Data("""
    {
      "siblings": [
        {"rfilename":"config.json"},
        {"rfilename":"tokenizer.json"},
        {"rfilename":"chat_template.jinja"},
        {"rfilename":"optiq_metadata.json"},
        {"rfilename":"model.safetensors"},
        {"rfilename":"optiq/optiq_vision.safetensors"}
      ]
    }
    """.utf8)
    let config = Data("""
    {
      "quantization": {"group_size": 64, "bits": 4, "mode": "optiq"},
      "optiq_vision": {"sidecar": "optiq/optiq_vision.safetensors"}
    }
    """.utf8)
    let optiqMetadata = Data("""
    {
      "method": "optiq_mixed_precision",
      "per_layer": {
        "model.layers.0.self_attn.q_proj": {"group_size": 64, "bits": 4},
        "model.layers.0.self_attn.k_proj": {"group_size": 64, "bits": 8}
      }
    }
    """.utf8)
    client.dataResponses["https://huggingface.co/mlx-community/Gemma-OptiQ/resolve/main/config.json"] = config
    client.downloadResponses["https://huggingface.co/mlx-community/Gemma-OptiQ/resolve/main/config.json"] = config
    client.downloadResponses["https://huggingface.co/mlx-community/Gemma-OptiQ/resolve/main/tokenizer.json"] = Data("{}".utf8)
    client.downloadResponses["https://huggingface.co/mlx-community/Gemma-OptiQ/resolve/main/chat_template.jinja"] = Data("{{ messages }}".utf8)
    client.downloadResponses["https://huggingface.co/mlx-community/Gemma-OptiQ/resolve/main/optiq_metadata.json"] = optiqMetadata
    client.downloadResponses["https://huggingface.co/mlx-community/Gemma-OptiQ/resolve/main/model.safetensors"] = Data("weights".utf8)
    client.downloadResponses["https://huggingface.co/mlx-community/Gemma-OptiQ/resolve/main/optiq/optiq_vision.safetensors"] = Data("vision".utf8)
    let service = ModelDownloadService(rootURL: root, httpClient: client)

    let result = await service.downloadHuggingFace(repoID: "mlx-community/Gemma-OptiQ", token: nil)
    let model = try #require(LocalModelScanner(rootURL: root).scan().first)

    #expect(result.success)
    #expect(model.quantization?.kind == "optiq")
    #expect(model.quantization?.mixedBits == [4, 8])
    #expect(model.readiness?.status == "ready")
    #expect(client.downloadRequests.contains("https://huggingface.co/mlx-community/Gemma-OptiQ/resolve/main/optiq_metadata.json"))
    #expect(client.downloadRequests.contains("https://huggingface.co/mlx-community/Gemma-OptiQ/resolve/main/optiq/optiq_vision.safetensors"))
}

@Test func modelScopeDownloadUpdatesStatusAndWritesMainSnapshot() async throws {
    let root = try temporaryDirectory()
    let client = FakeModelDownloadHTTPClient()
    client.dataResponses["https://modelscope.cn/api/v1/models/mlx-community/Tiny-4bit/repo/files"] = Data("""
    {"Success":true,"Data":{"Files":[{"Name":"config.json"},{"Name":"tokenizer.json"},{"Name":"model.safetensors"}]}}
    """.utf8)
    client.downloadResponses["https://modelscope.cn/api/v1/models/mlx-community/Tiny-4bit/repo?Revision=master&FilePath=config.json"] = Data("{}".utf8)
    client.downloadResponses["https://modelscope.cn/api/v1/models/mlx-community/Tiny-4bit/repo?Revision=master&FilePath=tokenizer.json"] = Data("{}".utf8)
    client.downloadResponses["https://modelscope.cn/api/v1/models/mlx-community/Tiny-4bit/repo?Revision=master&FilePath=model.safetensors"] = Data("weights".utf8)
    let service = ModelDownloadService(rootURL: root, httpClient: client)

    let start = await service.startModelScopeDownload(repoID: "mlx-community/Tiny-4bit")

    #expect(start.success)
    let completed = try await waitForCompletedDownload(in: service, repoID: "mlx-community/Tiny-4bit")
    #expect(completed.progressPct == 100)
    #expect(LocalModelScanner(rootURL: root).scan().map(\.source) == ["ms"])
}

private actor ProgressRecorder {
    private var values: [ModelDownloadProgress] = []

    func record(_ progress: ModelDownloadProgress) {
        values.append(progress)
    }

    func percentages() -> [Double] {
        values.map(\.percent)
    }
}

private final class FakeModelDownloadHTTPClient: ModelDownloadHTTPClient, @unchecked Sendable {
    var dataResponses: [String: Data] = [:]
    var downloadResponses: [String: Data] = [:]
    private(set) var dataRequests: [String] = []
    private(set) var downloadRequests: [String] = []
    private let lock = NSLock()

    func data(for request: URLRequest) async throws -> (Data, HTTPURLResponse) {
        let key = try requestKey(request)
        lock.withLock {
            dataRequests.append(key)
        }
        guard let data = lock.withLock({ dataResponses[key] }) else {
            throw URLError(.fileDoesNotExist)
        }
        return (data, response(for: request, statusCode: 200))
    }

    func download(for request: URLRequest) async throws -> (URL, HTTPURLResponse) {
        let key = try requestKey(request)
        lock.withLock {
            downloadRequests.append(key)
        }
        guard let data = lock.withLock({ downloadResponses[key] }) else {
            throw URLError(.fileDoesNotExist)
        }
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("ironmlx-download-\(UUID().uuidString)")
        try data.write(to: url)
        return (url, response(for: request, statusCode: 200))
    }

    private func requestKey(_ request: URLRequest) throws -> String {
        try #require(request.url).absoluteString
    }

    private func response(for request: URLRequest, statusCode: Int) -> HTTPURLResponse {
        HTTPURLResponse(url: request.url!, statusCode: statusCode, httpVersion: nil, headerFields: nil)!
    }
}

private func waitForCompletedDownload(
    in service: ModelDownloadService,
    repoID: String
) async throws -> ModelDownloadStatus {
    for _ in 0..<50 {
        if let status = await service.downloadStatuses().first(where: { $0.repoID == repoID }),
           status.status != "downloading" {
            return status
        }
        try await Task.sleep(nanoseconds: 20_000_000)
    }
    throw URLError(.timedOut)
}
