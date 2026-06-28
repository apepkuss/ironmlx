import Foundation

public protocol ModelDownloadHTTPClient: Sendable {
    func data(for request: URLRequest) async throws -> (Data, HTTPURLResponse)
    func download(for request: URLRequest) async throws -> (URL, HTTPURLResponse)
}

public struct URLSessionModelDownloadHTTPClient: ModelDownloadHTTPClient {
    public init() {}

    public func data(for request: URLRequest) async throws -> (Data, HTTPURLResponse) {
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw URLError(.badServerResponse)
        }
        try validate(http)
        return (data, http)
    }

    public func download(for request: URLRequest) async throws -> (URL, HTTPURLResponse) {
        let (url, response) = try await URLSession.shared.download(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw URLError(.badServerResponse)
        }
        try validate(http)
        return (url, http)
    }

    private func validate(_ response: HTTPURLResponse) throws {
        guard (200..<300).contains(response.statusCode) else {
            throw ModelDownloadHTTPError(statusCode: response.statusCode)
        }
    }
}

public struct ModelDownloadHTTPError: LocalizedError, Sendable {
    public var statusCode: Int

    public var errorDescription: String? {
        "Download endpoint returned HTTP \(statusCode)."
    }
}

public struct HuggingFaceSearchResult: Codable, Equatable, Sendable {
    public var id: String
    public var modelId: String?
    public var downloads: Int?
    public var likes: Int?
    public var pipelineTag: String?

    enum CodingKeys: String, CodingKey {
        case id
        case modelId
        case downloads
        case likes
        case pipelineTag = "pipeline_tag"
    }
}

public struct ModelDownloadProgress: Equatable, Sendable {
    public var percent: Double
    public var filename: String
}

public struct ModelDownloadCompletion: Codable, Equatable, Sendable {
    public var success: Bool
    public var message: String?
    public var error: String?
    public var code: String?
    public var repoID: String?

    enum CodingKeys: String, CodingKey {
        case success
        case message
        case error
        case code
        case repoID = "repo_id"
    }
}

public struct ModelDownloadStartResponse: Codable, Equatable, Sendable {
    public var success: Bool
    public var status: String
    public var repoID: String
    public var error: String?
    public var code: String?

    enum CodingKeys: String, CodingKey {
        case success
        case status
        case repoID = "repo_id"
        case error
        case code
    }
}

public struct ModelDownloadStatus: Codable, Equatable, Sendable {
    public var repoID: String
    public var status: String
    public var progressPct: Double
    public var error: String?
    public var errorCode: String?

    enum CodingKeys: String, CodingKey {
        case repoID = "repo_id"
        case status
        case progressPct = "progress_pct"
        case error
        case errorCode = "error_code"
    }
}

public actor ModelDownloadService {
    private let rootURL: URL
    private let httpClient: any ModelDownloadHTTPClient
    private let huggingFaceEndpoint: URL
    private let modelScopeEndpoint: URL
    private var statuses: [String: ModelDownloadStatus] = [:]

    public init(
        rootURL: URL = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".ironmlx", isDirectory: true),
        httpClient: any ModelDownloadHTTPClient = URLSessionModelDownloadHTTPClient(),
        huggingFaceEndpoint: URL = URL(string: "https://huggingface.co")!,
        modelScopeEndpoint: URL = URL(string: "https://modelscope.cn/api/v1/models")!
    ) {
        self.rootURL = rootURL
        self.httpClient = httpClient
        self.huggingFaceEndpoint = huggingFaceEndpoint
        self.modelScopeEndpoint = modelScopeEndpoint
    }

    public func searchHuggingFace(query: String, sort: String) async throws -> [HuggingFaceSearchResult] {
        var components = URLComponents(
            url: huggingFaceEndpoint
                .appendingPathComponent("api")
                .appendingPathComponent("models"),
            resolvingAgainstBaseURL: false
        )
        components?.queryItems = [
            URLQueryItem(name: "search", value: query),
            URLQueryItem(name: "sort", value: sort),
            URLQueryItem(name: "direction", value: "-1"),
            URLQueryItem(name: "limit", value: "20"),
            URLQueryItem(name: "filter", value: "mlx"),
        ]
        guard let url = components?.url else {
            throw URLError(.badURL)
        }

        let (data, _) = try await httpClient.data(for: URLRequest(url: url))
        return try JSONDecoder().decode([HuggingFaceSearchResult].self, from: data)
    }

    public func downloadHuggingFace(
        repoID: String,
        token: String?,
        progress: @Sendable (ModelDownloadProgress) async -> Void = { _ in }
    ) async -> ModelDownloadCompletion {
        let repoID = repoID.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !repoID.isEmpty else {
            return failure(message: "missing repo_id", code: "repo_not_found", repoID: repoID)
        }
        if modelExistsLocally(repoID) {
            return failure(
                message: "Model \(repoID) already exists locally",
                code: nil,
                repoID: repoID
            )
        }

        do {
            try await runHuggingFaceDownload(repoID: repoID, token: token, progress: progress)
            return ModelDownloadCompletion(
                success: true,
                message: "Model \(repoID) downloaded successfully.",
                error: nil,
                code: nil,
                repoID: repoID
            )
        } catch let error as DownloadFailure {
            return failure(message: error.message, code: error.code, repoID: error.repoID)
        } catch {
            return failure(message: error.localizedDescription, code: "hf_download_failed", repoID: repoID)
        }
    }

    public func startModelScopeDownload(repoID: String) async -> ModelDownloadStartResponse {
        let repoID = repoID.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !repoID.isEmpty else {
            return ModelDownloadStartResponse(
                success: false,
                status: "error",
                repoID: repoID,
                error: "missing repo_id",
                code: "repo_not_found"
            )
        }
        if modelExistsLocally(repoID) {
            return ModelDownloadStartResponse(
                success: false,
                status: "error",
                repoID: repoID,
                error: "Model \(repoID) already exists locally",
                code: nil
            )
        }
        if statuses[repoID]?.status == "downloading" {
            return ModelDownloadStartResponse(
                success: false,
                status: "error",
                repoID: repoID,
                error: "\(repoID) is already downloading",
                code: nil
            )
        }

        statuses[repoID] = ModelDownloadStatus(
            repoID: repoID,
            status: "downloading",
            progressPct: 0,
            error: nil,
            errorCode: nil
        )

        Task {
            await self.runModelScopeDownload(repoID: repoID)
        }

        return ModelDownloadStartResponse(
            success: true,
            status: "accepted",
            repoID: repoID,
            error: nil,
            code: nil
        )
    }

    public func downloadStatuses() -> [ModelDownloadStatus] {
        statuses.values.sorted { $0.repoID.localizedStandardCompare($1.repoID) == .orderedAscending }
    }

    private func runHuggingFaceDownload(
        repoID: String,
        token: String?,
        progress: @Sendable (ModelDownloadProgress) async -> Void
    ) async throws {
        let info = try await fetchHuggingFaceModelInfo(repoID: repoID, token: token)
        let fileNames = info.siblings.compactMap(\.rfilename)
        guard fileNames.contains("config.json") else {
            throw DownloadFailure(repoID: repoID, code: "repo_missing_config", message: "Repository \"\(repoID)\" is missing config.json.")
        }
        guard fileNames.contains("tokenizer.json") else {
            throw DownloadFailure(repoID: repoID, code: "repo_missing_tokenizer", message: "Repository \"\(repoID)\" is missing tokenizer.json.")
        }

        let snapshot = snapshotURL(sourceDirectory: "models", repoID: repoID, snapshotID: "main")
        let optionalFiles = [
            "tokenizer_config.json",
            "generation_config.json",
            "special_tokens_map.json",
            "model.safetensors.index.json",
        ].filter { fileNames.contains($0) }
        var files = ["config.json", "tokenizer.json"] + optionalFiles

        if fileNames.contains("model.safetensors.index.json"),
           let indexData = try? await fetchHuggingFaceFileData(repoID: repoID, filename: "model.safetensors.index.json", token: token),
           let shards = safetensorsShards(from: indexData),
           !shards.isEmpty {
            let missingShards = shards.filter { !fileNames.contains($0) }
            if !missingShards.isEmpty {
                if fileNames.contains("model.safetensors") {
                    files.append("model.safetensors")
                } else {
                    throw DownloadFailure(
                        repoID: repoID,
                        code: "repo_broken_shards",
                        message: "Repository \"\(repoID)\" has a broken model.safetensors.index.json."
                    )
                }
            } else {
                files.append(contentsOf: shards)
            }
        } else if fileNames.contains("model.safetensors") {
            files.append("model.safetensors")
        } else {
            let safetensors = fileNames.filter { $0.hasSuffix(".safetensors") }.sorted()
            if safetensors.isEmpty {
                throw DownloadFailure(
                    repoID: repoID,
                    code: "missing_safetensors",
                    message: "Repository \"\(repoID)\" does not include .safetensors weights."
                )
            }
            files.append(contentsOf: safetensors)
        }

        files = uniqueValidRepositoryPaths(files)
        try FileManager.default.createDirectory(at: snapshot, withIntermediateDirectories: true)

        for (index, filename) in files.enumerated() {
            try await downloadHuggingFaceFile(repoID: repoID, filename: filename, token: token, to: snapshot)
            await progress(
                ModelDownloadProgress(
                    percent: Double(index + 1) / Double(files.count) * 100,
                    filename: filename
                )
            )
        }
    }

    private func fetchHuggingFaceModelInfo(repoID: String, token: String?) async throws -> HuggingFaceModelInfo {
        let url = huggingFaceURL(pathComponents: ["api", "models"] + repoPathComponents(repoID))
        var request = URLRequest(url: url)
        applyHuggingFaceToken(token, to: &request)
        do {
            let (data, _) = try await httpClient.data(for: request)
            return try JSONDecoder().decode(HuggingFaceModelInfo.self, from: data)
        } catch let error as ModelDownloadHTTPError where error.statusCode == 404 || error.statusCode == 401 || error.statusCode == 403 {
            throw DownloadFailure(repoID: repoID, code: "repo_not_found", message: "Repository \"\(repoID)\" does not exist, or is private/gated.")
        }
    }

    private func fetchHuggingFaceFileData(repoID: String, filename: String, token: String?) async throws -> Data {
        let url = huggingFaceURL(pathComponents: repoPathComponents(repoID) + ["resolve", "main"] + repositoryPathComponents(filename))
        var request = URLRequest(url: url)
        applyHuggingFaceToken(token, to: &request)
        let (data, _) = try await httpClient.data(for: request)
        return data
    }

    private func downloadHuggingFaceFile(repoID: String, filename: String, token: String?, to snapshot: URL) async throws {
        let url = huggingFaceURL(pathComponents: repoPathComponents(repoID) + ["resolve", "main"] + repositoryPathComponents(filename))
        var request = URLRequest(url: url)
        applyHuggingFaceToken(token, to: &request)
        try await downloadFile(request: request, filename: filename, to: snapshot)
    }

    private func runModelScopeDownload(repoID: String) async {
        do {
            try await downloadModelScope(repoID: repoID)
            statuses[repoID] = ModelDownloadStatus(
                repoID: repoID,
                status: "completed",
                progressPct: 100,
                error: nil,
                errorCode: nil
            )
        } catch let error as DownloadFailure {
            statuses[repoID] = ModelDownloadStatus(
                repoID: repoID,
                status: "failed",
                progressPct: statuses[repoID]?.progressPct ?? 0,
                error: error.message,
                errorCode: error.code
            )
        } catch {
            statuses[repoID] = ModelDownloadStatus(
                repoID: repoID,
                status: "failed",
                progressPct: statuses[repoID]?.progressPct ?? 0,
                error: error.localizedDescription,
                errorCode: nil
            )
        }
    }

    private func downloadModelScope(repoID: String) async throws {
        let info = try await fetchModelScopeFiles(repoID: repoID)
        if info.success == false {
            throw DownloadFailure(repoID: repoID, code: "repo_not_found", message: "ModelScope rejected repo \"\(repoID)\".")
        }
        let fileNames = info.data?.files.compactMap(\.name) ?? []
        guard !fileNames.isEmpty else {
            throw DownloadFailure(repoID: repoID, code: "repo_not_found", message: "Repository \"\(repoID)\" has no files.")
        }
        guard fileNames.contains("config.json") else {
            throw DownloadFailure(repoID: repoID, code: "repo_missing_config", message: "Repository \"\(repoID)\" is missing config.json.")
        }
        guard fileNames.contains("tokenizer.json") else {
            throw DownloadFailure(repoID: repoID, code: "repo_missing_tokenizer", message: "Repository \"\(repoID)\" is missing tokenizer.json.")
        }

        let snapshot = snapshotURL(sourceDirectory: "models-ms", repoID: repoID, snapshotID: "main")
        try FileManager.default.createDirectory(at: snapshot, withIntermediateDirectories: true)
        let files = uniqueValidRepositoryPaths(
            fileNames.filter {
                !$0.hasPrefix(".") && $0 != "README.md" && $0 != "configuration.json"
            }
        )

        for (index, filename) in files.enumerated() {
            let url = modelScopeURL(repoID: repoID, filename: filename)
            try await downloadFile(request: URLRequest(url: url), filename: filename, to: snapshot)
            statuses[repoID]?.progressPct = Double(index + 1) / Double(files.count) * 100
        }
    }

    private func fetchModelScopeFiles(repoID: String) async throws -> ModelScopeFilesResponse {
        let url = modelScopeEndpoint
            .appendingPathComponent(repoID)
            .appendingPathComponent("repo")
            .appendingPathComponent("files")
        let request = URLRequest(url: url)
        do {
            let (data, _) = try await httpClient.data(for: request)
            return try JSONDecoder().decode(ModelScopeFilesResponse.self, from: data)
        } catch let error as ModelDownloadHTTPError where error.statusCode == 404 || error.statusCode == 401 || error.statusCode == 403 {
            throw DownloadFailure(repoID: repoID, code: "repo_not_found", message: "ModelScope returned HTTP \(error.statusCode) for repo \"\(repoID)\".")
        }
    }

    private func downloadFile(request: URLRequest, filename: String, to snapshot: URL) async throws {
        let destination = snapshot.appendingPathComponent(filename)
        try FileManager.default.createDirectory(
            at: destination.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        let partial = destination.appendingPathExtension("partial")
        if FileManager.default.fileExists(atPath: partial.path) {
            try FileManager.default.removeItem(at: partial)
        }
        let (downloaded, _) = try await httpClient.download(for: request)
        if FileManager.default.fileExists(atPath: destination.path) {
            try FileManager.default.removeItem(at: destination)
        }
        try FileManager.default.moveItem(at: downloaded, to: partial)
        try FileManager.default.moveItem(at: partial, to: destination)
    }

    private func modelExistsLocally(_ repoID: String) -> Bool {
        LocalModelScanner(rootURL: rootURL).resolveModelPath(for: repoID) != nil
    }

    private func snapshotURL(sourceDirectory: String, repoID: String, snapshotID: String) -> URL {
        rootURL
            .appendingPathComponent(sourceDirectory, isDirectory: true)
            .appendingPathComponent("models--" + repoID.replacingOccurrences(of: "/", with: "--"), isDirectory: true)
            .appendingPathComponent("snapshots", isDirectory: true)
            .appendingPathComponent(snapshotID, isDirectory: true)
    }

    private func applyHuggingFaceToken(_ token: String?, to request: inout URLRequest) {
        guard let token = token?.trimmingCharacters(in: .whitespacesAndNewlines),
              !token.isEmpty else {
            return
        }
        request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
    }

    private func huggingFaceURL(pathComponents: [String]) -> URL {
        pathComponents.reduce(huggingFaceEndpoint) { url, component in
            url.appendingPathComponent(component)
        }
    }

    private func modelScopeURL(repoID: String, filename: String) -> URL {
        var components = URLComponents(
            url: modelScopeEndpoint
                .appendingPathComponent(repoID)
                .appendingPathComponent("repo"),
            resolvingAgainstBaseURL: false
        )
        components?.queryItems = [
            URLQueryItem(name: "Revision", value: "master"),
            URLQueryItem(name: "FilePath", value: filename),
        ]
        return components?.url ?? modelScopeEndpoint
    }

    private func repoPathComponents(_ repoID: String) -> [String] {
        repoID.split(separator: "/").map(String.init)
    }

    private func repositoryPathComponents(_ filename: String) -> [String] {
        filename.split(separator: "/").map(String.init)
    }

    private func uniqueValidRepositoryPaths(_ files: [String]) -> [String] {
        var seen = Set<String>()
        var result: [String] = []
        for file in files {
            guard !file.isEmpty,
                  !file.hasPrefix("/"),
                  !file.split(separator: "/").contains(".."),
                  seen.insert(file).inserted else {
                continue
            }
            result.append(file)
        }
        return result
    }

    private func safetensorsShards(from data: Data) -> [String]? {
        guard let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let weightMap = object["weight_map"] as? [String: Any] else {
            return nil
        }
        let shards = Set(weightMap.values.compactMap { $0 as? String })
        return shards.sorted()
    }

    private func failure(message: String, code: String?, repoID: String) -> ModelDownloadCompletion {
        ModelDownloadCompletion(
            success: false,
            message: nil,
            error: message,
            code: code,
            repoID: repoID
        )
    }
}

private struct DownloadFailure: Error {
    var repoID: String
    var code: String?
    var message: String
}

private struct HuggingFaceModelInfo: Decodable {
    var siblings: [HuggingFaceSibling]
}

private struct HuggingFaceSibling: Decodable {
    var rfilename: String?
}

private struct ModelScopeFilesResponse: Decodable {
    var success: Bool?
    var data: ModelScopeFilesData?

    enum CodingKeys: String, CodingKey {
        case success = "Success"
        case data = "Data"
    }
}

private struct ModelScopeFilesData: Decodable {
    var files: [ModelScopeFile]

    enum CodingKeys: String, CodingKey {
        case files = "Files"
    }
}

private struct ModelScopeFile: Decodable {
    var name: String?

    enum CodingKeys: String, CodingKey {
        case name = "Name"
    }
}
