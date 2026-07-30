import Foundation

public struct RemoteModelFile: Equatable, Sendable {
    public var path: String
    public var size: Int64
    public var sha256: String?
    public var blobID: String?
    public var etag: String?

    public var isWeight: Bool {
        path.hasSuffix(".safetensors")
    }
}

public struct ResolvedModelRepository: Equatable, Sendable {
    public var provider: ModelRepositoryProvider
    public var repoID: String
    public var requestedRevision: String
    public var commitSHA: String
    public var files: [RemoteModelFile]
}

public struct ModelRepositoryResolver: Sendable {
    private let httpClient: any ModelDownloadHTTPClient
    private let huggingFaceEndpoint: URL
    private let modelScopeAPIEndpoint: URL
    private let modelScopeGitEndpoint: URL

    public init(
        httpClient: any ModelDownloadHTTPClient,
        huggingFaceEndpoint: URL,
        modelScopeAPIEndpoint: URL,
        modelScopeGitEndpoint: URL = URL(string: "https://www.modelscope.cn")!
    ) {
        self.httpClient = httpClient
        self.huggingFaceEndpoint = huggingFaceEndpoint
        self.modelScopeAPIEndpoint = modelScopeAPIEndpoint
        self.modelScopeGitEndpoint = modelScopeGitEndpoint
    }

    public func resolve(
        provider: ModelRepositoryProvider,
        repoID: String,
        token: String?
    ) async throws -> ResolvedModelRepository {
        switch provider {
        case .huggingFace:
            try await resolveHuggingFaceRevision(repoID: repoID, revision: nil, token: token)
        case .modelScope:
            try await resolveModelScope(repoID: repoID)
        }
    }

    public func request(
        for file: RemoteModelFile,
        repository: ResolvedModelRepository,
        token: String?
    ) throws -> URLRequest {
        let url: URL
        switch repository.provider {
        case .huggingFace:
            url = pathURL(
                base: huggingFaceEndpoint,
                components: repoComponents(repository.repoID)
                    + ["resolve", repository.commitSHA]
                    + repositoryPathComponents(file.path)
            )
        case .modelScope:
            var components = URLComponents(
                url: pathURL(
                    base: modelScopeAPIEndpoint,
                    components: repoComponents(repository.repoID) + ["repo"]
                ),
                resolvingAgainstBaseURL: false
            )
            components?.queryItems = [
                URLQueryItem(name: "Revision", value: repository.commitSHA),
                URLQueryItem(name: "FilePath", value: file.path),
            ]
            guard let resolved = components?.url else {
                throw URLError(.badURL)
            }
            url = resolved
        }
        var request = URLRequest(url: url)
        if repository.provider == .huggingFace {
            applyHuggingFaceToken(token, to: &request)
        }
        return request
    }

    public func resolveHuggingFace(
        repoID: String,
        revision: String,
        token: String?
    ) async throws -> ResolvedModelRepository {
        try await resolveHuggingFaceRevision(repoID: repoID, revision: revision, token: token)
    }

    private func resolveHuggingFaceRevision(
        repoID: String,
        revision: String?,
        token: String?
    ) async throws -> ResolvedModelRepository {
        var endpoint = pathURL(base: huggingFaceEndpoint, components: ["api", "models"] + repoComponents(repoID))
        if let revision {
            endpoint = pathURL(base: endpoint, components: ["revision", revision])
        }
        var components = URLComponents(
            url: endpoint,
            resolvingAgainstBaseURL: false
        )
        components?.queryItems = [URLQueryItem(name: "blobs", value: "true")]
        guard let url = components?.url else {
            throw URLError(.badURL)
        }
        var request = URLRequest(url: url)
        applyHuggingFaceToken(token, to: &request)
        let data: Data
        do {
            (data, _) = try await httpClient.data(for: request)
        } catch let error as ModelDownloadHTTPError
            where error.statusCode == 401 || error.statusCode == 403 || error.statusCode == 404
        {
            throw RepositoryResolutionError.notFound(repoID)
        }
        let info = try JSONDecoder().decode(HuggingFaceRepositoryInfo.self, from: data)
        guard let commitSHA = info.sha?.lowercased(),
              ModelSnapshotVerifier.isCommitSHA(commitSHA)
        else {
            throw RepositoryResolutionError.invalidCommit("Hugging Face did not return a full commit SHA.")
        }
        let files = try info.siblings.map { sibling in
            guard let path = sibling.rfilename,
                  let size = sibling.size ?? sibling.lfs?.size,
                  size >= 0
            else {
                throw RepositoryResolutionError.incompleteMetadata(
                    sibling.rfilename ?? "<unknown>",
                    "missing size"
                )
            }
            return RemoteModelFile(
                path: path,
                size: size,
                sha256: sibling.lfs?.sha256?.lowercased(),
                blobID: sibling.blobID,
                etag: nil
            )
        }
        return ResolvedModelRepository(
            provider: .huggingFace,
            repoID: repoID,
            requestedRevision: revision ?? ModelRepositoryProvider.huggingFace.mutableRevision,
            commitSHA: commitSHA,
            files: try validatedFiles(files)
        )
    }

    private func resolveModelScope(repoID: String) async throws -> ResolvedModelRepository {
        let commitSHA = try await resolveModelScopeMaster(repoID: repoID)
        var components = URLComponents(
            url: pathURL(
                base: modelScopeAPIEndpoint,
                components: repoComponents(repoID) + ["repo", "files"]
            ),
            resolvingAgainstBaseURL: false
        )
        components?.queryItems = [
            URLQueryItem(name: "Revision", value: commitSHA),
            URLQueryItem(name: "Recursive", value: "true"),
        ]
        guard let url = components?.url else {
            throw URLError(.badURL)
        }
        let data: Data
        do {
            (data, _) = try await httpClient.data(for: URLRequest(url: url))
        } catch let error as ModelDownloadHTTPError
            where error.statusCode == 401 || error.statusCode == 403 || error.statusCode == 404
        {
            throw RepositoryResolutionError.notFound(repoID)
        }
        let response = try JSONDecoder().decode(ModelScopeRepositoryFilesResponse.self, from: data)
        guard response.success != false else {
            throw RepositoryResolutionError.notFound(repoID)
        }
        let files = try (response.data?.files ?? []).compactMap { file -> RemoteModelFile? in
            if file.type == "tree" {
                return nil
            }
            guard let path = file.path ?? file.name,
                  let size = file.size,
                  let sha256 = file.sha256?.lowercased(),
                  size >= 0,
                  sha256.count == 64,
                  sha256.allSatisfy(\.isHexDigit)
            else {
                throw RepositoryResolutionError.incompleteMetadata(
                    file.path ?? file.name ?? "<unknown>",
                    "missing size or SHA-256"
                )
            }
            return RemoteModelFile(path: path, size: size, sha256: sha256, blobID: nil, etag: nil)
        }
        return ResolvedModelRepository(
            provider: .modelScope,
            repoID: repoID,
            requestedRevision: ModelRepositoryProvider.modelScope.mutableRevision,
            commitSHA: commitSHA,
            files: try validatedFiles(files)
        )
    }

    private func resolveModelScopeMaster(repoID: String) async throws -> String {
        let repository = repoComponents(repoID)
        guard repository.count == 2 else {
            throw RepositoryResolutionError.invalidCommit("ModelScope repository ID must be organization/model.")
        }
        var components = URLComponents(
            url: pathURL(
                base: modelScopeGitEndpoint,
                components: [repository[0], repository[1] + ".git", "info", "refs"]
            ),
            resolvingAgainstBaseURL: false
        )
        components?.queryItems = [URLQueryItem(name: "service", value: "git-upload-pack")]
        guard let url = components?.url else {
            throw URLError(.badURL)
        }
        let (data, _) = try await httpClient.data(for: URLRequest(url: url))
        guard let advertisement = String(data: data, encoding: .utf8) else {
            throw RepositoryResolutionError.invalidCommit("ModelScope returned an invalid Git ref advertisement.")
        }
        let marker = " refs/heads/master"
        for line in advertisement.components(separatedBy: "\n") where line.contains(marker) {
            guard let markerRange = line.range(of: marker) else {
                continue
            }
            let prefix = line[..<markerRange.lowerBound]
            let candidate = String(prefix.suffix(40))
            guard ModelSnapshotVerifier.isCommitSHA(candidate)
            else {
                continue
            }
            return candidate.lowercased()
        }
        throw RepositoryResolutionError.invalidCommit("ModelScope master could not be resolved to a full commit SHA.")
    }

    private func validatedFiles(_ files: [RemoteModelFile]) throws -> [RemoteModelFile] {
        var seen = Set<String>()
        var validated: [RemoteModelFile] = []
        for file in files.sorted(by: { $0.path < $1.path }) {
            guard file.size >= 0 else {
                throw RepositoryResolutionError.incompleteMetadata(file.path, "negative size")
            }
            if let sha256 = file.sha256,
               sha256.count != 64 || !sha256.allSatisfy(\.isHexDigit) {
                throw RepositoryResolutionError.incompleteMetadata(
                    file.path,
                    "invalid SHA-256"
                )
            }
            _ = try ModelSnapshotVerifier.safeFileURL(
                path: file.path,
                beneath: URL(fileURLWithPath: "/repository", isDirectory: true)
            )
            guard seen.insert(file.path).inserted else {
                throw RepositoryResolutionError.incompleteMetadata(file.path, "duplicate path")
            }
            validated.append(file)
        }
        return validated
    }

    private func pathURL(base: URL, components: [String]) -> URL {
        components.reduce(base) { $0.appendingPathComponent($1) }
    }

    private func repoComponents(_ repoID: String) -> [String] {
        repoID.split(separator: "/").map(String.init)
    }

    private func repositoryPathComponents(_ path: String) -> [String] {
        path.split(separator: "/").map(String.init)
    }

    private func applyHuggingFaceToken(_ token: String?, to request: inout URLRequest) {
        guard let token = token?.trimmingCharacters(in: .whitespacesAndNewlines),
              !token.isEmpty
        else {
            return
        }
        request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
    }
}

public enum RepositoryResolutionError: LocalizedError {
    case notFound(String)
    case invalidCommit(String)
    case incompleteMetadata(String, String)

    public var errorDescription: String? {
        switch self {
        case let .notFound(repoID):
            "Repository \(repoID) does not exist or requires credentials."
        case let .invalidCommit(detail):
            detail
        case let .incompleteMetadata(path, detail):
            "Repository file \(path) has incomplete identity metadata: \(detail)."
        }
    }
}

private struct HuggingFaceRepositoryInfo: Decodable {
    var sha: String?
    var siblings: [HuggingFaceRepositorySibling]
}

private struct HuggingFaceRepositorySibling: Decodable {
    var rfilename: String?
    var size: Int64?
    var blobID: String?
    var lfs: HuggingFaceLFSFile?

    enum CodingKeys: String, CodingKey {
        case rfilename
        case size
        case blobID = "blobId"
        case lfs
    }
}

private struct HuggingFaceLFSFile: Decodable {
    var sha256: String?
    var size: Int64?
}

private struct ModelScopeRepositoryFilesResponse: Decodable {
    var success: Bool?
    var data: ModelScopeRepositoryFilesData?

    enum CodingKeys: String, CodingKey {
        case success = "Success"
        case data = "Data"
    }
}

private struct ModelScopeRepositoryFilesData: Decodable {
    var files: [ModelScopeRepositoryFile]

    enum CodingKeys: String, CodingKey {
        case files = "Files"
    }
}

private struct ModelScopeRepositoryFile: Decodable {
    var name: String?
    var path: String?
    var type: String?
    var size: Int64?
    var sha256: String?

    enum CodingKeys: String, CodingKey {
        case name = "Name"
        case path = "Path"
        case type = "Type"
        case size = "Size"
        case sha256 = "Sha256"
    }
}
