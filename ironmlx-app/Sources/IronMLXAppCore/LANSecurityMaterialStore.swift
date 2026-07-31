import CryptoKit
import Foundation
import Security

public enum LANSecurityMaterialError: LocalizedError, Equatable {
    case invalidLANAddress
    case keychain(OSStatus)
    case materialMissing
    case certificateGenerationFailed

    public var errorDescription: String? {
        switch self {
        case .invalidLANAddress:
            return "LAN mode requires a concrete active LAN IP address."
        case .keychain(let status):
            return "The LAN credential could not be accessed in Keychain (status \(status))."
        case .materialMissing:
            return "LAN authentication or TLS material is missing."
        case .certificateGenerationFailed:
            return "The LAN TLS certificate could not be generated."
        }
    }
}

public struct LANSecurityMetadata: Equatable, Sendable {
    public var credentialID: String
    public var certificateFingerprint: String

    public init(credentialID: String, certificateFingerprint: String) {
        self.credentialID = credentialID
        self.certificateFingerprint = certificateFingerprint
    }
}

public struct BackendSecurityBootstrap: Encodable, Equatable, Sendable {
    public var apiKeySHA256: Data
    public var tlsCertificatePEM: Data
    public var tlsPrivateKeyPEM: Data

    enum CodingKeys: String, CodingKey {
        case apiKeySHA256 = "api_key_sha256_base64"
        case tlsCertificatePEM = "tls_certificate_pem_base64"
        case tlsPrivateKeyPEM = "tls_private_key_pem_base64"
    }
}

private struct StoredLANSecurityMaterial: Codable {
    var lanHost: String
    var apiKey: String
    var caCertificatePEM: Data
    var serverCertificatePEM: Data
    var serverPrivateKeyPEM: Data

    var fingerprint: String {
        SHA256.hash(data: caCertificatePEM).map { String(format: "%02x", $0) }.joined()
    }

    var bootstrap: BackendSecurityBootstrap {
        BackendSecurityBootstrap(
            apiKeySHA256: Data(SHA256.hash(data: Data(apiKey.utf8))),
            tlsCertificatePEM: serverCertificatePEM + caCertificatePEM,
            tlsPrivateKeyPEM: serverPrivateKeyPEM
        )
    }
}

public final class LANSecurityMaterialStore: @unchecked Sendable {
    public static let shared = LANSecurityMaterialStore()

    private let service: String
    private let fileManager: FileManager

    public init(fileManager: FileManager = .default) {
        self.service = "com.ironmlx.lan-security.v1"
        self.fileManager = fileManager
    }

    init(fileManager: FileManager = .default, service: String) {
        self.service = service
        self.fileManager = fileManager
    }

    public func ensureMaterial(
        lanHost: String,
        credentialID: String?
    ) throws -> LANSecurityMetadata {
        guard EndpointPayload.isSafeLANAddress(lanHost) else {
            throw LANSecurityMaterialError.invalidLANAddress
        }
        if let credentialID,
           let material = try load(credentialID: credentialID),
           material.lanHost == lanHost {
            return LANSecurityMetadata(
                credentialID: credentialID,
                certificateFingerprint: material.fingerprint
            )
        }
        let credentialID = UUID().uuidString.lowercased()
        let material = try Self.generateMaterial(lanHost: lanHost, fileManager: fileManager)
        try save(material, credentialID: credentialID)
        return LANSecurityMetadata(
            credentialID: credentialID,
            certificateFingerprint: material.fingerprint
        )
    }

    public func bootstrap(credentialID: String, lanHost: String) throws -> Data {
        guard let material = try load(credentialID: credentialID), material.lanHost == lanHost else {
            throw LANSecurityMaterialError.materialMissing
        }
        return try JSONEncoder().encode(material.bootstrap)
    }

    public func apiKey(credentialID: String) throws -> String {
        guard let material = try load(credentialID: credentialID) else {
            throw LANSecurityMaterialError.materialMissing
        }
        return material.apiKey
    }

    public func caCertificate(credentialID: String) throws -> Data {
        guard let material = try load(credentialID: credentialID) else {
            throw LANSecurityMaterialError.materialMissing
        }
        return material.caCertificatePEM
    }

    public func rotate(lanHost: String) throws -> LANSecurityMetadata {
        try ensureMaterial(lanHost: lanHost, credentialID: nil)
    }

    public func deleteMaterial(credentialID: String) throws {
        let status = SecItemDelete(baseQuery(credentialID: credentialID) as CFDictionary)
        guard status == errSecSuccess || status == errSecItemNotFound else {
            throw LANSecurityMaterialError.keychain(status)
        }
    }

    private func load(credentialID: String) throws -> StoredLANSecurityMaterial? {
        var query = baseQuery(credentialID: credentialID)
        query[kSecReturnData as String] = true
        query[kSecMatchLimit as String] = kSecMatchLimitOne
        var result: CFTypeRef?
        let status = SecItemCopyMatching(query as CFDictionary, &result)
        if status == errSecItemNotFound {
            return nil
        }
        guard status == errSecSuccess, let data = result as? Data else {
            throw LANSecurityMaterialError.keychain(status)
        }
        return try JSONDecoder().decode(StoredLANSecurityMaterial.self, from: data)
    }

    private func save(_ material: StoredLANSecurityMaterial, credentialID: String) throws {
        let data = try JSONEncoder().encode(material)
        var query = baseQuery(credentialID: credentialID)
        query[kSecValueData as String] = data
        let status = SecItemAdd(query as CFDictionary, nil)
        guard status == errSecSuccess else {
            throw LANSecurityMaterialError.keychain(status)
        }
    }

    private func baseQuery(credentialID: String) -> [String: Any] {
        [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: credentialID,
        ]
    }

    private static func generateMaterial(
        lanHost: String,
        fileManager: FileManager
    ) throws -> StoredLANSecurityMaterial {
        let root = fileManager.temporaryDirectory
            .appendingPathComponent("ironmlx-lan-tls-\(UUID().uuidString)", isDirectory: true)
        try fileManager.createDirectory(
            at: root,
            withIntermediateDirectories: false,
            attributes: [.posixPermissions: 0o700]
        )
        defer { try? fileManager.removeItem(at: root) }

        let caKey = root.appendingPathComponent("ca-key.pem")
        let caCertificate = root.appendingPathComponent("ca-certificate.pem")
        let serverKey = root.appendingPathComponent("server-key.pem")
        let serverRequest = root.appendingPathComponent("server.csr")
        let serverCertificate = root.appendingPathComponent("server-certificate.pem")
        let extensions = root.appendingPathComponent("extensions.cnf")
        try "subjectAltName=IP:\(lanHost)\nextendedKeyUsage=serverAuth\nkeyUsage=digitalSignature,keyEncipherment\n"
            .write(to: extensions, atomically: true, encoding: .utf8)

        try runOpenSSL(["genrsa", "-out", caKey.path, "3072"])
        try runOpenSSL([
            "req", "-x509", "-new", "-sha256", "-days", "3650",
            "-key", caKey.path, "-subj", "/CN=IronMLX Local CA", "-out", caCertificate.path,
        ])
        try runOpenSSL(["genrsa", "-out", serverKey.path, "2048"])
        try runOpenSSL([
            "req", "-new", "-sha256", "-key", serverKey.path,
            "-subj", "/CN=\(lanHost)", "-out", serverRequest.path,
        ])
        try runOpenSSL([
            "x509", "-req", "-sha256", "-days", "825",
            "-in", serverRequest.path, "-CA", caCertificate.path, "-CAkey", caKey.path,
            "-CAcreateserial", "-extfile", extensions.path, "-out", serverCertificate.path,
        ])
        try fileManager.setAttributes([.posixPermissions: 0o600], ofItemAtPath: caKey.path)
        try fileManager.setAttributes([.posixPermissions: 0o600], ofItemAtPath: serverKey.path)

        var random = [UInt8](repeating: 0, count: 32)
        guard SecRandomCopyBytes(kSecRandomDefault, random.count, &random) == errSecSuccess else {
            throw LANSecurityMaterialError.certificateGenerationFailed
        }
        let key = Data(random).base64EncodedString()
            .replacingOccurrences(of: "+", with: "-")
            .replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: "=", with: "")
        return StoredLANSecurityMaterial(
            lanHost: lanHost,
            apiKey: "imx_\(key)",
            caCertificatePEM: try Data(contentsOf: caCertificate),
            serverCertificatePEM: try Data(contentsOf: serverCertificate),
            serverPrivateKeyPEM: try Data(contentsOf: serverKey)
        )
    }

    private static func runOpenSSL(_ arguments: [String]) throws {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/openssl")
        process.arguments = arguments
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice
        do {
            try process.run()
            process.waitUntilExit()
        } catch {
            throw LANSecurityMaterialError.certificateGenerationFailed
        }
        guard process.terminationStatus == 0 else {
            throw LANSecurityMaterialError.certificateGenerationFailed
        }
    }
}
