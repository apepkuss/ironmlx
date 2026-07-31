import Foundation
import Testing

@testable import IronMLXAppCore

@Test func lanConfigPersistsOnlyPublicCredentialMetadata() throws {
    let config = AppConfig(
        networkMode: "lan",
        lanHost: "192.168.1.24",
        lanCredentialID: "credential-id",
        lanCertificateFingerprint: "certificate-fingerprint"
    )
    let json = try #require(String(data: JSONEncoder().encode(config), encoding: .utf8))

    #expect(json.contains("credential-id"))
    #expect(json.contains("certificate-fingerprint"))
    #expect(!json.contains("api_key"))
    #expect(!json.contains("private_key"))
    #expect(!json.contains("imx_"))
}

@Test func backendSecurityBootstrapUsesOnlyDigestAndTLSMaterial() throws {
    let bootstrap = BackendSecurityBootstrap(
        apiKeySHA256: Data(repeating: 7, count: 32),
        tlsCertificatePEM: Data("certificate".utf8),
        tlsPrivateKeyPEM: Data("private-key".utf8)
    )
    let object = try #require(
        JSONSerialization.jsonObject(with: JSONEncoder().encode(bootstrap)) as? [String: Any]
    )

    #expect(object["api_key_sha256_base64"] != nil)
    #expect(object["tls_certificate_pem_base64"] != nil)
    #expect(object["tls_private_key_pem_base64"] != nil)
    #expect(object["api_key"] == nil)
}

@Test func lanLaunchValidationRejectsMissingBootstrap() {
    let config = BackendLaunchConfiguration(
        executableURL: URL(fileURLWithPath: "/tmp/ironmlx"),
        host: "127.0.0.1",
        port: 9068,
        networkMode: "lan",
        lanHost: nil,
        securityBootstrapStdin: false
    )

    #expect(config.validationError == .lanBindAddressRequired)
}

@MainActor
@Test func lanKeychainFailuresUseAStableLocalizedErrorCode() throws {
    let json = DashboardBridge.settingsErrorJSON(LANSecurityMaterialError.keychain(-34_018))
    let object = try #require(
        JSONSerialization.jsonObject(with: Data(json.utf8)) as? [String: Any]
    )

    #expect(object["status"] as? String == "error")
    #expect(object["code"] as? String == "lan_keychain_unavailable")
    #expect(object["os_status"] as? Int == -34_018)
}

@Test func p06DashboardSecurityUIIsLocalizedForEverySupportedLanguage() throws {
    let html = try String(
        contentsOfFile: "Sources/IronMLXAppCore/Resources/dashboard2.html",
        encoding: .utf8
    )
    let requiredKeys = [
        "network_mode", "network_mode_desc", "network_mode_local", "network_mode_lan",
        "lan_interface", "lan_interface_desc", "lan_access_credentials", "lan_access_credentials_desc",
        "copy_api_key", "copy_ca_certificate", "rotate_credentials",
        "endpoint_address", "copy_endpoint", "runtime_network_local", "runtime_network_lan",
        "authentication_status", "authentication_status_local", "authentication_status_api_key",
        "endpoints_hint_local", "endpoints_hint_lan",
        "msg_lan_api_key_copied", "msg_lan_ca_certificate_copied",
        "msg_lan_security_rotated_and_copied", "err_lan_address_invalid",
        "err_lan_keychain_unavailable", "err_lan_security_material_missing",
        "err_lan_certificate_generation_failed", "err_lan_mode_required",
        "err_lan_security_rotation_rolled_back", "err_lan_security_rotation_failed",
        "err_lan_security_action_failed", "err_network_restart_rolled_back",
        "err_settings_invalid",
    ]
    for key in requiredKeys {
        let definitionCount = html.components(separatedBy: "\(key): \"").count - 1
        #expect(definitionCount == 5, "\(key) must be defined for all five languages")
    }

    let requiredBindings = [
        "network_mode", "network_mode_desc", "network_mode_local", "network_mode_lan",
        "lan_interface", "lan_interface_desc", "lan_access_credentials", "lan_access_credentials_desc",
        "copy_api_key", "copy_ca_certificate", "rotate_credentials",
    ]
    for key in requiredBindings {
        #expect(html.contains("data-i18n=\"\(key)\""), "\(key) must be bound to the UI")
    }

    #expect(!html.contains("服务当前无认证"))
    #expect(!html.contains("服務目前無驗證"))
    #expect(!html.contains("このサーバーには認証がありません"))
    #expect(!html.contains("이 서버는 인증이 없습니다"))
    #expect(!html.contains("Host 改为「所有接口 (0.0.0.0)」"))
    #expect(!html.contains("Host 改為「所有介面 (0.0.0.0)」"))
}

@Test func lanSecurityMaterialRoundTripsInMacOSKeychainWhenEnabled() throws {
    guard ProcessInfo.processInfo.environment["IRONMLX_KEYCHAIN_INTEGRATION"] == "1" else {
        return
    }
    let lanHost = try #require(EndpointPayload.localNetworkInterfaces().first?.ip)
    let store = LANSecurityMaterialStore(
        service: "com.ironmlx.lan-security.test.\(UUID().uuidString.lowercased())"
    )
    let metadata = try store.ensureMaterial(lanHost: lanHost, credentialID: nil)
    defer { try? store.deleteMaterial(credentialID: metadata.credentialID) }

    #expect(try store.apiKey(credentialID: metadata.credentialID).hasPrefix("imx_"))
    #expect(
        try String(
            decoding: store.caCertificate(credentialID: metadata.credentialID),
            as: UTF8.self
        ).contains("BEGIN CERTIFICATE")
    )
    #expect(try !store.bootstrap(credentialID: metadata.credentialID, lanHost: lanHost).isEmpty)
}
