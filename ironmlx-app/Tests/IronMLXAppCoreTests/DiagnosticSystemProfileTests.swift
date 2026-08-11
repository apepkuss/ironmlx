import Foundation
import Testing

@testable import IronMLXAppCore

@Test func diagnosticBuildIdentityNeverReadsRepositoryFallbacks() {
    let identity = DiagnosticBuildIdentity(bundle: .main)

    #expect(!identity.appVersion.isEmpty)
    #expect(!identity.appBuild.isEmpty)
    #expect(!identity.sourceCommit.isEmpty)
    #expect(!identity.mlxCommit.isEmpty)
    #expect(!identity.distributionChannel.isEmpty)
}

@Test func diagnosticSystemSnapshotHasOnlyNonIdentifyingFields() throws {
    let data = try JSONEncoder.ironMLXDiagnostic.encode(DiagnosticSystemProfiler().snapshot())
    let json = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])

    for prohibited in ["username", "hostname", "serial_number", "apple_id", "mac_address"] {
        #expect(json[prohibited] == nil)
    }
    #expect(json["physical_memory_bytes"] != nil)
    #expect(json["signature_validity"] != nil)
    #expect(json["notarization_status"] != nil)
}

@Test func bundleScriptsInjectAndVerifyDiagnosticBuildIdentity() throws {
    let build = try String(contentsOfFile: "../scripts/build-app-bundle.sh", encoding: .utf8)
    let verify = try String(contentsOfFile: "../scripts/verify-app-bundle.sh", encoding: .utf8)
    let preview = try String(contentsOfFile: "../scripts/package-development-preview.sh", encoding: .utf8)

    for key in [
        "IronMLXSourceCommit", "IronMLXSourceTreeState", "IronMLXMLXCommit",
        "IronMLXDistributionChannel", "IronMLXDeveloperIDSigned", "IronMLXNotarizationStatus",
    ] {
        #expect(build.contains(key), "build script missing \(key)")
        #expect(verify.contains(key), "verify script missing \(key)")
        #expect(preview.contains(key), "preview script missing \(key)")
    }
    #expect(!build.contains("/Users/xin"))
    #expect(!preview.contains("/Users/xin"))
}
