import Darwin
import Foundation
import Security

public struct DiagnosticBuildIdentity: Sendable, Equatable {
    public var appVersion: String
    public var appBuild: String
    public var sourceCommit: String
    public var sourceTreeState: String
    public var mlxCommit: String
    public var distributionChannel: String
    public var declaredDeveloperIDStatus: String
    public var declaredNotarizationStatus: String

    public init(bundle: Bundle = .main) {
        let info = bundle.infoDictionary ?? [:]
        appVersion = info["CFBundleShortVersionString"] as? String ?? "unavailable"
        appBuild = info["CFBundleVersion"] as? String ?? "unavailable"
        sourceCommit = info["IronMLXSourceCommit"] as? String ?? "unavailable"
        sourceTreeState = info["IronMLXSourceTreeState"] as? String ?? "unavailable"
        mlxCommit = info["IronMLXMLXCommit"] as? String ?? "unavailable"
        distributionChannel = info["IronMLXDistributionChannel"] as? String ?? "unavailable"
        declaredDeveloperIDStatus = info["IronMLXDeveloperIDSigned"] as? String ?? "unavailable"
        declaredNotarizationStatus = info["IronMLXNotarizationStatus"] as? String ?? "unavailable"
    }
}

public struct DiagnosticSystemProfiler: Sendable {
    public init() {}

    public func snapshot(
        bundleURL: URL = Bundle.main.bundleURL,
        buildIdentity: DiagnosticBuildIdentity = DiagnosticBuildIdentity()
    ) -> DiagnosticSystemSnapshot {
        let signature = signatureState(bundleURL: bundleURL)
        let stapled = stapledTicketState(bundleURL: bundleURL)
        let notarization = stapled == "present"
            ? "stapled"
            : buildIdentity.declaredNotarizationStatus
        return DiagnosticSystemSnapshot(
            macOSVersion: ProcessInfo.processInfo.operatingSystemVersionString,
            macOSBuild: sysctlString("kern.osversion") ?? "unavailable",
            chip: sysctlString("machdep.cpu.brand_string") ?? "Apple Silicon",
            physicalMemoryBytes: ProcessInfo.processInfo.physicalMemory,
            appArchitecture: architecture,
            signatureValidity: signature.validity,
            signatureKind: signature.kind,
            developerIDStatus: signature.developerID,
            notarizationStatus: notarization,
            stapledTicketStatus: stapled
        )
    }

    private var architecture: String {
        #if arch(arm64)
        "arm64"
        #elseif arch(x86_64)
        "x86_64"
        #else
        "unknown"
        #endif
    }

    private func signatureState(bundleURL: URL) -> (validity: String, kind: String, developerID: String) {
        var code: SecStaticCode?
        guard SecStaticCodeCreateWithPath(bundleURL as CFURL, [], &code) == errSecSuccess,
              let code else {
            return ("unavailable", "unavailable", "unavailable")
        }
        let validity = SecStaticCodeCheckValidity(code, [], nil) == errSecSuccess ? "valid" : "invalid"
        var information: CFDictionary?
        guard SecCodeCopySigningInformation(code, SecCSFlags(rawValue: kSecCSSigningInformation), &information) == errSecSuccess,
              let values = information as? [String: Any] else {
            return (validity, "unavailable", "unavailable")
        }
        let teamID = values[kSecCodeInfoTeamIdentifier as String] as? String
        let adHoc = teamID?.isEmpty != false
        var developerIDRequirement: SecRequirement?
        let developerIDRequirementText = """
        anchor apple generic and certificate 1[field.1.2.840.113635.100.6.2.6] exists \
        and certificate leaf[field.1.2.840.113635.100.6.1.13] exists
        """
        let hasDeveloperID = SecRequirementCreateWithString(
            developerIDRequirementText as CFString,
            [],
            &developerIDRequirement
        ) == errSecSuccess && developerIDRequirement.map {
            SecStaticCodeCheckValidity(code, [], $0) == errSecSuccess
        } == true
        return (
            validity,
            adHoc ? "ad_hoc" : "signed",
            adHoc ? "unsigned" : (hasDeveloperID ? "developer_id" : "not_developer_id")
        )
    }

    private func stapledTicketState(bundleURL: URL) -> String {
        var code: SecStaticCode?
        guard SecStaticCodeCreateWithPath(bundleURL as CFURL, [], &code) == errSecSuccess,
              let code else { return "unavailable" }
        var information: CFDictionary?
        guard SecCodeCopySigningInformation(code, SecCSFlags(rawValue: kSecCSSigningInformation), &information) == errSecSuccess,
              let values = information as? [String: Any] else { return "unavailable" }
        return values[kSecCodeInfoStapledNotarizationTicket as String] == nil ? "absent" : "present"
    }

    private func sysctlString(_ name: String) -> String? {
        var size = 0
        guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 1, size < 4_096 else { return nil }
        var bytes = [CChar](repeating: 0, count: size)
        guard sysctlbyname(name, &bytes, &size, nil, 0) == 0 else { return nil }
        let unsigned = bytes.map(UInt8.init).prefix { $0 != 0 }
        return String(decoding: unsigned, as: UTF8.self)
    }
}
