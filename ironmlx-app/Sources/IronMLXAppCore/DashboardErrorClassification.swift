import Foundation

public protocol DashboardErrorCodeProviding: Error {
    var dashboardErrorCode: String { get }
}

public enum DashboardErrorClassifier {
    public static func code(for error: Error) -> String {
        (error as? any DashboardErrorCodeProviding)?.dashboardErrorCode
            ?? "operation_failed"
    }
}

extension BundledRuntimeLayoutError: DashboardErrorCodeProviding {
    public var dashboardErrorCode: String {
        "bundled_runtime_invalid"
    }
}

extension BackendBinaryResolverError: DashboardErrorCodeProviding {
    public var dashboardErrorCode: String {
        "bundled_runtime_invalid"
    }
}

extension BackendProcessError: DashboardErrorCodeProviding {
    public var dashboardErrorCode: String {
        switch self {
        case .invalidLaunchConfiguration:
            "settings_invalid"
        case .externalHelperNotAllowed, .bundledMetallibArgumentRequired:
            "bundled_runtime_invalid"
        }
    }
}

extension BackendRuntimeSupervisorError: DashboardErrorCodeProviding {
    public var dashboardErrorCode: String {
        switch self {
        case .instanceAlreadyRunning:
            BackendRuntimeFailureCode.instanceAlreadyRunning.rawValue
        case .readinessFailed:
            BackendRuntimeFailureCode.backendReadinessFailed.rawValue
        case .launchFailed:
            "operation_failed"
        }
    }
}

extension ModelMtpRuntimeError: DashboardErrorCodeProviding {
    public var dashboardErrorCode: String {
        switch self {
        case .noCompatibleMtp:
            "mtp_incompatible"
        case .mtpPathNotFound:
            "mtp_model_not_found"
        }
    }
}

extension ModelDFlash2RuntimeError: DashboardErrorCodeProviding {
    public var dashboardErrorCode: String {
        switch self {
        case .targetPathNotFound:
            "dflash2_target_not_found"
        case .noCompatibleDraft, .draftPathNotFound:
            "dflash2_draft_not_found"
        case .incompatibleAccelerationConfiguration:
            "dflash2_acceleration_conflict"
        }
    }
}
