import Foundation

public enum DashboardInitialRoute: String, Equatable {
    case status
    case onboarding
    case modelsManage
    case modelsDownload
}

public struct AppLaunchPlan: Equatable {
    public var backendModelReference: String?
    public var dashboardRoute: DashboardInitialRoute

    public init(
        backendModelReference: String?,
        dashboardRoute: DashboardInitialRoute
    ) {
        self.backendModelReference = backendModelReference
        self.dashboardRoute = dashboardRoute
    }
}

public struct AppLaunchPlanner {
    public init() {}

    public func plan(config: AppConfig, localModels: [LocalModel]) -> AppLaunchPlan {
        if let model = config.lastModel?.trimmingCharacters(in: .whitespacesAndNewlines),
           !model.isEmpty {
            return AppLaunchPlan(
                backendModelReference: model,
                dashboardRoute: .status
            )
        }

        if localModels.isEmpty {
            return AppLaunchPlan(
                backendModelReference: nil,
                dashboardRoute: .onboarding
            )
        }

        return AppLaunchPlan(
            backendModelReference: nil,
            dashboardRoute: .modelsManage
        )
    }
}
