import Foundation

public enum DashboardInitialRoute: String, Equatable {
    case status
    case onboarding
    case modelsManage
    case modelsDownload
}

public struct AppLaunchPlan: Equatable {
    public var backendModelReferences: [String]
    public var dashboardRoute: DashboardInitialRoute

    public init(
        backendModelReferences: [String],
        dashboardRoute: DashboardInitialRoute
    ) {
        self.backendModelReferences = backendModelReferences
        self.dashboardRoute = dashboardRoute
    }
}

public struct AppLaunchPlanner {
    public init() {}

    public func plan(config: AppConfig, localModels: [LocalModel]) -> AppLaunchPlan {
        let models = config.restoredModelReferences
        if !models.isEmpty {
            return AppLaunchPlan(
                backendModelReferences: models,
                dashboardRoute: .status
            )
        }

        if localModels.isEmpty {
            return AppLaunchPlan(
                backendModelReferences: [],
                dashboardRoute: .onboarding
            )
        }

        return AppLaunchPlan(
            backendModelReferences: [],
            dashboardRoute: .modelsManage
        )
    }
}
