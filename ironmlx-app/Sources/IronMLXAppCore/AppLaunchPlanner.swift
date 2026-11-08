import Foundation

public enum DashboardInitialRoute: String, Equatable {
    case status
    case onboarding
    case modelsManage
    case modelsDownload
    /// The Agent page (external agents), opened from the chat history panel.
    case agents
}

/// The first window the App shows after the backend starts.
public enum LaunchSurface: Equatable {
    case dashboard(DashboardInitialRoute)
    /// The native window that offers the recommended starter model.
    case noModelWindow
}

public struct AppLaunchPlan: Equatable {
    public var backendModelReferences: [String]
    /// The Dashboard route, also used when the backend fails to start.
    public var dashboardRoute: DashboardInitialRoute
    public var firstSurface: LaunchSurface

    public init(
        backendModelReferences: [String],
        dashboardRoute: DashboardInitialRoute,
        firstSurface: LaunchSurface? = nil
    ) {
        self.backendModelReferences = backendModelReferences
        self.dashboardRoute = dashboardRoute
        self.firstSurface = firstSurface ?? .dashboard(dashboardRoute)
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
                dashboardRoute: .onboarding,
                firstSurface: .noModelWindow
            )
        }

        return AppLaunchPlan(
            backendModelReferences: [],
            dashboardRoute: .modelsManage
        )
    }
}
