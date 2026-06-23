import Foundation
import Testing

@testable import IronMLXAppCore

@Test func restoresLastModelRegardlessOfPersistedAutoStartFlag() throws {
    let json = """
    {
      "host": "127.0.0.1",
      "port": 9068,
      "auto_start": false,
      "last_model": "  mlx-community/Qwen3-0.6B-4bit  ",
      "language": "en"
    }
    """
    let config = try JSONDecoder().decode(AppConfig.self, from: Data(json.utf8))

    let plan = AppLaunchPlanner().plan(config: config, localModels: [])

    #expect(plan.backendModelReference == "mlx-community/Qwen3-0.6B-4bit")
    #expect(plan.dashboardRoute == .status)
}

@Test func showsOnboardingWhenNoDefaultModelAndNoDownloadedModelsExist() {
    let plan = AppLaunchPlanner().plan(config: AppConfig(), localModels: [])

    #expect(plan.backendModelReference == nil)
    #expect(plan.dashboardRoute == .onboarding)
}

@Test func opensModelManagementWhenDownloadedModelsExistButNoDefaultModelIsSet() {
    let localModel = LocalModel(
        id: "mlx-community/Qwen3-0.6B-4bit",
        repoID: "mlx-community/Qwen3-0.6B-4bit",
        source: "hf",
        sizeMB: 512
    )

    let plan = AppLaunchPlanner().plan(config: AppConfig(), localModels: [localModel])

    #expect(plan.backendModelReference == nil)
    #expect(plan.dashboardRoute == .modelsManage)
}
