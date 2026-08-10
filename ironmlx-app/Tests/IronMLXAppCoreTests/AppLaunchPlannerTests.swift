import Foundation
import Testing

@testable import IronMLXAppCore

@Test func doesNotRestoreUnloadedDefaultModelAsLoadedModel() throws {
    let json = """
    {
      "host": "127.0.0.1",
      "port": 9068,
      "default_model": "  mlx-community/Qwen3-0.6B-4bit  ",
      "language": "en"
    }
    """
    let config = try JSONDecoder().decode(AppConfig.self, from: Data(json.utf8))

    let plan = AppLaunchPlanner().plan(config: config, localModels: [])

    #expect(plan.backendModelReferences == [])
    #expect(plan.dashboardRoute == .onboarding)
}

@Test func restoresPersistedLoadedModelsWithDefaultFirst() {
    let config = AppConfig(
        defaultModel: "mlx-community/Default-4bit",
        loadedModels: [
            "mlx-community/Other-4bit",
            "mlx-community/Default-4bit",
        ]
    )

    let plan = AppLaunchPlanner().plan(config: config, localModels: [])

    #expect(plan.backendModelReferences == [
        "mlx-community/Default-4bit",
        "mlx-community/Other-4bit",
    ])
    #expect(plan.dashboardRoute == .status)
}

@Test func showsOnboardingWhenNoDefaultModelAndNoDownloadedModelsExist() {
    let plan = AppLaunchPlanner().plan(config: AppConfig(), localModels: [])

    #expect(plan.backendModelReferences == [])
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

    #expect(plan.backendModelReferences == [])
    #expect(plan.dashboardRoute == .modelsManage)
}
