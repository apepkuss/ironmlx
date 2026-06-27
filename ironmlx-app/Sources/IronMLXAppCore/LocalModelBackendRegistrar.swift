import Foundation

public enum LocalModelBackendRegistrar {
    @discardableResult
    public static func register(
        localModels: [LocalModel],
        defaultModel: String?,
        scanner: LocalModelScanner,
        parameterStore: ModelParameterStore,
        client: any BackendModelLoading
    ) async -> [String] {
        let defaultModel = AppConfig.normalizedModelReference(defaultModel)
        var failed: [String] = []
        for model in localModels {
            guard let modelDir = scanner.resolveModelPath(for: model.id) else {
                failed.append(model.id)
                IronMLXAppLogger.error("Failed to register local model \(model.id): model path not found")
                continue
            }
            do {
                _ = try await client.registerModel(
                    model: model.id,
                    modelDir: modelDir,
                    setDefault: model.id == defaultModel,
                    maxCacheCap: ModelLoadParameters.maxCacheCap(
                        for: model.id,
                        scanner: scanner,
                        parameterStore: parameterStore
                    ),
                    samplingDefaults: parameterStore.parameters(for: model.id)?.samplingDefaults ?? .empty
                )
            } catch {
                failed.append(model.id)
                IronMLXAppLogger.error("Failed to register local model \(model.id): \(error)")
            }
        }
        return failed
    }
}
