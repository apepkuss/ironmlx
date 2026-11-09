import Foundation

public extension Notification.Name {
    static let ironMLXMenuLanguageDidChange = Notification.Name("IronMLXMenuLanguageDidChange")
    static let ironMLXLoadedModelsDidChange = Notification.Name("IronMLXLoadedModelsDidChange")
    static let ironMLXBackendRuntimeDidChange = Notification.Name("IronMLXBackendRuntimeDidChange")
    /// `ModelLoadActivityCenter.activities` changed; the object is the center.
    static let ironMLXModelLoadsDidChange = Notification.Name("IronMLXModelLoadsDidChange")
}
