import Foundation
import Testing

@testable import IronMLXAppCore

@Test func backendRuntimeFailureCodeDetectsOnlyStableInstanceMarker() {
    #expect(
        BackendRuntimeFailureCode.detect(
            in: "Error: ironmlx_instance_already_running: another backend owns the lock"
        ) == .instanceAlreadyRunning
    )
    #expect(BackendRuntimeFailureCode.detect(in: "address already in use") == nil)
}

@Test func backendInstanceConflictTextIsLocalized() {
    #expect(
        BackendInstanceConflictPresentation.text(language: "zh-Hans").title
            == "已有 IronMLX 实例正在运行"
    )
    #expect(
        BackendInstanceConflictPresentation.text(language: "en").message
            .contains("Only one IronMLX backend")
    )
}
