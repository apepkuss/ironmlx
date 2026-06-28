import Foundation
import Testing

@testable import IronMLXAppCore

@MainActor
@Test func dashboardBridgeRegistersSchedulerProfileGenerationHandlers() {
    #expect(DashboardBridge.handlerNames.contains("generateSchedulerProfile"))
    #expect(DashboardBridge.handlerNames.contains("refreshSchedulerProfileStatus"))
}

@Test func dashboardIncludesSchedulerProfileGenerationControls() throws {
    let html = try String(
        contentsOfFile: "Sources/IronMLXAppCore/Resources/dashboard2.html",
        encoding: .utf8
    )
    let modelsManageTab = try #require(html.range(of: #"id="tab-models-manage""#))
    let profileCard = try #require(html.range(of: #"id="profile-generation-card""#))
    let benchmarkPage = try #require(html.range(of: #"id="page-benchmark""#))

    #expect(modelsManageTab.lowerBound < profileCard.lowerBound)
    #expect(profileCard.lowerBound < benchmarkPage.lowerBound)
    #expect(html.contains(#"id="profile-model-select""#))
    #expect(html.contains("profile-generate-btn"))
    #expect(html.contains("onSchedulerProfileStatus"))
    #expect(html.contains("document.getElementById('profile-model-select')"))
}

@Test func dashboardIncludesSchedulerProfileHelpTooltip() throws {
    let html = try String(
        contentsOfFile: "Sources/IronMLXAppCore/Resources/dashboard2.html",
        encoding: .utf8
    )

    #expect(html.contains(#"id="profile-help-trigger""#))
    #expect(html.contains(#"aria-describedby="profile-help-tooltip""#))
    #expect(html.contains(#"id="profile-help-tooltip""#))
    #expect(html.contains(#"data-i18n="scheduler_profile_help_title""#))
    #expect(html.contains(#"data-i18n="scheduler_profile_help_body""#))
    #expect(html.contains(#"data-i18n="scheduler_profile_help_flow""#))
}

@Test func dashboardIncludesSchedulerProfileProgressAndResultSummary() throws {
    let html = try String(
        contentsOfFile: "Sources/IronMLXAppCore/Resources/dashboard2.html",
        encoding: .utf8
    )

    #expect(html.contains(#"id="profile-progress""#))
    #expect(html.contains(#"class="profile-progress-bar""#))
    #expect(html.contains(#"id="profile-result""#))
    #expect(html.contains(#"id="profile-result-stored""#))
    #expect(html.contains(#"id="profile-result-runtime-details""#))
    #expect(html.contains(#"data-i18n="profile_running_hint""#))
    #expect(html.contains(#"data-i18n="profile_store_path""#))
    #expect(html.contains(#"data-i18n="profile_report_path""#))
    #expect(html.contains("profileResultEl"))
    #expect(html.contains("stored_runtime_profile"))
    #expect(html.contains("runtime_profile"))
}

@Test func schedulerProfileGenerationPlanBuildsCalibrateCommand() throws {
    let outputRoot = try temporaryDirectory()
    let plan = SchedulerProfileGenerationPlan(
        executableURL: URL(fileURLWithPath: "/tmp/ironmlx"),
        ironBenchURL: URL(fileURLWithPath: "/tmp/iron-bench"),
        request: SchedulerProfileGenerationRequest(
            model: "mlx-community/Tiny-4bit",
            modelPath: "/tmp/Tiny-4bit/snapshot",
            selectionProfile: "balanced"
        ),
        outputRootURL: outputRoot,
        timestamp: Date(timeIntervalSince1970: 1_717_171_717)
    )

    #expect(plan.processURL.path == "/tmp/ironmlx")
    #expect(plan.arguments == [
        "scheduler-autotune",
        "calibrate",
        "--model", "/tmp/Tiny-4bit/snapshot",
        "--model-name", "mlx-community/Tiny-4bit",
        "--iron-bench-bin", "/tmp/iron-bench",
        "--output-dir", outputRoot.appendingPathComponent("Tiny-4bit-20240531-160837").path,
        "--write-profile", outputRoot.appendingPathComponent("Tiny-4bit-20240531-160837/scheduler-profile.json").path,
        "--selection-profile", "balanced",
    ])
    #expect(plan.logURL.path == outputRoot.appendingPathComponent("Tiny-4bit-20240531-160837/calibrate.log").path)
}

@Test func schedulerProfileGenerationStatusParsesCalibrateOutputPaths() {
    let status = SchedulerProfileGenerationStatus.completed(
        request: SchedulerProfileGenerationRequest(
            model: "mlx-community/Tiny-4bit",
            modelPath: "/tmp/Tiny-4bit/snapshot",
            selectionProfile: "agent-long-prompt"
        ),
        exitCode: 0,
        logTail: """
        calibration: /tmp/out/calibration.json
        runtime_profile: /tmp/out/scheduler-profile.json
        stored_runtime_profile: /Users/xin/.ironmlx/scheduler-profiles/profiles/tiny.json
        """,
        outputDirectory: "/tmp/out"
    )

    #expect(status.state == "succeeded")
    #expect(status.runtimeProfile == "/tmp/out/scheduler-profile.json")
    #expect(status.storedRuntimeProfile == "/Users/xin/.ironmlx/scheduler-profiles/profiles/tiny.json")
    #expect(status.outputDirectory == "/tmp/out")
}

@Test func schedulerProfileGenerationCurrentStatusIncludesLiveLogTailWhileRunning() async throws {
    let root = try temporaryDirectory()
    let outputRoot = root.appendingPathComponent("reports", isDirectory: true)
    let executable = root.appendingPathComponent("fake-ironmlx")
    try """
    #!/bin/sh
    echo "calibration started"
    sleep 1
    echo "runtime_profile: /tmp/runtime-profile.json"
    echo "stored_runtime_profile: /tmp/stored-profile.json"
    """.write(to: executable, atomically: true, encoding: .utf8)
    try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: executable.path)

    let service = SchedulerProfileGenerationService(outputRootURL: outputRoot)
    let request = SchedulerProfileGenerationRequest(
        model: "mlx-community/Tiny-4bit",
        modelPath: root.path,
        selectionProfile: "balanced"
    )
    _ = service.start(executableURL: executable, ironBenchURL: nil, request: request)

    var observed: SchedulerProfileGenerationStatus?
    for _ in 0..<50 {
        let status = service.currentStatus()
        if status.state == "running",
           status.logTail?.contains("calibration started") == true {
            observed = status
            break
        }
        try await Task.sleep(nanoseconds: 20_000_000)
    }

    #expect(observed?.state == "running")
    #expect(observed?.logTail?.contains("calibration started") == true)

    for _ in 0..<80 {
        if service.currentStatus().state != "running" {
            return
        }
        try await Task.sleep(nanoseconds: 20_000_000)
    }
}
