import Foundation
import Testing

@testable import IronMLXAppCore

@MainActor
@Test func dashboardBridgeRegistersSchedulerProfileGenerationHandlers() {
    #expect(DashboardBridge.handlerNames.contains("previewSchedulerProfileGeneration"))
    #expect(DashboardBridge.handlerNames.contains("generateSchedulerProfile"))
    #expect(DashboardBridge.handlerNames.contains("cancelSchedulerProfileGeneration"))
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
    #expect(html.contains("profile-level-select"))
    #expect(html.contains("profile-cancel-btn"))
    #expect(html.contains("requestSchedulerProfilePreview"))
    #expect(html.contains("onSchedulerProfilePreview"))
    #expect(html.contains("onSchedulerProfileStatus"))
    #expect(html.contains("document.getElementById('profile-model-select')"))
}

@Test func schedulerProfileGenerationPreviewUsesResolvedMatrixSize() throws {
    let outputRoot = try temporaryDirectory()
    let qwenModel = outputRoot.appendingPathComponent("qwen", isDirectory: true)
    try FileManager.default.createDirectory(at: qwenModel, withIntermediateDirectories: true)
    try #"{"model_type":"qwen3_5"}"#.write(
        to: qwenModel.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )

    let standard = SchedulerProfileGenerationPreview(
        request: SchedulerProfileGenerationRequest(
            model: "mlx-community/Tiny-4bit",
            modelPath: "/tmp/Tiny-4bit/snapshot",
            selectionProfile: "balanced",
            calibrationLevel: "standard",
            pagedPrefixCacheDir: "/tmp/prefix"
        )
    )
    #expect(standard.success)
    #expect(standard.totalJobs == 48)
    #expect(standard.estimatedMinSeconds == 2_400)
    #expect(standard.estimatedMaxSeconds == 3_600)

    let quickWithoutPrefixCache = SchedulerProfileGenerationPreview(
        request: SchedulerProfileGenerationRequest(
            model: "mlx-community/Tiny-4bit",
            modelPath: "/tmp/Tiny-4bit/snapshot",
            calibrationLevel: "quick"
        )
    )
    #expect(quickWithoutPrefixCache.totalJobs == 8)
    #expect(quickWithoutPrefixCache.estimatedMinSeconds == 240)
    #expect(quickWithoutPrefixCache.estimatedMaxSeconds == 480)

    let qwenFull = SchedulerProfileGenerationPreview(
        request: SchedulerProfileGenerationRequest(
            model: "mlx-community/Qwen",
            modelPath: qwenModel.path,
            calibrationLevel: "full",
            mtpModelDir: "/tmp/mtp",
            pagedPrefixCacheDir: "/tmp/prefix"
        )
    )
    #expect(qwenFull.totalJobs == 64)
    #expect(qwenFull.estimatedMinSeconds == 4_800)
    #expect(qwenFull.estimatedMaxSeconds == 6_000)
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
    #expect(html.contains(#"data-i18n="scheduler_profile_help_levels""#))
    #expect(html.contains("快速验证最多运行 16 jobs"))
    #expect(html.contains("标准校准最多运行 48 jobs"))
    #expect(html.contains("完整校准最多运行 96 jobs"))
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
    let expectedPrefix = [
        "scheduler-autotune",
        "calibrate",
        "--model", "/tmp/Tiny-4bit/snapshot",
        "--model-name", "mlx-community/Tiny-4bit",
        "--iron-bench-bin", "/tmp/iron-bench",
        "--output-dir", outputRoot.appendingPathComponent("Tiny-4bit-20240531-160837").path,
        "--write-profile", outputRoot.appendingPathComponent("Tiny-4bit-20240531-160837/scheduler-profile.json").path,
        "--selection-profile", "balanced",
        "--kv-quant", "none",
        "--max-cache-cap", "32768",
    ]
    #expect(Array(plan.arguments.prefix(expectedPrefix.count)) == expectedPrefix)
    #expect(containsFlagValue(plan.arguments, flag: "--concurrency", value: "1,4"))
    #expect(containsFlagValue(plan.arguments, flag: "--runs", value: "3"))
    #expect(plan.arguments.filter { $0 == "--candidate" }.count == 12)
    #expect(plan.logURL.path == outputRoot.appendingPathComponent("Tiny-4bit-20240531-160837/calibrate.log").path)
}

@Test func schedulerProfileGenerationPlansExposeQuickStandardAndFullMatrices() throws {
    let outputRoot = try temporaryDirectory()
    let qwenModel = outputRoot.appendingPathComponent("qwen", isDirectory: true)
    let gemmaModel = outputRoot.appendingPathComponent("gemma", isDirectory: true)
    try FileManager.default.createDirectory(at: qwenModel, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: gemmaModel, withIntermediateDirectories: true)
    try #"{"model_type":"qwen3_5"}"#.write(
        to: qwenModel.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )
    try #"{"model_type":"gemma4"}"#.write(
        to: gemmaModel.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )

    func plan(level: String) -> SchedulerProfileGenerationPlan {
        SchedulerProfileGenerationPlan(
            executableURL: URL(fileURLWithPath: "/tmp/ironmlx"),
            ironBenchURL: nil,
            request: SchedulerProfileGenerationRequest(
                model: "mlx-community/Tiny-4bit",
                modelPath: "/tmp/Tiny-4bit/snapshot",
                calibrationLevel: level,
                pagedPrefixCacheDir: "/tmp/prefix"
            ),
            outputRootURL: outputRoot,
            timestamp: Date(timeIntervalSince1970: 1_717_171_717)
        )
    }

    let quick = plan(level: "quick")
    #expect(quick.arguments.filter { $0 == "--candidate" }.count == 4)
    #expect(containsFlagValue(quick.arguments, flag: "--concurrency", value: "1,2"))
    #expect(containsFlagValue(quick.arguments, flag: "--duration", value: "10"))

    let standard = plan(level: "standard")
    #expect(standard.arguments.filter { $0 == "--candidate" }.count == 12)
    #expect(containsFlagValue(standard.arguments, flag: "--concurrency", value: "1,4"))
    #expect(containsFlagValue(standard.arguments, flag: "--duration", value: "15"))

    let full = plan(level: "full")
    #expect(full.arguments.filter { $0 == "--candidate" }.isEmpty)
    #expect(containsFlagValue(full.arguments, flag: "--concurrency", value: "1,2,4,8"))
    #expect(!full.arguments.contains("--duration"))

    func totalJobs(level: String, modelPath: String, mtpModelDir: String? = nil) -> Int? {
        SchedulerProfileGenerationStatus.running(
            request: SchedulerProfileGenerationRequest(
                model: "mlx-community/Tiny-4bit",
                modelPath: modelPath,
                calibrationLevel: level,
                mtpModelDir: mtpModelDir,
                pagedPrefixCacheDir: "/tmp/prefix"
            ),
            outputDirectory: "/tmp/output"
        ).totalJobs
    }
    #expect(totalJobs(level: "quick", modelPath: gemmaModel.path) == 16)
    #expect(totalJobs(level: "standard", modelPath: gemmaModel.path) == 48)
    #expect(totalJobs(level: "full", modelPath: gemmaModel.path) == 96)
    #expect(totalJobs(level: "standard", modelPath: qwenModel.path, mtpModelDir: "/tmp/mtp") == 32)
    #expect(totalJobs(level: "full", modelPath: qwenModel.path, mtpModelDir: "/tmp/mtp") == 64)
    #expect(totalJobs(level: "standard", modelPath: gemmaModel.path, mtpModelDir: "/tmp/mtp") == 48)
    #expect(totalJobs(level: "full", modelPath: gemmaModel.path, mtpModelDir: "/tmp/mtp") == 96)
}

@Test func schedulerProfileGenerationPlanIncludesRuntimeContext() throws {
    let outputRoot = try temporaryDirectory()
    let plan = SchedulerProfileGenerationPlan(
        executableURL: URL(fileURLWithPath: "/tmp/ironmlx"),
        ironBenchURL: nil,
        request: SchedulerProfileGenerationRequest(
            model: "mlx-community/Tiny-4bit",
            modelPath: "/tmp/Tiny-4bit/snapshot",
            mtpModelDir: "/tmp/Tiny-4bit/mtp",
            mtpDraftTokens: 3,
            kvQuant: "k3v4",
            pagedPrefixCacheDir: "/tmp/prefix",
            prefixLruCacheMaxBytes: 1_048_576,
            ssdPrefixCacheMaxGB: 8,
            activeKvOffload: true,
            memoryLimitTotalGB: 96,
            memoryLimitModelGB: 64,
            maxCacheCap: 65_536
        ),
        outputRootURL: outputRoot,
        timestamp: Date(timeIntervalSince1970: 1_717_171_717)
    )

    #expect(containsFlagValue(plan.arguments, flag: "--mtp-model-dir", value: "/tmp/Tiny-4bit/mtp"))
    #expect(containsFlagValue(plan.arguments, flag: "--mtp-draft-tokens", value: "3"))
    #expect(containsFlagValue(plan.arguments, flag: "--kv-quant", value: "k3v4"))
    #expect(containsFlagValue(plan.arguments, flag: "--paged-prefix-cache-dir", value: "/tmp/prefix"))
    #expect(containsFlagValue(plan.arguments, flag: "--prefix-lru-cache-max-bytes", value: "1048576"))
    #expect(containsFlagValue(plan.arguments, flag: "--ssd-prefix-cache-max-gb", value: "8"))
    #expect(plan.arguments.contains("--active-kv-offload"))
    #expect(containsFlagValue(plan.arguments, flag: "--memory-limit-total-gb", value: "96"))
    #expect(containsFlagValue(plan.arguments, flag: "--memory-limit-model-gb", value: "64"))
    #expect(containsFlagValue(plan.arguments, flag: "--max-cache-cap", value: "65536"))
}

private func containsFlagValue(_ arguments: [String], flag: String, value: String) -> Bool {
    for index in arguments.indices.dropLast() where arguments[index] == flag {
        if arguments[arguments.index(after: index)] == value {
            return true
        }
    }
    return false
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
        stored_runtime_profile: /Users/xin/.ironmlx/scheduler-profiles/profiles/tiny--test-host--agent-long-prompt--model-hash--context-hash.json
        """,
        outputDirectory: "/tmp/out"
    )

    #expect(status.state == "succeeded")
    #expect(status.runtimeProfile == "/tmp/out/scheduler-profile.json")
    #expect(status.storedRuntimeProfile == "/Users/xin/.ironmlx/scheduler-profiles/profiles/tiny--test-host--agent-long-prompt--model-hash--context-hash.json")
    #expect(status.outputDirectory == "/tmp/out")
}

@Test func schedulerProfileGenerationStatusReadsManifestProgressAndETA() throws {
    let outputDirectory = try temporaryDirectory()
    let manifest = """
    {
      "jobs": [
        {"candidate_idx":0,"concurrency":1,"cache_state":"cold","output_json":"candidate-000.json"},
        {"candidate_idx":1,"concurrency":4,"cache_state":"warm","output_json":"candidate-001.json"}
      ]
    }
    """
    try manifest.write(
        to: outputDirectory.appendingPathComponent("run-order.json"),
        atomically: true,
        encoding: .utf8
    )
    try "{}".write(
        to: outputDirectory.appendingPathComponent("candidate-000.json"),
        atomically: true,
        encoding: .utf8
    )
    let startedAt = Date(timeIntervalSince1970: 1_000)
    var status = SchedulerProfileGenerationStatus.running(
        request: SchedulerProfileGenerationRequest(
            model: "mlx-community/Tiny-4bit",
            modelPath: "/tmp/model"
        ),
        outputDirectory: outputDirectory.path,
        startedAt: startedAt
    )

    status.refreshProgress(
        logTail: "[scheduler-autotune] job 1/2 stage=completed elapsed_s=12.5",
        now: Date(timeIntervalSince1970: 1_030)
    )

    #expect(status.completedJobs == 1)
    #expect(status.totalJobs == 2)
    #expect(status.elapsedSeconds == 30)
    #expect(status.estimatedRemainingSeconds == 12.5)
    #expect(status.currentJob == "candidate 1 · concurrency 4 · warm")
    #expect(status.currentStage?.contains("stage=completed") == true)
}

@Test func schedulerProfileGenerationCurrentStatusIncludesLiveLogTailWhileRunning() async throws {
    let root = try temporaryDirectory()
    let outputRoot = root.appendingPathComponent("reports", isDirectory: true)
    let executable = root.appendingPathComponent("fake-ironmlx")
    let readyURL = root.appendingPathComponent("calibration-ready")
    let releaseURL = root.appendingPathComponent("calibration-release")
    try """
    #!/bin/sh
    echo "calibration started"
    touch "\(readyURL.path)"
    while [ ! -f "\(releaseURL.path)" ]; do sleep 0.02; done
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
    defer {
        try? Data().write(to: releaseURL)
    }

    for _ in 0..<250 {
        if FileManager.default.fileExists(atPath: readyURL.path) {
            break
        }
        try await Task.sleep(nanoseconds: 20_000_000)
    }
    try #require(FileManager.default.fileExists(atPath: readyURL.path))

    let observed = service.currentStatus()
    #expect(observed.state == "running")
    #expect(observed.logTail?.contains("calibration started") == true)

    try Data().write(to: releaseURL)

    for _ in 0..<250 {
        let status = service.currentStatus()
        if status.state != "running" {
            #expect(status.state == "succeeded")
            return
        }
        try await Task.sleep(nanoseconds: 20_000_000)
    }
    Issue.record("calibration did not finish after the test released it")
}

@Test func schedulerProfileGenerationCanBeCancelledCooperatively() async throws {
    let root = try temporaryDirectory()
    let outputRoot = root.appendingPathComponent("reports", isDirectory: true)
    let executable = root.appendingPathComponent("fake-ironmlx")
    try """
    #!/bin/sh
    trap 'echo calibration_cancelled: test; exit 1' INT TERM
    echo "calibration started"
    while true; do sleep 1; done
    """.write(to: executable, atomically: true, encoding: .utf8)
    try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: executable.path)

    let service = SchedulerProfileGenerationService(outputRootURL: outputRoot)
    let request = SchedulerProfileGenerationRequest(
        model: "mlx-community/Tiny-4bit",
        modelPath: root.path,
        calibrationLevel: "quick"
    )
    _ = service.start(executableURL: executable, ironBenchURL: nil, request: request)

    for _ in 0..<50 where service.currentStatus().state != "running" {
        try await Task.sleep(nanoseconds: 20_000_000)
    }
    let cancelling = service.cancel()
    #expect(cancelling.state == "cancelling")

    for _ in 0..<100 {
        let status = service.currentStatus()
        if status.state == "cancelled" {
            #expect(status.success == false)
            return
        }
        try await Task.sleep(nanoseconds: 20_000_000)
    }
    Issue.record("calibration did not reach cancelled state")
}
