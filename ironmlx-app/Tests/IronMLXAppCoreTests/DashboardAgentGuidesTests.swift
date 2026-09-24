import Foundation
import Testing

@testable import IronMLXAppCore

@Test func dashboardAgentPageUsesListAndDetailProviderGuides() throws {
    let html = try String(
        contentsOfFile: "Sources/IronMLXAppCore/Resources/dashboard2.html",
        encoding: .utf8
    )

    #expect(html.contains(#"class="agent-guide-shell""#))
    #expect(html.contains(#"data-agent-guide="hermes""#))
    #expect(html.contains(#"data-agent-guide="omp""#))
    #expect(html.contains(#"data-agent-guide="dsh""#))
    let ompNavIndex = try #require(html.range(of: #"data-agent-guide="omp""#)?.lowerBound)
    let hermesNavIndex = try #require(html.range(of: #"data-agent-guide="hermes""#)?.lowerBound)
    let dshNavIndex = try #require(html.range(of: #"<button class="agent-guide-nav-item" type="button" data-agent-guide="dsh""#)?.lowerBound)
    #expect(ompNavIndex < hermesNavIndex)
    #expect(hermesNavIndex < dshNavIndex)
    #expect(html.contains(#"class="agent-guide-nav-item active" type="button" data-agent-guide="omp" aria-selected="true""#))
    #expect(html.contains(#"class="agent-guide-nav-item" type="button" data-agent-guide="hermes" aria-selected="false""#))
    #expect(html.contains(#"id="agent-guide-hermes" role="tabpanel" aria-label="Hermes Agent" hidden"#))
    #expect(html.contains(#"id="agent-guide-omp" role="tabpanel" aria-label="oh-my-pi">"#))
    #expect(html.contains(#"id="agent-guide-dsh" role="tabpanel" aria-label="DeepSeek Harness" hidden"#))
    #expect(html.contains("let currentAgentGuide = null"))
    #expect(html.contains("const firstAgentGuide = document.querySelector('[data-agent-guide]')"))
    #expect(html.contains("selectAgentGuide(firstAgentGuide.dataset.agentGuide)"))
    #expect(html.contains(#"data-agent-logo="hermes" src="hermes-agent-logo.svg""#))
    #expect(html.contains(#"data-agent-logo="omp" src="oh-my-pi-logo.svg""#))
    #expect(!html.contains(#"class="agent-guide-mark" aria-hidden="true">H</span>"#))
    #expect(!html.contains(#"aria-hidden="true">π</span>"#))
    #expect(html.contains(#"<svg data-nav-icon="agent""#))
    #expect(!html.contains(#"M3 3h12a1 1 0 011 1v8"#))
    #expect(html.contains(#"id="agent-guide-hermes" role="tabpanel""#))
    #expect(html.contains(#"id="agent-guide-omp" role="tabpanel""#))
    #expect(html.contains(#"id="agent-guide-dsh" role="tabpanel""#))
    #expect(html.contains(#"class="agent-guide-version-badge""#))
    #expect(html.contains("<strong>v0.20.0+</strong>"))
    #expect(html.contains("<strong>17.2.12+</strong>"))
    #expect(html.contains(#"agent_applicable_version: "适用版本""#))
    #expect(html.components(separatedBy: #"class="agent-code-copy""#).count - 1 == 6)
    #expect(html.components(separatedBy: #"data-copy-label="copy_configuration""#).count - 1 == 3)
    #expect(html.components(separatedBy: #"data-copy-label="copy_commands""#).count - 1 == 3)
    #expect(html.contains(#"class="agent-code-copy-icon""#))
    #expect(html.contains("function showAgentCopySuccess(button)"))
    #expect(html.contains("setAgentCopyButtonLabel(button, true)"))
    #expect(html.components(separatedBy: #"agent_copied: ""#).count - 1 == 5)
    #expect(html.components(separatedBy: #"agent_copy_failed: ""#).count - 1 == 5)
    #expect(html.contains("function selectAgentGuide(agent)"))
    #expect(html.contains("function selectHermesProfileMode(mode)"))
    #expect(html.contains("function renderAgentGuideConfiguration()"))
    #expect(html.contains("['hermes', 'omp', 'dsh'].forEach"))

    let resourcesDirectory = "Sources/IronMLXAppCore/Resources"
    #expect(FileManager.default.fileExists(atPath: "\(resourcesDirectory)/hermes-agent-logo.svg"))
    #expect(FileManager.default.fileExists(atPath: "\(resourcesDirectory)/oh-my-pi-logo.svg"))

    let bundleBuildScript = try String(contentsOfFile: "../scripts/build-app-bundle.sh", encoding: .utf8)
    let bundleVerifyScript = try String(contentsOfFile: "../scripts/verify-app-bundle.sh", encoding: .utf8)
    for logo in ["hermes-agent-logo.svg", "oh-my-pi-logo.svg"] {
        #expect(bundleBuildScript.contains(logo), "App assembly omits Agent logo: \(logo)")
        #expect(bundleVerifyScript.contains(logo), "Bundle verification omits Agent logo: \(logo)")
    }
}

@Test func dashboardAgentGuidesGenerateResponsesProviderConfiguration() throws {
    let html = try String(
        contentsOfFile: "Sources/IronMLXAppCore/Resources/dashboard2.html",
        encoding: .utf8
    )

    #expect(html.contains(#"'    transport: "codex_responses"'"#))
    #expect(html.contains(#"'    api: openai-responses'"#))
    #expect(html.contains(#"'      type: openai-models-list'"#))
    #expect(html.contains("'ironmlx/' + ompModel"))
    #expect(html.contains("apiFetch('/v1/models')"))
    #expect(html.contains("entry.context_window"))
    #expect(html.contains("entry.max_output_tokens"))
    #expect(html.contains("Math.min(4096, dshCapacity.maxOutputTokens"))
    #expect(html.contains("document.getElementById('copy-dsh-config').disabled = !dshValid"))
    #expect(html.contains("'    provider: ironmlx-local'"))
    #expect(html.contains("'        api: openai-responses'"))
    #expect(html.contains("'        cacheRetention: none'"))
    #expect(html.contains("'            maxTokens: ' + (dshOutput || 'OUTPUT_BUDGET')"))
    for level in ["off", "minimal", "low", "medium", "high", "xhigh", "max"] {
        #expect(html.contains(#"<option value="\#(level)" data-i18n="dsh_reasoning_\#(level)""#))
    }
    #expect(html.contains("'        reasoning: ' + dshReasoning"))
    #expect(html.contains("'              high: high'"))
    #expect(html.contains("'            reasoningEfforts: false'"))
    #expect(html.contains(".agent-guide-field {\n    min-width: 0;"))
    #expect(html.contains(#"id="dsh-official-guide" class="agent-guide-docs-link""#))
    #expect(html.contains("https://deepseek-harness.github.io/deepseek-harness/guide/providers#%E6%B7%BB%E5%8A%A0%E8%87%AA%E5%AE%9A%E4%B9%89%E6%8F%90%E4%BE%9B%E6%96%B9"))
    #expect(html.contains("https://deepseek-harness.github.io/deepseek-harness/en/guide/providers#add-a-custom-provider"))
    #expect(html.contains("if (dshGuide) dshGuide.href = lang === 'zh-Hans'"))
    #expect(html.contains("--patch ' + dshPatch"))
    #expect(html.contains("Math.max(64000"))
    #expect(html.contains("Hermes Agent v0.20.0 及以上版本需要至少 64K context tokens"))
    #expect(html.contains(#"data-hermes-profile-mode="dedicated""#))
    #expect(html.contains(#"data-hermes-profile-mode="default""#))
    #expect(html.contains("let hermesProfileMode = 'dedicated'"))
    #expect(html.contains("~/.hermes/profiles/ironmlx/config.yaml"))
    #expect(html.contains("~/.hermes/config.yaml"))
    #expect(html.contains("hermes profile create ironmlx"))
    #expect(html.contains("hermes --profile ironmlx"))
    #expect(html.contains("hermes --profile default"))
    #expect(html.contains("hermesCommand + ' --tui'"))
    #expect(html.contains("Desktop：选择 ironmlx profile，然后新建会话。"))
    #expect(!html.contains("agent_full_guide"))
    #expect(!html.contains("agent-guide-doc-link"))
    #expect(!html.contains(#"docs/hermes-agent.md"#))
    #expect(!html.contains(#"docs/oh-my-pi.md"#))
    #expect(!html.contains(#"docs/dsh.md"#))
}

@Test func dshAgentGuideDocumentsTheVerifiedOverlay() throws {
    let english = try String(contentsOfFile: "../docs/dsh.md", encoding: .utf8)
    let chinese = try String(contentsOfFile: "../docs/zh-CN/dsh.md", encoding: .utf8)
    let bridge = try String(contentsOfFile: "Sources/IronMLXAppCore/DashboardBridge.swift", encoding: .utf8)

    for guide in [english, chinese] {
        #expect(guide.contains("ddefc45fbc"))
        #expect(guide.contains("ironmlx.patch.yml"))
        #expect(guide.contains("--patch"))
        #expect(guide.contains("apiKeyEnv: IRONMLX_API_KEY"))
        #expect(guide.contains("contextWindow: 16384"))
        #expect(guide.contains("maxTokens: 4096"))
        #expect(guide.contains("tool_call"))
        #expect(guide.contains("tool_result"))
    }
    #expect(bridge.contains(#"case "/v1/models":"#))
    #expect(bridge.contains(#"client.fetchData(path: path)"#))
}

@Test func hermesAgentGuideRecommendsAnIsolatedProfile() throws {
    let guide = try String(contentsOfFile: "../docs/zh-CN/hermes-agent.md", encoding: .utf8)

    #expect(guide.contains("hermes profile create ironmlx"))
    #expect(guide.contains("~/.hermes/profiles/ironmlx/config.yaml"))
    #expect(guide.contains("hermes --profile ironmlx --tui"))
    #expect(guide.contains("Desktop：选择 `ironmlx` profile 后新建会话。"))
    #expect(guide.contains("~/.hermes/config.yaml"))
    #expect(guide.contains("若使用默认 profile，请将命令中的 `--profile ironmlx` 改为 `--profile default`。"))
}

@MainActor
@Test func legacyDashboardAgentIntegrationsAreRemoved() throws {
    let html = try String(
        contentsOfFile: "Sources/IronMLXAppCore/Resources/dashboard2.html",
        encoding: .utf8
    )

    for legacyName in ["OpenClaw", "IronHermes", "switchChatTab", "checkOpenClaw", "checkIronHermes"] {
        #expect(!html.contains(legacyName), "legacy Agent page reference remains: \(legacyName)")
    }

    for legacyHandler in ["openOpenClawChat", "openOpenClawDashboard", "checkOpenClaw", "checkIronHermes"] {
        #expect(!DashboardBridge.handlerNames.contains(legacyHandler))
    }
}
