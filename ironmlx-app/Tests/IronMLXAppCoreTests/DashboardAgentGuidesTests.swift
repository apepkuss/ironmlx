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
    #expect(html.contains(#"data-agent-logo="dsh" src="deepseek-harness-logo.svg""#))
    #expect(html.contains(#"srcset="deepseek-harness-logo-dark.svg""#))
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
    #expect(html.contains("function selectDshSetupMode(mode)"))
    #expect(html.contains("function renderAgentGuideConfiguration()"))
    #expect(html.contains("['hermes', 'omp', 'dsh'].forEach"))
    #expect(html.contains(#"data-dsh-setup-mode="desktop" checked"#))
    #expect(html.contains(#"data-dsh-setup-mode="cli" onchange"#))
    #expect(html.contains(#"data-dsh-setup-panel="cli" hidden"#))
    #expect(html.contains(#"data-dsh-setup-panel="desktop">"#))
    #expect(html.contains("let dshSetupMode = 'desktop';"))
    let desktopSetupOption = try #require(html.range(of: #"data-dsh-setup-mode="desktop""#))
    let cliSetupOption = try #require(html.range(of: #"data-dsh-setup-mode="cli""#))
    #expect(desktopSetupOption.lowerBound < cliSetupOption.lowerBound)

    let resourcesDirectory = "Sources/IronMLXAppCore/Resources"
    let bundleBuildScript = try String(contentsOfFile: "../scripts/build-app-bundle.sh", encoding: .utf8)
    let bundleVerifyScript = try String(contentsOfFile: "../scripts/verify-app-bundle.sh", encoding: .utf8)
    for logo in [
        "deepseek-harness-logo-dark.svg",
        "deepseek-harness-logo.svg",
        "hermes-agent-logo.svg",
        "oh-my-pi-logo.svg",
    ] {
        #expect(FileManager.default.fileExists(atPath: "\(resourcesDirectory)/\(logo)"))
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
    #expect(html.contains(#"id="dsh-desktop-full-guide" class="agent-guide-docs-link""#))
    #expect(html.contains("https://github.com/apepkuss/ironmlx/blob/dev/docs/dsh.md#dsh-desktop-gui-setup"))
    #expect(html.contains("https://github.com/apepkuss/ironmlx/blob/dev/docs/zh-CN/dsh.md#dsh-desktop-gui-%E9%85%8D%E7%BD%AE"))
    #expect(html.contains("if (dshDesktopGuide) dshDesktopGuide.href = lang === 'zh-Hans' || lang === 'zh-Hant'"))
    #expect(html.contains(#"id="dsh-desktop-endpoint""#))
    #expect(html.contains("dshDesktopEndpoint.textContent = endpoint"))
    #expect(html.contains(#"data-i18n="dsh_desktop_step_7""#))
    #expect(html.contains("IronMLX advertises 4096 when the model has no independent limit"))
    #expect(html.contains("模型没有独立上限时，IronMLX 会提供 4096"))
    #expect(html.contains("ironmlx-local</dd>"))
    #expect(html.contains("openai-responses</dd>"))
    #expect(html.contains("local</dd>"))
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
    #expect(!html.contains(#"href="docs/dsh.md""#))
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
        #expect(guide.contains("DSH Desktop"))
        #expect(guide.contains("ironmlx-local"))
        #expect(guide.contains("openai-responses"))
        #expect(guide.contains("http://127.0.0.1:9068/v1"))
        #expect(guide.contains("local"))
        #expect(guide.contains("TTS"))
        #expect(guide.contains("default_max_output_tokens"))
        #expect(guide.contains("8192"))
        #expect(guide.contains("16384"))
    }
    #expect(bridge.contains(#"case "/v1/models":"#))
    #expect(bridge.contains(#"client.fetchData(path: path)"#))
    #expect(bridge.contains(#"let isIronMLXDSHGuide = url.host == "github.com""#))
    #expect(bridge.contains(#"/apepkuss/ironmlx/blob/dev/docs/dsh.md"#))
    #expect(bridge.contains(#"/apepkuss/ironmlx/blob/dev/docs/zh-CN/dsh.md"#))
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
