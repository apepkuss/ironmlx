import Foundation

/// Ids of the entries on the Dashboard Agent page (`data-agent-guide`).
public enum AgentGuideID: String, CaseIterable, Sendable {
    case dsh
    case opencode
    case hermes
    case omp
    case dshDesktop = "dsh-desktop"
}

/// Picks the agent the Agent page opens on when it is reached from chat.
enum AgentPreselector {
    /// Preference order; the first installed agent wins.
    static let order: [AgentGuideID] = [.dsh, .opencode, .hermes, .omp, .dshDesktop]

    /// An installed agent when there is one, otherwise DeepSeek Harness.
    static func choose(installed: Set<AgentGuideID>) -> AgentGuideID {
        order.first(where: installed.contains) ?? .dsh
    }
}

protocol AgentInstallationChecking {
    func installed() async -> Set<AgentGuideID>
}

/// Reuses each integration service's `inspect()`, which only reads files and
/// runs `--version`; nothing is configured or written. DeepSeek Harness counts
/// only for the official Desktop app or official CLI.
@MainActor
struct AgentInstallationChecker: AgentInstallationChecking {
    /// The five checks run concurrently so the slowest one bounds the delay.
    func installed() async -> Set<AgentGuideID> {
        async let harness = Self.harnessInstalled()
        async let opencode = OpenCodeIntegrationService().inspect().code
        async let hermes = HermesIntegrationService().inspect().code
        async let omp = OMPIntegrationService().inspect().result.code
        async let desktop = DSHDesktopIntegrationService().inspect().code
        var found: Set<AgentGuideID> = []
        if await harness { found.insert(.dsh) }
        if Self.isInstalled(await opencode) { found.insert(.opencode) }
        if Self.isInstalled(await hermes) { found.insert(.hermes) }
        if Self.isInstalled(await omp) { found.insert(.omp) }
        if Self.isInstalled(await desktop) { found.insert(.dshDesktop) }
        return found
    }

    private static func harnessInstalled() async -> Bool {
        let service = DeepSeekHarnessIntegrationService()
        async let desktop = service.inspect(mode: .desktop).code
        async let cli = service.inspect(mode: .cli).code
        let (desktopCode, cliCode) = (await desktop, await cli)
        return isInstalled(desktopCode) || isInstalled(cliCode)
    }

    /// `cli_unrecognized` is a `dsh` command that is not the official CLI.
    nonisolated static func isInstalled(_ code: String) -> Bool {
        code != "not_installed" && code != "cli_unrecognized"
    }
}
