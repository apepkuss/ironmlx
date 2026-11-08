import AppKit
import Testing

@testable import IronMLXAppCore

@MainActor
@Suite("App window activation")
struct AppWindowActivationTests {
    @Test func anyVisibleWindowKeepsTheAppRegular() {
        var applied: [NSApplication.ActivationPolicy] = []
        let activation = AppWindowActivation { applied.append($0) }

        activation.shown(.chat)
        #expect(activation.policy == .regular)
        activation.shown(.dashboard)
        // Closing the Dashboard opened from chat keeps chat in the Dock and Cmd-Tab.
        activation.hidden(.dashboard)
        #expect(activation.policy == .regular)
        #expect(applied.last == .regular)
        activation.shown(.noModel)
        activation.hidden(.chat)
        #expect(activation.policy == .regular)
        activation.hidden(.noModel)
        #expect(activation.policy == .accessory)
        #expect(applied.last == .accessory)
    }

    @Test func repeatedEventsChangeNothing() {
        var applied: [NSApplication.ActivationPolicy] = []
        let activation = AppWindowActivation { applied.append($0) }
        activation.hidden(.chat)
        #expect(applied.isEmpty)
        activation.shown(.chat)
        activation.shown(.chat)
        #expect(applied == [.regular])
        activation.hidden(.chat)
        activation.hidden(.chat)
        #expect(applied == [.regular, .accessory])
    }
}
