import AppKit
import Testing

@testable import IronMLXAppCore

@MainActor
@Test func applicationMenuProvidesStandardAboutPanelAndQuitActions() {
    let menu = ApplicationMenuBuilder.makeMainMenu()
    let applicationMenu = menu.item(withTitle: "IronMLX")?.submenu

    #expect(
        applicationMenu?.item(withTitle: "About IronMLX")?.action
            == #selector(NSApplication.orderFrontStandardAboutPanel(_:))
    )
    #expect(applicationMenu?.item(withTitle: "About IronMLX")?.target === NSApp)
    #expect(
        applicationMenu?.item(withTitle: "Third-Party Notices…")?.action
            == #selector(ApplicationLegalNoticesPresenter.showThirdPartyNotices(_:))
    )
    #expect(
        applicationMenu?.item(withTitle: "Third-Party Notices…")?.target
            === ApplicationLegalNoticesPresenter.shared
    )
    #expect(applicationMenu?.item(withTitle: "Quit IronMLX")?.action == #selector(NSApplication.terminate(_:)))
    #expect(applicationMenu?.item(withTitle: "Quit IronMLX")?.keyEquivalent == "q")
}

@MainActor
@Test func applicationEditMenuRoutesClipboardCommandsToFirstResponder() {
    let menu = ApplicationMenuBuilder.makeMainMenu()
    let editMenu = menu.item(withTitle: "Edit")?.submenu

    #expect(editMenu?.item(withTitle: "Undo")?.action == Selector(("undo:")))
    #expect(editMenu?.item(withTitle: "Redo")?.action == Selector(("redo:")))
    #expect(editMenu?.item(withTitle: "Cut")?.action == #selector(NSText.cut(_:)))
    #expect(editMenu?.item(withTitle: "Copy")?.action == #selector(NSText.copy(_:)))
    #expect(editMenu?.item(withTitle: "Paste")?.action == #selector(NSText.paste(_:)))
    #expect(editMenu?.item(withTitle: "Select All")?.action == #selector(NSText.selectAll(_:)))
    #expect(editMenu?.item(withTitle: "Undo")?.keyEquivalent == "z")
    #expect(editMenu?.item(withTitle: "Redo")?.keyEquivalent == "z")
    #expect(editMenu?.item(withTitle: "Redo")?.keyEquivalentModifierMask.contains(.shift) == true)
    #expect(editMenu?.item(withTitle: "Paste")?.keyEquivalent == "v")
}
