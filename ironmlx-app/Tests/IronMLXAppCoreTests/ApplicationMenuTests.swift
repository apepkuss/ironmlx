import AppKit
import Testing

@testable import IronMLXAppCore

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
