import AppKit

@MainActor
public enum ApplicationMenuBuilder {
    public static func makeMainMenu() -> NSMenu {
        let menu = NSMenu()
        addApplicationMenu(to: menu)
        addEditMenu(to: menu)
        return menu
    }

    private static func addApplicationMenu(to menu: NSMenu) {
        let item = NSMenuItem(title: "ironmlx", action: nil, keyEquivalent: "")
        let submenu = NSMenu(title: "ironmlx")
        let quit = NSMenuItem(
            title: "Quit ironmlx",
            action: #selector(NSApplication.terminate(_:)),
            keyEquivalent: "q"
        )
        quit.target = NSApp
        submenu.addItem(quit)
        item.submenu = submenu
        menu.addItem(item)
    }

    private static func addEditMenu(to menu: NSMenu) {
        let item = NSMenuItem(title: "Edit", action: nil, keyEquivalent: "")
        let submenu = NSMenu(title: "Edit")
        submenu.addItem(editItem("Undo", Selector(("undo:")), "z"))
        let redo = editItem("Redo", Selector(("redo:")), "z")
        redo.keyEquivalentModifierMask = [.command, .shift]
        submenu.addItem(redo)
        submenu.addItem(.separator())
        submenu.addItem(editItem("Cut", #selector(NSText.cut(_:)), "x"))
        submenu.addItem(editItem("Copy", #selector(NSText.copy(_:)), "c"))
        submenu.addItem(editItem("Paste", #selector(NSText.paste(_:)), "v"))
        submenu.addItem(.separator())
        submenu.addItem(editItem("Select All", #selector(NSText.selectAll(_:)), "a"))
        item.submenu = submenu
        menu.addItem(item)
    }

    private static func editItem(_ title: String, _ action: Selector, _ keyEquivalent: String) -> NSMenuItem {
        let item = NSMenuItem(title: title, action: action, keyEquivalent: keyEquivalent)
        item.target = nil
        return item
    }
}
