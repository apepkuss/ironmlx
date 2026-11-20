import Carbon.HIToolbox

/// A system-wide hotkey registered with Carbon `RegisterEventHotKey`, which needs no
/// Accessibility permission. Registration fails only when another program registered the same
/// combination this way; it cannot see a menu shortcut of the same keys in another App.
///
/// While registered, the system delivers the keystroke to this App as a hotkey event instead of
/// a key-down, so the handler is the one place the combination takes effect.
@MainActor
final class GlobalHotKey {
    /// "IMLX": tags this App's hotkeys so a handler ignores hotkeys registered elsewhere in the process.
    nonisolated static let signature: OSType = 0x494D_4C58
    private static var nextID: UInt32 = 1

    fileprivate let id: UInt32
    fileprivate let handler: @MainActor () -> Void
    private var hotKeyRef: EventHotKeyRef?
    private var handlerRef: EventHandlerRef?

    /// Registers `keyCode` (a `kVK_*` value) with Carbon `modifiers` (`cmdKey`, `optionKey`, …);
    /// nil when the event handler cannot be installed or the combination is taken.
    init?(keyCode: UInt32, modifiers: UInt32, handler: @escaping @MainActor () -> Void) {
        id = Self.nextID
        Self.nextID &+= 1
        self.handler = handler

        var pressed = EventTypeSpec(eventClass: OSType(kEventClassKeyboard), eventKind: UInt32(kEventHotKeyPressed))
        var installed: EventHandlerRef?
        let installStatus = InstallEventHandler(
            GetApplicationEventTarget(),
            globalHotKeyPressed,
            1,
            &pressed,
            Unmanaged.passUnretained(self).toOpaque(),
            &installed
        )
        guard installStatus == noErr, let installed else {
            return nil
        }
        handlerRef = installed

        var registered: EventHotKeyRef?
        let registerStatus = RegisterEventHotKey(
            keyCode,
            modifiers,
            EventHotKeyID(signature: Self.signature, id: id),
            GetApplicationEventTarget(),
            0,
            &registered
        )
        guard registerStatus == noErr, let registered else {
            // deinit removes the installed handler.
            return nil
        }
        hotKeyRef = registered
    }

    isolated deinit {
        if let hotKeyRef {
            UnregisterEventHotKey(hotKeyRef)
        }
        if let handlerRef {
            RemoveEventHandler(handlerRef)
        }
    }

    /// The HUD hotkey, ⌃⌥⌘H. Not ⌥⌘H, which is "Hide Others" in almost every Mac App.
    static let hudDisplay = "⌃⌥⌘H"

    static func hudToggle(handler: @escaping @MainActor () -> Void) -> GlobalHotKey? {
        GlobalHotKey(keyCode: UInt32(kVK_ANSI_H), modifiers: UInt32(controlKey | optionKey | cmdKey), handler: handler)
    }
}

/// Carbon dispatches application events on the main thread.
private func globalHotKeyPressed(
    _ next: EventHandlerCallRef?,
    _ event: EventRef?,
    _ userData: UnsafeMutableRawPointer?
) -> OSStatus {
    guard let event, let userData else {
        return OSStatus(eventNotHandledErr)
    }
    var pressedID = EventHotKeyID()
    let status = GetEventParameter(
        event,
        EventParamName(kEventParamDirectObject),
        EventParamType(typeEventHotKeyID),
        nil,
        MemoryLayout<EventHotKeyID>.size,
        nil,
        &pressedID
    )
    guard status == noErr, pressedID.signature == GlobalHotKey.signature else {
        return OSStatus(eventNotHandledErr)
    }
    // The pointer is the `GlobalHotKey` itself, which removes this handler before it goes away.
    nonisolated(unsafe) let owner = userData
    return MainActor.assumeIsolated {
        let hotKey = Unmanaged<GlobalHotKey>.fromOpaque(owner).takeUnretainedValue()
        guard pressedID.id == hotKey.id else {
            return OSStatus(eventNotHandledErr)
        }
        hotKey.handler()
        return noErr
    }
}
