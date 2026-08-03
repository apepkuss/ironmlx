import AppKit

@MainActor
public final class ApplicationLegalNoticesPresenter: NSObject {
    public static let shared = ApplicationLegalNoticesPresenter()

    @objc public func showThirdPartyNotices(_: Any?) {
        let alert = NSAlert()
        alert.messageText = "IronMLX Third-Party Notices"
        alert.addButton(withTitle: "Close")

        guard let noticesURL = Bundle.main.url(
            forResource: "THIRD_PARTY_NOTICES",
            withExtension: "md",
            subdirectory: "Legal"
        ), let notices = try? String(contentsOf: noticesURL, encoding: .utf8)
        else {
            alert.informativeText = "The bundled third-party notices could not be loaded."
            alert.alertStyle = .warning
            alert.runModal()
            return
        }

        let textView = NSTextView(frame: NSRect(x: 0, y: 0, width: 680, height: 460))
        textView.isEditable = false
        textView.isSelectable = true
        textView.font = .monospacedSystemFont(ofSize: 11, weight: .regular)
        textView.string = notices

        let scrollView = NSScrollView(frame: textView.frame)
        scrollView.hasVerticalScroller = true
        scrollView.autohidesScrollers = true
        scrollView.documentView = textView
        alert.accessoryView = scrollView
        alert.runModal()
    }
}
