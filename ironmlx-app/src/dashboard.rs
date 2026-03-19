//! Native dashboard window — Clash Verge-inspired layout with left sidebar navigation.

use std::sync::OnceLock;

use objc2::rc::Retained;
use objc2::{MainThreadMarker, msg_send};
use objc2_app_kit::*;
use objc2_foundation::*;

// ---------------------------------------------------------------------------
// Global dashboard window reference (OnceLock is Send+Sync)
// ---------------------------------------------------------------------------

// NSWindow is not Send+Sync, so we use a raw pointer wrapped in a Send+Sync newtype
struct JsonSwap(*const std::ffi::c_void);
unsafe impl Send for JsonSwap {}
unsafe impl Sync for JsonSwap {}

static DASHBOARD_WINDOW: OnceLock<std::sync::Mutex<Option<JsonSwap>>> = OnceLock::new();

fn window_lock() -> &'static std::sync::Mutex<Option<JsonSwap>> {
    DASHBOARD_WINDOW.get_or_init(|| std::sync::Mutex::new(None))
}

// Navigation items: (icon, label)
const NAV_ITEMS: &[(&str, &str)] = &[
    ("\u{1F3E0}", "Status"),          // 🏠
    ("\u{1F916}", "Models"),          // 🤖
    ("\u{1F4AC}", "Chat"),            // 💬
    ("\u{2699}\u{FE0F}", "Settings"), // ⚙️
    ("\u{1F4DD}", "Logs"),            // 📝
    ("\u{26A1}", "Benchmark"),        // ⚡
];

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Show the dashboard window (create if needed, bring to front if exists).
pub fn show_dashboard(mtm: MainThreadMarker) {
    let app = NSApplication::sharedApplication(mtm);

    // Show in Dock when dashboard is open
    app.setActivationPolicy(NSApplicationActivationPolicy::Regular);

    let mut guard = window_lock().lock().unwrap();

    if let Some(ref ptr) = *guard {
        let window: &NSWindow = unsafe { &*(ptr.0 as *const NSWindow) };
        window.makeKeyAndOrderFront(None);
        app.activateIgnoringOtherApps(true);
        return;
    }

    let window = create_dashboard_window(mtm);
    window.makeKeyAndOrderFront(None);
    app.activateIgnoringOtherApps(true);
    let raw = Retained::into_raw(window) as *const std::ffi::c_void;
    *guard = Some(JsonSwap(raw));
}

// ---------------------------------------------------------------------------
// Window creation
// ---------------------------------------------------------------------------

fn create_dashboard_window(mtm: MainThreadMarker) -> Retained<NSWindow> {
    let frame = NSRect::new(NSPoint::new(200.0, 200.0), NSSize::new(900.0, 600.0));
    let style = NSWindowStyleMask(
        NSWindowStyleMask::Titled.0
            | NSWindowStyleMask::Closable.0
            | NSWindowStyleMask::Miniaturizable.0
            | NSWindowStyleMask::Resizable.0,
    );

    let window = unsafe {
        NSWindow::initWithContentRect_styleMask_backing_defer(
            mtm.alloc(),
            frame,
            style,
            NSBackingStoreType(2), // Buffered
            false,
        )
    };

    window.setTitle(ns_string!("ironmlx"));
    window.setMinSize(NSSize::new(700.0, 450.0));
    window.center();

    // Transparent titlebar for Clash Verge-like look
    window.setTitlebarAppearsTransparent(true);
    window.setTitleVisibility(NSWindowTitleVisibility::Hidden);

    // Build content
    let content = build_content(mtm);
    window.setContentView(Some(&content));

    window
}

// ---------------------------------------------------------------------------
// Content: sidebar + main area using frame-based layout
// We use autoresizingMask for simplicity with objc2
// ---------------------------------------------------------------------------

fn build_content(mtm: MainThreadMarker) -> Retained<NSView> {
    let container = unsafe {
        NSView::initWithFrame(
            mtm.alloc(),
            NSRect::new(NSPoint::ZERO, NSSize::new(900.0, 600.0)),
        )
    };

    // Sidebar (left, fixed 200pt width)
    let sidebar = build_sidebar(mtm, 200.0, 600.0);
    unsafe {
        sidebar.setAutoresizingMask(NSAutoresizingMaskOptions(16)); // NSViewHeightSizable = 16
    }

    // Content area (right, fills remaining)
    let content = build_main_area(mtm, 700.0, 600.0);
    unsafe {
        content.setFrameOrigin(NSPoint::new(200.0, 0.0));
        content.setAutoresizingMask(NSAutoresizingMaskOptions(2 | 16)); // NSViewWidthSizable=2 | NSViewHeightSizable=16
    }

    unsafe {
        container.setAutoresizesSubviews(true);
        container.addSubview(&sidebar);
        container.addSubview(&content);
    }

    container
}

fn build_sidebar(mtm: MainThreadMarker, width: f64, height: f64) -> Retained<NSView> {
    let sidebar = unsafe {
        NSView::initWithFrame(
            mtm.alloc(),
            NSRect::new(NSPoint::ZERO, NSSize::new(width, height)),
        )
    };
    // Solid dark background
    unsafe {
        sidebar.setWantsLayer(true);
        if let Some(layer) = sidebar.layer() {
            let bg = NSColor::colorWithSRGBRed_green_blue_alpha(0.12, 0.12, 0.14, 1.0);
            let cg: *const std::ffi::c_void = msg_send![&bg, CGColor];
            let _: () = msg_send![&*layer, setBackgroundColor: cg];
        }
    }

    // Header: logo + title
    let header_y = height - 36.0 - 50.0; // 36pt from top for traffic lights
    let header = build_sidebar_header(mtm, 16.0, header_y, width - 32.0);
    unsafe {
        sidebar.addSubview(&header);
    }

    // Nav items
    let mut y = header_y - 50.0; // more space between header and nav
    for (i, (icon, label)) in NAV_ITEMS.iter().enumerate() {
        let item = build_nav_item(mtm, icon, label, i, 12.0, y, width - 24.0);
        y -= 38.0;
        unsafe {
            sidebar.addSubview(&item);
        }
    }

    // Bottom status
    let status = build_sidebar_status(mtm, 16.0, 12.0, width - 32.0);
    unsafe {
        sidebar.addSubview(&status);
    }

    sidebar
}

fn build_sidebar_header(mtm: MainThreadMarker, x: f64, y: f64, width: f64) -> Retained<NSView> {
    let view = unsafe {
        NSView::initWithFrame(
            mtm.alloc(),
            NSRect::new(NSPoint::new(x, y), NSSize::new(width, 40.0)),
        )
    };

    // Logo
    let icon_bytes = include_bytes!("../../assets/menubar-icon@2x.png");
    let ns_data = unsafe { NSData::with_bytes(icon_bytes) };
    if let Some(image) = unsafe { NSImage::initWithData(mtm.alloc(), &ns_data) } {
        unsafe {
            image.setSize(NSSize::new(24.0, 24.0));
        }
        let iv = unsafe {
            let iv = NSImageView::initWithFrame(
                mtm.alloc(),
                NSRect::new(NSPoint::new(0.0, 8.0), NSSize::new(24.0, 24.0)),
            );
            iv.setImage(Some(&image));
            iv
        };
        unsafe {
            view.addSubview(&iv);
        }
    }

    // Title text
    let title = unsafe {
        let tf = NSTextField::labelWithString(ns_string!("ironmlx"), mtm);
        tf.setFont(Some(&NSFont::boldSystemFontOfSize(18.0)));
        tf.setFrame(NSRect::new(
            NSPoint::new(32.0, 8.0),
            NSSize::new(120.0, 24.0),
        ));
        tf
    };
    unsafe {
        view.addSubview(&title);
    }

    view
}

fn build_nav_item(
    mtm: MainThreadMarker,
    icon: &str,
    label: &str,
    tag: usize,
    x: f64,
    y: f64,
    width: f64,
) -> Retained<NSView> {
    // Use NSButton for clickability
    let title = format!("{} {}", icon, label);
    let button = unsafe {
        let btn = NSButton::initWithFrame(
            mtm.alloc(),
            NSRect::new(NSPoint::new(x, y), NSSize::new(width, 34.0)),
        );
        btn.setTitle(&NSString::from_str(&title));
        btn.setFont(Some(&NSFont::systemFontOfSize(13.0)));
        btn.setAlignment(NSTextAlignment::Left);
        btn.setBordered(false);
        btn.setTag(tag as isize);

        // Style: transparent background, left-aligned text
        btn.setBezelStyle(NSBezelStyle(0)); // no bezel
        btn
    };

    // Highlight active item
    if tag == 0 {
        unsafe {
            button.setWantsLayer(true);
            if let Some(layer) = button.layer() {
                let color = NSColor::controlAccentColor();
                let cg: *const std::ffi::c_void = msg_send![&color, CGColor];
                let _: () = msg_send![&*layer, setBackgroundColor: cg];
                let _: () = msg_send![&*layer, setCornerRadius: 8.0f64];
            }
        }
    }

    let view: Retained<NSView> = unsafe { Retained::cast(button) };
    view
}

fn build_sidebar_status(mtm: MainThreadMarker, x: f64, y: f64, width: f64) -> Retained<NSView> {
    let view = unsafe {
        NSView::initWithFrame(
            mtm.alloc(),
            NSRect::new(NSPoint::new(x, y), NSSize::new(width, 50.0)),
        )
    };

    let model = unsafe {
        let tf = NSTextField::labelWithString(ns_string!("Model: \u{2014}"), mtm);
        tf.setFont(Some(&NSFont::systemFontOfSize(11.0)));
        tf.setTextColor(Some(&NSColor::secondaryLabelColor()));
        tf.setFrame(NSRect::new(
            NSPoint::new(0.0, 26.0),
            NSSize::new(width, 16.0),
        ));
        tf
    };

    let mem = unsafe {
        let tf = NSTextField::labelWithString(ns_string!("Memory: \u{2014}"), mtm);
        tf.setFont(Some(&NSFont::systemFontOfSize(11.0)));
        tf.setTextColor(Some(&NSColor::secondaryLabelColor()));
        tf.setFrame(NSRect::new(
            NSPoint::new(0.0, 6.0),
            NSSize::new(width, 16.0),
        ));
        tf
    };

    unsafe {
        view.addSubview(&model);
        view.addSubview(&mem);
    }

    view
}

// ---------------------------------------------------------------------------
// Main content area (right side)
// ---------------------------------------------------------------------------

fn build_main_area(mtm: MainThreadMarker, width: f64, height: f64) -> Retained<NSView> {
    let view = unsafe {
        NSView::initWithFrame(
            mtm.alloc(),
            NSRect::new(NSPoint::ZERO, NSSize::new(width, height)),
        )
    };

    // Page title
    let title = unsafe {
        let tf = NSTextField::labelWithString(ns_string!("Status"), mtm);
        tf.setFont(Some(&NSFont::boldSystemFontOfSize(24.0)));
        tf.setFrame(NSRect::new(
            NSPoint::new(24.0, height - 36.0 - 32.0),
            NSSize::new(300.0, 30.0),
        ));
        tf
    };

    let subtitle = unsafe {
        let tf = NSTextField::labelWithString(
            ns_string!("Server overview and real-time monitoring"),
            mtm,
        );
        tf.setFont(Some(&NSFont::systemFontOfSize(13.0)));
        tf.setTextColor(Some(&NSColor::secondaryLabelColor()));
        tf.setFrame(NSRect::new(
            NSPoint::new(24.0, height - 36.0 - 56.0),
            NSSize::new(400.0, 20.0),
        ));
        tf
    };

    // Placeholder cards
    let card1 = build_card(
        mtm,
        "Server Status",
        "Running",
        24.0,
        height - 180.0,
        200.0,
        80.0,
    );
    let card2 = build_card(
        mtm,
        "Active Memory",
        "\u{2014}",
        240.0,
        height - 180.0,
        200.0,
        80.0,
    );
    let card3 = build_card(
        mtm,
        "Peak Memory",
        "\u{2014}",
        456.0,
        height - 180.0,
        200.0,
        80.0,
    );

    unsafe {
        view.addSubview(&title);
        view.addSubview(&subtitle);
        view.addSubview(&card1);
        view.addSubview(&card2);
        view.addSubview(&card3);
    }

    view
}

fn build_card(
    mtm: MainThreadMarker,
    title: &str,
    value: &str,
    x: f64,
    y: f64,
    w: f64,
    h: f64,
) -> Retained<NSView> {
    let card = unsafe {
        let v = NSBox::initWithFrame(
            mtm.alloc(),
            NSRect::new(NSPoint::new(x, y), NSSize::new(w, h)),
        );
        v.setBoxType(NSBoxType::Custom);
        v.setCornerRadius(12.0);
        v.setBorderWidth(0.5);
        v.setBorderColor(&NSColor::separatorColor());
        v.setFillColor(&NSColor::colorWithSRGBRed_green_blue_alpha(
            0.15, 0.15, 0.18, 0.6,
        ));
        v.setTitlePosition(unsafe { std::mem::transmute(0u64) }); // NoTitle
        v
    };

    let title_tf = unsafe {
        let tf = NSTextField::labelWithString(&NSString::from_str(title), mtm);
        tf.setFont(Some(&NSFont::systemFontOfSize(11.0)));
        tf.setTextColor(Some(&NSColor::secondaryLabelColor()));
        tf.setFrame(NSRect::new(
            NSPoint::new(14.0, h - 30.0),
            NSSize::new(w - 28.0, 16.0),
        ));
        tf
    };

    let value_tf = unsafe {
        let tf = NSTextField::labelWithString(&NSString::from_str(value), mtm);
        tf.setFont(Some(&NSFont::boldSystemFontOfSize(20.0)));
        tf.setFrame(NSRect::new(
            NSPoint::new(14.0, 10.0),
            NSSize::new(w - 28.0, 28.0),
        ));
        tf
    };

    unsafe {
        card.addSubview(&title_tf);
        card.addSubview(&value_tf);
    }

    let view: Retained<NSView> = unsafe { Retained::cast(card) };
    view
}
