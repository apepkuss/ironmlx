//! Native dashboard window — Clash Verge-inspired layout with left sidebar navigation.

use std::sync::{Mutex, OnceLock};

use objc2::rc::Retained;
use objc2::runtime::Sel;
use objc2::{MainThreadMarker, msg_send, sel};
use objc2_app_kit::*;
use objc2_foundation::*;

// ---------------------------------------------------------------------------
// Global state
// ---------------------------------------------------------------------------

struct RawPtr(*const std::ffi::c_void);
unsafe impl Send for RawPtr {}
unsafe impl Sync for RawPtr {}

static DASHBOARD_WINDOW: OnceLock<Mutex<Option<RawPtr>>> = OnceLock::new();
fn window_lock() -> &'static Mutex<Option<RawPtr>> {
    DASHBOARD_WINDOW.get_or_init(|| Mutex::new(None))
}

// Page views stored globally for navigation switching
static PAGES: OnceLock<Mutex<Vec<RawPtr>>> = OnceLock::new();
fn pages_lock() -> &'static Mutex<Vec<RawPtr>> {
    PAGES.get_or_init(|| Mutex::new(Vec::new()))
}

// Nav highlight backgrounds for switching
static NAV_HIGHLIGHTS: OnceLock<Mutex<Vec<RawPtr>>> = OnceLock::new();
fn nav_highlights_lock() -> &'static Mutex<Vec<RawPtr>> {
    NAV_HIGHLIGHTS.get_or_init(|| Mutex::new(Vec::new()))
}

// Nav handler instance
static NAV_HANDLER: OnceLock<Mutex<Option<RawPtr>>> = OnceLock::new();
fn nav_handler_lock() -> &'static Mutex<Option<RawPtr>> {
    NAV_HANDLER.get_or_init(|| Mutex::new(None))
}

// Navigation items: (icon, label)
const NAV_ITEMS: &[(&str, &str)] = &[
    ("\u{1F3E0}", "Status"),
    ("\u{1F916}", "Models"),
    ("\u{1F4AC}", "Chat"),
    ("\u{2699}\u{FE0F}", "Settings"),
    ("\u{1F4DD}", "Logs"),
    ("\u{26A1}", "Benchmark"),
];

// Status card value labels — stored for polling updates
// Keys: "status", "active_mem", "peak_mem", "models", "uptime", "default_model"
static STATUS_LABELS: OnceLock<Mutex<std::collections::HashMap<&'static str, RawPtr>>> =
    OnceLock::new();
fn status_labels_lock() -> &'static Mutex<std::collections::HashMap<&'static str, RawPtr>> {
    STATUS_LABELS.get_or_init(|| Mutex::new(std::collections::HashMap::new()))
}

// Sidebar status labels
static SIDEBAR_MODEL_LABEL: OnceLock<Mutex<Option<RawPtr>>> = OnceLock::new();
static SIDEBAR_MEM_LABEL: OnceLock<Mutex<Option<RawPtr>>> = OnceLock::new();

/// Start polling /health API every 5 seconds to update Status page
pub fn start_status_polling(port: u16) {
    let url = format!("http://localhost:{}/health", port);
    std::thread::spawn(move || {
        loop {
            std::thread::sleep(std::time::Duration::from_secs(5));
            if let Ok(body) = std::process::Command::new("curl")
                .args(["-s", &url])
                .output()
            {
                if body.status.success() {
                    if let Ok(json) = serde_json::from_slice::<serde_json::Value>(&body.stdout) {
                        update_status_labels(&json);
                    }
                }
            }
        }
    });
}

fn update_status_labels(json: &serde_json::Value) {
    let labels = status_labels_lock().lock().unwrap();

    let set_label = |key: &str, text: &str| {
        if let Some(ptr) = labels.get(key) {
            let tf: &NSTextField = unsafe { &*(ptr.0 as *const NSTextField) };
            unsafe {
                tf.setStringValue(&NSString::from_str(text));
            }
        }
    };

    let status = json
        .get("status")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");
    set_label("status", if status == "ok" { "Running" } else { status });

    if let Some(mem) = json.get("memory") {
        let active = mem.get("active_mb").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let peak = mem.get("peak_mb").and_then(|v| v.as_f64()).unwrap_or(0.0);
        set_label("active_mem", &format!("{:.0} MB", active));
        set_label("peak_mem", &format!("{:.0} MB", peak));
    }

    let model = json
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("\u{2014}");
    set_label("default_model", model);

    if let Some(started) = json.get("started_at").and_then(|v| v.as_i64()) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;
        let elapsed = now - started;
        let h = elapsed / 3600;
        let m = (elapsed % 3600) / 60;
        set_label("uptime", &format!("{}h {}m", h, m));
    }

    // Update sidebar labels
    if let Ok(guard) = SIDEBAR_MODEL_LABEL.get_or_init(|| Mutex::new(None)).lock() {
        if let Some(ptr) = guard.as_ref() {
            let tf: &NSTextField = unsafe { &*(ptr.0 as *const NSTextField) };
            let short_model = model.split('/').last().unwrap_or(model);
            unsafe {
                tf.setStringValue(&NSString::from_str(&format!("Model: {}", short_model)));
            }
        }
    }
    if let Ok(guard) = SIDEBAR_MEM_LABEL.get_or_init(|| Mutex::new(None)).lock() {
        if let Some(ptr) = guard.as_ref() {
            let tf: &NSTextField = unsafe { &*(ptr.0 as *const NSTextField) };
            if let Some(mem) = json.get("memory") {
                let active = mem.get("active_mb").and_then(|v| v.as_f64()).unwrap_or(0.0);
                unsafe {
                    tf.setStringValue(&NSString::from_str(&format!("Memory: {:.0} MB", active)));
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// NavHandler — handles sidebar button clicks
// ---------------------------------------------------------------------------

use objc2::runtime::{AnyClass, AnyObject, ClassBuilder};

static NAV_HANDLER_CLASS: OnceLock<&'static AnyClass> = OnceLock::new();

extern "C" fn nav_clicked(_this: *mut AnyObject, _sel: Sel, sender: *mut AnyObject) {
    let idx: isize = unsafe { msg_send![sender, tag] };
    switch_page(idx as usize);
}

fn nav_handler_class() -> &'static AnyClass {
    NAV_HANDLER_CLASS.get_or_init(|| {
        let superclass = AnyClass::get(c"NSObject").unwrap();
        let mut builder = ClassBuilder::new(c"IronNavHandler", superclass).unwrap();

        unsafe {
            builder.add_method(
                sel!(navClicked:),
                nav_clicked as extern "C" fn(*mut AnyObject, Sel, *mut AnyObject),
            );
        }

        builder.register()
    })
}

fn switch_page(idx: usize) {
    let pages = pages_lock().lock().unwrap();
    let highlights = nav_highlights_lock().lock().unwrap();

    // Hide all pages, show selected
    for (i, ptr) in pages.iter().enumerate() {
        let view: &NSView = unsafe { &*(ptr.0 as *const NSView) };
        unsafe {
            view.setHidden(i != idx);
        }
    }

    // Update nav highlights — show/hide background views
    for (i, ptr) in highlights.iter().enumerate() {
        let bg: &NSView = unsafe { &*(ptr.0 as *const NSView) };
        unsafe {
            bg.setHidden(i != idx);
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

pub fn show_dashboard(mtm: MainThreadMarker) {
    let app = NSApplication::sharedApplication(mtm);
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
    *guard = Some(RawPtr(raw));

    // Start polling on first dashboard open
    static POLLING_STARTED: OnceLock<bool> = OnceLock::new();
    POLLING_STARTED.get_or_init(|| {
        let port = crate::config::AppConfig::load().port;
        start_status_polling(port);
        true
    });
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
            NSBackingStoreType(2),
            false,
        )
    };

    window.setTitle(ns_string!("ironmlx"));
    window.setMinSize(NSSize::new(700.0, 450.0));
    window.center();
    window.setTitlebarAppearsTransparent(true);
    window.setTitleVisibility(NSWindowTitleVisibility::Hidden);

    let content = build_content(mtm);
    window.setContentView(Some(&content));

    window
}

// ---------------------------------------------------------------------------
// Content layout
// ---------------------------------------------------------------------------

fn build_content(mtm: MainThreadMarker) -> Retained<NSView> {
    let w = 900.0f64;
    let h = 600.0f64;
    let sw = 200.0f64; // sidebar width
    let cw = w - sw; // content width

    let container = unsafe {
        let v = NSView::initWithFrame(mtm.alloc(), NSRect::new(NSPoint::ZERO, NSSize::new(w, h)));
        v.setAutoresizesSubviews(true);
        v
    };

    // Create NavHandler
    let cls = nav_handler_class();
    let handler: Retained<AnyObject> = unsafe {
        let obj: *mut AnyObject = msg_send![cls, alloc];
        let obj: *mut AnyObject = msg_send![obj, init];
        Retained::from_raw(obj).unwrap()
    };
    let handler_ptr = Retained::into_raw(handler) as *const std::ffi::c_void;
    *nav_handler_lock().lock().unwrap() = Some(RawPtr(handler_ptr));

    // Sidebar
    let sidebar = build_sidebar(mtm, sw, h);
    unsafe {
        sidebar.setAutoresizingMask(NSAutoresizingMaskOptions(16));
    }

    // Pages container
    let pages_container = unsafe {
        let v = NSView::initWithFrame(
            mtm.alloc(),
            NSRect::new(NSPoint::new(sw, 0.0), NSSize::new(cw, h)),
        );
        v.setAutoresizesSubviews(true);
        v.setAutoresizingMask(NSAutoresizingMaskOptions(2 | 16));
        v
    };

    // Build all pages
    let page_builders: Vec<fn(MainThreadMarker, f64, f64) -> Retained<NSView>> = vec![
        build_status_page,
        build_models_page,
        build_chat_page,
        build_settings_page,
        build_logs_page,
        build_benchmark_page,
    ];

    let mut page_ptrs = Vec::new();
    for (i, builder) in page_builders.iter().enumerate() {
        let page = builder(mtm, cw, h);
        unsafe {
            page.setAutoresizingMask(NSAutoresizingMaskOptions(2 | 16));
            page.setHidden(i != 0); // Only Status visible initially
            pages_container.addSubview(&page);
        }
        let ptr = Retained::into_raw(page) as *const std::ffi::c_void;
        page_ptrs.push(RawPtr(ptr));
    }
    *pages_lock().lock().unwrap() = page_ptrs;

    unsafe {
        container.addSubview(&sidebar);
        container.addSubview(&pages_container);
    }

    container
}

// ---------------------------------------------------------------------------
// Sidebar
// ---------------------------------------------------------------------------

fn build_sidebar(mtm: MainThreadMarker, width: f64, height: f64) -> Retained<NSView> {
    let sidebar = unsafe {
        NSView::initWithFrame(
            mtm.alloc(),
            NSRect::new(NSPoint::ZERO, NSSize::new(width, height)),
        )
    };
    // Use system color that auto-adapts to light/dark theme
    unsafe {
        sidebar.setWantsLayer(true);
        if let Some(layer) = sidebar.layer() {
            let bg = NSColor::controlBackgroundColor();
            let cg: *const std::ffi::c_void = msg_send![&bg, CGColor];
            let _: () = msg_send![&*layer, setBackgroundColor: cg];
        }
    }

    // Header
    let header_y = height - 36.0 - 50.0;
    let header = build_sidebar_header(mtm, 16.0, header_y, width - 32.0);
    unsafe {
        sidebar.addSubview(&header);
    }

    // Nav items
    let handler_guard = nav_handler_lock().lock().unwrap();
    let handler_ptr = handler_guard.as_ref().unwrap().0;
    let action = sel!(navClicked:);

    let mut highlight_ptrs = Vec::new();
    let mut y = header_y - 50.0;
    for (i, (icon, label)) in NAV_ITEMS.iter().enumerate() {
        // Highlight background view (behind button)
        let highlight_bg = unsafe {
            let v = NSView::initWithFrame(
                mtm.alloc(),
                NSRect::new(NSPoint::new(12.0, y), NSSize::new(width - 24.0, 34.0)),
            );
            v.setWantsLayer(true);
            if let Some(layer) = v.layer() {
                let _: () = msg_send![&*layer, setCornerRadius: 8.0f64];
                let color = NSColor::controlAccentColor().colorWithAlphaComponent(0.25);
                let cg: *const std::ffi::c_void = msg_send![&color, CGColor];
                let _: () = msg_send![&*layer, setBackgroundColor: cg];
            }
            v.setHidden(i != 0); // Only Status highlight visible initially
            v
        };
        // Store raw pointer without consuming the Retained — sidebar.addSubview retains it
        let highlight_ptr = &*highlight_bg as *const NSView as *const std::ffi::c_void;
        highlight_ptrs.push(RawPtr(highlight_ptr));

        let btn = build_nav_button(
            mtm,
            icon,
            label,
            i,
            12.0,
            y,
            width - 24.0,
            handler_ptr,
            action,
        );
        y -= 38.0;
        unsafe {
            sidebar.addSubview(&highlight_bg);
            sidebar.addSubview(&btn);
        }
    }
    *nav_highlights_lock().lock().unwrap() = highlight_ptrs;

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
            NSRect::new(NSPoint::new(x, y), NSSize::new(width, 44.0)),
        )
    };

    let icon_bytes = include_bytes!("../../assets/menubar-icon@2x.png");
    let ns_data = unsafe { NSData::with_bytes(icon_bytes) };
    if let Some(image) = unsafe { NSImage::initWithData(mtm.alloc(), &ns_data) } {
        unsafe {
            image.setSize(NSSize::new(32.0, 32.0));
        }
        let iv = unsafe {
            let iv = NSImageView::initWithFrame(
                mtm.alloc(),
                NSRect::new(NSPoint::new(0.0, 6.0), NSSize::new(32.0, 32.0)),
            );
            iv.setImage(Some(&image));
            iv
        };
        unsafe {
            view.addSubview(&iv);
        }
    }

    let title = unsafe {
        let tf = NSTextField::labelWithString(ns_string!("IronMLX"), mtm);
        tf.setFont(Some(&NSFont::boldSystemFontOfSize(22.0)));
        tf.setTextColor(Some(&NSColor::labelColor()));
        tf.setFrame(NSRect::new(
            NSPoint::new(40.0, 8.0),
            NSSize::new(130.0, 28.0),
        ));
        tf
    };
    unsafe {
        view.addSubview(&title);
    }

    view
}

fn build_nav_button(
    mtm: MainThreadMarker,
    icon: &str,
    label: &str,
    tag: usize,
    x: f64,
    y: f64,
    width: f64,
    handler: *const std::ffi::c_void,
    action: Sel,
) -> Retained<NSButton> {
    let title = format!("  {} {}", icon, label);
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
        btn.setBezelStyle(NSBezelStyle(0));

        // Set action
        let handler_obj: &NSObject = &*(handler as *const NSObject);
        btn.setTarget(Some(handler_obj));
        btn.setAction(Some(action));

        // Disable focus ring — highlight handled by separate background view
        let cell: &NSCell = msg_send![&btn, cell];
        let _: () = msg_send![cell, setFocusRingType: 1i64]; // NSFocusRingTypeNone = 1

        btn
    };

    button
}

fn build_sidebar_status(mtm: MainThreadMarker, x: f64, y: f64, width: f64) -> Retained<NSView> {
    let view = unsafe {
        NSView::initWithFrame(
            mtm.alloc(),
            NSRect::new(NSPoint::new(x, y), NSSize::new(width, 50.0)),
        )
    };

    let model_tf = unsafe {
        let tf = NSTextField::labelWithString(ns_string!("Model: \u{2014}"), mtm);
        tf.setFont(Some(&NSFont::systemFontOfSize(11.0)));
        tf.setTextColor(Some(&NSColor::secondaryLabelColor()));
        tf.setFrame(NSRect::new(
            NSPoint::new(0.0, 26.0),
            NSSize::new(width, 16.0),
        ));
        tf
    };
    // Store sidebar model label pointer
    *SIDEBAR_MODEL_LABEL
        .get_or_init(|| Mutex::new(None))
        .lock()
        .unwrap() = Some(RawPtr(
        &*model_tf as *const NSTextField as *const std::ffi::c_void,
    ));

    let mem_tf = unsafe {
        let tf = NSTextField::labelWithString(ns_string!("Memory: \u{2014}"), mtm);
        tf.setFont(Some(&NSFont::systemFontOfSize(11.0)));
        tf.setTextColor(Some(&NSColor::secondaryLabelColor()));
        tf.setFrame(NSRect::new(
            NSPoint::new(0.0, 6.0),
            NSSize::new(width, 16.0),
        ));
        tf
    };
    // Store sidebar memory label pointer
    *SIDEBAR_MEM_LABEL
        .get_or_init(|| Mutex::new(None))
        .lock()
        .unwrap() = Some(RawPtr(
        &*mem_tf as *const NSTextField as *const std::ffi::c_void,
    ));

    unsafe {
        view.addSubview(&model_tf);
        view.addSubview(&mem_tf);
    }

    view
}

// ---------------------------------------------------------------------------
// Page builders
// ---------------------------------------------------------------------------

fn build_status_page(mtm: MainThreadMarker, width: f64, height: f64) -> Retained<NSView> {
    let view = make_page_view(mtm, width, height);

    let title = make_title(mtm, "Status", height);
    let subtitle = make_subtitle(mtm, "Server overview and real-time monitoring", height);

    let card_y = height - 200.0;
    let card_w = (width - 24.0 * 2.0 - 16.0 * 2.0) / 3.0;

    let card1 = build_status_card(
        mtm,
        "Server Status",
        "Running",
        "status",
        24.0,
        card_y,
        card_w,
        90.0,
    );
    let card2 = build_status_card(
        mtm,
        "Active Memory",
        "\u{2014}",
        "active_mem",
        24.0 + card_w + 16.0,
        card_y,
        card_w,
        90.0,
    );
    let card3 = build_status_card(
        mtm,
        "Peak Memory",
        "\u{2014}",
        "peak_mem",
        24.0 + (card_w + 16.0) * 2.0,
        card_y,
        card_w,
        90.0,
    );

    let card_y2 = card_y - 110.0;
    let card4 = build_status_card(
        mtm,
        "Loaded Models",
        "1",
        "models",
        24.0,
        card_y2,
        card_w,
        90.0,
    );
    let card5 = build_status_card(
        mtm,
        "Uptime",
        "\u{2014}",
        "uptime",
        24.0 + card_w + 16.0,
        card_y2,
        card_w,
        90.0,
    );
    let card6 = build_status_card(
        mtm,
        "Default Model",
        "\u{2014}",
        "default_model",
        24.0 + (card_w + 16.0) * 2.0,
        card_y2,
        card_w,
        90.0,
    );

    unsafe {
        view.addSubview(&title);
        view.addSubview(&subtitle);
        view.addSubview(&card1);
        view.addSubview(&card2);
        view.addSubview(&card3);
        view.addSubview(&card4);
        view.addSubview(&card5);
        view.addSubview(&card6);
    }

    view
}

fn build_models_page(mtm: MainThreadMarker, width: f64, height: f64) -> Retained<NSView> {
    let view = make_page_view(mtm, width, height);

    let title = make_title(mtm, "Models", height);
    let subtitle = make_subtitle(mtm, "Manage loaded models", height);

    // Load model section
    let load_label = make_label(mtm, "Load Model", 24.0, height - 140.0, 100.0, true);
    let input = unsafe {
        let tf = NSTextField::initWithFrame(
            mtm.alloc(),
            NSRect::new(
                NSPoint::new(130.0, height - 142.0),
                NSSize::new(width - 260.0, 26.0),
            ),
        );
        tf.setPlaceholderString(Some(ns_string!("HuggingFace repo ID or local path")));
        tf.setFont(Some(&NSFont::systemFontOfSize(13.0)));
        tf.setBezeled(true);
        tf
    };
    let load_btn = make_button(mtm, "Load", width - 110.0, height - 142.0, 80.0);

    // Model list header
    let list_header = make_label(mtm, "Loaded Models", 24.0, height - 190.0, 200.0, true);

    // Placeholder model entry
    let model_card = build_card(
        mtm,
        "mlx-community/Qwen3-0.6B-4bit",
        "default \u{2022} running",
        24.0,
        height - 290.0,
        width - 48.0,
        70.0,
    );

    unsafe {
        view.addSubview(&title);
        view.addSubview(&subtitle);
        view.addSubview(&load_label);
        view.addSubview(&input);
        view.addSubview(&load_btn);
        view.addSubview(&list_header);
        view.addSubview(&model_card);
    }

    view
}

fn build_chat_page(mtm: MainThreadMarker, width: f64, height: f64) -> Retained<NSView> {
    let view = make_page_view(mtm, width, height);

    let title = make_title(mtm, "Chat", height);
    let subtitle = make_subtitle(mtm, "Interactive conversation with your model", height);

    let placeholder = unsafe {
        let tf = NSTextField::labelWithString(
            ns_string!("Chat is available via Web Admin Panel\nhttp://localhost:8080/admin"),
            mtm,
        );
        tf.setFont(Some(&NSFont::systemFontOfSize(14.0)));
        tf.setTextColor(Some(&NSColor::secondaryLabelColor()));
        tf.setAlignment(NSTextAlignment::Center);
        tf.setFrame(NSRect::new(
            NSPoint::new(24.0, height / 2.0 - 20.0),
            NSSize::new(width - 48.0, 40.0),
        ));
        tf
    };

    unsafe {
        view.addSubview(&title);
        view.addSubview(&subtitle);
        view.addSubview(&placeholder);
    }

    view
}

fn build_settings_page(mtm: MainThreadMarker, width: f64, height: f64) -> Retained<NSView> {
    let view = make_page_view(mtm, width, height);

    let title = make_title(mtm, "Settings", height);
    let subtitle = make_subtitle(mtm, "Server configuration", height);

    let mut y = height - 150.0;
    let label_w = 120.0;
    let field_w = width - label_w - 80.0;
    let row_h = 36.0;

    let fields = [
        ("Host", "127.0.0.1", true),
        ("Port", "8080", true),
        ("Temperature", "1.0", false),
        ("Top P", "1.0", false),
        ("Max Tokens", "2048", false),
        ("HF Endpoint", "https://huggingface.co", false),
    ];

    for (label_text, value, readonly) in &fields {
        let label = make_label(mtm, label_text, 24.0, y, label_w, false);
        let field = unsafe {
            let tf = NSTextField::initWithFrame(
                mtm.alloc(),
                NSRect::new(
                    NSPoint::new(24.0 + label_w + 12.0, y),
                    NSSize::new(field_w, 24.0),
                ),
            );
            tf.setStringValue(&NSString::from_str(value));
            tf.setFont(Some(&NSFont::systemFontOfSize(13.0)));
            tf.setBezeled(true);
            tf.setEditable(!readonly);
            if *readonly {
                tf.setTextColor(Some(&NSColor::secondaryLabelColor()));
            }
            tf
        };
        y -= row_h;

        unsafe {
            view.addSubview(&label);
            view.addSubview(&field);
        }
    }

    let save_btn = make_button(mtm, "Save Settings", 24.0 + label_w + 12.0, y - 10.0, 120.0);
    unsafe {
        view.addSubview(&title);
        view.addSubview(&subtitle);
        view.addSubview(&save_btn);
    }

    view
}

fn build_logs_page(mtm: MainThreadMarker, width: f64, height: f64) -> Retained<NSView> {
    let view = make_page_view(mtm, width, height);

    let title = make_title(mtm, "Logs", height);
    let subtitle = make_subtitle(mtm, "Server activity", height);

    // Filter buttons
    let filters = ["All", "Info", "Warn", "Error"];
    let mut fx = 24.0;
    for label in &filters {
        let btn = make_button(mtm, label, fx, height - 140.0, 60.0);
        unsafe {
            view.addSubview(&btn);
        }
        fx += 70.0;
    }

    // Log area (NSScrollView + NSTextView)
    let scroll = unsafe {
        let sv = NSScrollView::initWithFrame(
            mtm.alloc(),
            NSRect::new(
                NSPoint::new(24.0, 24.0),
                NSSize::new(width - 48.0, height - 190.0),
            ),
        );
        sv.setHasVerticalScroller(true);
        sv.setBorderType(NSBorderType::BezelBorder);
        sv
    };

    let text_view = unsafe {
        let tv = NSTextView::initWithFrame(
            mtm.alloc(),
            NSRect::new(NSPoint::ZERO, NSSize::new(width - 64.0, height - 190.0)),
        );
        tv.setEditable(false);
        tv.setFont(Some(&NSFont::monospacedSystemFontOfSize_weight(
            11.0,
            unsafe { std::mem::transmute(0.0f64) },
        )));
        tv.setString(ns_string!(
            "[2026-03-18 22:30:01] INFO  Server started on port 8080\n\
             [2026-03-18 22:30:02] INFO  Loaded model: mlx-community/Qwen3-0.6B-4bit\n\
             [2026-03-18 22:30:15] INFO  POST /v1/chat/completions - 200 (1.2s)\n\
             [2026-03-18 22:31:03] WARN  Memory usage above 80%\n\
             [2026-03-18 22:31:45] INFO  POST /v1/chat/completions - 200 (0.8s)\n"
        ));
        tv
    };

    unsafe {
        scroll.setDocumentView(Some(&text_view));
        let scroll_view: Retained<NSView> = Retained::cast(scroll);
        view.addSubview(&title);
        view.addSubview(&subtitle);
        view.addSubview(&scroll_view);
    }

    view
}

fn build_benchmark_page(mtm: MainThreadMarker, width: f64, height: f64) -> Retained<NSView> {
    let view = make_page_view(mtm, width, height);

    let title = make_title(mtm, "Benchmark", height);
    let subtitle = make_subtitle(mtm, "Performance testing", height);

    // Form
    let mut y = height - 150.0;

    let prompt_label = make_label(mtm, "Prompt", 24.0, y, 90.0, false);
    let prompt_field = unsafe {
        let tf = NSTextField::initWithFrame(
            mtm.alloc(),
            NSRect::new(NSPoint::new(120.0, y), NSSize::new(width - 240.0, 24.0)),
        );
        tf.setStringValue(ns_string!("Explain quantum computing in simple terms."));
        tf.setFont(Some(&NSFont::systemFontOfSize(13.0)));
        tf.setBezeled(true);
        tf
    };
    y -= 36.0;

    let tokens_label = make_label(mtm, "Max Tokens", 24.0, y, 90.0, false);
    let tokens_field = unsafe {
        let tf = NSTextField::initWithFrame(
            mtm.alloc(),
            NSRect::new(NSPoint::new(120.0, y), NSSize::new(80.0, 24.0)),
        );
        tf.setStringValue(ns_string!("100"));
        tf.setFont(Some(&NSFont::systemFontOfSize(13.0)));
        tf.setBezeled(true);
        tf
    };
    y -= 40.0;

    let run_btn = make_button(mtm, "Run Benchmark", 120.0, y, 130.0);

    // Results cards
    y -= 50.0;
    let result_label = make_label(mtm, "Results", 24.0, y, 100.0, true);
    y -= 10.0;

    let card_w = (width - 48.0 - 32.0) / 3.0;
    let card1 = build_card(mtm, "TTFT", "\u{2014}", 24.0, y - 90.0, card_w, 80.0);
    let card2 = build_card(
        mtm,
        "Tokens/sec",
        "\u{2014}",
        24.0 + card_w + 16.0,
        y - 90.0,
        card_w,
        80.0,
    );
    let card3 = build_card(
        mtm,
        "Total Time",
        "\u{2014}",
        24.0 + (card_w + 16.0) * 2.0,
        y - 90.0,
        card_w,
        80.0,
    );

    unsafe {
        view.addSubview(&title);
        view.addSubview(&subtitle);
        view.addSubview(&prompt_label);
        view.addSubview(&prompt_field);
        view.addSubview(&tokens_label);
        view.addSubview(&tokens_field);
        view.addSubview(&run_btn);
        view.addSubview(&result_label);
        view.addSubview(&card1);
        view.addSubview(&card2);
        view.addSubview(&card3);
    }

    view
}

// ---------------------------------------------------------------------------
// Status card (registers value label for polling)
// ---------------------------------------------------------------------------

fn build_status_card(
    mtm: MainThreadMarker,
    title: &str,
    value: &str,
    key: &'static str,
    x: f64,
    y: f64,
    w: f64,
    h: f64,
) -> Retained<NSView> {
    let (card, value_label_ptr) = build_card_with_label(mtm, title, value, x, y, w, h);
    if let Some(ptr) = value_label_ptr {
        status_labels_lock()
            .lock()
            .unwrap()
            .insert(key, RawPtr(ptr));
    }
    card
}

// ---------------------------------------------------------------------------
// Shared UI helpers
// ---------------------------------------------------------------------------

fn make_page_view(mtm: MainThreadMarker, width: f64, height: f64) -> Retained<NSView> {
    unsafe {
        NSView::initWithFrame(
            mtm.alloc(),
            NSRect::new(NSPoint::ZERO, NSSize::new(width, height)),
        )
    }
}

fn make_title(mtm: MainThreadMarker, text: &str, height: f64) -> Retained<NSTextField> {
    unsafe {
        let tf = NSTextField::labelWithString(&NSString::from_str(text), mtm);
        tf.setFont(Some(&NSFont::boldSystemFontOfSize(24.0)));
        tf.setFrame(NSRect::new(
            NSPoint::new(24.0, height - 36.0 - 32.0),
            NSSize::new(300.0, 30.0),
        ));
        tf
    }
}

fn make_subtitle(mtm: MainThreadMarker, text: &str, height: f64) -> Retained<NSTextField> {
    unsafe {
        let tf = NSTextField::labelWithString(&NSString::from_str(text), mtm);
        tf.setFont(Some(&NSFont::systemFontOfSize(13.0)));
        tf.setTextColor(Some(&NSColor::secondaryLabelColor()));
        tf.setFrame(NSRect::new(
            NSPoint::new(24.0, height - 36.0 - 58.0),
            NSSize::new(500.0, 20.0),
        ));
        tf
    }
}

fn make_label(
    mtm: MainThreadMarker,
    text: &str,
    x: f64,
    y: f64,
    width: f64,
    bold: bool,
) -> Retained<NSTextField> {
    unsafe {
        let tf = NSTextField::labelWithString(&NSString::from_str(text), mtm);
        if bold {
            tf.setFont(Some(&NSFont::boldSystemFontOfSize(14.0)));
        } else {
            tf.setFont(Some(&NSFont::systemFontOfSize(13.0)));
        }
        tf.setFrame(NSRect::new(NSPoint::new(x, y), NSSize::new(width, 20.0)));
        tf
    }
}

fn make_button(
    mtm: MainThreadMarker,
    title: &str,
    x: f64,
    y: f64,
    width: f64,
) -> Retained<NSButton> {
    unsafe {
        let btn = NSButton::initWithFrame(
            mtm.alloc(),
            NSRect::new(NSPoint::new(x, y), NSSize::new(width, 28.0)),
        );
        btn.setTitle(&NSString::from_str(title));
        btn.setFont(Some(&NSFont::systemFontOfSize(12.0)));
        btn.setBezelStyle(NSBezelStyle::Rounded);
        btn
    }
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
    build_card_with_label(mtm, title, value, x, y, w, h).0
}

fn build_card_with_label(
    mtm: MainThreadMarker,
    title: &str,
    value: &str,
    x: f64,
    y: f64,
    w: f64,
    h: f64,
) -> (Retained<NSView>, Option<*const std::ffi::c_void>) {
    let card = unsafe {
        let v = NSBox::initWithFrame(
            mtm.alloc(),
            NSRect::new(NSPoint::new(x, y), NSSize::new(w, h)),
        );
        v.setBoxType(NSBoxType::Custom);
        v.setCornerRadius(12.0);
        v.setBorderWidth(0.5);
        v.setBorderColor(&NSColor::separatorColor());
        v.setFillColor(&NSColor::windowBackgroundColor());
        v.setTitlePosition(unsafe { std::mem::transmute(0u64) });
        v
    };

    let title_tf = unsafe {
        let tf = NSTextField::labelWithString(&NSString::from_str(title), mtm);
        tf.setFont(Some(&NSFont::systemFontOfSize(11.0)));
        tf.setTextColor(Some(&NSColor::secondaryLabelColor()));
        tf.setFrame(NSRect::new(
            NSPoint::new(14.0, h - 28.0),
            NSSize::new(w - 28.0, 16.0),
        ));
        tf
    };

    let value_tf = unsafe {
        let tf = NSTextField::labelWithString(&NSString::from_str(value), mtm);
        tf.setFont(Some(&NSFont::boldSystemFontOfSize(20.0)));
        tf.setFrame(NSRect::new(
            NSPoint::new(14.0, 12.0),
            NSSize::new(w - 28.0, 28.0),
        ));
        tf
    };

    // Get pointer before addSubview (addSubview retains it)
    let value_ptr = &*value_tf as *const NSTextField as *const std::ffi::c_void;

    unsafe {
        card.addSubview(&title_tf);
        card.addSubview(&value_tf);
    }

    (unsafe { Retained::cast(card) }, Some(value_ptr))
}
