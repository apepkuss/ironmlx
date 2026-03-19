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
    ("\u{1F4DD}", "Logs"),
    ("\u{26A1}", "Benchmark"),
    ("\u{2699}\u{FE0F}", "Settings"),
];

// Status card value labels — stored for polling updates
// Keys: "status", "active_mem", "peak_mem", "models", "uptime", "default_model"
static STATUS_LABELS: OnceLock<Mutex<std::collections::HashMap<&'static str, RawPtr>>> =
    OnceLock::new();
fn status_labels_lock() -> &'static Mutex<std::collections::HashMap<&'static str, RawPtr>> {
    STATUS_LABELS.get_or_init(|| Mutex::new(std::collections::HashMap::new()))
}

// Memory history for chart (last 60 samples = 5 minutes at 5s interval)
const MEM_HISTORY_LEN: usize = 300; // 1s interval × 300 = 5 minutes
static MEM_HISTORY: OnceLock<Mutex<(Vec<f64>, Vec<f64>)>> = OnceLock::new();
fn mem_history_lock() -> &'static Mutex<(Vec<f64>, Vec<f64>)> {
    MEM_HISTORY.get_or_init(|| Mutex::new((Vec::new(), Vec::new())))
}
// Chart view pointer for triggering redraws
static CHART_VIEW_PTR: OnceLock<Mutex<Option<RawPtr>>> = OnceLock::new();

// ---------------------------------------------------------------------------
// MemoryChartView — custom NSView with drawRect for real-time line chart
// ---------------------------------------------------------------------------

static CHART_VIEW_CLASS: OnceLock<&'static AnyClass> = OnceLock::new();

extern "C" fn chart_draw_rect(_this: *mut AnyObject, _sel: Sel, _dirty_rect: NSRect) {
    let history = mem_history_lock().lock().unwrap();
    let (active, peak) = &*history;

    let this_view: &NSView = unsafe { &*(_this as *const NSView) };
    let bounds = this_view.bounds();
    let w = bounds.size.width;
    let h = bounds.size.height;
    let pad_x = 8.0;
    let pad_bottom = 12.0;
    let pad_top = 8.0;
    let chart_w = w - pad_x * 2.0;
    let chart_h = h - pad_bottom - pad_top;

    let max_val = active
        .iter()
        .chain(peak.iter())
        .cloned()
        .fold(100.0f64, f64::max)
        * 1.3; // 30% headroom
    let y_scale = chart_h / max_val;
    let dx = chart_w / (MEM_HISTORY_LEN as f64 - 1.0);

    let get_points = |data: &[f64]| -> Vec<(f64, f64)> {
        (0..MEM_HISTORY_LEN)
            .map(|i| {
                let di = i as isize - (MEM_HISTORY_LEN as isize - data.len() as isize);
                let v = if di >= 0 && (di as usize) < data.len() {
                    data[di as usize]
                } else {
                    0.0
                };
                (pad_x + i as f64 * dx, pad_bottom + v * y_scale)
            })
            .collect()
    };

    let active_pts = get_points(active);
    let _peak = peak; // used for horizontal reference line

    unsafe {
        // Background — match content area
        NSColor::colorWithSRGBRed_green_blue_alpha(0.925, 0.925, 0.925, 1.0).set();
        NSBezierPath::bezierPathWithRoundedRect_xRadius_yRadius(bounds, 6.0, 6.0).fill();

        // Horizontal grid lines (4 lines)
        NSColor::colorWithSRGBRed_green_blue_alpha(0.88, 0.88, 0.88, 1.0).set();
        for i in 1..=4 {
            let gy = pad_bottom + (chart_h * i as f64 / 5.0);
            let grid = NSBezierPath::new();
            grid.moveToPoint(NSPoint::new(pad_x, gy));
            grid.lineToPoint(NSPoint::new(w - pad_x, gy));
            grid.setLineWidth(0.5);
            grid.stroke();
        }

        // Bottom baseline
        NSColor::colorWithSRGBRed_green_blue_alpha(0.85, 0.85, 0.85, 1.0).set();
        let baseline = NSBezierPath::new();
        baseline.moveToPoint(NSPoint::new(pad_x, pad_bottom));
        baseline.lineToPoint(NSPoint::new(w - pad_x, pad_bottom));
        baseline.setLineWidth(0.5);
        baseline.stroke();

        // Active memory filled area
        let fill_path = NSBezierPath::new();
        fill_path.moveToPoint(NSPoint::new(active_pts[0].0, pad_bottom));
        for &(x, y) in &active_pts {
            fill_path.lineToPoint(NSPoint::new(x, y));
        }
        fill_path.lineToPoint(NSPoint::new(active_pts.last().unwrap().0, pad_bottom));
        fill_path.closePath();
        NSColor::colorWithSRGBRed_green_blue_alpha(0.22, 0.56, 0.96, 0.12).set();
        fill_path.fill();

        // Active memory line
        let line = NSBezierPath::new();
        for (i, &(x, y)) in active_pts.iter().enumerate() {
            if i == 0 {
                line.moveToPoint(NSPoint::new(x, y));
            } else {
                line.lineToPoint(NSPoint::new(x, y));
            }
        }
        NSColor::colorWithSRGBRed_green_blue_alpha(0.22, 0.56, 0.96, 1.0).set();
        line.setLineWidth(2.0);
        line.stroke();

        // Peak memory — horizontal reference line at peak value
        let peak_val = peak.iter().cloned().fold(0.0f64, f64::max);
        if peak_val > 0.0 {
            let peak_y = pad_bottom + peak_val * y_scale;
            let peak_line = NSBezierPath::new();
            peak_line.moveToPoint(NSPoint::new(pad_x, peak_y));
            peak_line.lineToPoint(NSPoint::new(w - pad_x, peak_y));
            NSColor::colorWithSRGBRed_green_blue_alpha(0.96, 0.56, 0.22, 0.6).set();
            peak_line.setLineWidth(1.0);
            let pattern: [f64; 2] = [4.0, 3.0];
            let _: () =
                msg_send![&*peak_line, setLineDash: pattern.as_ptr() count: 2i64 phase: 0.0f64];
            peak_line.stroke();
        }
    }
}

fn chart_view_class() -> &'static AnyClass {
    CHART_VIEW_CLASS.get_or_init(|| {
        let superclass = AnyClass::get(c"NSView").unwrap();
        let mut builder = ClassBuilder::new(c"IronMemoryChart", superclass).unwrap();

        unsafe {
            builder.add_method(
                sel!(drawRect:),
                chart_draw_rect as extern "C" fn(*mut AnyObject, Sel, NSRect),
            );
        }

        builder.register()
    })
}

// Sidebar status labels
static SIDEBAR_MODEL_LABEL: OnceLock<Mutex<Option<RawPtr>>> = OnceLock::new();
static SIDEBAR_MEM_LABEL: OnceLock<Mutex<Option<RawPtr>>> = OnceLock::new();

/// Start polling /health API every 5 seconds to update Status page
pub fn start_status_polling(port: u16) {
    let url = format!("http://localhost:{}/health", port);
    std::thread::spawn(move || {
        loop {
            std::thread::sleep(std::time::Duration::from_secs(1));
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

        // Record memory history for chart
        if let Ok(mut hist) = mem_history_lock().lock() {
            // If peak increased since last sample, insert the peak value
            // as a data point before the current active — this captures
            // the transient spike that the 1s polling missed
            let prev_peak = hist.1.last().cloned().unwrap_or(0.0);
            if peak > prev_peak && peak > active {
                hist.0.push(peak); // insert peak spike
                hist.1.push(peak);
                if hist.0.len() > MEM_HISTORY_LEN {
                    hist.0.remove(0);
                }
                if hist.1.len() > MEM_HISTORY_LEN {
                    hist.1.remove(0);
                }
            }
            hist.0.push(active);
            hist.1.push(peak);
            if hist.0.len() > MEM_HISTORY_LEN {
                hist.0.remove(0);
            }
            if hist.1.len() > MEM_HISTORY_LEN {
                hist.1.remove(0);
            }
        }

        // Trigger chart redraw via dispatch to main queue
        if let Ok(guard) = CHART_VIEW_PTR.get_or_init(|| Mutex::new(None)).lock() {
            if let Some(ptr) = guard.as_ref() {
                let raw = ptr.0 as usize;
                dispatch2::Queue::main().exec_async(move || {
                    let chart: &NSView = unsafe { &*(raw as *const NSView) };
                    unsafe {
                        chart.setNeedsDisplay(true);
                    }
                });
            }
        }
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

// ---------------------------------------------------------------------------
// SettingsHandler — handles settings page button actions
// ---------------------------------------------------------------------------

// Store Host/Port text field pointers for reading values
static SETTINGS_HOST_FIELD: OnceLock<Mutex<Option<RawPtr>>> = OnceLock::new();
static SETTINGS_PORT_FIELD: OnceLock<Mutex<Option<RawPtr>>> = OnceLock::new();

static SETTINGS_HANDLER_CLASS: OnceLock<&'static AnyClass> = OnceLock::new();

// Tag constants for settings buttons
const TAG_SAVE_RESTART: isize = 100;
const TAG_LANG_EN: isize = 110;
const TAG_LANG_ZH: isize = 111;
const TAG_THEME_SYSTEM: isize = 120;
const TAG_THEME_LIGHT: isize = 121;
const TAG_THEME_DARK: isize = 122;

extern "C" fn settings_action(_this: *mut AnyObject, _sel: Sel, sender: *mut AnyObject) {
    let tag: isize = unsafe { msg_send![sender, tag] };
    match tag {
        TAG_SAVE_RESTART => save_and_restart(),
        TAG_LANG_EN => set_language("en"),
        TAG_LANG_ZH => set_language("zh"),
        TAG_THEME_SYSTEM => set_theme(None),
        TAG_THEME_LIGHT => set_theme(Some("NSAppearanceNameAqua")),
        TAG_THEME_DARK => set_theme(Some("NSAppearanceNameDarkAqua")),
        _ => {}
    }
}

fn settings_handler_class() -> &'static AnyClass {
    SETTINGS_HANDLER_CLASS.get_or_init(|| {
        let superclass = AnyClass::get(c"NSObject").unwrap();
        let mut builder = ClassBuilder::new(c"IronSettingsHandler", superclass).unwrap();
        unsafe {
            builder.add_method(
                sel!(settingsAction:),
                settings_action as extern "C" fn(*mut AnyObject, Sel, *mut AnyObject),
            );
        }
        builder.register()
    })
}

fn save_and_restart() {
    // Read Host/Port values
    let host = if let Ok(guard) = SETTINGS_HOST_FIELD.get_or_init(|| Mutex::new(None)).lock() {
        guard.as_ref().map(|ptr| {
            let tf: &NSTextField = unsafe { &*(ptr.0 as *const NSTextField) };
            tf.stringValue().to_string()
        })
    } else {
        None
    };
    let port = if let Ok(guard) = SETTINGS_PORT_FIELD.get_or_init(|| Mutex::new(None)).lock() {
        guard.as_ref().map(|ptr| {
            let tf: &NSTextField = unsafe { &*(ptr.0 as *const NSTextField) };
            tf.stringValue().to_string()
        })
    } else {
        None
    };

    // Save to config
    let mut config = crate::config::AppConfig::load();
    if let Some(h) = host {
        // host is stored in server config, not app config — for now just log
        eprintln!("[settings] Host: {}", h);
    }
    if let Some(p) = &port {
        if let Ok(port_num) = p.parse::<u16>() {
            config.port = port_num;
        }
    }
    config.save();

    // Restart server
    eprintln!("[settings] Restarting server on port {}...", config.port);
    let mut server = crate::app_delegate::SERVER.lock().unwrap();
    if let Some(ref mut mgr) = *server {
        mgr.stop();
        if let Some(model) = &config.last_model {
            let _ = mgr.start(model);
        }
    }
}

// Navigation label translations
static NAV_LANGUAGE: OnceLock<Mutex<&'static str>> = OnceLock::new();
fn nav_language() -> &'static Mutex<&'static str> {
    NAV_LANGUAGE.get_or_init(|| Mutex::new("en"))
}

const NAV_LABELS_EN: &[&str] = &["Status", "Models", "Chat", "Logs", "Benchmark", "Settings"];
const NAV_LABELS_ZH: &[&str] = &[
    "\u{72B6}\u{6001}", // 状态
    "\u{6A21}\u{578B}", // 模型
    "\u{5BF9}\u{8BDD}", // 对话
    "\u{65E5}\u{5FD7}", // 日志
    "\u{57FA}\u{51C6}", // 基准
    "\u{8BBE}\u{7F6E}", // 设置
];

/// Get translated string for current language
fn t(key: &str) -> &str {
    let lang = *nav_language().lock().unwrap();
    match (lang, key) {
        // Status page
        ("zh", "status") => "\u{72B6}\u{6001}",
        ("zh", "server") => "\u{670D}\u{52A1}\u{5668}",
        ("zh", "uptime") => "\u{8FD0}\u{884C}\u{65F6}\u{95F4}",
        ("zh", "current_model") => "\u{5F53}\u{524D}\u{6A21}\u{578B}",
        ("zh", "gpu_memory") => "GPU \u{5185}\u{5B58}",
        ("zh", "active") => "\u{6D3B}\u{8DC3}:",
        ("zh", "peak") => "\u{5CF0}\u{503C}:",
        ("zh", "gpu_hint") => {
            "GPU \u{5185}\u{5B58}\u{5728}\u{9996}\u{6B21}\u{63A8}\u{7406}\u{65F6}\u{5206}\u{914D}"
        }
        // Settings page
        ("zh", "settings") => "\u{8BBE}\u{7F6E}",
        ("zh", "model") => "\u{6A21}\u{578B}",
        ("zh", "appearance") => "\u{5916}\u{89C2}",
        ("zh", "host") => "\u{4E3B}\u{673A}",
        ("zh", "port") => "\u{7AEF}\u{53E3}",
        ("zh", "temperature") => "Temperature",
        ("zh", "top_p") => "Top P",
        ("zh", "max_tokens") => "Max Tokens",
        ("zh", "endpoint") => "\u{670D}\u{52A1}\u{7AEF}\u{70B9}",
        ("zh", "language") => "\u{8BED}\u{8A00}",
        ("zh", "theme") => "\u{4E3B}\u{9898}",
        ("zh", "save_restart") => "\u{4FDD}\u{5B58}\u{5E76}\u{91CD}\u{542F}",
        ("zh", "theme_system") => "\u{1F5A5} \u{8DDF}\u{968F}\u{7CFB}\u{7EDF}",
        ("zh", "theme_light") => "\u{2600} \u{6D45}\u{8272}",
        ("zh", "theme_dark") => "\u{1F319} \u{6DF1}\u{8272}",
        // English defaults
        (_, "status") => "Status",
        (_, "server") => "Server",
        (_, "uptime") => "Uptime",
        (_, "current_model") => "Current Model",
        (_, "gpu_memory") => "GPU Memory",
        (_, "active") => "Active:",
        (_, "peak") => "Peak:",
        (_, "gpu_hint") => "GPU memory is allocated on first inference",
        (_, "settings") => "Settings",
        (_, "model") => "Model",
        (_, "appearance") => "Appearance",
        (_, "host") => "Host",
        (_, "port") => "Port",
        (_, "temperature") => "Temperature",
        (_, "top_p") => "Top P",
        (_, "max_tokens") => "Max Tokens",
        (_, "endpoint") => "Endpoint",
        (_, "language") => "Language",
        (_, "theme") => "Theme",
        (_, "save_restart") => "Save & Restart",
        (_, "theme_system") => "\u{1F5A5} System",
        (_, "theme_light") => "\u{2600} Light",
        (_, "theme_dark") => "\u{1F319} Dark",
        _ => key,
    }
}

fn set_language(lang: &'static str) {
    *nav_language().lock().unwrap() = lang;

    // Update nav buttons
    let labels = if lang == "zh" {
        NAV_LABELS_ZH
    } else {
        NAV_LABELS_EN
    };
    if let Ok(buttons) = NAV_BUTTON_PTRS
        .get_or_init(|| Mutex::new(Vec::new()))
        .lock()
    {
        for (i, ptr) in buttons.iter().enumerate() {
            if i < labels.len() {
                let btn: &NSButton = unsafe { &*(ptr.0 as *const NSButton) };
                let icon = NAV_ITEMS[i].0;
                let title = format!("  {}  {}", icon, labels[i]);
                unsafe {
                    btn.setTitle(&NSString::from_str(&title));
                }
            }
        }
    }

    // Rebuild Status page (index 0) and Settings page (index 5)
    let mtm = unsafe { MainThreadMarker::new_unchecked() };
    let pages = pages_lock().lock().unwrap();
    for &idx in &[0usize, 5] {
        if idx < pages.len() {
            let page_view: &NSView = unsafe { &*(pages[idx].0 as *const NSView) };
            let frame = page_view.frame();
            let was_hidden = page_view.isHidden();

            // Remove all subviews
            unsafe {
                let subs = page_view.subviews();
                for i in (0..subs.len()).rev() {
                    let sub = subs.objectAtIndex(i);
                    sub.removeFromSuperview();
                }
            }

            // Rebuild content into same view
            let w = frame.size.width;
            let h = frame.size.height;
            let new_page = if idx == 0 {
                build_status_page(mtm, w, h)
            } else {
                build_settings_page(mtm, w, h)
            };

            // Copy subviews from new page to existing page
            unsafe {
                let new_subs = new_page.subviews();
                for i in 0..new_subs.len() {
                    let sub = new_subs.objectAtIndex(i);
                    page_view.addSubview(&sub);
                }
                page_view.setHidden(was_hidden);
            }
        }
    }
}

// Store nav button pointers for language switching
static NAV_BUTTON_PTRS: OnceLock<Mutex<Vec<RawPtr>>> = OnceLock::new();

// ---------------------------------------------------------------------------
// WindowDelegate — handles window close
// ---------------------------------------------------------------------------

static WIN_DELEGATE_CLASS: OnceLock<&'static AnyClass> = OnceLock::new();
static WIN_DELEGATE_INSTANCE: OnceLock<Mutex<Option<RawPtr>>> = OnceLock::new();

extern "C" fn window_should_close(
    _this: *mut AnyObject,
    _sel: Sel,
    _sender: *mut AnyObject,
) -> objc2::runtime::Bool {
    eprintln!("[dashboard] windowShouldClose called");
    objc2::runtime::Bool::YES
}

extern "C" fn window_will_close(_this: *mut AnyObject, _sel: Sel, _notif: *mut AnyObject) {
    eprintln!("[dashboard] windowWillClose called");
    // Switch back to accessory mode (hide from Dock)
    unsafe {
        let app: *mut AnyObject =
            msg_send![AnyClass::get(c"NSApplication").unwrap(), sharedApplication];
        let _: () = msg_send![app, setActivationPolicy: 1i64];
    }
}

fn window_delegate_class() -> &'static AnyClass {
    WIN_DELEGATE_CLASS.get_or_init(|| {
        let superclass = AnyClass::get(c"NSObject").unwrap();
        let mut builder = ClassBuilder::new(c"IronWindowDelegate", superclass).unwrap();
        unsafe {
            builder.add_method(
                sel!(windowShouldClose:),
                window_should_close
                    as extern "C" fn(*mut AnyObject, Sel, *mut AnyObject) -> objc2::runtime::Bool,
            );
            builder.add_method(
                sel!(windowWillClose:),
                window_will_close as extern "C" fn(*mut AnyObject, Sel, *mut AnyObject),
            );
        }
        builder.register()
    })
}

fn window_delegate_instance() -> &'static Mutex<Option<RawPtr>> {
    let lock = WIN_DELEGATE_INSTANCE.get_or_init(|| Mutex::new(None));
    let mut guard = lock.lock().unwrap();
    if guard.is_none() {
        let cls = window_delegate_class();
        let obj: *mut AnyObject = unsafe {
            let obj: *mut AnyObject = msg_send![cls, alloc];
            msg_send![obj, init]
        };
        *guard = Some(RawPtr(obj as *const std::ffi::c_void));
    }
    drop(guard);
    lock
}

fn set_theme(appearance_name: Option<&str>) {
    let guard = window_lock().lock().unwrap();
    if let Some(ref ptr) = *guard {
        let window: &NSWindow = unsafe { &*(ptr.0 as *const NSWindow) };
        unsafe {
            match appearance_name {
                Some(name) => {
                    let ns_name = NSString::from_str(name);
                    let appearance: *mut AnyObject = msg_send![
                        AnyClass::get(c"NSAppearance").unwrap(),
                        appearanceNamed: &*ns_name
                    ];
                    let _: () = msg_send![window, setAppearance: appearance];
                }
                None => {
                    // System default — set appearance to nil
                    let _: () = msg_send![window, setAppearance: std::ptr::null::<AnyObject>()];
                }
            }
        }
    }
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
        if window.isVisible() {
            window.makeKeyAndOrderFront(None);
        } else {
            window.makeKeyAndOrderFront(None);
        }
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

    window.setTitle(ns_string!("IRONMLX"));
    window.setMinSize(NSSize::new(700.0, 450.0));
    window.center();
    unsafe {
        window.setReleasedWhenClosed(false);
    }

    // Set window delegate for close handling
    {
        let guard = window_delegate_instance().lock().unwrap();
        if let Some(ref ptr) = *guard {
            unsafe {
                let _: () = msg_send![&*window, setDelegate: ptr.0 as *const AnyObject];
            }
        }
    }

    // Hide native title text, use custom centered label instead
    window.setTitleVisibility(NSWindowTitleVisibility::Hidden);

    let content = build_content(mtm);
    window.setContentView(Some(&content));

    // Add centered title label on themeFrame — offset past traffic light buttons
    unsafe {
        if let Some(content_view) = window.contentView() {
            if let Some(theme_frame) = content_view.superview() {
                let frame = theme_frame.frame();
                let title_label = NSTextField::labelWithString(ns_string!("IRONMLX"), mtm);
                title_label.setFont(Some(&NSFont::titleBarFontOfSize(13.0)));
                title_label.setTextColor(Some(&NSColor::windowFrameTextColor()));
                title_label.setAlignment(NSTextAlignment::Center);
                // x=80 to avoid covering close/minimize/zoom buttons
                title_label.setFrame(NSRect::new(
                    NSPoint::new(80.0, frame.size.height - 24.0),
                    NSSize::new(frame.size.width - 160.0, 16.0),
                ));
                title_label.setAutoresizingMask(NSAutoresizingMaskOptions(2 | 8));
                theme_frame.addSubview(&title_label);
            }
        }
    }

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

    // Pages container — content area background
    let pages_container = unsafe {
        let v = NSView::initWithFrame(
            mtm.alloc(),
            NSRect::new(NSPoint::new(sw, 0.0), NSSize::new(cw, h)),
        );
        v.setAutoresizesSubviews(true);
        v.setAutoresizingMask(NSAutoresizingMaskOptions(2 | 16));
        // Content area: rgb(236,236,236)
        v.setWantsLayer(true);
        if let Some(layer) = v.layer() {
            let bg = NSColor::colorWithSRGBRed_green_blue_alpha(0.925, 0.925, 0.925, 1.0);
            let cg: *const std::ffi::c_void = msg_send![&bg, CGColor];
            let _: () = msg_send![&*layer, setBackgroundColor: cg];
        }
        v
    };

    // Vertical separator line between sidebar and content
    let separator = unsafe {
        let v = NSView::initWithFrame(
            mtm.alloc(),
            NSRect::new(NSPoint::new(sw - 0.5, 0.0), NSSize::new(0.5, h)),
        );
        v.setWantsLayer(true);
        if let Some(layer) = v.layer() {
            let bg = NSColor::separatorColor();
            let cg: *const std::ffi::c_void = msg_send![&bg, CGColor];
            let _: () = msg_send![&*layer, setBackgroundColor: cg];
        }
        v.setAutoresizingMask(NSAutoresizingMaskOptions(16)); // height sizable
        v
    };

    // Build all pages
    let page_builders: Vec<fn(MainThreadMarker, f64, f64) -> Retained<NSView>> = vec![
        build_status_page,
        build_models_page,
        build_chat_page,
        build_logs_page,
        build_benchmark_page,
        build_settings_page,
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
        container.addSubview(&separator);
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
    // Sidebar: rgb(245,245,245)
    unsafe {
        sidebar.setWantsLayer(true);
        if let Some(layer) = sidebar.layer() {
            let bg = NSColor::colorWithSRGBRed_green_blue_alpha(0.961, 0.961, 0.961, 1.0);
            let cg: *const std::ffi::c_void = msg_send![&bg, CGColor];
            let _: () = msg_send![&*layer, setBackgroundColor: cg];
        }
    }

    // Header — logo + IronMLX
    let header_y = height - 28.0 - 60.0;
    let header = build_sidebar_header(mtm, 16.0, header_y, width - 32.0);
    unsafe {
        sidebar.addSubview(&header);
    }

    // Nav items — below header
    let handler_guard = nav_handler_lock().lock().unwrap();
    let handler_ptr = handler_guard.as_ref().unwrap().0;
    let action = sel!(navClicked:);

    let mut highlight_ptrs = Vec::new();
    let mut nav_btn_ptrs = Vec::new();
    let mut y = header_y - 50.0;
    for (i, (icon, label)) in NAV_ITEMS.iter().enumerate() {
        // Highlight background view (behind button)
        let highlight_bg = unsafe {
            let v = NSView::initWithFrame(
                mtm.alloc(),
                NSRect::new(NSPoint::new(12.0, y), NSSize::new(width - 24.0, 38.0)),
            );
            v.setWantsLayer(true);
            if let Some(layer) = v.layer() {
                let _: () = msg_send![&*layer, setCornerRadius: 8.0f64];
                // Light blue highlight like Clash Verge
                let color = NSColor::colorWithSRGBRed_green_blue_alpha(0.86, 0.91, 1.0, 1.0);
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
        y -= 42.0;
        let btn_ptr = &*btn as *const NSButton as *const std::ffi::c_void;
        nav_btn_ptrs.push(RawPtr(btn_ptr));
        unsafe {
            sidebar.addSubview(&highlight_bg);
            sidebar.addSubview(&btn);
        }
    }
    *nav_highlights_lock().lock().unwrap() = highlight_ptrs;
    *NAV_BUTTON_PTRS
        .get_or_init(|| Mutex::new(Vec::new()))
        .lock()
        .unwrap() = nav_btn_ptrs;

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
            NSRect::new(NSPoint::new(x, y), NSSize::new(width, 79.0)),
        )
    };

    let icon_bytes = include_bytes!("../../assets/sidebar-logo@2x.png");
    let ns_data = unsafe { NSData::with_bytes(icon_bytes) };
    if let Some(image) = unsafe { NSImage::initWithData(mtm.alloc(), &ns_data) } {
        unsafe {
            image.setSize(NSSize::new(128.0, 60.0));
        }
        let iv = unsafe {
            let iv = NSImageView::initWithFrame(
                mtm.alloc(),
                NSRect::new(NSPoint::new(0.0, 17.0), NSSize::new(128.0, 60.0)),
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
        tf.setFont(Some(&NSFont::boldSystemFontOfSize(13.0)));
        tf.setTextColor(Some(&NSColor::secondaryLabelColor()));
        tf.setFrame(NSRect::new(
            NSPoint::new(84.0, 17.0),
            NSSize::new(90.0, 16.0),
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
    let title = format!("  {}  {}", icon, label); // extra space between icon and text
    let button = unsafe {
        let btn = NSButton::initWithFrame(
            mtm.alloc(),
            NSRect::new(NSPoint::new(x, y), NSSize::new(width, 38.0)),
        );
        btn.setTitle(&NSString::from_str(&title));
        btn.setFont(Some(&NSFont::systemFontOfSize(14.0)));
        btn.setAlignment(NSTextAlignment::Left);
        btn.setBordered(false);
        btn.setTag(tag as isize);
        btn.setBezelStyle(NSBezelStyle(0));
        // Dark gray text via contentTintColor
        btn.setContentTintColor(Some(&NSColor::colorWithSRGBRed_green_blue_alpha(
            0.3, 0.3, 0.33, 1.0,
        )));

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
            NSRect::new(NSPoint::new(x, y), NSSize::new(width, 30.0)),
        )
    };

    let version = env!("CARGO_PKG_VERSION");
    let version_tf = unsafe {
        let tf =
            NSTextField::labelWithString(&NSString::from_str(&format!("Version {}", version)), mtm);
        tf.setFont(Some(&NSFont::systemFontOfSize(11.0)));
        tf.setTextColor(Some(&NSColor::tertiaryLabelColor()));
        tf.setAlignment(NSTextAlignment::Center);
        tf.setFrame(NSRect::new(
            NSPoint::new(0.0, 6.0),
            NSSize::new(width, 16.0),
        ));
        tf
    };

    unsafe {
        view.addSubview(&version_tf);
    }

    view
}

// ---------------------------------------------------------------------------
// Page builders
// ---------------------------------------------------------------------------

fn build_status_page(mtm: MainThreadMarker, width: f64, height: f64) -> Retained<NSView> {
    let view = make_page_view(mtm, width, height);
    let title = make_title(mtm, t("status"), height);

    // === Layout constants ===
    // Title bar is 90pt, title text centered in it
    // Title bottom edge ~ height - 90
    let section_gap = 24.0;
    let card_h = 80.0;
    let gap = 12.0;

    // === Row 1: 3 cards ===
    let cards_top = height - 90.0 - section_gap;
    let card_y = cards_top - card_h;
    let narrow_w = 130.0;
    let wide_w = width - 24.0 * 2.0 - gap * 2.0 - narrow_w * 2.0;

    let card1 = build_status_card(
        mtm,
        t("server"),
        "Running",
        "status",
        24.0,
        card_y,
        narrow_w,
        card_h,
    );
    let card2 = build_status_card(
        mtm,
        t("uptime"),
        "\u{2014}",
        "uptime",
        24.0 + narrow_w + gap,
        card_y,
        narrow_w,
        card_h,
    );
    let card3 = build_status_card(
        mtm,
        t("current_model"),
        "\u{2014}",
        "default_model",
        24.0 + (narrow_w + gap) * 2.0,
        card_y,
        wide_w,
        card_h,
    );

    // === Separator between cards and GPU Memory ===
    let sep_y = card_y - 24.0;
    let separator = unsafe {
        let v = NSView::initWithFrame(
            mtm.alloc(),
            NSRect::new(NSPoint::new(24.0, sep_y), NSSize::new(width - 48.0, 0.5)),
        );
        v.setWantsLayer(true);
        if let Some(layer) = v.layer() {
            let bg = NSColor::separatorColor();
            let cg: *const std::ffi::c_void = msg_send![&bg, CGColor];
            let _: () = msg_send![&*layer, setBackgroundColor: cg];
        }
        v
    };

    // === GPU Memory header row ===
    let mem_row_y = sep_y - 24.0 - 18.0; // 24pt gap + 18pt label height
    let mem_title = make_label(mtm, t("gpu_memory"), 24.0, mem_row_y, 120.0, true);

    let active_dot = unsafe {
        let tf = NSTextField::labelWithString(ns_string!("\u{2014}"), mtm);
        tf.setFont(Some(&NSFont::boldSystemFontOfSize(13.0)));
        tf.setTextColor(Some(&NSColor::colorWithSRGBRed_green_blue_alpha(
            0.22, 0.56, 0.96, 1.0,
        )));
        tf.setFrame(NSRect::new(
            NSPoint::new(144.0, mem_row_y),
            NSSize::new(18.0, 18.0),
        ));
        tf
    };
    let active_lbl = make_label(mtm, t("active"), 164.0, mem_row_y, 55.0, false);
    let active_val = build_status_label(mtm, "0 MB", "active_mem", 220.0, mem_row_y, 100.0);

    let peak_dot = unsafe {
        let tf = NSTextField::labelWithString(ns_string!("\u{2014}"), mtm);
        tf.setFont(Some(&NSFont::boldSystemFontOfSize(13.0)));
        tf.setTextColor(Some(&NSColor::colorWithSRGBRed_green_blue_alpha(
            0.96, 0.56, 0.22, 1.0,
        )));
        tf.setFrame(NSRect::new(
            NSPoint::new(338.0, mem_row_y),
            NSSize::new(18.0, 18.0),
        ));
        tf
    };
    let peak_lbl = make_label(mtm, t("peak"), 358.0, mem_row_y, 45.0, false);
    let peak_val = build_status_label(mtm, "0 MB", "peak_mem", 404.0, mem_row_y, 100.0);

    // === Memory chart ===
    let chart_top = mem_row_y - 20.0;
    let chart_bottom = 24.0;
    let chart_h_val = (chart_top - chart_bottom).max(40.0);
    let chart_w = width - 48.0;

    let cls = chart_view_class();
    let chart: Retained<NSView> = unsafe {
        let obj: *mut AnyObject = msg_send![cls, alloc];
        let obj: *mut AnyObject = msg_send![obj, initWithFrame: NSRect::new(
            NSPoint::new(24.0, chart_bottom),
            NSSize::new(chart_w, chart_h_val),
        )];
        Retained::from_raw(obj as *mut NSView).unwrap()
    };

    *CHART_VIEW_PTR
        .get_or_init(|| Mutex::new(None))
        .lock()
        .unwrap() = Some(RawPtr(&*chart as *const NSView as *const std::ffi::c_void));

    // Hint text below chart
    let hint = unsafe {
        let tf = NSTextField::labelWithString(&NSString::from_str(t("gpu_hint")), mtm);
        tf.setFont(Some(&NSFont::systemFontOfSize(10.0)));
        tf.setTextColor(Some(&NSColor::tertiaryLabelColor()));
        tf.setFrame(NSRect::new(
            NSPoint::new(24.0, 8.0),
            NSSize::new(width - 48.0, 14.0),
        ));
        tf
    };

    unsafe {
        view.addSubview(&title);
        view.addSubview(&card1);
        view.addSubview(&card2);
        view.addSubview(&card3);
        view.addSubview(&separator);
        view.addSubview(&mem_title);
        view.addSubview(&active_dot);
        view.addSubview(&active_lbl);
        view.addSubview(&active_val);
        view.addSubview(&peak_dot);
        view.addSubview(&peak_lbl);
        view.addSubview(&peak_val);
        view.addSubview(&chart);
        view.addSubview(&hint);
    }

    view
}

fn build_models_page(mtm: MainThreadMarker, width: f64, height: f64) -> Retained<NSView> {
    let view = make_page_view(mtm, width, height);

    let title = make_title(mtm, "Models", height);

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
        view.addSubview(&placeholder);
    }

    view
}

fn build_settings_card(
    mtm: MainThreadMarker,
    section_title: &str,
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
        v.setCornerRadius(8.0);
        v.setBorderWidth(0.5);
        v.setBorderColor(&NSColor::separatorColor());
        v.setFillColor(&NSColor::controlBackgroundColor());
        v.setTitlePosition(unsafe { std::mem::transmute(0u64) });
        v
    };

    // Section title inside card
    let header = unsafe {
        let tf = NSTextField::labelWithString(&NSString::from_str(section_title), mtm);
        tf.setFont(Some(&NSFont::boldSystemFontOfSize(13.0)));
        tf.setFrame(NSRect::new(
            NSPoint::new(16.0, h - 34.0),
            NSSize::new(w - 32.0, 18.0),
        ));
        tf
    };
    unsafe {
        card.addSubview(&header);
    }

    unsafe { Retained::cast(card) }
}

fn build_settings_page(mtm: MainThreadMarker, width: f64, height: f64) -> Retained<NSView> {
    let view = make_page_view(mtm, width, height);
    let title = make_title(mtm, t("settings"), height);

    let pad = 16.0;
    let row_h = 32.0;
    let card_gap = 16.0;
    let scroll_top = height - 90.0;

    // Scrollable content area below title
    let scroll = unsafe {
        let sv = NSScrollView::initWithFrame(
            mtm.alloc(),
            NSRect::new(NSPoint::ZERO, NSSize::new(width, scroll_top)),
        );
        sv.setHasVerticalScroller(true);
        sv.setBorderType(NSBorderType(0));
        sv.setDrawsBackground(false);
        sv.setAutoresizingMask(NSAutoresizingMaskOptions(2 | 16));
        sv
    };

    let card_w = width - 48.0;
    let label_w = 110.0;
    let field_w = card_w - label_w - 48.0;

    // Card heights (generous)
    let c1_h = 130.0;
    let c2_h = 140.0;
    let c3_h = 90.0;
    let c4_h = 120.0;

    let total_h: f64 =
        card_gap + c1_h + card_gap + c2_h + card_gap + c3_h + card_gap + c4_h + card_gap;
    let content_h = total_h.max(scroll_top);

    let doc_view = unsafe {
        NSView::initWithFrame(
            mtm.alloc(),
            NSRect::new(NSPoint::ZERO, NSSize::new(width, content_h)),
        )
    };
    unsafe {
        scroll.setDocumentView(Some(&doc_view));
    }

    // Helper: add field inside a card (relative to card origin)
    let add_card_field = |card: &NSView,
                          mtm: MainThreadMarker,
                          label: &str,
                          value: &str,
                          readonly: bool,
                          fy: f64| {
        let lbl = unsafe {
            let tf = NSTextField::labelWithString(&NSString::from_str(label), mtm);
            tf.setFont(Some(&NSFont::systemFontOfSize(13.0)));
            tf.setFrame(NSRect::new(
                NSPoint::new(pad, fy),
                NSSize::new(label_w, 18.0),
            ));
            tf
        };
        let field = unsafe {
            let tf = NSTextField::initWithFrame(
                mtm.alloc(),
                NSRect::new(
                    NSPoint::new(pad + label_w + 8.0, fy - 2.0),
                    NSSize::new(field_w, 22.0),
                ),
            );
            tf.setStringValue(&NSString::from_str(value));
            tf.setFont(Some(&NSFont::systemFontOfSize(13.0)));
            tf.setBezeled(true);
            tf.setEditable(!readonly);
            if readonly {
                tf.setTextColor(Some(&NSColor::secondaryLabelColor()));
            }
            tf
        };
        unsafe {
            card.addSubview(&lbl);
            card.addSubview(&field);
        }
    };

    // Layout from top of doc_view (top = content_h)
    let mut y = content_h - card_gap;

    // Create settings handler
    let settings_cls = settings_handler_class();
    let settings_handler: Retained<AnyObject> = unsafe {
        let obj: *mut AnyObject = msg_send![settings_cls, alloc];
        let obj: *mut AnyObject = msg_send![obj, init];
        Retained::from_raw(obj).unwrap()
    };
    let sh_ptr = &*settings_handler as *const AnyObject;
    let settings_action_sel = sel!(settingsAction:);

    // === Card 1: Server ===
    y -= c1_h;
    let card1 = build_settings_card(mtm, t("server"), 24.0, y, card_w, c1_h);
    // Button right edge aligns with input field right edge
    let field_right = pad + label_w + 8.0 + field_w;
    let btn_w = 100.0;
    let btn_y = c1_h - 38.0; // aligned with header centerline
    let save_restart_btn = make_button(mtm, t("save_restart"), field_right - btn_w, btn_y, btn_w);
    unsafe {
        // Blue button style without keyEquivalent (avoid intercepting window close)
        save_restart_btn.setBezelColor(Some(&NSColor::colorWithSRGBRed_green_blue_alpha(
            0.0, 0.478, 1.0, 1.0,
        )));
        save_restart_btn.setTag(TAG_SAVE_RESTART);
        save_restart_btn.setTarget(Some(&*(sh_ptr as *const NSObject)));
        save_restart_btn.setAction(Some(settings_action_sel));
        card1.addSubview(&save_restart_btn);
    }
    // Host field — store pointer
    let host_fy = c1_h - 46.0 - row_h;
    add_card_field(&card1, mtm, t("host"), "127.0.0.1", false, host_fy);
    // Find the field we just added (last subview)
    unsafe {
        let subs = card1.subviews();
        let last = subs.objectAtIndex(subs.len() - 1);
        let ptr = &*last as *const NSView as *const std::ffi::c_void;
        *SETTINGS_HOST_FIELD
            .get_or_init(|| Mutex::new(None))
            .lock()
            .unwrap() = Some(RawPtr(ptr));
    }
    // Port field — store pointer
    let port_fy = c1_h - 46.0 - row_h * 2.0;
    add_card_field(&card1, mtm, t("port"), "8080", false, port_fy);
    unsafe {
        let subs = card1.subviews();
        let last = subs.objectAtIndex(subs.len() - 1);
        let ptr = &*last as *const NSView as *const std::ffi::c_void;
        *SETTINGS_PORT_FIELD
            .get_or_init(|| Mutex::new(None))
            .lock()
            .unwrap() = Some(RawPtr(ptr));
    }

    // === Card 2: Model ===
    y -= card_gap;
    y -= c2_h;
    let card2 = build_settings_card(mtm, t("model"), 24.0, y, card_w, c2_h);
    add_card_field(
        &card2,
        mtm,
        t("temperature"),
        "1.0",
        false,
        c2_h - 40.0 - row_h,
    );
    add_card_field(
        &card2,
        mtm,
        t("top_p"),
        "1.0",
        false,
        c2_h - 40.0 - row_h * 2.0,
    );
    add_card_field(
        &card2,
        mtm,
        t("max_tokens"),
        "2048",
        false,
        c2_h - 40.0 - row_h * 3.0,
    );

    // === Card 3: HuggingFace ===
    y -= card_gap;
    y -= c3_h;
    let card3 = build_settings_card(mtm, "HuggingFace", 24.0, y, card_w, c3_h);
    add_card_field(
        &card3,
        mtm,
        t("endpoint"),
        "https://huggingface.co",
        false,
        c3_h - 40.0 - row_h,
    );

    // === Card 4: Appearance ===
    y -= card_gap;
    y -= c4_h;
    let card4 = build_settings_card(mtm, t("appearance"), 24.0, y, card_w, c4_h);

    // Language row
    let lang_y = c4_h - 40.0 - row_h;
    let lang_lbl = unsafe {
        let tf = NSTextField::labelWithString(&NSString::from_str(t("language")), mtm);
        tf.setFont(Some(&NSFont::systemFontOfSize(13.0)));
        tf.setFrame(NSRect::new(
            NSPoint::new(pad, lang_y),
            NSSize::new(label_w, 18.0),
        ));
        tf
    };
    unsafe {
        card4.addSubview(&lang_lbl);
    }
    let lang_btns = [("English", TAG_LANG_EN), ("\u{4E2D}\u{6587}", TAG_LANG_ZH)];
    let mut lx = pad + label_w + 8.0;
    for (label, tag) in &lang_btns {
        let btn = make_button(mtm, label, lx, lang_y - 2.0, 70.0);
        unsafe {
            btn.setTag(*tag);
            btn.setTarget(Some(&*(sh_ptr as *const NSObject)));
            btn.setAction(Some(settings_action_sel));
            card4.addSubview(&btn);
        }
        lx += 78.0;
    }

    // Theme row
    let theme_y = c4_h - 40.0 - row_h * 2.0;
    let theme_lbl = unsafe {
        let tf = NSTextField::labelWithString(&NSString::from_str(t("theme")), mtm);
        tf.setFont(Some(&NSFont::systemFontOfSize(13.0)));
        tf.setFrame(NSRect::new(
            NSPoint::new(pad, theme_y),
            NSSize::new(label_w, 18.0),
        ));
        tf
    };
    unsafe {
        card4.addSubview(&theme_lbl);
    }
    let theme_btns = [
        (t("theme_system"), TAG_THEME_SYSTEM),
        (t("theme_light"), TAG_THEME_LIGHT),
        (t("theme_dark"), TAG_THEME_DARK),
    ];
    let mut tx = pad + label_w + 8.0;
    for (label, tag) in &theme_btns {
        let btn = make_button(mtm, label, tx, theme_y - 2.0, 80.0);
        unsafe {
            btn.setTag(*tag);
            btn.setTarget(Some(&*(sh_ptr as *const NSObject)));
            btn.setAction(Some(settings_action_sel));
            card4.addSubview(&btn);
        }
        tx += 88.0;
    }

    // Keep settings handler alive
    std::mem::forget(settings_handler);

    unsafe {
        doc_view.addSubview(&card1);
        doc_view.addSubview(&card2);
        doc_view.addSubview(&card3);
        doc_view.addSubview(&card4);
        // Scroll to top
        let top_point = NSPoint::new(0.0, content_h);
        doc_view.scrollPoint(top_point);
        let scroll_view: Retained<NSView> = Retained::cast(scroll);
        view.addSubview(&title);
        view.addSubview(&scroll_view);
    }

    view
}

fn build_logs_page(mtm: MainThreadMarker, width: f64, height: f64) -> Retained<NSView> {
    let view = make_page_view(mtm, width, height);

    let title = make_title(mtm, "Logs", height);

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
        view.addSubview(&scroll_view);
    }

    view
}

fn build_benchmark_page(mtm: MainThreadMarker, width: f64, height: f64) -> Retained<NSView> {
    let view = make_page_view(mtm, width, height);

    let title = make_title(mtm, "Benchmark", height);

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

/// Build a status label registered for polling updates.
fn build_status_label(
    mtm: MainThreadMarker,
    value: &str,
    key: &'static str,
    x: f64,
    y: f64,
    width: f64,
) -> Retained<NSTextField> {
    let tf = unsafe {
        let tf = NSTextField::labelWithString(&NSString::from_str(value), mtm);
        tf.setFont(Some(&NSFont::boldSystemFontOfSize(13.0)));
        tf.setFrame(NSRect::new(NSPoint::new(x, y), NSSize::new(width, 18.0)));
        tf
    };
    let ptr = &*tf as *const NSTextField as *const std::ffi::c_void;
    status_labels_lock()
        .lock()
        .unwrap()
        .insert(key, RawPtr(ptr));
    tf
}

// ---------------------------------------------------------------------------
// Shared UI helpers
// ---------------------------------------------------------------------------

fn make_page_view(mtm: MainThreadMarker, width: f64, height: f64) -> Retained<NSView> {
    let view = unsafe {
        NSView::initWithFrame(
            mtm.alloc(),
            NSRect::new(NSPoint::ZERO, NSSize::new(width, height)),
        )
    };

    // Title bar background — same color as sidebar
    let title_bar_h = 90.0;
    let title_bar = unsafe {
        let v = NSView::initWithFrame(
            mtm.alloc(),
            NSRect::new(
                NSPoint::new(0.0, height - title_bar_h),
                NSSize::new(width, title_bar_h),
            ),
        );
        v.setWantsLayer(true);
        if let Some(layer) = v.layer() {
            // Same color as sidebar: rgb(245,245,245)
            let bg = NSColor::colorWithSRGBRed_green_blue_alpha(0.961, 0.961, 0.961, 1.0);
            let cg: *const std::ffi::c_void = msg_send![&bg, CGColor];
            let _: () = msg_send![&*layer, setBackgroundColor: cg];
        }
        v.setAutoresizingMask(NSAutoresizingMaskOptions(2 | 8)); // width + top
        v
    };

    unsafe {
        view.setAutoresizesSubviews(true);
        view.addSubview(&title_bar);
    }

    view
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
        v.setCornerRadius(8.0);
        v.setBorderWidth(0.5);
        v.setBorderColor(&NSColor::separatorColor());
        // Cards use controlBackgroundColor — lightest layer
        v.setFillColor(&NSColor::controlBackgroundColor());
        v.setTitlePosition(unsafe { std::mem::transmute(0u64) });
        // Add subtle shadow for depth
        v.setWantsLayer(true);
        if let Some(layer) = v.layer() {
            let _: () = msg_send![&*layer, setShadowOpacity: 0.04f32];
            let _: () = msg_send![&*layer, setShadowRadius: 2.0f64];
            let shadow_offset: objc2_core_foundation::CGSize = objc2_core_foundation::CGSize {
                width: 0.0,
                height: -1.0,
            };
            let _: () = msg_send![&*layer, setShadowOffset: shadow_offset];
        }
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
        tf.setFont(Some(&NSFont::systemFontOfSize_weight(18.0, unsafe {
            std::mem::transmute(0.3f64)
        })));
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
