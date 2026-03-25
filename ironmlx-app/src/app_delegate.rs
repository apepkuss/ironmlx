use std::cell::RefCell;
use std::sync::Mutex;

use objc2::rc::Retained;
use objc2::runtime::AnyClass;
use objc2::sel;
use objc2::{MainThreadMarker, MainThreadOnly, define_class, msg_send};
use objc2_app_kit::{
    NSApplication, NSApplicationDelegate, NSColor, NSImage, NSMenu, NSMenuItem, NSStatusBar,
    NSStatusItem, NSWorkspace,
};
use objc2_foundation::{NSNotification, NSObject, NSObjectProtocol, NSString, NSURL, ns_string};

use crate::config::AppConfig;
use crate::server_manager::{ServerManager, ServerStatus};

// Global state
pub static SERVER: Mutex<Option<ServerManager>> = Mutex::new(None);
static CONFIG: Mutex<Option<AppConfig>> = Mutex::new(None);

thread_local! {
    static STATUS_ITEM: RefCell<Option<Retained<NSStatusItem>>> = const { RefCell::new(None) };
    static MENU_HANDLER: RefCell<Option<Retained<MenuHandler>>> = const { RefCell::new(None) };
}

#[allow(dead_code)]
fn port() -> u16 {
    CONFIG
        .lock()
        .unwrap()
        .as_ref()
        .map(|c| c.port)
        .unwrap_or(8080)
}

// ---------------------------------------------------------------------------
// AppDelegate
// ---------------------------------------------------------------------------

define_class!(
    #[unsafe(super(NSObject))]
    #[thread_kind = MainThreadOnly]
    #[name = "AppDelegate"]
    #[ivars = ()]
    pub struct AppDelegate;

    unsafe impl NSObjectProtocol for AppDelegate {}

    unsafe impl NSApplicationDelegate for AppDelegate {
        #[unsafe(method(applicationDidFinishLaunching:))]
        fn app_did_finish_launching(&self, _notif: &NSNotification) {
            setup_global_state();

            let mtm = MainThreadMarker::from(self);

            // First launch check
            if crate::welcome::is_first_launch() {
                let mut cfg = CONFIG.lock().unwrap();
                if let Some(ref mut c) = *cfg
                    && !crate::welcome::show_welcome(mtm, c)
                {
                    let app = NSApplication::sharedApplication(mtm);
                    app.terminate(None);
                    return;
                }
            }

            // Restore saved language before building menu
            {
                let config = crate::config::AppConfig::load();
                let lang: &'static str = match config.language.as_str() {
                    "zh" | "zh-Hans" => "zh",
                    "zh-Hant" => "zh-Hant",
                    "ja" => "ja",
                    "ko" => "ko",
                    _ => "en",
                };
                *crate::i18n::nav_language().lock().unwrap() = lang;
            }

            setup_status_bar(mtm);

            // Auto-start server if configured
            let should_start = CONFIG
                .lock()
                .unwrap()
                .as_ref()
                .map(|c| c.auto_start && c.last_model.is_some())
                .unwrap_or(false);

            if should_start {
                let model = CONFIG
                    .lock()
                    .unwrap()
                    .as_ref()
                    .and_then(|c| c.last_model.clone())
                    .unwrap_or_default();
                if let Some(ref srv) = *SERVER.lock().unwrap() {
                    let _ = srv.start(&model);
                }
                // Refresh menu to show running state
                refresh_menu(mtm);
            } else {
                // No model configured — show onboarding via Web Dashboard
                let has_model = CONFIG
                    .lock()
                    .unwrap()
                    .as_ref()
                    .map(|c| c.last_model.is_some())
                    .unwrap_or(false);
                if !has_model {
                    crate::web_dashboard::show_web_dashboard(mtm);
                }
            }
        }

        #[unsafe(method(applicationWillTerminate:))]
        fn app_will_terminate(&self, _notif: &NSNotification) {
            if let Some(ref srv) = *SERVER.lock().unwrap() {
                srv.stop();
            }
        }

        #[unsafe(method(applicationShouldTerminateAfterLastWindowClosed:))]
        fn should_terminate_after_last_window_closed(&self, _sender: &NSApplication) -> bool {
            false // menubar app should keep running
        }
    }
);

impl AppDelegate {
    pub fn new(mtm: MainThreadMarker) -> Retained<Self> {
        unsafe { msg_send![mtm.alloc::<Self>(), init] }
    }
}

// ---------------------------------------------------------------------------
// MenuHandler — receives menu item actions via selectors
// ---------------------------------------------------------------------------

define_class!(
    #[unsafe(super(NSObject))]
    #[thread_kind = MainThreadOnly]
    #[name = "MenuHandler"]
    #[ivars = ()]
    pub struct MenuHandler;

    unsafe impl NSObjectProtocol for MenuHandler {}

    impl MenuHandler {
        #[unsafe(method(openWebDashboard:))]
        fn open_web_dashboard(&self, _sender: &NSMenuItem) {
            if let Some(mtm) = MainThreadMarker::new() {
                crate::web_dashboard::show_web_dashboard(mtm);
            }
        }

        #[unsafe(method(openChat:))]
        fn open_chat(&self, _sender: &NSMenuItem) {
            if std::path::Path::new("/Applications/Moss.app").exists() {
                std::thread::spawn(|| {
                    let output = std::process::Command::new("/usr/bin/open")
                        .arg("-a")
                        .arg("/Applications/Moss.app")
                        .env_clear()
                        .env("HOME", std::env::var("HOME").unwrap_or_default())
                        .env("PATH", "/usr/bin:/bin:/usr/sbin:/sbin")
                        .output();
                    if let Err(e) = output {
                        eprintln!("[menu] openChat: failed to open Moss: {e}");
                    }
                });
            }
        }

        #[unsafe(method(startServer:))]
        fn start_server(&self, _sender: &NSMenuItem) {
            let model = CONFIG
                .lock()
                .unwrap()
                .as_ref()
                .and_then(|c| c.last_model.clone())
                .unwrap_or_default();
            if model.is_empty() {
                return;
            }
            if let Some(ref srv) = *SERVER.lock().unwrap() {
                let _ = srv.start(&model);
            }
            let mtm = MainThreadMarker::from(self);
            refresh_menu(mtm);
        }

        #[unsafe(method(stopServer:))]
        fn stop_server(&self, _sender: &NSMenuItem) {
            if let Some(ref srv) = *SERVER.lock().unwrap() {
                srv.stop();
            }
            let mtm = MainThreadMarker::from(self);
            refresh_menu(mtm);
        }

        #[unsafe(method(restartServer:))]
        fn restart_server(&self, _sender: &NSMenuItem) {
            let model = CONFIG
                .lock()
                .unwrap()
                .as_ref()
                .and_then(|c| c.last_model.clone())
                .unwrap_or_default();
            if let Some(ref srv) = *SERVER.lock().unwrap() {
                let _ = srv.restart(&model);
            }
            let mtm = MainThreadMarker::from(self);
            refresh_menu(mtm);
        }

        #[unsafe(method(openPreferences:))]
        fn open_preferences(&self, _sender: &NSMenuItem) {
            let mtm = MainThreadMarker::from(self);
            let mut cfg = CONFIG.lock().unwrap();
            if let Some(ref mut c) = *cfg {
                crate::preferences::show_preferences(mtm, c);
            }
        }

        #[unsafe(method(checkForUpdates:))]
        fn check_for_updates(&self, _sender: &NSMenuItem) {
            let mtm = MainThreadMarker::from(self);
            match crate::updater::check_for_update() {
                Some(info) => crate::updater::show_update_alert(mtm, &info),
                None => crate::updater::show_no_update_alert(mtm),
            }
        }

        #[unsafe(method(quitApp:))]
        fn quit_app(&self, _sender: &NSMenuItem) {
            if let Some(ref srv) = *SERVER.lock().unwrap() {
                srv.stop();
            }
            let mtm = MainThreadMarker::from(self);
            let app = NSApplication::sharedApplication(mtm);
            app.terminate(None);
        }
    }
);

impl MenuHandler {
    pub fn new(mtm: MainThreadMarker) -> Retained<Self> {
        unsafe { msg_send![mtm.alloc::<Self>(), init] }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn setup_global_state() {
    let config = AppConfig::load();
    let h = config.host.clone();
    let p = config.port;
    let ml = config.memory_limit_total;
    let hc = config.hot_cache_gb;
    let cc = config.cold_cache_gb;
    let ms = config.max_sequences;
    let icb = config.init_cache_blocks;
    let ce = config.cache_enabled;
    let cd = config.cache_dir.clone();
    *CONFIG.lock().unwrap() = Some(config);
    *SERVER.lock().unwrap() = Some(ServerManager::new(&h, p, ml, hc, cc, ms, icb, ce, &cd));
}

fn setup_status_bar(mtm: MainThreadMarker) {
    let status_bar = NSStatusBar::systemStatusBar();
    let status_item = status_bar.statusItemWithLength(-1.0);

    if let Some(button) = status_item.button(mtm) {
        // Load embedded narwhal icon as template image
        let icon_bytes = include_bytes!("../../assets/menubar-icon@2x.png");
        let ns_data = objc2_foundation::NSData::with_bytes(icon_bytes);
        if let Some(image) = objc2_app_kit::NSImage::initWithData(mtm.alloc(), &ns_data) {
            image.setSize(objc2_foundation::NSSize::new(34.0, 22.0));
            image.setTemplate(true); // adapts to light/dark menubar
            button.setImage(Some(&image));
        } else {
            // Fallback to emoji if image fails
            button.setTitle(ns_string!("\u{26A1}"));
        }
    }

    // Create menu handler
    let handler = MenuHandler::new(mtm);
    MENU_HANDLER.with(|mh| {
        *mh.borrow_mut() = Some(handler);
    });

    let menu = build_menu(mtm);
    status_item.setMenu(Some(&menu));

    STATUS_ITEM.with(|si| {
        *si.borrow_mut() = Some(status_item);
    });
}

fn build_menu(mtm: MainThreadMarker) -> Retained<NSMenu> {
    let menu = NSMenu::new(mtm);

    let status = SERVER
        .lock()
        .unwrap()
        .as_ref()
        .map(|s| s.status())
        .unwrap_or(ServerStatus::Stopped);

    let is_running = status == ServerStatus::Running;

    use crate::i18n::t;

    // Helper: create SF Symbol image for menu items
    let sf_icon = |name: &str| -> Option<Retained<NSImage>> {
        let ns_name = NSString::from_str(name);
        NSImage::imageWithSystemSymbolName_accessibilityDescription(&ns_name, None)
    };

    // ── Status row ──
    let status_label = if is_running {
        t("menu_status_running")
    } else {
        t("menu_status_stopped")
    };
    let status_item = unsafe {
        let item = NSMenuItem::initWithTitle_action_keyEquivalent(
            mtm.alloc::<NSMenuItem>(),
            &NSString::from_str(status_label),
            None,
            &NSString::from_str(""),
        );

        // Green/red circle.fill icon with tint
        if let Some(icon) = sf_icon("circle.fill") {
            let tint_color = if is_running {
                NSColor::systemGreenColor()
            } else {
                NSColor::systemRedColor()
            };
            let config: Retained<objc2::runtime::AnyObject> = msg_send![
                AnyClass::get(c"NSImageSymbolConfiguration").unwrap(),
                configurationWithHierarchicalColor: &*tint_color
            ];
            let tinted: Retained<NSImage> = msg_send![&*icon,
                imageWithSymbolConfiguration: &*config
            ];
            item.setImage(Some(&tinted));
        }

        // Green/red attributed title text
        let color = if is_running {
            NSColor::systemGreenColor()
        } else {
            NSColor::systemRedColor()
        };
        let fg_key: &NSString = ns_string!("NSColor");
        let attrs: Retained<objc2::runtime::AnyObject> = msg_send![
            AnyClass::get(c"NSDictionary").unwrap(),
            dictionaryWithObject: &*color,
            forKey: fg_key
        ];
        let title_ns = NSString::from_str(status_label);
        let attr_str: Retained<objc2::runtime::AnyObject> = msg_send![
            msg_send![AnyClass::get(c"NSAttributedString").unwrap(), alloc],
            initWithString: &*title_ns,
            attributes: &*attrs
        ];
        let _: () = msg_send![&*item, setAttributedTitle: &*attr_str];

        item.setEnabled(false);
        item
    };
    menu.addItem(&status_item);

    // ── Model name (gray, secondary) ──
    let model_name = CONFIG
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|c| c.last_model.clone())
        .unwrap_or_else(|| "\u{2014}".to_string());
    let model_item = make_item(mtm, &model_name, None, "");
    if let Some(icon) = sf_icon("cube") {
        model_item.setImage(Some(&icon));
    }
    model_item.setEnabled(false);
    menu.addItem(&model_item);

    menu.addItem(&NSMenuItem::separatorItem(mtm));

    // ── Dashboard ──
    let dashboard = make_item(mtm, t("menu_dashboard"), Some(sel!(openWebDashboard:)), "d");
    if let Some(icon) = sf_icon("square.grid.2x2") {
        dashboard.setImage(Some(&icon));
    }
    dashboard.setEnabled(is_running);
    menu.addItem(&dashboard);

    // ── Chat with Moss ──
    let moss_installed = std::path::Path::new("/Applications/Moss.app").exists();
    let chat = make_item(mtm, t("menu_chat"), Some(sel!(openChat:)), "");
    if let Some(icon) = sf_icon("bubble.left.and.bubble.right") {
        chat.setImage(Some(&icon));
    }
    chat.setEnabled(moss_installed);
    menu.addItem(&chat);

    menu.addItem(&NSMenuItem::separatorItem(mtm));

    // ── Start / Stop ──
    if is_running {
        let stop = make_item(mtm, t("menu_stop"), Some(sel!(stopServer:)), "");
        if let Some(icon) = sf_icon("stop.fill") {
            stop.setImage(Some(&icon));
        }
        menu.addItem(&stop);
    } else {
        let start = make_item(mtm, t("menu_start"), Some(sel!(startServer:)), "");
        if let Some(icon) = sf_icon("play.fill") {
            start.setImage(Some(&icon));
        }
        menu.addItem(&start);
    }

    // ── Restart ──
    let restart = make_item(mtm, t("menu_restart"), Some(sel!(restartServer:)), "");
    if let Some(icon) = sf_icon("arrow.clockwise") {
        restart.setImage(Some(&icon));
    }
    restart.setEnabled(is_running);
    menu.addItem(&restart);

    menu.addItem(&NSMenuItem::separatorItem(mtm));

    // ── Check for Updates ──
    let updates = make_item(mtm, t("menu_updates"), Some(sel!(checkForUpdates:)), "");
    if let Some(icon) = sf_icon("arrow.down.circle") {
        updates.setImage(Some(&icon));
    }
    menu.addItem(&updates);

    menu.addItem(&NSMenuItem::separatorItem(mtm));

    // ── Quit ──
    let quit = make_item(mtm, t("menu_quit"), Some(sel!(quitApp:)), "q");
    if let Some(icon) = sf_icon("power") {
        quit.setImage(Some(&icon));
    }
    menu.addItem(&quit);

    menu
}

fn make_item(
    mtm: MainThreadMarker,
    title: &str,
    action: Option<objc2::runtime::Sel>,
    key: &str,
) -> Retained<NSMenuItem> {
    let ns_title = NSString::from_str(title);
    let ns_key = NSString::from_str(key);
    let item = unsafe {
        NSMenuItem::initWithTitle_action_keyEquivalent(
            mtm.alloc::<NSMenuItem>(),
            &ns_title,
            action,
            &ns_key,
        )
    };
    // Set target to our MenuHandler
    MENU_HANDLER.with(|mh| {
        if let Some(ref handler) = *mh.borrow() {
            unsafe { item.setTarget(Some(handler)) };
        }
    });
    item
}

pub fn open_url(url_str: &str) {
    let ns_url_str = NSString::from_str(url_str);
    if let Some(url) = NSURL::URLWithString(&ns_url_str) {
        NSWorkspace::sharedWorkspace().openURL(&url);
    }
}

pub fn refresh_menu(mtm: MainThreadMarker) {
    STATUS_ITEM.with(|si| {
        if let Some(ref status_item) = *si.borrow() {
            let menu = build_menu(mtm);
            status_item.setMenu(Some(&menu));
        }
    });
}

/// Restart the backend server (callable from web_dashboard bridge)
/// Reloads config from disk so host/port changes take effect.
pub fn restart_server() {
    // Reload config from disk
    let fresh_config = crate::config::AppConfig::load();
    let model = fresh_config.last_model.clone().unwrap_or_default();
    let new_host = fresh_config.host.clone();
    let new_port = fresh_config.port;
    let new_mem_limit = fresh_config.memory_limit_total;
    let new_hc = fresh_config.hot_cache_gb;
    let new_cc = fresh_config.cold_cache_gb;
    let new_ms = fresh_config.max_sequences;
    let new_icb = fresh_config.init_cache_blocks;
    let new_ce = fresh_config.cache_enabled;
    let new_cd = fresh_config.cache_dir.clone();

    // Update in-memory CONFIG
    *CONFIG.lock().unwrap() = Some(fresh_config);

    // Update all settings and restart
    if let Some(ref mut srv) = *SERVER.lock().unwrap() {
        srv.set_host(&new_host);
        srv.set_port(new_port);
        srv.set_memory_limit_total(new_mem_limit);
        srv.set_hot_cache_gb(new_hc);
        srv.set_cold_cache_gb(new_cc);
        srv.set_max_sequences(new_ms);
        srv.set_init_cache_blocks(new_icb);
        srv.set_cache_enabled(new_ce);
        srv.set_cache_dir(&new_cd);
        let _ = srv.restart(&model);
    }
}
