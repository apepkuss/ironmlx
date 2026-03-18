use std::cell::RefCell;
use std::sync::Mutex;

use objc2::rc::Retained;
use objc2::{MainThreadMarker, MainThreadOnly, define_class, msg_send};
use objc2_app_kit::{
    NSApplication, NSApplicationDelegate, NSMenu, NSMenuItem, NSStatusBar, NSStatusItem,
    NSWorkspace,
};
use objc2_foundation::{NSNotification, NSObject, NSObjectProtocol, NSString, NSURL, ns_string};

use crate::config::AppConfig;
use crate::server_manager::{ServerManager, ServerStatus};

// Global state for server and config (Send + Sync safe)
static SERVER: Mutex<Option<ServerManager>> = Mutex::new(None);
static CONFIG: Mutex<Option<AppConfig>> = Mutex::new(None);

// Main-thread-only state stored in thread_local
thread_local! {
    static STATUS_ITEM: RefCell<Option<Retained<NSStatusItem>>> = const { RefCell::new(None) };
}

fn port() -> u16 {
    CONFIG
        .lock()
        .unwrap()
        .as_ref()
        .map(|c| c.port)
        .unwrap_or(8080)
}

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
            }
        }

        #[unsafe(method(applicationWillTerminate:))]
        fn app_will_terminate(&self, _notif: &NSNotification) {
            if let Some(ref srv) = *SERVER.lock().unwrap() {
                srv.stop();
            }
        }
    }
);

impl AppDelegate {
    pub fn new(mtm: MainThreadMarker) -> Retained<Self> {
        unsafe { msg_send![mtm.alloc::<Self>(), init] }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn setup_global_state() {
    let config = AppConfig::load();
    let p = config.port;
    *CONFIG.lock().unwrap() = Some(config);
    *SERVER.lock().unwrap() = Some(ServerManager::new(p));
}

fn setup_status_bar(mtm: MainThreadMarker) {
    let status_bar = unsafe { NSStatusBar::systemStatusBar() };
    let status_item = status_bar.statusItemWithLength(-1.0);

    if let Some(button) = status_item.button(mtm) {
        button.setTitle(ns_string!("\u{26A1}"));
    }

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

    let status_text = match status {
        ServerStatus::Stopped => "Status: Stopped",
        ServerStatus::Starting => "Status: Starting...",
        ServerStatus::Running => "Status: Running",
        ServerStatus::Failed => "Status: Failed",
    };

    // Status (disabled)
    let status_item = make_item(mtm, status_text, "");
    status_item.setEnabled(false);
    menu.addItem(&status_item);
    menu.addItem(&NSMenuItem::separatorItem(mtm));

    // Dashboard
    let dashboard = make_item(mtm, "Dashboard", "d");
    dashboard.setEnabled(is_running);
    menu.addItem(&dashboard);

    // Chat
    let chat = make_item(mtm, "Chat with ironmlx", "");
    chat.setEnabled(is_running);
    menu.addItem(&chat);

    menu.addItem(&NSMenuItem::separatorItem(mtm));

    // Start / Stop
    if is_running {
        menu.addItem(&make_item(mtm, "Stop Server", ""));
    } else {
        menu.addItem(&make_item(mtm, "Start Server", ""));
    }

    // Restart
    let restart = make_item(mtm, "Restart Server", "");
    restart.setEnabled(is_running);
    menu.addItem(&restart);

    menu.addItem(&NSMenuItem::separatorItem(mtm));

    // Quit
    menu.addItem(&make_item(mtm, "Quit ironmlx", "q"));

    menu
}

fn make_item(mtm: MainThreadMarker, title: &str, key: &str) -> Retained<NSMenuItem> {
    let ns_title = NSString::from_str(title);
    let ns_key = NSString::from_str(key);
    unsafe {
        NSMenuItem::initWithTitle_action_keyEquivalent(
            mtm.alloc::<NSMenuItem>(),
            &ns_title,
            None,
            &ns_key,
        )
    }
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
