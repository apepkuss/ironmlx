use std::cell::RefCell;
use std::sync::Mutex;

use objc2::rc::Retained;
use objc2::sel;
use objc2::{MainThreadMarker, MainThreadOnly, define_class, msg_send};
use objc2_app_kit::{
    NSApplication, NSApplicationDelegate, NSMenu, NSMenuItem, NSStatusBar, NSStatusItem,
    NSWorkspace,
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
                if let Some(ref mut c) = *cfg {
                    if !crate::welcome::show_welcome(mtm, c) {
                        let app = NSApplication::sharedApplication(mtm);
                        app.terminate(None);
                        return;
                    }
                }
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
        #[unsafe(method(openDashboard:))]
        fn open_dashboard(&self, _sender: &NSMenuItem) {
            if let Some(mtm) = MainThreadMarker::new() {
                crate::dashboard::show_dashboard(mtm);
            }
        }

        #[unsafe(method(openChat:))]
        fn open_chat(&self, _sender: &NSMenuItem) {
            let p = port();
            open_url(&format!("http://localhost:{}/admin", p));
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
    let p = config.port;
    *CONFIG.lock().unwrap() = Some(config);
    *SERVER.lock().unwrap() = Some(ServerManager::new(p));
}

fn setup_status_bar(mtm: MainThreadMarker) {
    let status_bar = unsafe { NSStatusBar::systemStatusBar() };
    let status_item = status_bar.statusItemWithLength(-1.0);

    if let Some(button) = status_item.button(mtm) {
        // Load embedded narwhal icon as template image
        let icon_bytes = include_bytes!("../../assets/menubar-icon@2x.png");
        let ns_data = unsafe { objc2_foundation::NSData::with_bytes(icon_bytes) };
        if let Some(image) = unsafe { objc2_app_kit::NSImage::initWithData(mtm.alloc(), &ns_data) }
        {
            unsafe {
                image.setSize(objc2_foundation::NSSize::new(34.0, 22.0));
                image.setTemplate(true); // adapts to light/dark menubar
            }
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

    let status_text = match status {
        ServerStatus::Stopped => "Status: Stopped",
        ServerStatus::Starting => "Status: Starting...",
        ServerStatus::Running => "Status: Running",
        ServerStatus::Failed => "Status: Failed",
    };

    // Status (disabled, informational)
    let status_item = make_item(mtm, status_text, None, "");
    status_item.setEnabled(false);
    menu.addItem(&status_item);
    menu.addItem(&NSMenuItem::separatorItem(mtm));

    // Dashboard
    let dashboard = make_item(mtm, "Dashboard", Some(sel!(openDashboard:)), "d");
    dashboard.setEnabled(is_running);
    menu.addItem(&dashboard);

    // Chat
    let chat = make_item(mtm, "Chat with ironmlx", Some(sel!(openChat:)), "");
    chat.setEnabled(is_running);
    menu.addItem(&chat);

    menu.addItem(&NSMenuItem::separatorItem(mtm));

    // Start / Stop
    if is_running {
        menu.addItem(&make_item(mtm, "Stop Server", Some(sel!(stopServer:)), ""));
    } else {
        menu.addItem(&make_item(
            mtm,
            "Start Server",
            Some(sel!(startServer:)),
            "",
        ));
    }

    // Restart
    let restart = make_item(mtm, "Restart Server", Some(sel!(restartServer:)), "");
    restart.setEnabled(is_running);
    menu.addItem(&restart);

    menu.addItem(&NSMenuItem::separatorItem(mtm));

    // Preferences
    menu.addItem(&make_item(
        mtm,
        "Preferences...",
        Some(sel!(openPreferences:)),
        ",",
    ));

    // Check for Updates
    menu.addItem(&make_item(
        mtm,
        "Check for Updates...",
        Some(sel!(checkForUpdates:)),
        "",
    ));

    menu.addItem(&NSMenuItem::separatorItem(mtm));

    // Quit
    menu.addItem(&make_item(mtm, "Quit ironmlx", Some(sel!(quitApp:)), "q"));

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
