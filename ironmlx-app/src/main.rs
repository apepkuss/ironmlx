use objc2::MainThreadMarker;
use objc2::runtime::ProtocolObject;
use objc2_app_kit::{NSApplication, NSApplicationDelegate};

mod app_delegate;
mod config;
mod server_manager;

fn main() {
    let mtm = MainThreadMarker::new().expect("ironmlx-app must be launched on the main thread");

    let app = NSApplication::sharedApplication(mtm);

    let delegate = app_delegate::AppDelegate::new(mtm);
    let proto = ProtocolObject::from_retained(delegate);
    app.setDelegate(Some(&proto));

    unsafe { app.run() };
}
