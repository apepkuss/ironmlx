use objc2::MainThreadMarker;
use objc2_app_kit::{NSAlert, NSAlertFirstButtonReturn, NSAlertSecondButtonReturn, NSAlertStyle};
use objc2_foundation::{NSPoint, NSRect, NSSize, NSString};

use crate::config::AppConfig;

pub fn show_preferences(mtm: MainThreadMarker, config: &mut AppConfig) {
    use objc2_app_kit::NSView;

    let alert = NSAlert::new(mtm);
    alert.setMessageText(&NSString::from_str("Preferences"));
    alert.setAlertStyle(NSAlertStyle::Informational);

    // Create an accessory view with text fields
    let view = NSView::initWithFrame(
        mtm.alloc::<NSView>(),
        NSRect::new(NSPoint::new(0.0, 0.0), NSSize::new(300.0, 120.0)),
    );

    // Port field
    let port_label = create_label(mtm, "Port:");
    port_label.setFrame(NSRect::new(
        NSPoint::new(0.0, 90.0),
        NSSize::new(75.0, 22.0),
    ));
    let port_field = create_text_field(mtm, &config.port.to_string(), 80.0, 90.0, 200.0);
    view.addSubview(&port_label);
    view.addSubview(&port_field);

    // Model field
    let model_label = create_label(mtm, "Model:");
    model_label.setFrame(NSRect::new(
        NSPoint::new(0.0, 55.0),
        NSSize::new(75.0, 22.0),
    ));
    let model_field = create_text_field(
        mtm,
        config.last_model.as_deref().unwrap_or(""),
        80.0,
        55.0,
        200.0,
    );
    view.addSubview(&model_label);
    view.addSubview(&model_field);

    // Auto-start label
    let auto_text = format!(
        "Auto-start: {}",
        if config.auto_start { "Yes" } else { "No" }
    );
    let auto_label = create_label(mtm, &auto_text);
    auto_label.setFrame(NSRect::new(
        NSPoint::new(0.0, 20.0),
        NSSize::new(280.0, 22.0),
    ));
    view.addSubview(&auto_label);

    alert.setAccessoryView(Some(&view));
    alert.addButtonWithTitle(&NSString::from_str("Save"));
    alert.addButtonWithTitle(&NSString::from_str("Toggle Auto-start"));
    alert.addButtonWithTitle(&NSString::from_str("Cancel"));

    let response = alert.runModal();

    if response == NSAlertFirstButtonReturn {
        // Save
        let port_str = port_field.stringValue().to_string();
        if let Ok(p) = port_str.parse::<u16>() {
            config.port = p;
        }
        let model_str = model_field.stringValue().to_string();
        if !model_str.is_empty() {
            config.last_model = Some(model_str);
        }
        config.save();
    } else if response == NSAlertSecondButtonReturn {
        // Toggle auto-start
        config.auto_start = !config.auto_start;
        config.save();
        // Show again to reflect change
        show_preferences(mtm, config);
    }
    // Cancel — do nothing
}

fn create_label(
    mtm: MainThreadMarker,
    text: &str,
) -> objc2::rc::Retained<objc2_app_kit::NSTextField> {
    objc2_app_kit::NSTextField::labelWithString(&NSString::from_str(text), mtm)
}

fn create_text_field(
    mtm: MainThreadMarker,
    value: &str,
    x: f64,
    y: f64,
    width: f64,
) -> objc2::rc::Retained<objc2_app_kit::NSTextField> {
    let field = objc2_app_kit::NSTextField::textFieldWithString(&NSString::from_str(value), mtm);
    field.setFrame(NSRect::new(NSPoint::new(x, y), NSSize::new(width, 22.0)));
    field.setEditable(true);
    field.setBezeled(true);
    field
}
