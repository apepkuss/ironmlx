use objc2::MainThreadMarker;
use objc2_app_kit::{NSAlert, NSAlertFirstButtonReturn, NSAlertStyle, NSTextField, NSView};
use objc2_foundation::{NSPoint, NSRect, NSSize, NSString};

use crate::config::AppConfig;

/// Show first-launch welcome dialog.
/// Returns true if user completed setup, false if cancelled.
pub fn show_welcome(mtm: MainThreadMarker, config: &mut AppConfig) -> bool {
    // Step 1: Ask for model ID
    let alert = NSAlert::new(mtm);
    alert.setMessageText(&NSString::from_str("Welcome to ironmlx!"));
    alert.setInformativeText(&NSString::from_str(
        "ironmlx is a local LLM inference engine for Apple Silicon.\n\n\
         Enter a HuggingFace model ID to get started.\n\
         Example: mlx-community/Qwen3-0.6B-4bit",
    ));
    alert.setAlertStyle(NSAlertStyle::Informational);

    // Create accessory view with text field for model ID
    let view = unsafe {
        NSView::initWithFrame(
            mtm.alloc::<NSView>(),
            NSRect::new(NSPoint::new(0.0, 0.0), NSSize::new(300.0, 30.0)),
        )
    };

    let model_field = unsafe {
        NSTextField::textFieldWithString(&NSString::from_str("mlx-community/Qwen3-0.6B-4bit"), mtm)
    };
    unsafe {
        model_field.setFrame(NSRect::new(
            NSPoint::new(0.0, 0.0),
            NSSize::new(300.0, 24.0),
        ));
        model_field.setEditable(true);
        model_field.setBezeled(true);
    }
    unsafe {
        view.addSubview(&model_field);
    }

    alert.setAccessoryView(Some(&view));
    alert.addButtonWithTitle(&NSString::from_str("Start"));
    alert.addButtonWithTitle(&NSString::from_str("Quit"));

    let response = alert.runModal();

    if response == NSAlertFirstButtonReturn {
        let model_id = model_field.stringValue().to_string();
        if model_id.is_empty() {
            return false;
        }
        config.last_model = Some(model_id);
        config.save();
        true
    } else {
        false
    }
}

/// Check if this is the first launch (no config file exists)
pub fn is_first_launch() -> bool {
    !AppConfig::config_path().exists()
}
