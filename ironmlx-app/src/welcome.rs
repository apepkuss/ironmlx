use objc2::MainThreadMarker;
use objc2_app_kit::{
    NSAlert, NSAlertFirstButtonReturn, NSAlertSecondButtonReturn, NSAlertStyle, NSModalResponseOK,
    NSOpenPanel,
};
use objc2_foundation::NSString;

use crate::config::AppConfig;

/// Show first-launch welcome dialog.
/// Returns true if user completed setup, false if cancelled.
pub fn show_welcome(mtm: MainThreadMarker, config: &mut AppConfig) -> bool {
    let alert = NSAlert::new(mtm);
    alert.setMessageText(&NSString::from_str("Welcome to ironmlx!"));
    alert.setInformativeText(&NSString::from_str(
        "ironmlx is a local LLM inference engine for Apple Silicon.\n\n\
         To get started, please select a model directory or enter a HuggingFace model ID.",
    ));
    alert.setAlertStyle(NSAlertStyle::Informational);
    alert.addButtonWithTitle(&NSString::from_str("Choose Model Directory"));
    alert.addButtonWithTitle(&NSString::from_str("Use Default (HuggingFace)"));
    alert.addButtonWithTitle(&NSString::from_str("Quit"));

    let response = alert.runModal();

    if response == NSAlertFirstButtonReturn {
        // Open folder picker
        let panel = NSOpenPanel::openPanel(mtm);
        panel.setCanChooseDirectories(true);
        panel.setCanChooseFiles(false);
        panel.setAllowsMultipleSelection(false);
        panel.setMessage(Some(&NSString::from_str("Select model directory")));

        let result = panel.runModal();
        if result == NSModalResponseOK {
            if let Some(url) = panel.URL() {
                if let Some(path) = url.path() {
                    config.model_dir = Some(path.to_string());
                    config.save();
                    return true;
                }
            }
        }
        false
    } else if response == NSAlertSecondButtonReturn {
        // Use default HF cache
        config.model_dir = Some("~/.cache/huggingface/hub".to_string());
        config.save();
        true
    } else {
        // Quit
        false
    }
}

/// Check if this is the first launch (no config file exists)
pub fn is_first_launch() -> bool {
    !AppConfig::config_path().exists()
}
