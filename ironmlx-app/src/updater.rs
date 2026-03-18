use serde::Deserialize;

const CURRENT_VERSION: &str = env!("CARGO_PKG_VERSION");
const GITHUB_REPO: &str = "apepkuss/ironmlx";

#[derive(Debug, Deserialize)]
struct GitHubRelease {
    tag_name: String,
    html_url: String,
    name: String,
}

#[derive(Debug)]
pub struct UpdateInfo {
    pub current: String,
    pub latest: String,
    pub url: String,
    pub name: String,
}

/// Check GitHub for newer version. Returns Some(UpdateInfo) if update available.
pub fn check_for_update() -> Option<UpdateInfo> {
    let url = format!(
        "https://api.github.com/repos/{}/releases/latest",
        GITHUB_REPO
    );

    let output = std::process::Command::new("curl")
        .args(["-s", "-H", "User-Agent: ironmlx", &url])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let body = String::from_utf8_lossy(&output.stdout);
    let release: GitHubRelease = serde_json::from_str(&body).ok()?;

    let latest = release.tag_name.trim_start_matches('v').to_string();

    if is_newer(&latest, CURRENT_VERSION) {
        Some(UpdateInfo {
            current: CURRENT_VERSION.to_string(),
            latest,
            url: release.html_url,
            name: release.name,
        })
    } else {
        None
    }
}

fn is_newer(latest: &str, current: &str) -> bool {
    let parse = |v: &str| -> Vec<u32> { v.split('.').filter_map(|s| s.parse().ok()).collect() };
    let l = parse(latest);
    let c = parse(current);
    l > c
}

/// Show update notification using NSAlert
pub fn show_update_alert(mtm: objc2::MainThreadMarker, info: &UpdateInfo) {
    use objc2_app_kit::{NSAlert, NSAlertFirstButtonReturn, NSAlertStyle};
    use objc2_foundation::NSString;

    let alert = NSAlert::new(mtm);
    alert.setMessageText(&NSString::from_str("Update Available"));
    alert.setInformativeText(&NSString::from_str(&format!(
        "{}\n\nCurrent: v{}\nLatest: v{}",
        info.name, info.current, info.latest
    )));
    alert.setAlertStyle(NSAlertStyle::Informational);
    alert.addButtonWithTitle(&NSString::from_str("Open Download Page"));
    alert.addButtonWithTitle(&NSString::from_str("Later"));

    let response = alert.runModal();
    if response == NSAlertFirstButtonReturn {
        crate::app_delegate::open_url(&info.url);
    }
}

/// Show a "no update" message when user manually checks
pub fn show_no_update_alert(mtm: objc2::MainThreadMarker) {
    use objc2_app_kit::{NSAlert, NSAlertStyle};
    use objc2_foundation::NSString;

    let alert = NSAlert::new(mtm);
    alert.setMessageText(&NSString::from_str("No Update Available"));
    alert.setInformativeText(&NSString::from_str(&format!(
        "You are running the latest version (v{}).",
        CURRENT_VERSION
    )));
    alert.setAlertStyle(NSAlertStyle::Informational);
    alert.addButtonWithTitle(&NSString::from_str("OK"));
    alert.runModal();
}
