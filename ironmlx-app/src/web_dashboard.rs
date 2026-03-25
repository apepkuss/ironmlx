//! Web-based dashboard window using WKWebView.
//!
//! Opens a native NSWindow with an embedded WKWebView that loads a local HTML file.
//! Supports JS→Rust communication via WKScriptMessageHandler.

use std::sync::{Mutex, OnceLock};

use objc2::rc::Retained;
use objc2::runtime::AnyObject;
use objc2::{MainThreadMarker, MainThreadOnly, define_class, msg_send, sel};
use objc2_app_kit::*;
use objc2_foundation::*;
use objc2_web_kit::{
    WKScriptMessage, WKScriptMessageHandler, WKUIDelegate, WKWebView, WKWebViewConfiguration,
};

// ---------------------------------------------------------------------------
// Singleton window
// ---------------------------------------------------------------------------

struct RawPtr(*const std::ffi::c_void);
unsafe impl Send for RawPtr {}
unsafe impl Sync for RawPtr {}

static WEB_DASHBOARD_WINDOW: OnceLock<Mutex<Option<RawPtr>>> = OnceLock::new();

// Simple app log buffer
static APP_LOG_BUFFER: OnceLock<Mutex<Vec<String>>> = OnceLock::new();

fn app_log_buf() -> &'static Mutex<Vec<String>> {
    APP_LOG_BUFFER.get_or_init(|| Mutex::new(Vec::new()))
}

/// Push a log message to the app log buffer (call from anywhere in ironmlx-app).
#[allow(dead_code)]
pub fn app_log(msg: &str) {
    let mut buf = app_log_buf().lock().unwrap();
    // Use Unix timestamp since we don't have chrono
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    buf.push(format!("{} - ironmlx-app - INFO - {}", ts, msg));
    // Cap at 1000 entries
    if buf.len() > 1000 {
        let drain = buf.len() - 1000;
        buf.drain(..drain);
    }
}

fn window_lock() -> &'static Mutex<Option<RawPtr>> {
    WEB_DASHBOARD_WINDOW.get_or_init(|| Mutex::new(None))
}

// ---------------------------------------------------------------------------
// WKScriptMessageHandler — receives messages from JavaScript
// ---------------------------------------------------------------------------

define_class!(
    #[unsafe(super(NSObject))]
    #[thread_kind = MainThreadOnly]
    #[name = "WebMessageHandler"]
    #[ivars = ()]
    pub struct WebMessageHandler;

    unsafe impl NSObjectProtocol for WebMessageHandler {}

    unsafe impl WKScriptMessageHandler for WebMessageHandler {
        #[unsafe(method(userContentController:didReceiveScriptMessage:))]
        fn did_receive_message(
            &self,
            _controller: &objc2_web_kit::WKUserContentController,
            message: &WKScriptMessage,
        ) {
            let name = unsafe { message.name() }.to_string();
            let body: *const AnyObject = unsafe { msg_send![message, body] };
            if body.is_null() {
                return;
            }
            let body_str: String = unsafe {
                let ns: &NSString = &*(body as *const NSString);
                ns.to_string()
            };

            match name.as_str() {
                "scanLocalModels" => {
                    // Scan HF cache directory for downloaded models
                    let models_json = scan_local_models();
                    if let Ok(guard) = window_lock().lock()
                        && let Some(ref ptr) = *guard
                    {
                        let win: &NSWindow = unsafe { &*(ptr.0 as *const NSWindow) };
                        if let Some(cv) = win.contentView() {
                            let escaped = models_json
                                .replace('\\', "\\\\")
                                .replace('\'', "\\'")
                                .replace('\n', "\\n");
                            let js =
                                NSString::from_str(&format!("onLocalModelsScanned('{}')", escaped));
                            unsafe {
                                let _: () = msg_send![&*cv, evaluateJavaScript: &*js, completionHandler: std::ptr::null::<AnyObject>()];
                            }
                        }
                    }
                }
                "setLanguage" => handle_set_language(&body_str),
                "setTheme" => handle_set_theme(&body_str),
                "setDefaultModel" => handle_set_default_model(&body_str),
                "deleteModels" => {
                    // Delete model directories from ~/.ironmlx/models/
                    let model_ids: Vec<String> =
                        serde_json::from_str(&body_str).unwrap_or_default();
                    let models_dir = crate::config::ironmlx_root().join("models");
                    let mut deleted = Vec::new();
                    let mut current_default = crate::config::AppConfig::load().last_model;
                    for id in &model_ids {
                        // model id "mlx-community/Qwen3-0.6B-4bit" -> dir "models--mlx-community--Qwen3-0.6B-4bit"
                        let dir_name = format!("models--{}", id.replace('/', "--"));
                        let path = models_dir.join(&dir_name);
                        if path.exists() {
                            let _ = std::fs::remove_dir_all(&path);
                            deleted.push(id.clone());
                        }
                        // If deleted model was the default, clear it
                        if current_default.as_deref() == Some(id.as_str()) {
                            current_default = None;
                            let mut config = crate::config::AppConfig::load();
                            config.last_model = None;
                            config.save();
                        }
                    }
                    // Notify JS that deletion is complete, then rescan
                    let result = serde_json::to_string(&deleted).unwrap_or_default();
                    if let Ok(guard) = window_lock().lock()
                        && let Some(ref ptr) = *guard
                    {
                        let win: &NSWindow = unsafe { &*(ptr.0 as *const NSWindow) };
                        if let Some(cv) = win.contentView() {
                            let js = NSString::from_str(&format!(
                                "onModelsDeleted('{}')",
                                result.replace('\'', "\\'")
                            ));
                            unsafe {
                                let _: () = msg_send![&*cv, evaluateJavaScript: &*js, completionHandler: std::ptr::null::<AnyObject>()];
                            }
                        }
                    }
                }
                "loadModel" | "forceLoadModel" => {
                    let model_id = body_str.clone();
                    let is_force = name == "forceLoadModel";
                    let win_ptr_raw = window_lock()
                        .lock()
                        .ok()
                        .and_then(|g| g.as_ref().map(|p| p.0));
                    let win_send = win_ptr_raw.map(RawPtr);
                    std::thread::spawn(move || {
                        let port = crate::config::AppConfig::load().port;
                        let result = if !is_force {
                            // Check memory first
                            let model_size = estimate_model_size_mb(&model_id);
                            let available = get_available_gpu_mb(port);
                            if model_size > 0.0 && available > 0.0 && model_size > available {
                                format!(
                                    "{{\"warning\":true,\"model_id\":\"{}\",\"required_mb\":{:.0},\"available_mb\":{:.0}}}",
                                    model_id.replace('"', "\\\""),
                                    model_size,
                                    available
                                )
                            } else {
                                do_load_model(&model_id, port)
                            }
                        } else {
                            do_load_model(&model_id, port)
                        };
                        eval_js_on_window(
                            win_send,
                            &format!("onModelLoaded('{}')", result.replace('\'', "\\'")),
                        );
                    });
                }
                "unloadModel" => {
                    let model_id = body_str.clone();
                    let win_ptr_raw = window_lock()
                        .lock()
                        .ok()
                        .and_then(|g| g.as_ref().map(|p| p.0));
                    let win_send = win_ptr_raw.map(RawPtr);
                    std::thread::spawn(move || {
                        let port = crate::config::AppConfig::load().port;
                        let result = do_unload_model(&model_id, port);
                        eval_js_on_window(
                            win_send,
                            &format!("onModelUnloaded('{}')", result.replace('\'', "\\'")),
                        );
                    });
                }
                "syncLoadedModels" => {
                    let win_ptr_raw = window_lock()
                        .lock()
                        .ok()
                        .and_then(|g| g.as_ref().map(|p| p.0));
                    let win_send = win_ptr_raw.map(RawPtr);
                    std::thread::spawn(move || {
                        let port = crate::config::AppConfig::load().port;
                        // Retry up to 10 times — server may still be loading model
                        let mut resp = String::new();
                        for _ in 0..10 {
                            if let Ok(r) = reqwest::blocking::get(format!(
                                "http://127.0.0.1:{}/v1/models",
                                port
                            )) && let Ok(text) = r.text()
                                && text.contains("\"data\"")
                            {
                                resp = text;
                                break;
                            }
                            std::thread::sleep(std::time::Duration::from_secs(1));
                        }
                        // Extract model IDs from response
                        let ids: Vec<String> = serde_json::from_str::<serde_json::Value>(&resp)
                            .ok()
                            .and_then(|v| {
                                v["data"].as_array().map(|arr| {
                                    arr.iter()
                                        .filter_map(|m| m["id"].as_str().map(String::from))
                                        .collect()
                                })
                            })
                            .unwrap_or_default();
                        let json = serde_json::to_string(&ids).unwrap_or_default();
                        eval_js_on_window(
                            win_send,
                            &format!("onLoadedModelsSynced('{}')", json.replace('\'', "\\'")),
                        );
                    });
                }
                "checkMoss" => {
                    let installed = std::path::Path::new("/Applications/Moss.app").exists();
                    let status = format!("{{\"installed\":{}}}", installed);
                    eval_js_on_window_sync(&format!(
                        "onMossStatus('{}')",
                        status.replace('\'', "\\'")
                    ));
                }
                "downloadMoss" => {
                    // Open GitHub Releases page in default browser
                    let url = "https://github.com/apepkuss/Aries/releases";
                    let url_ns = objc2_foundation::NSURL::URLWithString(&NSString::from_str(url));
                    if let Some(url_obj) = url_ns {
                        unsafe {
                            let workspace = NSWorkspace::sharedWorkspace();
                            let _: () = msg_send![&*workspace, openURL: &*url_obj];
                        }
                    }
                }
                "openMossDesktop" => {
                    // Launch Moss.app in a background thread to avoid blocking
                    std::thread::spawn(|| {
                        let output = std::process::Command::new("/usr/bin/open")
                            .arg("-a")
                            .arg("/Applications/Moss.app")
                            .env_clear()
                            .env("HOME", std::env::var("HOME").unwrap_or_default())
                            .env("PATH", "/usr/bin:/bin:/usr/sbin:/sbin")
                            .output();
                        if let Err(e) = output {
                            eprintln!("Failed to open Moss: {e}");
                        }
                    });
                }
                "saveSettings" => {
                    // Save settings to app_config.json and check if restart needed
                    let body = body_str.to_string();
                    let win_ptr_raw = window_lock()
                        .lock()
                        .ok()
                        .and_then(|g| g.as_ref().map(|p| p.0));
                    let win_send = win_ptr_raw.map(RawPtr);
                    std::thread::spawn(move || {
                        let mut needs_restart = false;
                        if let Ok(new_settings) = serde_json::from_str::<serde_json::Value>(&body) {
                            let mut config = crate::config::AppConfig::load();
                            let old_host = config.host.clone();
                            let old_port = config.port;

                            // Update config fields that exist in AppConfig
                            if let Some(host) = new_settings.get("host").and_then(|v| v.as_str()) {
                                config.host = host.to_string();
                            }
                            if let Some(port) = new_settings.get("port").and_then(|v| v.as_u64()) {
                                config.port = port as u16;
                            }
                            let mut lang_changed: Option<String> = None;
                            if let Some(lang) =
                                new_settings.get("language").and_then(|v| v.as_str())
                            {
                                config.language = lang.to_string();
                                lang_changed = Some(lang.to_string());
                            }
                            if let Some(theme) = new_settings.get("theme").and_then(|v| v.as_str())
                            {
                                config.theme = Some(theme.to_string());
                            }

                            // Check if restart-requiring settings changed
                            if config.host != old_host || config.port != old_port {
                                needs_restart = true;
                            }

                            // Save config
                            config.save();

                            // Sync language to menubar if changed
                            if let Some(ref lc) = lang_changed {
                                let lang_code = lc.clone();
                                dispatch2::DispatchQueue::main().exec_async(move || {
                                    handle_set_language(&lang_code);
                                });
                            }
                        }

                        // Notify JS
                        let result =
                            format!("{{\"success\":true,\"needs_restart\":{}}}", needs_restart);
                        let escaped = result.replace('\\', "\\\\").replace('\'', "\\'");
                        let js_code = format!("onSettingsSaved('{}')", escaped);
                        let inner_send = win_send.map(|p| RawPtr(p.0));
                        dispatch2::DispatchQueue::main().exec_async(move || {
                            if let Some(ref rp) = inner_send {
                                let win: &objc2_app_kit::NSWindow =
                                    unsafe { &*(rp.0 as *const objc2_app_kit::NSWindow) };
                                if let Some(cv) = win.contentView() {
                                    let js = NSString::from_str(&js_code);
                                    unsafe {
                                        let _: () = msg_send![&*cv, evaluateJavaScript: &*js, completionHandler: std::ptr::null::<objc2::runtime::AnyObject>()];
                                    }
                                }
                            }
                        });
                    });
                }
                "restartServer" => {
                    // Restart the backend server in background, notify JS when done
                    let win_ptr_raw = window_lock()
                        .lock()
                        .ok()
                        .and_then(|g| g.as_ref().map(|p| p.0));
                    let win_send = win_ptr_raw.map(RawPtr);
                    std::thread::spawn(move || {
                        crate::app_delegate::restart_server();
                        // Wait a moment for server to come up
                        std::thread::sleep(std::time::Duration::from_secs(2));
                        let new_port = crate::config::AppConfig::load().port;
                        let js_code = format!("onServerRestarted({})", new_port);
                        let inner_send = win_send.map(|p| RawPtr(p.0));
                        dispatch2::DispatchQueue::main().exec_async(move || {
                            if let Some(ref rp) = inner_send {
                                let win: &objc2_app_kit::NSWindow =
                                    unsafe { &*(rp.0 as *const objc2_app_kit::NSWindow) };
                                if let Some(cv) = win.contentView() {
                                    let js = NSString::from_str(&js_code);
                                    unsafe {
                                        let _: () = msg_send![&*cv, evaluateJavaScript: &*js, completionHandler: std::ptr::null::<objc2::runtime::AnyObject>()];
                                    }
                                }
                            }
                        });
                    });
                }
                "saveModelParams" => {
                    // Save model parameters to ~/.ironmlx/config/model_params.json
                    let body = body_str.to_string();
                    std::thread::spawn(move || {
                        let params_path = crate::config::ironmlx_root()
                            .join("config")
                            .join("model_params.json");
                        // Load existing params
                        let mut all_params: serde_json::Map<String, serde_json::Value> =
                            if params_path.exists() {
                                let data =
                                    std::fs::read_to_string(&params_path).unwrap_or_default();
                                serde_json::from_str(&data).unwrap_or_default()
                            } else {
                                serde_json::Map::new()
                            };
                        // Parse new params and merge
                        if let Ok(new_params) = serde_json::from_str::<serde_json::Value>(&body)
                            && let Some(model_id) =
                                new_params.get("model_id").and_then(|v| v.as_str())
                        {
                            all_params.insert(model_id.to_string(), new_params.clone());
                        }
                        // Write back
                        if let Some(parent) = params_path.parent() {
                            let _ = std::fs::create_dir_all(parent);
                        }
                        if let Ok(data) = serde_json::to_string_pretty(&all_params) {
                            let _ = std::fs::write(&params_path, data);
                        }
                    });
                }
                "downloadModel" => {
                    // Download model from HuggingFace
                    let body = body_str.to_string();
                    let win_ptr_raw = window_lock()
                        .lock()
                        .ok()
                        .and_then(|g| g.as_ref().map(|p| p.0));
                    let win_send = win_ptr_raw.map(RawPtr);
                    std::thread::spawn(move || {
                        let result = (|| -> Result<String, String> {
                            let req: serde_json::Value =
                                serde_json::from_str(&body).map_err(|e| e.to_string())?;
                            let repo_id = req
                                .get("repo_id")
                                .and_then(|v| v.as_str())
                                .ok_or("missing repo_id")?;
                            let token = req.get("token").and_then(|v| v.as_str());

                            // Use hf_hub to download
                            let mut builder = hf_hub::api::sync::ApiBuilder::new()
                                .with_cache_dir(crate::config::ironmlx_root().join("models"));
                            if let Some(t) = token
                                && !t.is_empty()
                            {
                                builder = builder.with_token(Some(t.to_string()));
                            }
                            let api = builder.build().map_err(|e| e.to_string())?;
                            let repo = api.model(repo_id.to_string());

                            // Download key files
                            for file in &[
                                "config.json",
                                "tokenizer.json",
                                "tokenizer_config.json",
                                "model.safetensors",
                                "model.safetensors.index.json",
                            ] {
                                match repo.get(file) {
                                    Ok(_) => {}
                                    Err(_) => {
                                        // Some files are optional
                                        if *file != "model.safetensors.index.json" {
                                            // Try to continue
                                        }
                                    }
                                }
                            }
                            Ok(format!("Model {} downloaded successfully.", repo_id))
                        })();

                        let js_code = match result {
                            Ok(msg) => {
                                let escaped = msg.replace('\\', "\\\\").replace('\'', "\\'");
                                format!(
                                    "onDownloadComplete('{{\"success\":true,\"message\":\"{}\"}}')",
                                    escaped
                                )
                            }
                            Err(e) => {
                                let escaped = e
                                    .replace('\\', "\\\\")
                                    .replace('\'', "\\'")
                                    .replace('"', "\\\"");
                                format!(
                                    "onDownloadComplete('{{\"success\":false,\"error\":\"{}\"}}')",
                                    escaped
                                )
                            }
                        };

                        let inner_send = win_send.map(|p| RawPtr(p.0));
                        dispatch2::DispatchQueue::main().exec_async(move || {
                            if let Some(ref rp) = inner_send {
                                let win: &objc2_app_kit::NSWindow =
                                    unsafe { &*(rp.0 as *const objc2_app_kit::NSWindow) };
                                if let Some(cv) = win.contentView() {
                                    let js = NSString::from_str(&js_code);
                                    unsafe {
                                        let _: () = msg_send![&*cv, evaluateJavaScript: &*js, completionHandler: std::ptr::null::<objc2::runtime::AnyObject>()];
                                    }
                                }
                            }
                        });
                    });
                }
                "searchHF" => {
                    // Search HuggingFace models API
                    let body = body_str.to_string();
                    let win_ptr_raw = window_lock()
                        .lock()
                        .ok()
                        .and_then(|g| g.as_ref().map(|p| p.0));
                    let win_send = win_ptr_raw.map(RawPtr);
                    std::thread::spawn(move || {
                        let result = (|| -> Result<String, String> {
                            let req: serde_json::Value =
                                serde_json::from_str(&body).map_err(|e| e.to_string())?;
                            let query = req.get("query").and_then(|v| v.as_str()).unwrap_or("");
                            let sort = req
                                .get("sort")
                                .and_then(|v| v.as_str())
                                .unwrap_or("downloads");

                            let url = format!(
                                "https://huggingface.co/api/models?search={}&sort={}&direction=-1&limit=20&filter=mlx",
                                urlencoding::encode(query),
                                sort
                            );
                            let resp = reqwest::blocking::get(&url)
                                .map_err(|e| e.to_string())?
                                .text()
                                .map_err(|e| e.to_string())?;
                            Ok(resp)
                        })();

                        let js_code = match result {
                            Ok(data) => {
                                let escaped = data
                                    .replace('\\', "\\\\")
                                    .replace('\'', "\\'")
                                    .replace('\n', "");
                                format!("onSearchResults('{}')", escaped)
                            }
                            Err(_) => "onSearchResults('[]')".to_string(),
                        };

                        let inner_send = win_send.map(|p| RawPtr(p.0));
                        dispatch2::DispatchQueue::main().exec_async(move || {
                            if let Some(ref rp) = inner_send {
                                let win: &objc2_app_kit::NSWindow =
                                    unsafe { &*(rp.0 as *const objc2_app_kit::NSWindow) };
                                if let Some(cv) = win.contentView() {
                                    let js = NSString::from_str(&js_code);
                                    unsafe {
                                        let _: () = msg_send![&*cv, evaluateJavaScript: &*js, completionHandler: std::ptr::null::<objc2::runtime::AnyObject>()];
                                    }
                                }
                            }
                        });
                    });
                }
                "fetchAPI" => {
                    // Bridge: JS asks Rust to fetch a local API endpoint
                    let port = crate::config::AppConfig::load().port;
                    let url = format!("http://127.0.0.1:{}{}", port, body_str);
                    let win_ptr_raw = window_lock()
                        .lock()
                        .ok()
                        .and_then(|g| g.as_ref().map(|p| p.0));
                    let win_send = win_ptr_raw.map(RawPtr);
                    std::thread::spawn(move || {
                        let win_ptr = win_send.map(|w| w.0);
                        let result = std::process::Command::new("curl")
                            .args(["-s", "--max-time", "2", &url])
                            .output()
                            .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
                            .unwrap_or_else(|_| "null".to_string());
                        let escaped = result
                            .replace('\\', "\\\\")
                            .replace('\'', "\\'")
                            .replace('\n', "\\n");
                        let path_escaped = body_str.replace('\\', "\\\\").replace('\'', "\\'");
                        let js_code =
                            format!("onApiFetchResult('{}', '{}')", path_escaped, escaped);
                        let inner_send = win_ptr.map(RawPtr);
                        dispatch2::DispatchQueue::main().exec_async(move || {
                            if let Some(ref rp) = inner_send {
                                let ptr = rp.0;
                                let win: &NSWindow = unsafe { &*(ptr as *const NSWindow) };
                                if let Some(cv) = win.contentView() {
                                    let js = NSString::from_str(&js_code);
                                    unsafe {
                                        let _: () = msg_send![&*cv, evaluateJavaScript: &*js, completionHandler: std::ptr::null::<AnyObject>()];
                                    }
                                }
                            }
                        });
                    });
                }
                "getAppLogs" => {
                    // Return app logs by evaluating JS
                    let logs = app_log_buf().lock().unwrap().join("\n");
                    let escaped = logs
                        .replace('\\', "\\\\")
                        .replace('\'', "\\'")
                        .replace('\n', "\\n");
                    // Find the webview and evaluate JS
                    if let Ok(guard) = window_lock().lock()
                        && let Some(ref ptr) = *guard
                    {
                        let win: &NSWindow = unsafe { &*(ptr.0 as *const NSWindow) };
                        if let Some(cv) = win.contentView() {
                            let js = NSString::from_str(&format!("receiveAppLogs('{}')", escaped));
                            unsafe {
                                let _: () = msg_send![&*cv, evaluateJavaScript: &*js, completionHandler: std::ptr::null::<AnyObject>()];
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }
);

impl WebMessageHandler {
    fn new(mtm: MainThreadMarker) -> Retained<Self> {
        unsafe { msg_send![mtm.alloc::<Self>(), init] }
    }
}

// ---------------------------------------------------------------------------
// WKUIDelegate — handle JS alert() / confirm() / prompt()
// ---------------------------------------------------------------------------

define_class!(
    #[unsafe(super(NSObject))]
    #[thread_kind = MainThreadOnly]
    #[name = "WebUIDelegate"]
    pub struct WebUIDelegate;

    unsafe impl NSObjectProtocol for WebUIDelegate {}

    unsafe impl WKUIDelegate for WebUIDelegate {
        // alert()
        #[unsafe(method(webView:runJavaScriptAlertPanelWithMessage:initiatedByFrame:completionHandler:))]
        unsafe fn run_alert(
            &self,
            _webview: &WKWebView,
            message: &NSString,
            _frame: &AnyObject,
            completion: &block2::Block<dyn Fn()>,
        ) {
            let msg = message.to_string();
            let mtm = unsafe { MainThreadMarker::new_unchecked() };
            let alert = NSAlert::new(mtm);
            alert.setMessageText(&NSString::from_str(&msg));
            alert.runModal();
            completion.call(());
        }

        // confirm()
        #[unsafe(method(webView:runJavaScriptConfirmPanelWithMessage:initiatedByFrame:completionHandler:))]
        unsafe fn run_confirm(
            &self,
            _webview: &WKWebView,
            message: &NSString,
            _frame: &AnyObject,
            completion: &block2::Block<dyn Fn(objc2::runtime::Bool)>,
        ) {
            let msg = message.to_string();
            let mtm = unsafe { MainThreadMarker::new_unchecked() };
            let alert = NSAlert::new(mtm);
            alert.setMessageText(&NSString::from_str(&msg));
            alert.addButtonWithTitle(&NSString::from_str("OK"));
            alert.addButtonWithTitle(&NSString::from_str("Cancel"));
            let response: isize = unsafe { msg_send![&*alert, runModal] };
            // NSAlertFirstButtonReturn = 1000
            let result = if response == 1000 {
                objc2::runtime::Bool::YES
            } else {
                objc2::runtime::Bool::NO
            };
            completion.call((result,));
        }
    }
);

impl WebUIDelegate {
    fn new(mtm: MainThreadMarker) -> Retained<Self> {
        unsafe { msg_send![mtm.alloc::<Self>(), init] }
    }
}

fn handle_set_theme(theme: &str) {
    use objc2::runtime::AnyClass;

    let appearance_name: Option<&str> = match theme {
        "light" => Some("NSAppearanceNameAqua"),
        "dark" => Some("NSAppearanceNameDarkAqua"),
        _ => None, // "system"
    };

    // Persist to config
    let mut config = crate::config::AppConfig::load();
    config.theme = match theme {
        "light" => Some("light".to_string()),
        "dark" => Some("dark".to_string()),
        _ => None,
    };
    config.save();

    // Apply to web dashboard window
    if let Ok(guard) = window_lock().lock()
        && let Some(ref ptr) = *guard
    {
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
                    let _: () = msg_send![window, setAppearance: std::ptr::null::<AnyObject>()];
                }
            }
        }
    }
}

fn scan_local_models() -> String {
    let cache_dir = crate::config::ironmlx_root().join("models");

    let mut models = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&cache_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if !name.starts_with("models--") {
                continue;
            }
            // Convert "models--mlx-community--Qwen3-0.6B-4bit" -> "mlx-community/Qwen3-0.6B-4bit"
            let model_id = name
                .strip_prefix("models--")
                .unwrap_or(&name)
                .replacen("--", "/", 1);

            // Calculate directory size
            let dir_path = entry.path();
            let size_bytes = dir_size(&dir_path);
            let size_mb = size_bytes as f64 / (1024.0 * 1024.0);

            // Try to detect model type from config.json
            let model_type = detect_model_type(&dir_path);

            models.push(format!(
                "{{\"id\":\"{}\",\"size_mb\":{:.1},\"type\":\"{}\"}}",
                model_id.replace('"', "\\\""),
                size_mb,
                model_type
            ));
        }
    }
    format!("[{}]", models.join(","))
}

fn dir_size(path: &std::path::Path) -> u64 {
    let mut total = 0u64;
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            let ft = entry
                .file_type()
                .unwrap_or_else(|_| entry.file_type().unwrap());
            if ft.is_dir() {
                total += dir_size(&entry.path());
            } else {
                total += entry.metadata().map(|m| m.len()).unwrap_or(0);
            }
        }
    }
    total
}

fn detect_model_type(model_dir: &std::path::Path) -> &'static str {
    // Look for config.json in snapshots
    let snapshots = model_dir.join("snapshots");
    if let Ok(entries) = std::fs::read_dir(&snapshots) {
        for entry in entries.flatten() {
            let config_path = entry.path().join("config.json");
            if config_path.exists() {
                if let Ok(data) = std::fs::read_to_string(&config_path)
                    && (data.contains("\"vision")
                        || data.contains("visual")
                        || data.contains("image_size"))
                {
                    return "vlm";
                }
                return "llm";
            }
        }
    }
    "llm"
}

/// Evaluate JS on the web dashboard window from any thread
fn eval_js_on_window(win_send: Option<RawPtr>, js_code: &str) {
    let js_owned = js_code.to_string();
    let inner_send = win_send;
    dispatch2::DispatchQueue::main().exec_async(move || {
        if let Some(ref rp) = inner_send {
            let win: &NSWindow = unsafe { &*(rp.0 as *const NSWindow) };
            if let Some(cv) = win.contentView() {
                let js = NSString::from_str(&js_owned);
                unsafe {
                    let _: () = msg_send![&*cv, evaluateJavaScript: &*js, completionHandler: std::ptr::null::<AnyObject>()];
                }
            }
        }
    });
}

/// Evaluate JS on the web dashboard window from the main thread (used in message handler)
fn eval_js_on_window_sync(js_code: &str) {
    if let Ok(guard) = window_lock().lock()
        && let Some(ref ptr) = *guard
    {
        let win: &NSWindow = unsafe { &*(ptr.0 as *const NSWindow) };
        if let Some(cv) = win.contentView() {
            let js = NSString::from_str(js_code);
            unsafe {
                let _: () = msg_send![&*cv, evaluateJavaScript: &*js, completionHandler: std::ptr::null::<AnyObject>()];
            }
        }
    }
}

/// Estimate model size in MB from local files
fn estimate_model_size_mb(model_id: &str) -> f64 {
    let models_dir = crate::config::ironmlx_root().join("models");
    let dir_name = format!("models--{}", model_id.replace('/', "--"));
    let path = models_dir.join(&dir_name);
    if path.exists() {
        dir_size(&path) as f64 / (1024.0 * 1024.0)
    } else {
        0.0
    }
}

/// Get available GPU memory from health API
fn get_available_gpu_mb(port: u16) -> f64 {
    let resp = reqwest::blocking::get(format!("http://127.0.0.1:{}/health", port))
        .and_then(|r| r.text())
        .unwrap_or_default();
    let v: serde_json::Value = serde_json::from_str(&resp).unwrap_or_default();
    let active = v["memory"]["active_mb"].as_f64().unwrap_or(0.0);
    // Try to get total GPU memory from device info
    let total = v["memory"]["total_mb"].as_f64().unwrap_or(0.0);
    if total > 0.0 {
        total - active
    } else {
        // Fallback: assume 75% of system memory is available for GPU
        0.0 // unknown, skip check
    }
}

/// Load a model via backend API (uses curl for reliability)
fn do_load_model(model_id: &str, port: u16) -> String {
    let url = format!("http://127.0.0.1:{}/v1/models/load", port);
    let payload = format!("{{\"model_dir\":\"{}\"}}", model_id.replace('"', "\\\""));
    let output = std::process::Command::new("/usr/bin/curl")
        .args([
            "-s",
            "-X",
            "POST",
            &url,
            "-H",
            "Content-Type: application/json",
            "-d",
            &payload,
        ])
        .output();
    match output {
        Ok(o) => {
            let body = String::from_utf8_lossy(&o.stdout).to_string();
            if body.contains("\"id\"") || body.contains("\"object\"") {
                format!(
                    "{{\"success\":true,\"model_id\":\"{}\"}}",
                    model_id.replace('"', "\\\"")
                )
            } else {
                format!(
                    "{{\"error\":\"Load failed: {}\"}}",
                    body.replace('"', "\\\"")
                )
            }
        }
        Err(e) => format!("{{\"error\":\"{}\"}}", e.to_string().replace('"', "\\\"")),
    }
}

/// Unload a model via backend API (uses curl for reliability)
fn do_unload_model(model_id: &str, port: u16) -> String {
    let url = format!("http://127.0.0.1:{}/v1/models/unload", port);
    let payload = format!("{{\"model\":\"{}\"}}", model_id.replace('"', "\\\""));
    let output = std::process::Command::new("/usr/bin/curl")
        .args([
            "-s",
            "-X",
            "POST",
            &url,
            "-H",
            "Content-Type: application/json",
            "-d",
            &payload,
        ])
        .output();
    match output {
        Ok(o) => {
            let body = String::from_utf8_lossy(&o.stdout).to_string();
            if body.contains("unloaded") || body.contains("ok") {
                format!(
                    "{{\"success\":true,\"model_id\":\"{}\"}}",
                    model_id.replace('"', "\\\"")
                )
            } else {
                format!(
                    "{{\"error\":\"Unload failed: {}\"}}",
                    body.replace('"', "\\\"")
                )
            }
        }
        Err(e) => format!("{{\"error\":\"{}\"}}", e.to_string().replace('"', "\\\"")),
    }
}

fn handle_set_default_model(model_id: &str) {
    let mut config = crate::config::AppConfig::load();
    config.last_model = Some(model_id.to_string());
    config.save();

    // Update menubar to reflect new model name
    let mtm = unsafe { MainThreadMarker::new_unchecked() };
    crate::app_delegate::refresh_menu(mtm);
}

fn handle_set_language(lang_code: &str) {
    let lang: &'static str = match lang_code {
        "zh-Hans" => "zh",
        "zh-Hant" => "zh-Hant",
        "ja" => "ja",
        "ko" => "ko",
        _ => "en",
    };

    // Persist to config
    let mut config = crate::config::AppConfig::load();
    config.language = lang.to_string();
    config.save();

    // Update native dashboard language state (if it's open)
    *crate::i18n::nav_language().lock().unwrap() = lang;

    // Rebuild menubar menu
    let mtm = unsafe { MainThreadMarker::new_unchecked() };
    crate::app_delegate::refresh_menu(mtm);
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Show the web dashboard window. If already open, bring it to front.
pub fn show_web_dashboard(mtm: MainThreadMarker) {
    // Switch to Regular activation policy so window can receive keyboard input
    let app = NSApplication::sharedApplication(mtm);
    app.setActivationPolicy(NSApplicationActivationPolicy::Regular);

    // Set up a main menu so fullscreen mode works correctly (menu bar + traffic lights)
    if app.mainMenu().is_none() {
        unsafe {
            let menu_bar = NSMenu::new(mtm);

            // App menu (Quit)
            let app_menu_item = NSMenuItem::new(mtm);
            let app_menu = NSMenu::new(mtm);
            let quit_item = NSMenuItem::initWithTitle_action_keyEquivalent(
                mtm.alloc(),
                &NSString::from_str("Quit"),
                Some(sel!(terminate:)),
                &NSString::from_str("q"),
            );
            app_menu.addItem(&quit_item);
            app_menu_item.setSubmenu(Some(&app_menu));
            menu_bar.addItem(&app_menu_item);

            // Edit menu (Undo/Redo/Cut/Copy/Paste/Select All)
            let edit_menu_item = NSMenuItem::new(mtm);
            let edit_menu = NSMenu::initWithTitle(mtm.alloc(), &NSString::from_str("Edit"));
            let undo = NSMenuItem::initWithTitle_action_keyEquivalent(
                mtm.alloc(),
                &NSString::from_str("Undo"),
                Some(sel!(undo:)),
                &NSString::from_str("z"),
            );
            let redo = NSMenuItem::initWithTitle_action_keyEquivalent(
                mtm.alloc(),
                &NSString::from_str("Redo"),
                Some(sel!(redo:)),
                &NSString::from_str("Z"),
            );
            let cut = NSMenuItem::initWithTitle_action_keyEquivalent(
                mtm.alloc(),
                &NSString::from_str("Cut"),
                Some(sel!(cut:)),
                &NSString::from_str("x"),
            );
            let copy = NSMenuItem::initWithTitle_action_keyEquivalent(
                mtm.alloc(),
                &NSString::from_str("Copy"),
                Some(sel!(copy:)),
                &NSString::from_str("c"),
            );
            let paste = NSMenuItem::initWithTitle_action_keyEquivalent(
                mtm.alloc(),
                &NSString::from_str("Paste"),
                Some(sel!(paste:)),
                &NSString::from_str("v"),
            );
            let select_all = NSMenuItem::initWithTitle_action_keyEquivalent(
                mtm.alloc(),
                &NSString::from_str("Select All"),
                Some(sel!(selectAll:)),
                &NSString::from_str("a"),
            );
            edit_menu.addItem(&undo);
            edit_menu.addItem(&redo);
            edit_menu.addItem(&NSMenuItem::separatorItem(mtm));
            edit_menu.addItem(&cut);
            edit_menu.addItem(&copy);
            edit_menu.addItem(&paste);
            edit_menu.addItem(&NSMenuItem::separatorItem(mtm));
            edit_menu.addItem(&select_all);
            edit_menu_item.setSubmenu(Some(&edit_menu));
            menu_bar.addItem(&edit_menu_item);

            app.setMainMenu(Some(&menu_bar));
        }
    }

    // Check if window already exists and is valid
    if let Ok(guard) = window_lock().lock()
        && let Some(ref ptr) = *guard
    {
        let win = ptr.0 as *const NSWindow;
        if !win.is_null() {
            unsafe {
                let w = &*win;
                w.makeKeyAndOrderFront(None);
                NSApplication::sharedApplication(mtm).activate();
            }
            return;
        }
    }

    // Create new window
    let window = create_web_dashboard_window(mtm);

    // Store the window pointer
    let ptr = &*window as *const NSWindow as *const std::ffi::c_void;
    *window_lock().lock().unwrap() = Some(RawPtr(ptr));

    // Show
    window.makeKeyAndOrderFront(None);
    NSApplication::sharedApplication(mtm).activate();

    // Leak to keep alive (menubar app manages lifecycle)
    let _ = Retained::into_raw(window);
}

// ---------------------------------------------------------------------------
// Window creation
// ---------------------------------------------------------------------------

fn create_web_dashboard_window(mtm: MainThreadMarker) -> Retained<NSWindow> {
    let w = 1100.0;
    let h = 700.0;

    let style = NSWindowStyleMask(
        NSWindowStyleMask::Titled.0
            | NSWindowStyleMask::Closable.0
            | NSWindowStyleMask::Miniaturizable.0
            | NSWindowStyleMask::Resizable.0,
    );

    let window = unsafe {
        let win = NSWindow::initWithContentRect_styleMask_backing_defer(
            mtm.alloc(),
            NSRect::new(NSPoint::new(200.0, 150.0), NSSize::new(w, h)),
            style,
            NSBackingStoreType(2), // NSBackingStoreBuffered
            false,
        );
        win.setTitle(&NSString::from_str(""));
        win.setMinSize(NSSize::new(700.0, 450.0));
        win.center();
        win.setReleasedWhenClosed(false);
        // Ensure menu bar and traffic lights auto-show in fullscreen
        let _: () = msg_send![&*win, setCollectionBehavior: 1u64 << 7]; // NSWindowCollectionBehaviorFullScreenPrimary
        win
    };

    // Create WKWebView with message handler
    let config = unsafe { WKWebViewConfiguration::new(mtm) };

    // Register JS→Rust message handler
    let handler = WebMessageHandler::new(mtm);
    let handler_obj = objc2::runtime::ProtocolObject::from_retained(handler);
    unsafe {
        let uc = config.userContentController();
        uc.addScriptMessageHandler_name(&handler_obj, &NSString::from_str("scanLocalModels"));
        uc.addScriptMessageHandler_name(&handler_obj, &NSString::from_str("setLanguage"));
        uc.addScriptMessageHandler_name(&handler_obj, &NSString::from_str("setTheme"));
        uc.addScriptMessageHandler_name(&handler_obj, &NSString::from_str("setDefaultModel"));
        uc.addScriptMessageHandler_name(&handler_obj, &NSString::from_str("deleteModels"));
        uc.addScriptMessageHandler_name(&handler_obj, &NSString::from_str("loadModel"));
        uc.addScriptMessageHandler_name(&handler_obj, &NSString::from_str("forceLoadModel"));
        uc.addScriptMessageHandler_name(&handler_obj, &NSString::from_str("unloadModel"));
        uc.addScriptMessageHandler_name(&handler_obj, &NSString::from_str("syncLoadedModels"));
        uc.addScriptMessageHandler_name(&handler_obj, &NSString::from_str("getAppLogs"));
        uc.addScriptMessageHandler_name(&handler_obj, &NSString::from_str("fetchAPI"));
        uc.addScriptMessageHandler_name(&handler_obj, &NSString::from_str("checkMoss"));
        uc.addScriptMessageHandler_name(&handler_obj, &NSString::from_str("downloadMoss"));
        uc.addScriptMessageHandler_name(&handler_obj, &NSString::from_str("openMossDesktop"));
        uc.addScriptMessageHandler_name(&handler_obj, &NSString::from_str("saveSettings"));
        uc.addScriptMessageHandler_name(&handler_obj, &NSString::from_str("restartServer"));
        uc.addScriptMessageHandler_name(&handler_obj, &NSString::from_str("saveModelParams"));
        uc.addScriptMessageHandler_name(&handler_obj, &NSString::from_str("downloadModel"));
        uc.addScriptMessageHandler_name(&handler_obj, &NSString::from_str("searchHF"));
    }

    let webview = unsafe {
        let wv = WKWebView::initWithFrame_configuration(
            mtm.alloc(),
            NSRect::new(NSPoint::ZERO, NSSize::new(w, h)),
            &config,
        );
        wv.setAutoresizingMask(NSAutoresizingMaskOptions(2 | 16)); // width + height flexible
        // Set UI delegate for JS alert()/confirm()/prompt()
        let ui_delegate = WebUIDelegate::new(mtm);
        let ui_delegate_obj = objc2::runtime::ProtocolObject::from_retained(ui_delegate);
        wv.setUIDelegate(Some(&ui_delegate_obj));
        // Leak the delegate to keep it alive
        std::mem::forget(ui_delegate_obj);
        wv
    };

    // Load embedded HTML via loadHTMLString (no CORS issues with http:// API calls)
    let config = crate::config::AppConfig::load();
    let current_lang = config.language;
    let default_model = config.last_model.clone().unwrap_or_default();
    let current_theme = match config.theme.as_deref() {
        Some("light") => "light",
        Some("dark") => "dark",
        _ => "system",
    };
    let html = include_str!("dashboard2.html");
    let port = config.port;
    let html_with_lang = html.replace(
        "/*__INIT_LANG__*/",
        &format!(
            "window.__IRONMLX_PORT__ = {}; window.__DEFAULT_MODEL__ = '{}'; {} setLanguage('{}'); setTheme('{}'); initLogs(); syncModelList(); syncLoadedModels(); initStatusPolling(); checkMossInstalled();{} {}",
            port,
            default_model,
            if !default_model.is_empty() {
                format!("window.__LOADED_MODELS__.add('{}');", default_model)
            } else {
                String::new()
            },
            current_lang,
            current_theme,
            if default_model.is_empty() { " showOnboarding();" } else { "" },
            {
                // Load saved model params
                let params_path = crate::config::ironmlx_root()
                    .join("config")
                    .join("model_params.json");
                if params_path.exists() {
                    let data = std::fs::read_to_string(&params_path).unwrap_or_default();
                    let escaped = data.replace('\\', "\\\\").replace('\'', "\\'").replace('\n', "");
                    format!("onModelParamsLoaded('{}');", escaped)
                } else {
                    String::new()
                }
            }
        ),
    );

    let html_ns = NSString::from_str(&html_with_lang);
    let base_url: Option<&NSURL> = None;
    unsafe {
        let _: Retained<AnyObject> =
            msg_send![&*webview, loadHTMLString: &*html_ns, baseURL: base_url];
    }

    // Set as content view
    let view: Retained<NSView> = unsafe { Retained::cast_unchecked::<NSView>(webview) };
    window.setContentView(Some(&view));

    window
}
