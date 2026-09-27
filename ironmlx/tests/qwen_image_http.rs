//! Real-model App-daemon lifecycle acceptance for Qwen Image 2.1.
//!
//! Set `IRONMLX_QWEN_IMAGE_MODEL_DIR` to the complete
//! `mlx-community/Qwen-Image-2.1-MLX-4bit` snapshot. Set
//! `IRONMLX_QWEN_IMAGE_HTTP_BUNDLE` to an assembled `IronMLX.app` to exercise
//! its Release helper and bundled metallib instead of the Cargo-built binary.

#[path = "common/ironmlx_process.rs"]
mod ironmlx_process;

use base64::Engine as _;
use serde_json::{json, Value};
use std::{
    path::PathBuf,
    process::{Child, Stdio},
    time::{Duration, Instant},
};

const MODEL_ID: &str = "mlx-community/Qwen-Image-2.1-MLX-4bit";

struct Server {
    child: Child,
    client: reqwest::Client,
    base: String,
    root: PathBuf,
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        if std::thread::panicking() {
            eprintln!(
                "preserving failed Qwen Image acceptance artifacts: {}",
                self.root.display()
            );
        } else {
            let _ = std::fs::remove_dir_all(&self.root);
        }
    }
}

impl Server {
    async fn start() -> Self {
        let port = std::net::TcpListener::bind("127.0.0.1:0")
            .expect("bind temporary port")
            .local_addr()
            .expect("temporary port address")
            .port();
        let root =
            std::env::temp_dir().join(format!("ironmlx-qwen-image-http-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).expect("create server test root");

        let mut command = if let Some(bundle) = std::env::var_os("IRONMLX_QWEN_IMAGE_HTTP_BUNDLE") {
            let bundle = PathBuf::from(bundle)
                .canonicalize()
                .expect("canonical Bundle path");
            let helper = bundle.join("Contents/Helpers/ironmlx");
            let metallib = bundle.join("Contents/Resources/mlx.metallib");
            assert!(helper.is_file() && metallib.is_file(), "incomplete Bundle");
            let mut command = std::process::Command::new(helper);
            for (key, _) in std::env::vars_os() {
                if key.to_string_lossy().starts_with("MLX_")
                    || key.to_string_lossy().starts_with("DYLD_")
                {
                    command.env_remove(key);
                }
            }
            command.arg("--mlx-metallib").arg(metallib);
            command.current_dir(&root);
            command
        } else {
            ironmlx_process::command()
        };
        let child = command
            .arg("serve")
            .arg("--port")
            .arg(port.to_string())
            .arg("--max-loaded-models")
            .arg("1")
            .arg("--memory-limit-total-gb")
            .arg("32")
            .arg("--memory-limit-model-gb")
            .arg("16")
            .stdout(Stdio::null())
            .stderr(std::fs::File::create(root.join("server.log")).expect("server log"))
            .spawn()
            .expect("spawn IronMLX App daemon");
        let server = Self {
            child,
            client: reqwest::Client::builder()
                .no_proxy()
                .timeout(Duration::from_secs(900))
                .build()
                .expect("HTTP client"),
            base: format!("http://127.0.0.1:{port}"),
            root,
        };
        for _ in 0..300 {
            if server
                .client
                .get(format!("{}/health", server.base))
                .send()
                .await
                .is_ok_and(|response| response.status().is_success())
            {
                return server;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        panic!("server startup failed; logs: {}", server.root.display());
    }

    async fn post(&self, path: &str, body: &Value) -> reqwest::Response {
        self.client
            .post(format!("{}{path}", self.base))
            .json(body)
            .send()
            .await
            .expect("HTTP request")
    }

    async fn post_multipart(
        &self,
        path: &str,
        form: reqwest::multipart::Form,
    ) -> reqwest::Response {
        self.client
            .post(format!("{}{path}", self.base))
            .multipart(form)
            .send()
            .await
            .expect("multipart HTTP request")
    }

    async fn loaded_models(&self) -> Vec<Value> {
        self.client
            .get(format!("{}/admin/api/models/loaded", self.base))
            .send()
            .await
            .expect("loaded models request")
            .json()
            .await
            .expect("loaded models JSON")
    }

    fn rss_kib(&self) -> u64 {
        let output = std::process::Command::new("/bin/ps")
            .args(["-o", "rss=", "-p", &self.child.id().to_string()])
            .output()
            .expect("read server RSS");
        assert!(output.status.success(), "ps failed: {output:?}");
        String::from_utf8(output.stdout)
            .expect("RSS is UTF-8")
            .trim()
            .parse()
            .expect("RSS is an integer")
    }
}

fn physical_memory_gib() -> f64 {
    let output = std::process::Command::new("/usr/sbin/sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .expect("read physical memory");
    assert!(output.status.success(), "sysctl failed: {output:?}");
    let bytes: u64 = String::from_utf8(output.stdout)
        .expect("physical memory is UTF-8")
        .trim()
        .parse()
        .expect("physical memory is an integer");
    bytes as f64 / 1024_f64.powi(3)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires the complete Qwen Image 2.1 MLX snapshot"]
async fn app_daemon_register_load_generate_and_unload() {
    let model_dir = PathBuf::from(
        std::env::var_os("IRONMLX_QWEN_IMAGE_MODEL_DIR")
            .expect("IRONMLX_QWEN_IMAGE_MODEL_DIR must be set"),
    )
    .canonicalize()
    .expect("canonical model directory");
    let full_acceptance = std::env::var_os("IRONMLX_QWEN_IMAGE_FULL_ACCEPTANCE").is_some();
    let physical_memory_gib = physical_memory_gib();
    if let Some(expected) = std::env::var_os("IRONMLX_QWEN_IMAGE_EXPECTED_MEMORY_GB") {
        let expected: f64 = expected
            .to_string_lossy()
            .parse()
            .expect("IRONMLX_QWEN_IMAGE_EXPECTED_MEMORY_GB must be numeric");
        assert!(
            (physical_memory_gib - expected).abs() <= 0.5,
            "physical memory is {physical_memory_gib:.2} GiB, expected {expected:.2} GiB"
        );
    }
    eprintln!("physical memory: {physical_memory_gib:.2} GiB");
    let server = Server::start().await;

    let registered = server
        .post(
            "/admin/api/models/register",
            &json!({"model":MODEL_ID,"model_dir":model_dir,"set_default":true}),
        )
        .await;
    assert_eq!(registered.status(), 200);
    let registered: Value = registered.json().await.expect("register response");
    assert_eq!(registered["status"], "registered");
    assert_eq!(registered["loaded_models"], json!([]));

    let started = Instant::now();
    let loaded = server
        .post(
            "/admin/api/models/load",
            &json!({"model":MODEL_ID,"model_dir":model_dir,"set_default":true}),
        )
        .await;
    assert_eq!(loaded.status(), 200);
    let loaded: Value = loaded.json().await.expect("load response");
    assert_eq!(loaded["status"], "loaded");
    let model = loaded["loaded_models"]
        .as_array()
        .and_then(|models| models.iter().find(|model| model["id"] == MODEL_ID))
        .expect("loaded Qwen Image model");
    assert_eq!(model["runtime_kind"], "image_generation");
    assert_eq!(model["architecture"], "qwen_image_2_1");
    eprintln!("Qwen Image load: {:?}", started.elapsed());

    let (generation_request, expected_dimensions) = if full_acceptance {
        (
            json!({
                "prompt":"A small red circle centered on a clean white background",
                "response_format":"b64_json",
                "seed":7
            }),
            (1024, 1024),
        )
    } else {
        (
            json!({
                "prompt":"A small red circle centered on a clean white background",
                "size":"256x256",
                "response_format":"b64_json",
                "seed":7,
                "inference_steps":2
            }),
            (256, 256),
        )
    };
    let started = Instant::now();
    let generated = server
        .post("/v1/images/generations", &generation_request)
        .await;
    assert_eq!(generated.status(), 200);
    let generated: Value = generated.json().await.expect("generation response");
    let data = generated["data"].as_array().expect("image data array");
    assert_eq!(data.len(), 1);
    let png = base64::engine::general_purpose::STANDARD
        .decode(data[0]["b64_json"].as_str().expect("base64 PNG"))
        .expect("decode base64 PNG");
    let image = image::load_from_memory(&png).expect("decode PNG");
    assert_eq!((image.width(), image.height()), expected_dimensions);
    assert!(image.color().has_alpha());
    eprintln!(
        "Qwen Image generation: {:?}; server RSS: {:.2} GiB; PNG bytes: {}",
        started.elapsed(),
        server.rss_kib() as f64 / 1024_f64.powi(2),
        png.len()
    );

    let unknown = server
        .post(
            "/v1/images/generations",
            &json!({"prompt":"test","response_formatt":"b64_json"}),
        )
        .await;
    assert_eq!(unknown.status(), 400);
    let unknown: Value = unknown.json().await.expect("unknown-field error");
    assert_eq!(unknown["error"]["code"], "invalid_json");

    let chat = server
        .post(
            "/v1/chat/completions",
            &json!({"model":MODEL_ID,"messages":[{"role":"user","content":"test"}]}),
        )
        .await;
    assert_eq!(chat.status(), 400);
    let chat: Value = chat.json().await.expect("task mismatch error");
    assert_eq!(chat["error"]["code"], "model_task_mismatch");

    let condition_part = reqwest::multipart::Part::bytes(png.clone())
        .file_name("condition.png")
        .mime_str("image/png")
        .expect("PNG content type");
    let mut edit_form = reqwest::multipart::Form::new()
        .part("image", condition_part)
        .text("prompt", "Change the centered circle from red to blue")
        .text("model", MODEL_ID)
        .text("response_format", "b64_json")
        .text("seed", "11");
    if full_acceptance {
        edit_form = edit_form.text("size", "1024x1024");
    } else {
        edit_form = edit_form
            .text("size", "256x256")
            .text("inference_steps", "2");
    }
    let started = Instant::now();
    let edited = server.post_multipart("/v1/images/edits", edit_form).await;
    assert_eq!(edited.status(), 200);
    let edited: Value = edited.json().await.expect("edit response");
    let edit_data = edited["data"].as_array().expect("edited image data array");
    assert_eq!(edit_data.len(), 1);
    let edited_png = base64::engine::general_purpose::STANDARD
        .decode(
            edit_data[0]["b64_json"]
                .as_str()
                .expect("edited base64 PNG"),
        )
        .expect("decode edited base64 PNG");
    let edited_image = image::load_from_memory(&edited_png).expect("decode edited PNG");
    assert_eq!(
        (edited_image.width(), edited_image.height()),
        expected_dimensions
    );
    assert!(edited_image.color().has_alpha());
    eprintln!(
        "Qwen Image edit: {:?}; server RSS: {:.2} GiB; PNG bytes: {}",
        started.elapsed(),
        server.rss_kib() as f64 / 1024_f64.powi(2),
        edited_png.len()
    );

    let mask_rejection = server
        .post_multipart(
            "/v1/images/edits",
            reqwest::multipart::Form::new()
                .part(
                    "image",
                    reqwest::multipart::Part::bytes(png.clone())
                        .file_name("condition.png")
                        .mime_str("image/png")
                        .expect("PNG content type"),
                )
                .part(
                    "mask",
                    reqwest::multipart::Part::bytes(png)
                        .file_name("mask.png")
                        .mime_str("image/png")
                        .expect("PNG content type"),
                )
                .text("prompt", "test"),
        )
        .await;
    assert_eq!(mask_rejection.status(), 400);
    let mask_rejection: Value = mask_rejection
        .json()
        .await
        .expect("mask rejection response");
    assert_eq!(mask_rejection["error"]["code"], "unsupported_image_mask");

    let unloaded = server
        .post("/admin/api/models/unload", &json!({"model":MODEL_ID}))
        .await;
    assert_eq!(unloaded.status(), 200);
    for _ in 0..300 {
        if !server
            .loaded_models()
            .await
            .iter()
            .any(|model| model["id"] == MODEL_ID)
        {
            return;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("Qwen Image model did not unload");
}
