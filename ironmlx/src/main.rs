mod api;
mod engine;
mod engine_handle;
mod state;

use std::sync::Arc;

use clap::Parser;
use engine::EngineCore;
use engine_handle::EngineHandle;
use state::AppState;
use tokio::sync::mpsc;

#[derive(Parser)]
#[command(name = "ironmlx", about = "Local LLM inference on Apple Silicon")]
struct Args {
    /// Path to model directory (containing config.json, *.safetensors, tokenizer.json)
    #[arg(long)]
    model: String,

    /// Host to bind to
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    /// Port to bind to
    #[arg(long, default_value_t = 8080)]
    port: u16,
}

#[tokio::main]
async fn main() {
    let args = Args::parse();

    println!("ironmlx starting...");
    println!("Loading model from: {}", args.model);

    // Initialize MLX runtime
    ironmlx_core::init();

    // Load model artifacts
    let (model, tokenizer, chat_template, eos_token_id, model_id, patch_size, spatial_merge_size) =
        match state::load_model(&args.model) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("Failed to load model: {}", e);
                std::process::exit(1);
            }
        };

    println!("Model loaded: {}", model_id);

    // Create engine command channel
    let (cmd_tx, cmd_rx) = mpsc::channel(256);
    let engine_handle = EngineHandle::new(cmd_tx);

    // Clone tokenizer for engine (engine needs its own copy for decode)
    let engine_tokenizer = tokenizer.clone();

    // Create CacheManager for prefix caching (SSD-backed)
    let num_layers = model.num_layers();
    let cache_dir = dirs::cache_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("/tmp"))
        .join("ironmlx")
        .join("kv_cache");
    let model_hash = model_id.replace('/', "_");
    let ssd_config = ironmlx_core::cache::SSDStoreConfig {
        cache_dir: cache_dir.clone(),
        max_size_bytes: 10 * 1024 * 1024 * 1024, // 10 GB
        model_hash,
    };
    let ssd_store =
        ironmlx_core::cache::SSDStore::new(ssd_config).expect("failed to create SSD cache store");
    let cache_manager = ironmlx_core::cache::CacheManager::new(ssd_store, num_layers);
    println!("  KV cache dir: {}", cache_dir.display());

    // Spawn engine on a dedicated OS thread.
    // SAFETY: MLX C handles are reference-counted. EngineCore runs sequentially
    // on a single thread — no concurrent access to model state.
    let mut engine = EngineCore::with_cache_manager(cmd_rx, model, engine_tokenizer, cache_manager);
    std::thread::spawn(move || {
        engine.run();
    });

    // Build app state
    let state = Arc::new(AppState {
        engine: engine_handle,
        tokenizer: Arc::new(tokenizer),
        chat_template,
        eos_token_id,
        model_id: model_id.clone(),
        patch_size,
        spatial_merge_size,
    });

    println!("Listening on {}:{}", args.host, args.port);

    // Build router
    let app = api::router(state);

    // Start server
    let listener = tokio::net::TcpListener::bind(format!("{}:{}", args.host, args.port))
        .await
        .expect("Failed to bind address");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .expect("Server error");

    println!("Server shut down gracefully.");
}

async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("Failed to install CTRL+C handler");
    println!("\nShutdown signal received...");
}
