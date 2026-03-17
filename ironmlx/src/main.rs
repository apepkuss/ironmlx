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

    // Spawn engine on a dedicated OS thread.
    // SAFETY: MLX C handles are reference-counted. EngineCore runs sequentially
    // on a single thread — no concurrent access to model state.
    let mut engine = EngineCore::new(cmd_rx, model, engine_tokenizer);
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
