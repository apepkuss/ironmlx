mod api;
mod engine;
mod engine_handle;
mod engine_pool;
mod state;

use std::sync::Arc;

use clap::Parser;
use engine_pool::EnginePool;
use state::AppState;

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

    // Create pool and load initial model
    let pool = EnginePool::new();
    let model_id = pool.load_model(&args.model).expect("failed to load model");

    println!("Model loaded: {}", model_id);

    // Build app state
    let state = Arc::new(AppState { pool });

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
