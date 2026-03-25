mod admin;
mod api;
mod config;
mod engine;
mod engine_handle;
mod engine_pool;
mod hardware;
mod memory_guard;
mod state;

use std::sync::Arc;

use std::sync::RwLock;

use clap::Parser;
use config::ServerConfig;
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

    /// Total memory limit in GB (0 = auto-detect from hardware)
    #[arg(long, default_value_t = 0.0)]
    memory_limit: f64,
}

#[tokio::main]
async fn main() {
    // Install panic hook to log crashes
    std::panic::set_hook(Box::new(|info| {
        let backtrace = std::backtrace::Backtrace::force_capture();
        let msg = format!("PANIC: {}\n\nBacktrace:\n{}", info, backtrace);
        eprintln!("{}", msg);
        let _ = std::fs::write("/tmp/ironmlx-panic.log", &msg);
    }));

    let args = Args::parse();

    println!("ironmlx starting...");

    // Initialize MLX runtime
    ironmlx_core::init();

    // Detect hardware
    let chip = hardware::ChipInfo::detect();
    println!("  Hardware: {}", chip);

    // Set memory limit
    let mem_limit = if args.memory_limit > 0.0 {
        (args.memory_limit * 1024.0 * 1024.0 * 1024.0) as usize
    } else {
        chip.recommended_memory_limit()
    };
    let guard = memory_guard::MemoryGuard::new(mem_limit, 0.9);
    println!(
        "  Memory limit: {:.1}GB",
        mem_limit as f64 / 1_073_741_824.0
    );
    let _ = guard; // guard lives for duration of process via set_memory_limit

    // Load config with CLI overrides
    let server_config = ServerConfig::load(&args.model, &args.host, args.port);
    let config = Arc::new(RwLock::new(server_config));

    println!("Loading model from: {}", args.model);

    // Create pool and load initial model
    let pool = EnginePool::new();
    let model_id = pool.load_model(&args.model).expect("failed to load model");

    println!("Model loaded: {}", model_id);

    // Build app state
    let state = Arc::new(AppState {
        pool,
        started_at: chrono::Utc::now().timestamp(),
        config: config.clone(),
        log_buffer: state::LogBuffer::new(config.read().unwrap().log_buffer_size),
        downloads: std::sync::Mutex::new(std::collections::HashMap::new()),
        benchmark_history: std::sync::Mutex::new(Vec::new()),
    });

    // Log startup event
    state.log_buffer.push(
        "info",
        &format!("Server started, model loaded: {}", model_id),
    );

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
