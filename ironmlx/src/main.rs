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
    #[arg(long, default_value_t = 9068)]
    port: u16,

    /// Total memory limit in GB (0 = auto-detect from hardware)
    #[arg(long, default_value_t = 0.0)]
    memory_limit: f64,

    /// Hot cache (in-memory KV) limit in GB (0 = disabled)
    #[arg(long, default_value_t = 0.0)]
    hot_cache_limit: f64,

    /// Cold cache (SSD KV) limit in GB (0 = disabled)
    #[arg(long, default_value_t = 10.0)]
    cold_cache_limit: f64,

    /// Max concurrent sequences (default 16)
    #[arg(long, default_value_t = 16)]
    max_sequences: usize,

    /// Initial cache blocks (0 = auto-calculate from hot cache limit)
    #[arg(long, default_value_t = 0)]
    init_cache_blocks: usize,

    /// Disable all caching
    #[arg(long, default_value_t = false)]
    no_cache: bool,

    /// SSD cache directory
    #[arg(long)]
    cache_dir: Option<String>,
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
    let mut server_config = ServerConfig::load(&args.model, &args.host, args.port);
    server_config.max_num_seqs = args.max_sequences;
    if args.no_cache {
        server_config.hot_cache_max_size_gb = 0.0;
        server_config.cache_max_size_gb = 0.0;
    } else {
        server_config.hot_cache_max_size_gb = args.hot_cache_limit;
        server_config.cache_max_size_gb = args.cold_cache_limit;
    }
    if let Some(ref dir) = args.cache_dir {
        server_config.cache_dir = Some(dir.clone());
    }
    // Auto-calculate hot cache: 0 means auto = GPU total memory / 4
    let hot_cache_bytes = if server_config.hot_cache_max_size_gb == 0.0 && !args.no_cache {
        let total = ironmlx_core::memory::get_memory_size().unwrap_or(0) as u64;
        let auto_bytes = total / 4;
        println!(
            "  Hot cache: auto ({:.1}GB)",
            auto_bytes as f64 / 1_073_741_824.0
        );
        auto_bytes
    } else {
        server_config.hot_cache_max_size_bytes()
    };
    let cold_cache_bytes = server_config.cold_cache_max_size_bytes();
    let cache_dir = server_config.cache_dir.clone();
    let config = Arc::new(RwLock::new(server_config));

    println!("Loading model from: {}", args.model);

    // Create pool and load initial model
    let pool = EnginePool::new();
    let model_id = pool
        .load_model(
            &args.model,
            hot_cache_bytes,
            cold_cache_bytes,
            cache_dir.as_deref(),
            args.max_sequences,
            args.init_cache_blocks,
        )
        .expect("failed to load model");

    println!("Model loaded: {}", model_id);

    // Build app state
    let state = Arc::new(AppState {
        pool,
        started_at: chrono::Utc::now().timestamp(),
        config: config.clone(),
        log_buffer: state::LogBuffer::new(config.read().unwrap().log_buffer_size),
        downloads: std::sync::Mutex::new(std::collections::HashMap::new()),
        benchmark_history: std::sync::Mutex::new(Vec::new()),
        total_tokens: std::sync::atomic::AtomicU64::new(0),
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
