//! Performance benchmark for ironmlx inference.
//!
//! Usage:
//!   cargo run --release --example benchmark -- <model_repo_or_dir>
//!   cargo run --release --example benchmark -- mlx-community/Qwen3-0.6B-4bit --tokens 200

use std::time::Instant;

use ironmlx_core::device::Device;
use ironmlx_core::generate::{SamplerConfig, Tokenizer, stream_generate};
use ironmlx_core::memory;
use ironmlx_core::model::{build_model_from_file, load_model_weights};
use ironmlx_core::stream::Stream;

struct BenchmarkResult {
    model_id: String,
    prompt_tokens: usize,
    generated_tokens: usize,
    load_time_ms: f64,
    total_time_ms: f64,
    tokens_per_sec: f64,
    time_to_first_token_ms: f64,
    peak_memory_mb: f64,
}

impl BenchmarkResult {
    fn print(&self) {
        println!("+-----------------------------------------+");
        println!("|      ironmlx Benchmark Results          |");
        println!("+-----------------------------------------+");
        println!("| Model:       {:>26} |", self.model_id);
        println!("| Prompt:      {:>22} tok |", self.prompt_tokens);
        println!("| Generated:   {:>22} tok |", self.generated_tokens);
        println!("+-----------------------------------------+");
        println!("| Load:        {:>22.1} ms |", self.load_time_ms);
        println!("| TTFT:        {:>22.1} ms |", self.time_to_first_token_ms);
        println!("| Total:       {:>22.1} ms |", self.total_time_ms);
        println!("| Throughput:  {:>20.1} tok/s |", self.tokens_per_sec);
        println!("| Peak memory: {:>21.1} MB |", self.peak_memory_mb);
        println!("+-----------------------------------------+");
    }
}

fn main() {
    ironmlx_core::init();

    let args: Vec<String> = std::env::args().collect();
    let model_path = args
        .get(1)
        .expect("Usage: benchmark <model_path> [--prompt <text>] [--tokens <n>] [--runs <n>]");

    let prompt = args
        .iter()
        .position(|a| a == "--prompt")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
        .unwrap_or("Explain the theory of relativity in simple terms.");

    let max_tokens: usize = args
        .iter()
        .position(|a| a == "--tokens")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);

    let num_runs: usize = args
        .iter()
        .position(|a| a == "--runs")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);

    // Resolve model directory
    let model_dir = if std::path::Path::new(model_path).exists() {
        std::path::PathBuf::from(model_path)
    } else {
        println!("Downloading model: {}", model_path);
        let api = hf_hub::api::sync::Api::new().expect("HF API failed");
        let repo = api.model(model_path.to_string());
        for f in &[
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "model.safetensors",
        ] {
            let _ = repo.get(f);
        }
        repo.get("config.json")
            .unwrap()
            .parent()
            .unwrap()
            .to_path_buf()
    };

    println!(
        "Benchmark: {} (max_tokens={}, runs={})\n",
        model_path, max_tokens, num_runs
    );

    // Load model
    let _ = memory::reset_peak_memory();
    let t_load = Instant::now();
    let stream = Stream::new(&Device::gpu());
    let weights = load_model_weights(&model_dir, &stream).expect("failed to load weights");
    let config_path = model_dir.join("config.json").to_string_lossy().to_string();
    let model = build_model_from_file(&config_path, &weights).expect("failed to build model");
    let load_time_ms = t_load.elapsed().as_secs_f64() * 1000.0;

    // Load tokenizer
    let tokenizer_path = model_dir
        .join("tokenizer.json")
        .to_string_lossy()
        .to_string();
    let tokenizer = Tokenizer::from_file(&tokenizer_path).expect("failed to load tokenizer");

    // EOS token
    let config_str = std::fs::read_to_string(model_dir.join("config.json")).unwrap();
    let raw: serde_json::Value = serde_json::from_str(&config_str).unwrap();
    let model_type = raw
        .get("model_type")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");
    let eos_token_id = if model_type == "qwen3_5" {
        raw.get("text_config")
            .and_then(|tc| tc.get("eos_token_id"))
            .and_then(|v| v.as_i64())
            .unwrap_or(151645) as i32
    } else {
        raw.get("eos_token_id")
            .and_then(|v| v.as_i64())
            .unwrap_or(2) as i32
    };

    let model_id = model_path.split('/').last().unwrap_or(model_path);
    let tokens = tokenizer.encode(prompt).expect("failed to encode");
    println!("Prompt: \"{}\" ({} tokens)\n", prompt, tokens.len());

    let sampler = SamplerConfig::greedy();

    for run in 0..num_runs {
        if num_runs > 1 {
            println!("--- Run {}/{} ---", run + 1, num_runs);
        }

        let _ = memory::reset_peak_memory();
        let mut generated_count = 0usize;
        let mut first_token_time: Option<f64> = None;
        let t_gen = Instant::now();

        let _reason = stream_generate(
            &model,
            &tokens,
            max_tokens,
            &sampler,
            eos_token_id,
            |_token| {
                generated_count += 1;
                if first_token_time.is_none() {
                    first_token_time = Some(t_gen.elapsed().as_secs_f64() * 1000.0);
                }
                true
            },
        )
        .expect("generation failed");

        let total_time_ms = t_gen.elapsed().as_secs_f64() * 1000.0;
        let ttft = first_token_time.unwrap_or(0.0);
        let decode_time_ms = total_time_ms - ttft;
        let tokens_per_sec = if decode_time_ms > 0.0 && generated_count > 1 {
            (generated_count - 1) as f64 / (decode_time_ms / 1000.0)
        } else {
            0.0
        };
        let peak_memory_mb = memory::get_peak_memory().unwrap_or(0) as f64 / 1_048_576.0;

        let result = BenchmarkResult {
            model_id: model_id.to_string(),
            prompt_tokens: tokens.len(),
            generated_tokens: generated_count,
            load_time_ms,
            total_time_ms,
            tokens_per_sec,
            time_to_first_token_ms: ttft,
            peak_memory_mb,
        };

        println!();
        result.print();
    }
}
