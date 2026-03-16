//! End-to-end verification: download a small quantized model and run inference.
//!
//! Usage:
//!   cargo run --release --example verify_e2e
//!
//! This downloads `mlx-community/SmolLM-135M-Instruct-4bit` (~50MB) on first run
//! and generates text from a prompt using chat template + greedy decoding.

use std::path::PathBuf;
use std::time::Instant;

use ironmlx_core::device::Device;
use ironmlx_core::generate::{
    ChatMessage, ChatTemplate, SamplerConfig, Tokenizer, stream_generate,
};
use ironmlx_core::model::{LlamaModel, ModelConfig, load_model_weights};
use ironmlx_core::stream::Stream;

fn download_model(repo_id: &str) -> PathBuf {
    println!("[1/5] Downloading model: {}", repo_id);
    let api = hf_hub::api::sync::Api::new().expect("failed to create HF API");
    let repo = api.model(repo_id.to_string());

    for filename in &[
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors",
    ] {
        match repo.get(filename) {
            Ok(path) => println!("  ✓ {} -> {}", filename, path.display()),
            Err(e) => {
                println!("  ⚠ {} not found ({})", filename, e);
            }
        }
    }

    let config_path = repo.get("config.json").expect("config.json is required");
    config_path.parent().unwrap().to_path_buf()
}

fn main() {
    ironmlx_core::init();

    let repo_id = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "mlx-community/SmolLM-135M-Instruct-4bit".to_string());

    // Step 1: Download
    let model_dir = download_model(&repo_id);
    println!("  Model directory: {}\n", model_dir.display());

    // Step 2: Load config + tokenizer + chat template
    println!("[2/5] Loading config, tokenizer, and chat template...");
    let config_path = model_dir.join("config.json").to_string_lossy().to_string();
    let config = ModelConfig::from_file(&config_path).expect("failed to load config");
    println!(
        "  Model: {} | layers={} | heads={} | hidden={} | vocab={}",
        config.model_type,
        config.num_hidden_layers,
        config.num_attention_heads,
        config.hidden_size,
        config.vocab_size,
    );
    if let Some(ref qc) = config.quantization {
        println!(
            "  Quantization: {}bit, group_size={}",
            qc.bits, qc.group_size
        );
    } else {
        println!("  Quantization: none (full precision)");
    }

    let tokenizer_path = model_dir
        .join("tokenizer.json")
        .to_string_lossy()
        .to_string();
    let tokenizer = Tokenizer::from_file(&tokenizer_path).expect("failed to load tokenizer");
    println!("  Tokenizer vocab size: {}", tokenizer.vocab_size());

    // Load chat template
    let tc_path = model_dir.join("tokenizer_config.json");
    let chat_template = if tc_path.exists() {
        match ChatTemplate::from_file(&tc_path.to_string_lossy()) {
            Ok(ct) => {
                println!("  Chat template: loaded from tokenizer_config.json");
                Some(ct)
            }
            Err(e) => {
                println!("  Chat template: failed to load ({}), using raw prompt", e);
                None
            }
        }
    } else {
        println!("  Chat template: not available, using raw prompt");
        None
    };
    println!();

    // Step 3: Load weights + build model
    println!("[3/5] Loading weights and building model...");
    let t0 = Instant::now();
    let stream = Stream::new(&Device::gpu());
    let weights = load_model_weights(&model_dir, &stream).expect("failed to load weights");
    println!("  Loaded {} weight tensors", weights.len());

    let model = LlamaModel::from_config(&config, &weights).expect("failed to build model");
    let load_time = t0.elapsed();
    println!("  Model built in {:.2}s\n", load_time.as_secs_f64());

    // Step 4: Apply chat template
    let user_prompt = "What is the capital of France?";
    let prompt = if let Some(ref ct) = chat_template {
        let messages = vec![
            ChatMessage::system("You are a helpful assistant."),
            ChatMessage::user(user_prompt),
        ];
        let formatted = ct
            .apply(&messages, true)
            .expect("failed to apply chat template");
        println!("[4/5] Chat template applied:");
        println!("  --- formatted prompt ---");
        for line in formatted.lines() {
            println!("  | {}", line);
        }
        println!("  -------------------------\n");
        formatted
    } else {
        println!("[4/5] Using raw prompt (no chat template)\n");
        user_prompt.to_string()
    };

    // Step 5: Generate
    println!("[5/5] Generating text...");
    let tokens = tokenizer.encode(&prompt).expect("failed to encode");
    println!("  Prompt: {} tokens\n", tokens.len());

    let sampler = SamplerConfig::greedy();
    let max_tokens = 100;
    let eos_token_id = 2;

    let mut generated_tokens = Vec::new();
    let t1 = Instant::now();

    let reason = stream_generate(
        &model,
        &tokens,
        max_tokens,
        &sampler,
        eos_token_id,
        |token| {
            generated_tokens.push(token);
            if let Ok(text) = tokenizer.decode_single(token) {
                print!("{}", text);
            }
            true
        },
    )
    .expect("generation failed");

    let gen_time = t1.elapsed();
    println!("\n");

    let full_text = tokenizer.decode(&generated_tokens).unwrap_or_default();
    let num_tokens = generated_tokens.len();
    let tokens_per_sec = num_tokens as f64 / gen_time.as_secs_f64();

    println!("--- Results ---");
    println!("  Stop reason: {:?}", reason);
    println!("  Generated: {} tokens", num_tokens);
    println!("  Full text: \"{}\"", full_text);
    println!(
        "  Time: {:.2}s ({:.1} tokens/sec)",
        gen_time.as_secs_f64(),
        tokens_per_sec
    );
    println!("  Model load: {:.2}s", load_time.as_secs_f64());
    println!("\n✅ End-to-end verification passed!");
}
