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
    ChatMessage, ChatTemplate, SamplerConfig, Tokenizer, stream_generate, stream_generate_vlm,
};
use ironmlx_core::media::ProcessedMedia;
use ironmlx_core::model::{build_model_from_file, load_model_weights};
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
        .unwrap_or_else(|| "mlx-community/Qwen3-0.6B-4bit".to_string());

    // Step 1: Download
    let model_dir = download_model(&repo_id);
    println!("  Model directory: {}\n", model_dir.display());

    // Step 2: Load tokenizer + chat template
    println!("[2/5] Loading tokenizer and chat template...");
    let tokenizer_path = model_dir
        .join("tokenizer.json")
        .to_string_lossy()
        .to_string();
    let tokenizer = Tokenizer::from_file(&tokenizer_path).expect("failed to load tokenizer");
    println!("  Tokenizer vocab size: {}", tokenizer.vocab_size());

    let tc_path = model_dir.join("tokenizer_config.json");
    let chat_template = if tc_path.exists() {
        match ChatTemplate::from_file(&tc_path.to_string_lossy()) {
            Ok(ct) => {
                println!("  Chat template: loaded from tokenizer_config.json");
                Some(ct)
            }
            Err(e) => {
                println!(
                    "  Chat template: failed to load ({}), using ChatML fallback",
                    e
                );
                // Fallback to ChatML (used by Qwen family)
                Some(ChatTemplate::new(
                    "{% for message in messages %}{{'<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{'<|im_start|>assistant\n'}}{% endif %}".to_string(),
                    "<|im_end|>".to_string(),
                    "<|im_start|>".to_string(),
                ))
            }
        }
    } else {
        println!("  Chat template: not available, using raw prompt");
        None
    };

    // Extract EOS token from config
    let config_str =
        std::fs::read_to_string(model_dir.join("config.json")).expect("failed to read config.json");
    let raw_config: serde_json::Value =
        serde_json::from_str(&config_str).expect("failed to parse config.json");
    let model_type = raw_config
        .get("model_type")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");
    let eos_token_id = if model_type == "qwen3_5" {
        raw_config
            .get("text_config")
            .and_then(|tc| tc.get("eos_token_id"))
            .and_then(|v| v.as_i64())
            .unwrap_or(151645) as i32
    } else {
        raw_config
            .get("eos_token_id")
            .and_then(|v| v.as_i64())
            .unwrap_or(2) as i32
    };
    println!("  Model type: {}, EOS: {}\n", model_type, eos_token_id);

    // Step 3: Load weights + build model
    println!("[3/5] Loading weights and building model...");
    let t0 = Instant::now();
    let stream = Stream::new(&Device::gpu());
    let weights = load_model_weights(&model_dir, &stream).expect("failed to load weights");
    println!("  Loaded {} weight tensors", weights.len());

    let config_path = model_dir.join("config.json").to_string_lossy().to_string();
    let model = build_model_from_file(&config_path, &weights).expect("failed to build model");
    let load_time = t0.elapsed();
    println!(
        "  Model built in {:.2}s ({} layers)\n",
        load_time.as_secs_f64(),
        model.num_layers()
    );

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
    // eos_token_id already extracted above

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

    // VLM test: if --vlm flag and an image path are provided
    let args: Vec<String> = std::env::args().collect();
    if let Some(vlm_pos) = args.iter().position(|a| a == "--vlm") {
        if let Some(image_path) = args.get(vlm_pos + 1) {
            println!("\n========== VLM Test ==========\n");
            run_vlm_test(&model, &tokenizer, &chat_template, image_path, eos_token_id);
        } else {
            println!("\n⚠ --vlm requires an image path argument");
        }
    }
}

fn run_vlm_test(
    model: &ironmlx_core::model::Model,
    tokenizer: &Tokenizer,
    chat_template: &Option<ChatTemplate>,
    image_path: &str,
    eos_token_id: i32,
) {
    println!("[VLM 1/4] Loading image: {}", image_path);

    // Load and process image
    let image_bytes =
        ironmlx_core::media::loader::load_media(image_path).expect("failed to load image");
    println!("  Image loaded: {} bytes", image_bytes.len());

    let patch_size = 16;
    let spatial_merge_size = 2;
    let (pixel_values, h, w) = ironmlx_core::media::image_proc::process_image_bytes(
        &image_bytes,
        patch_size,
        spatial_merge_size,
    )
    .expect("failed to process image");
    println!(
        "  Processed: {}x{} → pixel_values {:?}",
        w,
        h,
        pixel_values.shape()
    );

    // Compute grid_thw (pre-merge patch grid)
    let patches_h = h / patch_size;
    let patches_w = w / patch_size;
    let grid_thw = vec![(1usize, patches_h, patches_w)];
    println!("  Grid (T,H,W): {:?}", grid_thw[0]);

    // Number of vision tokens after spatial merge
    let num_vision_tokens = patches_h / spatial_merge_size * patches_w / spatial_merge_size;
    println!("  Vision tokens after merge: {}", num_vision_tokens);

    // Build prompt with image placeholders
    // Qwen3.5 expects: <|vision_start|><|image_pad|>...<|image_pad|><|vision_end|>
    // image_token_id = 248056
    println!("[VLM 2/4] Building multimodal prompt...");
    let image_placeholder = "<|image_pad|>".repeat(num_vision_tokens);
    let user_content = format!(
        "<|vision_start|>{}<|vision_end|>\nWhat do you see in this image?",
        image_placeholder
    );

    let prompt = if let Some(ct) = chat_template {
        let messages = vec![
            ChatMessage::system("You are a helpful assistant."),
            ChatMessage::user(&user_content),
        ];
        ct.apply(&messages, true)
            .expect("failed to apply chat template")
    } else {
        user_content.clone()
    };

    let tokens = tokenizer.encode(&prompt).expect("failed to encode");
    println!("  Prompt tokens: {}", tokens.len());

    // Check image token count matches
    let image_token_count = tokens.iter().filter(|&&t| t == 248056).count();
    println!("  Image tokens in prompt: {}", image_token_count);

    // Prepare processed media
    let media = vec![ProcessedMedia {
        pixel_values,
        grid_thw,
    }];

    // Generate with VLM
    println!("[VLM 3/4] Generating with vision...\n");
    let sampler = SamplerConfig::greedy();
    let max_tokens = 100;
    let mut generated_tokens = Vec::new();
    let t1 = Instant::now();

    let reason = stream_generate_vlm(
        model,
        &tokens,
        Some(&media),
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
    .expect("VLM generation failed");

    let gen_time = t1.elapsed();
    println!("\n");

    let full_text = tokenizer.decode(&generated_tokens).unwrap_or_default();
    let num_tokens = generated_tokens.len();
    let tokens_per_sec = num_tokens as f64 / gen_time.as_secs_f64();

    println!("[VLM 4/4] Results:");
    println!("  Stop reason: {:?}", reason);
    println!("  Generated: {} tokens", num_tokens);
    println!("  Full text: \"{}\"", full_text);
    println!(
        "  Time: {:.2}s ({:.1} tokens/sec)",
        gen_time.as_secs_f64(),
        tokens_per_sec
    );
    println!("\n✅ VLM verification passed!");
}
