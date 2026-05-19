//! P5d T4: dump first-step logits from Qwen35MoeModel for 5 prompts.
//! Output: reports/p5d-argmax/ironmlx_logits_p<N>.npy (N=0..4)
//!
//! Run with:
//!   IRONMLX_MOE_MODEL_DIR=<snap> MLX_DIR=$HOME/.local/mlx \
//!     cargo test -p ironmlx --release --test p5_qwen35_moe_logits_dump \
//!     -- --ignored --nocapture --test-threads=1

use mlx::Dtype;

use ironmlx::core::generate::build_position_ids;
use ironmlx::core::{Loader, Model, Tokenizer};
use ironmlx::models::qwen3_5_moe::MIN_KV_CACHE_CAP_FOR_GPU_PERF;
use ironmlx::models::Qwen35MoeModel;

const PROMPTS: [&str; 5] = [
    "Once upon a time, in a small village,",
    "The quick brown fox jumps over",
    "def fibonacci(n):\n    if n < 2:",
    "List three reasons why exercise is important:",
    "Translate to French: Good morning.",
];

fn locate_snapshot() -> String {
    if let Ok(p) = std::env::var("IRONMLX_MOE_MODEL_DIR") {
        return p;
    }
    let home = std::env::var("HOME").expect("HOME env");
    let snapshots =
        format!("{home}/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots");
    let entries = std::fs::read_dir(&snapshots).expect("snapshots dir missing");
    let first = entries
        .filter_map(|e| e.ok())
        .next()
        .expect("at least one snapshot");
    first.path().to_string_lossy().into_owned()
}

/// Minimal NPY writer for 1D fp32 array. NPY format:
/// magic(\x93NUMPY) + ver(1.0) + header_len(u16 le) + header(\n-padded) + data(little-endian)
fn write_npy_f32(path: &str, data: &[f32]) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;
    // Magic + version
    f.write_all(b"\x93NUMPY\x01\x00")?;
    // Header dict
    let header = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': ({},), }}",
        data.len()
    );
    // Pad to multiple of 64 bytes (incl. magic+ver+header_len field)
    let prefix_len = 10usize; // 6 (magic) + 2 (ver) + 2 (header_len field)
    let total_pre_pad = prefix_len + header.len() + 1; // +1 for trailing '\n'
    let pad_to = ((total_pre_pad + 63) / 64) * 64;
    let pad_count = pad_to - total_pre_pad;
    let padded_header = format!("{header}{}\n", " ".repeat(pad_count));
    let header_bytes = padded_header.as_bytes();
    let header_len_u16 = header_bytes.len() as u16;
    f.write_all(&header_len_u16.to_le_bytes())?;
    f.write_all(header_bytes)?;
    // Data: little-endian fp32
    for &x in data {
        f.write_all(&x.to_le_bytes())?;
    }
    Ok(())
}

#[test]
#[ignore]
fn p5d_dump_first_token_logits_for_5_prompts() {
    let dir = locate_snapshot();
    eprintln!("[T4-ironmlx] loading model from {dir}");
    let loader = Loader::open(std::path::Path::new(&dir)).expect("Loader::open");
    let tokenizer = Tokenizer::from_loader(&loader).expect("Tokenizer::from_loader");
    let model = Qwen35MoeModel::from_loader(&loader).expect("Qwen35MoeModel::from_loader");
    eprintln!(
        "[T4-ironmlx] model loaded: {} layers",
        model.config().num_hidden_layers
    );

    std::fs::create_dir_all("reports/p5d-argmax").expect("mkdir reports/p5d-argmax");

    for (idx, &prompt) in PROMPTS.iter().enumerate() {
        eprintln!("[T4-ironmlx] prompt {idx}: {prompt:.60}");
        let prompt_ids = tokenizer
            .encode(prompt, /* add_special_tokens */ false)
            .expect("encode");
        let s = prompt_ids.len() as i32;
        let ids_i32: Vec<i32> = prompt_ids.iter().map(|&t| t as i32).collect();
        let input_ids: mlx::Array = (&ids_i32[..], &[1_i32, s][..]).try_into().unwrap();
        let pos = build_position_ids(0, s).expect("build_position_ids");

        let cap = (s + 4).max(MIN_KV_CACHE_CAP_FOR_GPU_PERF);
        let mut cache = Model::make_cache(&model, 1, cap, Dtype::Bfloat16).expect("make_cache");
        let logits = Model::forward_on(
            &model,
            &input_ids,
            &pos,
            None,
            None,
            Some(&mut cache),
            mlx::StreamOrDevice::default(),
        )
        .expect("forward_on");

        // logits shape [1, 1, vocab] — flatten to 1D
        let v: Vec<f32> = mlx::ops::cast::astype(&logits, Dtype::Float32)
            .unwrap()
            .to_vec()
            .unwrap();

        let path = format!("reports/p5d-argmax/ironmlx_logits_p{idx}.npy");
        write_npy_f32(&path, &v).expect("write_npy");

        let argmax = v
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        eprintln!(
            "[T4-ironmlx] saved p{idx} ({} fp32 elements), argmax={argmax}",
            v.len()
        );
    }

    eprintln!("[T4-ironmlx] done — 5 .npy files in reports/p5d-argmax/");
}
