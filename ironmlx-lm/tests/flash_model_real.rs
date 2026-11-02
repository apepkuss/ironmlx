//! Opt-in complete text graph bring-up. This uses a private request cache;
//! passing does not expose the model through the production scheduler.
use ironmlx_lm::{
    core::{tensor_file::ModelTensorFiles, Loader},
    models::qwen4_exp::text_model::FlashTextModel,
};
use mlx::Dtype;
use sha2::{Digest, Sha256};
use std::{path::Path, sync::Arc};

#[test]
#[ignore = "requires IRONMLX_FLASH_SNAPSHOT, M5 Max and at least 128 GiB RAM"]
fn real_flash_text_prefill_and_decode() {
    let snapshot = std::env::var("IRONMLX_FLASH_SNAPSHOT").expect("snapshot required");
    mlx::memory::set_cache_limit(512 * 1024 * 1024);
    mlx::memory::set_wired_limit(96usize << 30).unwrap();
    let start = std::time::Instant::now();
    let loader = Loader::open_flash_text(Path::new(&snapshot), 82u64 << 30).unwrap();
    eprintln!(
        "loaded_bytes={} elapsed={:?}",
        loader.loaded_tensor_bytes(),
        start.elapsed()
    );
    let files = Arc::new(ModelTensorFiles::open(Path::new(&snapshot)).unwrap());
    let model = FlashTextModel::from_loader(
        &loader,
        files,
        32 * 1024 * 1024,
        16 * 1024 * 1024,
        64 * 1024 * 1024,
    )
    .unwrap();
    drop(loader);
    let tokenizer =
        tokenizers::Tokenizer::from_file(Path::new(&snapshot).join("tokenizer.json")).unwrap();
    let prompt = "<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n";
    let tokens = tokenizer.encode(prompt, false).unwrap().get_ids().to_vec();
    let oracle = std::env::var("IRONMLX_FLASH_MODEL_ORACLE")
        .ok()
        .map(|path| {
            let metadata: serde_json::Value = serde_json::from_reader(
                std::fs::File::open(Path::new(&path).with_extension("json")).unwrap(),
            )
            .unwrap();
            assert_eq!(
                metadata["reference_commit"],
                "ff788e98d82f6a931992a6a8ee0e74c7a7e6925e"
            );
            assert_eq!(
                metadata["mlx_commit"],
                "a8f66b5901791a38ff527f249e1f1a84f46c03b6"
            );
            for (filename, field) in [
                ("config.json", "config_sha256"),
                ("model.safetensors.index.json", "index_sha256"),
            ] {
                assert_eq!(
                    metadata[field],
                    format!(
                        "{:x}",
                        Sha256::digest(std::fs::read(Path::new(&snapshot).join(filename)).unwrap())
                    )
                );
            }
            assert_eq!(metadata["prompt_tokens"], serde_json::json!(tokens));
            assert_eq!(metadata["training"], false);
            let (arrays, _) = mlx::io::load_safetensors(&path).unwrap();
            (metadata, arrays)
        });
    let mut cache = model.new_cache(512).unwrap();
    let captures: Vec<usize> = if oracle.is_some() {
        (0..model.config.num_hidden_layers).collect()
    } else {
        Vec::new()
    };
    let mut output = model
        .forward_tokens_on(&tokens, &mut cache, &captures, ().into())
        .unwrap();
    if let Some((_, arrays)) = &oracle {
        for (index, capture) in output.captures.iter().enumerate() {
            let actual = capture
                .astype(Dtype::Float32)
                .unwrap()
                .to_vec::<f32>()
                .unwrap();
            let expected = arrays[&format!("prefill_layer_{index}")]
                .astype(Dtype::Float32)
                .unwrap()
                .to_vec::<f32>()
                .unwrap();
            let max = actual
                .iter()
                .zip(&expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            let rms = (actual
                .iter()
                .zip(&expected)
                .map(|(a, b)| ((a - b) as f64).powi(2))
                .sum::<f64>()
                / actual.len() as f64)
                .sqrt();
            eprintln!("prefill layer={index} max_abs={max} rms={rms}");
            assert!(
                max == 0.0 && rms == 0.0,
                "Flash prefill reference drift at layer {index}"
            );
        }
    }
    let mut generated = Vec::new();
    let mut top1_matches = 0;
    let mut reference_ok = true;
    for step in 0..24 {
        let logits = output
            .logits
            .astype(Dtype::Float32)
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        assert!(logits.iter().all(|n| n.is_finite()));
        let last = &logits[logits.len() - model.config.vocab_size as usize..];
        let next = last
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0 as u32;
        let feed = if let Some((metadata, arrays)) = &oracle {
            let expected = arrays[&format!("logits_{step}")]
                .astype(Dtype::Float32)
                .unwrap()
                .to_vec::<f32>()
                .unwrap();
            let max = last
                .iter()
                .zip(&expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            let rms = (last
                .iter()
                .zip(&expected)
                .map(|(a, b)| ((a - b) as f64).powi(2))
                .sum::<f64>()
                / last.len() as f64)
                .sqrt();
            let expected_token = metadata["generated_tokens"][step].as_u64().unwrap() as u32;
            top1_matches += usize::from(next == expected_token);
            eprintln!("reference step={step} max_abs={max} rms={rms} native_top1={next} reference_top1={expected_token}");
            reference_ok &= max == 0.0 && rms == 0.0;
            expected_token
        } else {
            next
        };
        generated.push(feed);
        if [248044, 248046].contains(&feed) {
            break;
        }
        output = model
            .forward_tokens_on(&[feed], &mut cache, &[], ().into())
            .unwrap();
    }
    eprintln!(
        "generated={generated:?} text={:?} elapsed={:?}",
        tokenizer.decode(&generated, false).unwrap(),
        start.elapsed()
    );
    assert!(!generated.is_empty());
    assert!(reference_ok, "Flash full-model reference drift");
    if oracle.is_some() {
        eprintln!(
            "fixed-trajectory top1 matches={top1_matches}/{}",
            generated.len()
        );
    }
    eprintln!("memory={:?}", mlx::memory::snapshot());
}
