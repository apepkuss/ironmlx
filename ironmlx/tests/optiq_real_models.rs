//! Real-checkpoint production gates for OptiQ quantization.
//!
//! Skipped unless IRONMLX_TEST_REAL_OPTIQ=1 because the checkpoints are large.
//! 35B MoE checkpoints are gated separately by IRONMLX_TEST_REAL_OPTIQ_MOE=1.

use std::path::PathBuf;

use ironmlx::core::generate::{GenerateRequest, GenerationStream, IMAGE_TOKEN_ID};
use ironmlx::core::{Model, QuantMode, Sampler, Tokenizer};
use ironmlx::models::{
    is_qwen36_moe_config, Gemma4Model, Qwen35Model, Qwen35MoeModel, Qwen36MoeModel,
};
use ironmlx::Loader;

fn snapshot_dir(repo: &str) -> Option<PathBuf> {
    let home = dirs::home_dir()?;
    let base = home.join(format!(
        ".ironmlx/models/models--mlx-community--{repo}/snapshots"
    ));
    let entries = std::fs::read_dir(&base).ok()?;
    entries.flatten().find_map(|entry| {
        let path = entry.path();
        path.is_dir().then_some(path)
    })
}

fn should_run() -> bool {
    std::env::var_os("IRONMLX_TEST_REAL_OPTIQ").as_deref() == Some("1".as_ref())
}

fn should_run_moe() -> bool {
    std::env::var_os("IRONMLX_TEST_REAL_OPTIQ_MOE").as_deref() == Some("1".as_ref())
}

fn assert_optiq_loader_contract(loader: &Loader, expected_global_bits: i32) {
    let global = loader.quant_meta().expect("global OptiQ quant meta");
    assert_eq!(global.mode, QuantMode::OptiQ);
    assert_eq!(global.bits, expected_global_bits);
    assert_eq!(global.group_size, 64);

    let mut quantized_prefixes = 0usize;
    for key in loader.keys().filter(|key| key.ends_with(".scales")) {
        let prefix = key.strip_suffix(".scales").expect("suffix just matched");
        let meta = loader
            .quant_meta_for(prefix)
            .unwrap_or_else(|| panic!("{prefix}: missing quant meta"));
        assert_eq!(meta.mode, QuantMode::OptiQ, "{prefix}");
        assert_eq!(meta.group_size, 64, "{prefix}");
        assert!(
            matches!(meta.bits, 2 | 4 | 8),
            "{prefix}: unexpected OptiQ bit width {}",
            meta.bits
        );
        assert!(
            loader.contains(&format!("{prefix}.weight")),
            "{prefix}: missing packed weight"
        );
        assert!(
            loader.contains(&format!("{prefix}.biases")),
            "{prefix}: missing OptiQ affine-compatible biases"
        );
        quantized_prefixes += 1;
    }
    assert!(
        quantized_prefixes > 0,
        "checkpoint has no quantized tensors"
    );
}

#[derive(Debug)]
struct MoeOptiqStats {
    switch_mlp_layers: usize,
    mixed_gate_up_layers: usize,
    mixed_triplet_layers: usize,
}

fn assert_moe_switch_mlp_optiq_contract(loader: &Loader) -> MoeOptiqStats {
    let mut switch_mlp_prefixes = loader
        .keys()
        .filter_map(|key| key.strip_suffix(".gate_proj.scales"))
        .filter(|prefix| prefix.contains(".switch_mlp"))
        .map(str::to_owned)
        .collect::<Vec<_>>();
    switch_mlp_prefixes.sort();
    switch_mlp_prefixes.dedup();

    assert!(
        !switch_mlp_prefixes.is_empty(),
        "checkpoint has no switch_mlp routed expert tensors"
    );

    let mut stats = MoeOptiqStats {
        switch_mlp_layers: 0,
        mixed_gate_up_layers: 0,
        mixed_triplet_layers: 0,
    };
    for prefix in switch_mlp_prefixes {
        let gate_prefix = format!("{prefix}.gate_proj");
        let up_prefix = format!("{prefix}.up_proj");
        let down_prefix = format!("{prefix}.down_proj");
        let gate = loader
            .quant_meta_for(&gate_prefix)
            .unwrap_or_else(|| panic!("{gate_prefix}: missing quant meta"));
        let up = loader
            .quant_meta_for(&up_prefix)
            .unwrap_or_else(|| panic!("{up_prefix}: missing quant meta"));
        let down = loader
            .quant_meta_for(&down_prefix)
            .unwrap_or_else(|| panic!("{down_prefix}: missing quant meta"));
        for (name, meta) in [
            (gate_prefix.as_str(), gate),
            (up_prefix.as_str(), up),
            (down_prefix.as_str(), down),
        ] {
            assert_eq!(meta.mode, QuantMode::OptiQ, "{name}");
            assert_eq!(meta.group_size, 64, "{name}");
            assert!(
                matches!(meta.bits, 2 | 4 | 8),
                "{name}: unexpected OptiQ bit width {}",
                meta.bits
            );
        }
        if gate != up {
            stats.mixed_gate_up_layers += 1;
        }
        if gate != up || gate != down {
            stats.mixed_triplet_layers += 1;
        }
        stats.switch_mlp_layers += 1;
    }
    stats
}

fn assert_short_text_generation<M: Model>(model: &M, tokenizer: &Tokenizer) {
    let prompt_ids = tokenizer
        .encode("Hello", true)
        .expect("tokenizer encode")
        .into_iter()
        .collect::<Vec<_>>();
    assert!(!prompt_ids.is_empty());

    let request = GenerateRequest {
        prompt_ids,
        max_new_tokens: 1,
        sampler: Sampler::greedy(),
        stop_token_ids: tokenizer.eos_token_ids().to_vec(),
        prefill_chunk_size: 0,
        decode_cadence_mid_chunk_cap: 0,
        kv_cache_turboquant_bits: None,
        pixel_values: None,
        image_grid_thw: None,
        image_spatial_merge_size: 2,
        image_token_id: IMAGE_TOKEN_ID,
    };
    let mut stream =
        GenerationStream::new_text_only(model, tokenizer, request).expect("GenerationStream");
    let event = stream
        .next_token()
        .expect("decode next token")
        .expect("one generated event");
    assert!(
        event.finish_reason.is_some(),
        "max_new_tokens=1 must finish"
    );
}

#[test]
fn qwen35_moe_optiq_real_checkpoint_loads_and_generates_when_requested() {
    if !should_run_moe() {
        eprintln!("IRONMLX_TEST_REAL_OPTIQ_MOE=1 not set; skipping real OptiQ Qwen3.5 MoE gate");
        return;
    }
    let Some(dir) = snapshot_dir("Qwen3.5-35B-A3B-OptiQ-4bit") else {
        eprintln!("Qwen3.5-35B-A3B-OptiQ-4bit cache absent; skipping");
        return;
    };

    let loader = Loader::open(&dir).expect("Loader::open Qwen3.5 MoE OptiQ");
    assert_optiq_loader_contract(&loader, 4);
    let stats = assert_moe_switch_mlp_optiq_contract(&loader);
    assert!(stats.switch_mlp_layers > 0);
    assert!(
        stats.mixed_triplet_layers > 0,
        "Qwen3.5 MoE OptiQ gate should exercise mixed routed-expert metadata, got {stats:?}"
    );

    let tokenizer = Tokenizer::from_loader(&loader).expect("Tokenizer::from_loader");
    let model = Qwen35MoeModel::from_loader(&loader).expect("Qwen35MoeModel::from_loader OptiQ");
    assert_short_text_generation(&model, &tokenizer);
}

#[test]
fn qwen36_moe_optiq_real_checkpoint_loads_and_generates_when_requested() {
    if !should_run_moe() {
        eprintln!("IRONMLX_TEST_REAL_OPTIQ_MOE=1 not set; skipping real OptiQ Qwen3.6 MoE gate");
        return;
    }
    let Some(dir) = snapshot_dir("Qwen3.6-35B-A3B-OptiQ-4bit") else {
        eprintln!("Qwen3.6-35B-A3B-OptiQ-4bit cache absent; skipping");
        return;
    };

    let loader = Loader::open(&dir).expect("Loader::open Qwen3.6 MoE OptiQ");
    assert_optiq_loader_contract(&loader, 4);
    assert!(
        is_qwen36_moe_config(loader.config_raw_value()),
        "checkpoint must be recognized by Qwen3.6 MoE facade"
    );
    let stats = assert_moe_switch_mlp_optiq_contract(&loader);
    assert!(stats.switch_mlp_layers > 0);
    assert!(
        stats.mixed_gate_up_layers > 0,
        "Qwen3.6 MoE OptiQ gate should exercise split gate/up dispatch, got {stats:?}"
    );

    let tokenizer = Tokenizer::from_loader(&loader).expect("Tokenizer::from_loader");
    let model = Qwen36MoeModel::from_loader(&loader).expect("Qwen36MoeModel::from_loader OptiQ");
    assert_short_text_generation(&model, &tokenizer);
}

#[test]
fn qwen35_optiq_real_checkpoint_loads_and_generates_when_requested() {
    if !should_run() {
        eprintln!("IRONMLX_TEST_REAL_OPTIQ=1 not set; skipping real OptiQ Qwen3.5 gate");
        return;
    }
    let Some(dir) = snapshot_dir("Qwen3.5-2B-OptiQ-4bit") else {
        eprintln!("Qwen3.5-2B-OptiQ-4bit cache absent; skipping");
        return;
    };

    let loader = Loader::open(&dir).expect("Loader::open Qwen3.5 OptiQ");
    assert_optiq_loader_contract(&loader, 4);
    assert_eq!(
        loader
            .quant_meta_for("model.embed_tokens")
            .expect("embed OptiQ meta")
            .bits,
        8
    );
    assert_eq!(
        loader
            .quant_meta_for("model.layers.0.linear_attn.in_proj_qkv")
            .expect("linear_attn OptiQ meta")
            .bits,
        8
    );
    assert_eq!(
        loader
            .quant_meta_for("model.layers.7.self_attn.q_proj")
            .expect("full attention OptiQ meta")
            .bits,
        4
    );

    let tokenizer = Tokenizer::from_loader(&loader).expect("Tokenizer::from_loader");
    let model = Qwen35Model::from_loader(&loader).expect("Qwen35Model::from_loader OptiQ");
    assert_short_text_generation(&model, &tokenizer);
}

#[test]
fn gemma4_optiq_real_checkpoint_loads_and_generates_when_requested() {
    if !should_run() {
        eprintln!("IRONMLX_TEST_REAL_OPTIQ=1 not set; skipping real OptiQ Gemma4 gate");
        return;
    }
    let Some(dir) = snapshot_dir("gemma-4-e4b-it-OptiQ-4bit") else {
        eprintln!("gemma-4-e4b-it-OptiQ-4bit cache absent; skipping");
        return;
    };

    let loader = Loader::open(&dir).expect("Loader::open Gemma4 OptiQ");
    assert_optiq_loader_contract(&loader, 4);
    assert_eq!(
        loader
            .quant_meta_for("model.layers.19.self_attn.q_proj")
            .expect("layer 19 q_proj OptiQ meta")
            .bits,
        4
    );
    assert_eq!(
        loader
            .quant_meta_for("model.layers.19.self_attn.k_proj")
            .expect("layer 19 k_proj OptiQ meta")
            .bits,
        8
    );
    assert_eq!(
        loader
            .quant_meta_for("model.layers.1.mlp.up_proj")
            .expect("layer 1 up_proj OptiQ meta")
            .bits,
        4
    );

    let tokenizer = Tokenizer::from_loader(&loader).expect("Tokenizer::from_loader");
    let model = Gemma4Model::from_loader(&loader).expect("Gemma4Model::from_loader OptiQ");
    assert_short_text_generation(&model, &tokenizer);
}
