//! Real-weight component qualification against pinned MLX-VLM, not a full
//! model or serving capability gate. Generate the oracle with the paired
//! scripts/qualify-qwen38-flash-next-layers.py command first.
use ironmlx_lm::{
    core::{
        cache::GatedDeltaCache,
        tensor_file::ModelTensorFiles,
        weights::{QuantMeta, QuantMode, WeightMap},
    },
    models::qwen4_exp::{
        attention::{QsaAttention, QsaState},
        config::FlashTextConfig,
        hyper::HyperConnection,
        ple::PleLayer,
        ple_rows::PleRows,
    },
    nn::GatedDeltaNet,
};
use mlx::{Array, Dtype};
use sha2::{Digest, Sha256};
use std::{collections::HashMap, path::Path, sync::Arc};

fn compare(name: &str, actual: &Array, expected: &Array) {
    assert_eq!(actual.shape(), expected.shape(), "{name} shape");
    assert_eq!(actual.dtype(), expected.dtype(), "{name} dtype");
    let a = actual
        .astype(Dtype::Float32)
        .unwrap()
        .to_vec::<f32>()
        .unwrap();
    let b = expected
        .astype(Dtype::Float32)
        .unwrap()
        .to_vec::<f32>()
        .unwrap();
    let mut peak = 0f32;
    let mut max = 0f32;
    let mut square = 0f64;
    let mut reference_square = 0f64;
    for (&a, &b) in a.iter().zip(&b) {
        assert!(a.is_finite() && b.is_finite());
        peak = peak.max(b.abs());
        max = max.max((a - b).abs());
        square += ((a - b) as f64).powi(2);
        reference_square += (b as f64).powi(2);
    }
    let relative_rms = (square / reference_square.max(1e-20)).sqrt();
    eprintln!("{name}: max_abs={max} reference_peak={peak} relative_rms={relative_rms}");
    // Independent BF16 graphs can differ by one rounding step. Require a
    // tight aggregate bound too, so a small subset of large values cannot
    // conceal a systematically wrong gate or normalization convention.
    assert!(
        max <= peak * 0.02 + 1e-6 && relative_rms <= 0.005,
        "{name} reference mismatch"
    );
}

#[test]
#[ignore = "requires IRONMLX_FLASH_SNAPSHOT and IRONMLX_FLASH_LAYER_ORACLE"]
fn real_hc_gdn_matches_pinned_reference() {
    let snapshot = std::env::var("IRONMLX_FLASH_SNAPSHOT").expect("snapshot required");
    let oracle_path = std::env::var("IRONMLX_FLASH_LAYER_ORACLE").expect("oracle required");
    let metadata: serde_json::Value = serde_json::from_reader(
        std::fs::File::open(Path::new(&oracle_path).with_extension("json")).unwrap(),
    )
    .unwrap();
    assert_eq!(
        metadata["reference_commit"],
        "ff788e98d82f6a931992a6a8ee0e74c7a7e6925e"
    );
    assert_eq!(metadata["training"], false);
    assert_eq!(
        metadata["mlx_commit"],
        "a8f66b5901791a38ff527f249e1f1a84f46c03b6"
    );
    for (filename, field) in [
        ("config.json", "config_sha256"),
        ("model.safetensors.index.json", "index_sha256"),
    ] {
        let hash = format!(
            "{:x}",
            Sha256::digest(std::fs::read(Path::new(&snapshot).join(filename)).unwrap())
        );
        assert_eq!(metadata[field], hash);
    }
    let config = serde_json::from_reader(
        std::fs::File::open(Path::new(&snapshot).join("config.json")).unwrap(),
    )
    .unwrap();
    let cfg = FlashTextConfig::from_root(&config).unwrap();
    let files = Arc::new(ModelTensorFiles::open(Path::new(&snapshot)).unwrap());
    let mut tensors = HashMap::new();
    let mut payload_hash = Sha256::new();
    for (name, info) in files.tensors().filter(|(name, _)| {
        name.starts_with("language_model.model.layers.0.linear_attn.")
            || name.starts_with("language_model.model.layers.0.attn_hyper_connection.")
            || name.starts_with("language_model.model.layers.3.self_attn.")
            || (name.starts_with("language_model.model.layers.1.ple.")
                && !name.contains(".ngram_embedding."))
    }) {
        payload_hash.update(
            files
                .read_rows(name, 0, info.shape[0], 64 * 1024 * 1024)
                .unwrap(),
        );
        let array = files
            .read_rows_array(name, 0, info.shape[0], 64 * 1024 * 1024)
            .unwrap();
        tensors.insert(
            name.strip_prefix("language_model.").unwrap().to_owned(),
            array,
        );
    }
    assert_eq!(
        metadata["selected_payload_sha256"],
        format!("{:x}", payload_hash.finalize())
    );
    let weights = WeightMap::new(
        tensors,
        Some(QuantMeta {
            group_size: 32,
            bits: 4,
            mode: QuantMode::Affine,
        }),
        HashMap::new(),
    );
    let hc =
        HyperConnection::from_weights(&weights, "model.layers.0.attn_hyper_connection", &cfg, true)
            .unwrap();
    let gdn =
        GatedDeltaNet::from_flash_weights(&weights, "model.layers.0.linear_attn", &cfg).unwrap();
    let (oracle, _) = mlx::io::load_safetensors(&oracle_path).unwrap();
    let mut cache = GatedDeltaCache::new_with_cap(
        1,
        cfg.linear_conv_kernel_dim,
        2 * cfg.linear_num_key_heads * cfg.linear_key_head_dim
            + cfg.linear_num_value_heads * cfg.linear_value_head_dim,
        cfg.linear_num_value_heads,
        cfg.linear_value_head_dim,
        cfg.linear_key_head_dim,
        Dtype::Bfloat16,
        64,
    )
    .unwrap();
    let branch = hc.branch_on(&oracle["hyper"], ().into()).unwrap();
    compare("hc_mixed", &branch.input, &oracle["mixed"]);
    let output = gdn
        .forward_on(&branch.input, None, None, Some(&mut cache), (), 0)
        .unwrap();
    compare("gdn_output", &output, &oracle["output"]);
    compare(
        "hc_combined",
        &branch.combine_on(&output, ().into()).unwrap(),
        &oracle["combined"],
    );
    compare("gdn_conv_state", cache.conv_state(), &oracle["conv_state"]);
    compare(
        "gdn_recurrent_state",
        cache.recurrent_state(),
        &oracle["recurrent_state"],
    );
    assert_eq!(
        cache.offsets(),
        &[metadata["sequence_length"].as_i64().unwrap() as i32]
    );
    let attention =
        QsaAttention::from_weights(&weights, "model.layers.3.self_attn", &cfg, 64 * 1024 * 1024)
            .unwrap();
    let mut state = QsaState::new(4096).unwrap();
    let output = attention
        .forward_on(&oracle["attention_input"], &mut state, ().into())
        .unwrap();
    compare("qsa_output", &output, &oracle["attention_output"]);
    compare(
        "qsa_keys",
        state.keys.as_ref().unwrap(),
        &oracle["attention_keys"],
    );
    compare(
        "qsa_values",
        state.values.as_ref().unwrap(),
        &oracle["attention_values"],
    );
    compare(
        "qsa_raw_index_keys",
        state.raw_index_keys.as_ref().unwrap(),
        &oracle["attention_raw_index_keys"],
    );
    compare(
        "qsa_block_keys",
        state.block_keys.as_ref().unwrap(),
        &oracle["attention_block_keys"],
    );
    let base = state.clone();
    let next = oracle["attention_input"]
        .slice(&[0, 0, 0][..], &[1, 1, cfg.hidden_size][..])
        .unwrap();
    let output = attention.forward_on(&next, &mut state, ().into()).unwrap();
    output.eval().unwrap();
    state.trim_on(base.offset, ().into()).unwrap();
    assert!(state.block_keys.is_none());
    let replay = attention.forward_on(&next, &mut state, ().into()).unwrap();
    assert_eq!(
        output
            .astype(Dtype::Float32)
            .unwrap()
            .to_vec::<f32>()
            .unwrap(),
        replay
            .astype(Dtype::Float32)
            .unwrap()
            .to_vec::<f32>()
            .unwrap()
    );
    assert_eq!(state.offset, base.offset + 1);
    let index = cfg.ple_index(0).unwrap();
    let width = cfg.ple_embed_dim as usize / index.heads();
    let rows = PleRows::new(
        Arc::clone(&files),
        "language_model.model.layers.1.ple.ple_embedding.ngram_embedding",
        index,
        width,
        4 * 1024 * 1024,
        8 * 1024 * 1024,
    )
    .unwrap();
    let ple = PleLayer::from_weights(&weights, "model.layers.1.ple", &cfg, rows).unwrap();
    let mut state = ple.new_state().unwrap();
    let tokens: Vec<u32> = oracle["ple_tokens"]
        .to_vec::<i32>()
        .unwrap()
        .into_iter()
        .map(|n| n as u32)
        .collect();
    let result = ple
        .forward_on(&oracle["hyper"], &tokens, &mut state, ().into())
        .unwrap();
    compare("ple_output", &result, &oracle["ple_output"]);
    compare(
        "ple_convolution",
        state.convolution.as_ref().unwrap(),
        &oracle["ple_conv_state"],
    );
    assert_eq!(state.history.position, tokens.len() as u64);
}
