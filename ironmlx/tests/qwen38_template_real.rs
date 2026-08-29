//! Metadata-only acceptance for the exact mlx-community Qwen3.8 template.
//! The test skips when the checkpoint metadata is not present locally.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use ironmlx::core::{
    preflight_model_metadata, ChatTemplate, NativeOutputDialect, TokenizerConfig, ToolDialect,
};

fn add_snapshot_dirs(found: &mut BTreeSet<PathBuf>, snapshots: &Path) {
    let Some(entries) = std::fs::read_dir(snapshots).ok() else {
        return;
    };
    for path in entries.flatten().map(|entry| entry.path()) {
        if path.join("chat_template.jinja").is_file() && path.join("config.json").is_file() {
            found.insert(path);
        }
    }
}

fn snapshot_dirs() -> Vec<PathBuf> {
    let mut found = BTreeSet::new();
    for env_name in ["QWEN38_MODEL", "QWEN38_4BIT_MODEL", "QWEN38_8BIT_MODEL"] {
        if let Some(path) = std::env::var_os(env_name).map(PathBuf::from) {
            found.insert(path);
        }
    }
    if let Some(home) = dirs::home_dir() {
        for bits in [4, 8] {
            let repo = format!("mlx-community--Qwen3.8-27B-{bits}bit");
            add_snapshot_dirs(
                &mut found,
                &home.join(format!(".ironmlx/models/huggingface/{repo}/snapshots")),
            );
            add_snapshot_dirs(
                &mut found,
                &home.join(format!(".ironmlx/models/models--{repo}/snapshots")),
            );
        }
    }
    found.into_iter().collect()
}

#[test]
fn exact_qwen38_templates_support_native_contract_and_reasoning_controls() {
    let model_dirs = snapshot_dirs();
    if model_dirs.is_empty() {
        eprintln!("Qwen3.8 metadata absent — skipping");
        return;
    }
    for model_dir in model_dirs {
        assert_qwen38_template(&model_dir);
    }
}

fn assert_qwen38_template(model_dir: &Path) {
    let preflight = preflight_model_metadata(model_dir).expect("Qwen3.8 metadata preflight");
    assert_eq!(preflight.model_type, "qwen3_5");
    assert_eq!(preflight.artifact_role, "base");
    let quantization = preflight.quantization.expect("Qwen3.8 quantization");
    assert_eq!(quantization.mode, "affine");
    assert!(
        matches!(quantization.bits, 4 | 8),
        "unexpected Qwen3.8 quantization in {}",
        model_dir.display()
    );
    assert_eq!(quantization.group_size, 64);

    let config = TokenizerConfig::from_model_dir(&model_dir).expect("tokenizer config");
    let source = config.chat_template.expect("standalone chat template");
    assert_eq!(
        NativeOutputDialect::detect("qwen3_5", &source),
        Some(NativeOutputDialect::Qwen38)
    );
    assert_eq!(
        ToolDialect::detect("qwen3_5", &source),
        Some(ToolDialect::Qwen35)
    );

    let template = ChatTemplate::new(&source).expect("compile Qwen3.8 template");
    let messages = serde_json::json!([
        {"role": "user", "content": "first"},
        {
            "role": "assistant",
            "content": "answer",
            "reasoning_content": "retained reasoning"
        },
        {"role": "user", "content": "continue"}
    ]);

    let default = template
        .render_serializable(messages.as_array().unwrap(), true, None)
        .expect("render default xhigh");
    assert!(default.contains("Reasoning effort is set to xhigh."));
    assert!(default.contains("retained reasoning"));
    assert!(default.ends_with("<|im_start|>assistant\n<think>\n"));

    for effort in ["low", "xhigh"] {
        let rendered = template
            .render_serializable(
                messages.as_array().unwrap(),
                true,
                Some(&serde_json::json!({"reasoning_effort": effort})),
            )
            .unwrap_or_else(|error| panic!("render {effort} effort: {error:#}"));
        assert!(
            rendered.contains(&format!("Reasoning effort is set to {effort}.")),
            "rendered prompt did not select {effort}"
        );
    }
    let medium = template
        .render_serializable(
            messages.as_array().unwrap(),
            true,
            Some(&serde_json::json!({"reasoning_effort": "medium"})),
        )
        .expect("render medium effort");
    assert!(!medium.contains("Reasoning effort is set to low."));
    assert!(!medium.contains("Reasoning effort is set to xhigh."));
    assert!(medium.ends_with("<|im_start|>assistant\n<think>\n"));

    let no_history = template
        .render_serializable(
            messages.as_array().unwrap(),
            true,
            Some(&serde_json::json!({"preserve_thinking": false})),
        )
        .expect("render without preserved reasoning");
    assert!(!no_history.contains("retained reasoning"));

    let disabled = template
        .render_serializable(
            messages.as_array().unwrap(),
            true,
            Some(&serde_json::json!({"enable_thinking": false})),
        )
        .expect("render thinking disabled");
    assert!(disabled.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
}
