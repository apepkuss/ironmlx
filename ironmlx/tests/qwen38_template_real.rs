//! Metadata-only acceptance for the exact mlx-community Qwen3.8 template.
//! The test skips when the checkpoint metadata is not present locally.

use std::path::PathBuf;

use ironmlx::core::{ChatTemplate, NativeOutputDialect, TokenizerConfig, ToolDialect};

fn snapshot_dir() -> Option<PathBuf> {
    let home = dirs::home_dir()?;
    let snapshots = home.join(".ironmlx/models/models--mlx-community--Qwen3.8-27B-4bit/snapshots");
    std::fs::read_dir(snapshots)
        .ok()?
        .flatten()
        .map(|entry| entry.path())
        .find(|path| path.join("chat_template.jinja").is_file())
}

#[test]
fn exact_qwen38_template_supports_native_contract_and_reasoning_controls() {
    let Some(model_dir) = snapshot_dir() else {
        eprintln!("Qwen3.8 metadata absent — skipping");
        return;
    };
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
