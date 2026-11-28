//! Deterministic tiny looped model against mlx-lm 0c4c99192230f1943dd468d297f0b8a64ae7b7c2.
use anyhow::Result;
use ironmlx_lm::{
    core::{chat_template::ChatTemplate, vision::DenseVlMethods, Loader, Model},
    models::Inkling2Model,
};
use mlx::{Array, Dtype};
use std::path::PathBuf;
fn init_mlx() -> Result<()> {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| std::env::set_var("MLX_ENABLE_TF32", "0"));
    if let Ok(root) = std::env::var("MLX_DIR") {
        mlx::metal::set_metallib_path(&format!("{root}/lib/mlx.metallib"))?;
    }
    Ok(())
}

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/inkling2")
}
fn close(a: &Array, b: &Array, tolerance: f32) -> Result<()> {
    assert_eq!(a.shape(), b.shape());
    let diff = a
        .astype(Dtype::Float32)?
        .try_sub(&b.astype(Dtype::Float32)?)?
        .abs()?
        .max(mlx::ops::All, false)?
        .item::<f32>()?;
    eprintln!("parity max_abs={diff}");
    assert!(diff < tolerance, "max absolute error {diff} >= {tolerance}");
    Ok(())
}
#[test]
#[serial_test::serial]
fn looped_text_chunked_cache_batch_and_vision_match_reference() -> Result<()> {
    init_mlx()?;
    for root in [fixture(), fixture().join("mixed")] {
        check_model_parity(&root)?;
    }
    Ok(())
}

fn check_model_parity(root: &std::path::Path) -> Result<()> {
    let loader = Loader::open_multimodal(root)?;
    let model = Inkling2Model::from_loader(&loader)?;
    let (reference, _) =
        mlx::io::load_safetensors(root.join("reference.safetensors").to_str().unwrap())?;
    let tokens: Vec<i32> = (1..=12).collect();
    let ids: Array = (tokens.as_slice(), &[1, 12][..]).try_into()?;
    let positions = Array::zeros((1, 12), Dtype::Int32)?;
    let mut caches = model.make_cache(1, 32, model.cache_dtype())?;
    let hidden =
        model.forward_text_hidden(&ids, &positions, None, None, Some(&mut caches), ().into())?;
    let full = model.project_hidden_on(&hidden, ().into())?;
    close(&full, &reference["logits"], 2e-4)?;
    let mut caches = model.make_cache(1, 32, model.cache_dtype())?;
    let mut parts = Vec::new();
    // The third chunk starts past the fixture's eight-token local window.
    for (start, end) in [(0, 5), (5, 9), (9, 12)] {
        let chunk = mlx::ops::indexing::slice(&ids, [0, start], [1, end])?;
        let hidden = model.forward_text_hidden(
            &chunk,
            &positions,
            None,
            None,
            Some(&mut caches),
            ().into(),
        )?;
        parts.push(model.project_hidden_on(&hidden, ().into())?);
    }
    close(
        &mlx::ops::shape::concatenate(&parts.iter().collect::<Vec<_>>(), 1)?,
        &reference["chunked"],
        2e-4,
    )?;
    // Per-visit convolution state survives a snapshot/restore, including loop passes.
    let snapshots: Vec<_> = caches.iter().map(|c| c.snapshot()).collect();
    let next: Array = (&[13_i32][..], &[1, 1][..]).try_into()?;
    let first = model.forward_on(&next, &positions, None, None, Some(&mut caches), ().into())?;
    first.eval()?;
    for (cache, snapshot) in caches.iter_mut().zip(&snapshots) {
        cache.restore(snapshot)?;
    }
    let replay = model.forward_on(&next, &positions, None, None, Some(&mut caches), ().into())?;
    close(&first, &replay, 1e-6)?;
    // Ragged right-padding must not advance a shorter row's convolution window.
    let batch: Array = (&[1_i32, 2, 3, 4, 5, 1, 2, 3, 0, 0][..], &[2, 5][..]).try_into()?;
    let mut batched = model.make_cache(2, 32, model.cache_dtype())?;
    let got = model.batched_prefill(
        &batch,
        &positions,
        &positions,
        &positions,
        &[5, 3],
        Some(&mut batched),
        ().into(),
    )?;
    for (row, length) in [(0, 5), (1, 3)] {
        let mut single = model.make_cache(1, 32, model.cache_dtype())?;
        let row_ids = mlx::ops::indexing::slice(&batch, [row, 0], [row + 1, length])?;
        let expected = model.forward_on(
            &row_ids,
            &positions,
            None,
            None,
            Some(&mut single),
            ().into(),
        )?;
        close(
            &mlx::ops::indexing::slice(&got, [row, 0, 0], [row + 1, 1, 120])?,
            &expected,
            2e-4,
        )?;
    }
    let vision =
        model.compute_vision_embeds(&[reference["pixels"].clone()], &[(1, 4, 6)], ().into())?;
    close(&vision, &reference["vision"], 2e-4)?;
    Ok(())
}
#[test]
fn checkpoint_template_preserves_tools_thinking_and_separate_image_channels() -> Result<()> {
    let template = ChatTemplate::new(include_str!("fixtures/inkling2/chat_template.jinja"))?;
    let rendered=template.render_serializable(&[serde_json::json!({"role":"user","content":"What is this?<|content_image|><|unused_200054|><|unused_200054|><|end_message|>"})],true,Some(&serde_json::json!({"enable_thinking":false})))?;
    assert!(rendered.contains("Thinking effort level: 0<|end_message|>"));
    assert!(rendered.ends_with("<|message_user|><|content_text|>What is this?<|end_message|><|message_user|><|content_image|><|unused_200054|><|unused_200054|><|end_message|>"));
    let history=template.render_serializable(&[
        serde_json::json!({"role":"assistant","content":"","reasoning_content":"Need weather.","tool_calls":[{"id":"call1","function":{"name":"weather","arguments":"{\"city\":\"Tokyo\"}"}}]}),
        serde_json::json!({"role":"tool","tool_call_id":"call1","content":"sunny"}),
    ],true,None)?;
    assert!(history.contains("<|message_model|><|content_thinking|>Need weather.<|end_message|>"));
    assert!(history.contains("<|message_model|>weather<|content_invoke_tool_json|>{\"name\":\"weather\",\"args\":{\"city\":\"Tokyo\"}}<|end_message|>"));
    assert!(history.contains("<|message_tool|>weather<|content_text|>sunny<|end_message|>"));
    Ok(())
}

#[test]
#[serial_test::serial]
#[ignore = "requires INKLING_MODEL and INKLING_REFERENCE real checkpoint/reference paths"]
fn real_checkpoint_logits_match_reference() -> Result<()> {
    init_mlx()?;
    let root = PathBuf::from(std::env::var("INKLING_REFERENCE")?);
    let data: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("real-reference.json"))?)?;
    let tokens: Vec<i32> = serde_json::from_value(data["prompt_ids"].clone())?;
    let (reference, _) =
        mlx::io::load_safetensors(root.join("real-reference.safetensors").to_str().unwrap())?;
    let loader = Loader::open_multimodal(&PathBuf::from(std::env::var("INKLING_MODEL")?))?;
    let model = Inkling2Model::from_loader(&loader)?;
    let ids: Array = (tokens.as_slice(), &[1, tokens.len() as i32][..]).try_into()?;
    let positions = Array::zeros((1, 1), Dtype::Int32)?;
    let mut caches = model.make_cache(1, 256, model.cache_dtype())?;
    let mut logits =
        model.forward_on(&ids, &positions, None, None, Some(&mut caches), ().into())?;
    let mut generated = Vec::new();
    for step in 0..32 {
        logits.eval()?;
        if step < 2 {
            let expected = &reference[if step == 0 {
                "prefill_logits"
            } else {
                "decode_logits"
            }];
            let av = logits
                .astype(Dtype::Float32)?
                .reshape((-1,))?
                .to_vec::<f32>()?;
            let bv = expected
                .astype(Dtype::Float32)?
                .reshape((-1,))?
                .to_vec::<f32>()?;
            let max = av
                .iter()
                .zip(&bv)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            let dot: f64 = av.iter().zip(&bv).map(|(a, b)| *a as f64 * *b as f64).sum();
            let norm = |x: &Vec<f32>| x.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
            let cosine = dot / (norm(&av) * norm(&bv));
            eprintln!("real step={step} max_abs={max} cosine={cosine}");
            assert!(cosine > 0.999);
        }
        let token = logits.argmax(-1, false)?.item::<u32>()? as i32;
        generated.push(token);
        if token == 200006 {
            break;
        }
        let next: Array = (&[token][..], &[1, 1][..]).try_into()?;
        logits = model.forward_on(&next, &positions, None, None, Some(&mut caches), ().into())?;
    }
    eprintln!("generated={generated:?}");
    let expected: Vec<i32> = serde_json::from_value(data["generated_ids"].clone())?;
    assert_eq!(generated, expected);
    Ok(())
}
