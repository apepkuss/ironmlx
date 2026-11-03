//! Opt-in real-model gate for two model-owned programs with the same cache key.
use ironmlx_core::{
    ane,
    m5_profile::{self, AneMode, M5ProfileMode, M5ProfileTarget},
};
use ironmlx_lm::{
    core::{model_input::build_position_ids_batched, Loader},
    models::Qwen35Model,
};
use mlx::{Array, Dtype};

#[test]
#[ignore = "requires ANE_REAL_MODEL, ANE_REAL_OUTPUT and qualified ANE hardware"]
fn unloading_one_model_does_not_invalidate_another_ane_context() {
    mlx::metal::set_metallib_path(&format!(
        "{}/lib/mlx.metallib",
        std::env::var("MLX_DIR").unwrap()
    ))
    .unwrap();
    m5_profile::install_for_target_with_ane(
        M5ProfileMode::Off,
        AneMode::Auto,
        false,
        M5ProfileTarget::Unqualified,
    );
    assert!(
        ane::eligible(ane::Model::Dense27B),
        "qualified real ANE required"
    );
    let path = std::env::var("ANE_REAL_MODEL").unwrap();
    let tokenizer =
        tokenizers::Tokenizer::from_file(std::path::Path::new(&path).join("tokenizer.json"))
            .unwrap();
    let tokens = tokenizer
        .encode(include_str!("../../docs/dflash2-server-api.md"), false)
        .unwrap()
        .get_ids()
        .iter()
        .map(|&id| id as i32)
        .take(2048)
        .collect::<Vec<_>>();
    assert_eq!(tokens.len(), 2048);
    let loader = Loader::open(std::path::Path::new(&path)).unwrap();
    let first = Qwen35Model::from_loader(&loader).unwrap();
    let second = Qwen35Model::from_loader(&loader).unwrap();
    drop(loader);
    let run = |model: &Qwen35Model| {
        let mut cache = model.make_cache(1, 4096, Dtype::Bfloat16).unwrap();
        let ids = Array::try_from((&tokens[..], [1, 2048])).unwrap();
        let positions = build_position_ids_batched(&[2048], 2048).unwrap();
        let before = ane::completed(ane::Model::Dense27B);
        let logits = model
            .forward_on(
                &ids,
                &positions,
                None,
                None,
                Some(&mut cache),
                mlx::StreamOrDevice::default(),
            )
            .unwrap()
            .astype(Dtype::Float32)
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        assert_eq!(
            ane::completed(ane::Model::Dense27B) - before,
            64,
            "GPU fallback must not pass this gate"
        );
        assert!(logits.iter().all(|v| v.is_finite()));
        logits
    };
    let _ = run(&first);
    let expected = run(&second);
    let both_bytes = ane::surface_bytes(ane::Model::Dense27B);
    assert!(both_bytes > 0);
    drop(first);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while ane::surface_bytes(ane::Model::Dense27B) != both_bytes / 2
        && std::time::Instant::now() < deadline
    {
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    let second_bytes = ane::surface_bytes(ane::Model::Dense27B);
    assert_eq!(
        second_bytes * 2,
        both_bytes,
        "only the unloaded model's surfaces must be released"
    );
    let actual = run(&second);
    assert!(
        actual
            .iter()
            .zip(&expected)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "unloading a peer program changed the survivor's output bits"
    );
    drop(second);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while ane::surface_bytes(ane::Model::Dense27B) > 0 && std::time::Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert_eq!(ane::surface_bytes(ane::Model::Dense27B), 0);
    std::fs::write(std::env::var("ANE_REAL_OUTPUT").unwrap(),serde_json::to_vec_pretty(&serde_json::json!({"result":"pass","both_surface_bytes":both_bytes,"survivor_surface_bytes":second_bytes,"surface_bytes_after_unload":0,"completed_ane_jobs":ane::completed(ane::Model::Dense27B),"survivor_output_bitwise_equal":true})).unwrap()).unwrap();
}
