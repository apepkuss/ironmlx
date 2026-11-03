//! Explicit opt-in ANE execution/quality gate; no passing skip on unavailable ANE.
use ironmlx_core::{
    ane,
    m5_profile::{self, AneMode, M5ProfileMode, M5ProfileTarget},
};
use ironmlx_lm::{
    core::{
        model_input::{build_decode_position_ids, build_position_ids_batched},
        Loader,
    },
    models::{
        dflash2::{DFlash2Target, DFlash2TargetForwardMode},
        Qwen35Model,
    },
};
use mlx::{Array, Dtype};
#[test]
#[ignore = "requires ANE_REAL_MODEL, ANE_REAL_OUTPUT and qualified ANE hardware"]
fn dense_ordinary_and_speculative_full_chunk() {
    mlx::metal::set_metallib_path(&format!(
        "{}/lib/mlx.metallib",
        std::env::var("MLX_DIR").unwrap()
    ))
    .unwrap();
    let mode: AneMode = std::env::var("ANE_REAL_MODE")
        .unwrap_or("auto".into())
        .parse()
        .unwrap();
    let route = std::env::var("ANE_REAL_ROUTE").unwrap_or("ordinary".into());
    let gpu_auto = std::env::var("ANE_REAL_GPU_PROFILE").as_deref() == Ok("auto");
    m5_profile::install_for_target_with_ane(
        if gpu_auto {
            M5ProfileMode::Auto
        } else {
            M5ProfileMode::Off
        },
        mode,
        gpu_auto,
        if gpu_auto {
            M5ProfileTarget::Qwen35Dense
        } else {
            M5ProfileTarget::Unqualified
        },
    );
    if mode == AneMode::Auto {
        assert!(
            ane::eligible(ane::Model::Dense27B),
            "actual qualified ANE required"
        );
    }
    let injected = std::env::var("ANE_REAL_INJECT_FAILURE").as_deref() == Ok("1");
    if injected {
        unsafe {
            std::env::set_var("IRONMLX_DIAGNOSTIC_ANE_FFN", "fail");
        }
    }
    let path = std::env::var("ANE_REAL_MODEL").unwrap();
    let mut loader = Loader::open(std::path::Path::new(&path)).unwrap();
    let tokenizer =
        tokenizers::Tokenizer::from_file(std::path::Path::new(&path).join("tokenizer.json"))
            .unwrap();
    let model = if route == "dflash" {
        Qwen35Model::from_loader_dflash2(&mut loader).unwrap()
    } else {
        Qwen35Model::from_loader(&loader).unwrap()
    };
    drop(loader);
    let execution_contract = ironmlx_lm::core::Model::execution_cache_fingerprint(&model);
    let source = std::env::var("ANE_REAL_TEXT_FILE")
        .map(|p| std::fs::read_to_string(p).unwrap())
        .unwrap_or_else(|_| include_str!("../../docs/dflash2-server-api.md").into());
    let tokens = tokenizer
        .encode(source, false)
        .unwrap()
        .get_ids()
        .iter()
        .map(|&v| v as i32)
        .collect::<Vec<_>>();
    let steps = std::env::var("ANE_REAL_STEPS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(4);
    assert!(tokens.len() > 2048 + steps);
    let summary = |values: &[f32], target: i32| {
        let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let lse = max
            + values
                .iter()
                .map(|v| (*v as f64 - max).exp())
                .sum::<f64>()
                .ln();
        let top1 = values
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
        serde_json::json!({"target":target,"top1":top1,"nll":lse-values[target as usize] as f64})
    };
    let mut teacher = Vec::new();
    let mut paired_stats = Vec::new();
    let mut rows = Vec::new();
    let mut times = Vec::new();
    let paired = std::env::var("ANE_REAL_PAIRED").as_deref() == Ok("1");
    let settings = if paired {
        vec![false, true, true, false, false, true, true, false].repeat(
            std::env::var("ANE_REAL_PAIR_BLOCKS")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(1),
        )
    } else {
        vec![
            mode == AneMode::Auto;
            std::env::var("ANE_REAL_REPEATS")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(2)
        ]
    };
    for enabled in settings {
        if paired {
            unsafe {
                std::env::set_var(
                    m5_profile::settings::DENSE_PREFILL_ANE_FFN_UNITS,
                    if enabled {
                        if gpu_auto {
                            "12"
                        } else {
                            "9"
                        }
                    } else {
                        "0"
                    },
                );
            }
        }
        let jobs_before = ane::completed(ane::Model::Dense27B);
        let mut cache = model.make_cache(1, 4096, Dtype::Bfloat16).unwrap();
        let ids = Array::try_from((&tokens[..2048], [1, 2048])).unwrap();
        let positions = build_position_ids_batched(&[2048], 2048).unwrap();
        let start = std::time::Instant::now();
        let out = if route == "dflash" {
            {
                let out = model
                    .dflash2_forward_target_on(
                        &ids,
                        &positions,
                        Some(&mut cache),
                        &[5],
                        DFlash2TargetForwardMode::Prefill,
                        mlx::StreamOrDevice::default(),
                    )
                    .unwrap();
                model
                    .dflash2_project_hidden_on(
                        &out.hidden
                            .slice(&[0, 2047, 0][..], &[1, 2048, 5120][..])
                            .unwrap(),
                        mlx::StreamOrDevice::default(),
                    )
                    .unwrap()
            }
        } else {
            model
                .forward_on(
                    &ids,
                    &positions,
                    None,
                    None,
                    Some(&mut cache),
                    mlx::StreamOrDevice::default(),
                )
                .unwrap()
        };
        let mut logits = vec![out.astype(Dtype::Float32).unwrap().to_vec::<f32>().unwrap()];
        let seconds = start.elapsed().as_secs_f64();
        times.push(seconds);
        paired_stats.push(serde_json::json!({"enabled":enabled,"seconds":seconds,"jobs":ane::completed(ane::Model::Dense27B)-jobs_before}));
        let jobs_this_run = ane::completed(ane::Model::Dense27B) - jobs_before;
        eprintln!("ANE_PAIR enabled={enabled} seconds={seconds:.6} jobs={jobs_this_run}");
        if !injected {
            if enabled {
                assert!(jobs_this_run > 0, "enabled run must execute ANE");
            } else {
                assert_eq!(jobs_this_run, 0);
            }
        }
        ironmlx_lm::nn::dense_ane_ffn_check().unwrap();
        teacher = vec![summary(&logits[0], tokens[2048])];
        for i in 0..steps {
            let ids = Array::try_from((&tokens[2048 + i..2049 + i], [1, 1])).unwrap();
            let positions = build_decode_position_ids(&[2048 + i as i32]).unwrap();
            let values = model
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
            teacher.push(summary(&values, tokens[2049 + i]));
            if i < 4 {
                logits.push(values);
            }
        }
        assert!(logits.iter().flatten().all(|v| v.is_finite()));
        rows = logits;
    }
    let surface_bytes = ane::surface_bytes(ane::Model::Dense27B);
    if mode == AneMode::Auto {
        assert!(surface_bytes > 0, "actual ANE surfaces required");
    } else {
        assert_eq!(surface_bytes, 0);
    }
    if injected {
        assert_eq!(
            ironmlx_lm::core::Model::execution_cache_fingerprint(&model),
            execution_contract,
            "failure must preserve the approximate prefix execution contract"
        );
        assert_eq!(ironmlx_lm::core::Model::auxiliary_memory_bytes(&model),0,
            "retired execution must keep its cache identity without forecasting new ANE allocations");
    }
    drop(model);
    let release_deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while ane::surface_bytes(ane::Model::Dense27B) > 0
        && std::time::Instant::now() < release_deadline
    {
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert_eq!(
        ane::surface_bytes(ane::Model::Dense27B),
        0,
        "model unload must release ANE surfaces"
    );
    let jobs = ane::completed(ane::Model::Dense27B);
    if injected {
        assert!(ane::retired(ane::Model::Dense27B));
        assert!(
            !ane::retired(ane::Model::Other),
            "ANE failures must not mark unadapted models as retired"
        );
        assert_eq!(jobs, 0);
    } else if mode == AneMode::Auto {
        assert!(jobs >= 64, "must execute actual ANE programs");
    } else {
        assert_eq!(jobs, 0);
    }
    std::fs::write(std::env::var("ANE_REAL_OUTPUT").unwrap(),serde_json::to_vec(&serde_json::json!({"mode":mode.as_str(),"route":route,"prefill_seconds":&times,"paired_stats":paired_stats,"completed_ane_jobs":jobs,"injected_failure":injected,"ane_surface_bytes":surface_bytes,"surface_bytes_after_unload":0,"prompt_tokens":&tokens[..2048],"teacher_forced":teacher,"logits":rows,"next_tokens":&tokens[2048..2053]})).unwrap()).unwrap();
    eprintln!(
        "ANE_REAL route={route} mode={} seconds={times:?} jobs={jobs}",
        mode.as_str()
    );
}
