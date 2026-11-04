//! Opt-in QSA arithmetic and branch-lifetime gate with actual checkpoint weights.
use ironmlx_lm::{
    core::{
        active_pages::{ActivePagesConfig, ActivePagesPool},
        tensor_file::ModelTensorFiles,
        weights::{QuantMeta, QuantMode, WeightMap},
    },
    models::qwen4_exp::{
        attention::{QsaAttention, QsaState},
        config::FlashTextConfig,
    },
};
use mlx::{Array, Dtype};
use std::{collections::HashMap, path::Path};

fn check(a: &Array, b: &Array, exact: bool) {
    assert_eq!(a.shape(), b.shape());
    let a = a.astype(Dtype::Float32).unwrap().to_vec::<f32>().unwrap();
    let b = b.astype(Dtype::Float32).unwrap().to_vec::<f32>().unwrap();
    if exact {
        assert_eq!(a, b, "prefill arithmetic changed");
        return;
    }
    let mut max = 0f32;
    let mut peak = 0f32;
    let mut err = 0f64;
    let mut norm = 0f64;
    for (&a, &b) in a.iter().zip(&b) {
        assert!(a.is_finite() && b.is_finite());
        max = max.max((a - b).abs());
        peak = peak.max(a.abs());
        err += ((a - b) as f64).powi(2);
        norm += (a as f64).powi(2);
    }
    let rms = (err / norm.max(1e-20)).sqrt();
    eprintln!("QSA indexed decode max_abs={max} peak={peak} relative_rms={rms}");
    assert_eq!(a, b, "indexed decode arithmetic differs from native SDPA");
}

#[test]
#[ignore = "requires IRONMLX_FLASH_SNAPSHOT and an Apple GPU"]
fn real_qsa_ssd_prefill_decode_rollback_and_forks() {
    let snapshot = std::env::var("IRONMLX_FLASH_SNAPSHOT").unwrap();
    let path = Path::new(&snapshot);
    let root: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(path.join("config.json")).unwrap()).unwrap();
    let cfg = FlashTextConfig::from_root(&root).unwrap();
    let files = ModelTensorFiles::open(path).unwrap();
    let mut tensors = HashMap::new();
    for (name, info) in files
        .tensors()
        .filter(|(name, _)| name.starts_with("language_model.model.layers.3.self_attn."))
    {
        tensors.insert(
            name.strip_prefix("language_model.").unwrap().to_owned(),
            files
                .read_rows_array(name, 0, info.shape[0], 64 << 20)
                .unwrap(),
        );
    }
    let weights = WeightMap::new(
        tensors,
        Some(QuantMeta {
            bits: 4,
            group_size: 32,
            mode: QuantMode::Affine,
        }),
        HashMap::new(),
    );
    let attention =
        QsaAttention::from_weights(&weights, "model.layers.3.self_attn", &cfg, 128 << 20).unwrap();
    let dir = std::env::temp_dir().join(format!("flash-qsa-state-{}", uuid::Uuid::new_v4()));
    let wide = std::env::var("IRONMLX_FLASH_ACTIVE_WIDE").as_deref() == Ok("1");
    let read_cache = if wide { 16 << 20 } else { 64 << 10 };
    let config = ActivePagesConfig {
        root: dir.clone(),
        cache_bytes: read_cache,
        pack_bytes: if wide { 128 << 20 } else { 8 << 20 },
        disk_bytes: 256 << 20,
        min_free_bytes: 0,
    };
    let pool = ActivePagesPool::new(config).unwrap();
    {
        let mut ram = QsaState::new(16384).unwrap();
        let mut ssd = QsaState::new(16384)
            .unwrap()
            .with_active_pages(pool.clone());
        let mut position = 0;
        let chunks = if wide {
            vec![2048, 2048, 2048, 2048, 3]
        } else {
            vec![2048, 2048, 2048, 3]
        };
        for seq in chunks {
            let data: Vec<f32> = (0..seq * cfg.hidden_size)
                .map(|i| {
                    (((i as i64 + position as i64 * cfg.hidden_size as i64) * 17 % 1009) as f32
                        - 504.)
                        / 2000.
                })
                .collect();
            let input = Array::try_from((data.as_slice(), [1, seq, cfg.hidden_size]))
                .unwrap()
                .astype(Dtype::Bfloat16)
                .unwrap();
            let a = attention.forward_on(&input, &mut ram, ().into()).unwrap();
            let b = attention.forward_on(&input, &mut ssd, ().into()).unwrap();
            check(&a, &b, true);
            position += seq;
        }
        assert!(ssd.keys.is_none() && ssd.values.is_none());
        let base_ram = ram.clone();
        let base_ssd = ssd.clone();
        let input = Array::try_from((
            vec![0.15f32; cfg.hidden_size as usize].as_slice(),
            [1, 1, cfg.hidden_size],
        ))
        .unwrap()
        .astype(Dtype::Bfloat16)
        .unwrap();
        for _ in 0..8 {
            check(
                &attention.forward_on(&input, &mut ram, ().into()).unwrap(),
                &attention.forward_on(&input, &mut ssd, ().into()).unwrap(),
                false,
            );
        }
        ram.trim_on(position + 3, ().into()).unwrap();
        ssd.trim_on(position + 3, ().into()).unwrap();
        check(
            &attention.forward_on(&input, &mut ram, ().into()).unwrap(),
            &attention.forward_on(&input, &mut ssd, ().into()).unwrap(),
            false,
        );
        for branch in 0..4 {
            let mut a = base_ram.clone();
            let mut b = base_ssd.clone();
            let input = mlx::ops::binary::multiply(
                &input,
                &Array::try_from((&[branch as f32 + 1.][..], ()))
                    .unwrap()
                    .astype(Dtype::Bfloat16)
                    .unwrap(),
            )
            .unwrap();
            for _ in 0..5 {
                check(
                    &attention.forward_on(&input, &mut a, ().into()).unwrap(),
                    &attention.forward_on(&input, &mut b, ().into()).unwrap(),
                    false,
                );
            }
            assert_eq!(base_ssd.offset, position);
        }
        let mut crossing_ram = base_ram.clone();
        let mut crossing_ssd = base_ssd.clone();
        crossing_ram.trim_on(4093, ().into()).unwrap();
        crossing_ssd.trim_on(4093, ().into()).unwrap();
        for _ in 0..8 {
            check(
                &attention
                    .forward_on(&input, &mut crossing_ram, ().into())
                    .unwrap(),
                &attention
                    .forward_on(&input, &mut crossing_ssd, ().into())
                    .unwrap(),
                false,
            );
        }
        let stats = pool.stats.snapshot();
        assert!(stats.written_bytes > 0 && stats.read_bytes > 0);
        assert!(stats.cache_bytes <= read_cache as u64);
        assert_eq!(stats.failures, 0);
        eprintln!("active QSA stats: {stats:?}");
    }
    let stats = pool.stats.snapshot();
    assert_eq!(stats.live_files, 0);
    assert_eq!(stats.disk_bytes, 0);
    assert_eq!(stats.cache_bytes, 0);
    drop(pool);
    assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 0);
    std::fs::remove_dir(dir).unwrap();
}
