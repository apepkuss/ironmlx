//! Real Flash target + author DFlash2 draft: exact ordinary/verify outputs,
//! zero/partial/full acceptance, PLE EOS history and continuation after commit.
use ironmlx_lm::{
    core::tensor_file::ModelTensorFiles,
    core::{
        cache::layer::{LayerCache, LayerCacheSnapshot},
        Loader, Model,
    },
    models::{
        dflash2::{
            DFlash2DraftModel, DFlash2Target, DFlash2TargetForwardMode as Mode, DFlash2TargetOutput,
        },
        qwen4_exp::text_model::{ExpertPagingOptions, FlashExecutionOptions, FlashTextModel},
    },
};
use mlx::{Array, Dtype};
use std::{path::Path, sync::Arc};

fn ids(tokens: &[u32]) -> Array {
    Array::try_from((tokens, [1, tokens.len() as i32])).unwrap()
}
fn exact(label: &str, a: &Array, b: &Array) {
    assert_eq!(a.shape(), b.shape(), "{label} shape");
    assert_eq!(a.dtype(), b.dtype(), "{label} dtype");
    let equal = a
        .equal(b)
        .unwrap()
        .reshape([-1])
        .unwrap()
        .all(-1, false)
        .unwrap()
        .to_vec::<bool>()
        .unwrap();
    assert_eq!(equal, [true], "{label} values differ");
}
fn snapshots(cache: &[LayerCache]) -> Vec<LayerCacheSnapshot> {
    cache.iter().map(LayerCache::snapshot).collect()
}
fn restore(cache: &mut [LayerCache], base: &[LayerCacheSnapshot]) {
    for (c, b) in cache.iter_mut().zip(base) {
        c.restore(b).unwrap();
    }
}
fn states(a: &[LayerCache], b: &[LayerCache]) {
    for (layer, (a, b)) in a.iter().zip(b).enumerate() {
        let (LayerCache::Flash(a), LayerCache::Flash(b)) = (a, b) else {
            panic!("Flash state")
        };
        assert_eq!(a.offsets(), b.offsets());
        for row in 0..a.offsets().len() {
            // Compare canonical persistent tensors, independent of RAM/SSD
            // representation. Rollback intentionally invalidates pooled index
            // keys, a derived cache rebuilt from raw index history. Continuation
            // logits/captures must still match exactly after that rebuild.
            let (a, len_a) = a.durable_prefix_for_row(row).unwrap();
            let (b, len_b) = b.durable_prefix_for_row(row).unwrap();
            assert_eq!(len_a, len_b);
            assert_eq!(a.len(), b.len());
            let mut meta_a = a[0].to_vec::<i32>().unwrap();
            let mut meta_b = b[0].to_vec::<i32>().unwrap();
            let sparse = meta_a[2] == 1;
            if sparse {
                meta_a[5] = 0;
                meta_b[5] = 0;
            }
            assert_eq!(meta_a, meta_b, "state metadata layer={layer} row={row}");
            for index in 1..a.len() {
                if sparse && index == 5 {
                    continue;
                }
                exact(
                    &format!("layer {layer} row {row} state {index}"),
                    &a[index],
                    &b[index],
                );
            }
        }
    }
}

fn positions(cache: &[LayerCache], seq: i32) -> Array {
    let LayerCache::Flash(first) = &cache[0] else {
        panic!("Flash cache")
    };
    let one: Vec<i32> = first
        .offsets()
        .iter()
        .flat_map(|start| (0..seq).map(move |i| start + i))
        .collect();
    Array::try_from((
        one.repeat(3).as_slice(),
        [3, first.offsets().len() as i32, seq],
    ))
    .unwrap()
}
fn run(
    model: &FlashTextModel,
    tokens: &[u32],
    cache: &mut [LayerCache],
    mode: Mode,
    taps: &[usize],
) -> DFlash2TargetOutput {
    model
        .dflash2_forward_target_on(
            &ids(tokens),
            &positions(cache, tokens.len() as i32),
            Some(cache),
            taps,
            mode,
            ().into(),
        )
        .unwrap()
}
fn joined(arrays: &[Array]) -> Array {
    let refs: Vec<&Array> = arrays.iter().collect();
    mlx::ops::shape::concatenate(&refs, 1).unwrap()
}

#[test]
#[ignore = "requires IRONMLX_FLASH_SNAPSHOT and IRONMLX_FLASH_DRAFT_SNAPSHOT"]
fn flash_pair_proposes_and_verifies_with_exact_transactional_state() {
    let (model, draft) = fixture();
    let taps = &draft.config().dflash_config.target_layer_ids;
    assert_eq!(taps, &[5, 15, 25, 35, 45]);
    let contexts: Vec<usize> = std::env::var("IRONMLX_FLASH_DFLASH_CONTEXTS")
        .unwrap_or_else(|_| "8,64".into())
        .split(',')
        .map(|n| n.parse().unwrap())
        .collect();
    for context in contexts {
        assert!((4..=4096).contains(&context));
        let mut prompt: Vec<u32> = (0..context).map(|n| 100 + n as u32 % 500).collect();
        prompt[2] = 248044;
        let mut ordinary = model
            .make_cache(1, (context + 32) as i32, Dtype::Bfloat16)
            .unwrap();
        let prefill = run(&model, &prompt, &mut ordinary, Mode::Prefill, taps);
        let base = snapshots(&ordinary);
        let mut verify = model
            .make_cache(1, (context + 32) as i32, Dtype::Bfloat16)
            .unwrap();
        let inputs = [42, 43, 248044, 44, 45, 46, 47, 48];
        let mut ordinary_hidden = Vec::new();
        let mut ordinary_context = Vec::new();
        let mut ordinary_logits = Vec::new();
        let mut prefixes = vec![snapshots(&ordinary)];
        for &token in &inputs {
            let output = run(&model, &[token], &mut ordinary, Mode::OrdinaryDecode, taps);
            ordinary_logits.push(
                model
                    .dflash2_project_hidden_on(&output.hidden, ().into())
                    .unwrap(),
            );
            ordinary_hidden.push(output.hidden);
            ordinary_context.push(output.context_hidden);
            prefixes.push(snapshots(&ordinary));
        }
        for width in [2, 4, 8] {
            restore(&mut verify, &base);
            let output = run(
                &model,
                &inputs[..width],
                &mut verify,
                Mode::GreedyVerify,
                taps,
            );
            exact(
                "verify hidden",
                &output.hidden,
                &joined(&ordinary_hidden[..width]),
            );
            exact(
                "verify HC context",
                &output.context_hidden,
                &joined(&ordinary_context[..width]),
            );
            exact(
                "verify logits",
                &model
                    .dflash2_project_hidden_on(&output.hidden, ().into())
                    .unwrap(),
                &joined(&ordinary_logits[..width]),
            );
            restore(&mut ordinary, &prefixes[width]);
            states(&ordinary, &verify);
        }
        for accepted in [0, 1, 4, 8] {
            restore(&mut verify, &base);
            run(&model, &inputs, &mut verify, Mode::GreedyVerify, taps);
            model
                .dflash2_restore_target_prefix_on(&mut verify, &base, accepted, ().into())
                .unwrap();
            restore(&mut ordinary, &prefixes[accepted]);
            states(&ordinary, &verify);
            let expected = run(&model, &[52], &mut ordinary, Mode::OrdinaryDecode, taps);
            let actual = run(&model, &[52], &mut verify, Mode::OrdinaryDecode, taps);
            exact(
                "accepted continuation hidden",
                &actual.hidden,
                &expected.hidden,
            );
            exact(
                "accepted continuation context",
                &actual.context_hidden,
                &expected.context_hidden,
            );
            states(&ordinary, &verify);
            eprintln!("context={context} accepted={accepted}: exact state and continuation");
        }
        if context == 8 {
            let last = prefill
                .hidden
                .slice(
                    &[0, context as i32 - 1, 0][..],
                    &[1, context as i32, model.config.hidden_size][..],
                )
                .unwrap();
            let logits = model
                .dflash2_project_hidden_on(&last, ().into())
                .unwrap()
                .astype(Dtype::Float32)
                .unwrap()
                .to_vec::<f32>()
                .unwrap();
            let anchor = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0 as u32;
            let mut input = vec![draft.config().dflash_config.mask_token_id; 8];
            input[0] = anchor;
            let mut draft_cache = draft.make_cache(0).unwrap();
            let proposed = draft
                .propose_greedy_on(
                    &model,
                    &ids(&input),
                    &prefill.context_hidden,
                    &mut draft_cache,
                    (),
                )
                .unwrap();
            assert_eq!(proposed.shape().as_slice(), [1, 7]);
            let proposed = proposed
                .astype(Dtype::Uint32)
                .unwrap()
                .to_vec::<u32>()
                .unwrap();
            assert!(proposed.iter().all(|&id| id < model.config.vocab_size));
            eprintln!("real Flash DFlash2 proposal: anchor={anchor} draft={proposed:?}");
        }
    }
    assert_active_cleanup(&model);
    eprintln!(
        "memory={:?} expert_cache={:?}",
        mlx::memory::snapshot(),
        model.expert_cache_stats().unwrap()
    );
}

fn fixture() -> (FlashTextModel, DFlash2DraftModel) {
    let path = std::env::var("IRONMLX_FLASH_SNAPSHOT").expect("target snapshot required");
    let draft_path =
        std::env::var("IRONMLX_FLASH_DRAFT_SNAPSHOT").expect("draft snapshot required");
    let paged = std::env::var("IRONMLX_FLASH_EXPERT_CACHE_MIB")
        .ok()
        .map(|n| n.parse::<usize>().unwrap() * 1024 * 1024);
    mlx::memory::set_cache_limit(128 * 1024 * 1024);
    mlx::memory::set_wired_limit(if paged.is_some() {
        24usize << 30
    } else {
        96usize << 30
    })
    .unwrap();
    let loader = if paged.is_some() {
        Loader::open_flash_text_paged(Path::new(&path), 8u64 << 30)
    } else {
        Loader::open_flash_text(Path::new(&path), 82u64 << 30)
    }
    .unwrap();
    let files = Arc::new(ModelTensorFiles::open(Path::new(&path)).unwrap());
    let model = FlashTextModel::from_loader_with_paging(
        &loader,
        files,
        32 * 1024 * 1024,
        16 * 1024 * 1024,
        64 * 1024 * 1024,
        paged.map(|cache_bytes| ExpertPagingOptions {
            cache_bytes,
            staging_bytes: 16 * 1024 * 1024,
            // Long active-state qualification uses the production load-plan
            // pack ceiling; the legacy short-context stress fixture is 64 MiB.
            pack_bytes: if std::env::var("IRONMLX_FLASH_ACTIVE_TEST_ROOT").is_ok() {
                2 << 30
            } else {
                64 << 20
            },
        }),
        FlashExecutionOptions::from_environment().unwrap(),
    )
    .unwrap();
    drop(loader);
    #[cfg(unix)]
    let model = if let Ok(root) = std::env::var("IRONMLX_FLASH_ACTIVE_TEST_ROOT") {
        model
            .with_active_state(ironmlx_lm::core::active_pages::ActivePagesConfig {
                root: root.into(),
                cache_bytes: 64 << 10,
                pack_bytes: 16 << 20,
                disk_bytes: 2 << 30,
                min_free_bytes: 0,
            })
            .unwrap()
    } else {
        model
    };
    let draft_loader = Loader::open_dflash2(Path::new(&draft_path)).unwrap();
    let draft =
        DFlash2DraftModel::from_loader_for_flash(&draft_loader, &model.config, None).unwrap();
    drop(draft_loader);
    (model, draft)
}

#[test]
#[ignore = "requires actual Flash target and draft snapshots"]
fn flash_b2_b4_verify_ragged_commit_matches_independent_requests() {
    use ironmlx_lm::core::cache::layer::adopt_layer_cache_rows;
    let (model, draft) = fixture();
    let taps = &draft.config().dflash_config.target_layer_ids;
    let accepted = [0, 1, 4, 8];
    let inputs = [42, 43, 248044, 44, 45, 46, 47, 48];
    let mut independent = Vec::new();
    let mut bases = Vec::new();
    let mut committed = Vec::new();
    let mut expected_hidden = Vec::new();
    let mut expected_context = Vec::new();
    let long = std::env::var("IRONMLX_FLASH_ACTIVE_TEST_ROOT").is_ok();
    let cap = if long { 4160 } else { 64 };
    for row in 0..4 {
        let lengths = if long {
            [4093, 4096, 4097, 4100]
        } else {
            [8, 12, 17, 24]
        };
        let prompt: Vec<u32> = (0..lengths[row])
            .map(|n| {
                if n == 2 {
                    248044
                } else {
                    100 + row as u32 * 50 + n as u32
                }
            })
            .collect();
        let mut cache = model.make_cache(1, cap, Dtype::Bfloat16).unwrap();
        for chunk in prompt.chunks(2048) {
            run(&model, chunk, &mut cache, Mode::Prefill, taps);
        }
        eprintln!("B4 base row={row} prefilled tokens={}", prompt.len());
        bases.push(snapshots(&cache));
        let mut prefix = snapshots(&cache);
        let mut hidden = Vec::new();
        let mut context = Vec::new();
        for (depth, &token) in inputs.iter().enumerate() {
            let output = run(&model, &[token], &mut cache, Mode::OrdinaryDecode, taps);
            hidden.push(output.hidden);
            context.push(output.context_hidden);
            if depth + 1 == accepted[row] {
                prefix = snapshots(&cache);
            }
        }
        expected_hidden.push(joined(&hidden));
        expected_context.push(joined(&context));
        committed.push(prefix);
        independent.push(cache);
    }
    for width in [2, 4] {
        let mut batch = model
            .make_cache(width as i32, cap, Dtype::Bfloat16)
            .unwrap();
        for row in 0..width {
            restore(&mut independent[row], &bases[row]);
            adopt_layer_cache_rows(&mut batch, &independent[row], row, 0).unwrap();
        }
        let base = snapshots(&batch);
        let flat: Vec<u32> = (0..width).flat_map(|_| inputs).collect();
        let output = model
            .dflash2_forward_target_on(
                &Array::try_from((flat.as_slice(), [width as i32, 8])).unwrap(),
                &positions(&batch, 8),
                Some(&mut batch),
                taps,
                Mode::GreedyVerify,
                ().into(),
            )
            .unwrap();
        let h: Vec<&Array> = expected_hidden[..width].iter().collect();
        let c: Vec<&Array> = expected_context[..width].iter().collect();
        exact(
            "batch verify hidden",
            &output.hidden,
            &mlx::ops::shape::concatenate(&h, 0).unwrap(),
        );
        exact(
            "batch verify captures",
            &output.context_hidden,
            &mlx::ops::shape::concatenate(&c, 0).unwrap(),
        );
        model
            .dflash2_restore_target_prefix_rows_on(&mut batch, &base, &accepted[..width], ().into())
            .unwrap();
        for row in 0..width {
            let mut actual = model.make_cache(1, cap, Dtype::Bfloat16).unwrap();
            adopt_layer_cache_rows(&mut actual, &batch, 0, row).unwrap();
            restore(&mut independent[row], &committed[row]);
            states(&actual, &independent[row]);
            let expected = run(
                &model,
                &[52],
                &mut independent[row],
                Mode::OrdinaryDecode,
                taps,
            );
            let actual_output = run(&model, &[52], &mut actual, Mode::OrdinaryDecode, taps);
            exact(
                "ragged continuation",
                &actual_output.hidden,
                &expected.hidden,
            );
            exact(
                "ragged continuation captures",
                &actual_output.context_hidden,
                &expected.context_hidden,
            );
            states(&actual, &independent[row]);
        }
        if std::env::var("IRONMLX_FLASH_PAGED_BATCHING").as_deref() != Ok("off") {
            let stats = model.paging_telemetry().unwrap().snapshot();
            assert_eq!(
                stats.experts.phases[2].request_width, width as u64,
                "verification must preserve its input request width"
            );
            assert!(stats.experts.phases[2].microbatches > 0);
        }
        eprintln!("B{width} Q8 exact verification and ragged prefix commit");
    }
    drop(independent);
    drop(bases);
    drop(committed);
    assert_active_cleanup(&model);
}

fn assert_active_cleanup(model: &FlashTextModel) {
    #[cfg(unix)]
    if std::env::var("IRONMLX_FLASH_ACTIVE_TEST_ROOT").is_ok() {
        let stats = model
            .paging_telemetry()
            .unwrap()
            .snapshot()
            .active_state
            .unwrap();
        eprintln!("DFlash2 active state after release: {stats:?}");
        assert!(stats.written_bytes > 0 && stats.read_bytes > 0);
        assert_eq!(stats.failures, 0);
        assert_eq!(stats.live_files, 0);
        assert_eq!(stats.disk_bytes, 0);
        assert_eq!(stats.cache_bytes, 0);
    }
}
