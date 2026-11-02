//! Pinned primary image-tower, three-axis positions and full decoder qualification.
use ironmlx_lm::{
    core::{
        model_input::build_position_ids_vl, tensor_file::ModelTensorFiles, vision::DenseVlMethods,
        Loader, Model,
    },
    models::{
        dflash2::{DFlash2Target, DFlash2TargetForwardMode},
        qwen4_exp::text_model::{ExpertPagingOptions, FlashTextModel},
    },
};
use mlx::{Array, Dtype};
use std::{path::Path, sync::Arc};
fn difference(label: &str, actual: &Array, expected: &Array, tolerance: f64) {
    assert_eq!(actual.shape(), expected.shape(), "{label}");
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
    let max = a
        .iter()
        .zip(&b)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let error = a
        .iter()
        .zip(&b)
        .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
        .sum::<f64>();
    let magnitude = b.iter().map(|b| (*b as f64).powi(2)).sum::<f64>();
    let relative = (error / magnitude.max(1e-30)).sqrt();
    eprintln!("{label} max={max} relative_rms={relative}");
    assert!(
        a.iter().all(|x| x.is_finite()) && relative <= tolerance,
        "{label}"
    );
}
#[test]
#[ignore = "requires IRONMLX_FLASH_SNAPSHOT and IRONMLX_FLASH_IMAGE_ORACLE"]
fn flash_image_tower_positions_full_graph_and_speculative_restore() {
    let snapshot = std::env::var("IRONMLX_FLASH_SNAPSHOT").unwrap();
    let oracle = std::env::var("IRONMLX_FLASH_IMAGE_ORACLE").unwrap();
    let meta: serde_json::Value = serde_json::from_reader(
        std::fs::File::open(Path::new(&oracle).with_extension("json")).unwrap(),
    )
    .unwrap();
    assert_eq!(
        meta["reference_commit"],
        "ff788e98d82f6a931992a6a8ee0e74c7a7e6925e"
    );
    assert_eq!(
        meta["mlx_commit"],
        "a8f66b5901791a38ff527f249e1f1a84f46c03b6"
    );
    let (arrays, _) = mlx::io::load_safetensors(&oracle).unwrap();
    mlx::memory::set_cache_limit(512 << 20);
    mlx::memory::set_wired_limit(24usize << 30).unwrap();
    let loader = Loader::open_flash_text_paged(Path::new(&snapshot), 8 << 30).unwrap();
    let model = FlashTextModel::from_loader_with_paging(
        &loader,
        Arc::new(ModelTensorFiles::open(Path::new(&snapshot)).unwrap()),
        64 << 20,
        16 << 20,
        64 << 20,
        Some(ExpertPagingOptions {
            cache_bytes: 2 << 30,
            staging_bytes: 16 << 20,
            pack_bytes: 2 << 30,
        }),
    )
    .unwrap();
    drop(loader);
    let flat = &arrays["pixel_values"];
    let n = flat.shape().as_slice()[0];
    let pixels = flat
        .reshape([n, 3, 2, 16, 16])
        .unwrap()
        .transpose_axes([0, 2, 1, 3, 4])
        .unwrap();
    let grid = arrays["image_grid_thw"].to_vec::<i32>().unwrap();
    let grids: Vec<_> = grid.chunks_exact(3).map(|x| (x[0], x[1], x[2])).collect();
    let ids = &arrays["input_ids"];
    let tokens = ids.to_vec::<i32>().unwrap();
    let positions = build_position_ids_vl(&tokens, &grids, 248056, 2).unwrap();
    difference(
        "three-axis positions",
        &positions,
        &arrays["position_ids"],
        0.0,
    );
    let vision = model
        .compute_vision_embeds(&[pixels], &grids, ().into())
        .unwrap();
    difference("vision tower", &vision, &arrays["vision_embeds"], 0.0);
    let captures: Vec<usize> = (0..48).collect();
    let cap = tokens.len() as i32 + 128;
    let mut reference_cache = model.make_cache(1, cap, Dtype::Bfloat16).unwrap();
    let mut actual_cache = model.make_cache(1, cap, Dtype::Bfloat16).unwrap();
    // Isolate the text graph from the tower's BF16 rounding before checking
    // native tower -> model behavior separately.
    let expected = model
        .dflash2_forward_target_vl_on(
            ids,
            &arrays["vision_embeds"],
            248056,
            &positions,
            Some(&mut reference_cache),
            &captures,
            ().into(),
        )
        .unwrap();
    for index in 0..48 {
        let capture = expected
            .context_hidden
            .slice(
                &[0, 0, index * 2560][..],
                &[1, tokens.len() as i32, (index + 1) * 2560][..],
            )
            .unwrap();
        difference(
            &format!("image prefill layer {index}"),
            &capture,
            &arrays[&format!("prefill_layer_{index}")],
            0.0,
        );
    }
    let native = model
        .dflash2_forward_target_vl_on(
            ids,
            &vision,
            248056,
            &positions,
            Some(&mut actual_cache),
            &[],
            ().into(),
        )
        .unwrap();
    let mut hidden = expected.hidden;
    let mut native_hidden = native.hidden;
    let delta = meta["rope_delta"].as_i64().unwrap() as i32;
    for step in 0..4 {
        let last = hidden.shape().as_slice()[1] - 1;
        let logits = model
            .project_hidden_on(&hidden, ().into())
            .unwrap()
            .slice(&[0, last, 0][..], &[1, last + 1, 248320][..])
            .unwrap();
        let native_logits = model
            .project_hidden_on(&native_hidden, ().into())
            .unwrap()
            .slice(&[0, last, 0][..], &[1, last + 1, 248320][..])
            .unwrap();
        difference(
            &format!("image logits {step}"),
            &logits,
            &arrays[&format!("logits_{step}")],
            0.0,
        );
        difference(
            &format!("native image logits {step}"),
            &native_logits,
            &logits,
            0.0,
        );
        let token = logits.argmax(-1, false).unwrap().to_vec::<u32>().unwrap()[0];
        assert_eq!(
            native_logits
                .argmax(-1, false)
                .unwrap()
                .to_vec::<u32>()
                .unwrap()[0],
            token
        );
        let next = Array::try_from((&[token][..], [1, 1])).unwrap();
        let pos =
            Array::try_from((&[tokens.len() as i32 + step + delta; 3][..], [3, 1, 1])).unwrap();
        hidden = model
            .forward_text_hidden(
                &next,
                &pos,
                None,
                None,
                Some(&mut reference_cache),
                ().into(),
            )
            .unwrap();
        native_hidden = model
            .forward_text_hidden(&next, &pos, None, None, Some(&mut actual_cache), ().into())
            .unwrap();
    }
    // Image positions survive speculative QSA trim and PLE/GDN prefix commit.
    let base: Vec<_> = actual_cache.iter().map(|x| x.snapshot()).collect();
    let verify_ids = Array::try_from((&[100u32, 101, 102, 103][..], [1, 4])).unwrap();
    let start = tokens.len() as i32 + 4 + delta;
    let positions = Array::try_from((
        &[start, start + 1, start + 2, start + 3].repeat(3)[..],
        [3, 1, 4],
    ))
    .unwrap();
    let speculative = model
        .dflash2_forward_target_on(
            &verify_ids,
            &positions,
            Some(&mut actual_cache),
            &[5, 15, 25, 35, 45],
            DFlash2TargetForwardMode::GreedyVerify,
            ().into(),
        )
        .unwrap();
    speculative.hidden.eval().unwrap();
    model
        .dflash2_restore_target_prefix_on(&mut actual_cache, &base, 2, ().into())
        .unwrap();
    let mut replay = model.make_cache(1, cap, Dtype::Bfloat16).unwrap();
    for (row, snapshot) in replay.iter_mut().zip(&base) {
        row.restore(snapshot).unwrap();
    }
    for i in 0..2 {
        model
            .forward_text_hidden(
                &verify_ids.slice(&[0, i][..], &[1, i + 1][..]).unwrap(),
                &positions.slice(&[0, 0, i][..], &[3, 1, i + 1][..]).unwrap(),
                None,
                None,
                Some(&mut replay),
                ().into(),
            )
            .unwrap()
            .eval()
            .unwrap();
    }
    let next = Array::try_from((&[104u32][..], [1, 1])).unwrap();
    let pos = Array::try_from((&[start + 2; 3][..], [3, 1, 1])).unwrap();
    let a = model
        .forward_on(&next, &pos, None, None, Some(&mut actual_cache), ().into())
        .unwrap();
    let b = model
        .forward_on(&next, &pos, None, None, Some(&mut replay), ().into())
        .unwrap();
    difference("image speculative rollback continuation", &a, &b, 0.0);
}

#[test]
#[ignore = "requires IRONMLX_FLASH_SNAPSHOT and IRONMLX_FLASH_TOWER_ORACLE"]
fn flash_non_square_image_preprocessing_and_tower_match_primary() {
    let snapshot = std::env::var("IRONMLX_FLASH_SNAPSHOT").unwrap();
    let oracle = std::env::var("IRONMLX_FLASH_TOWER_ORACLE").unwrap();
    let metadata: serde_json::Value = serde_json::from_reader(
        std::fs::File::open(Path::new(&oracle).with_extension("json")).unwrap(),
    )
    .unwrap();
    assert_eq!(
        metadata["reference_commit"],
        "ff788e98d82f6a931992a6a8ee0e74c7a7e6925e"
    );
    let (arrays, _) = mlx::io::load_safetensors(&oracle).unwrap();
    let loader = Loader::open_vision(Path::new(&snapshot)).unwrap();
    let config =
        serde_json::from_value(loader.config_raw_value()["vision_config"].clone()).unwrap();
    let slot = ironmlx_lm::models::vision::VisionSlot::from_loader(&loader, Some(&config)).unwrap();
    let flat = &arrays["pixel_values"];
    let n = flat.shape().as_slice()[0];
    let pixels = flat
        .reshape([n, 3, 2, 16, 16])
        .unwrap()
        .transpose_axes([0, 2, 1, 3, 4])
        .unwrap();
    let grid = arrays["image_grid_thw"].to_vec::<i32>().unwrap();
    let grids: Vec<_> = grid.chunks_exact(3).map(|x| (x[0], x[1], x[2])).collect();
    let image = std::fs::read(metadata["image"].as_str().unwrap()).unwrap();
    let bounds = ironmlx_lm::models::qwen3_5::image_processor::PixelBounds::for_model(
        Path::new(&snapshot),
        None,
    )
    .unwrap();
    let (native, h, w) =
        ironmlx_lm::models::qwen3_5::image_processor::preprocess_numpy_rescale(&image, bounds)
            .unwrap();
    assert_eq!(grids, [(1, h, w)]);
    difference(
        "non-square preprocessor",
        &native.astype(Dtype::Bfloat16).unwrap(),
        &pixels,
        0.0,
    );
    let output = slot.get().unwrap().forward(&native, &grids).unwrap();
    difference(
        "non-square vision tower",
        &output,
        &arrays["vision_embeds"],
        0.0,
    );
}
