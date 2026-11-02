//! Full ordinary/DFlash2 generation on the same Flash target, including a
//! multi-window run and a budget smaller than the draft block.
use ironmlx_core::sampler::Sampler;
use ironmlx_lm::{
    core::{tensor_file::ModelTensorFiles, Loader, Tokenizer},
    models::{
        qwen4_exp::text_model::{ExpertPagingOptions, FlashTextModel},
        DFlash2DraftModel,
    },
};
use ironmlx_runtime::core::{
    DFlash2TextGenerationStream, GenerateEvent, GenerateRequest, GenerationStream,
};
use std::{path::Path, sync::Arc};

fn request(prompt_ids: Vec<u32>, max_new_tokens: usize) -> GenerateRequest {
    GenerateRequest {
        priority: Default::default(),
        prompt_ids,
        max_new_tokens,
        sampler: Sampler::greedy(),
        stop_token_ids: vec![248044, 248046],
        prefill_chunk_size: 2048,
        decode_cadence_mid_chunk_cap: 1,
        kv_cache_turboquant_bits: None,
        pixel_values: None,
        image_grid_thw: None,
        image_spatial_merge_size: 2,
        image_token_id: 248056,
        constraint: None,
    }
}
fn drain(
    mut next: impl FnMut() -> anyhow::Result<Option<GenerateEvent>>,
) -> Vec<(u32, Option<&'static str>)> {
    let mut events = Vec::new();
    while let Some(event) = next().unwrap() {
        let finished = event.finish_reason.is_some();
        events.push((event.token, event.finish_reason));
        if finished {
            break;
        }
    }
    events
}

#[test]
#[ignore = "requires IRONMLX_FLASH_SNAPSHOT and IRONMLX_FLASH_DRAFT_SNAPSHOT"]
fn flash_ordinary_and_real_dflash2_generation_match_exactly() {
    let path = std::env::var("IRONMLX_FLASH_SNAPSHOT").expect("target snapshot required");
    let draft_path =
        std::env::var("IRONMLX_FLASH_DRAFT_SNAPSHOT").expect("draft snapshot required");
    mlx::memory::set_cache_limit(128 * 1024 * 1024);
    mlx::memory::set_wired_limit(24usize << 30).unwrap();
    let full = std::env::var("IRONMLX_FLASH_QUALIFY_FULL").as_deref() == Ok("1");
    if full {
        mlx::memory::set_wired_limit(96usize << 30).unwrap();
    }
    let loader = if full {
        Loader::open_flash_text(Path::new(&path), 82u64 << 30)
    } else {
        Loader::open_flash_text_paged(Path::new(&path), 8u64 << 30)
    }
    .unwrap();
    let tokenizer = Tokenizer::from_loader(&loader).unwrap();
    let files = Arc::new(ModelTensorFiles::open(Path::new(&path)).unwrap());
    let model = FlashTextModel::from_loader_with_paging(
        &loader,
        files,
        32 * 1024 * 1024,
        16 * 1024 * 1024,
        64 * 1024 * 1024,
        (!full).then_some(ExpertPagingOptions {
            cache_bytes: 64 * 1024 * 1024,
            staging_bytes: 16 * 1024 * 1024,
            pack_bytes: 64 * 1024 * 1024,
        }),
    )
    .unwrap();
    drop(loader);
    let loader = Loader::open_dflash2(Path::new(&draft_path)).unwrap();
    let draft = DFlash2DraftModel::from_loader_for_target(&loader, &model, None).unwrap();
    drop(loader);
    let image_fixture = std::env::var("IRONMLX_FLASH_IMAGE_ORACLE")
        .ok()
        .map(|path| mlx::io::load_safetensors(&path).unwrap().0);
    let prompt = tokenizer
        .encode(
            "<|im_start|>user\nGive a friendly greeting.<|im_end|>\n<|im_start|>assistant\n",
            false,
        )
        .unwrap();
    let prompt = if let Some(arrays) = &image_fixture {
        arrays["input_ids"]
            .astype(mlx::Dtype::Uint32)
            .unwrap()
            .to_vec::<u32>()
            .unwrap()
    } else {
        prompt
    };
    let make_request = |budget| {
        let mut request = request(prompt.clone(), budget);
        if let Some(arrays) = &image_fixture {
            let flat = &arrays["pixel_values"];
            let n = flat.shape().as_slice()[0];
            request.pixel_values = Some(vec![flat
                .reshape([n, 3, 2, 16, 16])
                .unwrap()
                .transpose_axes([0, 2, 1, 3, 4])
                .unwrap()]);
            let grid = arrays["image_grid_thw"].to_vec::<i32>().unwrap();
            request.image_grid_thw =
                Some(grid.chunks_exact(3).map(|x| (x[0], x[1], x[2])).collect());
            request.prefill_chunk_size = 8;
        }
        request
    };
    for budget in [17, 3] {
        let ordinary_start = std::time::Instant::now();
        let mut ordinary = if image_fixture.is_some() {
            GenerationStream::new(&model, &tokenizer, make_request(budget)).unwrap()
        } else {
            GenerationStream::new_text_only(&model, &tokenizer, make_request(budget)).unwrap()
        };
        let expected = drain(|| ordinary.next_token());
        drop(ordinary);
        let ordinary_seconds = ordinary_start.elapsed().as_secs_f64();
        let draft_start = std::time::Instant::now();
        let mut speculative = DFlash2TextGenerationStream::new_text_only(
            &model,
            &draft,
            &tokenizer,
            make_request(budget),
            8,
        )
        .unwrap();
        let actual = drain(|| speculative.next_token());
        let draft_seconds = draft_start.elapsed().as_secs_f64();
        eprintln!("timing full={full} budget={budget} ordinary_s={ordinary_seconds:.6} speculative_s={draft_seconds:.6} verify_order={}",std::env::var("IRONMLX_FLASH_VERIFY_ORDER").unwrap_or_else(|_|"layer".to_owned()));
        let metrics = speculative.metrics();
        assert_eq!(
            actual, expected,
            "Flash DFlash2 generation drift, budget={budget}"
        );
        if budget == 17 {
            assert!(
                metrics.windows > 0 && metrics.drafted_tokens > 0,
                "actual draft execution required"
            );
        }
        eprintln!(
            "budget={budget} tokens={} windows={} drafted={} accepted={} rollbacks={} exact",
            actual.len(),
            metrics.windows,
            metrics.drafted_tokens,
            metrics.accepted_draft_tokens,
            metrics.rollback_count
        );
    }
}
