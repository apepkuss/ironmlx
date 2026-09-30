//! Same-model serial/speculative equality and stage accounting, not API acceptance.
use anyhow::{ensure, Result};
use clap::Parser;
use ironmlx_core::sampler::Sampler;
use ironmlx_lm::{
    models::{dflash2::DFlash2DraftModel, Qwen35Model},
    Loader, Tokenizer,
};
use ironmlx_runtime::core::{
    dflash2::{DFlash2P2Options, DFlash2TextGenerationStream},
    generation_types::GenerateRequest,
};
use serde_json::{json, Value};
use std::{path::PathBuf, time::Instant};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    target: PathBuf,
    #[arg(long)]
    draft: PathBuf,
    #[arg(long)]
    prompts: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = 128)]
    max_tokens: usize,
    #[arg(long, default_value = "0,7", value_delimiter = ',')]
    budgets: Vec<usize>,
    #[arg(long)]
    require_natural_stop: bool,
    #[arg(long)]
    trace_layers: bool,
    #[arg(long, default_value_t = 0)]
    tree_nodes: usize,
}

fn trace_layers(target: &Qwen35Model, prompt: &[u32], tokens: &[u32]) -> Result<Value> {
    use ironmlx_lm::core::model_input::build_position_ids;
    use ironmlx_lm::{
        models::dflash2::{DFlash2Target, DFlash2TargetForwardMode as Mode},
        Model,
    };
    use mlx::{Array, Dtype};
    let layers = (0..target.config().num_hidden_layers as usize).collect::<Vec<_>>();
    let width = target.config().hidden_size as usize;
    let mut wide = target.make_cache(1, 8192, target.cache_dtype())?;
    let mut serial = target.make_cache(1, 8192, target.cache_dtype())?;
    let input: Array = (prompt, &[1, prompt.len() as i32][..]).try_into()?;
    let positions = build_position_ids(0, prompt.len() as i32)?;
    for cache in [&mut wide, &mut serial] {
        let out = target.dflash2_forward_target_on(
            &input,
            &positions,
            Some(cache),
            &layers,
            Mode::Prefill,
            ().into(),
        )?;
        mlx::transforms::eval(&[&out.hidden, &out.context_hidden])?;
    }
    let tokens = &tokens[..tokens.len().min(8)];
    let input: Array = (tokens, &[1, tokens.len() as i32][..]).try_into()?;
    let positions = build_position_ids(prompt.len() as i32, tokens.len() as i32)?;
    for layer in &mut wide {
        layer.begin_speculative_prefix_capture()?;
    }
    let mut expected_rows = (0..tokens.len()).collect::<Vec<_>>();
    let out = if std::env::var("IRONMLX_EXPERIMENTAL_DFLASH2_FLAT_TREE").as_deref() == Ok("1") {
        ensure!(
            tokens.len() == 8,
            "tree layer trace needs eight reference tokens"
        );
        let parents = [-1, 0, 0, 1, 2, 3, 4, 0];
        expected_rows = vec![0, 1, 1, 2, 2, 3, 3, 1];
        let nodes = expected_rows.iter().map(|&r| tokens[r]).collect::<Vec<_>>();
        let input: Array = (nodes.as_slice(), &[1, 8][..]).try_into()?;
        target.dflash2_forward_tree_on(
            &input,
            &parents,
            prompt.len() as i32,
            &mut wide,
            &layers,
            ().into(),
        )?
    } else {
        target.dflash2_forward_target_on(
            &input,
            &positions,
            Some(&mut wide),
            &layers,
            Mode::GreedyVerify,
            ().into(),
        )?
    };
    let actual = mlx::ops::cast::astype(&out.context_hidden, Dtype::Float32)?.to_vec::<f32>()?;
    let mut expected = Vec::new();
    for (i, token) in tokens.iter().enumerate() {
        let input: Array = (&[*token][..], &[1, 1][..]).try_into()?;
        let positions = build_position_ids((prompt.len() + i) as i32, 1)?;
        let out = target.dflash2_forward_target_on(
            &input,
            &positions,
            Some(&mut serial),
            &layers,
            Mode::OrdinaryDecode,
            ().into(),
        )?;
        expected
            .extend(mlx::ops::cast::astype(&out.context_hidden, Dtype::Float32)?.to_vec::<f32>()?);
    }
    let mut differences = Vec::new();
    for layer in layers.iter().copied() {
        let mut count = 0;
        let mut max_abs = 0.0_f32;
        for row in 0..tokens.len() {
            for column in 0..width {
                let index = row * layers.len() * width + layer * width + column;
                let expected_index =
                    expected_rows[row] * layers.len() * width + layer * width + column;
                if actual[index] != expected[expected_index] {
                    count += 1;
                    max_abs = max_abs.max((actual[index] - expected[expected_index]).abs());
                }
            }
        }
        differences.push(json!({"layer":layer,"mismatches":count,"max_abs":max_abs}));
    }
    println!(
        "first divergent layer: {:?}",
        differences
            .iter()
            .find(|x| x["mismatches"].as_u64() != Some(0))
    );
    Ok(json!(differences))
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        !args.output.exists(),
        "preserve existing report; choose a new output"
    );
    std::env::set_var("MLX_ENABLE_TF32", "0");
    let root = std::env::var("MLX_DIR")?;
    mlx::metal::set_metallib_path(&format!("{root}/lib/mlx.metallib"))?;
    let mut loader = Loader::open(&args.target)?;
    let tokenizer = Tokenizer::from_loader(&loader)?;
    let target = Qwen35Model::from_loader_dflash2(&mut loader)?;
    drop(loader);
    let loader = Loader::open_dflash2(&args.draft)?;
    let draft = DFlash2DraftModel::from_loader(&loader, target.config(), Some(4))?;
    drop(loader);
    let prompts: Vec<Value> = serde_json::from_slice(&std::fs::read(&args.prompts)?)?;
    let mut records = Vec::new();
    let mut all_equal = true;
    let mut traces = Vec::new();
    for prompt in prompts {
        let text = tokenizer.apply_chat_template(
            &[json!({"role":"user","content":prompt["prompt"]})],
            true,
            Some(&json!({"enable_thinking":false})),
        )?;
        let prompt_ids = tokenizer.encode(&text, false)?;
        let mut reference = None;
        for budget in &args.budgets {
            std::env::set_var(
                "IRONMLX_EXPERIMENTAL_DFLASH2_FIXED_BUDGET",
                budget.to_string(),
            );
            let request = GenerateRequest {
                priority: Default::default(),
                prompt_ids: prompt_ids.clone(),
                max_new_tokens: args.max_tokens,
                sampler: Sampler::greedy(),
                stop_token_ids: tokenizer.eos_token_ids().to_vec(),
                prefill_chunk_size: 0,
                decode_cadence_mid_chunk_cap: 1,
                kv_cache_turboquant_bits: None,
                pixel_values: None,
                image_grid_thw: None,
                image_spatial_merge_size: 2,
                image_token_id: 248_056,
                constraint: None,
            };
            let started = Instant::now();
            let mut stream = DFlash2TextGenerationStream::new_text_only_with_options(
                &target,
                &draft,
                &tokenizer,
                request,
                8,
                DFlash2P2Options {
                    tree_max_nodes: args.tree_nodes,
                    ..Default::default()
                },
            )?;
            let mut tokens = Vec::new();
            let mut stop = None;
            while let Some(event) = stream.next_token()? {
                tokens.push(event.token);
                if event.finish_reason.is_some() {
                    stop = event.finish_reason;
                }
            }
            let same = reference.as_ref().is_none_or(|r| r == &tokens);
            let mismatch = reference
                .as_ref()
                .and_then(|r: &Vec<u32>| r.iter().zip(&tokens).position(|(a, b)| a != b));
            if reference.is_none() {
                reference = Some(tokens.clone());
            }
            all_equal &= same;
            let natural = tokenizer
                .eos_token_ids()
                .contains(tokens.last().unwrap_or(&u32::MAX));
            let metrics = stream.metrics();
            println!(
                "{} d={} tokens={} equal={} first_mismatch={:?} stop={:?} tps={:.2}",
                prompt["id"],
                budget,
                tokens.len(),
                same,
                mismatch,
                stop,
                metrics.generation_tps
            );
            records.push(json!({"prompt":prompt["id"],"budget":budget,"tokens":tokens,"same_as_first_budget":same,
                "first_mismatch":mismatch,"natural_stop":natural,"finish":format!("{stop:?}"),"metrics":metrics,
                "wall_s":started.elapsed().as_secs_f64()}));
            std::fs::write(
                &args.output,
                serde_json::to_vec_pretty(&json!({"records":records,"all_equal":all_equal,
                "m5_qmm":std::env::var("IRONMLX_EXPERIMENTAL_M5_DFLASH2_QMM").ok(),"max_tokens":args.max_tokens}))?,
            )?;
            ensure!(
                !args.require_natural_stop || natural,
                "response did not stop naturally"
            );
        }
        if args.trace_layers {
            traces.push(json!({"prompt":prompt["id"],"layers":trace_layers(&target,&prompt_ids,reference.as_ref().unwrap())?}));
            std::fs::write(
                args.output.with_extension("layers.json"),
                serde_json::to_vec_pretty(&traces)?,
            )?;
        }
    }
    ensure!(
        all_equal,
        "serial/speculative tokens differ; inspect retained report"
    );
    Ok(())
}
