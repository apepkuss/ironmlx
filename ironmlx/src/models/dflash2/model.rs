use std::collections::BTreeMap;

use anyhow::anyhow;
use mlx::{Array, Dtype, StreamOrDevice};

use crate::core::Loader;
use crate::models::Qwen35Config;
use crate::nn::{Linear, RmsNorm};
use crate::Result;

use super::attention::DFlash2KvCache;
use super::config::DFlash2Config;
use super::layer::DFlash2DecoderLayer;
use super::selector::DFlash2CandidateSelector;
use super::{load_linear, DFlash2Target};

#[derive(Clone)]
pub struct DFlash2DraftCache {
    layers: Vec<DFlash2KvCache>,
}

impl DFlash2DraftCache {
    pub(crate) fn position_signature(&self) -> Result<(i32, i32)> {
        let first = self
            .layers
            .first()
            .ok_or_else(|| anyhow!("DFlash2 draft cache has no layers"))?;
        let signature = (first.processed(), first.len());
        if self
            .layers
            .iter()
            .any(|layer| (layer.processed(), layer.len()) != signature)
        {
            return Err(anyhow!("DFlash2 draft cache layer positions diverged"));
        }
        Ok(signature)
    }

    pub(crate) fn stack_rows_on(rows: &[&Self], target: StreamOrDevice) -> Result<Self> {
        let layer_count = rows
            .first()
            .ok_or_else(|| anyhow!("DFlash2 draft cache row stack cannot be empty"))?
            .layers
            .len();
        if rows.iter().any(|row| row.layers.len() != layer_count) {
            return Err(anyhow!("DFlash2 draft cache layer count mismatch"));
        }
        let mut layers = Vec::with_capacity(layer_count);
        for layer_index in 0..layer_count {
            let layer_rows = rows
                .iter()
                .map(|row| &row.layers[layer_index])
                .collect::<Vec<_>>();
            layers.push(DFlash2KvCache::stack_rows_on(&layer_rows, target)?);
        }
        Ok(Self { layers })
    }

    pub(crate) fn row_on(&self, row: usize, target: StreamOrDevice) -> Result<Self> {
        self.layers
            .iter()
            .map(|layer| layer.row_on(row, target))
            .collect::<Result<Vec<_>>>()
            .map(|layers| Self { layers })
    }
}

pub struct DFlash2DraftModel {
    config: DFlash2Config,
    fc: Linear,
    hidden_norm: RmsNorm,
    layers: Vec<DFlash2DecoderLayer>,
    norm: RmsNorm,
    selector: DFlash2CandidateSelector,
}

impl DFlash2DraftModel {
    pub fn from_loader(
        loader: &Loader,
        target: &Qwen35Config,
        draft_bits: Option<i32>,
    ) -> Result<Self> {
        if loader.quant_meta().is_some() {
            return Err(anyhow!(
                "DFlash2 execution requires the official unquantized BF16 draft checkpoint"
            ));
        }
        let config = DFlash2Config::from_loader(loader)?;
        config.ensure_target_compatible(target)?;
        validate_tensor_manifest(loader, &config)?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers as usize);
        for index in 0..config.num_hidden_layers {
            layers.push(DFlash2DecoderLayer::from_loader(
                loader, index, &config, draft_bits,
            )?);
        }
        Ok(Self {
            fc: load_linear(loader, "fc", draft_bits)?,
            hidden_norm: RmsNorm::from_loader(loader, "hidden_norm", config.rms_norm_eps)?,
            norm: RmsNorm::from_loader(loader, "norm", config.rms_norm_eps)?,
            selector: DFlash2CandidateSelector::from_loader(loader, &config, draft_bits)?,
            layers,
            config,
        })
    }

    pub fn config(&self) -> &DFlash2Config {
        &self.config
    }

    pub fn make_cache(&self, initial_offset: i32) -> Result<DFlash2DraftCache> {
        let max_size = self.config.sliding_window - 1;
        let layers = (0..self.layers.len())
            .map(|_| DFlash2KvCache::new(max_size, initial_offset))
            .collect::<Result<Vec<_>>>()?;
        Ok(DFlash2DraftCache { layers })
    }

    pub(crate) fn propose_greedy_on<T: DFlash2Target>(
        &self,
        target_model: &T,
        input_ids: &Array,
        target_hidden: &Array,
        cache: &mut DFlash2DraftCache,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        let target = target.into();
        let input_shape = input_ids.shape();
        let input_dims = input_shape.as_slice();
        if input_dims.len() != 2
            || input_dims[0] <= 0
            || input_dims[1] < 2
            || input_dims[1] > self.config.dflash_config.block_size
        {
            return Err(anyhow!(
                "DFlash2 proposal input must be [B,L] with B>0 and 2<=L<={}, got {input_dims:?}",
                self.config.dflash_config.block_size
            ));
        }
        let batch = input_dims[0];
        let _batch_stable_qmm = (batch > 1).then(crate::nn::batch_stable_qmm::linear_scope);
        let hidden_shape = target_hidden.shape();
        let hidden_dims = hidden_shape.as_slice();
        let expected_context = self.config.hidden_size
            * i32::try_from(self.config.dflash_config.target_layer_ids.len())?;
        if hidden_dims.len() != 3
            || hidden_dims[0] != batch
            || hidden_dims[1] <= 0
            || hidden_dims[2] != expected_context
        {
            return Err(anyhow!(
                "DFlash2 target hidden must be [B,S,{expected_context}] with B={batch}, got {hidden_dims:?}"
            ));
        }
        if cache.layers.len() != self.layers.len() {
            return Err(anyhow!(
                "DFlash2 cache layer count {} != model layer count {}",
                cache.layers.len(),
                self.layers.len()
            ));
        }
        let processed = cache.layers[0].processed();
        if cache
            .layers
            .iter()
            .any(|layer| layer.processed() != processed)
        {
            return Err(anyhow!("DFlash2 draft cache layer offsets diverged"));
        }

        let mut hidden = target_model.dflash2_embed_on(input_ids, target)?;
        let context = self
            .hidden_norm
            .forward_on(&self.fc.forward_on(target_hidden, target)?, target)?;
        let context_after = cache.layers[0].len_after_append(hidden_dims[1]);
        let mask = build_block_mask(
            input_dims[1],
            context_after,
            self.config.sliding_window,
            hidden.dtype(),
            target,
        )?;
        for (layer, layer_cache) in self.layers.iter().zip(cache.layers.iter_mut()) {
            hidden = layer.forward_on(&hidden, &context, &mask, layer_cache, target)?;
        }
        hidden = self.norm.forward_on(&hidden, target)?;
        let proposal_hidden = mlx::ops::indexing::slice_strided_on(
            &hidden,
            &[0_i32, 1, 0][..],
            &[batch, input_dims[1], self.config.hidden_size][..],
            &[1_i32, 1, 1][..],
            target,
        )?;
        let logits = target_model.dflash2_project_hidden_on(&proposal_hidden, target)?;
        let anchor = mlx::ops::indexing::slice_strided_on(
            input_ids,
            &[0_i32, 0][..],
            &[batch, 1][..],
            &[1_i32, 1][..],
            target,
        )?;
        self.selector
            .select_greedy_on(&proposal_hidden, &logits, &anchor, target)
    }
}

fn build_block_mask(
    query_len: i32,
    context_len: i32,
    sliding_window: i32,
    dtype: Dtype,
    target: StreamOrDevice,
) -> Result<Array> {
    let key_len = context_len + query_len;
    let mut values = vec![0.0_f32; (query_len * key_len) as usize];
    for query in 0..query_len {
        for key in 0..context_len {
            if context_len + query - key >= sliding_window {
                values[(query * key_len + key) as usize] = f32::NEG_INFINITY;
            }
        }
    }
    let mask: Array = (&values[..], &[1_i32, 1, query_len, key_len][..]).try_into()?;
    mlx::ops::cast::astype_on(&mask, dtype, target).map_err(Into::into)
}

fn validate_tensor_manifest(loader: &Loader, cfg: &DFlash2Config) -> Result<()> {
    let mut expected = BTreeMap::<String, Vec<i32>>::new();
    let h = cfg.hidden_size;
    let i = cfg.intermediate_size;
    let q = cfg.num_attention_heads * cfg.head_dim;
    let kv = cfg.num_key_value_heads * cfg.head_dim;
    let groups = h / cfg.dflash_config.conv_group_size;
    let kernel_projection = 2 * cfg.dflash_config.conv_kernel_size * groups;
    expected.insert("fc.weight".to_owned(), vec![h, h * cfg.num_hidden_layers]);
    expected.insert("hidden_norm.weight".to_owned(), vec![h]);
    expected.insert("norm.weight".to_owned(), vec![h]);
    expected.insert(
        "candidate_selector.hidden_projection.weight".to_owned(),
        vec![cfg.dflash_config.selector_rank, h],
    );
    expected.insert(
        "candidate_selector.predecessor_codebook".to_owned(),
        vec![cfg.vocab_size, cfg.dflash_config.selector_rank],
    );
    expected.insert(
        "candidate_selector.successor_codebook".to_owned(),
        vec![cfg.vocab_size, cfg.dflash_config.selector_rank],
    );
    for layer in 0..cfg.num_hidden_layers {
        let prefix = format!("layers.{layer}");
        for norm in ["input_layernorm", "post_attention_layernorm"] {
            expected.insert(format!("{prefix}.{norm}.weight"), vec![h]);
        }
        for (name, shape) in [
            ("q_proj", vec![q, h]),
            ("k_proj", vec![kv, h]),
            ("v_proj", vec![kv, h]),
            ("o_proj", vec![h, q]),
        ] {
            expected.insert(format!("{prefix}.self_attn.{name}.weight"), shape);
        }
        expected.insert(
            format!("{prefix}.self_attn.q_norm.weight"),
            vec![cfg.head_dim],
        );
        expected.insert(
            format!("{prefix}.self_attn.k_norm.weight"),
            vec![cfg.head_dim],
        );
        expected.insert(format!("{prefix}.mlp.gate_proj.weight"), vec![i, h]);
        expected.insert(format!("{prefix}.mlp.up_proj.weight"), vec![i, h]);
        expected.insert(format!("{prefix}.mlp.down_proj.weight"), vec![h, i]);
        for conv in ["attention_conv", "mlp_conv"] {
            expected.insert(
                format!("{prefix}.{conv}.base_kernel"),
                vec![2, cfg.dflash_config.conv_kernel_size, h],
            );
            expected.insert(
                format!("{prefix}.{conv}.kernel_projection.weight"),
                vec![kernel_projection, h],
            );
        }
    }

    let actual_keys = loader
        .keys()
        .map(str::to_owned)
        .collect::<std::collections::BTreeSet<_>>();
    let expected_keys = expected
        .keys()
        .cloned()
        .collect::<std::collections::BTreeSet<_>>();
    if actual_keys != expected_keys {
        let missing = expected_keys.difference(&actual_keys).collect::<Vec<_>>();
        let unexpected = actual_keys.difference(&expected_keys).collect::<Vec<_>>();
        return Err(anyhow!(
            "DFlash2 tensor manifest mismatch: missing={missing:?} unexpected={unexpected:?}"
        ));
    }
    for (key, expected_shape) in expected {
        let tensor = loader.tensor(&key)?;
        if tensor.dtype() != Dtype::Bfloat16 {
            return Err(anyhow!(
                "DFlash2 tensor {key} must be BF16, got {:?}",
                tensor.dtype()
            ));
        }
        if tensor.shape().as_slice() != expected_shape.as_slice() {
            return Err(anyhow!(
                "DFlash2 tensor {key} shape mismatch: expected={expected_shape:?} actual={:?}",
                tensor.shape().as_slice()
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;
    use std::time::{Duration, Instant};

    fn median_duration(mut values: Vec<Duration>) -> Duration {
        values.sort_unstable();
        values[values.len() / 2]
    }

    fn measure_median(mut run: impl FnMut()) -> Duration {
        run();
        let mut samples = Vec::with_capacity(5);
        for _ in 0..5 {
            let started = Instant::now();
            run();
            samples.push(started.elapsed());
        }
        median_duration(samples)
    }

    fn repeat_batch(row: &Array, batch: i32) -> Array {
        let rows = std::iter::repeat_n(row, batch as usize).collect::<Vec<_>>();
        mlx::ops::shape::concatenate(&rows, 0).expect("repeat batch row")
    }

    fn slice_batch_row(array: &Array, row: i32) -> Array {
        let shape = array.shape();
        let dims = shape.as_slice();
        mlx::ops::indexing::slice_strided(
            array,
            &[row, 0_i32, 0][..],
            &[row + 1, dims[1], dims[2]][..],
            &[1_i32, 1, 1][..],
        )
        .expect("slice batch row")
    }

    fn assert_array_exact(reference: &Array, candidate: &Array, label: &str) {
        let reference = mlx::ops::cast::astype(reference, Dtype::Float32)
            .expect("cast reference")
            .to_vec::<f32>()
            .expect("read reference");
        let candidate = mlx::ops::cast::astype(candidate, Dtype::Float32)
            .expect("cast candidate")
            .to_vec::<f32>()
            .expect("read candidate");
        assert_eq!(reference, candidate, "{label}");
    }

    #[test]
    #[serial(mlx_metal)]
    fn block_mask_preserves_noncausal_proposal_block_and_sliding_context() {
        let mask = build_block_mask(3, 4, 4, Dtype::Float32, StreamOrDevice::default())
            .expect("mask")
            .to_vec::<f32>()
            .expect("materialize");
        let row = |query: usize| &mask[query * 7..query * 7 + 7];
        assert_eq!(row(0), &[f32::NEG_INFINITY, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(
            row(1),
            &[
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0
            ]
        );
        assert_eq!(
            row(2),
            &[
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                0.0,
                0.0,
                0.0,
                0.0
            ]
        );
    }

    #[test]
    #[ignore = "loads the full local Qwen3.8 target and DFlash2 draft checkpoints"]
    #[serial(mlx_metal)]
    fn qwen38_dflash2_b2_kernel_spike_is_row_exact_and_reports_timings() {
        use crate::core::generate::build_position_ids;
        use crate::core::Loader;
        use crate::models::dflash2::{DFlash2Target, DFlash2TargetForwardMode};
        use crate::models::Qwen35Model;

        let target_dir = std::env::var("QWEN38_MODEL").expect("QWEN38_MODEL not set");
        let draft_dir = std::env::var("DFLASH2_MODEL").expect("DFLASH2_MODEL not set");
        let mut target_loader = Loader::open(std::path::Path::new(&target_dir))
            .expect("open Qwen3.8 target checkpoint");
        let target =
            Qwen35Model::from_loader_dflash2(&mut target_loader).expect("load DFlash2 target");
        let draft_loader = Loader::open_dflash2(std::path::Path::new(&draft_dir))
            .expect("open DFlash2 draft checkpoint");
        let draft = DFlash2DraftModel::from_loader(&draft_loader, target.config(), Some(4))
            .expect("load 4-bit runtime DFlash2 draft");

        let batch = 2_i32;
        let block = draft.config().dflash_config.block_size;
        let mask_token = draft.config().dflash_config.mask_token_id;
        let block_values = std::iter::once(100_u32)
            .chain(std::iter::repeat_n(mask_token, (block - 1) as usize))
            .collect::<Vec<_>>();
        let block_b1: Array = (&block_values[..], &[1_i32, block][..])
            .try_into()
            .expect("B1 proposal block");
        let block_b2 = repeat_batch(&block_b1, batch);
        let expected_context = draft.config().hidden_size
            * i32::try_from(draft.config().dflash_config.target_layer_ids.len())
                .expect("target layer count");
        let context_b1 = Array::zeros((1_i32, block, expected_context), target.hidden_dtype())
            .expect("B1 target context");
        let context_b2 = repeat_batch(&context_b1, batch);

        let run_draft_b1_pair = || {
            let mut first_cache = draft.make_cache(0).expect("first B1 draft cache");
            let first = draft
                .propose_greedy_on(
                    &target,
                    &block_b1,
                    &context_b1,
                    &mut first_cache,
                    StreamOrDevice::default(),
                )
                .expect("first B1 proposal");
            mlx::transforms::eval(&[&first]).expect("evaluate first B1 proposal");
            let mut second_cache = draft.make_cache(0).expect("second B1 draft cache");
            let second = draft
                .propose_greedy_on(
                    &target,
                    &block_b1,
                    &context_b1,
                    &mut second_cache,
                    StreamOrDevice::default(),
                )
                .expect("second B1 proposal");
            mlx::transforms::eval(&[&second]).expect("evaluate second B1 proposal");
            (first, second)
        };
        let run_draft_b2 = || {
            let mut cache = draft.make_cache(0).expect("B2 draft cache");
            let output = draft
                .propose_greedy_on(
                    &target,
                    &block_b2,
                    &context_b2,
                    &mut cache,
                    StreamOrDevice::default(),
                )
                .expect("B2 proposal");
            mlx::transforms::eval(&[&output]).expect("evaluate B2 proposal");
            output
        };
        let (draft_reference, _) = run_draft_b1_pair();
        let draft_b2 = run_draft_b2();
        let proposal_len = (block - 1) as usize;
        assert_eq!(
            draft_reference.to_vec::<u32>().expect("B1 draft tokens"),
            draft_b2.to_vec::<u32>().expect("B2 draft tokens")[..proposal_len]
        );
        assert_eq!(
            draft_reference.to_vec::<u32>().expect("B1 draft tokens"),
            draft_b2.to_vec::<u32>().expect("B2 draft tokens")[proposal_len..]
        );

        let verify_values = [101_u32, 102, 103, 104];
        let verify_b1: Array = (&verify_values[..], &[1_i32, 4_i32][..])
            .try_into()
            .expect("B1 verify input");
        let verify_b2 = repeat_batch(&verify_b1, batch);
        let positions_b1 = build_position_ids(0, 4).expect("B1 verify positions");
        let positions_b2 = mlx::ops::shape::broadcast_to(&positions_b1, &[3_i32, batch, 4_i32][..])
            .expect("B2 verify positions");
        let target_layers = &draft.config().dflash_config.target_layer_ids;

        let run_target_b1_pair = |mode| {
            let first = target
                .dflash2_forward_target_on(
                    &verify_b1,
                    &positions_b1,
                    None,
                    target_layers,
                    mode,
                    StreamOrDevice::default(),
                )
                .and_then(|output| {
                    target.dflash2_project_hidden_on(&output.hidden, StreamOrDevice::default())
                })
                .expect("first B1 target verify");
            mlx::transforms::eval(&[&first]).expect("evaluate first B1 target verify");
            let second = target
                .dflash2_forward_target_on(
                    &verify_b1,
                    &positions_b1,
                    None,
                    target_layers,
                    mode,
                    StreamOrDevice::default(),
                )
                .and_then(|output| {
                    target.dflash2_project_hidden_on(&output.hidden, StreamOrDevice::default())
                })
                .expect("second B1 target verify");
            mlx::transforms::eval(&[&second]).expect("evaluate second B1 target verify");
            (first, second)
        };
        let run_target_b2 = |mode| {
            let output = target
                .dflash2_forward_target_on(
                    &verify_b2,
                    &positions_b2,
                    None,
                    target_layers,
                    mode,
                    StreamOrDevice::default(),
                )
                .and_then(|output| {
                    target.dflash2_project_hidden_on(&output.hidden, StreamOrDevice::default())
                })
                .expect("B2 target verify");
            mlx::transforms::eval(&[&output]).expect("evaluate B2 target verify");
            output
        };

        for mode in [
            DFlash2TargetForwardMode::GreedyVerify,
            DFlash2TargetForwardMode::SampledVerify,
        ] {
            let (reference, _) = run_target_b1_pair(mode);
            let candidate = run_target_b2(mode);
            for row in 0..batch {
                assert_array_exact(
                    &reference,
                    &slice_batch_row(&candidate, row),
                    &format!("{mode:?} B2 row {row} diverged from B1"),
                );
            }
        }

        let draft_b1 = measure_median(|| {
            let _ = run_draft_b1_pair();
        });
        let draft_b2 = measure_median(|| {
            let _ = run_draft_b2();
        });
        let target_greedy_b1 = measure_median(|| {
            let _ = run_target_b1_pair(DFlash2TargetForwardMode::GreedyVerify);
        });
        let target_greedy_b2 = measure_median(|| {
            let _ = run_target_b2(DFlash2TargetForwardMode::GreedyVerify);
        });
        let target_sampled_b1 = measure_median(|| {
            let _ = run_target_b1_pair(DFlash2TargetForwardMode::SampledVerify);
        });
        let target_sampled_b2 = measure_median(|| {
            let _ = run_target_b2(DFlash2TargetForwardMode::SampledVerify);
        });
        let speedup =
            |serial: Duration, batched: Duration| serial.as_secs_f64() / batched.as_secs_f64();
        eprintln!(
            "dflash2_b2_spike draft_serial_b1_us={} draft_b2_us={} draft_speedup={:.3} target_greedy_serial_b1_us={} target_greedy_b2_us={} target_greedy_speedup={:.3} target_sampled_serial_b1_us={} target_sampled_b2_us={} target_sampled_speedup={:.3}",
            draft_b1.as_micros(),
            draft_b2.as_micros(),
            speedup(draft_b1, draft_b2),
            target_greedy_b1.as_micros(),
            target_greedy_b2.as_micros(),
            speedup(target_greedy_b1, target_greedy_b2),
            target_sampled_b1.as_micros(),
            target_sampled_b2.as_micros(),
            speedup(target_sampled_b1, target_sampled_b2),
        );
    }
}
