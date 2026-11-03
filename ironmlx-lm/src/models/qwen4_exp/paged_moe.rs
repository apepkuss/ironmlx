//! Flash adapter for the shared bounded expert cache. Router/shared weights
//! stay resident; only selected routed experts enter a pinned execution batch.
use super::config::FlashTextConfig;
use crate::{
    core::{
        paged_experts::PagedExperts,
        weights::{QuantMeta, WeightSource},
    },
    models::qwen3_5_moe::sparse_moe::{CompactExpertWeights, RoutedExperts},
    nn::{Linear, Mlp},
};
use anyhow::{ensure, Context, Result};
use mlx::{Array, Dtype, StreamOrDevice};
use std::{
    collections::{BTreeSet, HashMap},
    sync::{Arc, Mutex},
};

pub struct PagedMoe {
    router: Linear,
    shared: Mlp,
    shared_gate: Linear,
    store: Arc<Mutex<PagedExperts>>,
    table: usize,
    quant: QuantMeta,
    cfg: FlashTextConfig,
}

impl PagedMoe {
    pub fn from_weights(
        source: &(impl WeightSource + ?Sized),
        prefix: &str,
        cfg: &FlashTextConfig,
        store: Arc<Mutex<PagedExperts>>,
        table: usize,
    ) -> Result<Self> {
        let quant = source
            .quant_meta_for(&format!("{prefix}.switch_mlp.gate_proj"))
            .context("missing paged expert quantization metadata")?;
        ensure!(
            store
                .lock()
                .map_err(|_| anyhow::anyhow!("expert cache lock poisoned"))?
                .max_batch_experts(table)?
                >= cfg.num_experts_per_tok as usize,
            "expert cache/pack budget must fit one token's complete top-k working set"
        );
        Ok(Self {
            router: Linear::from_loader(source, &format!("{prefix}.gate"))?,
            shared: Mlp::from_loader(source, &format!("{prefix}.shared_expert"))?,
            shared_gate: Linear::from_loader(source, &format!("{prefix}.shared_expert_gate"))?,
            store,
            table,
            quant,
            cfg: cfg.clone(),
        })
    }

    pub fn forward_on(&self, input: &Array, target: StreamOrDevice, layer: i32) -> Result<Array> {
        let shape = input.shape();
        let dims = shape.as_slice();
        ensure!(
            dims.len() == 3 && dims[0] == 1 && dims[1] > 0 && dims[2] == self.cfg.hidden_size,
            "initial paged MoE path requires B1 [1,S,hidden]"
        );
        let seq = dims[1];
        let hidden = dims[2];
        let topk = self.cfg.num_experts_per_tok;
        let ane_pending = self.shared.flash_ane_begin_on(input, target)?;
        let logits = self.router.forward_on(input, target)?;
        let probs = mlx::ops::softmax_on(&logits, -1, true, target)?;
        let indices = mlx::ops::sort::argpartition_on(&probs, -topk, -1, target)?.slice(
            &[0, 0, self.cfg.num_experts - topk][..],
            &[1, seq, self.cfg.num_experts][..],
        )?;
        let scores = mlx::ops::indexing::take_along_axis_on(&probs, &indices, -1, target)?;
        let scores = if self.cfg.norm_topk_prob {
            mlx::ops::binary::divide_on(&scores, &scores.sum_on(-1, true, target)?, target)?
        } else {
            scores
        };
        let indices = indices.astype_on(Dtype::Uint32, target)?.to_vec::<u32>()?;
        let flat = input.reshape_on([seq, hidden], target)?;
        let scores = scores.reshape_on([seq, topk], target)?;
        let mut outputs = Vec::new();
        let mut start = 0usize;
        while start < seq as usize {
            let mut store = self
                .store
                .lock()
                .map_err(|_| anyhow::anyhow!("expert cache lock poisoned"))?;
            let limit = store.max_batch_experts(self.table)?;
            let mut chosen = BTreeSet::new();
            let mut end = start;
            while end < seq as usize {
                let mut next = chosen.clone();
                next.extend(
                    indices[end * topk as usize..(end + 1) * topk as usize]
                        .iter()
                        .copied(),
                );
                if next.len() > limit {
                    break;
                }
                chosen = next;
                end += 1;
            }
            ensure!(
                end > start,
                "expert working set cannot make progress inside its reservation"
            );
            let chosen: Vec<u32> = chosen.into_iter().collect();
            let batch = store.pack(self.table, &chosen, target)?;
            drop(store);
            let mapping: HashMap<u32, u32> = chosen
                .iter()
                .enumerate()
                .map(|(i, &id)| (id, i as u32))
                .collect();
            let local: Vec<u32> = indices[start * topk as usize..end * topk as usize]
                .iter()
                .map(|id| mapping[id])
                .collect();
            let local = Array::try_from((local.as_slice(), [(end - start) as i32, topk]))?;
            let mut compact = RoutedExperts::from_compact_affine(CompactExpertWeights {
                gate_up: batch.arrays.gate_up.clone(),
                down: batch.arrays.down.clone(),
                quant: self.quant,
            })?;
            let x = flat.slice(&[start as i32, 0][..], &[end as i32, hidden][..])?;
            compact.keep_gate_up_split()?;
            compact.set_dispatch_expert_count(self.cfg.num_experts)?;
            let score = scores.slice(&[start as i32, 0][..], &[end as i32, topk][..])?;
            let output = compact.apply_experts(&x, &local, &score, target, layer)?;
            // Finish every consumer before releasing the pins. Packed arrays
            // and decoder staging never accumulate across expert batches.
            output.eval()?;
            drop(compact);
            drop(batch);
            outputs.push(output);
            start = end;
        }
        let refs: Vec<&Array> = outputs.iter().collect();
        let routed = mlx::ops::shape::concatenate_on(&refs, 0, target)?
            .reshape_on([1, seq, hidden], target)?;
        let shared = self
            .shared
            .flash_ane_finish_on(ane_pending, input, &routed, target)?;
        let gate = {
            self.shared_gate
                .forward_singleton_gather_on(input, target)?
                .sigmoid_on(target)?
        };
        Ok(mlx::ops::binary::add_on(
            &routed,
            &mlx::ops::binary::multiply_on(&shared, &gate, target)?,
            target,
        )?)
    }
}
