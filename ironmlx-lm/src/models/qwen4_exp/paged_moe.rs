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
use std::{sync::atomic::Ordering::Relaxed, time::Instant};

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
            dims.len() == 3 && dims[0] > 0 && dims[1] > 0 && dims[2] == self.cfg.hidden_size,
            "paged MoE requires nonempty [B,S,hidden]"
        );
        let rows = (0..dims[0])
            .map(|r| input.slice(&[r, 0, 0][..], &[r + 1, dims[1], dims[2]][..]))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let owners: Vec<_> = (0..rows.len()).collect();
        let outputs = self.forward_rows_on(&rows, &owners, target, layer)?;
        Ok(mlx::ops::shape::concatenate_on(
            &outputs.iter().collect::<Vec<_>>(),
            0,
            target,
        )?)
    }

    /// Router/shared projections keep their B1 shape. Expert pieces with the
    /// same B1 QMV/QMM dispatch class share a pack and one gather execution.
    /// Ragged rows are interleaved fairly; outputs are restored by token offset.
    pub fn forward_rows_on(
        &self,
        inputs: &[Array],
        owners: &[usize],
        target: StreamOrDevice,
        layer: i32,
    ) -> Result<Vec<Array>> {
        ensure!(
            !inputs.is_empty() && owners.len() == inputs.len(),
            "invalid MoE request group"
        );
        let mut scores = Vec::new();
        let mut indices = Vec::new();
        let mut lengths = Vec::new();
        let k = self.cfg.num_experts_per_tok;
        for input in inputs {
            let shape = input.shape();
            let dims = shape.as_slice();
            ensure!(
                dims.len() == 3 && dims[0] == 1 && dims[1] > 0 && dims[2] == self.cfg.hidden_size,
                "invalid grouped MoE row"
            );
            let seq = dims[1];
            lengths.push(seq as usize);
            let logits = self.router.forward_on(input, target)?;
            let probs = mlx::ops::softmax_on(&logits, -1, true, target)?;
            let selected = mlx::ops::sort::argpartition_on(&probs, -k, -1, target)?.slice(
                &[0, 0, self.cfg.num_experts - k][..],
                &[1, seq, self.cfg.num_experts][..],
            )?;
            let score = mlx::ops::indexing::take_along_axis_on(&probs, &selected, -1, target)?;
            let score = if self.cfg.norm_topk_prob {
                mlx::ops::binary::divide_on(&score, &score.sum_on(-1, true, target)?, target)?
            } else {
                score
            };
            indices.extend(selected.astype_on(Dtype::Uint32, target)?.to_vec::<u32>()?);
            scores.push(score.reshape_on([seq, k], target)?);
        }
        let flat = mlx::ops::shape::concatenate_on(&inputs.iter().collect::<Vec<_>>(), 1, target)?
            .reshape_on([-1, self.cfg.hidden_size], target)?;
        let scores =
            mlx::ops::shape::concatenate_on(&scores.iter().collect::<Vec<_>>(), 0, target)?;
        let (limit, telemetry) = {
            let store = self
                .store
                .lock()
                .map_err(|_| anyhow::anyhow!("expert cache lock poisoned"))?;
            (
                store.max_batch_experts(self.table)?,
                Arc::clone(store.telemetry()),
            )
        };
        let groups = plan_groups(
            &indices,
            &lengths,
            owners,
            k as usize,
            limit,
            self.cfg.num_experts as usize,
        )?;
        telemetry.experts.current().batches.fetch_add(1, Relaxed);
        let classes = groups
            .iter()
            .flat_map(|g| g.pieces.iter().map(|p| p.class))
            .collect::<BTreeSet<_>>()
            .len();
        telemetry
            .experts
            .current()
            .arithmetic_splits
            .fetch_add(classes.saturating_sub(1) as u64, Relaxed);
        telemetry.experts.current().request_width.fetch_max(
            owners.iter().copied().collect::<BTreeSet<_>>().len() as u64,
            Relaxed,
        );
        telemetry
            .experts
            .current()
            .route_count
            .fetch_add(indices.len() as u64, Relaxed);
        let mut outputs = std::collections::BTreeMap::new();
        for (group_index, group) in groups.iter().enumerate() {
            telemetry
                .experts
                .current()
                .microbatches
                .fetch_add(1, Relaxed);
            telemetry.experts.current().max_width.fetch_max(
                group
                    .pieces
                    .iter()
                    .map(|p| p.owner)
                    .collect::<BTreeSet<_>>()
                    .len() as u64,
                Relaxed,
            );
            if group.budget_split {
                telemetry
                    .experts
                    .current()
                    .budget_splits
                    .fetch_add(1, Relaxed);
            }
            let chosen: Vec<_> = group.experts.iter().copied().collect();
            let batch = PagedExperts::pack_shared(&self.store, self.table, &chosen, target)?;
            if let Some(next) = groups.get(group_index + 1) {
                let waiting = Instant::now();
                let mut store = self
                    .store
                    .lock()
                    .map_err(|_| anyhow::anyhow!("expert cache lock poisoned"))?;
                telemetry
                    .experts
                    .current()
                    .lock_wait_us
                    .fetch_add(crate::core::paging_stats::elapsed_us(waiting), Relaxed);
                for &expert in &next.experts {
                    store.prefetch(self.table, expert)?;
                }
            }
            let mapping: HashMap<_, _> = chosen
                .iter()
                .enumerate()
                .map(|(i, &id)| (id, i as u32))
                .collect();
            let mut xs = Vec::new();
            let mut ws = Vec::new();
            let mut local = Vec::new();
            for piece in &group.pieces {
                xs.push(flat.slice(
                    &[piece.start as i32, 0][..],
                    &[piece.end as i32, self.cfg.hidden_size][..],
                )?);
                ws.push(scores.slice(&[piece.start as i32, 0][..], &[piece.end as i32, k][..])?);
                local.extend(
                    indices[piece.start * k as usize..piece.end * k as usize]
                        .iter()
                        .map(|id| mapping[id]),
                );
            }
            let count = group.pieces.iter().map(|p| p.end - p.start).sum::<usize>() as i32;
            let x = mlx::ops::shape::concatenate_on(&xs.iter().collect::<Vec<_>>(), 0, target)?;
            let score = mlx::ops::shape::concatenate_on(&ws.iter().collect::<Vec<_>>(), 0, target)?;
            let local = Array::try_from((local.as_slice(), [count, k]))?;
            let mut compact = RoutedExperts::from_compact_affine(CompactExpertWeights {
                gate_up: batch.arrays.gate_up.clone(),
                down: batch.arrays.down.clone(),
                quant: self.quant,
            })?;
            compact.keep_gate_up_split()?;
            compact.set_dispatch_expert_count(self.cfg.num_experts)?;
            compact
                .set_paged_reference_rows((group.pieces[0].end - group.pieces[0].start) as i32)?;
            let output = compact.apply_experts(&x, &local, &score, target, layer)?;
            output.eval()?;
            let mut offset = 0;
            for piece in &group.pieces {
                let len = (piece.end - piece.start) as i32;
                outputs.insert(
                    piece.start,
                    output.slice(&[offset, 0][..], &[offset + len, self.cfg.hidden_size][..])?,
                );
                offset += len;
            }
            drop(compact);
            drop(batch);
            self.store
                .lock()
                .map_err(|_| anyhow::anyhow!("expert cache lock poisoned"))?
                .finished_batch();
        }
        let routed =
            mlx::ops::shape::concatenate_on(&outputs.values().collect::<Vec<_>>(), 0, target)?;
        let mut offset = 0;
        let mut output = Vec::new();
        for (input, len) in inputs.iter().zip(lengths) {
            let row = routed
                .slice(
                    &[offset, 0][..],
                    &[offset + len as i32, self.cfg.hidden_size][..],
                )?
                .reshape_on([1, len as i32, self.cfg.hidden_size], target)?;
            offset += len as i32;
            let shared = self.shared.forward_on(input, target)?;
            let gate = self
                .shared_gate
                .forward_singleton_gather_on(input, target)?
                .sigmoid_on(target)?;
            output.push(mlx::ops::binary::add_on(
                &row,
                &mlx::ops::binary::multiply_on(&shared, &gate, target)?,
                target,
            )?);
        }
        Ok(output)
    }
}
#[derive(Clone, Debug)]
struct Piece {
    start: usize,
    end: usize,
    owner: usize,
    class: usize,
    experts: BTreeSet<u32>,
}
#[derive(Debug)]
struct ExpertGroup {
    pieces: Vec<Piece>,
    experts: BTreeSet<u32>,
    budget_split: bool,
}
fn plan_groups(
    indices: &[u32],
    lengths: &[usize],
    owners: &[usize],
    topk: usize,
    limit: usize,
    source_experts: usize,
) -> Result<Vec<ExpertGroup>> {
    ensure!(
        topk > 0
            && limit >= topk
            && source_experts > 0
            && lengths.len() == owners.len()
            && lengths.iter().sum::<usize>().checked_mul(topk) == Some(indices.len()),
        "invalid expert routing plan"
    );
    let mut per_owner: std::collections::BTreeMap<usize, std::collections::VecDeque<Piece>> =
        std::collections::BTreeMap::new();
    let mut base = 0;
    for (&len, &owner) in lengths.iter().zip(owners) {
        let mut start = base;
        while start < base + len {
            let mut experts = BTreeSet::new();
            let mut end = start;
            while end < base + len {
                let selected = &indices[end * topk..(end + 1) * topk];
                let added = selected.iter().filter(|id| !experts.contains(*id)).count();
                if experts.len() + added > limit {
                    break;
                }
                experts.extend(selected.iter().copied());
                end += 1;
            }
            ensure!(end > start, "expert working set cannot make progress");
            let routes = (end - start) * topk;
            let class = if routes < 64 {
                0
            } else if routes / source_experts < 4 {
                1
            } else {
                2
            };
            per_owner.entry(owner).or_default().push_back(Piece {
                start,
                end,
                owner,
                class,
                experts,
            });
            start = end;
        }
        base += len;
    }
    let mut pending: std::collections::BTreeMap<usize, ExpertGroup> =
        std::collections::BTreeMap::new();
    let mut groups = Vec::new();
    while per_owner.values().any(|p| !p.is_empty()) {
        for queue in per_owner.values_mut() {
            let Some(piece) = queue.pop_front() else {
                continue;
            };
            if let Some(group) = pending.get(&piece.class) {
                let added = piece
                    .experts
                    .iter()
                    .filter(|id| !group.experts.contains(*id))
                    .count();
                if group.experts.len() + added > limit {
                    let mut group = pending.remove(&piece.class).unwrap();
                    group.budget_split = true;
                    groups.push(group);
                }
            }
            let group = pending.entry(piece.class).or_insert_with(|| ExpertGroup {
                pieces: Vec::new(),
                experts: BTreeSet::new(),
                budget_split: false,
            });
            group.experts.extend(piece.experts.iter().copied());
            group.pieces.push(piece);
        }
    }
    groups.extend(pending.into_values());
    Ok(groups)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cohorts_preserve_reference_dispatch_and_working_set() {
        let indices: Vec<_> = (0..8).flat_map(|i| [i % 3, (i + 1) % 3]).collect();
        let groups = plan_groups(&indices, &[4, 4], &[0, 1], 2, 3, 512).unwrap();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].pieces.len(), 2);
        assert_eq!(groups[0].experts.len(), 3);
        let mut indices = vec![0; 63 * 10];
        indices.extend(vec![1; 300 * 10]);
        let groups = plan_groups(&indices, &[63, 300], &[0, 1], 10, 512, 512).unwrap();
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].pieces[0].class, 1);
        assert_eq!(groups[1].pieces[0].class, 2);
    }
}
