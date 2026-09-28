use anyhow::anyhow;
use mlx::{Array, StreamOrDevice};

use crate::core::Loader;
use crate::nn::Linear;
use crate::Result;

use super::DFlash2DraftTree;
use super::{config::DFlash2Config, load_linear};

pub(super) struct DFlash2CandidateSelector {
    predecessor_codebook: Array,
    successor_codebook: Array,
    hidden_projection: Linear,
    top_k: i32,
    vocab_size: i32,
}

impl DFlash2CandidateSelector {
    pub(super) fn from_loader(
        loader: &Loader,
        cfg: &DFlash2Config,
        draft_bits: Option<i32>,
    ) -> Result<Self> {
        Ok(Self {
            predecessor_codebook: loader
                .tensor("candidate_selector.predecessor_codebook")?
                .clone(),
            successor_codebook: loader
                .tensor("candidate_selector.successor_codebook")?
                .clone(),
            hidden_projection: load_linear(
                loader,
                "candidate_selector.hidden_projection",
                draft_bits,
            )?,
            top_k: cfg.dflash_config.selector_top_k,
            vocab_size: cfg.vocab_size,
        })
    }

    #[cfg(test)]
    fn from_components(
        predecessor_codebook: Array,
        successor_codebook: Array,
        hidden_projection: Linear,
        top_k: i32,
        vocab_size: i32,
    ) -> Self {
        Self {
            predecessor_codebook,
            successor_codebook,
            hidden_projection,
            top_k,
            vocab_size,
        }
    }

    pub(super) fn select_greedy_on(
        &self,
        hidden: &Array,
        logits: &Array,
        anchor_ids: &Array,
        target: StreamOrDevice,
    ) -> Result<Array> {
        let shape = logits.shape();
        let dims = shape.as_slice();
        if dims.len() != 3 || dims[0] <= 0 || dims[2] != self.vocab_size {
            return Err(anyhow!(
                "DFlash2 selector expected logits [B,L,{}] with B>0, got {dims:?}",
                self.vocab_size
            ));
        }
        let batch = dims[0];
        let length = dims[1];
        let partition = mlx::ops::sort::argpartition_on(logits, -self.top_k, -1, target)?;
        let candidates = mlx::ops::indexing::slice_strided_on(
            &partition,
            &[0_i32, 0, self.vocab_size - self.top_k][..],
            &[batch, length, self.vocab_size][..],
            &[1_i32, 1, 1][..],
            target,
        )?;
        let unary = mlx::ops::indexing::take_along_axis_on(logits, &candidates, -1, target)?;
        // Keep selector edge scores independent of the number of active rows.
        // This projection is small relative to the target LM head, while its
        // rounding can change the selected draft path and acceptance rate.
        let hidden = {
            let _product_stable_qmm = crate::nn::product_stable_qmm::scope();
            self.hidden_projection.forward_on(hidden, target)?
        };
        let mut predecessor = anchor_ids.reshape_on((batch,), target)?;
        let mut path = Vec::with_capacity(length as usize);
        for position in 0..length {
            let candidate_row = mlx::ops::indexing::slice_strided_on(
                &candidates,
                &[0_i32, position, 0][..],
                &[batch, position + 1, self.top_k][..],
                &[1_i32, 1, 1][..],
                target,
            )?
            .reshape_on((batch, self.top_k), target)?;
            let unary_row = mlx::ops::indexing::slice_strided_on(
                &unary,
                &[0_i32, position, 0][..],
                &[batch, position + 1, self.top_k][..],
                &[1_i32, 1, 1][..],
                target,
            )?
            .reshape_on((batch, self.top_k), target)?;
            let hidden_row = mlx::ops::indexing::slice_strided_on(
                &hidden,
                &[0_i32, position, 0][..],
                &[batch, position + 1, hidden.shape().as_slice()[2]][..],
                &[1_i32, 1, 1][..],
                target,
            )?;
            let predecessor_code = self
                .predecessor_codebook
                .take_on(&predecessor, 0, target)?
                .reshape_on((batch, 1, hidden.shape().as_slice()[2]), target)?;
            let successor_code = self.successor_codebook.take_on(&candidate_row, 0, target)?;
            let hidden_row =
                hidden_row.reshape_on((batch, 1, hidden.shape().as_slice()[2]), target)?;
            let edges = mlx::ops::reduction::sum_on(
                &(&(&predecessor_code * &hidden_row) * &successor_code),
                -1_i32,
                false,
                target,
            )?;
            let scores = &unary_row + &edges;
            let selected = mlx::ops::reduction::argmax_on(&scores, -1_i32, false, target)?
                .reshape_on((batch, 1), target)?;
            predecessor =
                mlx::ops::indexing::take_along_axis_on(&candidate_row, &selected, -1, target)?
                    .reshape_on((batch,), target)?;
            path.push(predecessor.clone());
        }
        let refs = path.iter().collect::<Vec<_>>();
        mlx::ops::shape::stack_on(&refs, 1, target).map_err(Into::into)
    }

    /// Build a bounded best-first tree from one proposal lattice. Candidate
    /// extraction and codebook gathers stay on the device; only the compact
    /// `[depth, top_k, rank]` lattice is materialized for the fifteen-node
    /// host priority walk.
    pub(super) fn select_tree_on(
        &self,
        hidden: &Array,
        logits: &Array,
        anchor_ids: &Array,
        max_nodes: usize,
        children_per_node: usize,
        target: StreamOrDevice,
    ) -> Result<DFlash2DraftTree> {
        anyhow::ensure!(
            (1..=DFlash2DraftTree::MAX_NODES).contains(&max_nodes),
            "DFlash2 tree max_nodes must be in [1, {}]",
            DFlash2DraftTree::MAX_NODES
        );
        anyhow::ensure!(
            children_per_node > 0,
            "DFlash2 tree children_per_node must be positive"
        );
        let shape = logits.shape();
        let dims = shape.as_slice();
        anyhow::ensure!(
            dims.len() == 3 && dims[0] == 1 && dims[1] > 0 && dims[2] == self.vocab_size,
            "DFlash2 tree selector expected logits [1,L,{}], got {dims:?}",
            self.vocab_size
        );
        let depth = dims[1] as usize;
        let rank = self.hidden_projection.out_features();
        let partition = mlx::ops::sort::argpartition_on(logits, -self.top_k, -1, target)?;
        let candidates = mlx::ops::indexing::slice_strided_on(
            &partition,
            &[0_i32, 0, self.vocab_size - self.top_k][..],
            &[1_i32, dims[1], self.vocab_size][..],
            &[1_i32, 1, 1][..],
            target,
        )?;
        let unary = mlx::ops::indexing::take_along_axis_on(logits, &candidates, -1, target)?;
        let projected = {
            let _product_stable_qmm = crate::nn::product_stable_qmm::scope();
            self.hidden_projection.forward_on(hidden, target)?
        };
        let flat_candidates = candidates.reshape_on((-1_i32,), target)?;
        let predecessor = self
            .predecessor_codebook
            .take_on(&flat_candidates, 0, target)?;
        let successor = self
            .successor_codebook
            .take_on(&flat_candidates, 0, target)?;
        let anchor = anchor_ids.reshape_on((-1_i32,), target)?;
        let anchor_predecessor = self.predecessor_codebook.take_on(&anchor, 0, target)?;
        let arrays = [
            &candidates,
            &unary,
            &projected,
            &predecessor,
            &successor,
            &anchor_predecessor,
        ];
        mlx::transforms::eval(&arrays)?;

        let lattice = HostDraftLattice {
            depth,
            width: self.top_k as usize,
            rank,
            candidates: candidates.to_vec::<u32>()?,
            unary: mlx::ops::cast::astype(&unary, mlx::Dtype::Float32)?.to_vec::<f32>()?,
            projected: mlx::ops::cast::astype(&projected, mlx::Dtype::Float32)?.to_vec::<f32>()?,
            predecessor: mlx::ops::cast::astype(&predecessor, mlx::Dtype::Float32)?
                .to_vec::<f32>()?,
            successor: mlx::ops::cast::astype(&successor, mlx::Dtype::Float32)?.to_vec::<f32>()?,
            anchor_predecessor: mlx::ops::cast::astype(&anchor_predecessor, mlx::Dtype::Float32)?
                .to_vec::<f32>()?,
        };
        lattice.best_first_tree(max_nodes, children_per_node.min(lattice.width))
    }
}

struct HostDraftLattice {
    depth: usize,
    width: usize,
    rank: usize,
    candidates: Vec<u32>,
    unary: Vec<f32>,
    projected: Vec<f32>,
    predecessor: Vec<f32>,
    successor: Vec<f32>,
    anchor_predecessor: Vec<f32>,
}

#[derive(Clone, Copy)]
struct FrontierNode {
    score: f64,
    parent: i32,
    token: u32,
    depth: usize,
    candidate: usize,
}

impl HostDraftLattice {
    fn vector<'a>(&self, values: &'a [f32], depth: usize, candidate: usize) -> &'a [f32] {
        let start = (depth * self.width + candidate) * self.rank;
        &values[start..start + self.rank]
    }

    fn child_scores(&self, predecessor: &[f32], depth: usize) -> Vec<f64> {
        let hidden = &self.projected[depth * self.rank..(depth + 1) * self.rank];
        let mut scores = (0..self.width)
            .map(|candidate| {
                let successor = self.vector(&self.successor, depth, candidate);
                let edge = predecessor
                    .iter()
                    .zip(hidden)
                    .zip(successor)
                    .map(|((&p, &h), &s)| f64::from(p) * f64::from(h) * f64::from(s))
                    .sum::<f64>();
                f64::from(self.unary[depth * self.width + candidate]) + edge
            })
            .collect::<Vec<_>>();
        let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let normalizer = scores
            .iter()
            .map(|score| (*score - max).exp())
            .sum::<f64>()
            .ln()
            + max;
        for score in &mut scores {
            *score -= normalizer;
        }
        scores
    }

    fn push_children(
        &self,
        frontier: &mut Vec<FrontierNode>,
        predecessor: &[f32],
        depth: usize,
        parent: i32,
        parent_score: f64,
        children: usize,
    ) {
        let scores = self.child_scores(predecessor, depth);
        let mut order = (0..self.width).collect::<Vec<_>>();
        order.sort_by(|&left, &right| {
            scores[right].total_cmp(&scores[left]).then_with(|| {
                self.candidates[depth * self.width + left]
                    .cmp(&self.candidates[depth * self.width + right])
            })
        });
        for candidate in order.into_iter().take(children) {
            frontier.push(FrontierNode {
                score: parent_score + scores[candidate],
                parent,
                token: self.candidates[depth * self.width + candidate],
                depth,
                candidate,
            });
        }
    }

    fn best_first_tree(&self, max_nodes: usize, children: usize) -> Result<DFlash2DraftTree> {
        anyhow::ensure!(
            self.depth > 0 && self.width > 0 && self.rank > 0,
            "empty DFlash2 lattice"
        );
        let mut frontier = Vec::new();
        self.push_children(
            &mut frontier,
            &self.anchor_predecessor,
            0,
            -1,
            0.0,
            children,
        );
        let mut tokens = Vec::with_capacity(max_nodes);
        let mut parents = Vec::with_capacity(max_nodes);
        while !frontier.is_empty() && tokens.len() < max_nodes {
            let next = frontier
                .iter()
                .enumerate()
                .max_by(|(_, left), (_, right)| {
                    left.score
                        .total_cmp(&right.score)
                        .then_with(|| right.token.cmp(&left.token))
                })
                .map(|(index, _)| index)
                .expect("non-empty frontier");
            let node = frontier.swap_remove(next);
            let node_index = i32::try_from(tokens.len())?;
            tokens.push(node.token);
            parents.push(node.parent);
            if node.depth + 1 < self.depth {
                let predecessor = self.vector(&self.predecessor, node.depth, node.candidate);
                self.push_children(
                    &mut frontier,
                    predecessor,
                    node.depth + 1,
                    node_index,
                    node.score,
                    children,
                );
            }
        }
        DFlash2DraftTree::new(tokens, parents)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    #[test]
    fn best_first_tree_is_bounded_deterministic_and_topological() {
        let lattice = HostDraftLattice {
            depth: 3,
            width: 2,
            rank: 1,
            candidates: vec![10, 11, 20, 21, 30, 31],
            unary: vec![2.0, 1.0, 2.0, 1.0, 2.0, 1.0],
            projected: vec![1.0; 3],
            predecessor: vec![1.0; 6],
            successor: vec![0.0; 6],
            anchor_predecessor: vec![1.0],
        };
        let first = lattice.best_first_tree(7, 2).expect("first tree");
        let second = lattice.best_first_tree(7, 2).expect("second tree");
        assert_eq!(first, second);
        assert_eq!(first.tokens.len(), 7);
        assert!(first
            .parents
            .iter()
            .enumerate()
            .all(|(node, parent)| *parent == -1 || (*parent as usize) < node));
        assert!(first.leaf_paths().len() <= 4);
        assert!(first.depths().into_iter().all(|depth| depth <= 3));
    }

    #[test]
    #[serial(mlx_metal)]
    fn selector_walk_uses_predecessor_dependent_edges() {
        let predecessor: Array = (
            &[1.0_f32, 0.0, 0.0, 1.0, 1.0, 1.0, -1.0, 0.0][..],
            &[4_i32, 2][..],
        )
            .try_into()
            .expect("predecessor");
        let successor: Array = (
            &[0.0_f32, 0.0, 2.0, 0.0, 0.0, 2.0, -2.0, 0.0][..],
            &[4_i32, 2][..],
        )
            .try_into()
            .expect("successor");
        let selector = DFlash2CandidateSelector::from_components(
            predecessor,
            successor,
            Linear::new_fp(
                (&[1.0_f32, 0.0, 0.0, 1.0][..], &[2_i32, 2][..])
                    .try_into()
                    .expect("identity"),
                None,
            ),
            2,
            4,
        );
        let hidden: Array = (&[1.0_f32, 0.0, 0.0, 1.0][..], &[1_i32, 2, 2][..])
            .try_into()
            .expect("hidden");
        let logits: Array = (
            &[0.0_f32, 1.0, 0.9, -1.0, 0.0, -1.0, 1.0, 0.9][..],
            &[1_i32, 2, 4][..],
        )
            .try_into()
            .expect("logits");
        let anchor: Array = (&[0_u32][..], &[1_i32][..]).try_into().expect("anchor");
        let selected = selector
            .select_greedy_on(&hidden, &logits, &anchor, StreamOrDevice::default())
            .expect("select")
            .to_vec::<u32>()
            .expect("tokens");
        assert_eq!(selected, vec![1, 2]);
    }

    #[test]
    #[serial(mlx_metal)]
    fn selector_walk_keeps_batch_rows_isolated() {
        let predecessor: Array = (
            &[1.0_f32, 0.0, 0.0, 1.0, 1.0, 1.0, -1.0, 0.0][..],
            &[4_i32, 2][..],
        )
            .try_into()
            .expect("predecessor");
        let successor: Array = (
            &[0.0_f32, 0.0, 2.0, 0.0, 0.0, 2.0, -2.0, 0.0][..],
            &[4_i32, 2][..],
        )
            .try_into()
            .expect("successor");
        let selector = DFlash2CandidateSelector::from_components(
            predecessor,
            successor,
            Linear::new_fp(
                (&[1.0_f32, 0.0, 0.0, 1.0][..], &[2_i32, 2][..])
                    .try_into()
                    .expect("identity"),
                None,
            ),
            2,
            4,
        );
        let hidden: Array = (
            &[1.0_f32, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0][..],
            &[2_i32, 2, 2][..],
        )
            .try_into()
            .expect("hidden");
        let logits: Array = (
            &[
                0.0_f32, 1.0, 0.9, -1.0, 0.0, -1.0, 1.0, 0.9, 0.0, 1.0, 0.9, -1.0, 0.0, -1.0, 1.0,
                0.9,
            ][..],
            &[2_i32, 2, 4][..],
        )
            .try_into()
            .expect("logits");
        let anchor: Array = (&[0_u32, 0][..], &[2_i32][..]).try_into().expect("anchor");
        let selected = selector
            .select_greedy_on(&hidden, &logits, &anchor, StreamOrDevice::default())
            .expect("select")
            .to_vec::<u32>()
            .expect("tokens");
        assert_eq!(selected, vec![1, 2, 1, 2]);
    }
}
