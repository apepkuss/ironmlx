use anyhow::anyhow;
use mlx::{Array, StreamOrDevice};

use crate::core::Loader;
use crate::nn::Linear;
use crate::Result;

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
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

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
