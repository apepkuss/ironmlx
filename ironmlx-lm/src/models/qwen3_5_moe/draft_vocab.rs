//! Draft vocabulary of the qualified Qwen3.6 MoE affine4 target.
//!
//! DFlash2 drafting projects its proposal hidden states through the target
//! LM head, which reads all 248,320 quantized rows for every window. Drafts
//! only propose tokens and verification decides every output token, so the
//! drafter projects onto a fixed subset instead: the 32,768 tokens most frequent
//! in the target's own greedy outputs for public prompt sets (general
//! instructions in English and Chinese, math, code), plus the tokens of the
//! current request that the subset lacks (its prompt and generated tokens, see
//! `DFlash2DraftCache::note_tokens`). Any other token can still be generated;
//! it is only never proposed.

use anyhow::{anyhow, ensure};
use mlx::{Array, StreamOrDevice};

use crate::nn::Linear;
use crate::Result;

/// Token ids, strictly ascending, little-endian `u32`. Built by
/// `reports/qwen36-moe-dflash2/arithmetic-optimization/draft-vocab/build_table.py`.
const TABLE: &[u8] = include_bytes!("qwen36_draft_vocab.u32le");
const TARGET_VOCAB: usize = 248_320;

pub(super) struct DraftVocab {
    table: Vec<u32>,
    ids: Array,
    head: Linear,
}

impl DraftVocab {
    /// The subset rows of `lm_head`, or `None` when the head is not the
    /// quantized 248,320-row head the table was built for.
    pub(super) fn from_lm_head(lm_head: &Linear) -> Result<Option<Self>> {
        let Some(parts) = lm_head.quantized_parts() else {
            return Ok(None);
        };
        if lm_head.out_features() != TARGET_VOCAB || parts.bias.is_some() {
            return Ok(None);
        }
        ensure!(
            TABLE.len().is_multiple_of(4),
            "draft vocabulary table is truncated"
        );
        let ids: Vec<u32> = TABLE
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        ensure!(
            ids.windows(2).all(|w| w[0] < w[1])
                && ids.last().is_some_and(|&id| (id as usize) < TARGET_VOCAB),
            "draft vocabulary table is not a strictly ascending id list"
        );
        let table = ids;
        let n = i32::try_from(table.len())?;
        let ids: Array = (&table[..], &[n][..]).try_into()?;
        let target = StreamOrDevice::default();
        let take = |a: &Array| a.take_on(&ids, 0, target);
        let weight = take(parts.weight)?;
        let scales = take(parts.scales)?;
        let biases = parts.biases.map(take).transpose()?;
        let mut evaluated = vec![&ids, &weight, &scales];
        evaluated.extend(biases.as_ref());
        mlx::transforms::eval(&evaluated)?;
        let head = Linear::new_quant_with_mode(
            weight,
            scales,
            biases,
            None,
            parts.group_size,
            parts.bits,
            parts.mode,
        );
        if head.out_features() != ids.size() {
            return Err(anyhow!("draft vocabulary head has the wrong row count"));
        }
        Ok(Some(Self { table, ids, head }))
    }

    pub(super) fn contains(&self, token: u32) -> bool {
        self.table.binary_search(&token).is_ok()
    }

    /// `(logits [.., N + E], ids [N + E])` for `hidden [.., H]`: the subset,
    /// then the `extra` request tokens (rows of the full `lm_head`).
    pub(super) fn project_on(
        &self,
        lm_head: &Linear,
        hidden: &Array,
        extra: &[u32],
        target: StreamOrDevice,
    ) -> Result<(Array, Array)> {
        let logits = self.head.forward_on(hidden, target)?;
        let Some(parts) = lm_head.quantized_parts().filter(|_| !extra.is_empty()) else {
            return Ok((logits, self.ids.clone()));
        };
        let extra_ids: Array = (extra, &[i32::try_from(extra.len())?][..]).try_into()?;
        let take = |a: &Array| a.take_on(&extra_ids, 0, target);
        let extra_head = Linear::new_quant_with_mode(
            take(parts.weight)?,
            take(parts.scales)?,
            parts.biases.map(take).transpose()?,
            None,
            parts.group_size,
            parts.bits,
            parts.mode,
        );
        let extra_logits = extra_head.forward_on(hidden, target)?;
        Ok((
            mlx::ops::shape::concatenate_on(&[&logits, &extra_logits], -1, target)?,
            mlx::ops::shape::concatenate_on(&[&self.ids, &extra_ids], 0, target)?,
        ))
    }
}
