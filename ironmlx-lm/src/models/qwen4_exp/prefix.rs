//! Versioned, row-local durable Flash state. Unlike rollback snapshots this
//! codec binds validated state to the destination model, never a saved Arc.
use super::{cache::FlashLayerCache, text_model::LayerState};
use crate::core::cache::{
    prefix_payload::{PrefixLayerKind, PrefixLayerSpec, PrefixTensorSpec},
    GatedDeltaCache,
};
use anyhow::{ensure, Context, Result};
use mlx::ops::{indexing::slice_strided, shape::concatenate};
use mlx::{Array, Dtype};

const VERSION: i32 = 1;
fn spec(dtype: Dtype, shape: impl Into<Vec<i32>>) -> PrefixTensorSpec {
    PrefixTensorSpec {
        dtype,
        shape: shape.into(),
    }
}

impl FlashLayerCache {
    pub fn durable_prefix_contract(&self) -> Result<String> {
        self.prefix_contract
            .clone()
            .context("Flash prefix execution identity unavailable")
    }
    pub fn durable_prefix_spec(&self, len: i32) -> Result<PrefixLayerSpec> {
        ensure!(
            len > 0 && len <= self.cap(),
            "Flash prefix exceeds capacity"
        );
        let cfg = self
            .prefix_config
            .as_ref()
            .context("Flash durable layout unavailable")?;
        let mut tensors = vec![spec(Dtype::Int32, vec![6])];
        match &self.rows[0] {
            LayerState::Linear { gdn, ple } => {
                tensors.push(PrefixTensorSpec::from_array(gdn.conv_state()));
                tensors.push(PrefixTensorSpec::from_array(gdn.recurrent_state()));
                tensors.push(spec(
                    Dtype::Uint32,
                    vec![if ple.is_some() {
                        (cfg.ngram_size - 1) as i32
                    } else {
                        1
                    }],
                ));
                tensors.push(spec(
                    Dtype::Bfloat16,
                    if ple.is_some() {
                        vec![
                            1,
                            (cfg.ple_conv_kernel_size - 1) * cfg.ngram_size as i32,
                            cfg.hidden_size * cfg.hc_count,
                        ]
                    } else {
                        vec![1]
                    },
                ));
            }
            LayerState::Sparse(_) => {
                tensors.extend([
                    spec(
                        Dtype::Bfloat16,
                        vec![1, cfg.num_key_value_heads, len, cfg.head_dim],
                    ),
                    spec(
                        Dtype::Bfloat16,
                        vec![1, cfg.num_key_value_heads, len, cfg.head_dim],
                    ),
                    spec(Dtype::Bfloat16, vec![1, len, cfg.indexer_head_dim]),
                    spec(Dtype::Int32, vec![3, 1, len]),
                    spec(
                        Dtype::Bfloat16,
                        vec![
                            1,
                            1,
                            (len / cfg.indexer_compress_ratio).max(1),
                            cfg.indexer_head_dim,
                        ],
                    ),
                ]);
            }
        }
        Ok(PrefixLayerSpec {
            kind: PrefixLayerKind::Flash,
            tensors,
        })
    }

    pub fn durable_prefix_for_row(&self, row: usize) -> Result<(Vec<Array>, i32)> {
        let state = self
            .rows
            .get(row)
            .context("Flash prefix row out of range")?;
        let len = state.position();
        let expected = self.durable_prefix_spec(len)?;
        let zero = |i: usize| -> Result<Array> {
            Ok(Array::zeros(
                expected.tensors[i].shape.as_slice(),
                expected.tensors[i].dtype,
            )?)
        };
        let mut meta = [VERSION, i32::try_from(self.layer)?, 0, len, 0, 0];
        let mut tensors = match state {
            LayerState::Linear { gdn, ple } => {
                let (history, convolution) = if let Some(ple) = ple {
                    ensure!(
                        ple.history.position == len as u64,
                        "Flash PLE position mismatch"
                    );
                    meta[4] = 1 | if ple.convolution.is_some() { 2 } else { 0 };
                    (
                        Array::try_from((
                            ple.history.tokens(),
                            [ple.history.tokens().len() as i32],
                        ))?,
                        ple.convolution.clone().unwrap_or(zero(4)?),
                    )
                } else {
                    (zero(3)?, zero(4)?)
                };
                vec![
                    gdn.conv_state().clone(),
                    gdn.recurrent_state().clone(),
                    history,
                    convolution,
                ]
            }
            LayerState::Sparse(qsa) => {
                meta[2] = 1;
                let mut arrays = [&qsa.keys, &qsa.values, &qsa.raw_index_keys, &qsa.positions]
                    .into_iter()
                    .map(|a| a.clone().context("incomplete Flash sparse prefix"))
                    .collect::<Result<Vec<_>>>()?;
                let block = if let Some(block) = &qsa.block_keys {
                    meta[5] = block.shape().as_slice()[2];
                    let capacity = expected.tensors[5].shape[2];
                    ensure!(
                        (1..=capacity).contains(&meta[5]),
                        "invalid Flash block count"
                    );
                    if meta[5] < capacity {
                        concatenate(
                            &[
                                block,
                                &Array::zeros(
                                    [1, 1, capacity - meta[5], expected.tensors[5].shape[3]],
                                    block.dtype(),
                                )?,
                            ],
                            2,
                        )?
                    } else {
                        block.clone()
                    }
                } else {
                    zero(5)?
                };
                arrays.push(block);
                arrays
            }
        };
        tensors.insert(0, Array::try_from((meta.as_slice(), [6]))?);
        self.validate_durable_prefix(&tensors, row, len)?;
        Ok((tensors, len))
    }

    /// All metadata and shapes are checked before any destination row changes.
    pub fn validate_durable_prefix(&self, tensors: &[Array], row: usize, len: i32) -> Result<()> {
        let template = self
            .rows
            .get(row)
            .context("Flash prefix row out of range")?;
        let expected = self.durable_prefix_spec(len)?;
        ensure!(
            tensors.len() == expected.tensors.len(),
            "Flash prefix tensor count mismatch"
        );
        for (tensor, spec) in tensors.iter().zip(&expected.tensors) {
            ensure!(
                tensor.dtype() == spec.dtype && tensor.shape().as_slice() == spec.shape,
                "Flash prefix tensor layout mismatch"
            );
        }
        let meta = tensors[0].to_vec::<i32>()?;
        ensure!(
            meta[0] == VERSION && meta[1] == i32::try_from(self.layer)? && meta[3] == len,
            "Flash prefix format/layer/position mismatch"
        );
        match template {
            LayerState::Linear { ple, .. } => {
                ensure!(
                    meta[2] == 0
                        && meta[5] == 0
                        && meta[4] & !3 == 0
                        && (meta[4] & 1 != 0) == ple.is_some()
                        && (meta[4] & 2 == 0 || ple.is_some()),
                    "Flash prefix linear flags mismatch"
                );
                if let Some(ple) = ple {
                    let mut history = ple.history.clone();
                    history.restore(
                        tensors[3].to_vec::<u32>()?,
                        len as u64,
                        self.prefix_config.as_ref().unwrap().vocab_size,
                    )?;
                }
            }
            LayerState::Sparse(_) => {
                let ratio = self.prefix_config.as_ref().unwrap().indexer_compress_ratio;
                ensure!(
                    meta[2] == 1 && meta[4] == 0 && (0..=len / ratio).contains(&meta[5]),
                    "Flash prefix sparse metadata mismatch"
                );
                ensure!(
                    tensors[4].to_vec::<i32>()?.iter().all(|&p| p >= 0),
                    "negative Flash MRoPE position"
                );
            }
        }
        Ok(())
    }

    pub fn restore_durable_prefix(
        &mut self,
        tensors: &[Array],
        row: usize,
        len: i32,
    ) -> Result<()> {
        self.validate_durable_prefix(tensors, row, len)?;
        let meta = tensors[0].to_vec::<i32>()?;
        let restored = match &self.rows[row] {
            LayerState::Linear { gdn, ple } => {
                let mut gdn = GatedDeltaCache::from_snapshot(&gdn.snapshot(), self.cap())?;
                gdn.restore_prefix_state_for_row_on(&tensors[1], &tensors[2], 0, len, ())?;
                let mut ple = ple.clone();
                if let Some(ple) = &mut ple {
                    ple.history.restore(
                        tensors[3].to_vec::<u32>()?,
                        len as u64,
                        self.prefix_config.as_ref().unwrap().vocab_size,
                    )?;
                    ple.convolution = (meta[4] & 2 != 0).then(|| tensors[4].clone());
                }
                LayerState::Linear {
                    gdn: Box::new(gdn),
                    ple,
                }
            }
            LayerState::Sparse(qsa) => {
                let mut qsa = qsa.clone();
                qsa.keys = Some(tensors[1].clone());
                qsa.values = Some(tensors[2].clone());
                qsa.raw_index_keys = Some(tensors[3].clone());
                qsa.positions = Some(tensors[4].clone());
                qsa.block_keys = if meta[5] > 0 {
                    Some(slice_strided(
                        &tensors[5],
                        [0, 0, 0, 0],
                        [1, 1, meta[5], tensors[5].shape().as_slice()[3]],
                        [1, 1, 1, 1],
                    )?)
                } else {
                    None
                };
                qsa.offset = len;
                LayerState::Sparse(qsa)
            }
        };
        self.rows[row] = restored;
        self.prefixes.clear();
        self.refresh_offsets();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::qwen4_exp::{attention::QsaState, config::FlashTextConfig, ple::PleState};
    use std::sync::Arc;
    fn cfg() -> FlashTextConfig {
        FlashTextConfig::from_root(
            &serde_json::from_str(include_str!(
                "../../../tests/fixtures/qwen38_flash_next/config.json"
            ))
            .unwrap(),
        )
        .unwrap()
    }
    fn linear() -> FlashLayerCache {
        let cfg = cfg();
        let mut cache = FlashLayerCache::new(
            vec![LayerState::Linear {
                gdn: Box::new(
                    GatedDeltaCache::new_with_cap(1, 4, 8, 1, 2, 32, Dtype::Bfloat16, 64).unwrap(),
                ),
                ple: Some(PleState::from_history(cfg.ple_index(0).unwrap().history())),
            }],
            Arc::new(()),
            0,
            64,
        )
        .unwrap();
        cache.prefix_config = Some(Arc::new(cfg));
        cache
    }
    #[test]
    #[serial_test::serial(mlx_metal)]
    fn durable_linear_preserves_eos_history_state_and_reset_template() {
        let mut source = linear();
        let cfg = source.prefix_config.clone().unwrap();
        let LayerState::Linear {
            gdn,
            ple: Some(ple),
        } = &mut source.rows[0]
        else {
            unreachable!()
        };
        let tokens = [17, cfg.eos_token_id, 21];
        gdn.advance(&[3]).unwrap();
        cfg.ple_index(0)
            .unwrap()
            .indices(&tokens, &mut ple.history)
            .unwrap();
        ple.convolution = Some(
            Array::zeros(
                [
                    1,
                    (cfg.ple_conv_kernel_size - 1) * cfg.ngram_size as i32,
                    cfg.hidden_size * cfg.hc_count,
                ],
                Dtype::Bfloat16,
            )
            .unwrap(),
        );
        source.refresh_offsets();
        let (payload, len) = source.durable_prefix_for_row(0).unwrap();
        let mut destination = linear();
        assert!(!Arc::ptr_eq(&source.instance, &destination.instance));
        destination
            .restore_durable_prefix(&payload, 0, len)
            .unwrap();
        assert_eq!(destination.offsets(), [3]);
        let LayerState::Linear {
            ple: Some(restored),
            ..
        } = &destination.rows[0]
        else {
            unreachable!()
        };
        assert_eq!(restored.history.tokens(), [cfg.eos_token_id, 21]);
        assert_eq!(restored.history.position, 3);
        assert!(restored.convolution.is_some());
        destination.reset().unwrap();
        let LayerState::Linear {
            ple: Some(restored),
            ..
        } = &destination.rows[0]
        else {
            unreachable!()
        };
        assert_eq!(restored.history.position, 0);
        assert_eq!(restored.history.tokens(), [cfg.eos_token_id; 2]);
        assert!(restored.convolution.is_none());
    }
    #[test]
    #[serial_test::serial(mlx_metal)]
    fn malformed_history_or_metadata_never_mutates_destination() {
        let mut source = linear();
        let LayerState::Linear {
            gdn,
            ple: Some(ple),
        } = &mut source.rows[0]
        else {
            unreachable!()
        };
        gdn.advance(&[3]).unwrap();
        ple.history.position = 3;
        let (mut payload, _) = source.durable_prefix_for_row(0).unwrap();
        let mut destination = linear();
        payload[3] = Array::try_from(([u32::MAX; 2].as_slice(), [2])).unwrap();
        assert!(destination.restore_durable_prefix(&payload, 0, 3).is_err());
        assert_eq!(destination.offsets(), [0]);
        payload[0] = Array::try_from(([2, 0, 0, 3, 1, 0].as_slice(), [6])).unwrap();
        assert!(destination.restore_durable_prefix(&payload, 0, 3).is_err());
        assert_eq!(destination.offsets(), [0]);
    }
    #[test]
    #[serial_test::serial(mlx_metal)]
    fn sparse_roundtrip_preserves_missing_or_completed_blocks_and_isolates_rows() {
        let cfg = cfg();
        let len = cfg.indexer_compress_ratio * 2;
        let mut cache = FlashLayerCache::new(
            vec![
                LayerState::Sparse(QsaState::new(len + 8).unwrap()),
                LayerState::Sparse(QsaState::new(len + 8).unwrap()),
            ],
            Arc::new(()),
            3,
            len + 8,
        )
        .unwrap();
        cache.prefix_config = Some(Arc::new(cfg.clone()));
        let spec = cache.durable_prefix_spec(len).unwrap();
        let mut payload = spec
            .tensors
            .iter()
            .map(|s| Array::zeros(s.shape.as_slice(), s.dtype).unwrap())
            .collect::<Vec<_>>();
        payload[0] = Array::try_from(([1, 3, 1, len, 0, 1].as_slice(), [6])).unwrap();
        cache.restore_durable_prefix(&payload, 1, len).unwrap();
        assert_eq!(cache.offsets(), [0, len]);
        let (encoded, _) = cache.durable_prefix_for_row(1).unwrap();
        assert_eq!(encoded[0].to_vec::<i32>().unwrap()[5], 1);
        payload[0] = Array::try_from(([1, 3, 1, len, 0, 0].as_slice(), [6])).unwrap();
        cache.restore_durable_prefix(&payload, 0, len).unwrap();
        let LayerState::Sparse(qsa) = &cache.rows[0] else {
            unreachable!()
        };
        assert!(qsa.block_keys.is_none());
        assert_eq!(
            cache.durable_prefix_for_row(0).unwrap().0[0]
                .to_vec::<i32>()
                .unwrap()[5],
            0
        );
    }
}
