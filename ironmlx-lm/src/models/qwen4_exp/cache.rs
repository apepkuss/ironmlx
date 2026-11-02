//! Composite request state for Flash's recurrent, PLE and QSA branches.
//! Row adoption/snapshots carry every auxiliary state, including EOS-reset
//! n-gram history. No cache row may cross loaded model or layer identities.
use super::text_model::{LayerSnapshot, LayerState};
use crate::core::cache::GatedDeltaCache;
use anyhow::{ensure, Result};
use mlx::{Array, Dtype, StreamOrDevice};
use std::sync::Arc;

impl LayerState {
    pub(crate) fn snapshot(&self) -> LayerSnapshot {
        match self {
            Self::Linear { gdn, ple } => LayerSnapshot::Linear(gdn.snapshot(), ple.clone()),
            Self::Sparse(state) => LayerSnapshot::Sparse(state.clone()),
        }
    }
    fn from_snapshot(snapshot: &LayerSnapshot, cap: i32) -> Result<Self> {
        Ok(match snapshot {
            LayerSnapshot::Linear(gdn, ple) => Self::Linear {
                gdn: GatedDeltaCache::from_snapshot(gdn, cap)?,
                ple: ple.clone(),
            },
            LayerSnapshot::Sparse(state) => {
                let mut state = state.clone();
                state.set_cap(cap)?;
                Self::Sparse(state)
            }
        })
    }
    pub(crate) fn position(&self) -> i32 {
        match self {
            Self::Linear { gdn, .. } => gdn.offsets()[0],
            Self::Sparse(state) => state.offset,
        }
    }
    fn arrays(&self) -> Vec<&Array> {
        match self {
            Self::Linear { gdn, ple } => {
                let mut arrays = vec![gdn.conv_state(), gdn.recurrent_state()];
                if let Some(conv) = ple.as_ref().and_then(|ple| ple.convolution.as_ref()) {
                    arrays.push(conv);
                }
                arrays
            }
            Self::Sparse(state) => [
                &state.keys,
                &state.values,
                &state.raw_index_keys,
                &state.block_keys,
                &state.positions,
            ]
            .into_iter()
            .filter_map(Option::as_ref)
            .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::qwen4_exp::{
        attention::QsaState,
        ple::PleState,
        ple_index::{PleIndex, PleIndexSpec},
    };
    fn index() -> PleIndex {
        PleIndex::new(PleIndexSpec {
            vocab_size: 100,
            eos_token_id: 99,
            ngram_size: 3,
            heads_per_ngram: 1,
            vocab_size_base: 101,
            ple_layer_index: 0,
            seed: 1234,
            alignment: 8,
            shard_count: 2,
        })
        .unwrap()
    }
    fn linear() -> LayerState {
        LayerState::Linear {
            gdn: GatedDeltaCache::new_with_cap(1, 4, 8, 1, 2, 32, Dtype::Bfloat16, 16).unwrap(),
            ple: Some(PleState::from_history(index().history())),
        }
    }
    fn advance(state: &mut LayerState, tokens: &[u32]) {
        let LayerState::Linear {
            gdn,
            ple: Some(ple),
        } = state
        else {
            panic!("linear fixture")
        };
        gdn.advance(&[tokens.len() as i32]).unwrap();
        index().indices(tokens, &mut ple.history).unwrap();
        ple.convolution = Some(mlx::ops::constructors::ones([1, 2, 8], Dtype::Bfloat16).unwrap());
    }
    #[test]
    #[serial_test::serial(mlx_metal)]
    fn adoption_restore_and_reset_include_ple_history_and_reject_foreign_state() {
        let instance = Arc::new(());
        let mut source =
            FlashLayerCache::new(vec![linear()], Arc::clone(&instance), 1, 16).unwrap();
        advance(&mut source.rows[0], &[12, 99, 14]);
        source.refresh_offsets();
        let mut destination =
            FlashLayerCache::new(vec![linear(), linear()], Arc::clone(&instance), 1, 16).unwrap();
        destination.adopt_row_from(&source, 1, 0).unwrap();
        assert_eq!(destination.offsets(), [0, 3]);
        let LayerState::Linear { ple: Some(ple), .. } = &destination.rows[1] else {
            panic!("PLE fixture")
        };
        assert_eq!(ple.history.tokens(), [99, 14]);
        assert!(ple.convolution.is_some());
        let base = destination.snapshot();
        destination.reset().unwrap();
        assert_eq!(destination.offsets(), [0, 0]);
        let LayerState::Linear { ple: Some(ple), .. } = &destination.rows[1] else {
            panic!("PLE fixture")
        };
        assert_eq!(ple.history.tokens(), [99, 99]);
        assert!(ple.convolution.is_none());
        destination.restore(&base).unwrap();
        assert_eq!(destination.offsets(), [0, 3]);
        let foreign = FlashLayerCache::new(vec![linear()], Arc::new(()), 1, 16).unwrap();
        assert!(destination.adopt_row_from(&foreign, 0, 0).is_err());
        let foreign_layer = FlashLayerCache::new(vec![linear()], instance, 2, 16).unwrap();
        assert!(destination.adopt_row_from(&foreign_layer, 0, 0).is_err());
        assert_eq!(destination.offsets(), [0, 3]);
    }
    #[test]
    #[serial_test::serial(mlx_metal)]
    fn ragged_prefix_commits_validate_all_rows_before_mutation() {
        let mut cache =
            FlashLayerCache::new(vec![linear(), linear()], Arc::new(()), 1, 16).unwrap();
        let base = cache.snapshot();
        cache.begin_speculative_prefix_capture().unwrap();
        for token in [10, 99, 12, 13] {
            for row in 0..2 {
                advance(&mut cache.rows[row], &[token]);
                cache.prefixes[row].push(cache.rows[row].snapshot());
            }
        }
        cache.refresh_offsets();
        assert!(cache.commit_prefix_rows(&base, &[2, 5], ().into()).is_err());
        assert_eq!(cache.offsets(), [4, 4]);
        cache.commit_prefix_rows(&base, &[2, 0], ().into()).unwrap();
        assert_eq!(cache.offsets(), [2, 0]);
        let LayerState::Linear { ple: Some(ple), .. } = &cache.rows[0] else {
            panic!("PLE fixture")
        };
        assert_eq!(ple.history.tokens(), [10, 99]);
        assert_eq!(ple.history.position, 2);
        let LayerState::Linear { ple: Some(ple), .. } = &cache.rows[1] else {
            panic!("PLE fixture")
        };
        assert_eq!(ple.history.tokens(), [99, 99]);
        assert_eq!(ple.history.position, 0);
    }
    #[test]
    #[serial_test::serial(mlx_metal)]
    fn qsa_commit_trims_raw_history_and_invalidates_completed_block_summaries() {
        let mut cache = FlashLayerCache::new(
            vec![LayerState::Sparse(QsaState::new(16).unwrap())],
            Arc::new(()),
            3,
            16,
        )
        .unwrap();
        let base = cache.snapshot();
        let LayerState::Sparse(state) = &mut cache.rows[0] else {
            unreachable!()
        };
        state.keys = Some(mlx::ops::constructors::ones([1, 2, 4, 32], Dtype::Bfloat16).unwrap());
        state.values = state.keys.clone();
        state.raw_index_keys =
            Some(mlx::ops::constructors::ones([1, 4, 32], Dtype::Bfloat16).unwrap());
        state.block_keys =
            Some(mlx::ops::constructors::ones([1, 1, 1, 32], Dtype::Bfloat16).unwrap());
        state.offset = 4;
        cache.refresh_offsets();
        cache.commit_prefix_rows(&base, &[2], ().into()).unwrap();
        let LayerState::Sparse(state) = &cache.rows[0] else {
            unreachable!()
        };
        assert_eq!(
            state.keys.as_ref().unwrap().shape().as_slice(),
            [1, 2, 2, 32]
        );
        assert_eq!(
            state.raw_index_keys.as_ref().unwrap().shape().as_slice(),
            [1, 2, 32]
        );
        assert!(state.block_keys.is_none());
        cache.reset().unwrap();
        assert_eq!(cache.offsets(), [0]);
        assert!(cache.state_arrays().is_empty());
    }
}

pub struct FlashLayerCache {
    pub(crate) rows: Vec<LayerState>,
    pub(crate) instance: Arc<()>,
    pub(crate) layer: usize,
    cap: i32,
    offsets: Vec<i32>,
    pub(crate) prefixes: Vec<Vec<LayerSnapshot>>,
}

#[derive(Clone)]
pub struct FlashLayerSnapshot {
    rows: Vec<LayerSnapshot>,
    instance: Arc<()>,
    layer: usize,
}

impl FlashLayerCache {
    pub(crate) fn new(
        rows: Vec<LayerState>,
        instance: Arc<()>,
        layer: usize,
        cap: i32,
    ) -> Result<Self> {
        ensure!(!rows.is_empty() && cap > 0, "invalid Flash layer cache");
        let offsets = rows.iter().map(LayerState::position).collect();
        Ok(Self {
            rows,
            instance,
            layer,
            cap,
            offsets,
            prefixes: Vec::new(),
        })
    }
    pub fn cap(&self) -> i32 {
        self.cap
    }
    pub fn dtype(&self) -> Dtype {
        Dtype::Bfloat16
    }
    pub fn offsets(&self) -> &[i32] {
        &self.offsets
    }
    pub fn state_arrays(&self) -> Vec<&Array> {
        self.rows.iter().flat_map(LayerState::arrays).collect()
    }
    pub fn diagnostic_tokens(&self) -> Vec<u32> {
        self.rows
            .iter()
            .flat_map(|state| match state {
                LayerState::Linear { ple: Some(ple), .. } => ple.history.tokens(),
                _ => &[],
            })
            .copied()
            .collect()
    }
    pub(crate) fn refresh_offsets(&mut self) {
        self.offsets = self.rows.iter().map(LayerState::position).collect();
    }
    pub fn reset(&mut self) -> Result<()> {
        for state in &mut self.rows {
            match state {
                LayerState::Linear { gdn, ple } => {
                    gdn.reset()?;
                    if let Some(ple) = ple {
                        ple.reset();
                    }
                }
                LayerState::Sparse(state) => state.reset(),
            }
        }
        self.refresh_offsets();
        self.prefixes.clear();
        Ok(())
    }
    pub fn snapshot(&self) -> FlashLayerSnapshot {
        FlashLayerSnapshot {
            rows: self.rows.iter().map(LayerState::snapshot).collect(),
            instance: Arc::clone(&self.instance),
            layer: self.layer,
        }
    }
    pub fn restore(&mut self, snapshot: &FlashLayerSnapshot) -> Result<()> {
        ensure!(
            Arc::ptr_eq(&self.instance, &snapshot.instance)
                && self.layer == snapshot.layer
                && self.rows.len() == snapshot.rows.len(),
            "Flash snapshot instance/layer/row mismatch"
        );
        let rows = snapshot
            .rows
            .iter()
            .map(|row| LayerState::from_snapshot(row, self.cap))
            .collect::<Result<Vec<_>>>()?;
        self.rows = rows;
        self.refresh_offsets();
        self.prefixes.clear();
        Ok(())
    }
    pub fn adopt_row_from(&mut self, source: &Self, destination: usize, row: usize) -> Result<()> {
        ensure!(
            Arc::ptr_eq(&self.instance, &source.instance)
                && self.layer == source.layer
                && destination < self.rows.len()
                && row < source.rows.len(),
            "Flash cache row identity/range mismatch"
        );
        let adopted = LayerState::from_snapshot(&source.rows[row].snapshot(), self.cap)?;
        self.rows[destination] = adopted;
        self.refresh_offsets();
        self.prefixes.clear();
        Ok(())
    }
    pub fn begin_speculative_prefix_capture(&mut self) -> Result<()> {
        ensure!(
            !self.rows.is_empty(),
            "cannot capture a borrowed Flash cache"
        );
        self.prefixes = (0..self.rows.len()).map(|_| Vec::new()).collect();
        Ok(())
    }
    pub fn discard_speculative_prefix_capture(&mut self) {
        self.prefixes.clear();
    }

    pub(crate) fn commit_prefix_rows(
        &mut self,
        base: &FlashLayerSnapshot,
        accepted: &[usize],
        target: StreamOrDevice,
    ) -> Result<()> {
        self.validate_prefix_rows(base, accepted)?;
        for (row, &count) in accepted.iter().enumerate() {
            if count == 0 {
                self.rows[row] = LayerState::from_snapshot(&base.rows[row], self.cap)?;
                continue;
            }
            match (&mut self.rows[row], &base.rows[row]) {
                (state @ LayerState::Linear { .. }, LayerSnapshot::Linear(..)) => {
                    *state = LayerState::from_snapshot(&self.prefixes[row][count - 1], self.cap)?;
                }
                (LayerState::Sparse(current), LayerSnapshot::Sparse(base)) => {
                    current.trim_on(base.offset + count as i32, target)?
                }
                _ => unreachable!("validated Flash layer kinds"),
            }
        }
        self.refresh_offsets();
        self.prefixes.clear();
        Ok(())
    }

    pub(crate) fn validate_prefix_rows(
        &self,
        base: &FlashLayerSnapshot,
        accepted: &[usize],
    ) -> Result<()> {
        ensure!(
            Arc::ptr_eq(&self.instance, &base.instance)
                && self.layer == base.layer
                && accepted.len() == self.rows.len()
                && base.rows.len() == self.rows.len(),
            "Flash prefix commit identity/row mismatch"
        );
        // Validate every row before changing any state in this layer.
        for (row, &count) in accepted.iter().enumerate() {
            match (&self.rows[row], &base.rows[row]) {
                (LayerState::Linear { .. }, LayerSnapshot::Linear(..)) => {
                    ensure!(
                        count == 0
                            || self
                                .prefixes
                                .get(row)
                                .is_some_and(|prefixes| count <= prefixes.len()),
                        "Flash recurrent prefix was not captured"
                    );
                }
                (LayerState::Sparse(current), LayerSnapshot::Sparse(base)) => {
                    ensure!(
                        i32::try_from(count)
                            .ok()
                            .and_then(|n| base.offset.checked_add(n))
                            .is_some_and(|end| end <= current.offset),
                        "Flash QSA prefix exceeds verified window"
                    );
                }
                _ => anyhow::bail!("Flash prefix commit layer kind mismatch"),
            }
        }
        Ok(())
    }
}
