# Scheduler Internal MTP B1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `b_max=1` text-only Qwen scheduler-internal MTP speculative decoding and switch `ironmlx-core-bench scheduler-text --mtp-model-dir` to that scheduler path.

**Architecture:** Keep `Scheduler<M>` generic over `M: Model` and add optional private MTP runtime state inside `Scheduler`. Add MTP-only methods with `M: MtpSpeculativeModel + DenseVlMethods` bounds, reusing the existing single-request MTP stream semantics and offsets-only rollback helpers. Bench uses those scheduler methods for `scheduler-text --b-max 1 --mtp-model-dir`, while normal scheduler, actor, VL, and multi-row paths remain unchanged.

**Tech Stack:** Rust, MLX arrays, `anyhow`, existing `MtpSpeculativeModel`, `MtpCache`, `LayerCacheSnapshot`, `Scheduler`, and `ironmlx-core-bench`.

---

## File Structure

- Modify `ironmlx/src/core/speculative.rs`
  - Make existing single-request helper functions visible to sibling core modules: `verify_input`, `sample_logits_positions`, `slice_hidden_position`, `restore_layer_cache`.
  - No behavior change.

- Modify `ironmlx/src/core/scheduler.rs`
  - Import MTP types and `VecDeque`.
  - Add private `SchedulerMtpState`.
  - Add `mtp_state: Option<SchedulerMtpState>` to `Scheduler<M>`.
  - Add `prefill_admitted_mtp_single`, `step_mtp_single`, `mtp_stats`, and private MTP window helpers.
  - Add unit tests with a scripted fake MTP model.

- Modify `ironmlx/src/bin/ironmlx-core-bench.rs`
  - Remove the scheduler-text MTP delegate to `MtpTextGenerationStream`.
  - Add a scheduler-internal MTP helper that admits one request, calls `prefill_admitted_mtp_single`, then loops `step_mtp_single`.
  - Keep `mtp-text` on `MtpTextGenerationStream`.

---

### Task 1: Write Scheduler MTP Red Tests

**Files:**
- Modify: `ironmlx/src/core/scheduler.rs`

- [ ] **Step 1: Add test imports**

Inside `mod tests` in `ironmlx/src/core/scheduler.rs`, extend the imports near the top of the test module:

```rust
use crate::core::cache::MtpCache;
use crate::core::speculative::{MtpSpeculativeConfig, MtpSpeculativeModel};
use crate::nn::MtpStepOutput;
```

- [ ] **Step 2: Add scripted fake MTP model helpers**

Insert this code inside `mod tests`, immediately after the existing `StepDecodeMaskModel` `DenseVlMethods` impl:

```rust
#[derive(Clone, Copy)]
struct FakeMtpHead;

#[derive(Default)]
struct ScriptedMtpSchedulerModel {
    first_token: u32,
    draft_tokens: std::sync::Mutex<VecDeque<u32>>,
    verify_sequences: std::sync::Mutex<VecDeque<Vec<u32>>>,
    project_calls: std::sync::Mutex<usize>,
}

impl ScriptedMtpSchedulerModel {
    fn new(first_token: u32, draft_tokens: Vec<u32>, verify_sequences: Vec<Vec<u32>>) -> Self {
        Self {
            first_token,
            draft_tokens: std::sync::Mutex::new(draft_tokens.into()),
            verify_sequences: std::sync::Mutex::new(verify_sequences.into()),
            project_calls: std::sync::Mutex::new(0),
        }
    }

    fn bump_first_full_cache(
        cache: Option<&mut [crate::nn::LayerCache]>,
        input_ids: &mlx::Array,
        per_row_lens: Option<&[i32]>,
    ) -> crate::Result<()> {
        let Some(cache) = cache else {
            return Ok(());
        };
        let dims = input_ids.shape();
        let dims = dims.as_slice();
        let batch = dims[0];
        let seq = dims[1];
        let lens_owned;
        let lens = if let Some(lens) = per_row_lens {
            lens
        } else {
            lens_owned = vec![seq; batch as usize];
            lens_owned.as_slice()
        };
        let k = mlx::Array::zeros((batch, 1_i32, seq, 1_i32), mlx::Dtype::Bfloat16)
            .map_err(|e| anyhow::anyhow!("fake k failed: {e:?}"))?;
        let v = mlx::Array::zeros((batch, 1_i32, seq, 1_i32), mlx::Dtype::Bfloat16)
            .map_err(|e| anyhow::anyhow!("fake v failed: {e:?}"))?;
        for layer in cache {
            if let crate::nn::LayerCache::Full(kv) = layer {
                kv.update_and_fetch(&k, &v, lens)?;
                break;
            }
        }
        Ok(())
    }
}

fn fake_logits_for_token_sequence(tokens: &[u32]) -> crate::Result<mlx::Array> {
    let seq = tokens.len();
    let vocab = 32_usize;
    let mut flat = vec![0.0_f32; seq * vocab];
    for (pos, &token) in tokens.iter().enumerate() {
        let token = token as usize;
        assert!(token < vocab, "fake token {token} must fit fake vocab {vocab}");
        flat[pos * vocab + token] = 100.0;
    }
    let logits: mlx::Array = (&flat[..], &[1_i32, seq as i32, vocab as i32][..])
        .try_into()
        .expect("fake logits [1,S,V]");
    Ok(logits)
}

impl crate::core::model::Model for ScriptedMtpSchedulerModel {
    fn make_cache(
        &self,
        batch: i32,
        cap: i32,
        dtype: mlx::Dtype,
    ) -> crate::Result<Vec<crate::nn::LayerCache>> {
        Ok(vec![crate::nn::LayerCache::Full(
            crate::core::KVCache::new(batch, 1, 1, 1, dtype, cap),
        )])
    }

    fn forward_on(
        &self,
        input_ids: &mlx::Array,
        _position_ids: &mlx::Array,
        per_row_lens: Option<&[i32]>,
        _decode_mask: Option<&mlx::Array>,
        cache: Option<&mut [crate::nn::LayerCache]>,
        _target: mlx::StreamOrDevice,
    ) -> crate::Result<mlx::Array> {
        Self::bump_first_full_cache(cache, input_ids, per_row_lens)?;
        fake_logits_for_token_sequence(&[self.first_token])
    }

    fn batched_prefill(
        &self,
        input_ids: &mlx::Array,
        _position_ids: &mlx::Array,
        _attention_mask: &mlx::Array,
        _linear_attention_mask: &mlx::Array,
        per_row_lens: &[i32],
        cache: Option<&mut [crate::nn::LayerCache]>,
        _target: mlx::StreamOrDevice,
    ) -> crate::Result<mlx::Array> {
        Self::bump_first_full_cache(cache, input_ids, Some(per_row_lens))?;
        fake_logits_for_token_sequence(&[self.first_token])
    }

    fn forward_text_hidden(
        &self,
        input_ids: &mlx::Array,
        _position_ids: &mlx::Array,
        per_row_lens: Option<&[i32]>,
        _decode_mask: Option<&mlx::Array>,
        cache: Option<&mut [crate::nn::LayerCache]>,
        _target: mlx::StreamOrDevice,
    ) -> crate::Result<mlx::Array> {
        Self::bump_first_full_cache(cache, input_ids, per_row_lens)?;
        let dims = input_ids.shape();
        let dims = dims.as_slice();
        mlx::Array::zeros((dims[0], dims[1], 4_i32), mlx::Dtype::Float32)
            .map_err(|e| anyhow::anyhow!("fake hidden failed: {e:?}"))
    }

    fn model_meta(&self) -> crate::core::memory_budget::ModelMeta {
        crate::core::memory_budget::test_meta_qwen35()
    }

    fn num_hidden_layers(&self) -> usize {
        1
    }
}

impl DenseVlMethods for ScriptedMtpSchedulerModel {
    fn batched_prefill_vl(
        &self,
        _input_ids: &mlx::Array,
        _position_ids: &mlx::Array,
        _attention_mask: &mlx::Array,
        _linear_attention_mask: &mlx::Array,
        _per_row_lens: &[i32],
        _per_row_pixel_values: &[Option<&[mlx::Array]>],
        _per_row_grid_thw: &[Option<&[(i32, i32, i32)]>],
        _image_token_id: i32,
        _cache: Option<&mut [crate::nn::LayerCache]>,
        _target: mlx::StreamOrDevice,
    ) -> crate::Result<mlx::Array> {
        unreachable!("scripted MTP scheduler tests are text-only")
    }

    fn compute_vision_embeds(
        &self,
        _pixel_values: &[mlx::Array],
        _grid_thw: &[(i32, i32, i32)],
        _target: mlx::StreamOrDevice,
    ) -> crate::Result<mlx::Array> {
        unreachable!("scripted MTP scheduler tests are text-only")
    }

    fn forward_vl_chunk(
        &self,
        _input_ids: &mlx::Array,
        _position_ids: &mlx::Array,
        _per_row_lens: Option<&[i32]>,
        _decode_mask: Option<&mlx::Array>,
        _cache: Option<&mut [crate::nn::LayerCache]>,
        _vision_embeds_slice: Option<&mlx::Array>,
        _image_token_id: i32,
        _target: mlx::StreamOrDevice,
    ) -> crate::Result<mlx::Array> {
        unreachable!("scripted MTP scheduler tests are text-only")
    }

    fn forward_vl_hidden(
        &self,
        _input_ids: &mlx::Array,
        _position_ids: &mlx::Array,
        _per_row_lens: Option<&[i32]>,
        _decode_mask: Option<&mlx::Array>,
        _cache: Option<&mut [crate::nn::LayerCache]>,
        _vision_embeds_slice: Option<&mlx::Array>,
        _image_token_id: i32,
        _target: mlx::StreamOrDevice,
    ) -> crate::Result<mlx::Array> {
        unreachable!("scripted MTP scheduler tests are text-only")
    }
}

impl MtpSpeculativeModel for ScriptedMtpSchedulerModel {
    type MtpHead = FakeMtpHead;

    fn load_mtp_head(&self, _loader: &crate::core::Loader) -> crate::Result<Self::MtpHead> {
        Ok(FakeMtpHead)
    }

    fn make_mtp_cache(
        &self,
        _mtp: &Self::MtpHead,
        batch: i32,
        cap: i32,
        dtype: mlx::Dtype,
    ) -> crate::Result<MtpCache> {
        MtpCache::new_with_cap(1, batch, 1, 1, 1, dtype, cap)
    }

    fn project_hidden_on(
        &self,
        hidden: &mlx::Array,
        _target: impl Into<mlx::StreamOrDevice>,
    ) -> crate::Result<mlx::Array> {
        let seq = hidden.shape().as_slice()[1] as usize;
        let mut calls = self.project_calls.lock().unwrap();
        let tokens = if *calls == 0 {
            vec![self.first_token]
        } else {
            self.verify_sequences
                .lock()
                .unwrap()
                .pop_front()
                .expect("verify sequence available")
        };
        *calls += 1;
        assert_eq!(tokens.len(), seq, "project logits sequence length must match hidden seq");
        fake_logits_for_token_sequence(&tokens)
    }

    fn mtp_forward_on(
        &self,
        _mtp: &Self::MtpHead,
        hidden_states: &mlx::Array,
        next_token_ids: &mlx::Array,
        _position_ids: &mlx::Array,
        _mask: Option<&mlx::Array>,
        mtp_cache: Option<&mut MtpCache>,
        _target: impl Into<mlx::StreamOrDevice>,
    ) -> crate::Result<MtpStepOutput> {
        if let Some(cache) = mtp_cache {
            let k = mlx::Array::zeros((1_i32, 1_i32, 1_i32, 1_i32), mlx::Dtype::Bfloat16)
                .map_err(|e| anyhow::anyhow!("fake mtp k failed: {e:?}"))?;
            let v = mlx::Array::zeros((1_i32, 1_i32, 1_i32, 1_i32), mlx::Dtype::Bfloat16)
                .map_err(|e| anyhow::anyhow!("fake mtp v failed: {e:?}"))?;
            cache.layer_mut(0).update_and_fetch(&k, &v, &[1])?;
        }
        let token = self
            .draft_tokens
            .lock()
            .unwrap()
            .pop_front()
            .expect("draft token available");
        Ok(MtpStepOutput {
            hidden_states: hidden_states.clone(),
            logits: fake_logits_for_token_sequence(&[token])?,
        })
    }
}

fn mtp_req(prompt_ids: Vec<u32>, max_new_tokens: usize) -> GenerateRequest {
    let mut req = mk_req(prompt_ids);
    req.max_new_tokens = max_new_tokens;
    req.stop_token_ids = vec![31];
    req
}
```

- [ ] **Step 3: Add gating tests**

Insert these tests after the helper code:

```rust
#[test]
fn mtp_prefill_rejects_bmax_gt_one() {
    let mut s = Scheduler::<ScriptedMtpSchedulerModel>::new(
        2,
        32768,
        crate::core::memory_budget::test_meta_qwen35(),
    )
    .expect("scheduler startup");
    s.admit(mtp_req(vec![1, 2, 3], 4)).expect("admit");

    let model = ScriptedMtpSchedulerModel::new(3, vec![4], vec![vec![4, 5]]);
    let cfg = MtpSpeculativeConfig::new(1, Sampler::greedy()).expect("mtp cfg");
    let err = s
        .prefill_admitted_mtp_single(&model, &FakeMtpHead, cfg)
        .expect_err("b_max > 1 must reject scheduler MTP");
    assert!(err.to_string().contains("b_max 1"), "unexpected err: {err}");
}

#[test]
fn mtp_prefill_rejects_vl_request() {
    let mut s = Scheduler::<ScriptedMtpSchedulerModel>::new(
        1,
        32768,
        crate::core::memory_budget::test_meta_qwen35(),
    )
    .expect("scheduler startup");
    let mut req = mtp_req(vec![1, 2, 3], 4);
    req.pixel_values = Some(vec![mlx::Array::zeros((1_i32, 1_i32), mlx::Dtype::Float32).unwrap()]);
    req.image_grid_thw = Some(vec![(1, 1, 1)]);
    s.admit(req).expect("admit");

    let model = ScriptedMtpSchedulerModel::new(3, vec![4], vec![vec![4, 5]]);
    let cfg = MtpSpeculativeConfig::new(1, Sampler::greedy()).expect("mtp cfg");
    let err = s
        .prefill_admitted_mtp_single(&model, &FakeMtpHead, cfg)
        .expect_err("VL request must reject scheduler MTP");
    assert!(err.to_string().contains("text-only"), "unexpected err: {err}");
}
```

- [ ] **Step 4: Add pending queue and rollback behavior tests**

Insert these tests after the gating tests:

```rust
#[test]
fn mtp_step_emits_one_pending_token_per_call() {
    let mut s = Scheduler::<ScriptedMtpSchedulerModel>::new(
        1,
        32768,
        crate::core::memory_budget::test_meta_qwen35(),
    )
    .expect("scheduler startup");
    let id = s.admit(mtp_req(vec![1, 2], 4)).expect("admit");
    let model = ScriptedMtpSchedulerModel::new(3, vec![4, 5], vec![vec![4, 5, 6]]);
    let cfg = MtpSpeculativeConfig::new(2, Sampler::greedy()).expect("mtp cfg");

    let first = s
        .prefill_admitted_mtp_single(&model, &FakeMtpHead, cfg)
        .expect("mtp prefill");
    assert_eq!(first, vec![StepEvent { id, token: 3, finish_reason: None }]);
    assert_eq!(s.get(id).unwrap().generated_tokens, vec![3]);

    let step_1 = s.step_mtp_single(&model, &FakeMtpHead).expect("step 1");
    assert_eq!(step_1, vec![StepEvent { id, token: 4, finish_reason: None }]);
    assert_eq!(s.get(id).unwrap().generated_tokens, vec![3, 4]);

    let step_2 = s.step_mtp_single(&model, &FakeMtpHead).expect("step 2");
    assert_eq!(step_2, vec![StepEvent { id, token: 5, finish_reason: None }]);
    assert_eq!(s.get(id).unwrap().generated_tokens, vec![3, 4, 5]);

    let step_3 = s.step_mtp_single(&model, &FakeMtpHead).expect("step 3");
    assert_eq!(step_3, vec![StepEvent { id, token: 6, finish_reason: Some("length") }]);
    assert_eq!(s.get(id).unwrap().generated_tokens, vec![3, 4, 5, 6]);

    let stats = s.mtp_stats().expect("mtp stats");
    assert_eq!(stats.windows, 1);
    assert_eq!(stats.drafted_tokens, 2);
    assert_eq!(stats.accepted_draft_tokens, 2);
    assert_eq!(stats.rollback_count, 0);
}

#[test]
fn mtp_step_updates_stats_after_mismatch() {
    let mut s = Scheduler::<ScriptedMtpSchedulerModel>::new(
        1,
        32768,
        crate::core::memory_budget::test_meta_qwen35(),
    )
    .expect("scheduler startup");
    let id = s.admit(mtp_req(vec![1, 2], 2)).expect("admit");
    let model = ScriptedMtpSchedulerModel::new(3, vec![8], vec![vec![4, 5]]);
    let cfg = MtpSpeculativeConfig::new(2, Sampler::greedy()).expect("mtp cfg");

    let first = s
        .prefill_admitted_mtp_single(&model, &FakeMtpHead, cfg)
        .expect("mtp prefill");
    assert_eq!(first, vec![StepEvent { id, token: 3, finish_reason: None }]);

    let stats = s.mtp_stats().expect("mtp stats after prefill window");
    assert_eq!(stats.windows, 1);
    assert_eq!(stats.drafted_tokens, 1);
    assert_eq!(stats.accepted_draft_tokens, 0);
    assert_eq!(stats.rollback_count, 1);

    let step = s.step_mtp_single(&model, &FakeMtpHead).expect("corrected step");
    assert_eq!(step, vec![StepEvent { id, token: 4, finish_reason: Some("length") }]);
    assert_eq!(s.get(id).unwrap().generated_tokens, vec![3, 4]);
}
```

- [ ] **Step 5: Run tests to verify RED**

Run:

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --lib mtp_
```

Expected: FAIL at compile time with errors including these missing scheduler methods:

```text
no method named `prefill_admitted_mtp_single` found
no method named `step_mtp_single` found
no method named `mtp_stats` found
```

Do not edit production code before this failure is observed.

---

### Task 2: Implement Scheduler MTP State and Methods

**Files:**
- Modify: `ironmlx/src/core/speculative.rs`
- Modify: `ironmlx/src/core/scheduler.rs`

- [ ] **Step 1: Expose existing speculative helpers to scheduler**

In `ironmlx/src/core/speculative.rs`, change these helper signatures:

```rust
pub(crate) fn verify_input(current_token: u32, draft_tokens: &[u32]) -> Vec<u32> {
```

```rust
pub(crate) fn sample_logits_positions(
    logits: &Array,
    sampler: Sampler,
    history: &[u32],
    prng_state: &mut Array,
) -> Result<Vec<u32>> {
```

```rust
pub(crate) fn slice_hidden_position(hidden: &Array, pos: i32) -> Result<Array> {
```

```rust
pub(crate) fn restore_layer_cache(
    cache: &mut [LayerCache],
    snapshots: &[LayerCacheSnapshot],
) -> Result<()> {
```

- [ ] **Step 2: Add scheduler imports**

At the top of `ironmlx/src/core/scheduler.rs`, change the collections import and add MTP imports:

```rust
use std::collections::{HashMap, VecDeque};
```

Add after the existing `use crate::core::sampler::Sampler;` line:

```rust
use crate::core::cache::MtpCache;
use crate::core::speculative::{
    restore_layer_cache, resolve_speculative_tokens, sample_logits_positions,
    slice_hidden_position, verify_input, MtpSpeculativeConfig, MtpSpeculativeModel,
    MtpSpeculativeStats,
};
```

- [ ] **Step 3: Add scheduler MTP state**

Insert after the `type PixelValuesSlice<'a>` alias:

```rust
struct SchedulerMtpState {
    cfg: MtpSpeculativeConfig,
    mtp_cache: MtpCache,
    pending_tokens: VecDeque<u32>,
    last_hidden: Array,
    stats: MtpSpeculativeStats,
}
```

- [ ] **Step 4: Add and initialize scheduler field**

Add this field to `pub struct Scheduler<M: Model>` after `dummy_position_ids`:

```rust
    /// Optional single-request MTP runtime state. Present only for the
    /// `b_max=1` scheduler-internal MTP path.
    mtp_state: Option<SchedulerMtpState>,
```

Initialize it in `Scheduler::new_with_state`:

```rust
            mtp_state: None,
```

Add this to `Debug` output after `has_dummy_position_ids`:

```rust
            .field("has_mtp_state", &self.mtp_state.is_some())
```

- [ ] **Step 5: Clear MTP state on eviction and evict_all**

In `Scheduler::evict`, after clearing a slot and before phase transitions, add:

```rust
        if self.active_count() == 0 {
            self.mtp_state = None;
        }
```

In `Scheduler::evict_all`, after `self.cache_rows.clear();`, add:

```rust
        self.mtp_state = None;
```

- [ ] **Step 6: Add MTP position helper**

Inside `impl<M: Model> Scheduler<M>`, after `reusable_dummy_position_ids`, add:

```rust
    fn mtp_position_ids(&mut self, model: &M, start_pos: i32, len: i32) -> Result<Array> {
        if model.requires_position_ids() {
            build_position_ids(start_pos, len)
        } else {
            self.reusable_dummy_position_ids()
        }
    }
```

- [ ] **Step 7: Add public MTP entry methods**

Inside `impl<M: Model> Scheduler<M>`, before `prefill_admitted`, add:

```rust
    pub fn mtp_stats(&self) -> Option<MtpSpeculativeStats> {
        self.mtp_state.as_ref().map(|state| state.stats)
    }

    pub fn prefill_admitted_mtp_single(
        &mut self,
        model: &M,
        mtp: &M::MtpHead,
        cfg: MtpSpeculativeConfig,
    ) -> Result<Vec<StepEvent>>
    where
        M: MtpSpeculativeModel + DenseVlMethods,
    {
        self.ensure_not_poisoned()?;
        match self.prefill_admitted_mtp_single_inner(model, mtp, cfg) {
            Ok(events) => Ok(events),
            Err(e) => {
                self.poisoned = true;
                Err(e)
            }
        }
    }

    pub fn step_mtp_single(&mut self, model: &M, mtp: &M::MtpHead) -> Result<Vec<StepEvent>>
    where
        M: MtpSpeculativeModel,
    {
        self.ensure_not_poisoned()?;
        match self.step_mtp_single_inner(model, mtp) {
            Ok(events) => Ok(events),
            Err(e) => {
                self.poisoned = true;
                Err(e)
            }
        }
    }
```

- [ ] **Step 8: Add MTP prefill implementation**

Insert these private methods after the public MTP entry methods:

```rust
    fn prefill_admitted_mtp_single_inner(
        &mut self,
        model: &M,
        mtp: &M::MtpHead,
        cfg: MtpSpeculativeConfig,
    ) -> Result<Vec<StepEvent>>
    where
        M: MtpSpeculativeModel + DenseVlMethods,
    {
        if self.b_max != 1 {
            return Err(anyhow!(
                "prefill_admitted_mtp_single currently requires b_max 1"
            ));
        }
        match self.phase {
            Phase::Idle | Phase::Admitting => {}
            Phase::Decoding | Phase::Finished => {
                return Err(anyhow!(
                    "prefill_admitted_mtp_single illegal in {:?} phase: call evict_all first",
                    self.phase
                ));
            }
        }

        let active_rows: Vec<usize> = self
            .slots
            .iter()
            .enumerate()
            .filter_map(|(row, slot)| slot.as_ref().map(|_| row))
            .collect();
        if active_rows.len() != 1 {
            return Err(anyhow!(
                "prefill_admitted_mtp_single requires exactly one admitted request, got {}",
                active_rows.len()
            ));
        }
        let row_idx = active_rows[0];
        let (id, prompt_ids, max_new_tokens, sampler, stop_token_ids, prefill_chunk_size) = {
            let state = self.slots[row_idx]
                .as_ref()
                .expect("active row implies slot is Some");
            if state.pixel_values.is_some() {
                return Err(anyhow!(
                    "prefill_admitted_mtp_single supports text-only requests"
                ));
            }
            (
                state.id,
                state.prompt_ids.clone(),
                state.max_new_tokens,
                state.sampler,
                state.stop_token_ids.clone(),
                state.prefill_chunk_size,
            )
        };
        if prompt_ids.is_empty() {
            return Err(anyhow!(
                "prefill_admitted_mtp_single: prompt_ids cannot be empty"
            ));
        }
        MtpSpeculativeConfig::new(cfg.max_draft_tokens, sampler)?;

        let prompt_len = prompt_ids.len();
        let cap = (prompt_len.saturating_add(max_new_tokens) as i32)
            .max(MIN_KV_CACHE_CAP_FOR_GPU_PERF);
        let dtype = Dtype::Bfloat16;
        if self.cache.is_some() {
            return Err(anyhow!(
                "prefill_admitted_mtp_single: cache already allocated before prefill"
            ));
        }
        self.cache = Some(model.make_cache(1, cap, dtype)?);
        self.cache_rows = vec![row_idx];

        let mtp_cache = model.make_mtp_cache(mtp, 1, cap, dtype)?;
        let prompt_len_i32 = prompt_len as i32;
        let mut pos = 0_i32;
        let last_prompt_hidden = loop {
            let remaining = prompt_len_i32 - pos;
            let n = if prefill_chunk_size == 0 {
                remaining
            } else {
                remaining.min(prefill_chunk_size.max(1))
            };
            let chunk_ids = &prompt_ids[pos as usize..(pos as usize + n as usize)];
            let chunk_arr: Array = (chunk_ids, &[1_i32, n][..]).try_into()?;
            let chunk_pos_ids = self.mtp_position_ids(model, pos, n)?;
            let hidden = {
                let cache = self
                    .cache
                    .as_mut()
                    .ok_or_else(|| anyhow!("prefill_admitted_mtp_single: cache absent"))?;
                model.forward_text_hidden(
                    &chunk_arr,
                    &chunk_pos_ids,
                    None,
                    None,
                    Some(cache.as_mut_slice()),
                    mlx::StreamOrDevice::default(),
                )?
            };
            if pos + n == prompt_len_i32 {
                break slice_hidden_position(&hidden, n - 1)?;
            }
            mlx::transforms::eval(&[&hidden])?;
            pos += n;
        };

        let first_logits = model.project_hidden_on(&last_prompt_hidden, mlx::StreamOrDevice::default())?;
        let mut compact_prng = self.compact_prng_state_for_rows(&[row_idx])?;
        let first_tokens = sample_logits_positions(
            &first_logits,
            sampler,
            &prompt_ids,
            &mut compact_prng,
        )?;
        self.scatter_prng_state_from_rows(&[row_idx], &compact_prng)?;
        let first_token = *first_tokens
            .first()
            .ok_or_else(|| anyhow!("prefill_admitted_mtp_single produced no first token"))?;

        let finish_reason = {
            let state = self.slots[row_idx]
                .as_mut()
                .expect("active row implies slot is Some");
            state.generated_tokens.push(first_token);
            state.real_len += 1;
            if stop_token_ids.contains(&first_token) {
                state.finished = true;
                state.finish_reason = Some("stop");
            } else if state.generated_tokens.len() >= state.max_new_tokens {
                state.finished = true;
                state.finish_reason = Some("length");
            }
            state.finish_reason
        };

        self.phase = if finish_reason.is_some() {
            Phase::Finished
        } else {
            Phase::Decoding
        };
        self.mtp_state = Some(SchedulerMtpState {
            cfg,
            mtp_cache,
            pending_tokens: VecDeque::new(),
            last_hidden: last_prompt_hidden,
            stats: MtpSpeculativeStats::default(),
        });

        if finish_reason.is_none() {
            self.fill_mtp_window_single(row_idx, model, mtp)?;
        }

        Ok(vec![StepEvent {
            id,
            token: first_token,
            finish_reason,
        }])
    }
```

- [ ] **Step 9: Add MTP step and window helpers**

Insert these methods after `prefill_admitted_mtp_single_inner`:

```rust
    fn step_mtp_single_inner(&mut self, model: &M, mtp: &M::MtpHead) -> Result<Vec<StepEvent>>
    where
        M: MtpSpeculativeModel,
    {
        if self.b_max != 1 {
            return Err(anyhow!("step_mtp_single currently requires b_max 1"));
        }
        if self.phase != Phase::Decoding {
            return Err(anyhow!(
                "step_mtp_single illegal in {:?} phase: call prefill_admitted_mtp_single first",
                self.phase
            ));
        }
        let row_idx = self
            .slots
            .iter()
            .position(|slot| matches!(slot, Some(state) if !state.finished))
            .ok_or_else(|| anyhow!("step_mtp_single: no unfinished row"))?;

        if self
            .mtp_state
            .as_ref()
            .ok_or_else(|| anyhow!("step_mtp_single: MTP state absent"))?
            .pending_tokens
            .is_empty()
        {
            self.fill_mtp_window_single(row_idx, model, mtp)?;
        }

        let token = {
            let mtp_state = self
                .mtp_state
                .as_mut()
                .ok_or_else(|| anyhow!("step_mtp_single: MTP state absent"))?;
            mtp_state
                .pending_tokens
                .pop_front()
                .ok_or_else(|| anyhow!("step_mtp_single: pending token queue is empty"))?
        };

        let (id, finish_reason) = {
            let state = self.slots[row_idx]
                .as_mut()
                .expect("unfinished row implies slot is Some");
            state.generated_tokens.push(token);
            state.real_len += 1;
            if state.stop_token_ids.contains(&token) {
                state.finished = true;
                state.finish_reason = Some("stop");
            } else if state.generated_tokens.len() >= state.max_new_tokens {
                state.finished = true;
                state.finish_reason = Some("length");
            }
            (state.id, state.finish_reason)
        };

        if finish_reason.is_some() {
            self.phase = Phase::Finished;
        } else if self
            .mtp_state
            .as_ref()
            .is_some_and(|state| state.pending_tokens.is_empty())
        {
            self.fill_mtp_window_single(row_idx, model, mtp)?;
        }

        Ok(vec![StepEvent {
            id,
            token,
            finish_reason,
        }])
    }

    fn fill_mtp_window_single(
        &mut self,
        row_idx: usize,
        model: &M,
        mtp: &M::MtpHead,
    ) -> Result<()>
    where
        M: MtpSpeculativeModel,
    {
        let mut mtp_state = self
            .mtp_state
            .take()
            .ok_or_else(|| anyhow!("fill_mtp_window_single: MTP state absent"))?;
        let result = self.fill_mtp_window_single_with_state(row_idx, &mut mtp_state, model, mtp);
        self.mtp_state = Some(mtp_state);
        result
    }

    fn fill_mtp_window_single_with_state(
        &mut self,
        row_idx: usize,
        mtp_state: &mut SchedulerMtpState,
        model: &M,
        mtp: &M::MtpHead,
    ) -> Result<()>
    where
        M: MtpSpeculativeModel,
    {
        let (prompt_ids, generated_tokens, max_new_tokens, sampler, stop_token_ids) = {
            let state = self.slots[row_idx]
                .as_ref()
                .ok_or_else(|| anyhow!("fill_mtp_window_single: row slot absent"))?;
            (
                state.prompt_ids.clone(),
                state.generated_tokens.clone(),
                state.max_new_tokens,
                state.sampler,
                state.stop_token_ids.clone(),
            )
        };
        let emitted = generated_tokens.len();
        let remaining = max_new_tokens.saturating_sub(emitted);
        if remaining == 0 {
            return Ok(());
        }
        let current_token = *generated_tokens
            .last()
            .ok_or_else(|| anyhow!("fill_mtp_window_single: no current token"))?;
        let mut history = Vec::with_capacity(prompt_ids.len() + generated_tokens.len());
        history.extend_from_slice(&prompt_ids);
        history.extend_from_slice(&generated_tokens);

        let draft_budget = mtp_state.cfg.max_draft_tokens.min(remaining);
        let mut compact_prng = self.compact_prng_state_for_rows(&[row_idx])?;
        let draft_tokens = self.draft_mtp_tokens_single(
            mtp_state,
            model,
            mtp,
            current_token,
            draft_budget,
            &history,
            sampler,
            &mut compact_prng,
        )?;
        let verify_input = verify_input(current_token, &draft_tokens);
        let verify_start_pos = (prompt_ids.len() + generated_tokens.len() - 1) as i32;
        let verify_pos_ids =
            self.mtp_position_ids(model, verify_start_pos, verify_input.len() as i32)?;
        let verify_arr: Array =
            (&verify_input[..], &[1_i32, verify_input.len() as i32][..]).try_into()?;

        let base_snapshot = {
            let cache = self
                .cache
                .as_ref()
                .ok_or_else(|| anyhow!("fill_mtp_window_single: main cache absent"))?;
            cache.iter().map(LayerCache::snapshot).collect::<Vec<_>>()
        };
        let verified_hidden = {
            let cache = self
                .cache
                .as_mut()
                .ok_or_else(|| anyhow!("fill_mtp_window_single: main cache absent"))?;
            model.forward_text_hidden(
                &verify_arr,
                &verify_pos_ids,
                None,
                None,
                Some(cache.as_mut_slice()),
                mlx::StreamOrDevice::default(),
            )?
        };
        let verified_logits =
            model.project_hidden_on(&verified_hidden, mlx::StreamOrDevice::default())?;
        let verified_tokens =
            sample_logits_positions(&verified_logits, sampler, &history, &mut compact_prng)?;
        self.scatter_prng_state_from_rows(&[row_idx], &compact_prng)?;

        let resolution = resolve_speculative_tokens(&draft_tokens, &verified_tokens)?;
        mtp_state.stats.windows += 1;
        mtp_state.stats.drafted_tokens += draft_tokens.len();
        mtp_state.stats.accepted_draft_tokens += resolution.accepted_draft_len;
        if resolution.needs_rollback {
            mtp_state.stats.rollback_count += 1;
        }

        mtp_state.last_hidden = if resolution.needs_rollback {
            {
                let cache = self
                    .cache
                    .as_mut()
                    .ok_or_else(|| anyhow!("fill_mtp_window_single: main cache absent"))?;
                restore_layer_cache(cache.as_mut_slice(), &base_snapshot)?;
            }
            let replay_len = resolution.accepted_verify_input_len;
            let replay_input = &verify_input[..replay_len];
            let replay_arr: Array =
                (replay_input, &[1_i32, replay_len as i32][..]).try_into()?;
            let replay_pos_ids =
                self.mtp_position_ids(model, verify_start_pos, replay_len as i32)?;
            let replay_hidden = {
                let cache = self
                    .cache
                    .as_mut()
                    .ok_or_else(|| anyhow!("fill_mtp_window_single: main cache absent"))?;
                model.forward_text_hidden(
                    &replay_arr,
                    &replay_pos_ids,
                    None,
                    None,
                    Some(cache.as_mut_slice()),
                    mlx::StreamOrDevice::default(),
                )?
            };
            slice_hidden_position(&replay_hidden, replay_len as i32 - 1)?
        } else {
            slice_hidden_position(
                &verified_hidden,
                resolution.accepted_verify_input_len as i32 - 1,
            )?
        };

        let mut tokens_to_append = resolution.tokens_to_append;
        if let Some(stop_idx) = tokens_to_append
            .iter()
            .position(|token| stop_token_ids.contains(token))
        {
            tokens_to_append.truncate(stop_idx + 1);
        }
        tokens_to_append.truncate(remaining);
        mtp_state.pending_tokens.extend(tokens_to_append);
        Ok(())
    }

    fn draft_mtp_tokens_single(
        &mut self,
        mtp_state: &mut SchedulerMtpState,
        model: &M,
        mtp: &M::MtpHead,
        current_token: u32,
        draft_budget: usize,
        history: &[u32],
        sampler: Sampler,
        prng_state: &mut Array,
    ) -> Result<Vec<u32>>
    where
        M: MtpSpeculativeModel,
    {
        let mtp_snapshot = mtp_state.mtp_cache.snapshot();
        let mut draft_tokens = Vec::with_capacity(draft_budget);
        let mut draft_history = history.to_vec();
        let mut input_hidden = mtp_state.last_hidden.clone();
        let mut input_token = current_token;
        let start_pos = (history.len() - 1) as i32;

        for offset in 0..draft_budget {
            let token_arr: Array = (&[input_token][..], &[1_i32, 1_i32][..]).try_into()?;
            let position_ids = self.mtp_position_ids(model, start_pos + offset as i32, 1)?;
            let output = model.mtp_forward_on(
                mtp,
                &input_hidden,
                &token_arr,
                &position_ids,
                None,
                Some(&mut mtp_state.mtp_cache),
                mlx::StreamOrDevice::default(),
            )?;
            let sampled =
                sample_logits_positions(&output.logits, sampler, &draft_history, prng_state)?;
            let next_token = *sampled
                .first()
                .ok_or_else(|| anyhow!("draft_mtp_tokens_single: MTP draft produced no token"))?;
            draft_tokens.push(next_token);
            draft_history.push(next_token);
            input_hidden = output.hidden_states;
            input_token = next_token;
        }

        mtp_state.mtp_cache.restore(&mtp_snapshot)?;
        Ok(draft_tokens)
    }
```

- [ ] **Step 10: Run scheduler tests to verify GREEN**

Run:

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --lib mtp_
```

Expected: PASS for the new scheduler MTP tests and any existing `mtp_` tests.

- [ ] **Step 11: Commit scheduler internal MTP**

Run:

```bash
git add ironmlx/src/core/speculative.rs ironmlx/src/core/scheduler.rs
git commit -m "feat: add scheduler internal mtp single path"
```

---

### Task 3: Switch Core Bench Scheduler-MTP Path

**Files:**
- Modify: `ironmlx/src/bin/ironmlx-core-bench.rs`

- [ ] **Step 1: Add bench stats contract test**

Inside the `tests` module in `ironmlx/src/bin/ironmlx-core-bench.rs`, add:

```rust
#[test]
fn scheduler_text_mtp_keeps_scheduler_mode_stats_contract() {
    let record = make_record(
        BenchMode::Scheduler,
        1.0,
        3.0,
        4,
        Some("length"),
        4,
        Some(MtpRecordStats {
            windows: 1,
            drafted_tokens: 2,
            accepted_draft_tokens: 1,
            rollback_count: 1,
            acceptance_rate: Some(0.5),
        }),
    );

    assert_eq!(record.mode, BenchMode::Scheduler);
    assert!(record.valid);
    let stats = record.mtp_stats.expect("scheduler MTP stats");
    assert_eq!(stats.windows, 1);
    assert_eq!(stats.rollback_count, 1);
    assert_eq!(stats.acceptance_rate, Some(0.5));
}
```

Run:

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --bin ironmlx-core-bench scheduler_text_mtp_keeps_scheduler_mode_stats_contract
```

Expected: exits 0. This protects the scheduler-MTP JSON record shape; the functional path switch is verified by the real smoke in Task 5 because a Tokenizer-free fake bench run would duplicate the scheduler tests.

- [ ] **Step 2: Confirm speculative import still includes stream type**

At the top of `ironmlx-core-bench.rs`, keep this import unchanged:

```rust
use ironmlx::core::speculative::{
    MtpSpeculativeConfig, MtpSpeculativeModel, MtpSpeculativeStats, MtpTextGenerationStream,
};
```

`MtpTextGenerationStream` remains required by `mtp-text`, while `scheduler-text --mtp-model-dir` will stop delegating to it.

- [ ] **Step 3: Pass effective cap into scheduler-MTP helper**

In `run_once_qwen`, replace:

```rust
run_scheduler_mtp_single_request(model, mtp, tokenizer, prompt_ids, args)
```

with:

```rust
run_scheduler_mtp_single_request(model, mtp, tokenizer, prompt_ids, args, effective_cap_max)
```

Change the helper signature:

```rust
fn run_scheduler_mtp_single_request<M>(
    model: &M,
    mtp: &M::MtpHead,
    tokenizer: &Tokenizer,
    prompt_ids: &[u32],
    args: &Args,
    effective_cap_max: usize,
) -> Result<Record>
where
    M: MtpSpeculativeModel + DenseVlMethods,
```

- [ ] **Step 4: Replace delegate implementation**

Replace the body of `run_scheduler_mtp_single_request` with:

```rust
{
    let mut scheduler = Scheduler::<M>::new(1, effective_cap_max, model.model_meta())
        .context("Scheduler::new")?;
    let request = make_request(model, tokenizer, prompt_ids, args);
    let cfg = MtpSpeculativeConfig::new(args.mtp_draft_tokens, request.sampler)?;
    let started = Instant::now();
    let _request_id = scheduler.admit(request)?;
    let first_events = scheduler.prefill_admitted_mtp_single(model, mtp, cfg)?;
    let mut generated = first_events.len();
    let mut finish_reason = first_events.first().and_then(|event| event.finish_reason);
    let ttft_ms = started.elapsed().as_secs_f64() * 1000.0;

    while finish_reason.is_none() && generated < args.max_tokens {
        let events = scheduler.step_mtp_single(model, mtp)?;
        if events.is_empty() {
            break;
        }
        generated += events.len();
        finish_reason = events.first().and_then(|event| event.finish_reason);
    }
    mlx::transforms::synchronize()?;
    let e2e_ms = started.elapsed().as_secs_f64() * 1000.0;
    let mtp_stats = scheduler
        .mtp_stats()
        .ok_or_else(|| anyhow!("scheduler MTP run produced no MTP stats"))?
        .into();
    Ok(make_record(
        args.mode,
        ttft_ms,
        e2e_ms,
        generated,
        finish_reason,
        args.max_tokens,
        Some(mtp_stats),
    ))
}
```

- [ ] **Step 5: Run bench unit tests**

Run:

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --bin ironmlx-core-bench mtp
```

Expected: PASS, including existing MTP CLI tests and the new scheduler stats contract test.

- [ ] **Step 6: Commit bench path switch**

Run:

```bash
git add ironmlx/src/bin/ironmlx-core-bench.rs
git commit -m "feat: use scheduler internal mtp in core bench"
```

---

### Task 4: Focused Rust Verification

**Files:**
- No edits expected.

- [ ] **Step 1: Run cargo fmt**

Run:

```bash
cargo fmt
```

Expected: exits 0.

- [ ] **Step 2: Run nightly fmt check**

Run:

```bash
cargo +nightly fmt --all -- --check
```

Expected: exits 0.

- [ ] **Step 3: Run focused scheduler tests**

Run:

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --lib mtp_
```

Expected: exits 0.

- [ ] **Step 4: Run focused bench tests**

Run:

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --bin ironmlx-core-bench mtp
```

Expected: exits 0.

- [ ] **Step 5: Run diff hygiene**

Run:

```bash
git diff --check
```

Expected: exits 0.

---

### Task 5: Full Rust Verification and Real Smoke

**Files:**
- No edits expected unless verification exposes a defect.

- [ ] **Step 1: Run clippy**

Run:

```bash
MLX_DIR=$HOME/.local/mlx cargo +nightly clippy --all-features --workspace -- -D warnings
```

Expected: exits 0. External `mlx-sys` C++ header warnings may appear; Rust/clippy warnings must be absent.

- [ ] **Step 2: Run release build**

Run:

```bash
MLX_DIR=$HOME/.local/mlx cargo build --release
```

Expected: exits 0. External `mlx-sys` C++ header warnings may appear.

- [ ] **Step 3: Ensure smoke prompt exists**

Run:

```bash
test -f /tmp/ironmlx-core-bench-prompt.txt && wc -c /tmp/ironmlx-core-bench-prompt.txt
```

Expected: exits 0 and prints a nonzero byte count for the prompt file. If it exits nonzero, stop before smoke verification and create `/tmp/ironmlx-core-bench-prompt.txt` with this exact one-line content, then rerun this step:

```text
请用一句话介绍 MTP speculative decoding。
```

- [ ] **Step 4: Run real scheduler-internal MTP smoke**

Run:

```bash
target/release/ironmlx-core-bench \
  --model /Users/xin/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3 \
  --prompt-file /tmp/ironmlx-core-bench-prompt.txt \
  --mode scheduler-text \
  --b-max 1 \
  --mtp-model-dir /Users/xin/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MTP-4bit/snapshots/ab6f59bc6627196c611ab8851638651078170485 \
  --max-tokens 16 \
  --runs 1 \
  --warmup-runs 0 \
  --prefill-chunk-size 0 \
  --out /tmp/ironmlx-core-bench-scheduler-mtp-internal-smoke.json
```

Expected: exits 0.

- [ ] **Step 5: Inspect smoke JSON**

Run:

```bash
rg '"mode"|"mtp_draft_tokens"|"mtp_stats"|"acceptance_rate"|"generation_tps"|"valid"' /tmp/ironmlx-core-bench-scheduler-mtp-internal-smoke.json
```

Expected output includes:

```text
"mode": "scheduler-text"
"mtp_draft_tokens": 1
"valid": true
"mtp_stats": {
```

- [ ] **Step 6: Verify batched scheduler rejection still holds**

Run:

```bash
target/release/ironmlx-core-bench \
  --model /Users/xin/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3 \
  --prompt-file /tmp/ironmlx-core-bench-prompt.txt \
  --mode scheduler-text \
  --b-max 2 \
  --mtp-model-dir /Users/xin/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MTP-4bit/snapshots/ab6f59bc6627196c611ab8851638651078170485 \
  --max-tokens 1 \
  --runs 1 \
  --warmup-runs 0 \
  --prefill-chunk-size 0 \
  --out /tmp/ironmlx-core-bench-scheduler-mtp-negative.json 2>&1
```

Expected: exits 1 with:

```text
--mode scheduler-text with --mtp-model-dir currently requires --b-max 1
```

---

### Task 6: Final Commit and Closeout

**Files:**
- Commit any verification-driven edits if Task 4 or Task 5 required fixes.

- [ ] **Step 1: Check status**

Run:

```bash
git status --short
```

Expected: clean if Tasks 2 and 3 commits were sufficient; otherwise only files changed by verification fixes.

- [ ] **Step 2: Commit verification-driven fixes if present**

If `git status --short` shows modified source files after verification fixes, run:

```bash
git add ironmlx/src/core/speculative.rs ironmlx/src/core/scheduler.rs ironmlx/src/bin/ironmlx-core-bench.rs
git commit -m "fix: stabilize scheduler internal mtp smoke"
```

Expected: commit succeeds. Skip this step when the worktree is already clean.

- [ ] **Step 3: Record final commit list**

Run:

```bash
git log --oneline -6
```

Expected: includes these new implementation commits after the plan commit:

```text
feat: add scheduler internal mtp single path
feat: use scheduler internal mtp in core bench
```

- [ ] **Step 4: Final status**

Run:

```bash
git status --short
```

Expected: no output.
