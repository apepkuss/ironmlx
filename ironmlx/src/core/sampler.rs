//! Sampler — full pipeline.
//!
//! Pipeline order (each step optional):
//! 1. repetition penalty (divide-by for tokens in history)
//! 2. frequency / presence penalty (subtract `count*freq + presence`)
//! 3. temperature scaling (zero ⇒ greedy short-circuit)
//! 4. top_k mask
//! 5. min_p mask (relative to top-1 prob)
//! 6. top_p (nucleus) mask — coarse surrogate for now (P1 follow-up
//!    tracks tightening once we have an exact gather-along-sorted-axis
//!    primitive)
//! 7. greedy: argmax | sample: categorical(num_samples=1)

use std::cell::Cell;

use mlx::{
    ops::{indexing, reduction, sort, unary, All},
    random, Array,
};

use crate::Result;

/// Configurable sampler. Build via [`Sampler::greedy`] then chain
/// `with_*` setters. The PRNG key is split internally on every
/// [`Sampler::sample`] call so successive calls draw distinct samples.
pub struct Sampler {
    /// Temperature; `<= 0` short-circuits to greedy argmax.
    pub temperature: f32,
    /// Optional top-k filter applied before sampling.
    pub top_k: Option<i32>,
    /// Optional top-p (nucleus) probability mass cap.
    pub top_p: Option<f32>,
    /// Optional min-p relative threshold (vs. top-1 prob).
    pub min_p: Option<f32>,
    /// Optional repetition penalty (divides logits of history tokens).
    pub repetition_penalty: Option<f32>,
    /// Optional frequency penalty (subtracts `count * freq`).
    pub frequency_penalty: Option<f32>,
    /// Optional presence penalty (subtracts `presence` for any token in history).
    pub presence_penalty: Option<f32>,
    /// PRNG seed (used to lazily mint the first key).
    pub seed: u64,
    key: Cell<Option<Array>>,
}

impl std::fmt::Debug for Sampler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Sampler")
            .field("temperature", &self.temperature)
            .field("top_k", &self.top_k)
            .field("top_p", &self.top_p)
            .field("min_p", &self.min_p)
            .field("repetition_penalty", &self.repetition_penalty)
            .field("frequency_penalty", &self.frequency_penalty)
            .field("presence_penalty", &self.presence_penalty)
            .field("seed", &self.seed)
            .finish()
    }
}

impl Clone for Sampler {
    fn clone(&self) -> Self {
        // We snapshot the configuration; the PRNG key state is a
        // runtime detail and is reset to the configured seed for the
        // clone (each generation owns its own stream).
        Self {
            temperature: self.temperature,
            top_k: self.top_k,
            top_p: self.top_p,
            min_p: self.min_p,
            repetition_penalty: self.repetition_penalty,
            frequency_penalty: self.frequency_penalty,
            presence_penalty: self.presence_penalty,
            seed: self.seed,
            key: Cell::new(None),
        }
    }
}

impl Sampler {
    /// Greedy sampler (`temperature = 0`, no penalties / filters).
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_k: None,
            top_p: None,
            min_p: None,
            repetition_penalty: None,
            frequency_penalty: None,
            presence_penalty: None,
            seed: 0,
            key: Cell::new(None),
        }
    }

    /// Set temperature (`<= 0` ⇒ greedy).
    pub fn with_temperature(mut self, t: f32) -> Self {
        self.temperature = t;
        self
    }
    /// Set top-k cutoff. Panics if `k <= 0`.
    pub fn with_top_k(mut self, k: i32) -> Self {
        assert!(k > 0, "top_k must be positive (got {k})");
        self.top_k = Some(k);
        self
    }
    /// Set top-p (nucleus) cumulative probability cap. Panics if `p` is
    /// outside `(0.0, 1.0]`.
    pub fn with_top_p(mut self, p: f32) -> Self {
        assert!(p > 0.0 && p <= 1.0, "top_p must be in (0.0, 1.0] (got {p})");
        self.top_p = Some(p);
        self
    }
    /// Set min-p (relative to top-1 prob). Panics if `p` is outside
    /// `(0.0, 1.0]`.
    pub fn with_min_p(mut self, p: f32) -> Self {
        assert!(p > 0.0 && p <= 1.0, "min_p must be in (0.0, 1.0] (got {p})");
        self.min_p = Some(p);
        self
    }
    /// Set repetition penalty.
    pub fn with_repetition_penalty(mut self, p: f32) -> Self {
        self.repetition_penalty = Some(p);
        self
    }
    /// Set frequency penalty.
    pub fn with_frequency_penalty(mut self, p: f32) -> Self {
        self.frequency_penalty = Some(p);
        self
    }
    /// Set presence penalty.
    pub fn with_presence_penalty(mut self, p: f32) -> Self {
        self.presence_penalty = Some(p);
        self
    }
    /// Set PRNG seed.
    pub fn with_seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }

    /// Returns `true` iff this sampler is in the "default greedy"
    /// configuration: `temperature <= 0` and no penalties / filters
    /// (`top_k`, `top_p`, `min_p`, repetition / frequency / presence
    /// penalty all `None` / zero). Used by [`sample_batch`] (3e.1a) to
    /// pick the vectorized argmax fast path when every active row's
    /// sampler is greedy.
    ///
    /// Distinct from [`is_pipelinable`] which permits non-greedy
    /// temperature as long as penalties are off: pipelined decode only
    /// requires no host-side penalty math, whereas `is_greedy` requires
    /// the full greedy short-circuit at `Sampler::sample` line ~210.
    pub fn is_greedy(&self) -> bool {
        self.temperature <= 0.0
            && self.top_k.is_none()
            && self.top_p.is_none()
            && self.min_p.is_none()
            && self.repetition_penalty.is_none()
            && self.frequency_penalty.is_none()
            && self.presence_penalty.is_none()
    }

    /// Returns `true` iff this sampler can be driven by the pipelined
    /// (async-eval) decode path. The pipelined path requires:
    /// - greedy short-circuit active (`temperature <= 0.0`)
    /// - no repetition / frequency / presence penalty (those force
    ///   `logits.to_vec()` to host, defeating the pipeline).
    ///
    /// Callers that get `false` must use the synchronous [`Sampler::sample`]
    /// path. There is no silent fallback; this predicate is checked
    /// explicitly at `GenerationStream::new` time.
    pub fn is_pipelinable(&self) -> bool {
        self.temperature <= 0.0
            && self.repetition_penalty.is_none()
            && self.frequency_penalty.is_none()
            && self.presence_penalty.is_none()
    }

    /// Greedy-only async sampling. Returns the lazy argmax Array — the caller
    /// is responsible for materialization via `.item()` (or `async_eval` to
    /// pre-dispatch the work for pipelining).
    ///
    /// Returns `Err` if any non-greedy parameter is configured. The caller
    /// must then use [`Sampler::sample`].
    pub fn sample_async_greedy(&self, logits: &Array) -> Result<Array> {
        if !self.is_pipelinable() {
            return Err(anyhow::anyhow!(
                "sample_async_greedy: only greedy (temperature <= 0, no penalties) is supported"
            ));
        }
        // argmax with keepdims=false matches sample()'s greedy short-circuit
        // at line 178 in this same file.
        Ok(reduction::argmax(logits, All, false)?)
    }

    fn ensure_key(&self) -> Result<Array> {
        if let Some(k) = self.key.take() {
            // Took it — split for next call and return one half.
            let (a, b) = random::split(&k)?;
            self.key.set(Some(a));
            return Ok(b);
        }
        let k = random::key(self.seed)?;
        let (a, b) = random::split(&k)?;
        self.key.set(Some(a));
        Ok(b)
    }

    /// Sample a single token id from `logits` (1-D `[vocab]`).
    /// `history` feeds repetition / frequency / presence penalties.
    pub fn sample(&self, logits: &Array, history: &[u32]) -> Result<u32> {
        let mut logits = logits.clone();

        // 1. repetition penalty
        if let Some(p) = self.repetition_penalty {
            if !history.is_empty() && (p - 1.0).abs() > f32::EPSILON {
                logits = apply_repetition_penalty(&logits, history, p)?;
            }
        }

        // 2. frequency / presence penalty
        if self.frequency_penalty.unwrap_or(0.0).abs() > f32::EPSILON
            || self.presence_penalty.unwrap_or(0.0).abs() > f32::EPSILON
        {
            let f = self.frequency_penalty.unwrap_or(0.0);
            let pp = self.presence_penalty.unwrap_or(0.0);
            logits = apply_freq_presence_penalty(&logits, history, f, pp)?;
        }

        // 3. greedy short-circuit
        if self.temperature <= 0.0 {
            let idx = reduction::argmax(&logits, All, false)?;
            return Ok(idx.item::<u32>()?);
        }

        // temperature scaling (scalar-RHS Mul panics on shape error;
        // for finite f32 / 1-D logits this cannot fail)
        let inv_t = 1.0_f32 / self.temperature;
        let mut logits = &logits * inv_t;

        // 4. top_k
        if let Some(k) = self.top_k {
            logits = apply_top_k(&logits, k)?;
        }
        // 5. min_p
        if let Some(p) = self.min_p {
            logits = apply_min_p(&logits, p)?;
        }
        // 6. top_p
        if let Some(p) = self.top_p {
            if p < 1.0 {
                logits = apply_top_p(&logits, p)?;
            }
        }

        // 7. categorical sample
        let key = self.ensure_key()?;
        let sample = random::categorical(&logits)
            .num_samples(1)
            .key(&key)
            .sample()?;
        Ok(sample.item::<u32>()?)
    }
}

/// Batched per-row sampling for `Scheduler::step` and
/// `Scheduler::prefill_admitted_inner` (B1-p2.3e.1a).
///
/// `logits` shape: `[B, vocab]` (caller already collapsed the
/// `[B, 1, vocab]` step output to drop the seq=1 dim).
/// `samplers` and `histories` must be length `B`, indexed in row
/// order. Each row's sampler is cloned per-request at admit time
/// (`RequestState::sampler`) so this borrow does not contend with
/// concurrent admits.
///
/// Returns `[B]` `Vec<u32>` of sampled token ids, one per row.
///
/// # Routing (spec §4.1)
/// - **All-greedy fast path** (every `samplers[b].is_greedy()`):
///   single `argmax(logits, axis=-1)` GPU dispatch → one
///   `.to_vec::<u32>()` host transfer for the whole batch. Replaces
///   B sequential `.item()` syncs (~1-3 ms each) with one
///   coalesced dispatch (~1-2 ms total). 3-4× per-step sampler
///   speedup at B=4.
/// - **Mixed / configured fallback** (any row not greedy): per-row
///   loop calling `Sampler::sample` exactly as the pre-3e.1a step
///   did. 3e.1b extends this fallback to vectorize temperature /
///   top-p / repetition penalty; top-k remains per-row pending a
///   custom Metal partial-sort kernel.
///
/// # Errors
/// - `samplers.len() != B` or `histories.len() != B` (`B` is
///   `logits.shape()[0]`).
/// - `logits` is not 2-D `[B, vocab]`.
/// - Underlying MLX argmax / `.to_vec` failures bubble up.
pub fn sample_batch(
    samplers: &[&Sampler],
    logits: &Array,
    histories: &[&[u32]],
) -> Result<Vec<u32>> {
    let shape = logits.shape();
    let dims = shape.as_slice();
    if dims.len() != 2 {
        anyhow::bail!(
            "sample_batch: logits must be 2-D [B, vocab]; got shape {:?}",
            dims
        );
    }
    let b = dims[0] as usize;
    if samplers.len() != b {
        anyhow::bail!("sample_batch: samplers.len()={} != B={}", samplers.len(), b);
    }
    if histories.len() != b {
        anyhow::bail!(
            "sample_batch: histories.len()={} != B={}",
            histories.len(),
            b
        );
    }

    // All-greedy fast path.
    if samplers.iter().all(|s| s.is_greedy()) {
        // `argmax(logits, axis=-1, keepdims=false)` over [B, vocab]
        // returns [B] u32 indices. One GPU dispatch, one host sync.
        let ids = reduction::argmax(logits, -1, false)?;
        let tokens: Vec<u32> = ids.to_vec()?;
        if tokens.len() != b {
            anyhow::bail!(
                "sample_batch: argmax returned {} tokens, expected B={}",
                tokens.len(),
                b
            );
        }
        return Ok(tokens);
    }

    // Mixed / configured fallback: per-row sequential. 3e.1b will
    // vectorize this for non-top-k configs.
    let mut tokens = Vec::with_capacity(b);
    for (i, sampler) in samplers.iter().enumerate() {
        // Slice row i out of [B, vocab] into [1, vocab] then reshape
        // to [vocab] for Sampler::sample.
        let row = indexing::slice_strided_on(
            logits,
            &[i as i32, 0_i32][..],
            &[i as i32 + 1, dims[1]][..],
            &[1_i32, 1_i32][..],
            (),
        )?;
        let row_flat = row.reshape(&[dims[1]][..])?;
        tokens.push(sampler.sample(&row_flat, histories[i])?);
    }
    Ok(tokens)
}

/// Per-row config tensors used by [`configured_pipeline`]. Each
/// field is shape `[B]`; no-op defaults make the corresponding op
/// behave as identity for rows that don't need it.
#[allow(dead_code)]
struct PerRowConfigs {
    /// `[B] f32`. None → 1.0 (identity divisor).
    temp: Array,
    /// `[B] i32`. None → `vocab_size` (no clip).
    top_k: Array,
    /// `[B] f32`. None → 1.0 (no nucleus cut).
    top_p: Array,
    /// `[B] f32`. None → 0.0 (no min_p floor).
    min_p: Array,
    /// `[B] f32`. None → 1.0 (no repetition penalty).
    rep_pen: Array,
    /// `[B] f32`. None → 0.0 (no frequency penalty).
    freq_pen: Array,
    /// `[B] f32`. None → 0.0 (no presence penalty).
    pres_pen: Array,
    /// True if any row has rep_pen / freq_pen / pres_pen set —
    /// drives the history-bincount short-circuit.
    need_history: bool,
}

#[allow(dead_code)]
fn collect_per_row_configs(samplers: &[&Sampler], vocab: i32) -> Result<PerRowConfigs> {
    let b = samplers.len();
    let mut temp = Vec::with_capacity(b);
    let mut top_k = Vec::with_capacity(b);
    let mut top_p = Vec::with_capacity(b);
    let mut min_p = Vec::with_capacity(b);
    let mut rep_pen = Vec::with_capacity(b);
    let mut freq_pen = Vec::with_capacity(b);
    let mut pres_pen = Vec::with_capacity(b);
    let mut need_history = false;
    for s in samplers {
        // temperature: <=0 means greedy in per-row Sampler::sample, but
        // configured_pipeline is only entered when batch is mixed.
        // Greedy rows in a mixed batch use temp=1.0 (no-op).
        temp.push(if s.temperature > 0.0 {
            s.temperature
        } else {
            1.0
        });
        top_k.push(s.top_k.unwrap_or(vocab));
        top_p.push(s.top_p.unwrap_or(1.0));
        min_p.push(s.min_p.unwrap_or(0.0));
        rep_pen.push(s.repetition_penalty.unwrap_or(1.0));
        freq_pen.push(s.frequency_penalty.unwrap_or(0.0));
        pres_pen.push(s.presence_penalty.unwrap_or(0.0));
        if s.repetition_penalty.is_some()
            || s.frequency_penalty.is_some()
            || s.presence_penalty.is_some()
        {
            need_history = true;
        }
    }
    let dim = &[b as i32][..];
    Ok(PerRowConfigs {
        temp: (&temp[..], dim).try_into()?,
        top_k: (&top_k[..], dim).try_into()?,
        top_p: (&top_p[..], dim).try_into()?,
        min_p: (&min_p[..], dim).try_into()?,
        rep_pen: (&rep_pen[..], dim).try_into()?,
        freq_pen: (&freq_pen[..], dim).try_into()?,
        pres_pen: (&pres_pen[..], dim).try_into()?,
        need_history,
    })
}

/// Build `[B, vocab] u32` count tensor from per-row histories. CPU
/// bincount → device upload.
#[allow(dead_code)]
fn build_history_count(histories: &[&[u32]], vocab: usize) -> Result<Array> {
    let b = histories.len();
    let mut flat = vec![0_u32; b * vocab];
    for (row, hist) in histories.iter().enumerate() {
        let offset = row * vocab;
        for &tok in *hist {
            let idx = tok as usize;
            if idx < vocab {
                flat[offset + idx] = flat[offset + idx].saturating_add(1);
            }
        }
    }
    let arr: Array = (&flat[..], &[b as i32, vocab as i32][..]).try_into()?;
    Ok(arr)
}

/// Apply repetition + frequency + presence penalties as a single
/// fused op over `[B, vocab]` logits. Returns updated logits (or
/// `logits.clone()` if `history_count.is_none()`).
#[allow(dead_code)]
fn apply_penalties(
    logits: &Array,
    history_count: Option<&Array>,
    configs: &PerRowConfigs,
) -> Result<Array> {
    let Some(history_count) = history_count else {
        return Ok(logits.clone());
    };
    let history_count_f32 = history_count.astype(mlx::Dtype::Float32)?;
    let zero_u32: Array = (&[0_u32][..], ()).try_into()?;
    let history_mask_bool = mlx::ops::binary::greater(history_count, &zero_u32)?;
    let history_mask_f32 = history_mask_bool.astype(mlx::Dtype::Float32)?;

    let b = logits.shape().as_slice()[0];
    let rep_pen_bv = configs.rep_pen.reshape(&[b, 1][..])?;
    let freq_pen_bv = configs.freq_pen.reshape(&[b, 1][..])?;
    let pres_pen_bv = configs.pres_pen.reshape(&[b, 1][..])?;

    // Repetition: where(logit > 0, logit / rep_pen, logit * rep_pen) for seen tokens
    let one_f32: Array = (&[1.0_f32][..], ()).try_into()?;
    let rep_inv_bv = &one_f32 / &rep_pen_bv;
    let zero_f32: Array = (&[0.0_f32][..], ()).try_into()?;
    let positive_logit_mask = mlx::ops::binary::greater(logits, &zero_f32)?;
    let rep_factor = indexing::where_(&positive_logit_mask, &rep_inv_bv, &rep_pen_bv)?;
    let logits_rep_full = logits * &rep_factor;
    let logits_rep = indexing::where_(&history_mask_bool, &logits_rep_full, logits)?;

    // Frequency: logit -= freq_pen * count
    let freq_term = &freq_pen_bv * &history_count_f32;
    let logits_freq = &logits_rep - &freq_term;

    // Presence: logit -= pres_pen * (history_mask as f32)
    let pres_term = &pres_pen_bv * &history_mask_f32;
    let logits_pres = &logits_freq - &pres_term;

    Ok(logits_pres)
}

/// Configured-sampler vectorized pipeline. Called by [`sample_batch`]
/// when not all rows are greedy. See spec
/// `docs/superpowers/specs/2026-05-17-b1-p2-3e-1b-vectorize-configured-sampler-design.md`.
///
/// **mlx API verification (T0, plan §Step 0):**
/// - `mlx::random::categorical(logits=[B,vocab]).key(&single_key).sample() → [B]`
///   — single key + automatic row-independent batching. Per-row PRNG
///   reproducibility (each Sampler having its own seed) is NOT preserved
///   by the batched op; spec NG6 accepts this drift.
/// - `mlx::ops::sort::partition(kth, axis)` and `sort(axis)` both exist;
///   plan T0 §0.1 bench result: sort([B=4,vocab=151936]) measured 11.84 ms
///   (> 3 ms threshold) vs partition(kth=151886) measured 1.15 ms. Top_k
///   path chosen: partition(kth = vocab - top_k_max, axis=-1) (R2 mitigation,
///   sort exceeded threshold).
/// - `scatter_along_axis` not exposed in mlx Rust binding; top_p scatter
///   back uses `argsort(sort_idx) = inverse permutation` then
///   `take_along_axis(sorted_masked, inv_perm, -1)`. Verified in
///   `probe_argsort_inverse_permutation_identity`. Note: `argsort` returns
///   `Uint32`; `to_vec::<u32>()` must be used (not `i32`).
#[allow(dead_code)]
fn configured_pipeline(
    samplers: &[&Sampler],
    logits: &Array,
    histories: &[&[u32]],
) -> Result<Vec<u32>> {
    let _ = (samplers, logits, histories);
    anyhow::bail!("configured_pipeline: not yet implemented (3e.1b T1-T3)")
}

fn apply_repetition_penalty(logits: &Array, history: &[u32], p: f32) -> Result<Array> {
    // For each token id in history, scale `logits[id]` by `1/p` if the
    // logit is positive, by `p` if negative — this matches the HF
    // canonical implementation. We materialise a multiplier vector on
    // the host because cxx-mlx's scatter coverage is not yet strong
    // enough to do this fully on-device. Acceptable for P1 (vocab is
    // O(150k); cost ≪ a forward pass).
    let v: Vec<f32> = logits.to_vec()?;
    let mut mul = vec![1.0_f32; v.len()];
    for &t in history {
        let i = t as usize;
        if i >= v.len() {
            continue;
        }
        mul[i] = if v[i] > 0.0 { 1.0 / p } else { p };
    }
    let mul_arr: Array = (&mul[..], (mul.len() as i32,)).try_into()?;
    Ok(logits * &mul_arr)
}

fn apply_freq_presence_penalty(
    logits: &Array,
    history: &[u32],
    freq: f32,
    presence: f32,
) -> Result<Array> {
    let v: Vec<f32> = logits.to_vec()?;
    let mut counts = vec![0_u32; v.len()];
    for &t in history {
        if (t as usize) < counts.len() {
            counts[t as usize] += 1;
        }
    }
    let mut sub = vec![0.0_f32; v.len()];
    for (i, &c) in counts.iter().enumerate() {
        if c > 0 {
            sub[i] = c as f32 * freq + presence;
        }
    }
    let sub_arr: Array = (&sub[..], (sub.len() as i32,)).try_into()?;
    Ok(logits - &sub_arr)
}

/// Mask logits below the k-th largest. Ties at the k-th position are
/// excluded (strict `<` matches the `mask` semantics), so the output
/// may have fewer than `k` surviving tokens when duplicates exist at
/// the boundary.
fn apply_top_k(logits: &Array, k: i32) -> Result<Array> {
    // Sort ascending; cut threshold is `sorted[len - k]`.
    let sorted = sort::sort(logits, -1)?;
    let v_len = sorted.shape().as_slice().last().copied().unwrap_or(0);
    let cut_idx = (v_len - k).max(0);
    let threshold = sorted.slice((cut_idx,), (cut_idx + 1,))?;
    let neg_inf: Array = (&[f32::NEG_INFINITY][..], (1,)).try_into()?;
    let mask = mlx::ops::binary::less(logits, &threshold)?;
    Ok(indexing::where_(&mask, &neg_inf, logits)?)
}

fn apply_min_p(logits: &Array, p: f32) -> Result<Array> {
    let probs = unary::softmax(logits, All, false)?;
    let max_p = reduction::max(&probs, All, true)?;
    let threshold = &max_p * p;
    let mask = mlx::ops::binary::less(&probs, &threshold)?;
    let neg_inf: Array = (&[f32::NEG_INFINITY][..], (1,)).try_into()?;
    Ok(indexing::where_(&mask, &neg_inf, logits)?)
}

fn apply_top_p(logits: &Array, p: f32) -> Result<Array> {
    // MVP coarse surrogate (P1 follow-up tracks exact nucleus): mask
    // tokens whose individual softmax prob is below `(1 - p) / vocab`.
    // Exact nucleus needs gather-along-sorted-axis to translate the
    // sorted cumulative cut back onto the original positions, which
    // exceeds cxx-mlx's current scatter primitives.
    let probs = unary::softmax(logits, All, false)?;
    let vocab = probs.size() as f32;
    let threshold_scalar = (1.0_f32 - p) / vocab;
    let threshold: Array = (&[threshold_scalar][..], (1,)).try_into()?;
    let mask = mlx::ops::binary::less(&probs, &threshold)?;
    let neg_inf: Array = (&[f32::NEG_INFINITY][..], (1,)).try_into()?;
    Ok(indexing::where_(&mask, &neg_inf, logits)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx::Dtype;

    #[test]
    fn greedy_picks_argmax() {
        let logits: Array = (&[0.1_f32, 5.0, 2.0, -1.0][..], (4,)).try_into().unwrap();
        let s = Sampler::greedy();
        let id = s.sample(&logits, &[]).unwrap();
        assert_eq!(id, 1);
    }

    #[test]
    fn temperature_zero_is_greedy() {
        let logits: Array = (&[0.1_f32, 5.0, 2.0][..], (3,)).try_into().unwrap();
        let s = Sampler::greedy().with_temperature(0.0);
        assert_eq!(s.sample(&logits, &[]).unwrap(), 1);
    }

    #[test]
    fn repetition_penalty_demotes_history_tokens() {
        // Token 0 has the highest logit; with high repetition penalty,
        // picking 0 again should be suppressed in favour of token 1.
        let logits: Array = (&[5.0_f32, 4.0, 3.0][..], (3,)).try_into().unwrap();
        let s = Sampler::greedy().with_repetition_penalty(10.0);
        let id = s.sample(&logits, &[0]).unwrap();
        assert_eq!(id, 1);
    }

    #[test]
    fn temperature_sample_runs() {
        let logits = Array::zeros((10,), Dtype::Float32).unwrap();
        let s = Sampler::greedy()
            .with_temperature(1.0)
            .with_top_p(0.9)
            .with_seed(42);
        let id = s.sample(&logits, &[]).unwrap();
        assert!((id as i32) < 10);
    }

    #[test]
    fn is_pipelinable_accepts_greedy() {
        assert!(Sampler::greedy().is_pipelinable());
    }

    #[test]
    fn is_pipelinable_rejects_temperature() {
        assert!(!Sampler::greedy().with_temperature(0.7).is_pipelinable());
    }

    #[test]
    fn is_pipelinable_rejects_repetition_penalty() {
        assert!(!Sampler::greedy()
            .with_repetition_penalty(1.1)
            .is_pipelinable());
    }

    #[test]
    fn is_pipelinable_rejects_frequency_penalty() {
        assert!(!Sampler::greedy()
            .with_frequency_penalty(0.5)
            .is_pipelinable());
    }

    #[test]
    fn is_pipelinable_rejects_presence_penalty() {
        assert!(!Sampler::greedy()
            .with_presence_penalty(0.5)
            .is_pipelinable());
    }

    #[test]
    fn sample_async_greedy_returns_lazy_array_with_correct_token() {
        // Construct a [vocab=8] f32 Array with the max at index 3.
        let logits_data: Vec<f32> = vec![0.1, 0.2, 0.3, 5.0, 0.4, 0.5, 0.6, 0.7];
        let logits: mlx::Array = (logits_data.as_slice(), &[8_i32][..])
            .try_into()
            .expect("build logits array");

        let s = Sampler::greedy();
        let result = s.sample_async_greedy(&logits).expect("sample_async_greedy");

        // Pin the shape contract: argmax with keepdims=false returns either
        // 0-D scalar or [1]. Either is fine for the pipeline (Task 3 will
        // .reshape((1,1)) which accepts both), but the test asserts the
        // actual shape to surface any future MLX wrapper changes.
        let shape = result.shape();
        let shape_slice = shape.as_slice();
        assert!(
            shape_slice.is_empty() || shape_slice == [1_i32],
            "unexpected shape from argmax(All, keepdims=false): {shape_slice:?}"
        );

        // Materialise to confirm correct value.
        let token: u32 = result.item().expect("item");
        assert_eq!(token, 3, "expected argmax index 3, got {token}");
    }

    #[test]
    fn sample_async_greedy_rejects_temperature() {
        let logits_data: Vec<f32> = vec![0.1_f32; 4];
        let logits: mlx::Array = (logits_data.as_slice(), &[4_i32][..])
            .try_into()
            .expect("build logits array");

        let s = Sampler::greedy().with_temperature(0.7);
        let r = s.sample_async_greedy(&logits);
        assert!(
            r.is_err(),
            "non-greedy temperature must reject async-greedy path"
        );
    }

    #[test]
    fn sample_async_greedy_rejects_penalty() {
        let logits_data: Vec<f32> = vec![0.1_f32; 4];
        let logits: mlx::Array = (logits_data.as_slice(), &[4_i32][..])
            .try_into()
            .expect("build logits array");

        let s = Sampler::greedy().with_repetition_penalty(1.1);
        let r = s.sample_async_greedy(&logits);
        assert!(
            r.is_err(),
            "repetition_penalty must reject async-greedy path"
        );
    }

    // ── is_greedy tests ──────────────────────────────────────────────

    #[test]
    fn is_greedy_true_for_default() {
        let s = Sampler::greedy();
        assert!(s.is_greedy());
    }

    #[test]
    fn is_greedy_false_when_temperature_set() {
        let s = Sampler::greedy().with_temperature(0.7);
        assert!(!s.is_greedy());
    }

    #[test]
    fn is_greedy_false_when_top_p_set() {
        let s = Sampler::greedy().with_top_p(0.9);
        assert!(!s.is_greedy());
    }

    #[test]
    fn is_greedy_false_when_repetition_penalty_set() {
        let s = Sampler::greedy().with_repetition_penalty(1.1);
        assert!(!s.is_greedy());
    }

    // ── sample_batch tests ───────────────────────────────────────────

    fn make_logits_b_vocab(b: usize, vocab: usize, max_at_per_row: &[usize]) -> Array {
        // Build a [B, vocab] f32 Array with row i's argmax at column max_at_per_row[i].
        assert_eq!(b, max_at_per_row.len(), "max indices must match B");
        let mut flat: Vec<f32> = vec![0.0; b * vocab];
        for (i, &max_col) in max_at_per_row.iter().enumerate() {
            flat[i * vocab + max_col] = 100.0;
        }
        let arr: Array = (&flat[..], &[b as i32, vocab as i32][..])
            .try_into()
            .expect("logits Array");
        arr
    }

    #[test]
    fn sample_batch_all_greedy_returns_per_row_argmax() {
        let samplers_owned: Vec<Sampler> = (0..4).map(|_| Sampler::greedy()).collect();
        let samplers: Vec<&Sampler> = samplers_owned.iter().collect();
        let logits = make_logits_b_vocab(4, 64, &[3, 7, 17, 63]);
        let histories: Vec<&[u32]> = vec![&[], &[], &[], &[]];
        let tokens = sample_batch(&samplers, &logits, &histories).expect("sample_batch greedy");
        assert_eq!(tokens, vec![3, 7, 17, 63]);
    }

    #[test]
    fn sample_batch_b1_greedy() {
        // B=1 edge case.
        let s = Sampler::greedy();
        let samplers = vec![&s];
        let logits = make_logits_b_vocab(1, 32, &[15]);
        let histories: Vec<&[u32]> = vec![&[]];
        let tokens = sample_batch(&samplers, &logits, &histories).expect("sample_batch B=1");
        assert_eq!(tokens, vec![15]);
    }

    #[test]
    fn sample_batch_mismatched_samplers_errs() {
        let s = Sampler::greedy();
        let samplers = vec![&s, &s]; // 2 samplers
        let logits = make_logits_b_vocab(4, 32, &[0, 1, 2, 3]); // B=4
        let histories: Vec<&[u32]> = vec![&[]; 2];
        let r = sample_batch(&samplers, &logits, &histories);
        assert!(r.is_err());
        let msg = format!("{}", r.unwrap_err());
        assert!(msg.contains("samplers.len()"), "msg: {msg}");
    }

    #[test]
    fn sample_batch_mismatched_histories_errs() {
        let s = Sampler::greedy();
        let samplers = vec![&s, &s, &s, &s];
        let logits = make_logits_b_vocab(4, 32, &[0, 1, 2, 3]);
        let histories: Vec<&[u32]> = vec![&[], &[]]; // 2 histories
        let r = sample_batch(&samplers, &logits, &histories);
        assert!(r.is_err());
        let msg = format!("{}", r.unwrap_err());
        assert!(msg.contains("histories.len()"), "msg: {msg}");
    }

    #[test]
    fn sample_batch_3d_logits_errs() {
        let s = Sampler::greedy();
        let samplers = vec![&s];
        // 3D logits [1, 1, 32] — caller should slice to 2D first.
        let flat: Vec<f32> = vec![0.0; 32];
        let logits: Array = (&flat[..], &[1_i32, 1_i32, 32_i32][..]).try_into().unwrap();
        let histories: Vec<&[u32]> = vec![&[]];
        let r = sample_batch(&samplers, &logits, &histories);
        assert!(r.is_err());
        let msg = format!("{}", r.unwrap_err());
        assert!(msg.contains("2-D"), "msg: {msg}");
    }

    #[test]
    fn probe_categorical_batched_single_key_independent_rows() {
        use mlx::random;
        // Build [B=4, vocab=8] logits where each row has its argmax at a different col.
        let mut data: Vec<f32> = vec![0.0; 32];
        for i in 0..4 {
            data[i * 8 + i] = 100.0;
        } // row i argmax at col i
        let logits: Array = (&data[..], &[4_i32, 8_i32][..]).try_into().expect("logits");
        let key = random::key(42).expect("key");
        let tokens = random::categorical(&logits)
            .key(&key)
            .sample()
            .expect("sample");
        assert_eq!(
            tokens.shape().as_slice(),
            &[4],
            "categorical([B,vocab]) → [B]"
        );
        let v: Vec<u32> = tokens.to_vec().expect("to_vec");
        // Each row's argmax dominates → categorical concentrates on that col.
        assert_eq!(
            v,
            vec![0, 1, 2, 3],
            "row i should sample col i (skewed logits)"
        );
    }

    #[test]
    #[ignore] // bench-mode, run on demand
    fn probe_sort_vs_partition_vocab_151k() {
        use mlx::ops::sort;
        use std::time::Instant;
        let b = 4usize;
        let vocab = 151936usize;
        let data: Vec<f32> = (0..b * vocab).map(|i| (i as f32).sin()).collect();
        let arr: Array = (&data[..], &[b as i32, vocab as i32][..])
            .try_into()
            .unwrap();
        arr.eval().unwrap();

        let t0 = Instant::now();
        let sorted = sort::sort(&arr, -1).unwrap();
        sorted.eval().unwrap();
        let dt_sort = t0.elapsed();

        let t1 = Instant::now();
        let parted = sort::partition(&arr, (vocab - 50) as i32, -1).unwrap();
        parted.eval().unwrap();
        let dt_part = t1.elapsed();

        eprintln!(
            "[T0 bench] sort=[B=4,vocab=151936] {dt_sort:?} | partition(kth=151886) {dt_part:?}"
        );
    }

    #[test]
    fn probe_argsort_inverse_permutation_identity() {
        use mlx::ops::{indexing, sort};
        let b = 2usize;
        let vocab = 8usize;
        let data: Vec<f32> = vec![
            0.1, 0.05, 0.2, 0.3, 0.05, 0.1, 0.1, 0.1, // row 0
            0.2, 0.15, 0.1, 0.05, 0.1, 0.15, 0.15, 0.1, // row 1
        ];
        let probs: Array = (&data[..], &[b as i32, vocab as i32][..])
            .try_into()
            .unwrap();
        let idx = sort::argsort(&probs, -1).unwrap();
        let inv = sort::argsort(&idx, -1).unwrap();
        // For each row: take_along_axis(idx, inv, -1) should produce arange(vocab) per row
        let got = indexing::take_along_axis(&idx, &inv, -1).unwrap();
        let got_v: Vec<u32> = got.to_vec().unwrap();
        let expected: Vec<u32> = (0..vocab as u32).chain(0..vocab as u32).collect();
        assert_eq!(got_v, expected, "inverse permutation identity failed");
    }

    #[test]
    fn sample_batch_configured_fallback_matches_per_row() {
        // B=4 where ONE row has temperature → mixed batch → fallback.
        // Use Sampler::sample with fixed seed for deterministic compare.
        let s_greedy = Sampler::greedy();
        let s_temp = Sampler::greedy().with_temperature(0.7).with_seed(42);
        let samplers: Vec<&Sampler> = vec![&s_greedy, &s_temp, &s_greedy, &s_greedy];
        let logits = make_logits_b_vocab(4, 32, &[5, 10, 15, 20]);
        let histories: Vec<&[u32]> = vec![&[], &[], &[], &[]];

        // Vectorized batch path (will take fallback because s_temp not greedy).
        let tokens_batch =
            sample_batch(&samplers, &logits, &histories).expect("sample_batch mixed");

        // Per-row reference using fresh samplers (Sampler is !Clone-safe across
        // PRNG state; rebuild the with_seed(42) one to get the same key).
        let s_temp_ref = Sampler::greedy().with_temperature(0.7).with_seed(42);
        let mut tokens_ref: Vec<u32> = Vec::with_capacity(4);
        for (i, expected_argmax) in [5_usize, 10, 15, 20].iter().enumerate() {
            let row = indexing::slice_strided_on(
                &logits,
                &[i as i32, 0_i32][..],
                &[i as i32 + 1, 32_i32][..],
                &[1_i32, 1_i32][..],
                (),
            )
            .unwrap();
            let row_flat = row.reshape(&[32_i32][..]).unwrap();
            let s_ref = if i == 1 { &s_temp_ref } else { &s_greedy };
            tokens_ref.push(s_ref.sample(&row_flat, &[]).unwrap());
            // Greedy rows must produce their argmax index.
            if i != 1 {
                assert_eq!(tokens_ref[i] as usize, *expected_argmax);
            }
        }

        assert_eq!(tokens_batch, tokens_ref);
    }

    // ── PerRowConfigs / collect_per_row_configs tests ─────────────────

    #[test]
    fn collect_per_row_configs_defaults_and_overrides() {
        let s1 = Sampler::greedy().with_temperature(0.7);
        let s2 = Sampler::greedy()
            .with_top_p(0.9)
            .with_repetition_penalty(1.1);
        let s3 = Sampler::greedy();
        let samplers: Vec<&Sampler> = vec![&s1, &s2, &s3];
        let cfg = collect_per_row_configs(&samplers, 32000).expect("collect");
        let temp: Vec<f32> = cfg.temp.to_vec().expect("temp vec");
        assert_eq!(temp, vec![0.7, 1.0, 1.0]);
        let top_p: Vec<f32> = cfg.top_p.to_vec().expect("top_p vec");
        assert_eq!(top_p, vec![1.0, 0.9, 1.0]);
        let rep: Vec<f32> = cfg.rep_pen.to_vec().expect("rep vec");
        assert_eq!(rep, vec![1.0, 1.1, 1.0]);
        let top_k: Vec<i32> = cfg.top_k.to_vec().expect("top_k vec");
        assert_eq!(top_k, vec![32000, 32000, 32000]);
        assert!(cfg.need_history);
    }

    // ── build_history_count tests ────────────────────────────────────

    #[test]
    fn build_history_count_per_row_bincount() {
        let h0: &[u32] = &[3, 3, 5];
        let h1: &[u32] = &[7];
        let h2: &[u32] = &[];
        let histories: Vec<&[u32]> = vec![h0, h1, h2];
        let counts = build_history_count(&histories, 8).expect("counts");
        let v: Vec<u32> = counts.to_vec().expect("to_vec");
        assert_eq!(&v[0..8], &[0, 0, 0, 2, 0, 1, 0, 0]);
        assert_eq!(&v[8..16], &[0, 0, 0, 0, 0, 0, 0, 1]);
        assert_eq!(&v[16..24], &[0; 8]);
    }

    // ── apply_penalties tests ────────────────────────────────────────

    fn make_logits(b: usize, vocab: usize, fill: f32) -> Array {
        let v: Vec<f32> = vec![fill; b * vocab];
        (&v[..], &[b as i32, vocab as i32][..])
            .try_into()
            .expect("logits")
    }

    #[test]
    fn apply_penalties_repetition_divides_seen_when_positive() {
        let logits = make_logits(1, 8, 2.0);
        let h0: &[u32] = &[5];
        let s = Sampler::greedy().with_repetition_penalty(2.0);
        let samplers: Vec<&Sampler> = vec![&s];
        let cfg = collect_per_row_configs(&samplers, 8).expect("cfg");
        let history_count = build_history_count(&[h0], 8).expect("hc");
        let out = apply_penalties(&logits, Some(&history_count), &cfg).expect("out");
        let v: Vec<f32> = out.to_vec().expect("to_vec");
        assert!(
            (v[5] - 1.0).abs() < 1e-5,
            "row 0 token 5 should be 2.0/2.0=1.0; got {}",
            v[5]
        );
        assert!(
            (v[0] - 2.0).abs() < 1e-5,
            "row 0 token 0 unseen → unchanged"
        );
    }

    #[test]
    fn apply_penalties_frequency_subtracts_count_times_penalty() {
        let logits = make_logits(1, 8, 5.0);
        let h0: &[u32] = &[3, 3, 3];
        let s = Sampler::greedy().with_frequency_penalty(1.5);
        let samplers: Vec<&Sampler> = vec![&s];
        let cfg = collect_per_row_configs(&samplers, 8).expect("cfg");
        let history_count = build_history_count(&[h0], 8).expect("hc");
        let out = apply_penalties(&logits, Some(&history_count), &cfg).expect("out");
        let v: Vec<f32> = out.to_vec().expect("to_vec");
        assert!(
            (v[3] - 0.5).abs() < 1e-4,
            "row 0 token 3 should be 5.0-4.5=0.5; got {}",
            v[3]
        );
        assert!((v[0] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn apply_penalties_presence_subtracts_once_per_token() {
        let logits = make_logits(1, 8, 5.0);
        let h0: &[u32] = &[3, 3, 3];
        let s = Sampler::greedy().with_presence_penalty(1.5);
        let samplers: Vec<&Sampler> = vec![&s];
        let cfg = collect_per_row_configs(&samplers, 8).expect("cfg");
        let history_count = build_history_count(&[h0], 8).expect("hc");
        let out = apply_penalties(&logits, Some(&history_count), &cfg).expect("out");
        let v: Vec<f32> = out.to_vec().expect("to_vec");
        assert!(
            (v[3] - 3.5).abs() < 1e-4,
            "row 0 token 3 should be 5.0-1.5=3.5; got {}",
            v[3]
        );
        assert!((v[0] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn apply_penalties_short_circuit_when_no_history_needed() {
        let logits = make_logits(2, 8, 5.0);
        let s1 = Sampler::greedy().with_temperature(0.7);
        let s2 = Sampler::greedy().with_top_p(0.9);
        let samplers: Vec<&Sampler> = vec![&s1, &s2];
        let cfg = collect_per_row_configs(&samplers, 8).expect("cfg");
        assert!(!cfg.need_history);
        let out = apply_penalties(&logits, None, &cfg).expect("out");
        let v: Vec<f32> = out.to_vec().expect("to_vec");
        assert!(v.iter().all(|&x| (x - 5.0).abs() < 1e-5));
    }
}
