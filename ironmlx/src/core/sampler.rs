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
}
