use crate::array::Array;
use crate::error::Result;
use crate::ops;
use crate::random;
use crate::stream::Stream;

#[derive(Clone)]
pub struct SamplerConfig {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub repetition_penalty: f32,
    pub seed: Option<u64>,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_p: 1.0,
            top_k: -1,
            repetition_penalty: 1.0,
            seed: None,
        }
    }
}

impl SamplerConfig {
    /// Greedy decoding (temperature=0).
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            ..Default::default()
        }
    }
}

/// Sample a token from logits [batch, vocab_size]. Returns token id array.
pub fn sample(logits: &Array, config: &SamplerConfig, stream: &Stream) -> Result<Array> {
    // Greedy: just argmax
    if config.temperature <= 0.0 {
        return ops::argmax(logits, -1, false, stream);
    }

    // Temperature scaling
    let temp = Array::from_float(config.temperature);
    let mut scaled = ops::divide(logits, &temp, stream)?;

    // Top-K filtering
    if config.top_k > 0 {
        scaled = apply_top_k(&scaled, config.top_k, stream)?;
    }

    // Top-P (nucleus) filtering
    if config.top_p < 1.0 && config.top_p > 0.0 {
        scaled = apply_top_p(&scaled, config.top_p, stream)?;
    }

    // Sample from distribution
    let key = config.seed.map(|s| random::key(s).unwrap());
    random::categorical(&scaled, -1, key.as_ref(), stream)
}

fn apply_top_k(logits: &Array, k: i32, stream: &Stream) -> Result<Array> {
    let top_values = ops::topk(logits, k, -1, stream)?;
    // Get the minimum value in top-k (the threshold)
    let threshold = ops::min(&top_values, &[-1], true, stream)?;
    // where_(logits >= threshold, logits, -inf)
    let neg_inf = Array::from_float(f32::NEG_INFINITY);
    let ge = {
        let mut res = Array::new_empty();
        crate::error::check(unsafe {
            mlx_sys::mlx_greater_equal(
                res.as_raw_mut(),
                logits.as_raw(),
                threshold.as_raw(),
                stream.as_raw(),
            )
        })?;
        res
    };
    ops::where_(&ge, logits, &neg_inf, stream)
}

fn apply_top_p(logits: &Array, p: f32, stream: &Stream) -> Result<Array> {
    // Sort logits in descending order
    let neg_logits = ops::neg(logits, stream)?;

    // Get sorted indices (ascending of negated = descending of original)
    let sorted_indices = ops::argsort_axis(&neg_logits, -1, stream)?;

    // Get sorted logits (descending)
    let sorted_neg = ops::sort_axis(&neg_logits, -1, stream)?;
    let sorted_logits = ops::neg(&sorted_neg, stream)?;

    // Softmax to get probabilities, then cumulative sum
    let probs = ops::softmax(&sorted_logits, &[-1], stream)?;
    let cum_probs = ops::cumsum(&probs, -1, false, true, stream)?;

    // Create mask: cumulative probability > p
    let p_arr = Array::from_float(p);
    let mask = {
        let mut res = Array::new_empty();
        crate::error::check(unsafe {
            mlx_sys::mlx_greater(
                res.as_raw_mut(),
                cum_probs.as_raw(),
                p_arr.as_raw(),
                stream.as_raw(),
            )
        })?;
        res
    };

    // Set masked positions to -inf in sorted logits
    let neg_inf = Array::from_float(f32::NEG_INFINITY);
    let filtered_sorted = ops::where_(&mask, &neg_inf, &sorted_logits, stream)?;

    // Unsort: scatter back to original order
    let unsort_indices = ops::argsort_axis(&sorted_indices, -1, stream)?;

    // Gather filtered values back to original order
    let mut result = Array::new_empty();
    crate::error::check(unsafe {
        mlx_sys::mlx_take_along_axis(
            result.as_raw_mut(),
            filtered_sorted.as_raw(),
            unsort_indices.as_raw(),
            -1,
            stream.as_raw(),
        )
    })?;
    Ok(result)
}
