use crate::array::Array;
use crate::error::Result;
use crate::ops;
use crate::stream::Stream;

/// Pad a batch of token sequences to the same length.
/// Returns (padded_tokens [batch, max_len], seq_lengths [batch]).
/// Padding is done on the RIGHT with `pad_token_id`.
pub fn right_pad_sequences(
    sequences: &[&[i32]],
    pad_token_id: i32,
    stream: &Stream,
) -> Result<(Array, Vec<usize>)> {
    let batch_size = sequences.len();
    let max_len = sequences.iter().map(|s| s.len()).max().unwrap_or(0);
    let seq_lengths: Vec<usize> = sequences.iter().map(|s| s.len()).collect();

    // Build padded 2D array [batch, max_len]
    let mut flat = vec![pad_token_id; batch_size * max_len];
    for (i, seq) in sequences.iter().enumerate() {
        for (j, &tok) in seq.iter().enumerate() {
            flat[i * max_len + j] = tok;
        }
    }

    let arr = Array::from_slice_i32(&flat);
    let arr = ops::reshape(&arr, &[batch_size as i32, max_len as i32], stream)?;

    Ok((arr, seq_lengths))
}

/// Create a padding mask: 1 for real tokens, 0 for padding.
/// Returns mask [batch, max_len] as f32.
pub fn create_padding_mask(
    seq_lengths: &[usize],
    max_len: usize,
    _stream: &Stream,
) -> Result<Array> {
    let batch_size = seq_lengths.len();
    let mut mask_data = vec![0.0f32; batch_size * max_len];
    for (i, &len) in seq_lengths.iter().enumerate() {
        for j in 0..len {
            mask_data[i * max_len + j] = 1.0;
        }
    }
    let mask = Array::from_slice_f32_shape(&mask_data, &[batch_size as i32, max_len as i32]);
    Ok(mask)
}

/// Create a combined causal + padding attention mask for batched forward.
/// Returns mask [batch, 1, max_len, max_len] where:
/// - Causal: position i can only attend to positions <= i
/// - Padding: padded positions are masked out (set to -inf)
pub fn create_batched_causal_mask(
    seq_lengths: &[usize],
    max_len: usize,
    stream: &Stream,
) -> Result<Array> {
    let batch_size = seq_lengths.len();
    let neg_inf = f32::NEG_INFINITY;

    // Build [batch, max_len, max_len] mask
    let mut mask_data = vec![neg_inf; batch_size * max_len * max_len];
    for (b, &seq_len) in seq_lengths.iter().enumerate() {
        for i in 0..seq_len {
            for j in 0..=i {
                // Position i can attend to position j (causal + within sequence)
                mask_data[b * max_len * max_len + i * max_len + j] = 0.0;
            }
        }
        // Padding positions (i >= seq_len) stay as -inf (can't attend to anything)
    }

    let mask = Array::from_slice_f32_shape(
        &mask_data,
        &[batch_size as i32, max_len as i32, max_len as i32],
    );
    // Reshape to [batch, 1, max_len, max_len] for attention broadcasting
    let mask = ops::reshape(
        &mask,
        &[batch_size as i32, 1, max_len as i32, max_len as i32],
        stream,
    )?;
    Ok(mask)
}
