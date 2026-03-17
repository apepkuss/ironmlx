use crate::array::Array;
use crate::dtype::Dtype;
use crate::error::Result;
use crate::ops;
use crate::stream::Stream;
use crate::vector::VectorArray;

/// Compute M-RoPE position IDs for mixed text + vision tokens.
/// Returns position_ids: `Vec<[i32; 3]>` where:
///   - Index 0: height positions
///   - Index 1: width positions
///   - Index 2: temporal positions
/// For text tokens, all 3 values have the same sequential position.
/// For image tokens, each value has the corresponding spatial/temporal coordinate.
pub fn compute_mrope_position_ids(
    tokens: &[i32],
    grid_thw: &[(usize, usize, usize)],
    image_token_id: i64,
    video_token_id: i64,
) -> Vec<[i32; 3]> {
    let seq_len = tokens.len();
    let mut position_ids = Vec::with_capacity(seq_len);
    let mut text_pos = 0i32;
    let mut grid_idx = 0;

    let mut i = 0;
    while i < seq_len {
        let token = tokens[i] as i64;

        if (token == image_token_id || token == video_token_id) && grid_idx < grid_thw.len() {
            let (t, h, w) = grid_thw[grid_idx];

            // Assign spatial/temporal positions for this image's patches
            for frame in 0..t {
                for row in 0..h {
                    for col in 0..w {
                        if i < seq_len {
                            position_ids.push([row as i32, col as i32, frame as i32]);
                            i += 1;
                        }
                    }
                }
            }

            grid_idx += 1;
            text_pos = position_ids
                .last()
                .map(|p| p[0].max(p[1]).max(p[2]))
                .unwrap_or(0)
                + 1;
        } else {
            position_ids.push([text_pos, text_pos, text_pos]);
            text_pos += 1;
            i += 1;
        }
    }

    position_ids
}

/// Convert M-RoPE position IDs to an Array of shape [3, L].
pub fn position_ids_to_array(position_ids: &[[i32; 3]], stream: &Stream) -> Result<Array> {
    let l = position_ids.len();
    // Layout: [3, L] in row-major order
    let mut data = vec![0i32; 3 * l];
    for (j, ids) in position_ids.iter().enumerate() {
        data[j] = ids[0]; // row 0: height
        data[l + j] = ids[1]; // row 1: width
        data[2 * l + j] = ids[2]; // row 2: temporal
    }
    let flat = Array::from_slice_i32(&data);
    ops::reshape(&flat, &[3, l as i32], stream)
}

/// Apply M-RoPE to a tensor by splitting head_dim into sections.
///
/// x: [B, n_heads, L, head_dim]
/// position_ids: [3, L] — per-section position IDs
/// mrope_section: [s0, s1, s2] — number of rotation pairs per section (e.g., [11, 11, 10])
/// rope_base: theta base for frequency computation
pub fn apply_mrope(
    x: &Array,
    position_ids: &Array, // [3, L]
    mrope_section: &[usize; 3],
    rope_base: f32,
    stream: &Stream,
) -> Result<Array> {
    let shape = x.shape();
    let b = shape[0];
    let n_heads = shape[1];
    let l = shape[2];
    let head_dim = shape[3];

    // Section dims (each pair takes 2 dims: cos, sin)
    let section_dims: Vec<usize> = mrope_section.iter().map(|s| s * 2).collect();
    let total_rope_dim: usize = section_dims.iter().sum();

    // Split x along head_dim into sections + passthrough
    let mut results = Vec::new();
    let mut offset = 0i32;

    for (sec_idx, &sec_dim) in section_dims.iter().enumerate() {
        let sec_dim_i32 = sec_dim as i32;

        // Slice this section from x: [B, n_heads, L, sec_dim]
        let x_sec = ops::slice(
            x,
            &[0, 0, 0, offset],
            &[b, n_heads, l, offset + sec_dim_i32],
            &[1, 1, 1, 1],
            stream,
        )?;

        // Get position IDs for this section: [1, L] → [L]
        let pos = ops::slice(
            position_ids,
            &[sec_idx as i32, 0],
            &[(sec_idx + 1) as i32, l],
            &[1, 1],
            stream,
        )?;
        let pos = ops::reshape(&pos, &[l], stream)?;

        // Apply rotary embedding with per-token positions
        let rotated = apply_rope_with_positions(&x_sec, &pos, sec_dim, rope_base, stream)?;
        results.push(rotated);

        offset += sec_dim_i32;
    }

    // Passthrough remaining dims if head_dim > total_rope_dim
    if (total_rope_dim as i32) < head_dim {
        let pass = ops::slice(
            x,
            &[0, 0, 0, total_rope_dim as i32],
            &[b, n_heads, l, head_dim],
            &[1, 1, 1, 1],
            stream,
        )?;
        results.push(pass);
    }

    // Concatenate all sections along last axis
    let refs: Vec<&Array> = results.iter().collect();
    let va = VectorArray::from_arrays(&refs);
    ops::concatenate(&va, 3, stream)
}

/// Apply rotary position embeddings with per-token position IDs.
/// x: [B, n_heads, L, dim]
/// positions: [L] — position index for each token
/// dim: number of dimensions (must be even)
/// base: RoPE theta base
fn apply_rope_with_positions(
    x: &Array,
    positions: &Array,
    dim: usize,
    base: f32,
    stream: &Stream,
) -> Result<Array> {
    let shape = x.shape();
    let b = shape[0];
    let n_heads = shape[1];
    let l = shape[2];

    // Compute frequency bands: 1 / (base^(2i/dim)) for i in 0..dim/2
    let half_dim = dim / 2;
    let mut freqs_data = vec![0.0f32; half_dim];
    for (i, f) in freqs_data.iter_mut().enumerate() {
        *f = 1.0 / base.powf(2.0 * i as f32 / dim as f32);
    }
    let freqs = Array::from_slice_f32_shape(&freqs_data, &[1, half_dim as i32]); // [1, half_dim]

    // positions: [L] → [L, 1]
    let pos_f32 = ops::astype(positions, Dtype::Float32, stream)?;
    let pos = ops::reshape(&pos_f32, &[l, 1], stream)?;

    // angles: [L, half_dim] = positions @ freqs
    let angles = ops::matmul(&pos, &freqs, stream)?; // [L, half_dim]

    // cos, sin: [1, 1, L, half_dim] for broadcasting
    let cos = ops::cos(&angles, stream)?;
    let cos = ops::reshape(&cos, &[1, 1, l, half_dim as i32], stream)?;
    let sin = ops::sin(&angles, stream)?;
    let sin = ops::reshape(&sin, &[1, 1, l, half_dim as i32], stream)?;

    // Split x into first half and second half along last dim
    let x1 = ops::slice(
        x,
        &[0, 0, 0, 0],
        &[b, n_heads, l, half_dim as i32],
        &[1, 1, 1, 1],
        stream,
    )?;
    let x2 = ops::slice(
        x,
        &[0, 0, 0, half_dim as i32],
        &[b, n_heads, l, dim as i32],
        &[1, 1, 1, 1],
        stream,
    )?;

    // Rotary: [x1*cos - x2*sin, x1*sin + x2*cos]
    let r1 = ops::subtract(
        &ops::multiply(&x1, &cos, stream)?,
        &ops::multiply(&x2, &sin, stream)?,
        stream,
    )?;
    let r2 = ops::add(
        &ops::multiply(&x1, &sin, stream)?,
        &ops::multiply(&x2, &cos, stream)?,
        stream,
    )?;

    // Concatenate along last dim
    let va = VectorArray::from_arrays(&[&r1, &r2]);
    ops::concatenate(&va, 3, stream)
}
