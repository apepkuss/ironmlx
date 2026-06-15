//! TurboQuant quantize and dequantize operations.
//!
//! Operates on raw bf16 byte buffers with shape [B, H, S, D].
//! Each head_dim vector is independently quantized using full-dim WHT rotation
//! (matching the GPU `hadamard_transform` which operates on the entire head_dim).

use half::bf16;

use crate::codebook::Codebook;
use crate::pack;
use crate::wht;

/// Compressed KV cache data for one tensor (K or V).
pub struct TurboCompressed {
    /// Packed block data: [scale(f16) | packed_indices] per vector.
    /// Each vector of head_dim values produces one norm + packed indices.
    pub data: Vec<u8>,
    /// Original shape [B, H, S, D].
    pub shape: [i32; 4],
    /// Bit-width (3 or 4).
    pub bits: u8,
    /// Bytes per vector (2 bytes norm + packed indices for head_dim values).
    pub vec_bytes: usize,
}

/// WHT rotation context, reusable across quantize/dequantize calls.
pub struct RotationContext {
    /// Single sign pattern for full head_dim (matching GPU full-dim WHT).
    signs: Vec<f32>,
    /// Head dimension.
    head_dim: usize,
}

impl RotationContext {
    /// Create rotation context for the given head_dim.
    /// Uses full-dim WHT (one rotation over the entire head_dim vector),
    /// matching the GPU `hadamard_transform` behavior.
    pub fn new(head_dim: usize, seed: u64) -> Self {
        assert!(
            head_dim.is_power_of_two(),
            "head_dim {head_dim} must be power of 2 for WHT"
        );
        let signs = wht::generate_signs(head_dim, seed);
        Self { signs, head_dim }
    }
}

/// Bytes per compressed vector: 2 (f16 norm) + packed indices for head_dim values.
fn vec_compressed_bytes(bits: u8, head_dim: usize) -> usize {
    // Number of BLOCK_SIZE groups needed to pack head_dim indices
    let n_groups = head_dim.div_ceil(pack::BLOCK_SIZE);
    2 + n_groups * pack::packed_bytes(bits)
}

/// Quantize a bf16 KV tensor using full-dim WHT.
///
/// `data`: raw bf16 bytes, shape `[B, H, S, D]`.
/// `shape`: [B, H, S, D].
/// `bits`: 3 or 4.
/// `rotation`: WHT rotation context (full head_dim).
pub fn quantize(
    data: &[u8],
    shape: [i32; 4],
    bits: u8,
    rotation: &RotationContext,
) -> TurboCompressed {
    let [b, h, s, d] = shape;
    let (b, h, s, d) = (b as usize, h as usize, s as usize, d as usize);
    assert_eq!(data.len(), b * h * s * d * 2, "bf16 data size mismatch");
    assert_eq!(d, rotation.head_dim, "head_dim mismatch");

    // Codebook calibrated for full head_dim (WHT produces N(0, 1/head_dim) coordinates)
    let codebook = Codebook::new(bits, d);
    let vb = vec_compressed_bytes(bits, d);
    let total_vecs = b * h * s;

    let mut output = vec![0u8; total_vecs * vb];

    for vec_idx in 0..total_vecs {
        let byte_offset = vec_idx * d * 2;
        let vec_f32 = bf16_bytes_to_f32(&data[byte_offset..byte_offset + d * 2]);

        // 1. Extract norm
        let norm = l2_norm(&vec_f32);
        let safe_norm = if norm > 1e-10 { norm } else { 1.0 };

        // 2. Normalize
        let normalized: Vec<f32> = vec_f32.iter().map(|&v| v / safe_norm).collect();

        // 3. Full-dim WHT rotation
        let rotated = wht::rotate_forward(&normalized, &rotation.signs);

        // 4. Scalar quantize all head_dim elements
        let mut all_indices: Vec<u8> = rotated
            .iter()
            .map(|&val| codebook.nearest_index(val))
            .collect();

        // 5. Pack: [f16 norm | packed_indices_group_0 | packed_indices_group_1 | ...]
        let out_offset = vec_idx * vb;
        let norm_f16 = bf16::from_f32(norm);
        output[out_offset..out_offset + 2].copy_from_slice(&norm_f16.to_le_bytes());

        // Pack indices in BLOCK_SIZE groups
        let mut pack_offset = out_offset + 2;
        for group in all_indices.chunks_mut(pack::BLOCK_SIZE) {
            // Pad to BLOCK_SIZE if last group is short
            let mut block = [0u8; 32];
            block[..group.len()].copy_from_slice(group);

            match bits {
                3 => {
                    let packed = pack::pack_3bit(&block);
                    output[pack_offset..pack_offset + 12].copy_from_slice(&packed);
                    pack_offset += 12;
                }
                4 => {
                    let packed = pack::pack_4bit(&block);
                    output[pack_offset..pack_offset + 16].copy_from_slice(&packed);
                    pack_offset += 16;
                }
                _ => unreachable!(),
            }
        }
    }

    TurboCompressed {
        data: output,
        shape,
        bits,
        vec_bytes: vb,
    }
}

/// Dequantize compressed data back to bf16 bytes.
pub fn dequantize(compressed: &TurboCompressed, rotation: &RotationContext) -> Vec<u8> {
    let [b, h, s, d] = compressed.shape;
    let (b, h, s, d) = (b as usize, h as usize, s as usize, d as usize);
    let codebook = Codebook::new(compressed.bits, d);
    let vb = compressed.vec_bytes;
    let total_vecs = b * h * s;

    let mut output = vec![0u8; total_vecs * d * 2];

    for vec_idx in 0..total_vecs {
        let in_offset = vec_idx * vb;

        // 1. Read norm
        let norm =
            bf16::from_le_bytes([compressed.data[in_offset], compressed.data[in_offset + 1]])
                .to_f32();

        // 2. Unpack all indices
        let mut all_indices: Vec<u8> = Vec::with_capacity(d);
        let mut pack_offset = in_offset + 2;
        let n_groups = d.div_ceil(pack::BLOCK_SIZE);
        for _ in 0..n_groups {
            let group_indices = match compressed.bits {
                3 => {
                    let mut packed = [0u8; 12];
                    packed.copy_from_slice(&compressed.data[pack_offset..pack_offset + 12]);
                    pack_offset += 12;
                    pack::unpack_3bit(&packed)
                }
                4 => {
                    let mut packed = [0u8; 16];
                    packed.copy_from_slice(&compressed.data[pack_offset..pack_offset + 16]);
                    pack_offset += 16;
                    pack::unpack_4bit(&packed)
                }
                _ => unreachable!(),
            };
            all_indices.extend_from_slice(&group_indices);
        }
        all_indices.truncate(d);

        // 3. Centroid lookup
        let mut recon: Vec<f32> = all_indices.iter().map(|&i| codebook.centroid(i)).collect();

        // 4. Norm correction
        let recon_norm = l2_norm(&recon);
        if recon_norm > 1e-10 {
            let scale = norm / recon_norm;
            for v in &mut recon {
                *v *= scale;
            }
        }

        // 5. Inverse WHT rotation (full-dim)
        let recovered = wht::rotate_inverse(&recon, &rotation.signs);

        // Convert f32 to bf16 bytes.
        let out_offset = vec_idx * d * 2;
        f32_to_bf16_bytes(&recovered, &mut output[out_offset..out_offset + d * 2]);
    }

    output
}

// Helpers

fn l2_norm(x: &[f32]) -> f32 {
    x.iter().map(|v| v * v).sum::<f32>().sqrt()
}

fn bf16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|chunk| bf16::from_le_bytes([chunk[0], chunk[1]]).to_f32())
        .collect()
}

fn f32_to_bf16_bytes(values: &[f32], output: &mut [u8]) {
    for (i, &v) in values.iter().enumerate() {
        let b = bf16::from_f32(v).to_le_bytes();
        output[i * 2] = b[0];
        output[i * 2 + 1] = b[1];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data(b: usize, h: usize, s: usize, d: usize, seed: u64) -> Vec<u8> {
        let n = b * h * s * d;
        let mut data = vec![0u8; n * 2];
        let mut rng = seed;
        for i in 0..n {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let val = ((rng & 0xFFFF) as f32 / 65536.0 - 0.5) * 2.0;
            let b16 = bf16::from_f32(val).to_le_bytes();
            data[i * 2] = b16[0];
            data[i * 2 + 1] = b16[1];
        }
        data
    }

    #[test]
    fn quantize_dequantize_roundtrip_3bit() {
        let (b, h, s, d) = (1, 1, 4, 128);
        let data = make_test_data(b, h, s, d, 42);
        let shape = [b as i32, h as i32, s as i32, d as i32];
        let rotation = RotationContext::new(d, 42);

        let compressed = quantize(&data, shape, 3, &rotation);
        let recovered = dequantize(&compressed, &rotation);

        assert_eq!(recovered.len(), data.len());

        let orig = bf16_bytes_to_f32(&data);
        let recon = bf16_bytes_to_f32(&recovered);
        let n = orig.len();
        let mse: f64 = orig
            .iter()
            .zip(recon.iter())
            .map(|(a, b)| ((*a - *b) as f64).powi(2))
            .sum::<f64>()
            / n as f64;

        let avg_norm: f64 = {
            let n_vecs = (b * h * s) as f64;
            let mut sum = 0.0f64;
            for v in 0..(b * h * s) {
                let start = v * d;
                let norm: f64 = orig[start..start + d]
                    .iter()
                    .map(|x| (*x as f64).powi(2))
                    .sum::<f64>()
                    .sqrt();
                sum += norm;
            }
            sum / n_vecs
        };
        let normalized_mse = mse / (avg_norm * avg_norm) * d as f64;
        assert!(
            normalized_mse < 0.2,
            "3-bit roundtrip normalized MSE = {normalized_mse:.6}, too high (raw MSE*d = {})",
            mse * d as f64,
        );
    }

    #[test]
    fn quantize_dequantize_roundtrip_4bit() {
        let (b, h, s, d) = (1, 1, 4, 128);
        let data = make_test_data(b, h, s, d, 42);
        let shape = [b as i32, h as i32, s as i32, d as i32];
        let rotation = RotationContext::new(d, 42);

        let compressed = quantize(&data, shape, 4, &rotation);
        let recovered = dequantize(&compressed, &rotation);

        let orig = bf16_bytes_to_f32(&data);
        let recon = bf16_bytes_to_f32(&recovered);
        let n = orig.len();
        let mse: f64 = orig
            .iter()
            .zip(recon.iter())
            .map(|(a, b)| ((*a - *b) as f64).powi(2))
            .sum::<f64>()
            / n as f64;

        let avg_norm: f64 = {
            let n_vecs = (b * h * s) as f64;
            let mut sum = 0.0f64;
            for v in 0..(b * h * s) {
                let start = v * d;
                let norm: f64 = orig[start..start + d]
                    .iter()
                    .map(|x| (*x as f64).powi(2))
                    .sum::<f64>()
                    .sqrt();
                sum += norm;
            }
            sum / n_vecs
        };
        let normalized_mse = mse / (avg_norm * avg_norm) * d as f64;
        assert!(
            normalized_mse < 0.1,
            "4-bit roundtrip normalized MSE = {normalized_mse:.6}, too high"
        );
    }

    #[test]
    fn compression_ratio() {
        let (b, h, s, d) = (1, 8, 256, 128);
        let data = make_test_data(b, h, s, d, 42);
        let shape = [b as i32, h as i32, s as i32, d as i32];
        let rotation = RotationContext::new(d, 42);

        let compressed = quantize(&data, shape, 3, &rotation);

        let original_bytes = data.len();
        let compressed_bytes = compressed.data.len();
        let ratio = original_bytes as f64 / compressed_bytes as f64;

        // Full-dim: per vector = 2 bytes norm + 4 groups x 12 bytes = 50 bytes
        // vs bf16: 128 x 2 = 256 bytes, ratio is about 5.12.
        assert!(
            ratio > 4.0 && ratio < 6.0,
            "compression ratio {ratio:.2} not in expected range"
        );
    }

    #[test]
    fn different_head_dims() {
        for d in [64, 128, 256] {
            let (b, h, s) = (1, 1, 2);
            let data = make_test_data(b, h, s, d, 42);
            let shape = [b as i32, h as i32, s as i32, d as i32];
            let rotation = RotationContext::new(d, 42);

            let compressed = quantize(&data, shape, 3, &rotation);
            let recovered = dequantize(&compressed, &rotation);
            assert_eq!(recovered.len(), data.len(), "d={d}: size mismatch");
        }
    }

    #[test]
    fn batch_correctness() {
        let (b, h, s, d) = (2, 4, 8, 128);
        let data = make_test_data(b, h, s, d, 42);
        let shape = [b as i32, h as i32, s as i32, d as i32];
        let rotation = RotationContext::new(d, 42);

        let compressed = quantize(&data, shape, 3, &rotation);
        let recovered = dequantize(&compressed, &rotation);

        let vec_bytes = d * 2;
        let first_orig = bf16_bytes_to_f32(&data[..vec_bytes]);
        let first_recon = bf16_bytes_to_f32(&recovered[..vec_bytes]);

        let last_start = (b * h * s - 1) * vec_bytes;
        let last_orig = bf16_bytes_to_f32(&data[last_start..last_start + vec_bytes]);
        let last_recon = bf16_bytes_to_f32(&recovered[last_start..last_start + vec_bytes]);

        let mse_first: f64 = first_orig
            .iter()
            .zip(first_recon.iter())
            .map(|(a, b)| ((*a - *b) as f64).powi(2))
            .sum::<f64>()
            / d as f64;
        let mse_last: f64 = last_orig
            .iter()
            .zip(last_recon.iter())
            .map(|(a, b)| ((*a - *b) as f64).powi(2))
            .sum::<f64>()
            / d as f64;

        assert!(mse_first < 1.0, "first vec MSE too high: {mse_first}");
        assert!(mse_last < 1.0, "last vec MSE too high: {mse_last}");
    }
}
