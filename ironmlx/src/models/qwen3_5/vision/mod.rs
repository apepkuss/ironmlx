//! Qwen3.5 vision tower (24-layer ViT) — see
//! `docs/superpowers/specs/2026-05-10-p6-vl-design.md` §4.2-4.5.

pub mod block;
pub mod merger;
pub mod patch_embed;

// VisionTower struct + forward 在 Task 12 填。

use mlx::Array;

/// Vision rotary frequency table: `freqs[s, i] = s * (1 / theta^(2i/dim))`,
/// `i ∈ [0, dim/2)`. Output shape: `[seqlen, dim/2]`.
pub fn build_rotary_freqs(seqlen: i32, dim: i32, theta: f32) -> Array {
    use mlx::ops;

    let half = dim / 2;

    let exponents: Vec<f32> = (0..half).map(|i| (2 * i) as f32 / dim as f32).collect();
    let exponents_arr: Array = (exponents.as_slice(), &[half][..]).try_into().unwrap();

    let theta_arr: Array = (&[theta][..], ()).try_into().unwrap();
    let theta_pow = ops::power(&theta_arr, &exponents_arr).unwrap();
    let inv_freq = ops::reciprocal(&theta_pow).unwrap();

    let seq: Vec<f32> = (0..seqlen).map(|i| i as f32).collect();
    let seq_arr: Array = (seq.as_slice(), &[seqlen][..]).try_into().unwrap();
    let seq2 = ops::shape::reshape(&seq_arr, &[seqlen, 1][..]).unwrap();
    let inv2 = ops::shape::reshape(&inv_freq, &[1, half][..]).unwrap();

    &seq2 * &inv2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotary_pos_emb_shape() {
        let freqs = build_rotary_freqs(8, 32, 10000.0);
        assert_eq!(freqs.shape().as_slice(), &[8, 16]); // dim/2 = 16 entries
    }

    #[test]
    fn rotary_pos_emb_values_match_mlx_vlm() {
        let freqs = build_rotary_freqs(4, 32, 10000.0);
        let v: Vec<f32> = freqs.to_vec().unwrap();
        let expected_1_1 = 1.0_f32 / 10000.0_f32.powf(2.0 / 32.0);
        assert!((v[0] - 0.0).abs() < 1e-5);
        assert!((v[16] - 1.0).abs() < 1e-5); // [1, 0]
        assert!((v[17] - expected_1_1).abs() < 1e-5); // [1, 1]
    }
}
