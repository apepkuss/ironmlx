//! Interleaved (traditional) RoPE wrapper for GLM-4.7-Flash (`glm4_moe_lite`).
//!
//! GLM applies decoupled RoPE on the `qk_rope_head_dim=64` channels only
//! (the 512 latent / 192 NoPE channels are NOT rotated). Layout is
//! INTERLEAVED (`traditional=true`, rotate-every-two): MLX's `rope.metal`
//! traditional branch rotates consecutive pairs `(2i, 2i+1)`. The real model
//! passes `base=cfg.rope_theta=1e6`, `scale=1.0`.
//!
//! Thin wrapper over `mlx::fast::rope_on` (scalar offset) and
//! `mlx::fast::rope_with_array_offset_on` (per-row `[B]` i32 offsets, for
//! non-uniform cache lengths).

use anyhow::Result;
use mlx::{Array, StreamOrDevice};

pub struct Glm4Rope {
    dims: i32,
    base: f32,
}

impl Glm4Rope {
    pub fn new(dims: i32, base: f32) -> Self {
        Self { dims, base }
    }

    /// `x`: `[B,H,S,dims]`; `offset`: `[B]` i32 per-row start position.
    /// Interleaved (`traditional=true`).
    pub fn apply(
        &self,
        x: &Array,
        offset: &Array,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        Ok(mlx::fast::rope_with_array_offset_on(
            x,
            self.dims,
            true,
            Some(self.base),
            1.0,
            offset,
            None,
            target,
        )?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arr(d: &[f32], s: &[i32]) -> Array {
        (d, s).try_into().unwrap()
    }

    fn off(d: &[i32], s: &[i32]) -> Array {
        (d, s).try_into().unwrap()
    }

    #[test]
    fn interleaved_matches_hand_computed() {
        // dims=4, base=10000 (small base for hand-calc), pos=1 (offset=[1]).
        // x = [1,0,0,1], shape [1,1,1,4].
        let dims = 4_i32;
        let base = 10000.0_f32;
        let pos = 1.0_f32;

        let rope = Glm4Rope::new(dims, base);
        let x = arr(&[1.0, 0.0, 0.0, 1.0], &[1, 1, 1, 4]);
        let offset = off(&[1], &[1]);
        let out = rope.apply(&x, &offset, StreamOrDevice::default()).unwrap();
        let got = out.to_vec::<f32>().unwrap();

        // Interleaved: rotate consecutive pairs (x0,x1),(x2,x3).
        // freq_i = base^(-2i/dims); theta = pos * freq_i.
        // pair i: (a,b) -> (a*cos - b*sin, a*sin + b*cos).
        let freq0 = base.powf(-2.0 * 0.0 / dims as f32);
        let freq1 = base.powf(-2.0 * 1.0 / dims as f32);
        let t0 = pos * freq0;
        let t1 = pos * freq1;
        // pair0 = (x0=1, x1=0): (cos t0, sin t0)
        let e0 = 1.0 * t0.cos() - 0.0 * t0.sin();
        let e1 = 1.0 * t0.sin() + 0.0 * t0.cos();
        // pair1 = (x2=0, x3=1): (-sin t1, cos t1)
        let e2 = 0.0 * t1.cos() - 1.0 * t1.sin();
        let e3 = 0.0 * t1.sin() + 1.0 * t1.cos();
        let expected = [e0, e1, e2, e3];

        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-4,
                "channel {i}: got {g}, expected {e}, diff {}",
                (g - e).abs()
            );
        }
    }

    #[test]
    fn per_row_offset_differs() {
        // x shape [2,1,1,4], both rows [1,0,0,1], offset=[0,3].
        // row0 (offset 0): pair0 is identity -> [1,0].
        // row1 (offset 3): pair0 rotated -> differs from [1,0].
        let dims = 4_i32;
        let base = 10000.0_f32;
        let rope = Glm4Rope::new(dims, base);
        let x = arr(&[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0], &[2, 1, 1, 4]);
        let offset = off(&[0, 3], &[2]);
        let out = rope.apply(&x, &offset, StreamOrDevice::default()).unwrap();
        let got = out.to_vec::<f32>().unwrap();

        // row0 pair0 == identity [1,0]
        assert!((got[0] - 1.0).abs() < 1e-4, "row0 ch0: got {}", got[0]);
        assert!((got[1] - 0.0).abs() < 1e-4, "row0 ch1: got {}", got[1]);

        // row1 (offset 3) pair0 must be rotated (differs from [1,0]).
        let freq0 = base.powf(-2.0 * 0.0 / dims as f32);
        let t0 = 3.0_f32 * freq0;
        let r0 = t0.cos();
        let r1 = t0.sin();
        assert!(
            (got[4] - r0).abs() < 1e-4,
            "row1 ch0: got {}, exp {r0}",
            got[4]
        );
        assert!(
            (got[5] - r1).abs() < 1e-4,
            "row1 ch1: got {}, exp {r1}",
            got[5]
        );
        // Sanity: row1 differs from identity (offset 3, freq0=1 -> theta=3 rad).
        assert!(
            (got[4] - 1.0).abs() > 1e-3 || (got[5] - 0.0).abs() > 1e-3,
            "row1 pair0 must be rotated, got [{},{}]",
            got[4],
            got[5]
        );
    }
}
