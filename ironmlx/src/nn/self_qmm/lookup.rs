//! Tile selection lookup: (device, shape, quant) → (BM, BN, BK).
//!
//! Stage 9 fills only the M1 Pro entry (refined by task 7 sweep) plus a
//! conservative global fallback. Stage 10 will expand to M Max / M Ultra /
//! M2+ / M3+ entries.

use std::sync::OnceLock;

/// Tile dimensions chosen for one quant matmul dispatch.
///
/// `bm` rows of output covered per threadgroup, `bn` columns, `bk` columns
/// of weights consumed per inner loop iteration. All three are kernel
/// template parameters in `kernel::dispatch_qmm_t`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tile {
    pub bm: i32,
    pub bn: i32,
    pub bk: i32,
}

/// Default fallback tile — used when device/shape doesn't match any
/// hardcoded entry. Conservative: small enough to fit any threadgroup
/// memory budget, broad enough to function on every Apple Silicon GPU
/// supported by MLX.
const DEFAULT_TILE: Tile = Tile {
    bm: 64,
    bn: 64,
    bk: 32,
};

/// Look up the optimal tile for the given (device, shape, quant) tuple.
///
/// `device_arch`: from [`mlx::metal::architecture()`]. Examples:
/// `"apple_g13s"` (M1 Pro 16-core GPU), `"apple_g13d"` (M1 Pro Max 32-core),
/// `"apple_g14g"` (M2), `"apple_g15p"` (M3 Pro).
///
/// Stage 9 only tunes the M1 Pro entry (Boss's reference machine). Other
/// devices fall back to a conservative default tile that's safe but not
/// necessarily optimal. Stage 10 will add explicit entries for additional
/// chips after sweeping each.
///
/// `_m`, `_n`, `_k`, `_bits`, `_group_size` are reserved for shape-aware
/// dispatch — stage 9 ignores them since only one tile is chosen per chip.
/// They're in the signature now so adding shape branches in stage 10
/// won't be a breaking API change at the qmm_t_on call site.
pub fn lookup_tile(
    device_arch: &str,
    _m: i32,
    _n: i32,
    _k: i32,
    _bits: i32,
    _group_size: i32,
) -> Tile {
    static WARNED: OnceLock<()> = OnceLock::new();
    match device_arch {
        // M1 Pro / M1 Pro Max GPU. Tile populated by task 7 sweep —
        // initial placeholder is (64, 128, 32), an educated guess at this
        // arch class (similar in spirit to llama.cpp's NRA=64/NRB=128
        // shape for Apple GPUs). Will be replaced once the sweep runs.
        "apple_g13s" | "apple_g13d" => Tile {
            bm: 64,
            bn: 128,
            bk: 32,
        },

        // All other devices: warn once per process and use the default.
        _ => {
            if WARNED.set(()).is_ok() {
                tracing::warn!(
                    target: "ironmlx::self_qmm::lookup",
                    device = device_arch,
                    "no tile entry for this device; using default fallback (BM=64, BN=64, BK=32). \
                     Stage 10 will add explicit entries for additional devices."
                );
            }
            DEFAULT_TILE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_m1_pro_returns_specific_tile() {
        let t = lookup_tile("apple_g13s", 2048, 9216, 2560, 4, 64);
        assert_eq!(t.bm, 64);
        assert_eq!(t.bn, 128);
        assert_eq!(t.bk, 32);

        let t = lookup_tile("apple_g13d", 2048, 9216, 2560, 4, 64);
        assert_eq!(t.bm, 64);
        assert_eq!(t.bn, 128);
        assert_eq!(t.bk, 32);
    }

    #[test]
    fn lookup_unknown_device_returns_default() {
        let t = lookup_tile("future_chip_xyz", 2048, 9216, 2560, 4, 64);
        assert_eq!(t.bm, 64);
        assert_eq!(t.bn, 64);
        assert_eq!(t.bk, 32);
    }
}
