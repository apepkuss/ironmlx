//! GatedDeltaNet kernel routes of the qualified Qwen3.8 Dense affine4 target.
//!
//! The Dense DFlash2 target forward arms this scope when the M5 profile
//! setting `DENSE_GDN_FAST_PATH` is on. Under it the GatedDeltaNet layers take
//! the same single-dispatch routes the Qwen3.6 MoE fast path uses
//! (`moe_fast_path`): the fused decode/verify preparation (conv-state
//! prepend, conv1d + SiLU, q/k norms and gates), the fused q/k RMS norms and
//! decay/beta gates, and the fused gated norm. Each route replays MLX's own
//! operators and reductions, so values are those of the op-by-op chain; only
//! the number of dispatches changes. Nothing else of the MoE fast path
//! (router, attention grouping, product-stable row groups) is armed here.

use std::cell::Cell;
use std::sync::OnceLock;

thread_local! {
    static DEPTH: Cell<u32> = const { Cell::new(0) };
}

pub(crate) struct Scope;

impl Drop for Scope {
    fn drop(&mut self) {
        DEPTH.with(|depth| depth.set(depth.get().saturating_sub(1)));
    }
}

pub(crate) fn scope() -> Scope {
    DEPTH.with(|depth| depth.set(depth.get().saturating_add(1)));
    Scope
}

pub(crate) fn is_armed() -> bool {
    DEPTH.with(|depth| depth.get() > 0)
}

/// M5 profile setting `DENSE_GDN_FAST_PATH`.
pub(crate) fn requested() -> bool {
    static REQUESTED: OnceLock<bool> = OnceLock::new();
    *REQUESTED.get_or_init(|| {
        ironmlx_core::m5_profile::flag(ironmlx_core::m5_profile::settings::DENSE_GDN_FAST_PATH)
    })
}

thread_local! {
    static PREFILL_DEPTH: Cell<u32> = const { Cell::new(0) };
}

/// Dense prefill projections of full 2048-row chunks dequantize the affine4
/// weight to BF16 and run a BF16 matmul (M5 profile setting
/// `DENSE_PREFILL_DEQUANT_GEMM`; MLX's own dequantize rounding, another
/// summation order than its quantized matmul). Inputs wider than 8192 (the
/// MLP down projection), where MLX's dequantize + matmul is slower than the
/// quantized matmul, keep the quantized matmul.
pub(crate) struct PrefillScope;

impl Drop for PrefillScope {
    fn drop(&mut self) {
        PREFILL_DEPTH.with(|depth| depth.set(depth.get().saturating_sub(1)));
    }
}

pub(crate) fn prefill_scope() -> PrefillScope {
    PREFILL_DEPTH.with(|depth| depth.set(depth.get().saturating_add(1)));
    PrefillScope
}

pub(crate) fn prefill_dequant_requested() -> bool {
    static REQUESTED: OnceLock<bool> = OnceLock::new();
    *REQUESTED.get_or_init(|| {
        ironmlx_core::m5_profile::flag(
            ironmlx_core::m5_profile::settings::DENSE_PREFILL_DEQUANT_GEMM,
        )
    })
}

/// Rows of a chunk that takes the BF16 route.
pub(crate) const PREFILL_DEQUANT_ROWS: usize = 2048;
/// Widest input that takes MLX's dequantize + matmul; wider ones (the MLP
/// down projection) keep the quantized matmul.
pub(crate) const PREFILL_DEQUANT_MAX_K: i32 = 8192;

/// M5 profile setting `DENSE_VERIFY_OMIT_GDN_STATE`.
pub(crate) fn verify_omit_gdn_state_requested() -> bool {
    static REQUESTED: OnceLock<bool> = OnceLock::new();
    *REQUESTED.get_or_init(|| {
        ironmlx_core::m5_profile::flag(
            ironmlx_core::m5_profile::settings::DENSE_VERIFY_OMIT_GDN_STATE,
        )
    })
}

/// M5 profile setting `DENSE_PREFILL_TILED_GEMM`.
pub(crate) fn prefill_tiled_requested() -> bool {
    static REQUESTED: OnceLock<bool> = OnceLock::new();
    *REQUESTED.get_or_init(|| {
        ironmlx_core::m5_profile::flag(ironmlx_core::m5_profile::settings::DENSE_PREFILL_TILED_GEMM)
    })
}

/// M5 profile setting `DENSE_PREFILL_FUSED_SWIGLU`.
pub(crate) fn prefill_fused_swiglu_requested() -> bool {
    static REQUESTED: OnceLock<bool> = OnceLock::new();
    *REQUESTED.get_or_init(|| {
        ironmlx_core::m5_profile::flag(
            ironmlx_core::m5_profile::settings::DENSE_PREFILL_FUSED_SWIGLU,
        )
    })
}

pub(crate) fn prefill_dequant_armed() -> bool {
    PREFILL_DEPTH.with(|depth| depth.get() > 0)
}

thread_local! {
    static ANE_DEPTH: Cell<u32> = const { Cell::new(0) };
}

/// Prefill chunks whose MLPs share their channels with the Neural Engine
/// (`ane_ffn`): armed by the Dense target forward for chunks starting before
/// `DENSE_PREFILL_ANE_FFN_MAX_POSITION`.
pub(crate) struct AneScope;

impl Drop for AneScope {
    fn drop(&mut self) {
        ANE_DEPTH.with(|depth| depth.set(depth.get().saturating_sub(1)));
    }
}

pub(crate) fn ane_scope() -> AneScope {
    ANE_DEPTH.with(|depth| depth.set(depth.get().saturating_add(1)));
    AneScope
}

pub(crate) fn ane_armed() -> bool {
    ANE_DEPTH.with(|depth| depth.get() > 0)
}

/// M5 profile setting `DENSE_PREFILL_ANE_FFN_MAX_POSITION` (`None`: no limit).
pub(crate) fn prefill_ane_ffn_max_position() -> Option<i32> {
    static VALUE: OnceLock<Option<i32>> = OnceLock::new();
    *VALUE.get_or_init(|| {
        ironmlx_core::m5_profile::setting(
            ironmlx_core::m5_profile::settings::DENSE_PREFILL_ANE_FFN_MAX_POSITION,
        )
        .and_then(|v| v.parse().ok())
    })
}

/// M5 profile setting `DENSE_PREFILL_ANE_FFN_PIPELINE`.
pub(crate) fn prefill_ane_ffn_pipelined() -> bool {
    static REQUESTED: OnceLock<bool> = OnceLock::new();
    *REQUESTED.get_or_init(|| {
        ironmlx_core::m5_profile::flag(
            ironmlx_core::m5_profile::settings::DENSE_PREFILL_ANE_FFN_PIPELINE,
        )
    })
}

/// M5 profile setting `DENSE_PREFILL_ANE_FFN_UNITS`: the Neural Engine's
/// 512-channel units of each full-chunk prefill MLP under the prefill scope
/// (`ane_ffn`), 0 when off.
pub(crate) fn prefill_ane_ffn_units() -> usize {
    static UNITS: OnceLock<usize> = OnceLock::new();
    *UNITS.get_or_init(|| {
        ironmlx_core::m5_profile::setting(
            ironmlx_core::m5_profile::settings::DENSE_PREFILL_ANE_FFN_UNITS,
        )
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
    })
}

/// GatedDeltaNet fused q/k norm and gate kernels: under either fast path.
pub(crate) fn gdn_kernels_armed() -> bool {
    super::moe_fast_path::is_armed() || is_armed()
}

/// GatedDeltaNet fused preparation and gated norm: the MoE fast path with
/// `FUSED_GLUE`, or the Dense scope.
pub(crate) fn gdn_fused_prep_armed() -> bool {
    (super::moe_fast_path::is_armed() && super::fused_norm::requested()) || is_armed()
}
