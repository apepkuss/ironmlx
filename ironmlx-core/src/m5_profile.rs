//! Default DFlash2 serving profile for Apple GPU generation 17+ (M5).
//!
//! On a g17+ GPU, DFlash2 serving enables the qualified candidate settings
//! below (affine4 tensor-unit QMM, lane attention, flat tf-v1 tree, fixed
//! draft budget, ragged linear batching, prefill kernels, ...). Each feature
//! keeps its own quantization, shape, request and memory checks and falls
//! back to the ordinary path when they do not hold. Other GPUs, other serving
//! modes and `--m5-dflash2-profile off` keep the previous defaults.
//!
//! Every setting is still keyed by its former `IRONMLX_EXPERIMENTAL_*`
//! variable: an explicitly set variable wins over the profile (`"1"` on, any
//! other value off), so individual features can be turned off or tested.
//!
//! [`install`] must run once at startup, before model loading; reads before
//! it see an inactive profile.

use std::sync::OnceLock;

/// Former experiment variables, now profile settings.
pub mod settings {
    pub const M5_DFLASH2_QMM: &str = "IRONMLX_EXPERIMENTAL_M5_DFLASH2_QMM";
    pub const M5_LANE_ATTN: &str = "IRONMLX_EXPERIMENTAL_M5_LANE_ATTN";
    pub const M5_ATTN_GROUP_TILES: &str = "IRONMLX_EXPERIMENTAL_M5_ATTN_GROUP_TILES";
    pub const M5_SHARED_WEIGHT_LAYOUT: &str = "IRONMLX_EXPERIMENTAL_M5_SHARED_WEIGHT_LAYOUT";
    pub const M5_PRECOMPILE: &str = "IRONMLX_EXPERIMENTAL_M5_PRECOMPILE";
    pub const DFLASH2_FIXED_BUDGET: &str = "IRONMLX_EXPERIMENTAL_DFLASH2_FIXED_BUDGET";
    pub const DFLASH2_SINGLE_PREFILL: &str = "IRONMLX_EXPERIMENTAL_DFLASH2_SINGLE_PREFILL";
    pub const DFLASH2_RADIX_TOPK: &str = "IRONMLX_EXPERIMENTAL_DFLASH2_RADIX_TOPK";
    pub const DFLASH2_FLAT_TREE: &str = "IRONMLX_EXPERIMENTAL_DFLASH2_FLAT_TREE";
    pub const DFLASH2_TREE_PROFILE: &str = "IRONMLX_EXPERIMENTAL_DFLASH2_TREE_PROFILE";
    pub const DFLASH2_RAGGED_LINEAR: &str = "IRONMLX_EXPERIMENTAL_DFLASH2_RAGGED_LINEAR";
    /// Shared draft budget of a ragged linear batch for targets whose single
    /// requests keep an adaptive budget (Qwen3.6 MoE window-cost policy).
    pub const DFLASH2_RAGGED_BUDGET: &str = "IRONMLX_EXPERIMENTAL_DFLASH2_RAGGED_BUDGET";
    /// Smallest ragged batch that decodes one token per row without drafts
    /// (0 disables batched one-token windows).
    pub const DFLASH2_RAGGED_ORDINARY_MIN_WIDTH: &str =
        "IRONMLX_EXPERIMENTAL_DFLASH2_RAGGED_ORDINARY_MIN_WIDTH";
    /// Consecutive batched one-token windows after which the next ragged
    /// window drafts (keeps speculation measured while batched).
    pub const DFLASH2_RAGGED_PROBE_INTERVAL: &str =
        "IRONMLX_EXPERIMENTAL_DFLASH2_RAGGED_PROBE_INTERVAL";
    /// Ragged multi-request rows run ordinary batched kernels (values not
    /// bit-identical to each row's B1 decode); B1 stays row-exact.
    pub const DFLASH2_RAGGED_FAST_ARITHMETIC: &str =
        "IRONMLX_EXPERIMENTAL_DFLASH2_RAGGED_FAST_ARITHMETIC";
    pub const PREFILL_MASKED_SOFTMAX: &str = "IRONMLX_EXPERIMENTAL_PREFILL_MASKED_SOFTMAX";
    pub const PREFILL_CHUNK_CACHE_RESET: &str = "IRONMLX_EXPERIMENTAL_PREFILL_CHUNK_CACHE_RESET";
    pub const PREFILL_QMM_MTILE_METALLIB: &str = "IRONMLX_EXPERIMENTAL_PREFILL_QMM_MTILE_METALLIB";
    pub const PREFILL_D256_NAX_METALLIB: &str = "IRONMLX_EXPERIMENTAL_PREFILL_D256_NAX_METALLIB";
    /// Fewest keys at which a causal prefill chunk of any GQA layout runs
    /// the D256 NAX attention (0: only the dense 27B layout).
    pub const PREFILL_D256_NAX_MIN_KEYS: &str = "IRONMLX_EXPERIMENTAL_PREFILL_D256_NAX_MIN_KEYS";
    /// Fewest keys at which query blocks of at most 8 rows (decode and
    /// speculative verify) run the split-K D256 NAX attention (0: never).
    pub const DECODE_D256_NAX_MIN_KEYS: &str = "IRONMLX_EXPERIMENTAL_DECODE_D256_NAX_MIN_KEYS";
    /// Prefill QMM of every affine4/g64 chunk shape through the BM128 M-tile
    /// kernel (bit-identical to MLX's).
    pub const PREFILL_QMM_MTILE_ANY_SHAPE: &str =
        "IRONMLX_EXPERIMENTAL_PREFILL_QMM_MTILE_ANY_SHAPE";
    pub const MLX_CACHE_MAX_MIB: &str = "IRONMLX_EXPERIMENTAL_MLX_CACHE_MAX_MIB";
    pub const QEMBEDDING_RUNTIME_COUNT: &str = "IRONMLX_EXPERIMENTAL_QEMBEDDING_RUNTIME_COUNT";
    /// Wire the loaded DFlash2 weights (MLX residency sets) at serve start.
    pub const WIRED_WEIGHTS: &str = "IRONMLX_EXPERIMENTAL_WIRED_WEIGHTS";
    /// Qwen3.6 MoE: expert-grouped gather qmv for multi-token verification.
    /// Qwen3.6 MoE affine4: route non-expert projections through the M5
    /// affine4 tensor-unit kernel (changes reduction order versus MLX).
    pub const M5_MOE_AFFINE4_PROJECTIONS: &str = "IRONMLX_EXPERIMENTAL_M5_MOE_AFFINE4_PROJECTIONS";
    /// Qwen3.6 MoE affine4: the LM head on the M5 affine4 route as well
    /// (keeps its own tile-layout copy under the shared weight layout).
    pub const M5_MOE_HEAD: &str = "IRONMLX_EXPERIMENTAL_M5_HEAD";
    /// Qwen3.6 MoE: routed and shared experts of narrow verify batches as
    /// expert tiles (`nn::moe_tile_experts`).
    pub const MOE_TILE_EXPERTS: &str = "IRONMLX_EXPERIMENTAL_MOE_TILE_EXPERTS";
    /// Qwen3.6 MoE: routed and shared experts of every batch on the M5
    /// tensor unit with one tiled resident copy of the expert weights
    /// (`nn::moe_m5_experts`).
    pub const MOE_M5_EXPERTS: &str = "IRONMLX_EXPERIMENTAL_MOE_M5_EXPERTS";
    /// GatedDelta speculative commit: replay only each row's accepted steps,
    /// state only.
    pub const GDN_COMMIT_KERNEL: &str = "IRONMLX_EXPERIMENTAL_GDN_COMMIT_KERNEL";
    /// First token position whose GatedDeltaNet prefill recurrence runs the
    /// long-chunk kernel (another summation order; unset: never).
    pub const GDN_LONG_SCAN_MIN_POSITION: &str = "IRONMLX_EXPERIMENTAL_GDN_LONG_SCAN_MIN_POSITION";
    /// Long-context B1 DFlash2: a window whose adaptive draft budget is at
    /// most this many tokens verifies a flat draft tree of that many nodes
    /// (on the split-K tree attention) instead of a linear draft.
    pub const DFLASH2_LONG_CONTEXT_TREE_NODES: &str =
        "IRONMLX_EXPERIMENTAL_DFLASH2_LONG_CONTEXT_TREE_NODES";
    /// DFlash2 drafter projections of ragged multi-request drafts on the M5
    /// affine4 route (single-request drafts keep the native projections).
    pub const DFLASH2_DRAFT_M5: &str = "IRONMLX_EXPERIMENTAL_DFLASH2_DRAFT_M5";
    /// Ragged drafts with the row-independent work batched across rows
    /// (context projections, norm + RoPE, attention).
    pub const DFLASH2_DRAFT_BATCHED: &str = "IRONMLX_EXPERIMENTAL_DFLASH2_DRAFT_BATCHED";
    /// Rows with fewer remaining tokens than the draft budget stay in the
    /// ragged batch; the window keeps only the tokens they still need.
    pub const DFLASH2_RAGGED_TAIL: &str = "IRONMLX_EXPERIMENTAL_DFLASH2_RAGGED_TAIL";
    /// Ragged windows count accepted tokens on the GPU and submit the cache
    /// commit right after the head, before the host reads the window.
    pub const DFLASH2_GPU_COMMIT: &str = "IRONMLX_EXPERIMENTAL_DFLASH2_GPU_COMMIT";
    /// Fixed draft budget of a request that ran ragged windows and is left
    /// decoding alone; 0 keeps its adaptive single-request policy.
    pub const DFLASH2_RAGGED_SOLO_BUDGET: &str = "IRONMLX_EXPERIMENTAL_DFLASH2_RAGGED_SOLO_BUDGET";
    /// M5 affine4 tensor kernel computes the group sums of x itself instead
    /// of a separate dependent dispatch per projection (same bits).
    pub const M5_FUSED_XSUM: &str = "IRONMLX_EXPERIMENTAL_M5_FUSED_XSUM";
    /// Every DFlash2 drafter projection in one resident M5 tiled layout, and
    /// every proposal on the M5 route (no checkpoint-layout copy kept).
    pub const DFLASH2_DRAFT_M5_RESIDENT: &str = "IRONMLX_EXPERIMENTAL_DFLASH2_DRAFT_M5_RESIDENT";
    /// M5 affine4 kernel: per-shape K-slice counts tuned on the M5 Max.
    pub const M5_TUNED_SPLIT_K: &str = "IRONMLX_EXPERIMENTAL_M5_TUNED_SPLIT_K";
    /// Bit-identical single-dispatch decoder glue: GatedDelta input
    /// preparation and gated norm, residual add + norm, the MoE residual add
    /// (`nn::fused_norm`, `nn::gdn_prep`).
    pub const FUSED_GLUE: &str = "IRONMLX_EXPERIMENTAL_FUSED_GLUE";
    /// DFlash2 serving runs a few short synthetic requests of the prompt
    /// length classes through the request pipeline before it accepts
    /// requests, so no shape-specialized kernel compiles inside a request.
    pub const DFLASH2_STARTUP_WARMUP: &str = "IRONMLX_EXPERIMENTAL_DFLASH2_STARTUP_WARMUP";
    /// Dense Qwen3.8 GatedDeltaNet layers take the single-dispatch preparation,
    /// q/k norm, gate and gated-norm kernels of the MoE fast path (same values).
    pub const DENSE_GDN_FAST_PATH: &str = "IRONMLX_EXPERIMENTAL_DENSE_GDN_FAST_PATH";
    /// Long prompts of equal length prefill one request at a time (each
    /// first token and its decoding start right after its own prefill).
    pub const DFLASH2_LONG_PREFILL_SINGLE: &str =
        "IRONMLX_EXPERIMENTAL_DFLASH2_LONG_PREFILL_SINGLE";
    /// Longest wait (milliseconds) of a long prompt for the decoding streams
    /// when no other request waits behind it (unset: the shared 3 s bound).
    /// Its wait delays only its own first token, while a stream still
    /// decoding when its prefill starts stalls for the whole prefill.
    pub const DFLASH2_LONG_ADMISSION_LAST_DEFER_MS: &str =
        "IRONMLX_EXPERIMENTAL_DFLASH2_LONG_ADMISSION_LAST_DEFER_MS";
    /// A long prompt waits for the decoding streams up to this many
    /// microseconds per prompt token past the long-prompt threshold (8192)
    /// when that is longer than the shared 3 s bound: its prefill stalls them
    /// for a time that grows with its length.
    pub const DFLASH2_LONG_ADMISSION_DEFER_US_PER_TOKEN: &str =
        "IRONMLX_EXPERIMENTAL_DFLASH2_LONG_ADMISSION_DEFER_US_PER_TOKEN";
    /// Dense prefill projections of full 2048-row chunks (inputs up to 8192
    /// wide) run as BF16 dequantize + matmul instead of the quantized matmul.
    pub const DENSE_PREFILL_DEQUANT_GEMM: &str = "IRONMLX_EXPERIMENTAL_DENSE_PREFILL_DEQUANT_GEMM";
    /// Dense prefill MLPs of full 2048-row chunks give the Neural Engine this
    /// many 512-channel units of their intermediate channels (W8A8, rotated
    /// int8 operands), run alongside the GPU's; unset or 0 keeps them all on
    /// the GPU.
    pub const DENSE_PREFILL_ANE_FFN_UNITS: &str =
        "IRONMLX_EXPERIMENTAL_DENSE_PREFILL_ANE_FFN_UNITS";
    /// Prefill chunks starting at or past this token position keep their MLP
    /// on the GPU alone (unset: no limit).
    pub const DENSE_PREFILL_ANE_FFN_MAX_POSITION: &str =
        "IRONMLX_EXPERIMENTAL_DENSE_PREFILL_ANE_FFN_MAX_POSITION";
    /// With `DENSE_PREFILL_ANE_FFN_UNITS`, a single-sequence full chunk runs
    /// its decoder layers as two 1024-row halves, interleaved layer by layer,
    /// so the Neural Engine computes one half's MLP share while the GPU runs
    /// the other half's attention and MLP; other chunks keep the MLP on the
    /// GPU.
    pub const DENSE_PREFILL_ANE_FFN_PIPELINE: &str =
        "IRONMLX_EXPERIMENTAL_DENSE_PREFILL_ANE_FFN_PIPELINE";
    /// Dense prefill projections of more than 128 rows read the single-resident
    /// M5 tiled weight layout with a fused dequantize + tensor-unit GEMM (MLX's
    /// dequantized values), so prefill and decode share one layout and no
    /// projection converts between them.
    pub const DENSE_PREFILL_TILED_GEMM: &str = "IRONMLX_EXPERIMENTAL_DENSE_PREFILL_TILED_GEMM";
    /// With `DENSE_PREFILL_TILED_GEMM`, a fused gate / up MLP projection's
    /// tiled GEMM applies SwiGLU in its epilogue (MLX's compiled SwiGLU of the
    /// rounded products, bit for bit) instead of writing both products.
    pub const DENSE_PREFILL_FUSED_SWIGLU: &str = "IRONMLX_EXPERIMENTAL_DENSE_PREFILL_FUSED_SWIGLU";
    /// Projections whose output width is a multiple of 64 keep 64-column
    /// tiles in the M5 tiled layout and take the 64-column (two-simdgroup)
    /// tensor-unit kernel for 1..=128 rows; the same per-group integer
    /// products and f32 scale / bias as the 32-column kernel.
    pub const M5_QMM_COOP: &str = "IRONMLX_EXPERIMENTAL_M5_QMM_COOP";
    /// The startup warmup also runs ragged (B4) linear windows at each batch
    /// width of a four-sequence server, so their batch-specialized draft and
    /// verify kernels compile before serving.
    pub const DFLASH2_STARTUP_RAGGED_WARMUP: &str =
        "IRONMLX_EXPERIMENTAL_DFLASH2_STARTUP_RAGGED_WARMUP";
    /// The startup warmup also runs one full prefill chunk (as it does with
    /// `DENSE_PREFILL_ANE_FFN_UNITS`), so the full-chunk kernels compile
    /// before serving.
    pub const DFLASH2_STARTUP_FULL_CHUNK_WARMUP: &str =
        "IRONMLX_EXPERIMENTAL_DFLASH2_STARTUP_FULL_CHUNK_WARMUP";
    /// Captured multi-token linear verify windows run the GatedDeltaNet
    /// recurrence without its state output (the commit replays the accepted
    /// steps from the capture's base state); same outputs.
    pub const DENSE_VERIFY_OMIT_GDN_STATE: &str =
        "IRONMLX_EXPERIMENTAL_DENSE_VERIFY_OMIT_GDN_STATE";
    /// Ragged linear windows of fewer rows draft more tokens, up to this many
    /// verify rows in all (the row block the verify projections run at the
    /// same cost) and at most the draft block: a window of w rows drafts
    /// max(budget, rows / w - 1). Needs `DFLASH2_RAGGED_TAIL`.
    pub const DFLASH2_RAGGED_FILL_ROWS: &str = "IRONMLX_EXPERIMENTAL_DFLASH2_RAGGED_FILL_ROWS";
}

/// Model family a DFlash2 profile is qualified for. Each family carries its
/// own settings table; a family never inherits another family's settings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum M5ProfileTarget {
    /// Dense Qwen3.5-family targets (the Qwen3.8-27B lane qualification).
    Qwen35Dense,
    /// Qwen3.6-35B-A3B MoE targets.
    Qwen36Moe,
    /// No qualified profile: every setting stays off.
    Unqualified,
}

impl M5ProfileTarget {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Qwen35Dense => "qwen35-dense",
            Self::Qwen36Moe => "qwen36-moe",
            Self::Unqualified => "unqualified",
        }
    }

    fn values(self) -> &'static [(&'static str, &'static str)] {
        match self {
            Self::Qwen35Dense => PROFILE_VALUES,
            Self::Qwen36Moe => MOE_PROFILE_VALUES,
            Self::Unqualified => &[],
        }
    }
}

/// Qwen3.6 MoE profile values. Only settings whose MoE correctness and
/// end-to-end benefit were measured are listed. (An exact expert-grouped
/// gather qmv for 8+ verify rows was removed: on M5 Max, same-process A/B on
/// affine6/affine8, adaptive linear windows showed no measurable gain and the
/// wide shapes that used it stayed slower than adaptive linear.) The
/// runtime-count quantized embedding has identical
/// arithmetic and removes a Metal library compile on every new prompt length
/// (affine4 short prompts: six-task outputs unchanged, first-use prefill
/// spikes of 35-116 ms gone). A 512 MiB MLX buffer-cache ceiling (default
/// 2 GiB) lowered the short-prompt serving footprint by 1.0 GiB with unchanged
/// window costs; long-prompt prefill was not measured at this ceiling. Wired
/// weights removed the intermittent 40-80 ms slower prefills after the idle
/// gap between requests (0 of 48 against 6 of 24 in the paired sessions).
///
/// Concurrent requests decode fully speculatively (no one-token windows):
/// ragged rows draft 3 tokens with ordinary batched kernels, the affine4
/// projections, LM head and drafter run on the M5 tensor-unit route, routed
/// and shared experts run as expert tiles, the draft batches its rows, the
/// GatedDelta commit replays only accepted steps and is submitted from GPU
/// accepted counts, and rows near their limit stay in the ragged batch. A
/// request left alone at the end of a batch keeps the ragged budget instead
/// of calibrating the single-request policy (whose budget-0 calibration
/// windows are one-token windows): two interleaved rounds 367.7/365.7 tok/s
/// against 362.8/364.0 for the adaptive policy. Four-app B4 thermal round:
/// 364.97 tok/s against 346.3 for Splash 1.3.0; teacher-forced flips
/// 1.2-2.3% of positions, all within a reference margin of 0.63.
///
/// Routed and shared experts of every batch run on the M5 tensor unit from
/// one tiled resident copy of the expert weights (converted at start-up);
/// the drafter's projections live in one resident M5 layout and every
/// proposal takes that route; M5 projections use per-shape K-slice counts and
/// in-kernel group sums; GatedDelta input preparation, its gated norm and the
/// decoder's residual add + post norm (and the MoE residual add) are single
/// dispatches computing the op-by-op values bit for bit. Paired in-process B1
/// q8 windows: experts -4.3%, glue -2.1%, drafts -0.7%; verify q8 12.7 ->
/// 11.7 ms with the split counts.
///
/// The vision encoder loads on the first image request instead of at start:
/// text-only B1 serving footprint 22.23 -> 21.34 GB. Startup warmup requests compile the
/// prompt-length-specialized kernels before serving: the first run of a new
/// build no longer stalls 190-240 ms inside a request (prefill of a prompt
/// over 64 tokens, the drafter's first 33-48-row context projection).
const MOE_PROFILE_VALUES: &[(&str, &str)] = &[
    (settings::DFLASH2_RAGGED_LINEAR, "1"),
    (settings::DFLASH2_RAGGED_BUDGET, "3"),
    (settings::DFLASH2_RAGGED_FAST_ARITHMETIC, "1"),
    (settings::DFLASH2_RAGGED_TAIL, "1"),
    (settings::DFLASH2_GPU_COMMIT, "1"),
    (settings::DFLASH2_RAGGED_SOLO_BUDGET, "3"),
    (settings::DFLASH2_DRAFT_M5, "1"),
    (settings::DFLASH2_DRAFT_BATCHED, "1"),
    (settings::M5_MOE_AFFINE4_PROJECTIONS, "1"),
    (settings::M5_MOE_HEAD, "1"),
    (settings::M5_SHARED_WEIGHT_LAYOUT, "1"),
    (settings::M5_PRECOMPILE, "1"),
    (settings::MOE_TILE_EXPERTS, "1"),
    (settings::MOE_M5_EXPERTS, "1"),
    (settings::M5_TUNED_SPLIT_K, "1"),
    (settings::M5_FUSED_XSUM, "1"),
    (settings::FUSED_GLUE, "1"),
    (settings::DFLASH2_DRAFT_M5_RESIDENT, "1"),
    (settings::GDN_COMMIT_KERNEL, "1"),
    (settings::DFLASH2_STARTUP_WARMUP, "1"),
    (settings::QEMBEDDING_RUNTIME_COUNT, "1"),
    (settings::MLX_CACHE_MAX_MIB, "512"),
    (settings::WIRED_WEIGHTS, "1"),
    (settings::PREFILL_D256_NAX_MIN_KEYS, "2048"),
    (settings::DECODE_D256_NAX_MIN_KEYS, "8193"),
    (settings::GDN_LONG_SCAN_MIN_POSITION, "2048"),
    (settings::PREFILL_QMM_MTILE_ANY_SHAPE, "1"),
];

/// Default flat-tree size of the Qwen3.6 MoE profile when the option is not
/// given explicitly. Linear windows: tree7/tree15 measured slower than linear
/// DFlash2 on every affine width (and below ordinary decoding on 5/6/8-bit).
pub const MOE_DFLASH2_TREE_MAX_NODES: usize = 0;

/// Profile values of the settings (the qualified candidate configuration).
/// The prefill kernel libraries are set to the embedded copies at install.
///
/// Drafter and GatedDeltaNet routes (same target values; same-text paired
/// B1 windows on M5 Max, six tasks, token ids unchanged): the drafter's
/// projections run from one resident M5 layout (52.7 -> 48.7 ms per tree
/// window), and the GatedDeltaNet layers take the fused preparation, q/k
/// norm, gate and gated-norm kernels, tree windows included (48.8 -> 47.0
/// ms). Concurrent requests verify ragged rows at a draft budget of 3 with
/// batched M5 drafts: B4 throughput 165.9 -> 180.0 tok/s and per-request
/// decode 51.9 -> 57.9 tok/s against budget 7 (outputs unchanged); single
/// linear windows keep the fixed budget 7.
///
/// Long prompts: causal prefill chunks from 2048 keys take the D256 NAX
/// attention (the exact 2048-row rule stopped at 40960 keys), GatedDeltaNet
/// prefill chunks from position 2048 take the long-chunk scan, and prefill
/// projections take the BM128 M tile at every shape (in-process code-1 TTFT:
/// 30K 42.3 -> 41.2 s, 61K 103.6 -> 94.8 s, 128K 327.2 -> 272.6 s; short
/// prompts' outputs unchanged). Equal-length long prompts prefill one at a
/// time, so each request decodes right after its own prefill. Full
/// 2048-row prefill chunks of the projections up to 8192 inputs wide run as
/// BF16 dequantize + matmul (30K 40.9 -> 38.9 s, 61K 95.6 -> 92.0 s), and
/// GatedDeltaNet chunks from position 2048 take the fused preparation and
/// gated norm. Startup warmup requests compile the prompt-length-specialized
/// prefill kernels before serving (the short code-3 prompt's TTFT had risen to
/// 278 ms from an in-request compile). Lane attention groups six 16-row
/// tiles per threadgroup, so a 15-node tree's 96 query rows read the keys once
/// (128K code-1 tree windows 107.2 -> 98.8 ms; same token ids).
const PROFILE_VALUES: &[(&str, &str)] = &[
    (settings::M5_DFLASH2_QMM, "1"),
    (settings::M5_LANE_ATTN, "1"),
    (settings::M5_ATTN_GROUP_TILES, "6"),
    (settings::M5_SHARED_WEIGHT_LAYOUT, "1"),
    (settings::M5_PRECOMPILE, "1"),
    (settings::DFLASH2_FIXED_BUDGET, "7"),
    (settings::DFLASH2_SINGLE_PREFILL, "1"),
    (settings::DFLASH2_RADIX_TOPK, "1"),
    (settings::DFLASH2_FLAT_TREE, "1"),
    (settings::DFLASH2_TREE_PROFILE, "tf-v1"),
    (settings::DFLASH2_RAGGED_LINEAR, "1"),
    (settings::PREFILL_MASKED_SOFTMAX, "1"),
    (settings::PREFILL_CHUNK_CACHE_RESET, "1"),
    // 2048 MiB buffer-cache ceiling: short B4 peak 20.41 -> 18.86 GiB at
    // -1% throughput (512 MiB: 17.47 GiB but windows +9%, buffer churn).
    (settings::MLX_CACHE_MAX_MIB, "2048"),
    (settings::QEMBEDDING_RUNTIME_COUNT, "1"),
    (settings::DFLASH2_DRAFT_M5_RESIDENT, "1"),
    (settings::DENSE_GDN_FAST_PATH, "1"),
    (settings::DFLASH2_RAGGED_BUDGET, "3"),
    (settings::DFLASH2_DRAFT_M5, "1"),
    (settings::DFLASH2_DRAFT_BATCHED, "1"),
    (settings::PREFILL_D256_NAX_MIN_KEYS, "2048"),
    (settings::GDN_LONG_SCAN_MIN_POSITION, "2048"),
    (settings::PREFILL_QMM_MTILE_ANY_SHAPE, "1"),
    (settings::DFLASH2_LONG_PREFILL_SINGLE, "1"),
    (settings::DENSE_PREFILL_DEQUANT_GEMM, "1"),
    (settings::DFLASH2_STARTUP_WARMUP, "1"),
    // Full 2048-row prefill chunks give the Neural Engine 12 of the 34
    // 512-channel MLP units (W8A8, rotated int8 operands; nn::ane_ffn). With
    // 9 units over whole chunks: 30K in-process TTFT 37.7 -> 35.5 s; NLL
    // (28672 tokens each of C++ source and English docs) moves -0.0002 /
    // +0.0002 with top-1 agreement 99.55% / 99.16%, as a split computing the
    // ANE's channels on the GPU does (-0.0002 / -0.0000, 99.52% / 99.32%).
    (settings::DENSE_PREFILL_ANE_FFN_UNITS, "12"),
    // The halves of a full chunk interleave layer by layer, so the ANE's
    // share of one half overlaps the other half's attention and MLP; 12
    // units (14 / 16 equal or ANE-bound, more units for the full-attention
    // layers alone slower). Paired in-process 30K TTFT 30.7 / 30.8 s (9 units,
    // whole chunks) -> 29.6 / 29.7 s, 61K 74.4 / 75.2 -> 73.2 / 73.8 s. NLL as
    // above: -0.0003 / -0.0003, top-1 99.55% / 99.26%.
    (settings::DENSE_PREFILL_ANE_FFN_PIPELINE, "1"),
    // Chunks from token 65536 on keep the MLP on the GPU: in-process 61K
    // TTFT 91.7 -> 86.3 s with the split, but a 128K prompt lost the first
    // half's gain in its second (264.5 s with this limit, 263.2 s without the
    // split, 272 s with it throughout).
    (settings::DENSE_PREFILL_ANE_FFN_MAX_POSITION, "65536"),
    // Prefill projections read the single-resident tiled layout with a fused
    // dequantize + tensor-unit GEMM (MLX's dequantized values: relative error
    // against an f32 reference equal to MLX's dequantize + matmul; natural-text
    // NLL -0.00014 / +0.00008, top-1 99.85% / 99.74%). 2048 x 5120 -> 34816
    // 13.8 -> 11.9 ms, down 2048 x 17408 -> 5120 ~7.8 -> 6.1 ms; no layout
    // conversion between prefill and decode. ANE off, three consecutive 30K
    // prefills in one process: 124.6 / 123.2 -> 108.2 / 108.2 s.
    (settings::DENSE_PREFILL_TILED_GEMM, "1"),
    // 64-column tiles and kernel for projections of a multiple of 64 outputs
    // (row-invariant, the same integer products and f32 scale / bias): DRAM-
    // cold 32-row 34816 x 5120 0.302 -> 0.246 ms, 5120 x 17408 0.166 -> 0.127
    // ms; 1 and 16 rows no slower.
    (settings::M5_QMM_COOP, "1"),
    // Ragged (B4) windows at batch widths 4/3/2 in the startup warmup: a
    // first budget-7 four-row window had compiled for 0.5 s inside a request.
    (settings::DFLASH2_STARTUP_RAGGED_WARMUP, "1"),
    // One full prefill chunk in the startup warmup with the Neural Engine off
    // too: the full-tile prefill kernels otherwise compile inside the first
    // long prompt.
    (settings::DFLASH2_STARTUP_FULL_CHUNK_WARMUP, "1"),
    // Ragged (B4) linear verify windows skip the GatedDeltaNet state output
    // their commit never reads (12.6 MB per layer at four rows); same tokens.
    (settings::DENSE_VERIFY_OMIT_GDN_STATE, "1"),
    // The MoE family's B4 commit path and glue: the GatedDelta commit replays
    // only accepted steps and is submitted from GPU accepted counts, residual
    // add + post norm run as one dispatch (same bits), rows near their limit
    // stay in the ragged batch. Short B4: windows 45.33 -> 44.44 ms, 190.7 ->
    // 193.8 tok/s, six-task outputs unchanged.
    (settings::GDN_COMMIT_KERNEL, "1"),
    (settings::DFLASH2_GPU_COMMIT, "1"),
    (settings::FUSED_GLUE, "1"),
    (settings::DFLASH2_RAGGED_TAIL, "1"),
    // Narrower ragged windows fill 16 verify rows (budgets 3 / 4 / 7 at four /
    // three / two rows; the verify projections cost the same up to 16 rows):
    // short B4 194.1 -> 197.1 tok/s, 2.60 -> 2.65 tokens per window, outputs
    // unchanged.
    (settings::DFLASH2_RAGGED_FILL_ROWS, "16"),
    // The last waiting long prompt waits up to 10 s for the decoding streams
    // (B4 30K: a low-acceptance stream needs ~6 s for its 256 tokens; past
    // the 3 s bound it stalled for a whole 37 s prefill).
    (settings::DFLASH2_LONG_ADMISSION_LAST_DEFER_MS, "10000"),
    // Long prompts wait up to 0.14 ms per prompt token past 8192 (30K ~3.1 s,
    // 61K ~7.4 s, 128K ~17 s): a stream decoding 256 tokens needs up to ~7 s
    // at 61K and ~11 s at 128K, and past 3 s it stalled for a whole prefill
    // (B4 128K E2E median 836 -> 642 s with 0.1 ms per token).
    (settings::DFLASH2_LONG_ADMISSION_DEFER_US_PER_TOKEN, "140"),
    // SwiGLU in the gate / up tiled GEMM's epilogue (same bits): 2048 x 5120
    // -> 34816 + SwiGLU 12.67 -> 12.12 ms.
    (settings::DENSE_PREFILL_FUSED_SWIGLU, "1"),
];

/// DFlash2 serve defaults of the profile, applied only when the
/// corresponding option is not given explicitly.
pub const DFLASH2_TREE_MAX_NODES: usize = 15;
pub const DFLASH2_ADMISSION_DEADLINE_MS: u64 = 0;

/// Minimum Apple GPU generation of the profile (M5).
pub const MIN_GPU_GENERATION: u32 = 17;

/// Diagnostic only: pretend the GPU has this architecture name when
/// resolving the profile (for fallback tests on M5 hardware).
pub const ARCHITECTURE_OVERRIDE_ENV: &str = "IRONMLX_DIAGNOSTIC_M5_PROFILE_ARCHITECTURE";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum M5ProfileMode {
    Auto,
    Off,
}

impl M5ProfileMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Off => "off",
        }
    }
}

impl std::str::FromStr for M5ProfileMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "auto" => Ok(Self::Auto),
            "off" => Ok(Self::Off),
            other => Err(format!("expected `auto` or `off`, got `{other}`")),
        }
    }
}

/// Neural Engine use (`--ane`): `auto` follows the active profile (the Dense
/// prefill MLP share of full chunks), `off` keeps all computation on the GPU.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AneMode {
    #[default]
    Auto,
    Off,
}

impl AneMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Off => "off",
        }
    }
}

impl std::str::FromStr for AneMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "auto" => Ok(Self::Auto),
            "off" => Ok(Self::Off),
            other => Err(format!("expected `auto` or `off`, got `{other}`")),
        }
    }
}

/// Whether the Neural Engine takes part in serving.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AneStatus {
    /// Dense prefill MLPs share full chunks with the Neural Engine.
    Active,
    /// `--ane off`.
    Disabled,
    /// The profile is not active for a Dense target (MoE targets have no
    /// dense MLP to share), so nothing would use the Neural Engine.
    NotApplicable,
    /// Applicable, but the effective share is zero (an explicit variable).
    Inactive,
}

impl AneStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::Disabled => "disabled",
            Self::NotApplicable => "not_applicable",
            Self::Inactive => "inactive",
        }
    }
}

/// Pure Neural Engine decision from the mode, whether the Dense profile is
/// active, and the effective share in units.
pub fn resolve_ane(ane: AneMode, dense_profile_active: bool, units: usize) -> AneStatus {
    if ane == AneMode::Off {
        AneStatus::Disabled
    } else if !dense_profile_active {
        AneStatus::NotApplicable
    } else if units == 0 {
        AneStatus::Inactive
    } else {
        AneStatus::Active
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum M5ProfileStatus {
    Active,
    /// `--m5-dflash2-profile off`.
    Disabled,
    /// Not DFlash2 serving.
    NotDFlash2,
    /// The GPU is older than generation 17.
    UnsupportedGpu,
    /// The GPU architecture could not be read.
    GpuUnknown,
    /// DFlash2 serving whose target model family was not qualified with the
    /// profile settings (they were tuned for the dense Qwen3.8 lane).
    TargetNotQualified,
}

impl M5ProfileStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::Disabled => "disabled",
            Self::NotDFlash2 => "not_dflash2",
            Self::UnsupportedGpu => "unsupported_gpu",
            Self::GpuUnknown => "gpu_unknown",
            Self::TargetNotQualified => "target_not_qualified",
        }
    }
}

/// Apple GPU generation from an architecture name such as `applegpu_g17s`.
pub fn gpu_generation(architecture: &str) -> Option<u32> {
    architecture
        .strip_prefix("applegpu_g")?
        .chars()
        .take_while(char::is_ascii_digit)
        .collect::<String>()
        .parse()
        .ok()
}

/// Pure profile decision.
pub fn resolve(mode: M5ProfileMode, dflash2: bool, architecture: Option<&str>) -> M5ProfileStatus {
    resolve_for_target(mode, dflash2, true, architecture)
}

/// Pure profile decision for a DFlash2 target whose model family may not be
/// covered by the profile qualification.
pub fn resolve_for_target(
    mode: M5ProfileMode,
    dflash2: bool,
    target_qualified: bool,
    architecture: Option<&str>,
) -> M5ProfileStatus {
    if mode == M5ProfileMode::Off {
        return M5ProfileStatus::Disabled;
    }
    if !dflash2 {
        return M5ProfileStatus::NotDFlash2;
    }
    if !target_qualified {
        return M5ProfileStatus::TargetNotQualified;
    }
    match architecture.map(gpu_generation) {
        None | Some(None) => M5ProfileStatus::GpuUnknown,
        Some(Some(generation)) if generation >= MIN_GPU_GENERATION => M5ProfileStatus::Active,
        Some(Some(_)) => M5ProfileStatus::UnsupportedGpu,
    }
}

/// Outcome of configuring one prefill kernel library.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrefillLibrary {
    pub name: &'static str,
    /// `"profile"` (embedded copy), `"environment"` (explicit path) or
    /// `"off"`.
    pub source: &'static str,
    pub path: Option<String>,
    pub error: Option<String>,
}

#[derive(Clone, Debug)]
pub struct M5Profile {
    pub mode: M5ProfileMode,
    pub architecture: Option<String>,
    pub status: M5ProfileStatus,
    pub target: M5ProfileTarget,
    pub prefill_libraries: Vec<PrefillLibrary>,
    pub ane: AneMode,
}

impl M5Profile {
    pub fn is_active(&self) -> bool {
        self.status == M5ProfileStatus::Active
    }

    /// Whether the Neural Engine takes part (see [`AneStatus`]).
    pub fn ane_status(&self) -> AneStatus {
        let units = setting(settings::DENSE_PREFILL_ANE_FFN_UNITS)
            .and_then(|value| value.parse().ok())
            .unwrap_or(0);
        resolve_ane(
            self.ane,
            self.is_active() && self.target == M5ProfileTarget::Qwen35Dense,
            units,
        )
    }

    /// Effective value of every profile setting of the family (`None` =
    /// off/unset).
    pub fn settings(&self) -> Vec<(&'static str, Option<String>)> {
        self.target
            .values()
            .iter()
            .map(|(name, _)| (*name, effective_setting(name)))
            .collect()
    }

    /// Settings of the family requested with `1` that do not take effect
    /// because the real GPU lacks the tensor units they need (for example an
    /// explicit variable on a GPU older than generation 17).
    pub fn hardware_ignored_settings(&self) -> Vec<&'static str> {
        hardware_ignored(
            self.target.values().iter().map(|(name, _)| *name),
            setting,
            hardware_architecture().as_deref(),
        )
    }

    /// `name=value` of every setting with an effective value, or `none`:
    /// what the profile (or an explicit variable) actually turns on, as
    /// opposed to the profile merely being installed.
    pub fn effective_settings_summary(&self) -> String {
        let effective: Vec<String> = self
            .settings()
            .into_iter()
            .filter_map(|(name, value)| value.map(|value| format!("{name}={value}")))
            .collect();
        if effective.is_empty() {
            "none".to_owned()
        } else {
            effective.join(",")
        }
    }
}

static INSTALLED: OnceLock<M5Profile> = OnceLock::new();

/// The installed profile, if any.
pub fn installed() -> Option<&'static M5Profile> {
    INSTALLED.get()
}

pub fn active() -> bool {
    installed().is_some_and(M5Profile::is_active)
}

fn setting_with(
    explicit: Option<String>,
    active_values: Option<&[(&str, &str)]>,
    name: &str,
) -> Option<String> {
    if explicit.is_some() {
        return explicit;
    }
    active_values
        .and_then(|values| values.iter().find(|(key, _)| *key == name))
        .map(|(_, value)| (*value).to_owned())
}

/// `--ane off` keeps every Neural Engine share at zero, above an explicit
/// variable and the profile table.
fn ane_override(ane: AneMode, name: &str) -> Option<String> {
    (ane == AneMode::Off && name == settings::DENSE_PREFILL_ANE_FFN_UNITS).then(|| "0".to_owned())
}

/// Effective value of a profile setting: `--ane off` for the Neural Engine
/// share, otherwise an explicitly set variable, otherwise the active
/// profile's value for its target family, otherwise unset.
pub fn setting(name: &str) -> Option<String> {
    let profile = installed();
    if let Some(value) = profile.and_then(|profile| ane_override(profile.ane, name)) {
        return Some(value);
    }
    let active_values = profile
        .filter(|profile| profile.is_active())
        .map(|profile| profile.target.values());
    setting_with(std::env::var(name).ok(), active_values, name)
}

/// Boolean profile setting (`"1"` = on).
pub fn flag(name: &str) -> bool {
    setting(name).as_deref() == Some("1")
}

/// Settings that run M5 tensor-unit kernels: besides being set they need a
/// real Apple GPU of generation [`MIN_GPU_GENERATION`] or newer.
const TENSOR_UNIT_SETTINGS: &[&str] = &[settings::M5_MOE_AFFINE4_PROJECTIONS];

/// Metal architecture of the running GPU as MLX reports it. Unlike
/// [`current_architecture`], the diagnostic override is ignored: hardware
/// gates follow the real device.
pub fn hardware_architecture() -> Option<String> {
    static ARCHITECTURE: OnceLock<Option<String>> = OnceLock::new();
    ARCHITECTURE
        .get_or_init(|| mlx::metal::architecture().ok())
        .clone()
}

/// Whether a tensor-unit setting with `value` takes effect on a GPU whose
/// real architecture is `hardware_architecture`: it must be `1` and the
/// architecture a known Apple GPU of generation 17 or newer.
fn tensor_unit_setting_effective(value: Option<&str>, hardware_architecture: Option<&str>) -> bool {
    value == Some("1")
        && hardware_architecture
            .and_then(gpu_generation)
            .is_some_and(|generation| generation >= MIN_GPU_GENERATION)
}

/// Names among `names` that are tensor-unit settings requested with `1` (per
/// `value_of`) but that `hardware_architecture` cannot run.
fn hardware_ignored(
    names: impl Iterator<Item = &'static str>,
    value_of: impl Fn(&str) -> Option<String>,
    hardware_architecture: Option<&str>,
) -> Vec<&'static str> {
    names
        .filter(|name| {
            let value = value_of(name);
            TENSOR_UNIT_SETTINGS.contains(name)
                && value.as_deref() == Some("1")
                && !tensor_unit_setting_effective(value.as_deref(), hardware_architecture)
        })
        .collect()
}

/// [`flag`] for a setting that runs M5 tensor-unit kernels: an explicit
/// variable cannot enable it on a GPU without them.
pub fn tensor_unit_flag(name: &str) -> bool {
    tensor_unit_setting_effective(setting(name).as_deref(), hardware_architecture().as_deref())
}

/// [`setting`] as it takes effect: a tensor-unit setting the GPU cannot run
/// reports as unset.
fn effective_setting(name: &str) -> Option<String> {
    let value = setting(name);
    if TENSOR_UNIT_SETTINGS.contains(&name)
        && value.as_deref() == Some("1")
        && !tensor_unit_flag(name)
    {
        return None;
    }
    value
}

/// GPU architecture used for the profile decision.
pub fn current_architecture() -> Option<String> {
    std::env::var(ARCHITECTURE_OVERRIDE_ENV)
        .ok()
        .or_else(|| mlx::metal::architecture().ok())
}

/// Resolve and install the process profile (first call wins) and configure
/// the prefill kernels accordingly. Call before loading a model.
pub fn install(mode: M5ProfileMode, dflash2: bool) -> &'static M5Profile {
    install_for_target(mode, dflash2, M5ProfileTarget::Qwen35Dense)
}

/// [`install`] for a DFlash2 target family. An unqualified target keeps
/// every profile setting off; each qualified family uses only its own table.
pub fn install_for_target(
    mode: M5ProfileMode,
    dflash2: bool,
    target: M5ProfileTarget,
) -> &'static M5Profile {
    install_for_target_with_ane(mode, AneMode::Auto, dflash2, target)
}

/// [`install_for_target`] with the Neural Engine mode (`--ane`).
pub fn install_for_target_with_ane(
    mode: M5ProfileMode,
    ane: AneMode,
    dflash2: bool,
    target: M5ProfileTarget,
) -> &'static M5Profile {
    INSTALLED.get_or_init(|| {
        let architecture = current_architecture();
        let status = resolve_for_target(
            mode,
            dflash2,
            target != M5ProfileTarget::Unqualified,
            architecture.as_deref(),
        );
        // The prefill kernel libraries and masked softmax belong to the
        // dense lane qualification.
        let active = status == M5ProfileStatus::Active && target == M5ProfileTarget::Qwen35Dense;
        // Long causal prefill chunks of the MoE target take the D256 NAX
        // attention from its minimum key count (more accurate than MLX's D256
        // fallback against an f32 reference, and faster on long prompts).
        let moe_active = status == M5ProfileStatus::Active && target == M5ProfileTarget::Qwen36Moe;
        // The long-prompt prefill settings below come from the active
        // family's table: the MoE table, or the Dense one.
        let long_prefill_values = if moe_active {
            Some(MOE_PROFILE_VALUES)
        } else {
            active.then_some(PROFILE_VALUES)
        };
        let d256_min_keys = setting_with(
            std::env::var(settings::PREFILL_D256_NAX_MIN_KEYS).ok(),
            long_prefill_values,
            settings::PREFILL_D256_NAX_MIN_KEYS,
        )
        .and_then(|value| value.parse::<i32>().ok())
        .filter(|&keys| keys > 0);
        // Decode and verify query blocks of the MoE target take the split-K
        // D256 NAX attention from its minimum key count (a verify grouping
        // boundary, so no query group straddles it).
        let decode_min_keys = setting_with(
            std::env::var(settings::DECODE_D256_NAX_MIN_KEYS).ok(),
            long_prefill_values,
            settings::DECODE_D256_NAX_MIN_KEYS,
        )
        .and_then(|value| value.parse::<i32>().ok())
        .filter(|&keys| keys > 0);
        // MoE prefill projections take the BM128 QMM M tile at every shape.
        let qmm_any_shape = setting_with(
            std::env::var(settings::PREFILL_QMM_MTILE_ANY_SHAPE).ok(),
            long_prefill_values,
            settings::PREFILL_QMM_MTILE_ANY_SHAPE,
        )
        .as_deref()
            == Some("1");
        let prefill_libraries = vec![
            configure_library(
                "prefill_qmm_mtile",
                settings::PREFILL_QMM_MTILE_METALLIB,
                mlx::metal::prefill_shaders::PREFILL_QMM_MTILE_METALLIB,
                active || qmm_any_shape,
                mlx::metal::set_prefill_qmm_mtile_library,
            ),
            configure_library(
                "prefill_d256_nax",
                settings::PREFILL_D256_NAX_METALLIB,
                mlx::metal::prefill_shaders::PREFILL_D256_NAX_METALLIB,
                active || d256_min_keys.is_some() || decode_min_keys.is_some(),
                mlx::metal::set_prefill_d256_nax_library,
            ),
        ];
        mlx::metal::set_prefill_d256_nax_min_keys(d256_min_keys.unwrap_or(0));
        mlx::metal::set_decode_d256_nax_min_keys(decode_min_keys.unwrap_or(0));
        mlx::metal::set_prefill_qmm_mtile_any_shape(qmm_any_shape);
        let masked_softmax = setting_with(
            std::env::var(settings::PREFILL_MASKED_SOFTMAX).ok(),
            active.then_some(PROFILE_VALUES),
            settings::PREFILL_MASKED_SOFTMAX,
        )
        .as_deref()
            == Some("1");
        mlx::metal::set_prefill_masked_softmax(masked_softmax);
        M5Profile {
            mode,
            architecture,
            status,
            target,
            prefill_libraries,
            ane,
        }
    })
}

fn configure_library(
    name: &'static str,
    env: &str,
    embedded: &[u8],
    active: bool,
    set: fn(&str) -> bool,
) -> PrefillLibrary {
    let (source, path, error) = if let Ok(path) = std::env::var(env) {
        ("environment", Some(path), None)
    } else if active {
        match materialize_library(name, embedded) {
            Ok(path) => ("profile", Some(path), None),
            Err(error) => ("off", None, Some(error.to_string())),
        }
    } else {
        ("off", None, None)
    };
    let applied = set(path.as_deref().unwrap_or(""));
    let error = error.or_else(|| (!applied).then(|| "configured after first use".to_owned()));
    PrefillLibrary {
        name,
        source,
        path,
        error,
    }
}

/// Write an embedded library to a content-addressed file (MLX loads
/// libraries by path) and return the path. An existing file is reused only
/// if its bytes are identical.
fn materialize_library(name: &str, bytes: &[u8]) -> std::io::Result<String> {
    let dir = std::env::temp_dir().join("ironmlx-prefill-kernels");
    std::fs::create_dir_all(&dir)?;
    let path = dir.join(format!(
        "{name}-{}-{:016x}.metallib",
        bytes.len(),
        fnv1a64(bytes)
    ));
    if std::fs::read(&path).ok().as_deref() != Some(bytes) {
        let staging = dir.join(format!(
            ".{name}-{}-{}.tmp",
            std::process::id(),
            fnv1a64(bytes)
        ));
        std::fs::write(&staging, bytes)?;
        std::fs::rename(&staging, &path)?;
    }
    path.into_os_string()
        .into_string()
        .map_err(|_| std::io::Error::other("non-UTF-8 kernel cache path"))
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0100_0000_01b3)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_unit_settings_require_a_real_g17_gpu() {
        // Supported hardware, explicitly enabled.
        assert!(tensor_unit_setting_effective(
            Some("1"),
            Some("applegpu_g17s")
        ));
        assert!(tensor_unit_setting_effective(
            Some("1"),
            Some("applegpu_g18g")
        ));
        // Older GPU, unknown or unreadable architecture: an explicit `1`
        // does not override the hardware requirement.
        assert!(!tensor_unit_setting_effective(
            Some("1"),
            Some("applegpu_g16s")
        ));
        assert!(!tensor_unit_setting_effective(
            Some("1"),
            Some("applegpu_g13p")
        ));
        assert!(!tensor_unit_setting_effective(Some("1"), Some("unknown")));
        assert!(!tensor_unit_setting_effective(Some("1"), None));
        // Not requested.
        assert!(!tensor_unit_setting_effective(None, Some("applegpu_g17s")));
        assert!(!tensor_unit_setting_effective(
            Some("0"),
            Some("applegpu_g17s")
        ));
        // Only the MoE affine4 projections are gated; the dense settings keep
        // their own gates.
        assert_eq!(
            TENSOR_UNIT_SETTINGS,
            &[settings::M5_MOE_AFFINE4_PROJECTIONS]
        );
    }

    #[test]
    fn hardware_ignored_reports_tensor_unit_settings_the_gpu_cannot_run() {
        let moe = || {
            M5ProfileTarget::Qwen36Moe
                .values()
                .iter()
                .map(|(name, _)| *name)
        };
        let dense = || {
            M5ProfileTarget::Qwen35Dense
                .values()
                .iter()
                .map(|(name, _)| *name)
        };
        let requested =
            |name: &str| (name == settings::M5_MOE_AFFINE4_PROJECTIONS).then(|| "1".to_owned());
        // An explicit MoE affine4 request on a GPU without tensor units.
        assert_eq!(
            hardware_ignored(moe(), requested, Some("applegpu_g13p")),
            vec![settings::M5_MOE_AFFINE4_PROJECTIONS]
        );
        assert_eq!(
            hardware_ignored(moe(), requested, None),
            vec![settings::M5_MOE_AFFINE4_PROJECTIONS]
        );
        // The GPU runs it, it is not requested, or the family does not use it.
        assert!(hardware_ignored(moe(), requested, Some("applegpu_g17s")).is_empty());
        assert!(hardware_ignored(moe(), |_| None, Some("applegpu_g13p")).is_empty());
        assert!(hardware_ignored(dense(), requested, Some("applegpu_g13p")).is_empty());
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn hardware_gate_ignores_the_diagnostic_architecture_override() {
        // The diagnostic override feeds the profile decision only; the
        // hardware gate reads MLX's real architecture.
        let previous = std::env::var(ARCHITECTURE_OVERRIDE_ENV).ok();
        std::env::set_var(ARCHITECTURE_OVERRIDE_ENV, "applegpu_g99s");
        let hardware = hardware_architecture();
        assert_ne!(hardware.as_deref(), Some("applegpu_g99s"));
        assert_eq!(current_architecture().as_deref(), Some("applegpu_g99s"));
        match previous {
            Some(value) => std::env::set_var(ARCHITECTURE_OVERRIDE_ENV, value),
            None => std::env::remove_var(ARCHITECTURE_OVERRIDE_ENV),
        }
    }

    #[test]
    fn generation_parses_apple_gpu_names() {
        assert_eq!(gpu_generation("applegpu_g17s"), Some(17));
        assert_eq!(gpu_generation("applegpu_g16p"), Some(16));
        assert_eq!(gpu_generation("applegpu_g18"), Some(18));
        assert_eq!(gpu_generation("apple_g13s"), None);
        assert_eq!(gpu_generation("applegpu_gx"), None);
    }

    #[test]
    fn profile_is_active_only_for_dflash2_on_g17_and_newer() {
        use M5ProfileMode::{Auto, Off};
        assert_eq!(
            resolve(Auto, true, Some("applegpu_g17s")),
            M5ProfileStatus::Active
        );
        assert_eq!(
            resolve(Auto, true, Some("applegpu_g18p")),
            M5ProfileStatus::Active
        );
        assert_eq!(
            resolve(Auto, true, Some("applegpu_g16s")),
            M5ProfileStatus::UnsupportedGpu
        );
        assert_eq!(
            resolve(Auto, true, Some("apple_g13s")),
            M5ProfileStatus::GpuUnknown
        );
        assert_eq!(resolve(Auto, true, None), M5ProfileStatus::GpuUnknown);
        assert_eq!(
            resolve(Auto, false, Some("applegpu_g17s")),
            M5ProfileStatus::NotDFlash2
        );
        assert_eq!(
            resolve_for_target(Auto, true, false, Some("applegpu_g17s")),
            M5ProfileStatus::TargetNotQualified
        );
        assert_eq!(
            resolve_for_target(Off, true, false, Some("applegpu_g17s")),
            M5ProfileStatus::Disabled
        );
        assert_eq!(
            resolve(Off, true, Some("applegpu_g17s")),
            M5ProfileStatus::Disabled
        );
        assert_eq!("off".parse::<M5ProfileMode>(), Ok(Off));
        assert!("on".parse::<M5ProfileMode>().is_err());
    }

    #[test]
    fn explicit_setting_overrides_profile_value() {
        let dense = Some(M5ProfileTarget::Qwen35Dense.values());
        let name = settings::DFLASH2_FIXED_BUDGET;
        assert_eq!(setting_with(None, dense, name).as_deref(), Some("7"));
        assert_eq!(setting_with(None, None, name), None);
        assert_eq!(
            setting_with(Some("3".into()), dense, name).as_deref(),
            Some("3")
        );
        assert_eq!(
            setting_with(Some("0".into()), dense, settings::M5_LANE_ATTN).as_deref(),
            Some("0")
        );
        assert_eq!(
            setting_with(Some("1".into()), None, settings::M5_LANE_ATTN).as_deref(),
            Some("1")
        );
        // The prefill library paths are not table values; they come from
        // the embedded copies at install.
        assert_eq!(
            setting_with(None, dense, settings::PREFILL_QMM_MTILE_METALLIB),
            None
        );
    }

    #[test]
    fn ane_mode_parses_and_off_zeroes_only_the_neural_engine_share() {
        assert_eq!("auto".parse::<AneMode>(), Ok(AneMode::Auto));
        assert_eq!("off".parse::<AneMode>(), Ok(AneMode::Off));
        assert!("on".parse::<AneMode>().is_err());
        assert_eq!(AneMode::default(), AneMode::Auto);
        let units = settings::DENSE_PREFILL_ANE_FFN_UNITS;
        assert_eq!(ane_override(AneMode::Off, units).as_deref(), Some("0"));
        assert_eq!(ane_override(AneMode::Auto, units), None);
        assert_eq!(
            ane_override(AneMode::Off, settings::DENSE_PREFILL_ANE_FFN_PIPELINE),
            None
        );
        assert_eq!(ane_override(AneMode::Off, settings::M5_LANE_ATTN), None);
        // `auto` keeps the Dense table's share.
        let dense = Some(M5ProfileTarget::Qwen35Dense.values());
        assert_eq!(setting_with(None, dense, units).as_deref(), Some("12"));
    }

    #[test]
    fn ane_status_follows_mode_profile_and_share() {
        assert_eq!(resolve_ane(AneMode::Auto, true, 12), AneStatus::Active);
        assert_eq!(resolve_ane(AneMode::Off, true, 12), AneStatus::Disabled);
        assert_eq!(resolve_ane(AneMode::Off, false, 0), AneStatus::Disabled);
        assert_eq!(
            resolve_ane(AneMode::Auto, false, 12),
            AneStatus::NotApplicable
        );
        assert_eq!(resolve_ane(AneMode::Auto, true, 0), AneStatus::Inactive);
        assert_eq!(AneStatus::NotApplicable.as_str(), "not_applicable");
    }

    #[test]
    fn moe_profile_never_inherits_dense_lane_settings() {
        let moe = Some(M5ProfileTarget::Qwen36Moe.values());
        for (name, _) in PROFILE_VALUES {
            if *name == settings::QEMBEDDING_RUNTIME_COUNT
                || *name == settings::MLX_CACHE_MAX_MIB
                || *name == settings::DFLASH2_RAGGED_LINEAR
                || *name == settings::M5_SHARED_WEIGHT_LAYOUT
                || *name == settings::M5_PRECOMPILE
                || *name == settings::DFLASH2_DRAFT_M5_RESIDENT
                || *name == settings::DFLASH2_RAGGED_BUDGET
                || *name == settings::DFLASH2_DRAFT_M5
                || *name == settings::DFLASH2_DRAFT_BATCHED
                || *name == settings::PREFILL_D256_NAX_MIN_KEYS
                || *name == settings::GDN_LONG_SCAN_MIN_POSITION
                || *name == settings::PREFILL_QMM_MTILE_ANY_SHAPE
                || *name == settings::DFLASH2_STARTUP_WARMUP
                || *name == settings::GDN_COMMIT_KERNEL
                || *name == settings::DFLASH2_GPU_COMMIT
                || *name == settings::FUSED_GLUE
                || *name == settings::DFLASH2_RAGGED_TAIL
            {
                continue;
            }
            assert_eq!(setting_with(None, moe, name), None, "{name}");
        }
        // Ragged multi-request batching is the MoE family's own setting
        // (row-exact ragged verify and one-token windows), not inherited.
        assert_eq!(
            setting_with(None, moe, settings::DFLASH2_RAGGED_LINEAR).as_deref(),
            Some("1")
        );
        assert_eq!(
            setting_with(None, moe, settings::DFLASH2_RAGGED_BUDGET).as_deref(),
            Some("3")
        );
        assert_eq!(
            setting_with(None, moe, settings::DFLASH2_RAGGED_SOLO_BUDGET).as_deref(),
            Some("3")
        );
        // Fully speculative: no batched one-token windows, no probes.
        for name in [
            settings::DFLASH2_RAGGED_ORDINARY_MIN_WIDTH,
            settings::DFLASH2_RAGGED_PROBE_INTERVAL,
        ] {
            assert_eq!(setting_with(None, moe, name), None, "{name}");
        }
        for name in [
            settings::DFLASH2_RAGGED_FAST_ARITHMETIC,
            settings::DFLASH2_RAGGED_TAIL,
            settings::DFLASH2_GPU_COMMIT,
            settings::DFLASH2_DRAFT_M5,
            settings::DFLASH2_DRAFT_BATCHED,
            settings::M5_MOE_AFFINE4_PROJECTIONS,
            settings::M5_MOE_HEAD,
            settings::M5_SHARED_WEIGHT_LAYOUT,
            settings::M5_PRECOMPILE,
            settings::MOE_TILE_EXPERTS,
            settings::MOE_M5_EXPERTS,
            settings::M5_TUNED_SPLIT_K,
            settings::M5_FUSED_XSUM,
            settings::FUSED_GLUE,
            settings::DFLASH2_DRAFT_M5_RESIDENT,
            settings::GDN_COMMIT_KERNEL,
            settings::DFLASH2_STARTUP_WARMUP,
        ] {
            assert_eq!(
                setting_with(None, moe, name).as_deref(),
                Some("1"),
                "{name}"
            );
        }
        // Shared settings with target-specific values.
        assert_eq!(
            setting_with(None, moe, settings::QEMBEDDING_RUNTIME_COUNT).as_deref(),
            Some("1")
        );
        assert_eq!(
            setting_with(None, moe, settings::MLX_CACHE_MAX_MIB).as_deref(),
            Some("512")
        );
        assert_eq!(
            setting_with(None, moe, settings::WIRED_WEIGHTS).as_deref(),
            Some("1")
        );
        assert_eq!(
            setting_with(
                None,
                Some(M5ProfileTarget::Qwen35Dense.values()),
                settings::WIRED_WEIGHTS
            ),
            None
        );
        assert!(M5ProfileTarget::Unqualified.values().is_empty());
    }

    #[test]
    fn embedded_libraries_match_the_qualified_kernels() {
        // Same size as the candidate's libraries; the byte identity is
        // checked against the frozen files by the product verification.
        assert_eq!(
            mlx::metal::prefill_shaders::PREFILL_QMM_MTILE_METALLIB.len(),
            70_104
        );
        assert_eq!(
            mlx::metal::prefill_shaders::PREFILL_D256_NAX_METALLIB.len(),
            121_240
        );
    }

    #[test]
    fn materialized_library_is_content_addressed_and_reused() {
        let bytes = b"ironmlx test library";
        let first = materialize_library("unit_test", bytes).expect("written");
        let second = materialize_library("unit_test", bytes).expect("reused");
        assert_eq!(first, second);
        assert_eq!(std::fs::read(&first).expect("readable"), bytes);
        let other = materialize_library("unit_test", b"other bytes").expect("written");
        assert_ne!(first, other);
        let _ = std::fs::remove_file(first);
        let _ = std::fs::remove_file(other);
    }
}
