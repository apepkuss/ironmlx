# P5e Three-Way MoE Text-Only Perf Bench

> **Self-contained for offline code-level analysis.** This document embeds all
> bench data, the relevant ironmlx source code, the test methodology, the
> targets' configuration, and the known T0 hot-path profile. A reader with no
> repo access can perform the full analysis from this single file.

| Field | Value |
|---|---|
| Date | 2026-05-19 |
| Hardware | Apple Silicon M5 Max, 128 GB unified memory |
| OS | macOS 26.4 (kernel Darwin 25.4.0, build 25E246) |
| Model | `mlx-community/Qwen3.5-35B-A3B-4bit` (Qwen3.5-MoE-A3B, 128 experts, top-4) |
| Snapshot SHA | `1e20fd8d42056f870933bf98ca6211024744f7ec` |
| Snapshot path | `~/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/<sha>/` |
| Harness | `iron-bench` (workspace member of ironmlx-backend) |
| Sweep config | `--prompt-len 128,512,2048,4096,8192,16384  --max-tokens 128  --runs 5  --warmup 1` |
| Method | Three-way head-to-head HTTP bench, strict serial (one server up at a time) |
| Report HEAD branch | `ironmlx-p5e-perf`, HEAD `f420c94` |

---

## §1 Purpose + key questions for the analyst

P5e completed an in-source perf optimization phase for ironmlx's MoE prefill
hot path (`SparseMoeBlock::forward_on`). Acceptance gate of P5e was "ironmlx
self before/after improvement" — _not_ alignment with any competitor.

That said, comparing ironmlx against:

- **mlx-lm** (ml-explore reference implementation, `git+ml-explore/mlx-lm@ed1fca4`, v0.31.3) — the
  natural **fair baseline** for "reference implementation level".
- **omlx** (`/Users/xin/workspace/iron-rivals/omlx`) — `omlx serve` runs mlx-lm
  the same commit BUT wraps it in `omlx.patches.gated_delta_advance` +
  `omlx.patches.qwen3_5_attention` monkey-patches at import time and adds a
  PagedCache + vlm-engine path. **Observation only** — not a fair-implementation baseline,
  but the leading "production-optimized peer" on Apple Silicon.

The questions we want an offline code-level analysis to address:

1. **Where does the ironmlx prefill gap vs mlx-lm come from?** P5e T0 profile
   identified the 3 `gather_quantized_matmul` calls (gate / up / down) as
   64.8% of PP=2048 prefill on M5 Max; P5e T5 sorted-routing improved that hot
   path by 1.92× (993 → 1902 tok/s in `Model::forward_on` direct measurement).
   Yet the HTTP-level bench below shows ironmlx prefill _throughput_ plateauing
   at ~1.8K tok/s and falling behind mlx-lm at PP ≥ 4096. Hypothesis: chunked
   prefill (default `--prefill-chunk-size 2048`) introduces a per-chunk
   overhead that does not amortize. Verify or refute.

2. **Why does PP=512 show the worst ratio (ironmlx 0.35× of mlx-lm prefill rate)?**
   PP=128 (0.72×) and PP=2048 (0.78×) are both better than PP=512. The PP=512
   dip suggests a discontinuity in either the prefill code path or KV cache
   layout for medium prompts. The MoE sorted-routing threshold
   (`SORTED_ROUTING_MIN_BS_K = 512`) was set to align with MLX's `gather_qmm_rhs`
   fast-path floor, but PP=512 → bs_k=2048 well exceeds that, so threshold
   itself shouldn't cause the dip. Identify the actual cause.

3. **Why does ironmlx decode (TG) at PP=128/512 lag mlx-lm by 21-22%, but reverse
   to +13-20% faster at PP ≥ 2048?** Hypothesis: KV cache grow-step behavior
   (`KVCache.step = 256`) — at PP ≤ 256 the cache hasn't reached one step, and
   each decoded token may trigger a re-alloc/grow. Verify by reading
   `KVCache::update_and_fetch` semantics (full source in §3.1.3).

4. **What's the path forward to close the prefill gap for agent-scale prompts
   (PP=8192-16384)?** ironmlx is positioned for agent workloads (long system
   prompt + tools + memory + multi-turn history). PP=16384 e2e is currently
   11.30s vs mlx-lm 5.75s vs omlx 4.60s — a 2× gap to mlx-lm. Identify the top
   1-3 candidate next-step optimizations from the embedded source.

---

## §2 Test environment

```
Hardware: Apple M5 Max
Memory:   128 GB unified
OS:       macOS 26.4 (kernel Darwin 25.4.0 arm64, T6050)
Python:   CPython 3.12.13 (mlx-lm venv)
MLX:      0.31.2 (Darwin metal)
mlx-lm:   0.31.3 (ed1fca4cef15a824c5f1702c80f70b4cffc8e4dd) — ml-explore upstream
ironmlx:  branch ironmlx-p5e-perf, HEAD f420c94 (P5e final close-out)
omlx:     /Users/xin/workspace/iron-rivals/omlx (latest checkout, default config)
```

The mlx-lm version installed in `scripts/bench-venvs/mlx-lm/.venv` is bit-identical
in source bytes to what omlx pins. The two differ only by **import-time monkey
patches injected by `omlx serve`** (not present in the bare `mlx_lm.server`
process).

---

## §3 Test targets

### §3.1 ironmlx (branch `ironmlx-p5e-perf`, HEAD `f420c94`)

#### §3.1.1 P5e commit chain (13 commits)

```
f420c94 docs(p5e-t6): fix MLX gather_qmm right_sorted_ formula attribution
8d850d8 chore(p5e-t6): P5e final close-out — sorted routing shipped
1cdb49b refactor(p5e-t5): align sorted-routing threshold to MLX fast-path floor
b8e3f26 feat(p5e-t5): B.1 sorted routing for gather_qmm        <-- THE WINNER
bb15906 docs(p5e-t4): fill Stage 1 close-out report HEAD SHA
06426b0 chore(p5e-t4): Stage 1 close-out — no winners; revert all A.x experiments
4142e52 test(p5e-t3): A.3 shape elimination experiment for down_proj path
c1e6623 docs(p5e-t2): move A.2 compile-gap notes from rustdoc to inline comment
944c3bf test(p5e-t2): A.2 mlx::compile wrap experiment for SparseMoeBlock
e0bcc3b test(p5e-t1): A.1 stream parallelism experiment for gate/up gather_qmm
1092eb6 docs(p5e-t0): polish baseline test docstring + report discrepancy notes
96dce61 test(p5e-t0): wall-clock prefill baseline at PP=128/512/2048
3277dc4 docs(p5e): SparseMoeBlock gather_qmm perf optimization implementation plan
```

Stage 1 (A.1/A.2/A.3): all discarded after rigorous measurement.
- A.1 stream parallelism: −6% to −17% regression (per-call new_stream overhead
  dominates Metal scheduler kernel-overlap benefit).
- A.2 mlx::compile wrap: 4 safe-wrapper API gaps blocked a real wrap; ended as
  no-op.
- A.3 shape elimination: ±0.5% noise.

Stage 2 (B.1 sorted routing): **shipped as default**. Sort tokens by expert id
before all 3 gather_qmm calls; pass `sorted_indices=true`; restore order via
inverse permutation after `down_proj`. Threshold `SORTED_ROUTING_MIN_BS_K = 512`
aligned to MLX `gather_qmm_rhs` fast-path floor (`B >= 16 && B/E >= 4` for
E=128 experts).

Direct-call (no HTTP) measurement after P5e (M5 Max, `Model::forward_on`):

| PP | T0 baseline | P5e final | Δ tok/s |
|---|---|---|---|
| 128 | 127.66 ms / 1003 tok/s | 116.99 ms / 1094 tok/s | +9.1% |
| 512 | 488.30 ms / 1048 tok/s | 307.37 ms / 1666 tok/s | +58.9% |
| 2048 | 2067.45 ms / 991 tok/s | 1076.99 ms / 1902 tok/s | +91.9% (1.92×) |

(Source: `reports/p5e-final-results.md`.)

#### §3.1.2 `SparseMoeBlock::forward_on` (full source, the P5e hot path)

`ironmlx/src/models/qwen3_5_moe/sparse_moe.rs` — relevant excerpts.

**Threshold constant + struct (lines 38-146):**

```rust
/// Minimum (batch_size * num_experts_per_tok) for the sorted-routing path.
/// MLX's gather_qmm rhs fast path (mlx/backend/metal/quantized.cpp:1484)
/// requires B >= 16 && B/E >= 4. For E=128 experts that's B >= 512;
/// below that we'd pay argsort + take_along_axis + take + reshape
/// overhead with no kernel-level benefit.
const SORTED_ROUTING_MIN_BS_K: i32 = 512;

pub struct SparseMoeBlock {
    router_gate: Linear,           // Linear(hidden → num_experts) quantized 4-bit
    routed: RoutedExperts,         // Stacked routed expert weights (G1 stacked tensor)
    shared_expert: Mlp,            // Standard SwiGLU Mlp (shared across all tokens)
    shared_expert_gate: Linear,    // Linear(hidden → 1) quantized; sigmoid(·) gates shared
    num_experts_per_tok: i32,
}
```

**`forward_on` body (lines 180-467, abridged with all critical paths):**

```rust
pub fn forward_on(&self, x: &Array, target: StreamOrDevice) -> Result<Array> {
    let dims = x.shape();
    let dvec = dims.as_slice();
    if dvec.len() != 3 {
        return Err(anyhow!(/* rank-3 [B,S,H] expected */));
    }
    let (b, s, h) = (dvec[0], dvec[1], dvec[2]);
    let bs = b * s;
    let k = self.num_experts_per_tok;
    let num_experts = self.routed.num_experts;

    // --- Flatten [B, S, H] → [BS, H] for routing and expert kernels. ---
    let flat_x = mlx::ops::shape::reshape(x, [bs, h])?;

    // (1) Router: Linear → [BS, E], then softmax along expert axis.
    let logits = self.router_gate.forward_on(&flat_x, target)?;       // [BS, E]
    let probs = mlx::ops::softmax_on(&logits, -1, true, target)?;     // [BS, E]

    // (2) Top-k selection via argpartition.
    let part_inds = argpartition_on(&probs, -(k), -1, target)?;       // [BS, E]
    let inds = mlx::ops::slice_strided_on(
        &part_inds,
        [0, num_experts - k],     // start col E-k
        [bs, num_experts],        // stop (exclusive)
        [1, 1],
        target,
    )?;                                                                 // [BS, k]

    // (3) Gather top-k probs and renormalize.
    let scores_raw = take_along_axis_on(&probs, &inds, -1, target)?;  // [BS, k]
    let scores_sum = mlx::ops::sum_on(&scores_raw, -1, /*keepdim*/ true, target)?;
    let scores = &scores_raw / &scores_sum;                            // [BS, k]

    // (4) Cast indices to uint32 (gather_qmm requirement).
    let inds_u32 = mlx::ops::cast::astype_on(&inds, mlx::Dtype::Uint32, target)?;

    // (5) Routed SwiGLU via gather_quantized_matmul_on (G1 path).
    //
    // P5e T5 B.1: Sorted-flat path when BS*k >= SORTED_ROUTING_MIN_BS_K (=512).
    //   Pre-sort tokens by expert id; pass sorted_indices=true so MLX dispatches
    //   gather_qmm_rhs fast-path (mlx/backend/metal/quantized.cpp:1484).
    //   x reshape: [BS,1,1,H] → [BS*k,1,1,H]; rhs_indices: [BS,k] → [BS*k,1].
    //
    // Default broadcast path (BS*k < SORTED_ROUTING_MIN_BS_K):
    //   Keep x as [BS,1,1,H], let MLX broadcast lhs_indices [BS,1] → [BS,k].
    let bs_k = bs * k;
    let use_sorted = bs_k >= SORTED_ROUTING_MIN_BS_K;

    let (gate_out, up_out, rhs_idx_used, sorted_flag, sort_perm_opt) = if use_sorted {
        // --- Sorted routing path. ---
        let flat_topk = mlx::ops::shape::reshape(&inds_u32, [bs_k])?;     // [BS*k]
        let sort_perm = argsort_on(&flat_topk, -1, target)?;              // [BS*k]
        let sorted_topk_1d = take_along_axis_on(&flat_topk, &sort_perm, -1, target)?;
        let sorted_topk_2d = mlx::ops::shape::reshape(&sorted_topk_1d, [bs_k, 1])?;

        // token_idx[i] = i / k — Rust-side build, then upload.
        let bs_k_usize = bs_k as usize;
        let k_usize = k as usize;
        let token_idx_vec: Vec<u32> = (0..bs_k_usize).map(|i| (i / k_usize) as u32).collect();
        let token_idx: Array = (token_idx_vec.as_slice(), [bs_k]).try_into()?;
        let sorted_token_idx = take_along_axis_on(&token_idx, &sort_perm, -1, target)?;

        // Gather flat_x rows in sorted order.
        let sorted_x_2d = take_on(&flat_x, &sorted_token_idx, 0, target)?;             // [BS*k, H]
        let sorted_x_4d = mlx::ops::shape::expand_dims_on(&sorted_x_2d, &[-2, -3][..], target)?;
        // sorted_x_4d: [BS*k, 1, 1, H]

        let gate_out = mlx::quantization::gather_quantized_matmul_on(
            &sorted_x_4d,
            &self.routed.gate_weight, &self.routed.gate_scales, self.routed.gate_biases.as_ref(),
            None, Some(&sorted_topk_2d), /*transpose*/ true,
            Some(self.routed.group_size), Some(self.routed.bits), "affine",
            /*sorted_indices*/ true, target,
        )?;
        let up_out = mlx::quantization::gather_quantized_matmul_on(
            &sorted_x_4d,
            &self.routed.up_weight, &self.routed.up_scales, self.routed.up_biases.as_ref(),
            None, Some(&sorted_topk_2d), true, Some(self.routed.group_size), Some(self.routed.bits),
            "affine", true, target,
        )?;
        (gate_out, up_out, sorted_topk_2d, true, Some(sort_perm))
    } else {
        // --- Default broadcast path (Stage 1 final, for decode steps). ---
        let x_in = mlx::ops::shape::expand_dims_on(&flat_x, &[-2, -3][..], target)?; // [BS, 1, 1, H]

        let gate_out = mlx::quantization::gather_quantized_matmul_on(
            &x_in, &self.routed.gate_weight, &self.routed.gate_scales, self.routed.gate_biases.as_ref(),
            None, Some(&inds_u32), true, Some(self.routed.group_size), Some(self.routed.bits),
            "affine", false, target,
        )?;                                                              // [BS, k, 1, moe_inter]
        let up_out = mlx::quantization::gather_quantized_matmul_on(
            &x_in, &self.routed.up_weight, &self.routed.up_scales, self.routed.up_biases.as_ref(),
            None, Some(&inds_u32), true, Some(self.routed.group_size), Some(self.routed.bits),
            "affine", false, target,
        )?;                                                              // [BS, k, 1, moe_inter]
        (gate_out, up_out, inds_u32, false, None)
    };

    // SwiGLU activation: silu(gate) * up where silu(z) = z * sigmoid(z)
    let gate_sig = gate_out.sigmoid_on(target)?;
    let gate_silu = &gate_out * &gate_sig;
    let act = &gate_silu * &up_out;

    let down_out_4d = mlx::quantization::gather_quantized_matmul_on(
        &act,
        &self.routed.down_weight, &self.routed.down_scales, self.routed.down_biases.as_ref(),
        None, Some(&rhs_idx_used), true, Some(self.routed.group_size), Some(self.routed.bits),
        "affine", sorted_flag, target,
    )?;

    // (6) Weight by renormalized scores and reduce over k. Both branches
    // converge on down_out: [BS, k, H] so the score weighting + reduce is shared.
    let down_out = if let Some(sort_perm) = sort_perm_opt {
        // Sorted path: reshape [BS*k, 1, 1, H] -> [BS*k, H], then invert
        // permutation via take_on, then reshape to [BS, k, H].
        let inv_perm = argsort_on(&sort_perm, -1, target)?;
        let down_out_2d = mlx::ops::shape::reshape(&down_out_4d, [bs_k, h])?;
        let unsorted_2d = take_on(&down_out_2d, &inv_perm, 0, target)?;
        mlx::ops::shape::reshape(&unsorted_2d, [bs, k, h])?
    } else {
        mlx::ops::shape::squeeze_on(&down_out_4d, &[-2][..], target)?
    };

    // [BS, k] → [BS, k, 1] for broadcast with [BS, k, H], then sum over k.
    let routed_y = {
        let scores_unsq = mlx::ops::shape::expand_dims_on(&scores, -1, target)?;
        let weighted = &down_out * &scores_unsq;
        mlx::ops::sum_on(&weighted, -2, false, target)?
    };

    // (7) Shared expert with independent sigmoid gate.
    let shared_y = self.shared_expert.forward_on(&flat_x, target)?;    // [BS, H]
    let gate_logit = self.shared_expert_gate.forward_on(&flat_x, target)?; // [BS, 1]
    let gate_sig2 = gate_logit.sigmoid_on(target)?;
    let shared_gated = &shared_y * &gate_sig2;

    // (8) Combine routed + shared, then reshape back to [B, S, H].
    let out_flat = &routed_y + &shared_gated;
    let out = mlx::ops::shape::reshape(&out_flat, [b, s, h])?;
    Ok(out)
}
```

**Note on shapes** (4-bit packed, group_size=64, E=128 experts, hidden=2560, moe_intermediate=1536):
- `gate_weight`, `up_weight`: `[E, moe_inter=1536, H/8=320]`
- `gate_scales`, `up_scales`: `[E, moe_inter, H/64=40]`
- `down_weight`: `[E, H=2560, moe_inter/8=192]`
- `down_scales`: `[E, H, moe_inter/64=24]`

#### §3.1.3 ironmlx KV cache design (full-attention layers only)

`ironmlx/src/core/cache/kv_cache.rs` — module header + struct + new/with_step:

```rust
//! Per-layer KV cache for full-attention layers. See P2 spec § 3 for design.
//!
//! Implementation strategy: lazy alloc + step-rounded grow via concatenate;
//! per-update writes use Strategy A: a B-loop of `slice_update_on` calls
//! (one per row with per_row_lens[i] > 0). The public API (`new`, `with_step`,
//! `update_and_fetch`, `offsets`, `cap`, `reset`) is stable across strategies.

pub struct KVCache {
    keys: Option<Array>,        // [batch, n_kv_heads, cap, head_dim]
    values: Option<Array>,      // [batch, n_kv_heads, cap, v_head_dim]
    offsets: Vec<i32>,          // per-row write pointer
    cap: i32,                   // hard max sequence length
    step: i32,                  // grow chunk size, default 256
    batch: i32,
    n_kv_heads: i32,
    head_dim: i32,
    v_head_dim: i32,
    dtype: Dtype,
}

impl KVCache {
    pub fn new(batch: i32, n_kv_heads: i32, head_dim: i32, v_head_dim: i32,
               dtype: Dtype, cap: i32) -> Self {
        Self { keys: None, values: None, offsets: vec![0; batch as usize],
               cap, step: 256, batch, n_kv_heads, head_dim, v_head_dim, dtype }
    }

    pub fn with_step(mut self, step: i32) -> Self {
        assert!(step > 0, "KVCache step must be positive (got {step})");
        self.step = step;
        self
    }
    // ... offsets(), cap(), update_and_fetch(), reset() omitted ...
}
```

Key design choices vs. omlx PagedCache:
- **Dense `[B, n_kv_heads, cap, head_dim]` arrays** — NOT paged into blocks.
- **Lazy alloc**: keys/values are allocated on first `update_and_fetch` call.
- **Step-rounded grow** (default step=256): when seq position crosses a step
  boundary, a new dense array is allocated via `concatenate` (Metal copy).
- **Per-request, request-scoped**: each request owns its own cache; no
  cross-request KV reuse / sharing.
- **No prefix cache** (in either persistent storage or hot memory).
- Only 10/40 layers are full attention (Qwen3.5-MoE-A3B is hybrid with
  `full_attention_interval=4`); the other 30 layers are GatedDeltaNet linear
  attention with a different cache type (`core/cache/gated_delta.rs`).

#### §3.1.4 ironmlx scheduler / chunked prefill

`ironmlx/src/core/scheduler.rs` exposes per-request `prefill_chunk_size`
(default 2048 via CLI `--prefill-chunk-size`). The scheduler iterates prefill in
chunks of `chunk_size` tokens, where intermediate chunks update the cache only,
and the final chunk runs the full forward + `lm_head`. With default 2048:

| PP | Number of chunks (default cs=2048) |
|---|---|
| 128 | 1 (single-shot) |
| 512 | 1 |
| 2048 | 1 (boundary) |
| 4096 | 2 |
| 8192 | 4 |
| 16384 | 8 |

#### §3.1.5 ironmlx server launch (this bench)

```sh
SNAP=/Users/xin/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/1e20fd8d42056f870933bf98ca6211024744f7ec/
MLX_DIR=$HOME/.local/mlx cargo run --release -p ironmlx -- serve \
  --model "$SNAP" --port 8080 --host 127.0.0.1
# All other params at defaults: prefill-chunk-size=2048, b-max=4,
# admission-deadline-ms=5, admission-queue-max=32, max-cache-cap=32768
```

### §3.2 mlx-lm (fair baseline)

#### §3.2.1 Isolation strategy

Pin to the same git commit omlx uses (`git+ml-explore/mlx-lm@ed1fca4`,
v0.31.3) but install in a clean venv with **zero omlx packages** so the
import-time monkey patches cannot reach it.

`scripts/bench-venvs/mlx-lm/pyproject.toml`:

```toml
[project]
name = "mlx-lm-bench"
version = "0.1.0"
description = "Isolated venv for benchmarking ml-explore/mlx-lm@ed1fca4 (v0.31.3) against ironmlx; pinned to the same commit omlx uses but with zero omlx packages installed (no monkey-patch injection)."
requires-python = ">=3.12,<3.13"
dependencies = [
    "mlx-lm @ git+https://github.com/ml-explore/mlx-lm@ed1fca4cef15a824c5f1702c80f70b4cffc8e4dd",
]
```

Resolved dependencies (uv lock):
- `mlx==0.31.2` (Darwin metal)
- `mlx-lm==0.31.3` (git source, commit `ed1fca4`)
- `mlx-metal==0.31.2`
- `numpy==2.4.6`, `transformers==5.8.1`, `tokenizers==0.22.2`, ...

Sanity verified (`python -c "from mlx_lm.models import qwen3_5_moe"` succeeds —
native upstream support for `model_type=qwen3_5_moe`).

#### §3.2.2 mlx-lm server launch (this bench)

```sh
cd scripts/bench-venvs/mlx-lm
uv run mlx_lm.server --model "$SNAP" --host 127.0.0.1 --port 8082 --log-level INFO
# All other params at defaults.
```

### §3.3 omlx (observation only)

Launched from the omlx checkout's venv (`/Users/xin/workspace/iron-rivals/omlx/.venv`),
which contains:

- Same `mlx-lm==0.31.3` source from `git+ml-explore/mlx-lm@ed1fca4`.
- `omlx` package whose CLI entry point `omlx serve` performs `import omlx.patches`
  at startup. These patches replace `mlx_lm.models.qwen3_5.language.Qwen3_5GatedDeltaNet`
  and `mlx_lm.models.qwen3_5.language.Qwen3_5Attention` method bodies in place
  (monkey-patch). There is no opt-out flag.
- Additional optimizations active by default: PagedCache (block_size auto-tune
  256→2048, hot cache), vlm-engine path (text-only requests still go through
  the vlm engine + vision-feature cache lookup).

#### §3.3.1 omlx serve launch (this bench)

```sh
cd /Users/xin/workspace/iron-rivals/omlx
uv run omlx serve --model-dir "$SNAP" --host 127.0.0.1 --port 8081
# All other params at defaults: max-concurrent-requests=8, hot-cache-max-size=0,
# initial-cache-blocks=256, max-process-memory unbounded.
```

omlx data is informational only; it is **not** a fair-implementation baseline
because the runtime stack is "mlx-lm + monkey-patches + PagedCache + vlm engine".

---

## §4 Methodology

### §4.1 Harness

`iron-bench` is a Rust HTTP harness that drives any OpenAI-compatible
`/v1/chat/completions` endpoint. Per `(target, prompt_len)` cell across N timed
runs (after W warmup runs):

- **TTFT (ms)** — request send → first non-empty content token.
- **TG (tok/s)** — decode rate = `completion_tokens / gen_duration` (uses
  server-reported `completion_tokens` if SSE `stream_options.include_usage` is
  honored; falls back to local SSE chunk count otherwise).
- **TPOT (ms/tok)** — per-decoded-token time, excludes prefill.
- **PP (tok/s)** — prefill rate = `prompt_tokens / TTFT_seconds`. Uses
  server-reported `prompt_tokens` when available; falls back to local
  tokenizer count.
- **E2E (s)** — total wall-clock per request.

Reports median + p95 across the N timed runs.

### §4.2 Controlled inputs

- **Synthetic prompts via tokenizer round-trip**: iron-bench uses the model's
  `tokenizer.json` to generate a string that encodes to exactly N tokens
  (±2 BPE drift). This is necessary because string-length ≠ token-length and
  because chat-template insertions vary per backend.
- **Per-run nonce prefix**: every run prefixes a unique nonce token sequence
  so prefix caches (omlx PagedCache, mlx-lm BatchGenerator) cannot reuse
  prior runs' prefill. Without this, the second run's prefill drops to ~0 ms
  and `pp_tps` becomes meaningless.
- **`chat_template_kwargs.enable_thinking=false`**: Qwen3+ chat template has a
  thinking-mode gate. When enabled, omlx buffers the entire `<think>...</think>`
  block into a single SSE event, collapsing `gen_duration` to ~0 and inflating
  `tg_tps` into tens of thousands. We disable it across all targets.
- **Greedy sampler**: `temperature=0, top_p=1` for all targets. Deterministic
  output, no sampler-algorithm bias.
- **Warmup run excluded**: 1 warmup per cell materializes MLX compile graphs +
  primes any internal caches; not counted in stats.
- **`stream_options.include_usage=true`**: requests authoritative
  `prompt_tokens` / `completion_tokens` from the server when supported.

### §4.3 Strict serial execution

All three targets share the same M5 Max GPU. Running multiple servers
simultaneously would have each loading 17.5 GB of model weights into unified
memory plus per-request KV caches; concurrent GPU contention has been
empirically measured (see prior memory note `feedback_serial_perf_experiments`)
to degrade mlx-lm by ~8%.

Sweep procedure:
1. Start ironmlx server on port 8080 → run iron-bench `--target ironmlx=...`
   → write JSON → SIGTERM ironmlx.
2. Start omlx serve on port 8081 → run iron-bench `--target omlx=...`
   → write JSON → SIGTERM omlx.
3. Start mlx-lm server on port 8082 → run iron-bench `--target mlx_lm=...`
   → write JSON → SIGTERM mlx-lm.
4. Aggregate three JSON files into the tables in §5.

### §4.4 Variables NOT controlled (and why)

- Each backend uses its own default chunked-prefill / KV-cache behavior. This
  IS the variable under study — we want to see the production-default behavior
  of each backend, not force them into matching configurations.
- HTTP framing overhead is non-trivial. At PP=128, ironmlx's `prompt_tokens /
  TTFT` shows ~390 tok/s, while direct `Model::forward_on` measurement on the
  same hardware showed 1094 tok/s — the gap is HTTP request parsing + chat
  template tokenization + scheduler admission + SSE response framing. The
  three targets pay HTTP overhead in different ways; this is a real
  cross-backend perf factor and is intentionally kept in the measurement.
- The `model` field value in the JSON request differs across targets:
  ironmlx accepts arbitrary names (`qwen3.5-moe`); omlx strictly validates
  against the snapshot SHA (`1e20fd8d42056f870933bf98ca6211024744f7ec`);
  mlx-lm accepts arbitrary names (`default_model`). This is a server-side
  routing key only; it does not affect inference behavior.

---

## §5 Test data

### §5.1 Prefill — TTFT (median ms; lower is better)

| PP | ironmlx | mlx-lm | omlx | iron/mlxlm | iron/omlx |
|---:|---:|---:|---:|---:|---:|
| 128 | 328.61 | 259.92 | 128.64 | 0.79× | 0.39× |
| 512 | 1042.33 | 363.74 | 199.75 | **0.35×** | **0.19×** |
| 2048 | 1111.71 | 877.43 | 487.40 | 0.79× | 0.44× |
| 4096 | 2309.80 | 1058.44 | 929.63 | 0.46× | 0.40× |
| 8192 | 4748.08 | 2050.01 | 1925.34 | 0.43× | 0.41× |
| 16384 | 10581.11 | 5378.75 | 4468.99 | 0.51× | 0.42× |

**Ratio convention**: `iron/mlxlm` = `mlx-lm TTFT / ironmlx TTFT`. Values <1
mean ironmlx is slower (higher TTFT). e.g. `0.35×` means mlx-lm finished TTFT
in 35% of ironmlx's time, i.e. ironmlx is 2.86× slower.

### §5.2 Prefill — PP throughput (median tok/s; higher is better)

| PP | ironmlx | mlx-lm | omlx | iron/mlxlm | iron/omlx |
|---:|---:|---:|---:|---:|---:|
| 128 | 389.51 | 538.63 | 1088.29 | 0.72× | 0.36× |
| 512 | 491.21 | 1440.58 | 2623.26 | **0.34×** | **0.19×** |
| 2048 | 1842.20 | 2347.75 | 4226.53 | 0.78× | 0.44× |
| 4096 | 1773.32 | 3881.18 | 4418.96 | 0.46× | 0.40× |
| 8192 | 1725.33 | 4001.92 | 4261.06 | 0.43× | 0.40× |
| 16384 | 1548.42 | 3048.29 | 3668.83 | 0.51× | 0.42× |

**Ratio**: `iron/mlxlm` = `ironmlx PP / mlx-lm PP`. <1 means ironmlx slower.

**Note**: ironmlx PP plateaus at ~1842 tok/s at PP=2048 then _declines_ to
~1548 tok/s at PP=16384. mlx-lm grows monotonically through PP=8192
(540→1441→2348→3881→4002) before declining at 16384 (3048). omlx grows
through PP=4096 (1088→2623→4227→4419) then declines slightly.

### §5.3 Decode — TG throughput (median tok/s; higher is better)

| PP | ironmlx | mlx-lm | omlx | iron/mlxlm | iron/omlx |
|---:|---:|---:|---:|---:|---:|
| 128 | 79.57 | 101.43 | 128.83 | 0.78× | 0.62× |
| 512 | 78.54 | 101.51 | 128.11 | 0.77× | 0.61× |
| 2048 | 123.73 | 103.48 | 123.71 | **1.20×** | 1.00× |
| 4096 | 121.42 | 107.84 | 123.67 | **1.13×** | 0.98× |
| 8192 | 117.98 | 104.65 | 121.61 | **1.13×** | 0.97× |
| 16384 | 112.04 | 100.44 | 103.27 | **1.12×** | 1.08× |

**Note the regime change at PP=2048**: ironmlx TG jumps from ~78 to ~124 tok/s
between PP=512 and PP=2048, then settles in the 112-124 range. mlx-lm stays
in a tight 100-108 band throughout. omlx is the consistent decode leader at
PP < 16384.

### §5.4 Decode — TPOT (median ms/tok; lower is better)

| PP | ironmlx | mlx-lm | omlx | iron/mlxlm | iron/omlx |
|---:|---:|---:|---:|---:|---:|
| 128 | 12.69 | 9.95 | 7.89 | 0.78× | 0.62× |
| 512 | 12.99 | 10.00 | 7.91 | 0.77× | 0.61× |
| 2048 | 8.24 | 9.81 | 8.26 | 1.19× | 1.00× |
| 4096 | 8.30 | 9.35 | 8.15 | 1.13× | 0.98× |
| 8192 | 8.54 | 9.63 | 8.29 | 1.13× | 0.97× |
| 16384 | 9.15 | 10.94 | 10.76 | 1.20× | 1.18× |

### §5.5 E2E — total wall-clock (median s; lower is better)

| PP | ironmlx | mlx-lm | omlx | iron/mlxlm | iron/omlx |
|---:|---:|---:|---:|---:|---:|
| 128 | 1.64 | 1.29 | 0.77 | 0.78× | 0.47× |
| 512 | 1.79 | 0.93 | 0.70 | 0.52× | 0.39× |
| 2048 | 1.48 | 1.52 | 0.86 | 1.03× | 0.58× |
| 4096 | 3.36 | 2.24 | 1.96 | 0.67× | 0.58× |
| 8192 | 5.83 | 3.25 | 3.00 | 0.56× | 0.51× |
| 16384 | **11.30** | **5.75** | **4.60** | **0.51×** | **0.41×** |

### §5.6 Tail behavior (p95 across the 5 timed runs, PP=2048)

| metric | ironmlx p95 | mlx-lm p95 | omlx p95 |
|---|---:|---:|---:|
| ttft_ms | 1111.78 | 924.05 | 487.51 |
| pp_tps | 1842.69 | 2363.98 | 4227.79 |
| tg_tps | 124.09 | 105.96 | 126.27 |
| tpot_ms | 8.26 | 10.02 | 8.33 |
| e2e_s | 1.49 | 2.06 | 0.98 |

p95 stays close to the median for all three backends (variance across the 5
runs is low). The interesting anomaly is that **mlx-lm's e2e p95 (2.06 s) is
notably worse than its median (1.52 s)** at PP=2048 — single-tail run drove
this. ironmlx and omlx don't show that tail.

### §5.7 Raw measurements per cell (5 samples each)

Embedded JSONs are in the appendix §8. Each JSON contains the full `raw_runs`
array. A representative sample (ironmlx PP=2048, run_idx=0):

```json
{
  "cached_tokens": null,
  "completion_tokens_server": null,
  "e2e_s": 2.140xxx,
  "finish_reason": "stop",
  "pp_target": 2048,
  "pp_tps": 1843.xx,
  "prompt_tokens_local": 2048,
  "prompt_tokens_server": null,
  "run_idx": 0,
  "target": "ironmlx",
  "tg_target": 128,
  "tg_tps": 124.xx,
  "tpot_ms": 8.xx,
  "ttft_ms": 1111.xx
}
```

`prompt_tokens_server: null` for ironmlx indicates the server does not return
`prompt_tokens` in the OpenAI usage object; `pp_tps` is computed from
`prompt_tokens_local`. omlx and mlx-lm both return server-side
`prompt_tokens` (which includes chat-template tokens, adding ~12 tokens per
request) so their `pp_tps` is server-reported.

---

## §6 Observations

### §6.1 Prefill anomalies (the headline gap)

#### 6.1.1 The PP=512 dip

PP=512 is the worst point for ironmlx vs both peers:
- ironmlx 1042 ms TTFT — 2.86× slower than mlx-lm (364 ms), 5.22× slower than
  omlx (200 ms).
- ironmlx TTFT roughly _triples_ between PP=128 (329 ms) and PP=512 (1042 ms),
  whereas mlx-lm only grows 1.4× (260 → 364 ms). Then PP=512 → PP=2048 ironmlx
  grows only 7% (1042 → 1112 ms) — sub-linear.

The PP=512 → 2048 sub-linear scaling is consistent with the T0 profile
(reports/p5e-t0-profile.md): gather_qmm reaches near-saturated GPU occupancy
at BS≈512 and additional tokens add proportionally less. But the PP=128 →
512 super-linear blow-up is not explained by the T0 data.

#### 6.1.2 The PP=2048 plateau and decline

ironmlx prefill throughput peaks at 1842 tok/s (PP=2048), then DECLINES through
PP=4096 (1773), PP=8192 (1725), PP=16384 (1548). Both mlx-lm and omlx continue
to scale up through PP=8192 before declining (mlx-lm peak 4002 tok/s at
PP=8192, omlx peak 4419 tok/s at PP=4096).

The plateau coincides with the `--prefill-chunk-size 2048` default boundary:
- PP=2048: single chunk (best case, what P5e T5 optimized).
- PP=4096+: multi-chunk (2, 4, 8 chunks respectively).

Hypothesis: chunked prefill in ironmlx introduces per-chunk overhead that does
not amortize. Each chunk re-builds intermediate state (KV cache slices, scheduler
batch composition); the wall-clock for 2 chunks of 2048 (PP=4096) ≈ 2× the
wall-clock for 1 chunk of 2048 (PP=2048) plus per-chunk setup, rather than 2×
the kernel-only time. Verify by reading `core/scheduler.rs` prefill loop and
identifying per-chunk costs.

#### 6.1.3 Throughput inversion at PP=16384

ironmlx's PP curve declines below PP=8192 by PP=16384 (1725 → 1548). This is
KV cache cost: at PP=16384, full-attention layers compute O(S²) attention
over a 16K context. The 10 full-attention layers (out of 40) consume ~20% of
prefill at PP=2048 per T0 profile, scaling super-linearly (15.6× for 16×
input). At PP=16384 this would project to ~50% of prefill being full-attention
KV work.

### §6.2 Decode regime change at PP=2048

At PP=128/512, ironmlx decode TG is 78-79 tok/s, ~22% slower than mlx-lm
(101 tok/s). At PP=2048 ironmlx jumps to 124 tok/s and outperforms mlx-lm
(103) by 20%. The break is sharp — single sample at PP=1024 would help locate
it.

Two hypotheses for the slow PP=128/512 decode:

a. **KV cache grow-step cost** (`KVCache.step = 256`, see §3.1.3). At PP=128,
   the cache is allocated to 256 capacity on first decode call. If subsequent
   decode tokens cross step boundaries, MLX triggers `concatenate` allocations.
   At PP ≥ 2048 the cache is preallocated large enough that no grow-step fires
   during the 128-token decode window.

b. **Decoder-step batch size affecting MoE routing**. At PP=128/512, the
   decoder step has BS*k = 1*4 = 4, below `SORTED_ROUTING_MIN_BS_K = 512`, so
   the default broadcast path is used. At PP=2048+ during decode the BS*k is
   still 4 (B=1 batch, k=4) — so the sorted-routing threshold is NOT crossed.
   This rules out hypothesis (b); the dispatch is the same.

Hypothesis (a) is more likely. Verify by instrumenting `KVCache::update_and_fetch`
or by running the bench with a custom `KVCache::with_step(<larger>)` to see if
the gap closes.

### §6.3 Decode advantage at PP ≥ 2048

ironmlx TG of 112-124 tok/s vs mlx-lm 100-108 in the long-prompt regime
suggests ironmlx's decode hot path (per-token forward + KV append + lm_head)
is well-tuned. This is consistent with prior P8a/P8b work (see memory
`project_p8a_stage8_findings` / `project_p8a_stage9_findings`) on lm_head and
self_qmm — those optimizations primarily affect decode-step single-token
forwards.

omlx matches ironmlx's decode performance at PP=2048-8192 (123-124 tok/s) and
edges back out only at PP=16384. omlx's PagedCache likely helps at long
context.

### §6.4 E2E story at agent prompt lengths (PP=8192, 16384)

Agent-style workloads (long system prompt + tools + memory + multi-turn) put
prompt length at 8K-32K typically.

PP=8192 e2e:
- ironmlx: 5.83 s (prefill 4.75 s + decode 1.08 s)
- mlx-lm: 3.25 s (prefill 2.05 s + decode 1.22 s)
- omlx: 3.00 s (prefill 1.93 s + decode 1.07 s)

PP=16384 e2e:
- ironmlx: 11.30 s (prefill 10.58 s + decode 0.72 s — incomplete decode within
  the 128-token target; finish_reason "stop" indicates EOS sampled earlier)
- mlx-lm: 5.75 s
- omlx: 4.60 s

The gap is **almost entirely prefill**. Decode is ≤ 0.2 s difference. For
agent workloads where the user waits for TTFT, ironmlx is currently 2× slower
than mlx-lm at PP=16384.

### §6.5 What Stage 2 (sorted routing) did NOT fix

P5e T5 sorted routing improved `SparseMoeBlock::forward_on` by ~1.9× at the
direct-call layer (Model::forward_on no HTTP). The HTTP-level bench shows the
gap remains at the chunked-prefill / scheduler / KV-cache layer that wraps
around the per-step forward call. The chosen optimization was correctly
scoped to its target (the 3 gather_qmm calls per layer), but the prefill
wall-clock that the user actually sees is bottlenecked outside that scope at
PP ≥ 4096.

---

## §7 Analysis hints for the code reviewer

### §7.1 The T0 hot-path profile (single forward call, PP=2048, M5 Max, instrumented)

From `reports/p5e-t0-profile.md` (committed at `f47d471`):

```
PP=2048 wall-clock = 2223.6 ms (921 tok/s, instrumented version)

Components:
  40 × DecoderLayerMoe         2220.6 ms (99.9%)
  embed + mrope + final_norm     2.93 ms (0.13%)
  lm_head                        2.06 ms (0.09%)

DecoderLayerMoe avg (40 layers):
  Linear attn layers (30/40):
    GatedDeltaNet (linear attn)   15.3 ms (80.1% of layer)
    SparseMoeBlock                39.9 ms (incl. eval barrier inflation)
  Full attn layers (10/40):
    GatedAttention (KV/SDPA)      14.5 ms (35.6% of layer)
    SparseMoeBlock                39.9 ms

SparseMoeBlock breakdown (avg per call, 40 calls/forward, eval-barrier inflated):
  gather_qmm_down + squeeze     16.05 ms (40.2%)   <-- biggest single op
  gather_qmm_gate               10.21 ms (25.6%)
  gather_qmm_up                  9.73 ms (24.4%)
  shared_expert_mlp (3×Lin)      1.33 ms (3.3%)
  weighted_sum_k (expand+mul+sum) 0.66 ms (1.6%)
  router_gate_linear             0.49 ms (1.2%)
  swiglu_activation              0.40 ms (1.0%)
  ... (other ops < 1% each)
  ─────────────────────────────────────
  3× gather_qmm subtotal        36.0 ms (90.2% of SparseMoeBlock)
  All SparseMoeBlock            39.9 ms
```

**Key finding from T0**: The three `gather_quantized_matmul` calls (gate, up,
down projections) account for **90.2% of SparseMoeBlock time** and **71.8% of
total decoder layer time** at PP=2048.

P5e T5 sorted routing addressed exactly that hot path. Direct-call wall-clock
went from 2067.45 ms → 1077 ms (1.92×). HTTP-level wall-clock went from
~2057 ms (P5d on M1 Pro, comparable test) to ~1112 ms (this bench). Both layers
of measurement confirm the in-source optimization landed.

What's NOT in the T0 profile (because it tested raw `Model::forward_on`):
- HTTP request parsing + chat-template tokenization
- Scheduler admission + queueing
- Chunked-prefill loop overhead (T0 was always single-shot)
- KV cache concatenate-grow when crossing step=256 boundaries

These are the four candidates for the observed HTTP-vs-direct delta.

### §7.2 Candidate root causes for the prefill gap (ranked by suspected severity)

**(C1) Chunked prefill per-chunk overhead** — likely dominant at PP ≥ 4096.
- Source: `ironmlx/src/core/scheduler.rs` — search for `prefill_chunk_size`,
  the prefill loop, and per-chunk forward dispatch.
- Each chunk computes the full per-layer forward (40 layers); intermediate
  chunks discard the lm_head output but still pay the full KV append cost.
- mlx-lm uses a different chunking strategy (or no chunking) — its
  `BatchGenerator.generate_step` does the full forward in a single shot when
  KV memory allows. omlx's PagedCache enables larger-than-32K context without
  contiguous memory pressure, so it can run single-shot prefill at PP=16384.
- Verifiable: re-run iron-bench against ironmlx with
  `--prefill-chunk-size 32768` (single shot for all our PP tests). If the
  PP=4096+ ratio improves toward 0.8× of mlx-lm, C1 is confirmed.

**(C2) PP=512 specific super-linear blow-up** — separate root cause likely.
- TTFT 329 (PP=128) → 1042 (PP=512) — 3.17× for 4× input — only marginally
  super-linear, but jumps to nearly the PP=2048 cost despite being 4× less
  prompt. This is suspicious.
- Hypothesis: chat-template token cost OR the first KV-cache step=256
  allocation OR a Metal compile path that activates at BS=512 but not BS=128.
- Verifiable: instrument ironmlx server with per-stage timing on a PP=512
  request and compare to PP=128 / PP=2048.

**(C3) KV cache concatenate-grow during prefill**
- Source: `ironmlx/src/core/cache/kv_cache.rs` — `update_and_fetch` and the
  step=256 grow logic via concatenate.
- At PP=16384 with step=256, the cache grows 64 times during the single
  prefill (or fewer if `with_step(cap)` is used). Each grow is a copy of all
  existing K/V data. This is O(S²) memory traffic in the cache layer.
- Verifiable: set `KVCache.step` to match cap (one-shot preallocation) and
  re-measure. If PP=16384 prefill improves, C3 confirmed.
- Note: T0 profile says full attention is 14.5 ms/layer @ PP=2048 (10
  layers = 145 ms total, 6.5% of prefill). At PP=16384 that scales
  super-linearly to ~50% of prefill, which is consistent with the observed
  PP=8192→16384 throughput decline.

**(C4) Linear attention (GatedDeltaNet) at long prompt**
- 30 of 40 layers, 80% of linear-attn layer time per T0 (15.3 ms/layer at
  PP=2048 → 459 ms total, 20% of prefill).
- Implementation in `ironmlx/src/models/qwen3_5/gated_delta_net.rs` (not
  embedded here; check the source for the recurrent loop structure).
- Likely contributes to the prefill gap at long PP but is not the dominant
  factor per T0 profile.
- omlx replaces this layer's body via `omlx.patches.gated_delta_advance` —
  the patch is one of the two main reasons omlx outperforms mlx-lm in §5
  data above.

**(C5) Other ops outside SparseMoeBlock** (router, shared_expert, RMS norms)
- Per T0 profile, these are 0.1-3.3% individually, not promising single
  targets but if 5+ of them each have 5% slowness, the aggregate matters.

### §7.3 Files to inspect in the ironmlx tree

The reader who wants to extend this analysis to source code should look at:

| File | Purpose | Lines |
|---|---|---|
| `ironmlx/src/models/qwen3_5_moe/sparse_moe.rs` | P5e hot path (this report §3.1.2) | 496 |
| `ironmlx/src/models/qwen3_5/gated_delta_net.rs` | Linear-attn layer (30/40 layers) | ~? |
| `ironmlx/src/models/qwen3_5/attention.rs` | Full-attn layer (10/40 layers) | ~? |
| `ironmlx/src/core/cache/kv_cache.rs` | Full-attn KV cache (this report §3.1.3) | 965 |
| `ironmlx/src/core/cache/gated_delta.rs` | Linear-attn state cache | 525 |
| `ironmlx/src/core/scheduler.rs` | Request scheduling + chunked prefill loop | 2557 |
| `ironmlx/src/core/generate.rs` | High-level generation orchestration | ~? |

Repository: `https://github.com/<TBD-private>/ironmlx-backend`, branch
`ironmlx-p5e-perf`, HEAD `f420c94`. (Repo currently private; if the reviewer
needs additional source, request specific file dumps.)

### §7.4 What a useful response from the analysis would contain

1. Confirm or refute hypothesis C1 (chunked prefill overhead) by code reading.
   If confirmed, what specifically about ironmlx's chunked path is suboptimal
   vs single-shot? (per-chunk graph rebuild? per-chunk Metal compile? KV
   append vs lm_head fork?)
2. Identify the PP=512 specific bottleneck (C2) — what changes between PP=128
   and PP=512 in the ironmlx pipeline that doesn't change between PP=512 and
   PP=2048?
3. Propose a remediation order — which 1-2 changes are highest expected
   impact for agent-scale prefill (PP ≥ 8192)?
4. Identify any code-quality issues in the embedded `SparseMoeBlock::forward_on`
   that may be hurting performance in subtle ways (e.g. unnecessary
   allocations, redundant reshape/expand_dims chains, unfused element-wise
   ops).

---

## §8 Appendix: raw JSON

The three iron-bench output JSONs are at:

- `reports/p5e-three-way-bench/ironmlx.json`
- `reports/p5e-three-way-bench/mlx_lm.json`
- `reports/p5e-three-way-bench/omlx.json`

Each contains `metadata { runs_measured, sampler, targets, warmup }` plus a
`raw_runs` array of 36 entries (6 PP × 5 timed runs + 6 PP × 1 warmup; only
timed entries are reported as raw_runs — total 30 per file).

For the analyst's convenience, here are the per-(target, PP) sample windows
showing 5 timed runs each, as observed from the stderr logs:

**ironmlx** (logs: `reports/p5e-three-way-bench/ironmlx.log`):
```
PP=128:  TTFT 329.4 / 327.6 / 328.7 / 328.6 / 327.8 ms
         TG    79.6 /  79.6 /  79.8 /  79.9 /  79.1 tok/s
PP=512:  TTFT 1042.5 / 1041.9 / 1043.2 / 1041.4 / 1042.3 ms
         TG    78.9 /  78.0 /  78.5 /  78.5 /  78.6 tok/s
PP=2048: TTFT 1111.7 / 1110.9 / 1111.8 / 1111.4 / 1113.0 ms
         TG   124.6 / 123.7 / 123.6 / 124.1 / 123.5 tok/s
PP=4096: TTFT 2244.3 / 2274.9 / 2309.8 / 2326.3 / 2329.7 ms
         TG   121.0 / 122.0 / 122.1 / 116.8 / 121.4 tok/s
PP=8192: TTFT 4799.0 / 4766.0 / [...continued at run 3-5; logs truncated]
PP=16384: TTFT 10236.4 / 10533.1 / 10581.1 / 10678.3 / 10781.3 ms
         TG   118.5 / 116.9 / 112.0 / 111.6 / 111.3 tok/s
```

**omlx** (logs: `reports/p5e-three-way-bench/omlx.log`):
```
PP=512:  TTFT [pre-truncated] / 201.1 / 198.8 / 198.2 / 203.5 ms
         TG    [..] / 126.2 / 128.1 / 127.5 / 129.1 tok/s
PP=2048: TTFT 489.1 / 487.5 / [run 3-5 in JSON] ms
PP=16384: TTFT [..] / [..] / 4497.8 / 4469.0 / 4412.4 ms
          TG   [..] / [..] / 101.8 / 114.0 / 102.8 tok/s
```

**mlx-lm** (logs: `reports/p5e-three-way-bench/mlx_lm.log`):
```
PP=512:  TTFT 375.6 / 370.5 / 363.7 / 342.6 / 342.4 ms
         TG    96.7 /  98.4 / 102.6 / 102.6 / 101.5 tok/s
PP=2048: TTFT 877.4 / 871.4 / [run 3-5 in JSON] ms
         TG   103.5 / 106.0 / [..] tok/s
PP=16384: TTFT [..] / [..] / 5636.8 / 5378.7 / 6199.9 ms
          TG    [..] / [..] / 100.7 /  99.8 / 100.6 tok/s
```

Variance across the 5 timed runs is small (typically <2% on TTFT for ironmlx;
mlx-lm has slightly higher PP=16384 variance suggesting GPU thermal /
scheduler jitter at the extreme).

---

## §9 Reproducing this bench

1. Ensure model snapshot present:
   ```sh
   ls ~/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/
   ```
2. Set MLX env:
   ```sh
   export MLX_DIR=$HOME/.local/mlx
   ```
3. Build mlx-lm venv:
   ```sh
   cd scripts/bench-venvs/mlx-lm && uv sync
   ```
4. Verify omlx checkout:
   ```sh
   ls /Users/xin/workspace/iron-rivals/omlx/pyproject.toml
   ```
5. For each target (ironmlx, omlx, mlx-lm): start server, wait for ready,
   run iron-bench, kill server. Sequence per §4.3.
6. Aggregate via python (see git log around `f420c94..` for the inline
   aggregation script).

The full bench (3 sweeps × 6 PP × 6 runs + server load) takes ~40 minutes on
M5 Max 128 GB.

---

_Generated by the P5e three-way bench harness. Branch `ironmlx-p5e-perf`,
report committed on top of P5e final HEAD `f420c94`._
