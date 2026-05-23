# P5i.a T3 — Self-quant gather Metal kernel 立项可行性 (Level a; design only)

**Status:** Design memo, no kernel implementation. Per P5i.a spec § 4.4 + § 5,
any Level (a) Metal kernel rewrite requires explicit Boss Scope-gate approval
before P5i.b implementation work begins.

**Date:** 2026-05-23
**Branch:** `ironmlx-p5i-a-gather-qmm-feasibility` @ `5f9a269` (T2 fused gate+up landed)
**Source files referenced:**
- `ironmlx/src/nn/self_qmm/mod.rs` (175 LoC)
- `ironmlx/src/nn/self_qmm/kernel.rs` (129 LoC)
- `ironmlx/src/nn/self_qmm/lookup.rs` (111 LoC)
- `ironmlx/src/nn/self_qmm/metal/qmm_t.metal.in` (297 LoC)
- `ironmlx/src/models/qwen3_5_moe/sparse_moe.rs` (~1000 LoC; gather_qmm sites at L867-893, L901-924, L624-638)
- `/Users/xin/workspace/iron-rivals/mlx/mlx/backend/metal/quantized.cpp` (1751 LoC; `gather_qmm_rhs` L1215-1363)

---

## § 1 Summary

**Verdict: 暂缓 (defer P5i.b; first run T4 + execute P5i.c new-candidate discovery).**

A self-quant gather Metal kernel is **technically feasible** — the existing
self_qmm dense kernel (`metal/qmm_t.metal.in`) provides a directly relevant
precedent (+35% over MLX baseline at PP=2048 dense Linear per
`[project_p8a_stage9_findings]`), and extending its tile structure to a
gather variant is a well-scoped problem.

However, four factors push the recommendation to **暂缓** rather than
**立项**:

1. **Gather indirection cost is structurally unknown.** self_qmm's +35% gain
   came on a contiguous (M, K) × (N, K) matmul where every loader thread
   reads sequential memory. Gather mode adds an `expert_indices[token]`
   lookup that fragments the weight-load pattern across L2 cache lines and
   may halve or quarter the achievable speedup. Lower-bound ROI may not
   justify the 2-4 week impl cost.
2. **PP=128 may close without P5i.b.** Post-T2 gap is -6.13%; T4
   (gda_step_1a_in_proj_qkvz hot-path tweak per P5h+1) is a low-risk
   no-Scope-gate option that may push PP=128 past the +10% target.
3. **PP=512 -35.62% gap may not be purely kernel-bound.** T0b H4 was a
   small-PP hypothesis (kernel dispatch overhead at PP≤128). At PP=512 the
   gap could equally come from launch-count overhead (8 gather_qmm calls per
   layer × 28 layers × 2 chunks), KV-cache layout transitions, or the
   gather indirection itself in MLX upstream. Without an instrumented
   attribution pass (P5i.c), committing to P5i.b is premature.
4. **Existing self_qmm is M1-Pro-only tile-tuned.** Per
   `[project_cross_device_tuning_deferred]` M5 Max tile params are not
   tuned; a gather kernel rewrite would need both expert-aware AND
   device-aware tile lookup before delivering peak gain.

The memo below documents the design fully so that **if Boss authorizes
P5i.b**, the implementer has a complete starting plan. The Scope gate
remains closed until then.

---

## § 2 self_qmm precedent

### 2.1 What the dense self-quant kernel does

`ironmlx/src/nn/self_qmm/` implements a Metal kernel for the 4-bit
affine-quantized matmul `out[M,N] = x[M,K] @ dequant(w)[N,K]^T` where
`w` is packed uint32 (8 nibbles per word), `scales[N,K/64]` and
`biases[N,K/64]` are per-group affine parameters.

**Template structure** (`metal/qmm_t.metal.in:1-297`):
- Tile fixed at `(BM=32 batch rows, BN=64 weight rows, BK=32 inner K)` —
  ported from llama.cpp's `kernel_mul_mm_q4_K_f32` (NR1=32, NR0=64, NK=32).
- 128 threads per TG = 4 SGs × 32 lanes.
- Shared-mem layout: `sa[2048]` weights (BN×BK), `sb[1024]` activations
  (BM×BK), both organized as 8×8 simdgroup-frag blocks for direct
  `simdgroup_load` with `elements_per_row=8` (no stride redirection).
- Outer-K loop loads one (BM×BK) activation tile + one (BN×BK) dequant'd
  weight tile per iteration; inner-K loop runs 4 simdgroup MMA steps
  (BK/FRAG=4).
- Per-SG accumulator `simdgroup_float8x8 mc[8]` = 2 batch-frag × 4 wcol-frag
  flat layout. SG sg_y ∈ {0,1} covers 16 batch rows, sg_x ∈ {0,1} covers
  32 weight cols → 2×2 SG grid covers full BM×BN tile.
- Output store: `mc[i].thread_elements()` reinterpret as `vec<float, 2>`,
  bound-checked + cast to runtime out dtype.

**Tile lookup arity** (`lookup.rs:48`):
```
fn lookup_tile(device_arch: &str, _m, _n, _k, _bits, _group_size) -> Tile
```
Today: M1 Pro (`applegpu_g13s/g13d`) returns `(32, 64, 32)`; everything else
returns the same default with a once-per-process warning. The signature is
shape-aware-ready (m/n/k passed in) for stage 10+ branch expansion.

**Dispatch contract** (`kernel.rs:64`):
```
dispatch_qmm_t(x: &Array, w: &Array, scales: &Array, biases: &Array,
               bm: i32, bn: i32, bk: i32) -> Result<Array>
```
- `x.shape() = [..., K]` (leading dims flattened to M via product)
- `w.shape() = [N, K/8]` packed uint32
- `scales.shape() = [N, K/64]`, `biases.shape() = [N, K/64]`
- Grid: `(n_tiles_x * 128, n_tiles_y, 1)` (MLX dispatch_threads semantics —
  grid is total thread count not TG count); threadgroup `(128, 1, 1)`.
- Template ints: `M, N, K, BM, BN, BK` passed to
  `mx::fast::metal_kernel::template_int(...)` so MLX upstream auto-
  specializes per (shape, tile) tuple at PSO cache time.

### 2.2 Why +35% over MLX baseline

Per `[project_p8a_stage9_findings]`:
- `lm_head fix` contributed +15-17% (separate issue, not kernel-internal).
- Self-quant kernel itself delivered +35% on PP=2048 dense Linear at
  cool state on M1 Pro.

Root causes per the memo:
- llama.cpp Q4_K_M kernel structure has been hand-optimized for Apple
  Silicon SG-MMA + threadgroup memory bandwidth balance; reusing the same
  (NR1, NR0, NK) tile choice + shmem block layout captures most of the
  win without re-deriving from first principles.
- Dequant staging through thread registers (4×4 tile, `temp_w[16]`) lets
  the Metal compiler pipeline mul/add against threadgroup stores —
  measurable vs naive direct-to-shmem dequant.
- Vec8 activation load (1 op/thread/iter) saturates L1 read BW.

**MLX baseline being beaten** is `mlx::quantization::quantized_matmul_on`
calling MLX upstream's `qmm` (tile `(32, 64, 32)` for transpose-true, see
`mlx/backend/metal/quantized.cpp:1420` `qmm(...)` path). Notably, MLX
upstream's tile choice for `qmm` is the SAME as self_qmm's (32, 64, 32) —
the +35% win comes from kernel internals (shmem layout, dequant staging,
load patterns), NOT from tile selection.

### 2.3 Reusable vs adaptation needed for gather

| Component | Dense self_qmm | Gather variant | Reuse / adapt |
|---|---|---|---|
| Tile (BM, BN, BK) | (32, 64, 32) | TBD per indirection cost | adapt (likely smaller BM) |
| TG threads | 128 (4 SGs) | likely 128 | reuse |
| Shmem layout | sa[2048] sb[1024] | sa[2048] sb smaller | mostly reuse |
| Dequant staging | thread-reg 4×4 tile | reuse exactly | reuse |
| Activation load | vec8 contig from `x[M,K]` | vec8 contig but per-token expert lookup | adapt addressing only |
| Weight load | sequential `w[N,K/8]` | `w[E, N, K/8]` + expert_indices[token] | adapt addressing; same vec8 pattern |
| SG-MMA pattern | 4 SGs × mc[8] | reuse exactly | reuse |
| Output store | bound-check + cast | scatter into output[token, N] | mostly reuse |
| Tile lookup | (device, shape) | (device, shape, expert_count, k_per_expert) | extend arity |

**Key insight:** the inner MMA + dequant + shmem layout transfers nearly
verbatim. The deltas are confined to (a) how weights are addressed per TG
and (b) how the output is scattered. This bounds the implementation
complexity but **does not bound the performance impact** of the
indirection — that requires a prototype to measure.

---

## § 3 Gather kernel design pseudocode

### 3.1 Dispatch contract

Proposed signature (mirrors `dispatch_qmm_t` arity):
```
dispatch_gather_qmm_t(
    x: &Array,            // [B*k, 1, K] or [B, k, 1, K] flattened to [M, K]
                          //   (M = total token-expert pairs after gather)
    w: &Array,            // [E, N, K/8] packed uint32 (E = num_experts)
    scales: &Array,       // [E, N, K/64]
    biases: &Array,       // [E, N, K/64]
    indices: &Array,      // [M] uint32 — expert index per token-expert row
    bm: i32, bn: i32, bk: i32,
) -> Result<Array>        // [M, N]
```

Per spec § 5 only `sorted_indices=true` path is in scope for the first
prototype (i.e. `rhs_indices` already sorted by expert so consecutive M
rows share the same expert), matching the post-T1 C1 sorted layout at
`sparse_moe.rs:874-888`.

### 3.2 Tile + thread group layout

Two design options for handling expert indirection:

**Option A — Per-token expert lookup, expert-uniform within tile (RECOMMENDED for prototype):**
- BM (batch rows per TG) reduced from 32 to **8 or 16** so that a single TG's
  M-axis tile is highly likely to land entirely within one expert's run of
  sorted tokens (avoid mid-tile expert boundary crossing).
- TG checks `expert_id = indices[tile_m]` at TG entry; if all M rows in
  this tile have the same expert (the hot path under sorted_indices=true),
  the weight base pointer is computed once: `w_base = w + expert_id * (N * K/8)`.
- Cold path (tile straddles expert boundary): per-row weight base computed
  in the activation loader, single TG branches at the top.
- BN, BK unchanged: 64, 32.

```
// Pseudocode entry
const int row0 = tile_m;
const int row_max = min(tile_m + BM, M);
const uint exp0 = indices[row0];
const uint expN = indices[row_max - 1];
const bool uniform_expert = (exp0 == expN);

device const uint32_t* w_base   = w      + exp0 * (N * K/8);
device const T*        s_base   = scales + exp0 * (N * K/64);
device const T*        b_base   = biases + exp0 * (N * K/64);

if (uniform_expert) {
    // Hot path — same as dense qmm_t with rebased w/scales/biases pointers.
    // Outer-K loop reads w_base[n,k0..k0+BK/8], s_base[n,g], b_base[n,g].
    // Activation loader reads x[row0..row_max, k0..k0+BK].
    // Inner-K SG-MMA + output store unchanged.
} else {
    // Cold path — fall back to per-row expert lookup in dequant phase.
    // Each weight loader thread re-reads indices[row] and rebases.
    // ~10-20% slowdown but rare (only ~num_experts boundary tiles per BM rows).
}
```

**Option B — Per-thread expert lookup (REJECTED for prototype):**
- Every weight loader thread does `expert_id = indices[token_in_tile]` on
  every outer-K iteration. Simpler to reason about (no branch), but adds 1
  uint load per weight load and likely defeats compiler's address-coalescing
  inference. Likely 20-30% slower than Option A hot path.

**Recommended prototype tile:** `(BM=8, BN=64, BK=32)` on M1 Pro. BM=8 because:
- Qwen3.5 MoE Top-k typically routes 4-8 tokens per expert per chunk at
  PP=512 (per `[project_p5h_t3_findings]` `packed_tokens=4-32/expert`).
- BM=8 means most TG tiles fit entirely within one expert's run → uniform_expert
  hot path dominates.
- BM=8 with 128 threads/TG means 16 threads per batch row in the activation
  loader (vs 4 in dense BM=32) — adapt NL1 from 4 to 16, reducing per-thread
  K-positions from 8 to 2. Acceptable; vec8 load still works at the wider
  thread spread.
- Alternative `BM=16` if microbench shows BM=8 underutilizes SGs.

### 3.3 Output scatter

Output layout matches sparse_moe.rs:867-893 expectation: `out[M, N]` row
`token_i` written by thread fragments mapped to `out + token_i * N + col`.
Identical to dense self_qmm output store at `qmm_t.metal.in:278-297` —
**no scatter contention**, because each output row is written by exactly
one TG (M is partitioned across `n_tiles_y` TGs, no overlap).

The "scatter" terminology in the task description is misleading: under
sorted_indices=true with row-major `out[M, N]`, output writes are dense
and contiguous. Scatter contention would only arise if we wrote back to
the original token positions (unsorted), which is the caller's
responsibility (`sort_perm` inverse at `sparse_moe.rs:895`).

### 3.4 Indirection cost — the uncertainty

The wall-time question is: **how much does Option A's hot path lose vs the
identity case (E=1, indices=range(M))?**

Expected sources of slowdown:
1. **L2 cache footprint multiplier.** Dense self_qmm L2 working set =
   `N * K` weights ≈ Qwen3.5 routed expert size (`N ≈ moe_intermediate_size=512`,
   `K ≈ hidden_size=2048` for Qwen3.5-35B-A3B). Gather variant has E=256
   experts, and consecutive tiles likely touch 4-8 different expert slabs
   in PP=512. Per-expert weight slab = 512×2048 bytes × 0.5 (4-bit packed) =
   524 KB — single expert fits in L2 but full expert set (256 × 524 KB =
   131 MB) far exceeds L2. L2 thrash is real if expert routing scatters
   across many experts per layer per chunk.
2. **Branch overhead.** Even with the uniform_expert hot path dominating,
   the runtime check + pointer rebase costs ~5 cycles/tile.
3. **Bias/scales/biases base pointer recomputation.** One extra add per
   per-K-iteration vs dense kernel.

Rough estimate from microarchitecture: Option A hot path should deliver
**60-80% of dense kernel's per-tile throughput**, putting gather kernel
at **+10% to +20% over MLX gather_qmm baseline** (vs +35% for dense self_qmm).
Lower bound: **+5%** if L2 thrash dominates. Upper bound: **+25%** if expert
slabs fit in L2 and uniform_expert branch is perfectly predicted.

---

## § 4 Bench plan

### 4.1 Microbench harness

Mirror the structure of `ironmlx/src/nn/self_qmm/mod.rs` test module
(L92-175):
- Test in `ironmlx/src/nn/self_qmm/gather/tests.rs` (new module).
- Shape sweep covering Qwen3.5-35B-A3B-4bit MoE actual dimensions
  (per `models--mlx-community--Qwen3.5-35B-A3B-4bit/config.json`):
  - `E=256` (num_experts), `K=2048` (hidden_size), `N=512` (moe_intermediate_size)
  - `top_k=8` (num_experts_per_tok)
  - `M ∈ {32, 128, 512, 2048}` (BS*k values for PP ∈ {4, 16, 64, 256})
- Per-shape: warmup 5× → time 20× → median.
- Compare `dispatch_gather_qmm_t` vs `mlx::quantization::gather_quantized_matmul_on`
  (which routes to MLX upstream `gather_qmm_rhs`).
- Decision gate: each shape PASS if self-gather wall ≤ MLX wall × 0.85
  (i.e. ≥+15% gain).

### 4.2 Correctness oracle

Mirror `mod.rs:104-169` `run_variant` pattern:
- Deterministic raw weight data (no random seed).
- Generate `E` weight slabs, `M` tokens, `indices` permutation.
- Call both kernels; assert max-abs-diff < 0.5 (per dense kernel test).
- Test matrix: `{E=4, E=128} × {sorted_indices=true, false} × {bf16, fp16}`.

### 4.3 Integration smoke

After microbench PASS, wire into `sparse_moe.rs:874` behind
`IRONMLX_USE_SELF_GATHER_QMM=1` env flag (mirror `IRONMLX_USE_SELF_QMM=1`
at `mod.rs:17-20`). Run iron-bench against ironmlx with the flag set vs
unset at PP={128, 256, 512, 1024, 2048, 4096, 8192}.

---

## § 5 ROI estimate (with confidence)

### 5.1 Upper bound (+12% root_inclusive at PP=128/512)

Reasoning: assume gather indirection costs ~33% of theoretical kernel speedup
(self_qmm dense +35% → self-gather +23% at kernel level). At PP=512, MoE
gather_qmm_rhs accounts for ~50% of layer wall (3 calls per layer × 28
layers × 2 chunks; rough estimate from T2 +4.97% gain attributing
exclusively to fused gate+up). So:
- 23% kernel reduction × 50% gather_qmm share = **+11.5% wall reduction**
- Closes PP=512 from -35.62% to **-24%** (~12pp improvement).
- Closes PP=128 from -6.13% to **+5.4%** (would exceed +10% target).

**Confidence:** LOW. Three assumptions stacked:
- self_qmm +35% transfers cleanly with ≤33% indirection penalty (unverified
  until prototype).
- gather_qmm_rhs is 50% of layer wall (rough estimate; needs P5h+1
  re-measurement with first_token_sampling lazy-eval gap closed).
- M5 Max kernel performance follows M1 Pro pattern (per
  `[project_cross_device_tuning_deferred]` this is NOT established).

### 5.2 Lower bound (+3-5% root_inclusive)

Reasoning: assume gather indirection costs 70-80% of theoretical kernel
speedup (self_qmm dense +35% → self-gather +7-10% at kernel level), and
gather_qmm_rhs is only 40% of layer wall.
- 10% kernel reduction × 40% gather_qmm share = **+4% wall reduction**
- Closes PP=512 from -35.62% to **-32%** (~4pp improvement).
- Closes PP=128 from -6.13% to **-2.4%** (still below target unless T4 helps).

**Confidence:** MEDIUM. Closer to baseline assumption of indirection cost.

### 5.3 Source of uncertainty (sensitivity ranking)

1. **Gather indirection cost** (HIGHEST uncertainty) — could be 33-80%
   penalty. Only resolved by prototype microbench.
2. **gather_qmm_rhs share of layer wall** (MEDIUM uncertainty) — P5h+1
   measurement gap (first_token_sampling wrapper dominance) blocks
   confident attribution; T2 +4.97% gain is the only direct evidence.
3. **M5 Max kernel transferability** (MEDIUM uncertainty) — self_qmm tile
   tuned on M1 Pro; M5 Max may need different tile (per
   `[project_cross_device_tuning_deferred]`).
4. **L2 cache footprint at large PP** (MEDIUM uncertainty) — Qwen3.5-35B-A3B
   has 256 experts × ~524 KB packed weights per expert (= 131 MB total);
   total weight set far exceeds L2.

### 5.4 Anchored against P5i.a current state

| Lane | Pre-P5i.a (T0) | Post-T2 (current) | Upper-bound P5i.b (+12%) | Lower-bound P5i.b (+4%) | Target (omlx+10%) |
|---|---|---|---|---|---|
| PP=128 pp_tps | -13.31% | -6.13% | +5.4% | -2.4% | ≥+10% |
| PP=512 pp_tps | -37.83% | -35.62% | -24% | -32% | ≥+10% |

**Critical observation:** even the **upper bound P5i.b doesn't close
PP=512 to target** (would still be ~24% short of omlx+10%). PP=128 closes
under upper bound (+5.4% vs +10% target — still 4.6pp short of target;
upper bound assumes 100% T4 also lands which is independent).

This means **P5i.b alone cannot close the program target at PP=512**. Either
PP=512 target needs P5i.c additional work (scheduler / KV / launch
amortization), or the target itself needs reframing as "best-effort partial
gain".

---

## § 6 Cost estimate (P5i.b decomposition if 立项)

Per `[feedback_task_breakdown_bounded]` decompose into 4 sub-phases (≤5-7
tasks each):

### P5i.b.1 — Kernel implementation + microbench (1 week)
- B1.T1: Write `metal/gather_qmm_t.metal.in` template (Option A uniform-
  expert hot path + cold-path fallback).
- B1.T2: Write `nn/self_qmm/gather/{mod,kernel,lookup}.rs` mirroring dense
  self_qmm structure.
- B1.T3: Microbench harness + tile sweep (BM ∈ {8, 16, 32}, BK ∈ {32, 64})
  on M1 Pro target hardware.
- B1.T4: Determine tile choice from sweep + lock lookup_tile entry.

### P5i.b.2 — Correctness oracle + cold/hot validation (3-4 days)
- B2.T1: Generate correctness oracle test suite (E={4, 128}, sorted=true/false).
- B2.T2: Validate uniform_expert hot path vs cold path against MLX upstream
  for boundary-straddle cases.
- B2.T3: bf16 + fp16 dtype coverage.

### P5i.b.3 — Integration + iron-bench sweep (4-5 days)
- B3.T1: Wire env flag `IRONMLX_USE_SELF_GATHER_QMM=1` into sparse_moe.rs.
- B3.T2: Run iron-bench sweep (PP={128, 256, 512, 1024, 2048, 4096, 8192})
  with flag set vs unset.
- B3.T3: Bisect any regression (some PPs may regress if indirection cost
  dominates; flag gate prevents shipping in those cases).

### P5i.b.4 — Close-out + M5 Max tile (3-4 days)
- B4.T1: M5 Max tile sweep (per `[project_cross_device_tuning_deferred]`).
- B4.T2: Update `[project_cross_device_tuning_deferred]` and
  `[feedback_device_aware_tile]` memories.
- B4.T3: Close-out memo with bench evidence + verdict per shape.

**Total: ~2.5-3 weeks** under the assumption that no fundamental redesign
is needed mid-implementation. If indirection cost forces Option B fallback
or larger tile redesign, add ~1 week.

---

## § 7 Scope gate hook (per spec § 5)

**P5i.b is BLOCKED on explicit Boss approval before kernel implementation
begins.** Per `docs/superpowers/specs/2026-05-23-ironmlx-p5i-a-gather-qmm-feasibility-design.md`
§ 5:

> Per Codex P5i review Q7: any Level (a) self-quant gather Metal kernel
> rewrite work (writing new `.metal.in` template + `dispatch_gather_qmm_t`
> function + integrating into `ironmlx/src/nn/self_qmm/`) requires
> **explicit Boss approval BEFORE work begins**.

Within P5i.a: this memo + ROI analysis is the **entire** Level (a) work.
P5i.b would require a fresh spec + plan, requesting the Scope gate trigger
explicitly. P5i.a closing with this memo does NOT auto-dispatch P5i.b.

---

## § 8 Recommendation

**暂缓 (defer P5i.b). Recommended next steps:**

1. **Execute T4 first** (gda_step_1a_in_proj_qkvz hot-path tweak per P5h+1
   ranking). Low risk, no Scope gate. May close PP=128 to target without
   needing P5i.b. If T4 closes PP=128, the strongest argument for
   immediate P5i.b dispatch dissolves.

2. **Dispatch P5i.c new-candidate discovery** to attribute the residual
   PP=512 gap. The -35.62% gap is currently classified as "kernel-bound
   per T0b H4" but T0b H4 was a small-PP hypothesis. At PP=512 the gap
   could come from launch-count overhead (8 gather_qmm calls per layer ×
   28 layers × 2 chunks = 448 dispatches), KV-cache transition cost, or
   the indirection itself in MLX upstream. A measurement pass (with
   first_token_sampling lazy-eval gap closed per P5h+1) would clarify
   where the 2-4 weeks of P5i.b effort would actually land in attribution
   space.

3. **Reconsider P5i.b立项 after (1) + (2).** If P5i.c attribution confirms
   that ≥50% of the PP=512 gap is inside the gather_qmm kernel itself
   (not launch/scheduler/KV), AND T4 doesn't independently close PP=128,
   THEN dispatch P5i.b under Boss Scope gate with full design memo
   already prepared (this document).

**Rationale for 暂缓 vs 立项:**

- **2-4 weeks is a non-trivial commitment.** Opportunity cost vs P5i.c
  is real — P5i.c might find a +10pp closing strategy at <1 week cost.
- **Upper-bound ROI doesn't close PP=512 to target.** Even if everything
  goes right, PP=512 stays ~24% short. The kernel rewrite is therefore not
  sufficient on its own; it's a partial contribution that needs P5i.c
  anyway. Sequencing P5i.c first gives Boss better information about
  whether P5i.b's partial contribution is worth the cost.
- **PP=128 is close to target via cheaper paths (T4, fine-tuning).**
  Committing P5i.b for ~4pp closure at PP=128 is overkill if T4 +
  small-PP-specific tweaks can do the same.
- **Gather indirection cost is the dominant unknown.** Building a 2-4
  week kernel without first prototyping the indirection in a 1-day
  microbench (`gather_qmm` on dummy weights with controlled indices
  pattern) is high-risk. P5i.c could include a 1-day prototype that
  resolves the upper/lower bound ambiguity, then立项 P5i.b with much
  higher confidence.

**Rationale against 否决:**

- The PP=512 -35.62% gap is real and the kernel-bound classification has
  partial supporting evidence (T0b H4 + T2 saturation). T1+T2 collectively
  delivered ~2pp at PP=512, suggesting wrapper-level options have largely
  plateaued at this batch size.
- self_qmm precedent on dense Linear (+35%) demonstrates that the kernel-
  rewrite playbook works on ironmlx for at least some quant paths. There
  is no evidence yet that the gather variant is fundamentally hopeless.

**Bottom line:** the case for P5i.b is plausible but not yet strong enough
to commit 2-4 weeks. Sequence T4 + P5i.c first to either close PP=128
without P5i.b OR sharpen the attribution that justifies P5i.b.

---

## Appendix A — File pointer index

- self_qmm precedent: `/Users/xin/workspace/ironmlx-backend/ironmlx/src/nn/self_qmm/`
- gather_qmm call sites: `/Users/xin/workspace/ironmlx-backend/ironmlx/src/models/qwen3_5_moe/sparse_moe.rs:867`, `:905`, `:624`
- MLX upstream gather_qmm_rhs (baseline being benchmarked against):
  `/Users/xin/workspace/iron-rivals/mlx/mlx/backend/metal/quantized.cpp:1215-1363`
- P5i.a spec: `/Users/xin/workspace/ironmlx-backend/docs/superpowers/specs/2026-05-23-ironmlx-p5i-a-gather-qmm-feasibility-design.md`
- Stage 9 dense self_qmm findings memory: `project_p8a_stage9_findings`
- Cross-device tile tuning deferred memory: `project_cross_device_tuning_deferred`
- Bounded task breakdown principle memory: `feedback_task_breakdown_bounded`
- P5h T3 MoE substep findings memory: `project_p5h_t3_findings`
