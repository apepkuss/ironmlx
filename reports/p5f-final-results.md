# P5f Final — MoE Text-Only Known-Path Perf Close-Out

> **Self-contained for offline code-level analysis.** Embeds all bench
> data, T2 failure post-mortem, and remaining-gap attribution for P5g
> scope definition.

| Field | Value |
|---|---|
| Date | 2026-05-20 |
| Hardware | M5 Max 128 GB |
| Model | mlx-community/Qwen3.5-35B-A3B-4bit |
| Branch | ironmlx-p5e-perf |
| HEAD post-T3 | filled by close-out commit (see git log) |
| Spec | docs/superpowers/specs/2026-05-19-ironmlx-p5f-known-path-perf-design.md |
| Plan | docs/superpowers/plans/2026-05-19-ironmlx-p5f-known-path-perf.md |
| Harness | iron-bench |
| Sweep | `--prompt-len 128,512,2048,4096,8192,16384  --max-tokens 128  --runs 5  --warmup 1`, strict serial |

## §1 P5f ship summary

- **T1 SHIPPED** (commit 242f21a): CLI default `--b-max` changed from 4 → 1 (single-request optimized; multi-request via explicit `--b-max N > 1`).
- **T2 REVERTED** (not committed): GenerationStream single-shot when KV budget allows. Implementation completed per spec; iron-bench measured **negative ROI** (PP=8192 −4%, PP=16384 −12.5%). Reverted before commit per Boss "对系统产生的负面影响最小" principle. See §4 for full post-mortem.
- **T3 (this commit)**: close-out bench + report + sweep_full validation.

## §2 P5f vs P5e baseline measurement (default `--b-max 1`)

Median across 5 timed runs + 1 warmup (M5 Max 128 GB):

| PP | P5e baseline (b_max=4) | P5f current (b_max=1, T1) | P5f delta | omlx (default) | omlx+10% target | P5f vs target |
|---:|---:|---:|---:|---:|---:|---|
| 128 | 390 | 953 | +144.3% | 1078 | 1186 | −19.6% |
| 512 | 491 | 1577 | +221.1% | 2635 | 2898 | −45.6% |
| 2048 | 1842 | 1844 | +0.1% | 4230 | 4653 | −60.4% |
| 4096 | 1773 | 1827 | +3.1% | 4413 | 4855 | −62.4% |
| 8192 | 1725 | 1723 | −0.1% | 4347 | 4782 | −64.0% |
| 16384 | 1548 | 1598 | +3.2% | 3865 | 4252 | −62.4% |

P5e baseline column from `reports/p5f-baseline.md`. omlx column = omlx prefill PP tok/s from this T3 bench (re-measured today); +10% target = omlx × 1.10.

Observations:
- T1 delivers massive prefill ROI at PP=128/512 (+144% / +221%), confirming `b_max=4` scheduler-loop overhead was the dominant bottleneck at short PP for the single-request idle-server case.
- T1 is essentially neutral at PP=2048+ (±3%), as expected (those routes go through GenerationStream, not Scheduler admission loop).
- Residual gap to omlx+10% target is **largest at PP=2048-8192 (60-64%)**; this is the P5g target band.

## §3 Decode + e2e

### Decode TG (tok/s, median over 5 runs)

| PP | ironmlx TG | mlx_lm TG | omlx TG | omlx+10% target |
|---:|---:|---:|---:|---:|
| 128 | 124.55 | 113.20 | 128.65 | 141.52 |
| 512 | 124.46 | 113.41 | 127.17 | 139.89 |
| 2048 | 124.47 | 110.43 | 124.81 | 137.29 |
| 4096 | 121.96 | 108.08 | 123.96 | 136.36 |
| 8192 | 117.45 | 104.82 | 119.95 | 131.95 |
| 16384 | 112.18 | 106.21 | **101.67** | 111.84 |

Highlight: at PP=16384 ironmlx decode TG 112.18 tok/s **already beats omlx** 101.67 (+10.3%). At PP=128-8192, ironmlx decode is ~3-5% behind omlx — small fixed gap, likely scheduler poll cost.

### E2E (seconds for PP + 128 generated tokens; median)

| PP | ironmlx e2e (s) | mlx_lm e2e | omlx e2e | omlx−10% target |
|---:|---:|---:|---:|---:|
| 128 | 0.97 | 1.16 | 0.95 | 0.85 |
| 512 | 0.89 | 0.69 | 0.62 | 0.56 |
| 2048 | 1.80 | 1.54 | 1.50 | 1.35 |
| 4096 | 3.29 | 2.23 | 1.96 | 1.76 |
| 8192 | 5.84 | 3.29 | 2.95 | 2.66 |
| 16384 | 11.39 | 4.82 | 4.36 | 3.92 |

E2E gap dominated by prefill at long PP (PP=16384: ironmlx 11.39s vs omlx 4.36s — 2.6× wall-time, all in prefill phase).

## §4 T2 post-mortem (REVERTED, not shipped)

### What spec assumed

P5f spec §4 hypothesized: when prompt_len > prefill_chunk_size, the chunked-prefill loop's per-chunk `mlx::transforms::eval(&[&hidden])` barriers are a significant overhead; replacing chunked path with a single-shot forward (when KV budget allows) should yield 1.9-2.5× prefill speedup at PP=4096-16384.

### What measurement showed

iron-bench with T2 dispatch enabled (staged but not committed):

| PP | Baseline (chunked, T1 default) | T2 (single-shot when budget allows) | Delta |
|---:|---:|---:|---:|
| 4096 | 1773 | 1792 | +1% (noise) |
| 8192 | 1725 | 1660 | **−4%** |
| 16384 | 1548 | 1355 | **−12.5%** |

T2 implementer ran microbench at PP=4096 directly comparing 3 chunk strategies (forward_text_hidden loop calls):

| Strategy | Total ms |
|---|---:|
| Path A: `prefill_chunk_size=0` (single-shot) | 2222 |
| Path B: `prefill_chunk_size=2048` + T2 dispatch → single-shot | 2217 |
| Path C: `prefill_chunk_size=512` (8 chunks, bypass T2) | 2213 |

At PP=4096, all three strategies are ≈ equivalent (within 0.4%). But iron-bench at PP=8192/16384 shows clear single-shot regression.

### Re-attribution

Spec's "per-chunk eval barriers are the bottleneck" assumption is **wrong**. Microbench shows chunked vs single-shot are nearly identical at PP=4096 (eval barriers ARE nearly free on the M5 Max Metal command queue). The −4% to −12.5% regression at PP=8192/16384 comes from **GPU memory pressure**, not eval barriers:

- Single-shot at PP=16384 must materialize attention intermediates `[1, n_heads, 16384, 16384]` ≈ ~300 MB × n_heads ≈ multi-GB total during the single forward pass.
- Chunked path at chunk_size=2048 keeps intermediates bounded at `[1, n_heads, 2048, KV_offset]`, much smaller working set per call, better fit in Apple Silicon SLC + unified-memory bandwidth.

Mathematically both paths have the same O(N²) attention ops; in practice the memory-access pattern matters at long PP on Apple Silicon.

### Disposition

T2 reverted. `memory_budget` helpers, `generate.rs` dispatch change, and `p5f_long_prompt_single_shot` test were all kept in working tree only during testing and discarded via `git restore` after the negative measurement. **No T2 code in branch history**.

P5g scope (§7 below) is revised: long-prompt prefill optimization must NOT go through "bypass chunking" — it must address the actual bottleneck (memory pressure / kernel selection at long PP).

## §5 Validation gates

- `p5_qwen35_moe_smoke` (argmax=11 sentinel): PASS (default `b_max=1`)
- `p5_qwen35_moe_batched` (B=2 row-equiv): PASS
- `p5_qwen35_moe_http_smoke`: PASS
- Explicit `--b-max 4` smoke (multi-request capability preserved): PASS
- `sweep_full.sh` 19/19: **PASS in 168 seconds** (Qwen3.5-4B-MLX-4bit, M5 Max 128 GB)
- `cargo +nightly clippy --all-features --workspace --release -- -D warnings`: 0 warnings
- `cargo +nightly fmt --all -- --check`: clean
- `cargo build --release`: PASS

## §6 Residual gap to omlx+10% target (P5g scope drivers)

Per-PP attribution of residual gap after P5f T1 only (use canonical T0 profile `reports/p5e-t0-profile.md` for hot-path proportions):

| PP | P5f ironmlx tok/s | omlx+10% target | residual gap | Likely attribution (root cause for P5g) |
|---:|---:|---:|---:|---|
| 128 | 953 | 1186 | −19.6% | HTTP overhead + GatedDeltaNet recurrent fixed cost |
| 512 | 1577 | 2898 | −45.6% | GatedDeltaNet + GatedAttention still |
| 2048 | 1844 | 4653 | −60.4% | **Primary P5g target**: GatedDeltaNet (20%) + GatedAttention (6.5%) per T0 profile; long-prompt memory pressure (per §4 T2 finding) |
| 4096 | 1827 | 4855 | −62.4% | Same + memory pressure starts to bite |
| 8192 | 1723 | 4782 | −64.0% | Long-prompt memory pressure increasingly dominant |
| 16384 | 1598 | 4252 | −62.4% | Memory pressure + GatedAttention O(S²) |

## §7 P5g candidates (revised after T2 finding)

Ranked by expected impact:

1. **GatedDeltaNet independent profile + optimization** (linear attn, 30/40 layers, T0 profile 20% at PP=2048)
   - Read current `ironmlx/src/nn/gated_delta_net.rs`; profile per-op
   - Independent design improvement (NO copy from omlx.patches per [feedback_no_spec_from_competitors])

2. **GatedAttention optimization + memory-pressure mitigation** (full attn, 10/40 layers, super-linear scaling)
   - O(S²) growth dominates long PP
   - **NEW: must consider memory-access pattern** (T2 finding showed memory pressure beats compute count at long PP)
   - Candidates: SDPA dispatch tuning, attention intermediate streaming, KV layout for cache-friendly access

3. **Router bypass for single-request idle server** (if Scheduler admission/queue overhead > 50ms — needs measurement)

4. **Multi-request batching deferred capability (P5h / P6+, separate phase)** — per Boss directive (2026-05-19):
   - `--b-max N > 1` already functional; P5f only changed default
   - Future work items: PagedCache evaluation, ragged batching, dynamic b_max, admit_mid efficiency
   - Trigger: when ironmlx enters multi-user / agent-fleet deployment

## §8 Out of P5f scope (deferred capabilities)

- **T2 (single-shot fallback)**: REVERTED. Spec assumption wrong; needs P5g to consider memory-pressure-aware approach instead of bypass-chunking.
- **Multi-request batching default re-evaluation**: `--b-max N > 1` works today via explicit flag. The default-to-1 choice is single-request optimal; multi-request deployment will revisit default selection in P5h / P6+.
- **omlx-style PagedCache**: not aligned with current ironmlx design ([feedback_design_philosophy]). Reconsider only if multi-request scaling shows demand.
- **mlx::compile wrap**: still blocked by 4 safe-wrapper API gaps from P5e T2.

## §9 Cross-reference: omlx (observation only)

omlx achieves 4230 tok/s at PP=2048 (vs ironmlx P5f 1844 tok/s) via its 4-layer optimization stack:
1. `omlx.patches.gated_delta_advance` monkey-patches `Qwen3_5GatedDeltaNet`
2. `omlx.patches.qwen3_5_attention` monkey-patches `Qwen3_5Attention`
3. PagedCache (block 256→2048 auto-tune)
4. Engine = vlm path (text-only also routes through vlm engine)

Per [feedback_no_spec_from_competitors]: omlx is observation only, not an alignment target. ironmlx independently designs improvements based on its own architecture; reaching omlx+10% via independent design is the goal but the implementation path is independent.

## §10 P5f ship-state summary

What was the **measured ROI of T1 alone** for the common case (PP≤512):

- PP=128 prefill: 390 → 953 tok/s (2.44×)
- PP=512 prefill: 491 → 1577 tok/s (3.21×)
- PP=128 decode TG: 79 → 124.5 tok/s (1.58×)
- PP=512 decode TG: 79 → 124.5 tok/s (1.58×)

T1 alone delivers the bulk of expected sanity ROI. PP=2048+ unchanged by T1 (those routes go through GenerationStream, not Scheduler). Long-prompt gain requires P5g.

Multi-request batching is preserved as a deferred capability: `--b-max N > 1` continues to work and is fully functional; only the CLI default flipped to single-request-optimized. Re-evaluation of the default (and any new batching infrastructure such as PagedCache, ragged batching, dynamic b_max) is scheduled for P5h / P6+ if multi-user deployment demand surfaces.
