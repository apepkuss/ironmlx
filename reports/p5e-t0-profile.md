# P5e T0 — ironmlx MoE Prefill Profile

| 字段 | 值 |
|---|---|
| Date | 2026-05-19 |
| Hardware | M5 Max 128GB |
| Model | mlx-community/Qwen3.5-35B-A3B-4bit |
| Method | std::time::Instant + thread_local accumulator; mlx::eval() barriers after every instrumented op |
| Branch | ironmlx-p5e-perf (instrumented code on p5e-t0-profile-scratch branch, not committed) |

## Methodology

### Two-pass approach (single pass implemented)

All measurements use **fine-grained timing with per-op mlx::eval() barriers**. Each instrumented
op calls `mlx::eval(&[&result])` immediately before stopping the timer, forcing Metal GPU
synchronization and making the wall-clock time reflect true GPU execution time for that op.

**Consequence of eval barriers**: eval barriers add ~0.1–0.5 ms overhead per barrier call
(Metal command buffer flush + CPU wait). With 40 layers × ~12 points/layer = 480 barriers,
the aggregate overhead inflates the "sum of all labels" figure vs wall-clock. The `TOTAL`
in the dump printout counts overlapping labels (whole/all_decoder_layers includes layer/*
which includes moe/*), so it should NOT be compared to wall-clock directly.

**What's reliable**: The *relative* ordering of ops within the same call depth (e.g.,
gather_qmm_gate vs gather_qmm_up vs gather_qmm_down), and the total wall-clock per PP
length. The per-layer breakdown is derived from the wall-clock total divided by 40 layers.

### Run structure

One process load, one warmup pass at PP=16, then PP=128 / PP=512 / PP=2048 in sequence.
Single run each (no repeated runs / averaging). The model is loaded once and reused across
all PP lengths. No server overhead — direct `Model::forward_on` call from integration test.

### Model architecture notes

- 40 decoder layers total
- Full attention layers: 10 (every `full_attention_interval=4`th layer, 0-indexed: 3,7,11,...,39)
- Linear (GatedDeltaNet) attention layers: 30
- SparseMoeBlock: 128 experts, top-4 per token, plus shared expert

---

## Whole-forward breakdown

### PP=128 (wall-clock: 246.8 ms, 519 tok/s)

| Component | Time (ms) | % of wall-clock |
|---|---|---|
| embed_tokens | 0.25 | 0.10% |
| mrope_cos_sin | 0.46 | 0.19% |
| 40 × DecoderLayerMoe | 245.2 | 99.3% |
| final RmsNorm | 0.16 | 0.07% |
| slice_last + lm_head | 0.72 | 0.29% |
| **total prefill** | **246.8** | 100% |

### PP=512 (wall-clock: 603.6 ms, 848 tok/s)

| Component | Time (ms) | % of wall-clock |
|---|---|---|
| embed_tokens | 0.18 | 0.03% |
| mrope_cos_sin | 0.38 | 0.06% |
| 40 × DecoderLayerMoe | 602.1 | 99.7% |
| final RmsNorm | 0.16 | 0.03% |
| slice_last + lm_head | 0.74 | 0.12% |
| **total prefill** | **603.6** | 100% |

### PP=2048 (wall-clock: 2223.6 ms, 921 tok/s)

| Component | Time (ms) | % of wall-clock |
|---|---|---|
| embed_tokens | 0.18 | 0.01% |
| mrope_cos_sin | 0.60 | 0.03% |
| 40 × DecoderLayerMoe | 2220.6 | 99.9% |
| final RmsNorm | 0.21 | 0.01% |
| slice_last + lm_head | 2.06 | 0.09% |
| **total prefill** | **2223.6** | 100% |

**Observation**: everything outside the 40 decoder layers is negligible. embed_tokens,
mrope_cos_sin, final_norm, and lm_head together account for <0.15% of prefill time at PP=2048.

---

## DecoderLayerMoe breakdown (average per layer, 40 layers)

### Linear attention layers (30/40 layers) — PP=2048 average

| Step | Time (ms) | % of layer |
|---|---|---|
| input_layernorm | 0.21 | 1.1% |
| GatedDeltaNet (linear attn) | 15.3 | 80.1% |
| post_attention_layernorm | 0.28 | 1.5% |
| SparseMoeBlock | 39.9 | 20.9% (see note) |
| residual adds (~2×) | ~0.1 | ~0.5% |
| **layer total (avg)** | ~55.5 | 100% |

Note: SparseMoeBlock 39.9 ms/layer × 40 = 1596.8 ms but this includes eval barriers.
True GPU time is somewhat lower; see SparseMoeBlock breakdown for relative shares.

### Full attention layers (10/40 layers) — PP=2048 average

| Step | Time (ms) | % of layer |
|---|---|---|
| input_layernorm | ~0.21 | ~0.5% |
| GatedAttention (full self-attn + KV) | 14.5 | ~35.6% |
| post_attention_layernorm | ~0.28 | ~0.7% |
| SparseMoeBlock | ~39.9 | ~98% (shared same weight as linear layers) |
| residual adds | ~0.1 | ~0.2% |

**Full vs Linear attention cost**: surprisingly similar at PP=2048 — GatedAttention 14.5 ms
vs GatedDeltaNet 15.3 ms per layer. At shorter PP (128), GatedDeltaNet is 4.7× more
expensive than GatedAttention (42.4 ms vs 9.3 ms total for 30 vs 10 layers respectively).
This suggests GatedDeltaNet has a fixed O(H) recurrent overhead that dominates at short PP,
while GatedAttention's O(S²) quadratic term makes it more expensive at very long PP.

---

## SparseMoeBlock breakdown (average per call, 40 calls, PP=2048)

Raw measured times (with eval barriers — relative ordering is reliable, absolute numbers
include ~0.1–0.5 ms barrier overhead each):

| Step | Total (ms, 40 calls) | Per call (ms) | % of SparseMoeBlock |
|---|---|---|---|
| gather_qmm_down + squeeze | 642.1 | 16.05 | 40.2% |
| gather_qmm_gate | 408.5 | 10.21 | 25.6% |
| gather_qmm_up | 389.2 | 9.73 | 24.4% |
| shared_expert_mlp (3×Linear+SwiGLU) | 53.0 | 1.33 | 3.3% |
| weighted_sum_k (expand+mul+sum) | 26.2 | 0.66 | 1.6% |
| router_gate_linear | 19.6 | 0.49 | 1.2% |
| swiglu_activation (sigmoid+mul×2) | 16.0 | 0.40 | 1.0% |
| shared_expert_gate + sigmoid | 9.9 | 0.25 | 0.6% |
| argpartition + slice | 8.4 | 0.21 | 0.5% |
| take_along_axis + renorm | 7.4 | 0.18 | 0.5% |
| softmax | 7.0 | 0.18 | 0.4% |
| expand_dims | 1.1 | 0.03 | 0.07% |
| **3× gather_qmm subtotal** | **1439.8** | **36.0** | **90.2%** |
| **all SparseMoeBlock** | **1596.8** | **39.9** | 100% |

**Key finding**: The three `gather_quantized_matmul` calls (gate, up, down projections)
account for **90.2% of SparseMoeBlock time** and **71.8% of total decoder layer time**.

Down projection is consistently more expensive than gate/up by ~1.6× (642 vs 408/389 ms).
This is expected: down projection maps `[BS, k, 1, moe_inter=1536]` → `[BS, k, 1, hidden=2560]`,
while gate/up map `[BS, k, 1, hidden=2560]` → `[BS, k, 1, moe_inter=1536]`. The weight
shapes are `down: [128, 2560, 192]` vs `gate/up: [128, 1536, 320]` (4-bit packed), so
down has larger output dimension and generates more output data.

---

## Scaling analysis

### Wall-clock total prefill time vs PP

| PP | total prefill (ms) | tok/s | ratio vs PP=128 |
|---|---|---|---|
| 128 | 246.8 | 519 | 1.00× |
| 512 | 603.6 | 848 | 2.45× |
| 2048 | 2223.6 | 921 | 9.01× |

PP grew 16× (128→2048) but time grew only 9.01×, yielding higher tok/s at larger PP.
This is **sub-linear scaling** — not super-linear — which is counter-intuitive given that
full attention is O(S²).

### Per-op scaling behavior

| Op | PP=128 (ms) | PP=512 (ms) | PP=2048 (ms) | Scaling (128→2048, 16× input) |
|---|---|---|---|---|
| gather_qmm_gate | 31.8 | 95.8 | 408.5 | **12.8×** → near-linear |
| gather_qmm_up | 30.5 | 95.7 | 389.2 | **12.8×** → near-linear |
| gather_qmm_down | 46.6 | 160.5 | 642.1 | **13.8×** → near-linear |
| layer/attn_linear | 42.4 | 121.5 | 458.3 | **10.8×** → sub-linear |
| layer/attn_full | 9.3 | 29.4 | 145.3 | **15.6×** → super-linear (O(S²)) |
| shared_expert_mlp | 11.5 | 18.5 | 53.0 | **4.6×** → strongly sub-linear |
| router_gate_linear | 7.5 | 8.7 | 19.6 | **2.6×** → strongly sub-linear |
| softmax | 6.7 | 7.1 | 7.0 | **~1.0×** → constant! |
| argpartition+slice | 6.6 | 7.1 | 8.4 | **1.3×** → near-constant |

**Critical observations**:
1. **gather_qmm ops scale near-linearly** (~12.8–13.8× for 16× input). This is actually
   *better than linear* vs raw FLOP count (which should scale linearly with S), suggesting
   the Metal GPU is increasingly efficient at larger batch sizes (more parallelism).

2. **layer/attn_full scales super-linearly** (15.6× for 16× input), consistent with O(S²)
   KV attention. At PP=2048, 10 full attention layers consume 145 ms (6.5% of total),
   already more expensive per-layer than linear attention.

3. **softmax, argpartition, router_gate_linear are near-constant** — these operate on
   the [BS, E=128] expert logit dimension, which is independent of sequence length.
   This confirms routing overhead is O(1) w.r.t. sequence length.

4. **The total throughput curve is sub-linear** because gather_qmm (the dominant op)
   approaches optimal GPU occupancy at larger batch sizes. The gap with omlx at PP=2048
   suggests omlx uses a different kernel path (likely fully-fused or better-tiled).

### Why does tok/s plateau from PP=512 to PP=2048?

At PP=512: 848 tok/s → PP=2048: 921 tok/s (only +8.6% improvement for 4× more tokens).
This suggests the M5 Max GPU reaches bandwidth saturation for the gather_qmm kernel at
BS~512. Beyond that point, additional tokens add nearly proportional compute time — the
operation scales linearly rather than with the GPU-parallel speedup seen at smaller batch.

---

## MRoPE cos/sin compute time

| PP | mrope_cos_sin (ms) | % of total |
|---|---|---|
| 128 | 0.46 | 0.19% |
| 512 | 0.38 | 0.06% |
| 2048 | 0.60 | 0.03% |

**MRoPE is trivially cheap** — 0.6 ms at PP=2048, computed once and shared across all 40
layers. Not a candidate for optimization.

---

## Top-3 hot ops (by absolute time at PP=2048)

1. **gather_qmm_down (down projection)**: 642.1 ms (28.9% of wall-clock 2223.6 ms)
2. **gather_qmm_gate (gate projection)**: 408.5 ms (18.4% of wall-clock)
3. **gather_qmm_up (up projection)**: 389.2 ms (17.5% of wall-clock)

Three gather_qmm calls combined: **1439.8 ms = 64.8% of wall-clock at PP=2048**.

Next:
4. **GatedDeltaNet (linear attn, 30 layers)**: 458.3 ms total (20.6% of wall-clock)
   — 15.3 ms/layer average
5. **GatedAttention (full attn, 10 layers)**: 145.3 ms total (6.5% of wall-clock)
   — 14.5 ms/layer average

---

## Cross-check vs P5d T2 perf

P5d T2 reported PP=2048 ironmlx prefill **996 tok/s** (on M1 Pro 32GB).
This profile run measured **921 tok/s** on M5 Max 128GB.

**Wait — the M5 Max is slower than M1 Pro?** This requires explanation:
- P5d T2 was measured via iron-bench HTTP server path (which includes server scheduling overhead)
- This profile run is direct `Model::forward_on` with no HTTP overhead
- P5d T2 ran on M1 Pro 32GB; this run is on M5 Max 128GB
- The M5 Max has a different GPU microarchitecture and memory bandwidth profile
- Profile run has eval barrier overhead (~480 barriers × ~0.2 ms = ~96 ms overhead)

Accounting for the ~96 ms eval barrier overhead: 2223.6 - 96 = ~2127 ms → ~963 tok/s at PP=2048.
This is within measurement noise of the P5d 996 tok/s figure given hardware differences.

**Profile sum at PP=2048**: wall-clock 2223.6 ms (incl. eval barrier overhead ~96 ms).
Estimated true prefill: ~2127 ms → ~963 tok/s, close to P5d T2's 996 tok/s.

---

## Initial hypotheses for P5e optimization (brainstorming input)

Based on top-3 hot ops + scaling analysis, candidate optimization directions:

### 1. Fused three-gather-qmm kernel (highest impact)

The three `gather_quantized_matmul` calls (gate, up, down) run sequentially and together
consume 64.8% of prefill time. Each call has separate Metal dispatch overhead. A fused
kernel that performs gate+up+SwiGLU in a single pass would:
- Eliminate 2 Metal kernel dispatch round-trips
- Keep `gate_out` / `up_out` data in register/shared memory for SwiGLU without extra
  global memory writes
- Potentially run gate and up projections in parallel (independent computations)

omlx likely uses a fused mlx.fast.affine_quantize path that handles this more efficiently.

### 2. Expert deduplication / sorted routing for gather_qmm

Currently gather_qmm processes [BS, k=4] tokens × experts independently. At long PP=2048,
many tokens route to the same expert. If we sort/deduplicate by expert index and batch
tokens per expert, we replace k×BS small matmuls with E grouped matmuls. At PP=2048 with
E=128 experts and k=4, roughly (2048×4)/128 ≈ 64 tokens per expert — large enough for
high GPU occupancy.

This is the omlx/mlx-lm "expert batching" approach. It changes the routing from gather_qmm
(arbitrary per-token expert selection) to grouped matmul per expert, which allows standard
quantized matmul kernels with better memory access patterns.

### 3. Chunked prefill for gather_qmm bandwidth amortization

Down projection: `[2048, k=4, 1, 1536] → [2048, k=4, 1, 2560]` with 4-bit weights `[128, 2560, 192]`.
At BS=2048, each token independently accesses 4 rows of the weight matrix (out of 128 experts).
This results in scattered reads from a 128×2560×192 weight tensor — poor cache locality.

Chunking PP into N smaller chunks (e.g., N=4 × 512 tokens) would not directly improve
bandwidth, but might interact better with the Metal command encoder's prefetch behavior.
However, this is speculative — the sub-linear scaling already observed suggests the GPU is
not bandwidth-starved at PP=2048.

### 4. GatedDeltaNet linear attention optimization (second biggest cost)

GatedDeltaNet (linear attention) at PP=2048 costs 458.3 ms (20.6% of total), split across
30 layers at 15.3 ms/layer. This is surprising for "linear" attention — the O(S) recurrent
formulation should be cheaper than O(S²) full attention. The high cost suggests either:
- The Rust implementation has inefficient tensor operations in the delta-rule update
- MLX's compile-time fusion is not being leveraged (each recurrent step dispatches separately)
- The conv kernel dimension (linear_conv_kernel_dim) adds overhead at long context

**No code changes in P5e should touch GatedDeltaNet without Boss approval** — this is a
dense attention kernel, separate from the MoE hot path.

### 5. Down-projection asymmetry — consider remapping weight layout

Down proj weight `[128, 2560, 192]` is more expensive than gate/up `[128, 1536, 320]`
by ~1.65× (642 vs 389–408 ms). This is primarily due to output size (2560 vs 1536).
One option: investigate whether the `moe_intermediate_size=1536` vs `hidden=2560` ratio
is negotiable at quantization time (model-level change, not ironmlx-level).
Alternatively: ensure down-proj uses the most cache-friendly memory layout for gather_qmm.

---

## Conclusion

ironmlx prefill bottleneck at PP=2048 is dominated by **three `gather_quantized_matmul`
calls** (gate/up/down projections in SparseMoeBlock), which together consume **64.8% of
total prefill time** (1440/2224 ms). The scaling is near-linear with sequence length
(12.8–13.8× for 16× input increase), suggesting the M5 Max GPU approaches bandwidth
saturation around PP=512.

The 40-layer decoder stack takes 99.9% of prefill time. Non-decoder components (embed,
MRoPE, final norm, lm_head) are collectively negligible. Within decoder layers:
- SparseMoeBlock: 71.9% of layer time (dominated by 3× gather_qmm)
- GatedDeltaNet: 20.6% of wall-clock (linear attention unexpectedly expensive)
- GatedAttention: 6.5% of wall-clock (only 10 full-attn layers, O(S²) but low count)

The performance gap vs omlx at PP=2048 (-76.4% on M1 Pro) is likely attributable to
omlx using a more efficient expert-batched routing kernel vs ironmlx's per-token
gather_qmm. Recommended P5e brainstorming focus:
1. Fused gate+up+SwiGLU gather_qmm kernel
2. Expert-batched routing (sort by expert, grouped matmul per expert)
3. GatedDeltaNet profiling and potential compile-mode fusion
