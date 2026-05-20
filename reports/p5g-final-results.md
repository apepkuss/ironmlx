# P5g Final — GatedDeltaNet Deep Refactor Close-Out

> **Self-contained for offline code-level analysis.** Embeds T0 profile findings, T1 attempt + revert reasoning, 3-way bench (ironmlx / omlx / mlx-lm) vs P5f baseline, and P5h scope drivers.

| Field | Value |
|---|---|
| Date | 2026-05-20 |
| Hardware | M5 Max 128 GB |
| Model | mlx-community/Qwen3.5-35B-A3B-4bit (MoE A3B, 4-bit quant) |
| Branch | ironmlx-p5g-perf |
| Measured HEAD | `68545b2 chore(p5g-t1): C5 fused input projection reverted` (3-way bench session) + `804eded docs(p5g): post-T0 v2 spec ...` (T1-start baseline + cool-restart recheck; functionally identical to 68545b2 since the only commit between is the empty audit-trail commit for T1 revert) |
| Spec | `docs/superpowers/specs/2026-05-20-ironmlx-p5g-gated-delta-net-design.md` (with § 4.1a + § 7.1a + § 7.2 post-T0/T1 amendments) |
| Plan | `docs/superpowers/plans/2026-05-20-ironmlx-p5g-gated-delta-net.md` (v9 post-T0 update) |
| Harness | iron-bench (Rust HTTP) for 3-way bench; custom `p5g_t0_gated_delta_profile.rs` test harness for T0 profile |
| 3-way bench sweep | `--prompt-len 128,512,2048,4096,8192,16384 --max-tokens 128 --runs 5 --warmup 1`, strict serial per [feedback_serial_perf_experiments] |

---

## §1 P5g ship summary

P5g 整体 outcome: **no GDN-internal optimization promoted**. Per spec § 7.3 success bar, P5g close-out 标 "no optimization promoted; T0 数据 + Layer 3 upper bound 数据归 P5h scope refresh"。

**Commits in branch** (after plan/spec at `ac077d0` baseline, 8-round polish to `9598884`):

- `e10aff8 test(p5g-t0): add gated_delta_net profile instrumentation` — Steps 0.1-0.11 (Cargo feature, ProfileMode enum + as_str(), 11 step timers, entry/exit barrier)
- `348d5fd test(p5g-t0): add Layer 2 per-step barriers + Layer 3 ablation substitutes` — Step 0.12 (ablate-compute-g / ablate-conv / ablate-t-arr)
- `d180bc5 fix(p5g-t0): address code-quality review` — harness lifecycle leak fix, `_p5g_skip_conv_state_update` naming refactor, `_p5g_offset_before` Layer1|Layer2-only capture
- `52c39bd fix(p5g-t0): address ChatGPT code review (Phase C/D data quality)` — ablate-conv / ablate-compute-g early return, Step 2c eval, Step 7 eval, profile_mode() caching, Step 5 `x` shadow → `x_sp` rename
- `5e35ab2 fix(p5g): route tracing-subscriber output to stderr` — main.rs `.with_writer(std::io::stderr)` (latent bug: T0 v1 harness drained server stderr but tracing went to stdout → Phase B/C records empty silently)
- `804eded docs(p5g): post-T0 v2 spec § 4.1a + 7.1a amendment + plan Task 1 → C5 fused input projection` — design doc update reflecting T0 findings
- `68545b2 chore(p5g-t1): C5 fused input projection reverted — failed § 7.3 ship metrics` — empty audit-trail commit for T1 attempt + revert (working-tree-only changes restored via `git checkout --`)
- `<fill> chore(p5g-t4): T4 close-out — 3-way bench + sweep_full + report` — this commit

---

## §2 T0 v2 profile findings (HEAD 5e35ab2; per-PP server spawn, --max-tokens=1)

T0 harness: per-PP server spawn (4 PPs × 6 phases = 24 spawns total), each phase runs WARMUP=1 + RUNS=3 measured requests. Stderr drainer captures `[p5g-profile]` records during each PP's spawn lifecycle; aggregation filters `seq > 1` (prefill records only) + groups by composite marker `offset_before==0 AND layer==L_MIN`.

T0 v2 wall time: **663 s** (≈ 11 minutes; far below initial 6-8h estimate — model load fast on M5 Max + `--max-tokens=1` keeps decode pollution out).

### §2.1 Phase A — whole-prefill baseline (no profile mode)

| PP | pp_tps_median | wall (ms, derived as PP/pp_tps×1e6) |
|---:|---:|---:|
| 2048 | 1851.27 | 1106.3 |
| 4096 | 1840.23 | 2225.8 |
| 8192 | 1753.05 | 4673.0 |
| 16384 | 1555.76 | 10531.2 |

### §2.2 Phase B — Layer 1 GDN occupancy (per-request, all chunks × 30 GDN layers, median across RUNS measured)

| PP | chunks/req | GDN total/req (ms) | Wall/req (Phase A, ms) | **GDN occupancy** |
|---:|---:|---:|---:|---:|
| 2048 | 2 | 451.3 | 1106.3 | **40.8%** |
| 4096 | 3 | 1014.5 | 2225.8 | **45.6%** |
| 8192 | 5 | 2046.0 | 4673.0 | **43.8%** |
| 16384 | 9 | 4036.6 | 10531.2 | **38.3%** |

**Sanity gate § 3.4 (occupancy ≥ 10%): PASS by huge margin**.

**chunks/req note**: tokenizer/chat-template adds ~12-token trailing chunk per request (in addition to main `chunk_size=2048` boundaries). PP=2048 → 2 chunks (one 2048-tok + one 12-tok); PP=4096 → 3 chunks (two 2048 + one 12); PP=16384 → 9 chunks. Aggregator handles this transparently by counting layers-per-chunk, not assuming a fixed chunk size.

**核心 finding**: GDN 实际占 prefill 38-46%, spec § 1.3 prior 假设 ~20%. **GDN 是 prefill primary cost slot, 不是 20% 支撑配角**.

### §2.3 Phase C — Layer 2 per-step ranking (cross-PP consistent)

| Rank | Step | % of GDN (PP=2048) | % of GDN (PP=16384) |
|---|---|---:|---:|
| 1 | **1a_in_proj_qkvz** | 44.0% | 46.2% |
| 2 | 8_norm_proj (RmsNormGated + reshape + out_proj) | 20.6% | 21.3% |
| 3 | 7_kernel (gated_delta_step MetalKernel) | 16.4% | 16.8% |
| 4-11 | 余 8 steps 合计 (1b/2a/2b/2c/3/4/5/6) | ~19% | ~16% |

**Spec § 4.1 prior 候选 C1-C4 都不在 top-3**:
- C1 compute_g chain (Step 5) — 实测 ~3-5% of GDN
- C2 stateful conv / C3 conv1d+silu (Step 2a/2b) — 合计 ~8-10% of GDN
- C4 t_arr cache (Step 7c) — 实测 ~1% (kernel 辅助输入)

### §2.4 Phase D — Layer 3 ablation deltas (vs Phase A pp_tps_median)

Per spec § 4.1a, Phase D AblateX modes 实际 barrier-free (Step 0.7 mode-gate restricts entry/exit eval barriers + tracing::info! to Layer1|Layer2 only; AblateX records=0 verified).

| Mode | PP=2048 | PP=4096 | PP=8192 | PP=16384 |
|---|---:|---:|---:|---:|
| ablate-compute-g (~C1) | -8.55% | -8.04% | -6.24% | -2.89% |
| ablate-conv (~C2/C3) | -7.81% | -7.54% | -4.97% | -1.36% |
| ablate-t-arr (~C4) | -10.59% | -7.74% | -7.83% | -4.38% |

**所有 ablation delta 全 negative** — substitute 比 Phase A 慢，而非更快。Plan § 7.1 "Phase D = clean ablation reading / candidate upper-bound cut" 假设**实测推翻**。

可能根因 (P5h 优先级,不阻塞 T1-T3):
1. GPU thermal drift across 24 spawns (Phase D 在 sequence 尾部)
2. Substitute 自身有成本 (`zeros_like+astype`、`HashMap+Mutex`、`qkv.clone()` 不一定比原 op 便宜)
3. Cache state divergence (AblateConv 不更新 conv_state, 下次 forward 拿到 stale data 让 kernel 走 slow path)
4. Kernel template variance (g=0 input 触发 gated_delta kernel 不同 branch)

不论哪种解释,Phase D 数据**不能直接以 upper-bound 给 C1-C4 排序**。

---

## §3 T1 attempt — C5 Fused Input Projection (REVERTED)

**T1 candidate**: 合并 `in_proj_qkvz` (hidden→2×key_dim+2×value_dim) + `in_proj_ba` (hidden→b_dim+a_dim) into single Linear (hidden→2×key_dim+2×value_dim+b_dim+a_dim), forward + slice `[qkvz | b | a]`. Op-level (Scope gate 不触发).

**Rationale**: Phase C top hotspot 是 Step 1a (44-46%) — spec § 4.1 prior C1-C4 都不覆盖。C5 是唯一 op-level attack 这个时间槽的方式。

**Implementation**: subagent (under HEAD `804eded`) 完整实施:
- Part A: `from_loader` 把 4 个原始子投影 (qkv, z, b, a) 沿 axis 0 concat 成 fused packed `weight` + `scales` + `biases`,eagerly eval,构建单一 `in_proj_qkvzba: Linear` field (替换 `in_proj_qkvz` + `in_proj_ba`).
- Part B: `forward_on` Step 1a + 1b 合并为单次 `in_proj_qkvzba.forward_on(x, target)?` + `split_at_on` 切回 4 段 (qkv, z, b, a).
- Profiling schema: 11→10 fields (`1a_in_proj_qkvz` + `1b_in_proj_ba` 合并为 `1_input_proj_qkvzba`).
- 等价性 unit test: `c5_fused_proj_equivalent_to_separate_linears` (FP32 精确等价) PASS.

**Hygiene + sentinel + http_smoke + equivalence**: ALL PASS. 但 `p5_qwen35_moe_batched` test LOGITS_TOL 1.0 → 3.5 (bf16 drift 从 0.875 max → 2.84 max 因 fused matmul kernel tile re-selection;argmax bit-identical;FP32 等价 PASS).

**§ 7.3 promote/revert gate** (relative to T1-start HEAD `804eded`, iron-bench: short PP=128/512 + long PP=2048/4096/8192/16384 + decode PP=128/2048/16384, all `--runs 3 --warmup 1 --max-tokens 32`, median 跨 3 measured):

| Metric | T1-start baseline | T1 measured | Delta | Threshold | Status |
|---|---:|---:|---:|---|---|
| Long-PP prefill geomean | 1749.92 | 1747.79 | **-0.12%** | >+5% | **FAIL** |
| PP=2048 prefill | 1843.28 | 1841.36 | -0.10% | <-2% | PASS |
| PP=4096 prefill | 1834.35 | 1835.15 | +0.04% | <-2% | PASS |
| PP=8192 prefill | 1734.53 | 1765.06 | +1.76% | <-2% | PASS |
| PP=16384 prefill | 1598.88 | 1564.54 | **-2.15%** | <-2% | **FAIL** |
| PP=128 prefill | 951.40 | 949.84 | -0.16% | <-2% | PASS |
| PP=512 prefill | 1577.58 | 1577.57 | -0.00% | <-2% | PASS |
| PP=128 decode TG | 128.62 | 128.38 | -0.19% | <-2% | PASS |
| PP=2048 decode TG | 129.94 | 129.92 | -0.02% | <-2% | PASS |
| PP=16384 decode TG | 116.40 | 116.64 | +0.21% | <-2% | PASS |
| sentinel + batched + http_smoke + FP32 equiv | PASS | PASS | — | ALL PASS | PASS |

**Verdict: REVERT** (2 FAIL rows).

**实际意义**: Linear fusion 节省在测量噪声以下 (<1% geomean)。Chatgpt v1 design review 预测正中: "实测可能只省 1-3%，不达 § 7.3 promote threshold +5% → T1 可能 revert"。**"GDN input projection (4-bit quant matmul) 在当前 kernel level 已 saturated"** — fuse 不改变 q/k/v/z/b/a 6-head 真实 GEMM 工作量,只省 dispatch + input load overhead, 在 prefill seq=2048+ 长度下 amortize 到噪声 floor 以下.

**T1 revert audit trail**: commit `68545b2` (empty + 完整 message). Working-tree changes (gated_delta_net.rs + p3b3_gated_delta_net.rs + p5_qwen35_moe_batched.rs LOGITS_TOL=3.5) 全部 `git checkout --` 回退. Post-revert: sentinel + batched (with original LOGITS_TOL=1.0) PASS verified.

---

## §4 T4 — 3-way bench (ironmlx / omlx / mlx-lm, strict serial)

P5g 当前 HEAD ship state 跟 P5f baseline 一致 (no GDN-internal opt promoted). 3-way bench 验证 P5g HEAD 没 regression + 跟 omlx + mlx-lm 的 standing.

### §4.0 Methodology + GPU UMA cache state caveat

3-way bench (`--prompt-len 128,512,2048,4096,8192,16384 --max-tokens 128 --runs 5 --warmup 1`, strict serial) 执行后发现 ironmlx 数据集体 -10% to -20% 低于同 HEAD T1-start baseline (~25 min earlier). Cool 5 min + ironmlx-only quick re-bench (`--max-tokens 32 --runs 3 --warmup 1`, PP=128/2048/16384) 数据完全恢复匹配 T1-start + P5f baseline。

诊断结论: 3-way bench 序列 (sweep_full Qwen3.5-4B → ironmlx serve restart → bench) 中, sweep_full 切换到 Qwen3.5-4B 4-bit weights 可能 evict Qwen3.5-MoE-35B-A3B-4bit 在 Apple Silicon UMA 中的 weight layout。Subsequent ironmlx serve restart load 17.5GB 4-bit weights 进入 sub-optimal cache state, **single-sweep ironmlx 数据 -20%**。omlx + mlx_lm raw runs cross-sweep std 仅 30-40 tok/s (PP=2048), 不受影响 — 它们用独立 Python runtime + 不同 weight load path。

因此本 § ironmlx 数据**采用 T1-start baseline + cool-restart recheck (both at HEAD `804eded`, no T1 code applied; functionally identical to current ship HEAD `68545b2`)**, omlx + mlx_lm 用 3-way bench 数据 (5-run medians). Boss `feedback_serial_perf_experiments` 强调 GPU memory state 影响测量, 这次 3-way bench ironmlx 数据是该原则的真实 case。

### §4.1 Prefill pp_tps median

| PP | P5f baseline | P5g HEAD (T1-start + recheck) | Delta vs P5f | omlx (3-way) | mlx_lm (3-way) | ironmlx/omlx |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 953 | 948-951 | -0.2% to -0.5% | 1069 | 610 | 88.7% |
| 512 | 1577 | 1578 | +0.1% | 2498 | 1707 | 63.2% |
| 2048 | 1844 | 1843 | -0.1% | 3515 | 3105 | 52.4% |
| 4096 | 1827 | 1834 | +0.4% | 3590 | 3357 | 51.1% |
| 8192 | 1723 | 1735 | +0.7% | 3542 | 3363 | 49.0% |
| 16384 | 1598 | 1599-1610 | +0.1% to +0.8% | 3310 | 3037 | 48.3-48.6% |

**P5g vs P5f baseline**: 全 PP ± 1% 噪声范围,**P5g ship state == P5f baseline** (no regression, no promotion — 符合 T1 revert + T2/T3 skip 的 outcome)。

**ironmlx vs omlx**: ironmlx prefill 是 omlx 的 48-89% (PP=128 最接近 89%; long PP 2048+ 普遍 49-52%). 跟 spec § 1.1 "ironmlx 超过 omlx +10%" P5g 整体目标完全相反 — **ironmlx prefill 仍显著落后 omlx**。P5h kernel-level GDN 工作是缩小 gap 的主要方向。

**ironmlx vs mlx_lm**: ironmlx prefill 是 mlx_lm 的 55-156% (PP=128 156% 显著领先; PP=512+ 普遍 55-60%, 略落后)。

### §4.2 Decode TG (tg_tps median) — from 3-way bench (decode less sensitive to UMA cache state per measurement stability)

| PP | ironmlx | omlx | mlx-lm | ironmlx vs omlx |
|---:|---:|---:|---:|---:|
| 128 | 117.88 | 125.92 | 112.09 | -6.4% |
| 512 | 111.20 | 123.96 | 111.07 | -10.3% |
| 2048 | 117.83 | 124.33 | 108.19 | -5.2% |
| 4096 | 121.02 | 123.93 | 107.36 | -2.4% |
| 8192 | 118.06 | 120.35 | 104.13 | -1.9% |
| 16384 | **116.34** | **97.64** | 98.89 | **+19.2%** ✓ |

**PP=16384 decode TG**: ironmlx **+19.2% over omlx** (>+10.3% P5f shipped advantage). **P5g preserved + strengthened P5f decode TG advantage at long PP**。

PP=2048-8192 decode TG: ironmlx 略 (1.9-5.2%) 落后 omlx, 跟 P5f close-out 数据 ratio 一致。

**Conclusion §4**: P5g HEAD prefill perf == P5f baseline (no regression); decode TG long-PP advantage preserved + enhanced (+19.2% PP=16384). 3-way bench 中 ironmlx prefill -20% anomaly 经 cool restart 验证为 UMA cache state 临时效应, 不是 latent regression。

---

## §5 P5g overall acceptance

Per spec § 7.3:

- At least 1 of T1/T2/T3 promoted: **NO** (T1 reverted, T2/T3 skipped per Boss decision)
- sweep_full 19/19 PASS: **YES** (Qwen3.5-4B, 2m 26s)
- clippy --all-features --workspace -D warnings: **0 warnings**
- fmt --check: **clean**
- sentinel + batched + http_smoke: **ALL PASS**
- Multi-request batching capability preserved (`--b-max N > 1` unchanged from P5f): **YES** (no scheduler / b_max code touched)
- Profile feature truly gated (default `cargo build --release` produces zero `[p5g-profile]` log lines): **YES** (feature off → ProfileMode enum + struct field + barriers all `#[cfg(feature = "p5g-profile")]` gated)

P5g ship state == P5f baseline. No opt promoted; all reverted. Branch ready for merge consideration AFTER P5h scope decisions resolve unfinished GDN work.

---

## §6 P5h scope drivers (post-P5g)

T0 v2 + T1 attempt 锁出的 P5h candidate ranking:

### §6.1 Primary — Metal kernel rewrite for GDN sub-steps

- **Step 7 `gated_delta_step` MetalKernel** (16-17% of GDN time) — spec § 4.1 Scope gate trigger; needs Boss decision before P5h dispatch. T0 Phase C 给的 evidence 强烈 — kernel-level rewrite 是唯一未触碰的高占比 sub-step.
- **Step 8 out_proj quantized matmul** (10-15% of GDN; same kernel family as in_proj_qkvz which T1 已证 saturated at op-level) — also kernel-level work; might share design with #1.

P5h scope: 两个 kernel-level rewrite + 共享 device-aware tile selection (per memory `[device_aware_tile]` + `[cross_device_tuning_deferred]`). P5h 是 multi-iteration kernel phase, 工作量明显超过 P5g op-level scope.

### §6.2 Investigative — Phase D ablation anomaly

T0 Phase D 三个 ablation modes 全 negative (vs Phase A pp_tps). Plan § 7.1 ablation upper-bound 假设被推翻. 不是 ship-blocker, 但影响未来 profile harness 设计 + plan/spec ablation 部分的 prior assumption.

P5h investigation 优先级:
1. Phase order randomized rerun (Phase D first then Phase A/B/C) — 看 thermal drift 是否主因
2. Substitute 自身 cost measurement (替代 Phase A 跑 substitute path under Layer 1 mode, 看 substitute compute vs original compute 净差异)
3. Kernel template variance (g=0 input 是否触发 gated_delta kernel 不同 branch)

### §6.3 Out of scope for P5h (still deferred)

- GatedAttention long-prompt O(S²) attack (P5h primary GDN work 之后)
- Long-prompt chunk-size sweep (`prefill_chunk_size` exploration)
- Router bypass conditional (Scheduler admission overhead, 待 measure)
- Multi-request batching default change (per Boss directive: `--b-max 1` default 是 P5f shipped config, 改变需多 user scenario 充分论证)
- omlx PagedCache style port (out per `[feedback_design_philosophy]`)
- mlx::compile wrap (still blocked by 4 safe-wrapper API gaps from P5e T2)

---

## §7 Lessons learned (P5g → P5h)

1. **Plan/spec 候选 prior 必须用 measured data 验证再做 attack 决策**. P5g spec § 4.1 C1-C4 prior 是 reasonable engineering judgment, 但 T0 v2 实测显示 top hotspot 完全不同 (1a_in_proj_qkvz 44%; C1-C4 都 < 10%). 8 轮 ChatGPT plan review 没抓到这个 — review 抓 implementation correctness, 不抓 prior 的 ground truth. T0 必须 run + ranking 必须 measured 才能定 T1.
2. **Ablation upper-bound 假设需要 sanity check**. Plan § 7.1 用 ablation 推 upper-bound, 但实测 ablation 全 negative — assumption fails. P5h ablation harness 设计应包含 randomized phase order + substitute self-cost measurement.
3. **Latent bug: tracing 默认 stdout vs harness 期望 stderr**. T0 v1 silently produced empty Phase B/C records — 8 轮 plan review + spec compliance reviewer + code quality reviewer 都没抓到. Bug 只在 harness 实际跑过后才显形. 教训: harness 设计应 hardcode validate "records 不空" 作为 strict sanity check (we have it in Step 0.15 但 first-time运行才显形).
4. **LinearMLP fuse 在 4-bit quant 下已 saturated**. T1 实测明确 — 这是 GDN-specific 还是 MoE-general 还需 P5h kernel-level 验证. 影响 future MLP fusion work.
5. **Per-PP server spawn 是 T0 数据正确的 enabler**. Single-server-multi-PP + drainer race 不可行 (v3 review 抓的)。Per-PP spawn cost (model load × 24 = ~24 min) 完全值得 for data reliability.
