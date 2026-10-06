# DFlash2 前缀缓存零命中 prefill —— 结果

日期 2026-10-06。分支 `perf/dflash2-cache-miss-prefill`，worktree
`/Users/xin/workspace/perf-dflash2-cache-miss-prefill`，基线 dev
6c1774ce（与预期一致，dev 未变化）。候选已做本地检查点提交 2572078a（只用于固定候选）；后续诊断增强与补充验收证据分别提交，不改变冻结候选身份。未推送、未合并。测量协议见
`docs/protocol.md`（正式计时前写定）和 `docs/protocol-addendum-30k-hit.md`（30K 命中诊断前写定）。历史上 M5 产品化的短输入 TTFT 未达标结论不受本次影响。

## 状态

原正式计时 `timing-v1` 中，“30K 完全重复（命中）TTFT”未满足，这一结论保留不变。Boss 选择方案 (b) 后，按独立预登记协议做了 R1 专项确认：主判据和正确性均满足，见“R1 专项确认”一节。

- 原计时其余判据都满足；
- 30K 命中项由专项确认补足，但它的区间很宽，限制写在专项确认一节。

候选已在本地做检查点提交 2572078a；后续提交分别保存诊断增强与补充验收证据。是否合入由 Boss 决定。

**逐项状态：**

| 项目 | 状态 |
| --- | --- |
| 主目标（零命中短输入 TTFT） | 满足 |
| Decode、E2E、短输入命中 TTFT、30K 零命中 TTFT、内存峰值 | 满足 |
| 正确性（17 个场景逐 token）、命中能力、回退 | 满足（文档分叉的 token 证据已补齐，见“正确性”） |
| 30K 完全重复（命中）TTFT ≤1.10 | `timing-v1`：**未满足**（1.121，保留）；R1 专项确认：**满足**（0.848，区间 [0.654, 1.337]） |
| 物化双峰的根本原因 | 未查明 |
| AGENTS.md 检查、App bundle | 满足 |
| 内存下降中约 1.4 GiB 的来源 | 未知 |

## 结论

| 项目 | 结果 | 结论 |
| --- | --- | --- |
| 主场景：开缓存、预热后零命中的短输入 TTFT | C/B 0.789 [0.705, 0.840] | 满足（判据：≤0.95 且上界 <1） |
| Decode TPS | 1.001 [0.994, 1.008] | 满足 |
| E2E | 0.993 [0.987, 1.002] | 满足 |
| 短输入命中 TTFT | 1.058 [0.451, 1.291]，命中数一致 | 满足 |
| 30K 零命中 TTFT | 1.002 [0.988, 1.012] | 满足 |
| 30K 完全重复（命中）TTFT | 1.121 [0.896, 1.215] | **未满足**（判据点估计 ≤1.10）；见下文，属证据不足 |
| 内存峰值 | 生命周期 0.925，30K 未命中请求 0.925 | 满足（下降） |
| 输出正确性、缓存命中能力、回退 | 见“正确性” | 满足 |

**总体：** 主目标达到，零命中短输入 TTFT 降低约 21%，没有观察到正确性、命中能力、Decode、E2E 或内存退化。30K 命中 TTFT 这一项按登记判据未通过：
- 两侧数据都呈双峰，置信区间包含 1；
- 后续诊断把耗时差异定位到命中恢复的物化一步，慢峰在两臂都出现；
- 但没有找到与本次改动有关的机制，也不能排除紧跟 30K 零命中之后的第一次重复（R1）在候选上更常变慢。原计时未观察到频率差异，新诊断中候选偏慢。

既不能确认退化，也不能确认没有退化，记为“未满足（证据不足）”，没有补跑。

## 根因

有前缀缓存时，`new_scheduler_b1_text_only_with_cancellation` 要求 `prefix_cache.is_none()` 才走单图 prefill，所以缓存开启时即使零命中也走 `SchedulerB1`：
- **分块方式。** `dflash2_prefill_chunk_len` 在 SchedulerB1 下把不超过 chunk size（默认 2048）的 prompt 拆成 `[N-1] + [1]` 两次完整前向。
- **保存边界。** `should_cache_dflash2_prefill_boundary` 因此在 N-1 和 N 两个边界各保存一次条目。

诊断（`results/diagnose-phases-v1`）用默认关闭的阶段计时，只在已有同步点读时钟，分解结果如下。

**短输入、开缓存：**
- `[N-1]` 前向 100–170 ms；
- **额外的 `[1]` 前向 35–40 ms**；
- 两次保存每次 0.1–0.3 ms；
- 首个 logits 1.5–7.7 ms。

**短输入、关缓存（单图）：** 一次 `[N]` 前向 103–140 ms，与 `[N-1]` 相当。

**30K：** 两种方式分块相同（15×2048 + 12）；保存 0.1–0.4 ms，触发淘汰 17 条时为 29 ms。

**判断：** 瓶颈是 B1 的 `[N-1]+[1]` 拆分，不是缓存保存，也不是同步。30K 输入没有缓存相关的可优化开销。

**N-1 条目是否有用：** 场景实测中，追加、分叉在两侧都命中完整 prompt 的 N 条目（62/46/83 token；30K 为 30732），没有请求依赖 N-1 条目。

**进程首个请求：** 两侧都比预热后慢（B 236 ms，C 195 ms，中位数），这与 Metal 管线首次使用有关。诊断中开缓存进程首请求 794 ms、关缓存 158 ms，很可能是先运行的进程填充了系统级 Metal 管线磁盘缓存；正式计时用交错顺序和不计分的适应会话覆盖了这一影响。

## 改动（`binaries/ironmlx-candidate-f007c386.source.diff`）

改动在 `ironmlx-runtime/src/core/dflash2.rs`：
- 单图 prefill 的资格条件不变：M5 profile 的 `DFLASH2_SINGLE_PREFILL` 设置，加上 M5 affine4 路由指纹。
- 新增 `cold_single_prefill`：开了前缀缓存时，若查找后没有恢复任何前缀（`position == 0`，包括恢复失败回退冷 prefill 的情况），本次 prefill 改用单图方式，保存的条目来自同一次前向。
- 只要恢复了前缀，仍走 SchedulerB1。没有缓存的路径不变。
- 新增默认关闭的诊断 `IRONMLX_DIAGNOSTIC_DFLASH2_PREFILL_PHASES=1`，只记录各阶段耗时，不增加 eval。

`dflash2_actor.rs`：前缀指纹的 prefill 标签由 `scheduler-b1-chunk-v1` 改为
`scheduler-b1-chunk-v2-cold-single`，表明条目的生成规则变了。条目只存在进程内，不会与旧二进制的条目混用。

**有意保留的行为：**
- 长输入（> chunk size）两种方式分块相同，结果不变；
- M5 profile 关闭或单图 prefill 设置关闭时，回到原 SchedulerB1；
- 短输入零命中现在只保存完整 prompt 一个条目，不再保存 N-1 条目。

## R1 专项确认（`results/r1-confirmation-v1`，分析 `evidence/r1-confirmation-v1-analysis.json`）

**协议：** `docs/protocol-r1-confirmation.md`，运行前写定；顺序文件 `docs/r1-confirmation-order.json`。

**设置：**
- 使用原冻结的正式二进制 B b9c7b4fb、C f007c386，每个会话前核验哈希；不开诊断。
- 会话内容与 `run_timing.py` 完全相同；只取紧跟 30K 零命中之后的第一次完全重复（R1）。
- 一对不计分适应会话，加 24 个计分配对轮次（BC、CB 各 12 轮，种子 20261007）。
- 只运行一次，运行中没有查看结果。

**结果：**
- 50 个会话全部有效，24 个轮次全部计分，没有残留进程。
- **主判据：** C/B = 0.848 ≤ 1.10，**满足**。
  - 中位数 B 76.8 ms、C 65.2 ms；
  - 配对 bootstrap 95% 区间 [0.654, 1.337]。
- **每轮差值 C − B：** 中位数 −1.0 ms，区间 [−9.3, +10.2] ms；各轮原始值见分析文件的 `per_round`。
- **慢峰比例（服务端 prefill > 60 ms）：** B 12/24，C 8/24。
- **正确性满足：** 每次 R1 都命中全部 30732 token；R1 输出等于同会话 30K 零命中的输出；所有会话、两臂的 R1 输出相同。

**限制：**
- 两臂的 R1 都是双峰（约 30–40 ms 或 65–107 ms 的服务端 prefill），中位数比值对落在哪一峰很敏感。区间上界 1.337 超过 1.10，所以这批数据不能排除 R1 中位数存在超过 10% 的退化。按预登记规则，判据只看点估计。
- 本次没有观察到候选提高慢峰概率（C 8/24 对 B 12/24），但 24 轮的样本不足以确认概率相等。
- 物化双峰的根本原因仍未查明；它在两臂都出现。
- 原 `timing-v1` 的未满足结论保留，没有被本次结果替换。

## 正确性与缓存行为（`results/scenarios-main-v1`、`scenarios-fallback-v1`、`scenarios-docfork-v1`）

所有会话都开启 token-id 诊断；逐请求比较 prompt token、生成 token 和结束状态（`evidence/scenarios-*-compare.json`）。

- **C-on 零命中 = 无缓存参考。** C-on 的零命中输出与 B-off、C-off 逐 token 相同（短 code-1、knowledge-1、code-3 和 30K code-1）；完全重复复现零命中结果。
- **C-on = B-on。** 包括零命中、重复、追加、分叉、文档分叉在内的 17 个请求逐 token 相同，用户在开缓存时看到的输出没有变化。
  - 在 `scenarios-main-v1` 中，前 16 个有逐 token 证据；最后一个（30K 文档分叉）的 token 记录被截断，当时只验证了文本和结束原因。
  - `scenarios-docfork-v1` 重跑了 30K 链：4 个会话的 token 记录都完整写完，文档分叉在 C-on、B-on、B-off、C-off 之间逐 token 相同，结束状态相同（`evidence/scenarios-docfork-v1-compare.json`）。
  - 该链其余步骤与主场景同侧会话的文本哈希全部一致，说明重跑复现了原场景。原截断记录保留。
- **C-off = B-off。** 全部相同。
- **开缓存与关缓存的差异。** 追加和分叉在开、关缓存之间不同（例如短 code-1 分叉 490 对 428 token），B 和 C 完全一样，属于命中后续算本来就有的差异，不是本次引入。
- **命中能力。** C-on 每一步的命中 token 都等于 B-on：重复为完整 prompt，追加/分叉 62/46/83，30K 追加/分叉 30732；文档分叉两侧都是 0，因为共享前缀短于 30720 边界。短输入零命中的保存次数 B 为 2、C 为 1。
- **回退。**
  - M5 profile 关闭：C = B，逐 token。
  - 单图 prefill 强制关闭：C 与 B 文本一致，保存 2 次，说明回到原路径。
  - 1 GiB 缓存：30K 条目超过上限不保存，短输入正常命中。
  - 40 GB 内存上限：C = B 逐 token，两侧都发生淘汰（B 7 次、C 4 次），没有请求被拒。
  - 取消：30K 零命中请求在 3 s 后放弃，之后短请求正常；被取消的请求没有留下条目。
- **定时运行中的输出。** 每类请求的 B、C 输出完全相同，各轮之间也稳定。

## 配对计时（`results/timing-v1`，分析 `evidence/timing-v1-analysis.json`）

**设计：**
- 1 对不计分适应会话加 6 轮（BC, CB, CB, BC, BC, CB）；
- 每会话一个新进程，使用 App 默认参数和 App 环境，不开诊断；
- 所有被测零命中请求的命中 token 都是 0，所有重复都命中完整 prompt，审计全部通过。

中位数（ms，6 轮）：

| 请求 | B | C |
| --- | ---: | ---: |
| 预热后零命中 code-1 | 168.8 | 109.8 |
| code-3 | 194.8 | 157.8 |
| knowledge-1 | 181.3 | 175.8 |
| knowledge-2 | 168.2 | 110.0 |
| knowledge-3 | 174.6 | 159.3 |
| 进程首请求 code-2（描述性） | 236.4 | 195.2 |
| 短输入命中 code-1 / knowledge-1 | 20.8 / 20.3 | 20.6 / 23.0 |
| 追加 code-1（命中 62，描述性） | 1246 | 1213 |
| 30K 零命中（s） | 55.55 | 55.68 |
| 30K 重复 | 85.8 | 96.2 |

**Decode** 每任务比值 0.999–1.004，**E2E** 0.99–1.00。

**30K 重复项未满足的原因：**
- 两侧的服务端恢复耗时都是双峰：约 33 ms 或 80–108 ms；
- B 与 C 各有 3 轮在低峰、3 轮在高峰，而且慢峰都出现在第 2、3、5 轮，两臂的慢轮次完全重合（`results/timing-v1/run.json` 中每会话的 `server_prefill_ms`）；
- 中位数落在两峰之间，比值由少数高峰样本决定。

按协议记为未满足，不再补跑。后续诊断见下节。

## 30K 命中诊断（`results/diag30k-hit-v1`，汇总 `evidence/diag30k-hit-v1-summary.json`）

**协议：** `docs/protocol-addendum-30k-hit.md`，运行前写定。

**设置：**
- 8 个会话，顺序 Bd Cd Cd Bd Bd Cd Cd Bd；
- 二进制为基线和候选各自加上细分诊断（f2fc817d / c40265c9）；
- 每个会话先完全复现正式计时的步骤，再做 4 次 30K 完全重复；
- 每臂共 20 次 30K 命中，其中 R1 是紧跟 30K 零命中后的第一次重复。

**观察：**
- **耗时集中在物化。** load 最长 0.025 ms，恢复建图最长 0.10 ms，耗时差异全部在物化（eval 恢复的 KV/GDN 状态）。慢的物化往往伴随较慢的首 logits（6–11 ms，对比 1.6 ms）。
- **慢峰在两臂都有。** 物化或服务端 prefill 超过 60 ms 的请求：Bd 3/20、Cd 4/20；物化均值 33.3 对 34.9 ms。慢请求出现在 R1–R5 的各个位置。
- **未发现与内存计数器的对应关系。** 同一臂内，快、慢请求物化前的 MLX active、MLX cache 和 footprint 基本相同；接纳压力始终为 Normal，没有触发收缩或 `clear_cache`。这只说明这些计数器上看不到对应关系，不能证明慢峰与内存状态无关。
- **R1 都要新申请内存。** R1 前两臂的 MLX cache 都约 1.40 GB，物化时都要向系统新申请约 744 MiB，footprint 增量两臂相同。R2 以后 cache 约 2.87 GB，不再新申请，但仍会出现慢峰。
- **R1 的快慢比例。** 本次诊断中候选偏慢：Cd 3/4 慢，Bd 0/4 慢。原正式计时 timing-v1 未观察到频率差异：两臂都是 3/6 慢，且慢在同一批轮次。诊断二进制与正式二进制不同，两组数据不合并，也不作验收依据。

**判断：**
- 两臂都观察到物化慢峰。这不代表候选不会提高慢峰出现的概率。
- 记录到的内存计数器中，两臂之间系统性的差异只有一项：Cd 的 MLX active 低约 2.8 GB，与少存 N-1 条目一致。这一差异与 Cd 的 R1 偏慢之间，没有找到可解释的联系。
- 原计时未观察到 R1 慢峰频率的差异，新诊断中候选偏慢。两者样本都小，诊断用的又是不同的二进制，不足以判断候选是否让 R1 更常变慢。
- 慢峰的根本原因（GPU 或系统侧的瞬时状态，还是其他因素）没有查明。

**处理：**
- 没有找到可由本次改动解释、且可最小修复的机制，因此不做修复。
- 按协议，不为通过而重跑原判据；原判据仍记为未满足。

**内存：**
- 30K 零命中请求峰值 B 36.13 GiB、C 33.42 GiB。会话生命周期峰值就出现在这次请求，所以比值同为 0.925。
- 能解释的部分：会话末缓存 B 7.34 GB/19 条、C 6.07 GB/11 条，差 1.27 GB，对应 C 少存的 8 个 N-1 条目（每个短输入条目带完整 GDN 状态，约 0.16 GB）。
- 其余约 1.4 GiB 的来源未测清，记为未知。诊断运行中 Cd 的 MLX active 比 Bd 低约 2.8 GB；这只是记录到的事实，不据此扩大归因。

## 检查（AGENTS.md，`checks/`）

- `cargo fmt` 已执行；`cargo +nightly fmt --all -- --check` 通过。
- `cargo +nightly clippy --all-features --workspace -- -D warnings` exit 0。
- `cargo build --release` 通过，无警告；产出 `ironmlx` 与冻结候选 f007c386 一致。
- lib 测试：ironmlx-runtime 543、ironmlx 328、ironmlx-core 95，全部通过。core 首次运行因测试二进制旁缺 `mlx.metallib` 失败 74 项，日志保留为 `lib-tests-ironmlx-core-failed-no-metallib.txt`；临时复制 metallib 后通过，副本已删除。
- App bundle：见下节。
- **诊断改动（独立提交）。** 新增命中路径细分计时和接纳压力记录，都默认关闭。对该诊断源码做了：
  - nightly fmt 检查、strict clippy、workspace release 构建，均 exit 0、无警告；
  - runtime lib 测试 543 项通过（`checks/diag2/`）。

  这次只改了 runtime 中的诊断代码，所以没有重复 ironmlx 和 core 的测试，也没有重做 App bundle。

## App bundle

**构建与静态检查：**
- 命令：`MLX_SRC=/Users/xin/workspace/build-mlx-73ad5df2 ./scripts/build-app-bundle.sh`，日志 `checks/build-app-bundle.log`。
- 产物：`dist/IronMLX.app`（helper 03f6c13f…，mlx.metallib 26485555…，ad-hoc 签名，未公证）。
- `verify-app-bundle.sh` 通过（`checks/verify-app-bundle.txt`）。

**真实模型实测（`results/bundle-check-v1`）：** 用包内 helper 和包内 metallib，App 默认参数，App 环境。
- 状态：M5 profile 为 active，指纹带 `scheduler-b1-chunk-v2-cold-single`。
- 5 个请求全部有效，输出都与 `scenarios-main-v1` 的 C-on 相同：
  - 短 code-1、knowledge-1 零命中，各保存 1 次；
  - 两者的完全重复命中完整 prompt（62/46）；
  - 30K code-1 零命中保存 2 次。
- 单次 TTFT（描述性）：短零命中 95/107 ms，重复 14/25 ms，30K 零命中 50.4 s。

## 失败与过程记录（保留）

- `results/scenarios-main-v1`：每个会话最后一条 token-id 记录（30K 文档分叉）被截断。服务关闭时诊断还没写完约 150 KB 的一行。对比脚本对这一条改用输出文本哈希，并标明截断；`scenarios.py` 随后加入“等待记录写完再关闭”，回退集起没有再出现。
- `checks/lib-tests-ironmlx-core-failed-no-metallib.txt`：见上。
- 诊断二进制 `binaries/ironmlx-diag-phases-4efae5b0`（基线加阶段计时）、`ironmlx-baseline-diag2-f2fc817d` 和 `ironmlx-candidate-diag2-c40265c9` 仅用于诊断，不参与配对。
  - baseline-diag2 由候选源码临时关闭修复后构建，源码随后逐字节恢复，`*.source.diff` 记录了实际构建的源码。
  - 构建后 cargo 曾因文件时间戳没有重编，`target` 一度留着 baseline-diag2；随后 touch 源文件重建，已核对回到 c40265c9。冻结的副本不受影响。

## 复现

```bash
cd /Users/xin/workspace/perf-dflash2-cache-miss-prefill
MLX_DIR=/Users/xin/.local/mlx cargo build --release -p ironmlx
```

```bash
cd /Users/xin/workspace/perf-dflash2-cache-miss-prefill/benchmarks/dflash2-cache-miss-prefill/scripts
PY=/Users/xin/workspace/b1-rival-benchmark/artifacts/tensorfold/venv/bin/python
$PY diagnose.py --label <新标签> --binary ../binaries/ironmlx-diag-phases-4efae5b0
$PY scenarios.py --label <新标签> --set main --baseline ../binaries/ironmlx-baseline-b9c7b4fb --candidate ../binaries/ironmlx-candidate-f007c386
$PY scenarios.py --label <新标签> --set fallback --baseline ../binaries/ironmlx-baseline-b9c7b4fb --candidate ../binaries/ironmlx-candidate-f007c386
$PY run_timing.py --label <新标签> --baseline ../binaries/ironmlx-baseline-b9c7b4fb --candidate ../binaries/ironmlx-candidate-f007c386
python3 analyze_timing.py <新标签> --output ../evidence/<新标签>-analysis.json
python3 compare_scenarios.py <新标签> --pairs C-on:B-off C-on:C-off C-off:B-off C-on:B-on --output ../evidence/<新标签>-compare.json
$PY bundle_check.py --label <新标签>
$PY scenarios.py --label <新标签> --set docfork --baseline ../binaries/ironmlx-baseline-b9c7b4fb --candidate ../binaries/ironmlx-candidate-f007c386
$PY diagnose_30k_hit.py --label <新标签> --baseline-diag ../binaries/ironmlx-baseline-diag2-f2fc817d --candidate-diag ../binaries/ironmlx-candidate-diag2-c40265c9
$PY run_r1.py --label <新标签> --baseline ../binaries/ironmlx-baseline-b9c7b4fb --candidate ../binaries/ironmlx-candidate-f007c386
python3 analyze_r1.py <新标签> --output ../evidence/<新标签>-analysis.json
```

端口 18490 必须空闲；各脚本只清理自己启动的进程组。

**证据保存位置：** 协议、脚本、报告、摘要及三个补充实验的原始 `run.json` 保留在 Git。完整服务/过程日志、诊断检查日志和补录的 token-id 转储单独归档，逐 token 复核前需按 [保存与恢复说明](artifact-storage.md) 恢复。
- `MANIFEST.sha256` 只校验 Git 内材料；`ARCHIVED_ARTIFACTS.sha256` 记录移出的原始证据；`LOCAL_ARTIFACTS.sha256` 记录此前已忽略、仍在本地的五个实验二进制、footprint 采样和 probe 动态库。二进制另有身份或运行记录，源码补丁保持不变。
- 重跑实验需要对应版本的二进制、模型与 MLX 环境。当前源码的构建不等于恢复历史冻结二进制；复现历史对照时需使用对应源码身份和构建环境，重新构建的文件应记录新哈希。
- probe 动态库由 `tools/footprint_probe.c` 编译：`clang -dynamiclib -O2 -o tools/footprint-probe.dylib tools/footprint_probe.c`。

## 剩余问题与需要的决策

1. **30K 命中 TTFT 判据未满足**（证据不足）。
   - 已定位到物化一步的双峰，这一现象两臂都有，根本原因未查明；两臂都有慢峰不代表候选不会提高慢峰概率。
   - 紧跟 30K 零命中后的第一次重复（R1）：原计时未观察到频率差异，新诊断中候选偏慢。不同二进制的数据不合并，无法据此判断。

   需要 Boss 决定以下之一：
   - (a) 按现有证据接受这一项；
   - (b) 授权一次事先登记、样本量足够的专项验证。例如只测 R1，12 轮以上交错，判据与原判据相同，并事先写明只运行一次；
   - (c) 把命中物化的双峰作为独立问题另立任务；它在两臂都出现，但是否与本次改动有关尚未判定。

   Boss 选择了 (b)。专项确认满足（见“R1 专项确认”），但区间宽、根因未明，这些限制仍然存在。
2. **内存下降** 中约 1.4 GiB 的来源未知。
3. **`cached_tokens`** 仍未在 DFlash2 usage 中返回（不在本次范围）。
4. **测量范围。** 只测了 Qwen3.8-27B-4bit + DFlash2 草稿和 App 默认 chunk size（2048）。自定义 chunk size 时，单图 prefill 同样只在零命中时启用。
