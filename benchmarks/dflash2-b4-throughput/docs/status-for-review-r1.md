# DFlash2 B4 吞吐优化轮次 —— 执行情况与诊断汇总（供评审）

写于 2026-10-07。范围：本轮从开始到当前的全部指令、执行、测量、诊断、偏差和待决事项。
- 所有路径相对仓库 worktree `/Users/xin/workspace/perf-dflash2-b4-throughput`。
- `reports/` 下是不进 git 的大文件：二进制、原始结果、采样。
- `benchmarks/dflash2-b4-throughput/` 下是协议、脚本、摘要。

## 0. 当前状态一句话

- **第 1 项（ragged 窗口草稿合批）已保留：** 正式测量 B1 短输入 TTFT 未过门槛，经 Boss 授权的一次专项确认后通过。
- **第 2 项（verify 效率）出了问题：** 2a 已按诊断放弃；2b（ragged 窗口草稿预算降为 3）正式测量中 B1 短输入 TTFT 明确未过，诊断没有复现，也没有找到原因。
- **待决：** 2b 及第 2 项的结论、是否做进一步诊断、何时做整轮验收。
- **未做：** 没有提交、推送、合并；整轮验收和竞品对比都还没做。

## 1. 指令时间线

| 时间 | 指令要点 | 执行情况 |
| --- | --- | --- |
| 本轮开始 | 1. 从 dev 9f9c6580 建分支和 worktree，冻结整轮基线，测量 B4，拆分草稿、verify、commit/rollback、成员变化时缓存重建的耗时。2. 按实测固定有限的优化清单：草稿合批为首要候选；verify 效率和缓存复用按收益取舍；多合批组和长输入 prefill 不优先；不把新发现不断追加进本轮。3. 逐项实施和验收：与上一个已验收版本比较，同时保留整轮初始基线；主看 B4 吞吐和整批完成时间，检查逐请求延迟、B1 短/长输入、token ID 正确性和内存。4. 正式测量前固定配置、负载、样本数、配对顺序和门槛；诊断与正式测量分开；报告 CI95 和失败记录，不事后放宽、不补跑到通过。5. 全部完成后与整轮基线做整体验收、冻结版本，再做四应用竞品对比。另外：每项给出“保留／修改／放弃”；不合并 dev、不推送、不清理证据、不扩大生产默认启用范围；先完成基线与耗时拆分，再确定实现。 | 已完成：步骤 1、2；步骤 3 的第 1 项。第 2 项未完成。步骤 5 未开始。 |
| 第 1 项正式测量后 | Boss 选 (b)：授权一次 B1 短输入 TTFT 专项确认。冻结 A=4fc49e92、B=8d3a42dc；补充协议固定 24 个配对轮次、48 个会话，顺序 `AB BA BA AB AB BA BA AB` ×3；每个会话完整执行原流程；只运行一次，不提前停止、不剔除、不追加；判据为汇总点估计 ≤1.03；原 8 轮仍记未通过；合并 32 轮只作敏感性分析。通过后保留草稿合批，以 B 为基线推进 verify 优化；未通过或无效则保持未验收，不追加确认，不进入 verify 实现。缓存复用本轮不做；B4 之后仍保留长输入优化安排。 | 已执行，专项确认通过，随后开始第 2 项。 |
| 第 2 项诊断后 | Boss 询问诊断与前一指令的关系，确认理解：第 1 项已完成，第 2 项出了问题，这是诊断结果。 | 已回答。本文件为 Boss 要求的完整汇总。 |

## 2. 固定的环境与配置

**机器与模型：**
- M5 Max 128 GB。
- target：`mlx-community--Qwen3.8-27B-4bit`，snapshot 3e6447f0…；draft：`z-lab--Qwen3.8-27B-DFlash2`，snapshot 50307d4c…。

**构建：**
- 命令 `MLX_DIR=/Users/xin/.local/mlx cargo build --release -p ironmlx`。
- MLX：libmlx.a 27f1e5d1…，mlx.metallib adc6967e…，头文件对应 fork 73ad5df2。

**服务配置：**
- 产品默认值（M5 profile 自动启用），`--max-sequences 4`，`--max-cache-cap 8192`；B1 长输入用 40960。
- 不开前缀缓存，端口 18490。
- App 环境：只保留 PATH、HOME 等，`IRONMLX_LOG_LEVEL=warn`。正式测量不开任何诊断。

**工作负载（沿用 b1-b4-concurrency-three-app-v1）：**
- 三个 B4 批次组合，各四个短任务，回答长度约 450–1800 token：
  - 组合 1：code-1、code-2、code-3、knowledge-1；
  - 组合 2：knowledge-2、knowledge-3、code-1、code-2；
  - 组合 3：code-3、knowledge-1、knowledge-2、knowledge-3。
- 一个固定预热批。
- greedy，关闭思考，max_tokens 4096。
- B1 短输入：code-1、knowledge-1、code-3，逐个发送；B1 长输入：30K code-1，max_tokens 1024。

**二进制（`reports/dflash2-b4-throughput/binaries/`，均有 `.build.log`，源码对应 `.source.diff`）：**

| 二进制 | sha256 前 16 位 | 用途 |
| --- | --- | --- |
| ironmlx-round-baseline-4fc49e92 | 4fc49e9223f62468 | 整轮基线，dev 9f9c6580 干净构建（另有 identity.json） |
| ironmlx-round-diag-68969b30 | 68969b30388f23e0 | 基线加步骤诊断（默认关闭），只用于诊断 |
| ironmlx-item1-dev-85489e24 | 85489e2444a266f0 | 第 1 项变体 1（批隔离 QMM），只做了诊断 |
| **ironmlx-item1b-dev-8d3a42dc** | 8d3a42dc8ee05cc8 | **第 1 项候选 1b，已保留** |
| ironmlx-item2-diag-4de81a2c | 4de81a2c7a2cc3c1 | 1b 加 ragged 注意力计时诊断，只用于诊断 |
| ironmlx-item2b-diag-83e22bf5 | 83e22bf55ee1adbe | 1b 加 ragged 预算覆盖诊断，只用于诊断 |
| ironmlx-item2b-dev-f6a40e20 | f6a40e20c5a3bed7 | 2b 候选，正式测量未过 |

## 3. 步骤 1：整轮基线与耗时拆分（诊断，不作验收）

**运行：** `results/diag-plain-v1`、`diag-steplog-v1`、`diag-stages-v1`，每组 2 个会话，汇总在 `evidence/diagnosis-v1-steps.json`。

**诊断工具：** 新增默认关闭的步骤日志 `IRONMLX_DIAGNOSTIC_DFLASH2_STEP_LOG`，记录每个调度步各阶段的墙钟时间。屏障模式下另开已有的 `IRONMLX_DIAGNOSTIC_DFLASH2_WINDOW_STAGES=1`，并在 scatter 后加同步。三组的吞吐中位数分别为 117.2、121.1、116.0 tok/s，诊断开销在会话间波动范围内。

**B4 一批的构成：**

| 部分 | 占比 / 数值 |
| --- | --- |
| ragged 线性合批窗口 | 69–94% |
| 长尾单行树窗口（只剩一个请求） | 组合 1、2 为 24–29%，组合 3 约 4% |
| 成员变化时缓存重建（每批新建组 3 次，每次 13–35 ms）与 scatter（每次 5–20 ms） | 合计不到 0.5% |
| 主机侧（取 token、发布、key、选行） | 约 1.5% |

**窗口内部（屏障模式，单窗口中位数）：**

| 窗口 | 草稿 | verify | head | commit | 每窗口产出 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ragged 2 行 | 17.2 ms | 42.2 ms | 1.7 ms | 2.3 ms | 8 |
| ragged 3 行 | 24.7 ms | 66.6 ms | 2.7 ms | 2.9 ms | 10 |
| ragged 4 行 | 32.2 ms | 66.4 ms | 2.7 ms | 3.6 ms | 14 |
| 单行树窗口 | 11.9 ms | 44.6 ms | 1.7 ms | 4.1 ms | 4 |

ragged 窗口的草稿是逐行生成的，每行约 8 ms，开销随行数线性增长。

## 4. 本轮优化清单（由上述实测固定）

| 编号 | 候选 | 依据 | 当前结论 |
| --- | --- | --- | --- |
| 1 | ragged 窗口草稿合批 | 4 行窗口中草稿占 31%，开销随行数线性增长 | **保留** |
| 2 | ragged verify 效率 | 占 ragged 窗口的 63–68% | 2a 放弃；**2b 待决** |
| 3 | 成员变化时缓存重建与复用 | 不到整批的 0.5% | **放弃**（Boss 也指示本轮不做缓存复用） |
| — | commit/rollback | 约 3% | 不列入 |
| — | 长尾单行树窗口 | 24–29%，属于单请求解码效率 | 不列入，仅记录 |

## 5. 正式验收协议（`docs/protocol.md`，第 1 项正式测量前写定）

**对照：** A 为上一个已验收版本，B 为候选。

**会话：**
- B4 会话：8 个配对轮次，顺序 AB BA BA AB AB BA BA AB。每个会话都是新进程：预热批，三个批次组合（起始位置按 round%3 轮换），然后在同一进程里依次发三个 B1 短任务。会话间隔 10 s，批次间隔 2 s，B1 请求间隔 1 s。
- B1 长输入会话：4 个配对轮次，顺序 AB BA BA AB，容量 40960，先发一个短预热，再发 30K code-1。

**统计：** 每个组合或任务取各轮中位数之比，再取几何平均；区间用配对轮次 bootstrap（10000 次，种子 20261006）。

**门槛：**

| 指标 | 条件 |
| --- | --- |
| B4 吞吐 | ≥1.03 且下界 >1 |
| B4 整批完成时间 | ≤0.97 且上界 <1 |
| B4 逐请求 TTFT | ≤1.05 |
| B1 短输入 Decode | 下界 ≥0.97 |
| B1 短输入 TTFT、E2E | ≤1.03 |
| B1 长输入 TTFT | ≤1.03 |
| 内存峰值 | ≤1.02 |

**正确性：** 单独会话，开 token-id 诊断，B4 三个组合、预热批、B1 短输入、30K，共 21 个请求，要求 token ID 和结束原因全部一致。

**判定：** 全部满足为“保留”；B4 吞吐改善但某项非退化未过为“修改”，修改后按同一协议重测；吞吐门槛未过为“放弃”。

## 6. 第 1 项：ragged 窗口草稿合批（保留）

**实现：**
- 草稿模型新增 `propose_greedy_ragged_on`：投影、MLP、lm_head 和选择器对所有行展平计算，权重只读一遍；每行的上下文投影、RoPE、草稿缓存和注意力仍走单行路径。
- 修改的文件：`ironmlx-lm/src/models/dflash2/{attention,layer,model}.rs` 和 `ironmlx-runtime/src/core/dflash2.rs`。

**变体诊断（`results/item1-*`、`item1b-*`）：**

| 变体 | 4 行草稿耗时 | 与逐行草稿是否逐位一致 | 吞吐中位数（诊断） | 处理 |
| --- | ---: | --- | ---: | --- |
| 基线 | 32.2 ms | — | 117.2 | — |
| 1：批隔离 QMM（`quantized_matmul_batch_isolated`，按批次分别读权重） | 26.9 ms | 一致，1056 个窗口 0 差异 | 119.3 | 放弃，收益太小 |
| 1b：展平 QMM | 16.3 ms | 1042 个窗口中 229 个有差异 | 124.8 | 进入正式测量 |

1b 的草稿接受率从 0.371 降到 0.368，输出文本不变。

**正确性（`results/item1-correctness-v1`）：** 21/21 一致；接受率 A 0.318，B 0.317。

**正式测量（`results/item1-formal-v1`，A=4fc49e92，B=8d3a42dc，24 个会话全部有效）：**

| 指标 | B/A [95% 区间] | 结果 |
| --- | --- | --- |
| B4 吞吐 | 1.105 [1.088, 1.114] | 满足 |
| B4 整批完成时间 | 0.905 [0.898, 0.919] | 满足 |
| B4 逐请求 TTFT | 0.937 [0.842, 1.121] | 满足 |
| B1 短输入 Decode / E2E | 1.000 / 1.000 | 满足 |
| **B1 短输入 TTFT** | **1.039 [0.937, 1.200]** | **未满足**：code-3 中位数 146.9 → 177.7 ms，有两次 252 ms；code-1 相同；knowledge-1 B 更快 |
| B1 长输入 TTFT | 1.002 | 满足 |
| 内存 | 1.004 | 满足 |

**诊断（`results/diag-b1-ttft-v1`，A=4fc49e92，B=8d3a42dc）：**
- 3 轮，两种会话（先跑 B4 后发 B1；不跑 B4 直接发 B1），每种每臂各 27 次请求。
- code-3：先跑 B4 时 147.9 对 148.1 ms，不跑 B4 时 152.8 对 154.5 ms。
- prefill 以外的耗时两臂都约 2 ms；请求前 MLX 内存相同。
- 正式测量中的偏差没有复现。

**专项确认（`docs/protocol-b1-ttft-confirmation.md`，`results/b1-ttft-confirmation-v1`，48 个会话全部有效，只运行一次）：**

| 项目 | 新 24 轮 | 原 8 轮（保留，未通过） | 合并 32 轮（敏感性分析） |
| --- | --- | --- | --- |
| B1 短输入 TTFT 汇总 | **0.986 [0.950, 1.043]，通过** | 1.039 | 0.999 [0.961, 1.054] |
| code-3 | 0.997 [0.945, 1.161]；配对差值中位数 −0.4 ms；B 更慢 12/24 轮 | 146.9 → 177.7 ms | 1.057 [0.978, 1.190] |
| 参考：B4 吞吐 / 整批完成时间 | 1.114 / 0.898 | — | — |

点估计通过，但区间上界 1.043 超过 1.03，不能在统计上排除 3% 的退化。

**检查（`benchmarks/dflash2-b4-throughput/checks/item1/`，对应 8d3a42dc 的源码）：**
- nightly fmt、strict clippy、release 构建全部通过。clippy 首次在诊断代码中报出 `manual_is_multiple_of`，失败日志保留；修正后二进制哈希不变。
- ironmlx-lm 615 项、ironmlx-runtime 543 项 lib 测试通过。

## 7. 第 2 项：verify 效率

### 2a 合并 ragged verify 的逐行注意力：放弃

**诊断（`results/item2-attn-stages-v1`，二进制 4de81a2c，屏障模式）：** 每层逐行注意力块前后各加一次同步计时。

| 窗口 | verify | 逐行注意力 |
| --- | ---: | ---: |
| 2 行 | 50.3 ms | 6.2 ms |
| 3 行 | 76.0 ms | 6.5 ms |
| 4 行 | 66.1 ms | 5.9 ms |

- 逐行注意力合计只占整批时间的 6.0%（`evidence/diagnosis-summaries-v1.json`）。
- 这个数含每层两次同步的开销，且几乎不随行数增长，所以实际可省的空间更小，估计 1–2%，低于 3% 的门槛。

### 2b ragged 窗口减小草稿预算：正式测量未过，待决

**依据：** item1b 屏障诊断的接受长度分布，见 `evidence/diagnosis-summaries-v1.json`。

| 窗口 | 接受 0 个 | 接受满 7 个 | 每行期望产出（预算 7 / 4 / 3） |
| --- | ---: | ---: | --- |
| 2 行 | 25% | 22% | 3.94 / 3.14 / 2.78 |
| 3 行 | 29% | 14% | 3.31 / 2.78 / 2.51 |
| 4 行 | 28% | 16% | 3.56 / 2.94 / 2.62 |

**归类说明：** 2b 是我把它归到“verify 效率”下的子方向，理由是它减少被 verify 掉的无效 token。如果评审认为这属于追加的新发现、超出了固定清单，它应直接放弃。

**预算诊断（`results/item2b-budget-{none,k3,k4,k5}-v1`，二进制 83e22bf5，每组 2 个会话）：**

| 预算 | 吞吐中位数 | 整批完成中位数 |
| --- | ---: | ---: |
| 不覆盖（7） | 122.7 | 32.51 s |
| 3 | 125.8 | 31.46 s |
| 4 | 119.2 | 33.61 s |
| 5 | 116.6 | 34.11 s |

- **缺陷：** 四组依次运行，没有交错。对照组两个会话相差约 20%（147.8 / 150.0 对 119.5 / 122.2），说明存在明显的时间漂移。这组数据只用来选候选。

**实现：**
- `RAGGED_LINEAR_DRAFT_BUDGET = 3`：ragged 窗口的预算为 min(3, 行预算)，单行窗口不变。
- 保留诊断覆盖开关 `IRONMLX_DIAGNOSTIC_DFLASH2_RAGGED_BUDGET`。
- 候选 f6a40e20 同时包含默认关闭的 2a 注意力计时诊断。
- 协议补充写在 `docs/protocol.md` 的“第 2 项”一节，在正式运行前、预算诊断之后写定。

**正确性（`results/item2b-correctness-v1`）：** 21/21 一致；草稿数 35969 → 25284，接受率 0.317 → 0.414。

**正式测量（`results/item2b-formal-v1`，A=8d3a42dc，B=f6a40e20，24 个会话全部有效，分析在 `evidence/item2b-formal-v1-analysis.json`）：**

| 指标 | B/A [95% 区间] | 结果 |
| --- | --- | --- |
| B4 吞吐 | 1.032 [1.025, 1.037] | 满足（贴近门槛） |
| B4 整批完成时间 | 0.969 [0.964, 0.976] | 满足（贴近门槛） |
| B4 逐请求 TTFT | 0.993 | 满足 |
| B4 逐请求 E2E（报告） | 0.947 | — |
| B1 短输入 Decode / E2E | 1.006 / 0.998 | 满足 |
| **B1 短输入 TTFT** | **1.205 [1.058, 1.285]** | **未满足**（区间整体高于 1） |
| B1 长输入 TTFT | 1.003 | 满足 |
| 内存 | 1.017 | 满足 |

**正式测量中 B1 短输入 TTFT 的原始值（各 8 轮）：**

| 任务（会话内顺序） | A 中位数 | B 中位数 | B 各轮值 |
| --- | ---: | ---: | --- |
| code-1（B4 后第一个） | 143.9 | 144.6 | 无差异 |
| knowledge-1（第二个） | 108.6 | 135.6 | 154.0, 135.5, 137.9, 117.9, 135.7, 134.5, 184.4, 107.3 |
| code-3（第三个） | 160.1 | 223.2 | 264.1, 187.9, 257.4, 259.7, 177.9, 229.8, 216.5, 157.1 |

**诊断（`results/diag-b1-ttft-item2b-v1`，A=8d3a42dc，B=f6a40e20）：** 方法与 `diag-b1-ttft-v1` 相同，12 个会话全部有效。

| 条件 / 任务 | A 中位数 | B 中位数 |
| --- | ---: | ---: |
| 先跑 B4 / code-1 | 122.6 | 123.9 |
| 先跑 B4 / knowledge-1 | 136.6 | 120.3 |
| 先跑 B4 / code-3 | 154.4 | 160.0 |
| 不跑 B4 / code-1 | 116.0 | 128.6 |
| 不跑 B4 / knowledge-1 | 112.2 | 106.3 |
| 不跑 B4 / code-3 | 166.3 | 173.7 |

- prefill 以外两臂都约 1.8 ms，诊断中 B 的最大值为 210 ms。
- 正式测量中的系统性偏慢没有复现，原因不明。
- 诊断与正式流程的唯一已知差别：诊断在每个 B1 请求前后各调用一次 `/healthz`。
- 按协议，第 2 项的正式结论（B1 短输入 TTFT 未过）不因诊断而改变。没有可修改的机制，就无法走“修改后重测”的路径。

## 8. 跨项观察（未解释）

1. **同样的模式出现了两次。** 第 1 项和 2b 的正式测量中，都是 B 臂的 B1 短输入 TTFT 偏慢，且集中在会话内第二、第三个 B1 请求（第 1 项是 code-3，2b 是 knowledge-1 和 code-3）。随后的诊断，以及第 1 项那次 24 轮、流程与正式测量完全相同的专项确认，都没有复现。
2. **单次短输入 prefill 波动很大。** 两臂都在 100–265 ms 之间，偏慢全部发生在服务端 prefill 阶段。
3. **两种解释无法区分。** 这可能是同一个尚未识别的系统或测量因素，也可能是两件不相干的事；现有证据不足以判断。

## 9. 过程中的偏差与失误（如实记录）

- **监视器自匹配。** 多次用 `pgrep -f` 或 `ps | grep` 监视后台运行时，模式与监视命令自身匹配，导致监视器不退出或不报进度。只影响通知，不影响运行，之后改用 `[r]un_...` 写法。
- **命令链中断。** 冻结 2a 诊断二进制时，`git diff --no-index` 返回码为 1，中断了 `&&` 链，诊断第一次没有启动；文件已正确写出，随后手动重新启动。
- **2b 预算诊断没有交错。** 四组依次运行，存在时间漂移，只能用来选候选。
- **候选包含诊断代码。** 2b 候选 f6a40e20 带有 2a 的注意力计时诊断（默认关闭），A 和 B 之间的差异不只有预算。
- **2b 的归类。** 把 2b 归入第 2 项“verify 效率”是我的判断，见第 7 节说明。
- **AGENTS.md 检查不完整。** 第 2 项新增的代码（2a 注意力计时诊断、2b 预算常量和覆盖开关）只做了 `cargo fmt` 和 `-p ironmlx` release 构建，**还没有对当前源码重跑 strict clippy、workspace 构建和 lib 测试**。第 1 项的检查对应的是 8d3a42dc 的源码。

## 10. 当前代码状态（未提交）

基于 9f9c6580，共 7 个修改文件和 1 个新文件，+516 / −26：
- `ironmlx-lm/src/models/dflash2/{attention,layer,model}.rs`：第 1 项的 ragged 合批草稿（`propose_greedy_ragged_on`、`forward_ragged_on`）。
- `ironmlx-runtime/src/core/dflash2.rs`：
  - ragged 窗口改用合批草稿（第 1 项）；
  - 草稿对照诊断 `IRONMLX_DIAGNOSTIC_DFLASH2_RAGGED_DRAFT_CHECK`；
  - scatter 屏障诊断；
  - `RAGGED_LINEAR_DRAFT_BUDGET = 3` 和预算覆盖诊断（2b）；
  - `StageClock::note`。
- `ironmlx-runtime/src/core/dflash2_actor.rs`、`dflash2_step_diagnostic.rs`（新）、`mod.rs`：步骤日志诊断。
- `ironmlx-lm/src/nn/gated_attention.rs`：ragged 注意力计时诊断（2a）。

所有诊断默认关闭。如果 2b 放弃，`dflash2.rs` 中的预算常量需要回退，诊断代码的取舍待定。

## 11. 待决问题与我的建议

1. **2b 的结论。**
   - 建议：**放弃**。B4 收益只有 3.2%，贴近门槛；B1 短输入 TTFT 在正式测量中明确退化，原因不明，又找不到可修改的机制，无法按协议“修改后重测”。
   - 不建议追加专项确认：与“不追加确认、不补跑到通过”的原则冲突。
2. **第 2 项整体。** 若 2b 放弃，第 2 项整体放弃（2a 已放弃）。本轮计划内保留的优化只有第 1 项。
3. **是否先查清第 8 节的现象（可选）。**
   - 做法：按正式流程（不调用 `/healthz`）复现，在服务端记录 prefill 各阶段的耗时。
   - 性质：只作诊断，不用于验收。
   - 理由：这个现象两次影响了验收判断，查清后能提高之后测量的可信度。
4. **整轮验收。** 以第 1 项版本（如回退 2b 后的源码）对整轮基线 4fc49e92，按 `docs/protocol.md` 做整体验收；之前需补齐当前源码的 AGENTS.md 检查。之后冻结版本，再做四应用竞品对比。B4 之后还保留长输入优化的安排。

## 12. 文件索引

| 内容 | 路径 |
| --- | --- |
| 协议 | `benchmarks/dflash2-b4-throughput/docs/protocol.md`、`protocol-b1-ttft-confirmation.md` |
| 结果报告（逐项） | `benchmarks/dflash2-b4-throughput/docs/results.md`（第 2 项的正式结果见本文件） |
| 分析摘要 | `benchmarks/dflash2-b4-throughput/evidence/`：`diagnosis-v1-steps.json`、`item1-formal-v1-analysis.json`、`b1-ttft-confirmation-v1-analysis.json`、`item2b-formal-v1-analysis.json`、`diagnosis-summaries-v1.json` |
| 检查日志 | `benchmarks/dflash2-b4-throughput/checks/item1/` |
| 脚本 | `benchmarks/dflash2-b4-throughput/scripts/`：common、diagnose_b4、diagnose_b1、analyze_steps、run_formal、analyze_formal、run_correctness、run_b1_confirmation、analyze_b1_confirmation |
| 原始结果 | `reports/dflash2-b4-throughput/results/<label>/`：run.json、服务日志、footprint、步骤日志、token-id 记录 |
| 二进制与源码 | `reports/dflash2-b4-throughput/binaries/` |
