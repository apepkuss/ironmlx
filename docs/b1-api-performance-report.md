# IronMLX B1 API 性能优化对比报告

本报告针对同机 Qwen3.8-27B-4bit + DFlash2 的短 prompt、单请求串行 API
推理。首批 8 轮及追加 4 轮合并后的 12 轮复测均通过竞争性能门槛：IronMLX 三项指标均优于 oMLX，
Decode TPS 和 E2E 优于 TensorFold，但 TTFT 仍落后于 TensorFold。
**本阶段目标已达成，范围限于本报告固定的 B1 API 工作负载及实验配置。**
2026-09-30 用户允许披露模型已有缺陷，并按相对基线质量不退化验收；逐题对照及
代码功能复核后，该相对质量门槛通过。知识回答仍存在事实错误和字数超限，
不能将本结论理解为回答全部正确或模型整体能力资格通过。
没有提交、合并、发布或更改全局默认执行路径。

## 性能结果

以下为每应用 72 个请求的合并中位数，共 288 个正式请求，供直观阅读；正式判定使用下文的逐题
等权比值和置信区间，不能直接用这张表的比值替代。

| 应用 | TTFT 秒 | Decode tok/s | E2E 秒 |
|---|---:|---:|---:|
| 优化后 IronMLX | 0.1689 | 78.67 | 10.88 |
| oMLX 0.7.0rc1 | 0.2596 | 72.42 | 13.27 |
| Splash 1.1.0 | 0.2334 | 66.24 | 12.54 |
| TensorFold 0.3.6.2 | 0.1189 | 74.52 | 12.29 |

正式统计量是六道题的 IronMLX/对手中位数比值的几何平均。TTFT、E2E
比值低于 1 有利；TPS 高于 1 有利。以下区间是配对 session block bootstrap
的 95% 区间，10,000 次重采样，固定种子 20260929。

| 对手 | 指标 | IronMLX/对手 | 95% 区间 | 判定 |
|---|---|---:|---|---|
| oMLX | TTFT | 0.7038 | 0.5260–0.8018 | 优于 |
| oMLX | Decode TPS | 1.1812 | 1.1650–1.1950 | 优于 |
| oMLX | E2E | 0.8124 | 0.8021–0.8243 | 优于 |
| TensorFold | TTFT | 1.5654 | 1.1686–1.7828 | 落后 |
| TensorFold | Decode TPS | 1.0931 | 1.0878–1.1097 | 优于 |
| TensorFold | E2E | 0.8531 | 0.8394–0.8577 | 优于 |
| Splash | TTFT | 0.8023 | 0.5977–0.9118 | 优于 |
| Splash | Decode TPS | 1.1906 | 1.1786–1.2029 | 优于 |
| Splash | E2E | 0.8302 | 0.8239–0.8368 | 优于 |

共同 tokenizer 重新计数的 TPS 比值为 oMLX 1.1795、TensorFold 1.0929，
相应区间仍全部高于 1。不同服务是否计入 EOS 不改变该结论。
预先规定的代码与知识两个类别的退化检查均通过。结果不是“每题都胜出”：
例如 knowledge-1 的 IronMLX TTFT 为 0.3595 秒，oMLX 为 0.2552 秒；
knowledge-2 的 IronMLX TPS 为 73.45，oMLX 为 73.92、TensorFold 为 73.52，
该题仍略低。逐题数据完整保留。

“优于”要求整个区间位于 1 的有利一侧；“不弱于”要求点估计有利且区间
排除超过 3% 的退化，并检查每个类别点估计不得退化超过 3%。后者是明确的
3% 非劣界限，不是证明严格相等。本次两项 TensorFold 胜项达到更严格的
“优于”标准。统计区间只反映这组固定题和本机 session 变动，不代表所有任务、
设备或长上下文的总体置信度，也未做多重比较校正。

## 补测敏感性

以下均为逐题等权几何比值，不是合并中位数的比值。补测四轮只检查趋势，
不单独作为正式验收；未因补测结果更快而舍弃原始数据。

| 对手 | 指标 | 首批八轮 | 补测四轮 | 全部十二轮 |
|---|---|---:|---:|---:|
| oMLX | TTFT | 0.7049 | 0.6698 | 0.7038 |
| oMLX | TPS | 1.1843 | 1.1662 | 1.1812 |
| oMLX | E2E | 0.8115 | 0.8224 | 0.8124 |
| TensorFold | TTFT | 1.5680 | 1.4915 | 1.5654 |
| TensorFold | TPS | 1.0978 | 1.0893 | 1.0931 |
| TensorFold | E2E | 0.8504 | 0.8566 | 0.8531 |

## 十二轮逐题与分类结果

每格依次为 TTFT 秒 / Decode tok/s / E2E 秒，均取该题十二次中位数。

| 题目 | IronMLX | oMLX | Splash | TensorFold |
|---|---|---|---|---|
| code-1 | 0.1341/113.56/6.17 | 0.2683/86.13/8.21 | 0.2309/92.14/7.69 | 0.1152/91.14/7.98 |
| code-2 | 0.1546/106.22/13.34 | 0.2624/86.77/16.49 | 0.2306/87.85/16.40 | 0.1137/99.33/14.54 |
| code-3 | 0.2010/79.65/5.80 | 0.3293/66.48/7.24 | 0.2541/68.09/7.20 | 0.1459/74.91/7.05 |
| knowledge-1 | 0.3595/59.48/24.19 | 0.2552/48.57/30.65 | 0.2331/49.45/29.20 | 0.1203/54.06/30.53 |
| knowledge-2 | 0.1800/73.45/22.26 | 0.2527/73.92/25.22 | 0.2338/62.88/28.68 | 0.1197/73.52/24.48 |
| knowledge-3 | 0.1690/47.85/9.50 | 0.2510/41.45/11.23 | 0.2318/41.15/9.83 | 0.1125/43.67/10.37 |

类别几何均值比值 IronMLX/对手：

| 对手 | 类别 | TTFT | TPS | E2E |
|---|---|---:|---:|---:|
| omlx | code | 0.5644 | 1.2459 | 0.7872 |
| omlx | knowledge | 0.8775 | 1.1200 | 0.8385 |
| splash | code | 0.6754 | 1.2035 | 0.8076 |
| splash | knowledge | 0.9531 | 1.1778 | 0.8534 |
| tensorfold | code | 1.2969 | 1.1231 | 0.8359 |
| tensorfold | knowledge | 1.8896 | 1.0639 | 0.8707 |

## 协议和环境

协议在实现前冻结，见 [测试协议](b1-api-performance-protocol.md)。三道代码题
分别为合并闭区间、TypeScript LRU、并发数不超过 4 且保持顺序的异步执行；
三道知识题为事务隔离、HTTP 缓存、多位宽量化。所有应用输入完全相同，
每次请求 temperature=0、top_p=1、关闭 thinking、自然 EOS；4096 token
仅作安全上限。没有修改 prompt、重写答案、裁剪回答或强迫提前停止。

每个新服务进程先做两次不相关预热，再按轮次轮换题序；四应用按平衡顺序交错。
请求间隔 1 秒，应用间隔 10 秒。模型装载和权重准备不计入请求延迟；保留磁盘
prepared weights 和编译缓存，关闭可配置的 prompt/prefix 缓存。最终客户端
通过实际服务的 `/v1/chat/completions` 流式接口计时，不是内部 decode benchmark。
本阶段不覆盖 DSH Desktop，也没有更改 IronMLX 的 DSH Responses 集成。

设备是接交流电的 M5 Max、40 GPU 核、128 GiB 内存，报告 Metal 4 支持。
首批 32 次应用 session 的热状态快照没有记录系统热警告，但这不等于测得芯片
温度稳定，也不能证明没有降频。部分进程快照出现 Chrome CPU 突增，最高约
135%；因此保留首批全部数据，追加一个完整四应用平衡周期，不选择性删除慢样本。
补测仍捕捉到 Chrome 最高约 109% 的瞬时 CPU 负载，不能声称完全排除了后台
干扰；补测单独趋势和全部 12 轮区间均支持原结论。首批测量开始时间范围为
2026-09-29 14:30–15:35 UTC，补测从 2026-09-30 01:26 UTC 开始，属于
同机跨时段复测，不伪装成完全恒温、无其他桌面进程的实验室测量。

目标 snapshot 为 `3e6447f082e89cc7f0bc6e5441afd38dfce760ff`，drafter 为
`50307d4c4cde6860d4eee73e2547cd786fe8e8a4`，均使用本机共享权重。
模型 metadata、路径和 shard 大小已记录；未重新计算所有大型权重 shard 的
全文件校验和。Splash 的 prepared manifest 指向相同源 checkpoint。

## 生效配置和实现

| 应用 | 核心设置 |
|---|---|
| IronMLX | Release；BF16 KV；Q8 checkpoint；固定 draft budget 7；显式 15 节点 tree；M5 affine4/group64 lane；原生 tree attention；single prefill；radix top-k；attention group tiles 4 |
| oMLX | 发布包 CLI；B1；aggressive burst；关闭 cache 和 prefill chunk；DFlash2 affine4/group64 与其自适应策略 |
| Splash | 1.1.0 发布内容中的服务与 engine；相同 checkpoint 的 prepared weights；默认 INT8 KV；无磁盘 prompt cache；reasoning none |
| TensorFold | 固定 0.3.6.2 源码及环境；lane kernels 启用；parallel 1；prompt cache 0；无 snapshot reuse |

服务上下文配置为 8192。逐 session 的确切参数、版本、运行时、环境变量和
已脱敏配置保存在 `.command.json`，不能只凭上表重新猜测命令。
IronMLX 使用本机 MLX C++ 0.32.2 的静态库和对应 metallib。

主要优化不是单独调整线性深度阈值，而是组合了以下工作：

- 改编 TensorFold 的 M5 affine4 lane，准备专用 packed weights，并合并相关
  projection；只对匹配硬件、模型形状、BF16/affine4/group64 的路径启用。
- 唯一节点 tree 执行和 parent-indexed GDN；原生 tree attention 共享已提交
  prefix，只在 kernel 内处理各路径 tail，避免重复完整 prefix gather。
- 精确 BF16 radix top-k 替换本机 MLX 的全排序 argpartition；保留完整词表，
  处理全部并列值，不采用 TensorFold 的额外 draft 词表限制。
- B1 single prefill、selector casts 合并求值，以及固定四组 attention 调度。
  最后两项一起实测，没有独立消融，不能分配各自贡献。

TensorFold 的 MIT 声明保存在 `ironmlx-lm/src/nn/LICENSE.tensorfold`。
Apple 的 [Metal 4 inline ML 文档](https://developer.apple.com/documentation/metal/running-inline-ml-operations-in-a-shader-with-metal-4)
提供 tensor 操作能力依据，但具体收益来自本机实测，不从硬件说明推算。
没有声称已完成 allocator 的独立优化，也没有把 CPU 等待栈当作 GPU kernel
耗时。固定深度、lane、tree、同步变更共同影响结果，不能把全部收益归因于某一项。
失败和无收益方案详见 [实验记录](b1-api-performance-experiments.md)。

当前全部控制为 opt-in，普通执行路径保留。M5 以外的设备没有实机资格结果；
M6 不因架构编号符合而自动获得“实测合格”的声明。该原型保留普通和 prepared
权重两份，约 32 GiB 进程占用是需要继续优化的成本，不代表最低内存实现。

## 正确性与已接受的限制

所有六题的候选串行执行与 15 节点 tree 完整 token 序列逐个相等，直到自然 EOS。
这证明了这批输入上的 speculative/serial 一致性，不证明新 kernel 与普通 MLX
逐位相同。新 attention 有意采用不同归约与乘积精度；与普通 MLX 的边界夹具
最大绝对误差为 0.001953125，serial/verify 和 branching 边界测试通过。

本地回归通过 LM 613 项、runtime 537 项、CLI library 326 项；分别还有
14、7、2 项忽略，不能计作通过。另有 9 项聚焦 M5 kernel 测试及 top-k CPU
reference 检查。首次 library 回归因找不到 metallib 失败，日志保留；修正
测试运行环境后重跑通过。这不是 CI、其他硬件、长上下文或完整稳定性资格声明。

全部 288 个正式请求均自然停止、无 API 错误或 thinking 泄漏。同应用同题的
12 次输出 hash 一致，首个内容事件均重新编码为 1 token。12 份不同应用生成代码
及原始 IronMLX 基线的 3 份代码均通过同一功能测试；不等于 TypeScript
静态类型检查、取消行为或所有文字说明均正确。

知识回答包含数据库隔离事实错误和长度约束违例；原始 IronMLX 基线也有这些
缺陷。用户已明确允许保留并披露这些限制，采用相对质量不退化口径。
[完整质量审查](b1-api-performance-quality.md)记录逐题人工判断及不同错误的变化：
代码功能与要求覆盖未退化，未发现整体任务质量严重度增加，但不声称新旧错误
逐条相同、每句话都更准确，或通过独立的大样本模型能力评测。
答案长度不同确实影响 E2E；TPS 的独立提升成立，E2E 则是自然停止条件下
完成同类要求的实际耗时，不能把其全部改善解释为等 token 数的计算加速。

共同 tokenizer 的自然停止输出长度如下；IronMLX/对手的逐题等权长度比值
为 oMLX 0.9623、Splash 0.9896、TensorFold 0.9257。没有以长度更短本身
作为优化成功依据，也没有假设跨应用的回答逐字相同。

| 题目 | IronMLX | oMLX | Splash | TensorFold |
|---|---:|---:|---:|---:|
| code-1 | 685 | 685 | 687 | 717 |
| code-2 | 1393 | 1409 | 1420 | 1425 |
| code-3 | 447 | 460 | 473 | 517 |
| knowledge-1 | 1418 | 1477 | 1432 | 1644 |
| knowledge-2 | 1622 | 1847 | 1789 | 1791 |
| knowledge-3 | 447 | 456 | 395 | 448 |

## 复现和证据位置

工作目录为 `/Users/xin/workspace/b1-api-performance/ironmlx-backend`，分支
`perf/b1-api-performance`，基点 `a2aec98d887f405e4d3faf5827321441f9cf7987`。
原始数据位于该工作目录的 `reports/b1-api-performance/`，保留失败、终止和
无结论实验。正式数据前缀为 `final-candidate-v1`，每个应用、每轮有独立 JSON。
`final-candidate-v1-analysis.json` 是首批 8 轮结果，不被补测覆盖。
`final-candidate-v1-analysis-twelve.json` 是全部 12 轮统计，
`final-candidate-v1-sensitivity.json` 单独记录补测四轮；四轮不足以独立替代
冻结协议中的八轮最低样本数，因此只作敏感性检查。原始八轮、补测四轮以及
全部十二轮的结论方向一致：胜 oMLX 三项、胜 TensorFold TPS/E2E，TTFT 落后。

`candidate-source-v1/manifest.json` 和 `sources/`、`tracked.patch` 在正式测试前
记录了基点、完整修改源码和 hashes。测量的 IronMLX 二进制 SHA256：
`11ab79a9cce2575d1bb082b135c02bf8165fdb9e19fb75e63bc7a7971ffc61ee`。
客户端 SHA256：`8b4f54e37d18cf616c16ac1b7ee313234082e07c2104028790742ddacfabd606`。

按协议中的候选环境变量启动 `scripts/run_b1_api_sessions.py`，使用新的 label
以免覆盖既有证据；先确认端口 18480 空闲、没有其他推理或编译任务。

构建使用已记录的本机 MLX 安装，不能将另一版 MLX 与这次二进制混为同一候选。
恢复快照源码后，在上述工作目录执行以下命令，再校验构建产物及配置；构建
必须在测量之前完成。

```sh
source /Users/xin/.local/mlx/mlx-env.sh
cargo build --release -p ironmlx --bin ironmlx
```

分析独立复测的命令如下：

```sh
/Users/xin/workspace/b1-rival-benchmark/artifacts/tensorfold/venv/bin/python \
  scripts/analyze_b1_api.py reports/b1-api-performance \
  --label YOUR_NEW_LABEL --output reports/b1-api-performance/YOUR_NEW_LABEL-analysis.json
```

输出代码必须先人工审阅，再调用 `scripts/audit_b1_outputs.py --reviewed`；该工具
是功能夹具，不是安全沙箱。不要直接公开整个 reports 目录：oMLX runtime
settings 和部分历史服务日志含本机生成的认证信息。已脱敏的 command 元数据、
测量 JSON、分析及源码清单可用于复现；凭据不属于报告或可发布证据。

`scripts/audit_b1_evidence.py` 是测量结束后的离线审计工具，核对完整题目集合、
自然停止、输出 hashes、客户端与入口二进制身份，以及 Splash 的缓存计数。
它不启动推理、不修改测量结果，也不将高负载样本排除。首批审计记录为
`final-candidate-v1-evidence-eight.json`，全量审计为
`final-candidate-v1-evidence-twelve.json`，完整性检查均通过。测试后重新校验
791 个已记录的对手源码、运行时及二进制文件，未发现身份变化。串行与 tree
诊断 token 解码后，也分别与六份实际 API 回答逐字相等。
`cargo fmt --all -- --check` 通过；数据分析单元测试和 Python 静态检查单独记录。

## 完成审计

2026-09-30 按用户确认的相对质量口径完成最后审查。未重写冻结协议、原始回答
或失败记录；仅增加验收结论和审批记录。

| 要求 | 当前证据 | 结论 |
|---|---|---|
| 指定分支和候选实现 | 分支 perf/b1-api-performance；基点、源码快照及二进制 hashes | 符合 |
| 同机、同目标及 drafter、固定四应用版本 | 每轮 command 元数据、prepared manifest、候选 source manifest | 符合 |
| 实际 API、B1、关闭思考、自然停止、代码与知识任务 | 冻结客户端及六题；288 个有效请求；无截断或 reasoning 泄漏 | 符合 |
| 预先固定统计和公平比较 | 八轮协议加保留原样本的四轮补测；全量审计；背景负载限制已披露 | 符合本协议 |
| 三项胜 oMLX、至少两项不弱于 TensorFold | 十二轮逐题配对统计，95% CI；共同 tokenizer 复核 | 三项胜 oMLX，TPS/E2E 胜 TensorFold |
| 输出质量不因优化降低 | 用户批准的相对口径；逐题人工对照；新复核六份基线/候选代码全部通过 | 固定任务范围内通过，已有缺陷保留 |
| 数值与稳定性回归 | 六题完整 serial/tree token 相等并与 API 文本匹配；边界测试；1476 项本地回归 | 本阶段所需检查通过，忽略项不计为通过 |
| 可复现交付及保留失败实验 | 本报告、协议、脚本、原始数据、源码快照、实验记录 | 已保存 |
| 保留兼容路径和操作边界 | 实验开关 opt-in；未改变默认路径；没有提交、合并或发布 | 符合 |

最后复核产物为 `approved-relative-code-audit.json` 和
`approved-final-evidence-audit.json`。DSH E2E、跨设备资格、长上下文及全局默认
启用不属于本阶段完成声明；后续是否提交、合并或发布仍需相应授权。
