# Qwen3.8-27B affine4 DFlash2 回归归档

日期：2026-08-29

状态：`dd37fde` 回归通过。本记录补充 4-bit DFlash2 在当前优化提交上的
Greedy、Sampled 和跨 batch-width 精确性结果；8-bit 数据见对应的 affine8
归档。历史 P3 正式验收记录保持不变。

## 环境与测量契约

| 项目 | 配置 |
|---|---|
| 当前分支 | `feat/qwen3-8-27b-8bit@60f02b65ab0f3b0179adddf80f958944d6ff8655` |
| 数据执行 commit | `dd37fde67af113501f40ce893b55e7a5609907e1` |
| MLX | `fix/nax-qmm-product-shape-main@73ad5df20cb30be4192e5c4d0ae8130674773427` |
| 硬件 | Apple M5 Max，128 GB 统一内存 |
| Target | `mlx-community/Qwen3.8-27B-4bit@3e6447f082e89cc7f0bc6e5441afd38dfce760ff` |
| MTP | `mlx-community/Qwen3.8-27B-MTP-4bit@b643c01b6d3b094e325edb6ebd832e16c486c575` |
| DFlash2 | `z-lab/Qwen3.8-27B-DFlash2@50307d4c4cde6860d4eee73e2547cd786fe8e8a4` |
| DFlash2 配置 | block size 4，draft runtime 4-bit，Greedy/Sampled，B1/B2/B4 |
| 峰值内存 | `20,914,906,208 B` |

本记录中的正式 Q4 配置使用 128-token Greedy 输出；普通 decode 与 DFlash2
使用相同模型、prompt、cache 状态、warmup、cooldown 和执行顺序。Q3 数据为
补充测量，不与 Q4 正式对比表混合计算。

## Q4 Greedy 正式回归

| Batch | 当前 DFlash2（TPS） | 历史正式 TPS | 变化 | 结论 |
|---:|---:|---:|---:|---|
| B1 | 56.108 | 38.781 | +44.7% | 通过 |
| B2 | 60.254 | 44.578 | +35.2% | 通过 |
| B4 | 58.139 | 46.043 | +26.3% | 通过 |

B4 的第三组受持续负载影响为 `48.244 TPS`，仍高于历史正式值；三组中位数
通过。峰值内存与此前基线一致，无新增内存增长。

## Q3 补充吞吐测量

| Batch | DFlash2 aggregate TPS |
|---:|---:|
| B1 | 50.430 |
| B2 | 67.404 |
| B4 | 67.811 |

Q3 行仅作为同一提交上的补充观测，不替代 Q4 正式回归结果，也不用于推导
跨配置的性能承诺。

## 精确性回归

- Greedy 64-token：B4 与逐行 B1 的 token/context 严格一致。
- Sampled 256-token：B4 与逐行 B1 的 token/context 严格一致。
- Q4 B1/B2/B4 输出 hash 完全一致。
- batched prefill 启用后的 4-bit DFlash2 路径未出现输出或 cache 状态回归。

## 证据边界

- 结果仅适用于上述提交、MLX、checkpoint、硬件和请求条件。
- 本归档是回归摘要，不包含原始 JSON、prompt 正文或生成正文。
- 历史 P3 正式数据仍见
  [`dflash2-final-validation/2026-08-23`](../../dflash2-final-validation/2026-08-23/summary.md)；
  新旧数据应按各自提交和测量契约分别解读。
