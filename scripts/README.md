# 脚本用途索引

`scripts/` 保留构建、发布、协议验证及可复用的模型诊断工具。
手动工具未接入 CI 不代表没有用途；这里的用途分类也不代表已在当前模型和机器上完成运行验收。
已结束的一次性实验从工作树删除，历史实现可通过 Git 查阅。

| 类别 | 入口或文件组 | 保留用途 |
|---|---|---|
| MLX 与 App 构建 | `setup-mlx.sh`、`checkout-release-mlx.sh`、`build-app-bundle.sh`、`verify-app-bundle.sh` | 开发安装、固定依赖与 Bundle 构建检查 |
| 版本与发布 | `bump-version.sh`、`release-config.sh`、`package-*.sh`、`release-archives.py`、`sign-notarize-app.sh`、`publish-stable-release.py`、`verify-release-identity.py`、`verify-version-consistency.sh` | RC、stable 的版本、打包与发布链路 |
| 自动更新 | `configure-app-updates.py`、`package-app-update.py`、`publish-update-feed.py`、`update-key-public.swift` | 更新配置、归档签名与 feed 发布 |
| 更新验收辅助 | `validate-development-auto-update.sh`、`validate-update-installation.py`、`serve-development-appcast.py`、`generate-development-update-key.swift` | 隔离的本地更新与安装测试；部分辅助文件也供 RC/测试调用 |
| 分发材料 | `generate-third-party-materials.*`、`update-third-party-materials.sh`、`verify-third-party-materials.sh`、`generate-sbom.py`、`verify-sbom.sh`、`verify-distribution-materials.sh`、`release-legal-gate.sh` | Notices、许可证、SBOM 的生成与发布检查 |
| 仓库检查 | `verify-conventional-commits.sh`、`verify-license-policy.py`、`verify-model-distribution-boundary.sh`、`verify-secrets.sh` | CI 中的提交、许可证、模型排除及凭据检查 |
| SDK 协议 | `api-contract-sdk/` | CI 使用固定版本官方 SDK 验证 HTTP/SSE 协议 |
| 量化验收 | `quant_validation_matrix.py`、`affine56_prefill.py`、`mxfp_strict_decode.py` | 可配置模型的 HTTP 正确性、prefill/decode 与性能比较 |
| 推测解码与缓存 | `benchmark_prompt_lookup_matrix.py`、`benchmark_dflash2_tensor_batching_gate.py`、`benchmark_turboquant_prefix_cache_matrix.py` | 输出一致性、生命周期、并发与性能矩阵；历史结果归档到 `reports/benchmarks/` |
| TurboQuant 数值诊断 | `turboquant_kv_validate.sh`、`turboquant_kv_long_context_validate.sh` | 调用现存 Rust 诊断二进制，比较 KV 量化误差和长上下文性能 |
| 视觉回归 | `vl_server_smoke.sh`、`gemma4_vl_semantic_check.sh` | 视觉 HTTP 冒烟及多图、分块语义检查 |
| Gemma4 分析 | `gemma4_vl_profile.sh`、`gemma4_vl_profile_report.py`、`gemma4_drafter_active_kv_regression.py` | 视觉 profile 汇总及 drafter/Active KV 回归 |
| 测试与输入 | `test_*.py`、`tests/`、`fixtures/` | 上述工具的单元测试、发布测试及固定输入 |

模型诊断工具通常需要本地模型、MLX 环境及空闲 GPU；具体参数见脚本说明。
常规 Rust 回归应使用当前 `ironmlx/tests/` 的测试目标，例如
`cargo test --release -p ironmlx --test scheduler_actor -- --ignored --test-threads=1`，
并按相应测试要求配置模型。脚本修改后应执行对应测试，而不是无差别启动所有模型实验。

## Gemma4 Drafter / Active KV 回归

使用 `gemma4_drafter_active_kv_regression.py` 检查 Gemma4 target 与 assistant drafter、
paged prefix cache 和 Active KV offload 的组合。需要本地模型和足够的 GPU/内存，不属于默认 CI。
脚本会查找 `~/.ironmlx/models/` 下的模型快照；可通过
`IRONMLX_GEMMA4_E4B_MODEL_DIR`、`IRONMLX_GEMMA4_E4B_DRAFTER_DIR`、
`IRONMLX_GEMMA4_12B_MODEL_DIR`、`IRONMLX_GEMMA4_12B_DRAFTER_DIR` 覆盖路径。

在仓库根目录先生成执行计划：

```bash
python3 scripts/gemma4_drafter_active_kv_regression.py --dry-run --variant 12b_b2
```

准备好模型后，将 `--dry-run` 替换为 `--build` 执行回归。
检查涵盖模型加载、Active KV 策略和容量、swap 错误、MTP 状态及请求完成情况。

## TurboQuant / Prefix Cache 矩阵

`benchmark_turboquant_prefix_cache_matrix.py` 比较基础配置、仅 TurboQuant、
仅 Prefix Cache 和两者同时开启的四种组合，分别启动并停止服务，使用独立缓存目录测量冷缓存与热缓存。

```bash
python3 scripts/benchmark_turboquant_prefix_cache_matrix.py \
  --model-dir /path/to/model/snapshot \
  --prompt-len 2048,8192 --max-tokens 16 --runs 3 --kv-quant k3v4
```

可先添加 `--dry-run` 检查执行计划。其他参数见各脚本的 `--help`。
这两个工具默认写入 `reports/benchmarks/<工具名称>/<时间戳>/`，可通过 `--out-root` 覆盖。
输出包括执行命令、模型与运行元数据、JSON/CSV/Markdown 摘要；实际运行还会保存服务日志和 healthz 等证据。
`reports/` 不纳入 Git；需要共享的结果应另行归档。
