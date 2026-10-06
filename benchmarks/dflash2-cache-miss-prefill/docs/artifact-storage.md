# 验收材料的保存与恢复

本次精简只调整 `c37ae0e57` 新增材料的保存位置，不改变优化源码、实验结果、原失败结论或较早提交的证据。

## 仓库内保留

- 测量协议、固定会话顺序、复现与分析脚本、源码补丁、报告和结果摘要。
- `results/diag30k-hit-v1/run.json`、`results/r1-confirmation-v1/run.json`、`results/scenarios-docfork-v1/run.json` 原始数据，内容不变。
- R1 的逐轮原始时间、命中量、输出和身份信息仍可直接读取；`scripts/analyze_r1.py` 可以在不恢复日志、不启动模型的情况下重算专项结果。

## 仓库外归档

73 个完整服务日志、过程输出、测试日志及文档分叉 token-id 转储已移出 Git，保存在主工作区的 `reports/` 目录。它们不是无效数据；逐 token 复核和查看详细诊断日志时需要恢复。

- 本地目录：`/Users/xin/workspace/ironmlx-backend/reports/dflash2-cache-miss-prefill/`。
- 内容：`ARCHIVED_ARTIFACTS.sha256` 中的 73 个文件；目录自身的 `MANIFEST.sha256` 与该清单一致，另保留精简前的完整清单 `MANIFEST.before-slim.sha256`。
- 原始文件逐项 SHA256 已核对。目录位于功能 worktree 外，移除本功能 worktree 不会移除它；目前只是本地归档，不代表已公开上传。其他机器复核完整证据时需另外取得这些文件。

## 哈希清单

三份清单均使用 `SHA256  字节数  套件相对路径` 格式：

- `MANIFEST.sha256`：当前 Git 内文件，清单不列自身。全新 checkout 可直接核验。
- `ARCHIVED_ARTIFACTS.sha256`：上述 73 个归档文件。
- `LOCAL_ARTIFACTS.sha256`：此前已忽略的二进制、footprint 采样和 probe 动态库等本地产物；这些文件未移动，也不包含在本次归档中。

`MANIFEST.before-slim.sha256` 只用于保留精简前的身份记录，不应覆盖当前 Git 清单。

## 恢复归档证据

在 `benchmarks/dflash2-cache-miss-prefill` 目录执行：

```sh
archive=/Users/xin/workspace/ironmlx-backend/reports/dflash2-cache-miss-prefill
awk '{print $3}' ARCHIVED_ARTIFACTS.sha256 | tar -cf - -C "$archive" -T - | tar -xf -
awk '{print $1 "  " $3}' ARCHIVED_ARTIFACTS.sha256 | shasum -a 256 -c -
```

仅恢复清单列出的文件，不覆盖当前 `MANIFEST.sha256`；恢复出的文件受 `.gitignore` 保护。随后可按结果报告运行 `compare_scenarios.py`，重新核对文档分叉的 token 记录。

Git 内材料可单独校验：

```sh
awk '{print $1 "  " $3}' MANIFEST.sha256 | shasum -a 256 -c -
```
