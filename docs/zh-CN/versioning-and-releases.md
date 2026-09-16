# 版本与发布流程

[English](../versioning-and-releases.md)

## 单一产品版本

仓库根目录 `VERSION` 是产品版本的规范输入。Rust workspace、CLI、`healthz`、
App `CFBundleShortVersionString` 与发布 tag 必须保持一致。`CFBundleVersion` 是单调
递增的正整数构建号。

不要手动逐文件改版本。发布或跳版时运行：

```bash
scripts/bump-version.sh 0.2.0
```

脚本会更新 `VERSION`、workspace package version、内部显式依赖版本、Cargo.lock
与 App plist；版本变化时默认把 App build number 加一。需要指定构建号时：

```bash
scripts/bump-version.sh 0.2.0 7
```

完成后必须提交脚本生成的全部改动，并运行：

```bash
scripts/verify-version-consistency.sh
```

CI 会执行同一检查，并验证每个 workspace crate 都声明 `publish = false`。因此
IronMLX 不能通过 `cargo publish` 意外发布到 crates.io。

## Tag 与发布说明

Stable tag 使用 `vX.Y.Z`，并必须与 `VERSION` 一致。DMG、App About、CLI
`--version`、`healthz.version`、release tag 和 release notes 应引用同一产品版本。

RC 使用 `vX.Y.Z-rc.N` tag。

## 当前发布硬门禁

`scripts/release-legal-gate.sh` 检查已授权的分发开关和许可证、Notices、清单及可重现 SBOM。开关状态以 `scripts/release-config.sh` 为准；门禁通过不等于执行公开发布。签名、公证、tag 身份和显式发布步骤仍须通过各自检查。材料更新方法见[发布流水线](stable-release-pipeline.md)。

## 正式发布产物身份

正式打包前，使用明确的现有发布 tag 检查源码及 App：

```bash
python3 scripts/verify-release-identity.py v0.1.0
python3 scripts/verify-release-identity.py v0.1.0 dist/IronMLX.app
```

源码检查要求 `refs/tags/` 下的 tag 指向 HEAD、与 `VERSION` 一致，且工作区
clean，包括未忽略的未跟踪文件。支持 lightweight 和 annotated tag。提供 App
参数时，还要求产品版本、build number、来源提交一致，来源状态为 `clean`。
正式打包脚本始终执行两项检查；第三个参数为发布 tag，默认是 `v` 加 `VERSION`。
自动和手动触发的正式工作流均显式传入选定 tag。

此门禁检查身份元数据，不证明密码学构建来源，也不替代签名和公证检查。
本地开发构建及其静态 Bundle 检查仍允许 dirty 源码。

RC 身份验收使用独立的显式模式：

```bash
python3 scripts/verify-release-identity.py --candidate v0.1.0-rc.1 dist/IronMLX.app
```

候选模式只接受 `vX.Y.Z-rc.N`，N 必须为无前导零的正整数。基础版本 `X.Y.Z`
必须与 App 版本和 `VERSION` 一致，仍执行 clean、tag/HEAD、build number 和
Bundle 来源检查。正式打包和发布不启用此模式，继续拒绝 RC tag。

## RC 与稳定版发布入口

| 模式 | Tag | 工作流 |
| --- | --- | --- |
| RC | `vX.Y.Z-rc.N` | Release Candidate |
| stable | `vX.Y.Z` | Stable Release |

Tag push 和手动 `publish=false` 只构建和验证；显式 `publish=true` 才进入签名、公证和发布流程。
两者都要求仓库级更新公钥。凭据、执行顺序、更新通道与失败恢复统一见[发布流水线](stable-release-pipeline.md)。

## 归档内容检查

正式输出目录统一包含 `IronMLX-X.Y.Z.dmg`、`IronMLX-X.Y.Z.zip`、`SHA256SUMS`、
独立法律材料和 `THIRD_PARTY_LICENSES/`。ZIP 内以 `IronMLX-X.Y.Z/` 为根目录；
DMG 卷根目录仅包含 `IronMLX.app`、`Applications` 快捷方式和 `Documentation/`，
其中放置法律材料及 `THIRD_PARTY_LICENSES/`。输出目录必须不存在或为空；脚本不自动
删除旧产物。

无需 release tag 或 Developer ID 即可验证归档机制：

```bash
python3 scripts/release-archives.py assemble dist/IronMLX.app .build/archive-check
python3 scripts/release-archives.py verify dist/IronMLX.app .build/archive-check
```

检查涵盖当前材料/SBOM、产品版本和标识、完整校验和清单、实际 ZIP 解压与 DMG
只读挂载，以及 App 每个文件、执行位和符号链接与参考 Bundle 的一致性，包括
版本/来源元数据及内嵌法律材料。这不证明参考 Bundle 来自当前 clean 提交或已签名。
正式打包入口仍先执行身份、clean、分发授权、静态 Bundle、签名与 Gatekeeper 门禁。
仅内容验证产生的文件不能作为已批准的正式版发布。

凭据、签名、公证、更新源与发布恢复流程统一见[发布流水线](stable-release-pipeline.md)。
