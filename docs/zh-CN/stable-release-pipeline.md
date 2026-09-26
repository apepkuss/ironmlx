# RC 与稳定版发布流水线

[English](../stable-release-pipeline.md)

面向发布维护者。[版本与发布](versioning-and-releases.md)定义 tag 和产物身份；本文统一说明签名、发布与失败恢复。用户更新操作见[自动更新](automatic-updates.md)。

## 构建与验证模式

| 模式 | Tag | 分发 / 更新通道 | 公开 Release |
| --- | --- | --- | --- |
| RC | `vX.Y.Z-rc.N` | `release-candidate` | prerelease，`make_latest=false` |
| stable | `vX.Y.Z` | `stable` | 稳定版 |

两条工作流都从现有不可变 tag 的干净完整 checkout 构建，核对 VERSION、build、源码提交及 Bundle 身份。MLX 由 `scripts/release-config.sh` 固定，工具链版本以工作流为准。
Tag push 和手动 `publish=false` 只构建并检查归档，不读取签名 Secret、不创建 Release 或发布 feed。验证模式也要求仓库变量 `IRONMLX_UPDATE_PUBLIC_ED_KEY`。归档检查不能替代公证或 Gatekeeper 验收。

## 发布配置

显式手动 `publish=true` 传递已验证的精确 App ZIP，保留权限、框架软链接和签名。分发材料必须通过 `release-legal-gate.sh`；工作流检查授权开关，不修改该开关。
发布 job 使用 GitHub `stable-release` Environment；其部署规则按仓库发布策略配置。所需值如下：

| 类型 | 名称 | 内容 |
|---|---|---|
| Secret | `IRONMLX_DEVELOPER_ID_P12_BASE64` | 含 Developer ID Application 证书及私钥的 PKCS#12，Base64 编码 |
| Secret | `IRONMLX_DEVELOPER_ID_P12_PASSWORD` | 非空 PKCS#12 导出密码 |
| Variable | `IRONMLX_SIGNING_IDENTITY` | 完整 `Developer ID Application: Name (TEAMID)` |
| Variable | `IRONMLX_APPLE_TEAM_ID` | 证书 Team ID |
| Secret | `IRONMLX_NOTARY_KEY_ID` | App Store Connect 团队 API Key ID |
| Secret | `IRONMLX_NOTARY_ISSUER_ID` | 团队 API Issuer UUID |
| Secret | `IRONMLX_NOTARY_PRIVATE_KEY` | API Key 的 `.p8` 内容 |
| Secret | `IRONMLX_UPDATE_PRIVATE_ED_KEY` | 与构建公钥匹配的 Sparkle Ed25519 seed |

公钥变量保留在仓库级，因为构建 job 不使用生产 Environment。Sparkle 私钥也可由现有仓库 Secret 提供。私有凭据不得提交到仓库。

## 签名与发布顺序

1. 下载已验证候选包，重新核对精确源码/tag/Bundle 身份；签名前确定全部 Bundle 元数据。
2. 将凭据导入临时 keychain，从内到外签署 Sparkle、Rust helpers 和 App，启用 hardened runtime 与可信时间戳。保留 Downloader 的 sandbox/network entitlements，不添加 JIT 或禁用库验证例外。
3. 向 Apple 提交 App ZIP，必须收到 `Accepted`，再 staple、验证票据并执行签名和 Gatekeeper 检查；plist 状态声明不能替代票据证据。
4. 从已 staple 的 App 生成安装 ZIP/DMG，对 DMG 签名、公证、staple，更新校验和，再解压/挂载，与参考 App 和源码材料核对。
5. 生成独立的 App-only 更新 ZIP，使用 Sparkle 签署并验证 ZIP/XML，记录 `update.json` 及覆盖上传资产的 `RELEASE-SHA256SUMS`。安装 ZIP 不充当更新 ZIP；当前不生成差分更新。
6. 创建草稿，上传完整资产集合，下载并核验哈希，再核对远端 tag，最后按 RC/stable 状态公开。
7. 复验公开下载和 Release 身份后发布 feed。RC 只写自己的更新源，此操作不将 RC 提升到稳定版通道。

两种模式都保留 `IronMLX.app` 名称，共享打包目录为 `.build/stable-release`。App 和最终 DMG 分别提交公证。

## 失败与重试

构建、签名或公证失败时，在创建 Release 前停止。临时密钥、证书和 keychain 在正常失败/终止时清理，工作流还有 `always()` 清理。Apple 回执保存在 `.build/notarization`，不作为公开资产。
上传或草稿验证失败时保留草稿、不公开。普通重跑拒绝已存在的 Release，不替换资产或自动删除草稿；应检查原因后显式处理。
Release 已公开但 feed 发布失败时，用完全相同的公开 ZIP/XML/`update.json` 重试 `publish-update-feed.py`，不要重新构建或覆盖资产。相同 manifest 可幂等重试。

## 更新通道与版本规则

App 产品版本保持 `X.Y.Z`，RC tag/feed 显示版本附加 `-rc.N`。RC 后缀是候选版序号，不是 App build number。Sparkle 按正整数 `CFBundleVersion` 比较，每次更新都必须全局递增，包括新产品版本的首个 RC 和同产品版本的后续 RC。使用 `scripts/bump-version.sh X.Y.Z BUILD` 指定高于所有已发布 build 的新值；RC 工作流保留源码中的 build number。
更新源位于独立 `updates` 分支：`https://raw.githubusercontent.com/<owner>/<repo>/updates/stable.xml` 和同目录的 `release-candidate.xml`；feed 发布不修改源码分支或 release tag。
RC 条目带 `release-candidate` 通道标记，稳定版客户端不订阅；已安装 RC 切换稳定版需要主动安装/切换。
发布器拒绝倒退或冲突的 build，使用非强制更新；并发冲突时失败而不覆盖另一通道。首次发布前 feed 可以不存在；更新源缺失或不可用不能阻止现有 App 使用。

## 长期签名配置

更新使用的 Ed25519 key 与 Apple Developer ID 无关。现有工具可生成所需的
32 字节 seed；工具名称中的 development 不限制密钥用途：

```bash
swift scripts/generate-development-update-key.swift /absolute/private/path/update-key
```

目标文件须不存在，生成后权限为 0600，标准输出只有公钥。将公钥配置为 GitHub
仓库 variable `IRONMLX_UPDATE_PUBLIC_ED_KEY`，私钥文件内容配置为 secret
`IRONMLX_UPDATE_PRIVATE_ED_KEY`。私钥保存在仓库外；发布后换 key 需要单独迁移方案。
脚本会检查私钥派生的公钥与 App 内嵌公钥一致。源码测试只生成临时 key，不创建
长期密钥或修改 GitHub 设置。

正式 App 必须在构建、签名前配置 `IRONMLX_UPDATE_CHANNEL=stable`、对应
`IRONMLX_UPDATE_FEED_URL` 和 `IRONMLX_UPDATE_PUBLIC_ED_KEY`。RC 工作流自动填写
RC 通道和 URL。RC 和 stable 工作流的验证模式也要求配置长期公钥；实际发布还必须具备匹配的私钥 secret。

## 第三方材料更新与验证

### 目标与边界

发布材料生成为实际打入 `IronMLX.app` 的 macOS arm64 Release 产品建立可复现的工程
清单，而不是直接复制 `Cargo.lock` 全集。当前范围包括：

- `ironmlx` 与 `iron-bench` 两个 Release 二进制的默认 feature 依赖；
- `aarch64-apple-darwin` 目标，排除 dev dependencies，保留 build dependencies；
- Swift App 的外部 SwiftPM 包（当前为 Sparkle 与 ZIPFoundation）；
- MLX C++ 分叉及其 Release 构建实际纳入的 metal-cpp、fmt、nlohmann/json、
  gguflib；JACCL 是锁定 MLX checkout 内的组成部分；
- 直接打入 App Bundle 的第三方图形与品牌资源；
- 明确排除由 macOS 提供且未复制进 App 的系统 frameworks，以及由用户另行下载、
  受各自条款约束的模型权重；IronMLX 的责任边界见[模型权利边界](model-license-boundary.md)。

这些输出用于依赖漂移检测和保留第三方声明，不构成法律意见或公开分发授权。

### 规范输入与生成物

- `Cargo.lock`、两个产品 Cargo manifest 与 `about.toml`；
- `ironmlx-app/Package.swift`；
- `scripts/release-config.sh` 锁定的非官方 MLX 分叉 commit；
- `compliance/native-dependencies.json` 中锁定的原生依赖版本、源码完整性和许可证
  文件 SHA-256；
- `compliance/bundled-assets.json` 中锁定的第三方资源来源、上游与 Bundle 文件
  SHA-256、版权及许可证材料；
- `third-party-inventory.json`：规范化机器可读工程清单；
- `THIRD_PARTY_NOTICES.md`：组件、版本、许可表达式与许可证文件映射；
- `THIRD_PARTY_LICENSES/`：从锁定依赖源码提取的完整许可证原文。
- `SBOM.cdx.json`：从同一份清单确定性生成的 CycloneDX 1.6 软件物料清单。

MLX 条目标明 IronMLX 使用 `apepkuss/mlx` 分叉，而不是官方 MLX repo，并同时
记录精确 fork commit、官方 upstream repo 与 upstream base revision。生成器还会
验证 MLX/fmt/gguflib 的 Git commit，以及 metal-cpp/nlohmann JSON 下载归档、
第三方资源 Bundle 文件和所有许可证文件的 SHA-256。上游资源 SHA-256 作为已审查
的来源锁记录保留在规范输入中；离线生成器验证经过 App 适配后的 Bundle 文件。

### 更新流程

安装锁定工具并完成一次 MLX Release 配置/构建后，运行：

```bash
cargo install --locked --features cli --version 0.9.1 cargo-about
CARGO_ABOUT="$(command -v cargo-about)" scripts/update-third-party-materials.sh
scripts/verify-third-party-materials.sh
```

依赖变化后必须完整审查上述三个生成物的 Git diff，不允许只更新哈希以绕过失败。
CI 会先按 `Cargo.lock` 获取依赖，再使用同一版本的 `cargo-about` 离线扫描；在
App 构建所用的实际 MLX/CMake 输入就绪后重新生成到临时目录，并与 tracked 材料
逐字节比较。

### App 与归档

Release 构建将项目 `LICENSE`、`NOTICE`、`SBOM.cdx.json` 以及第三方材料复制到：

```text
IronMLX.app/Contents/Resources/Legal/
```

App 菜单的 `Third-Party Notices…` 可以读取并显示 bundled notices。RC 和 stable 打包时，
ZIP 根目录保留归档材料，DMG 根目录仅包含 `IronMLX.app`、`Applications` 快捷方式和
`Documentation/`；法律材料放在 `Documentation/` 中。`scripts/release-archives.py` 会解包
ZIP、挂载 DMG，并核对归档中的材料与源码目录一致。

### 模型权重排除检查

每个准备分发的 App、DMG 或 ZIP 都必须通过发布脚本的模型分发边界检查，确认产物
不含常见模型权重文件。该检查不能替代用户对上游许可证的审查，也不改变用户对
所下载模型的全部责任。

## 验证边界

`test_app_updates.py` 检查通道/build 规则和真实 Sparkle 签名，包括错误密钥与篡改拒绝。`validate-update-installation.py` 使用隔离 App 和 localhost HTTPS，结束后移除临时钥匙串信任；它不加载生产模型，也不证明完整生产升级通过。
本地 Apple/GitHub 失败路径测试验证顺序与清理，不等于远端验收。实际公证、Gatekeeper、安装、模型恢复和数据保留证据应绑定候选包单独记录；本文不声明某个版本已经完成验收。
