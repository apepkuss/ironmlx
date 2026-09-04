# 第三方依赖与许可证材料

## 目标与边界

P0-8A 为实际打入 `IronMLX.app` 的 macOS arm64 Release 产品建立可复现的工程
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

## 规范输入与生成物

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

## 更新流程

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

## App 与归档

Release 构建将项目 `LICENSE`、`NOTICE`、`SBOM.cdx.json` 以及第三方材料复制到：

```text
IronMLX.app/Contents/Resources/Legal/
```

App 菜单的 `Third-Party Notices…` 可以读取并显示 bundled notices。未来获准启用
development preview 打包时，DMG 与 ZIP 根目录也会包含相同材料；验证器会解包
ZIP、挂载 DMG，并逐项比较 App 内部和归档根目录的材料。

## P0-8B 保留门禁

以下内容不因 P0-8A 自动完成：

- 对每个许可证表达式、归属声明与闭源分发义务的最终法律复核；
- 对已生成 CycloneDX `SBOM.cdx.json` 的正式复核与批准；
- 模型权利边界声明的最终法律复核（声明 IronMLX 不重新授权模型，用户自行查阅上游条款）；
- 将 `IRONMLX_PUBLIC_DISTRIBUTION_READY` 改为 `true` 的明确授权；
- Developer ID 签名、公证、stapling 与真实最低目标机器验收。
