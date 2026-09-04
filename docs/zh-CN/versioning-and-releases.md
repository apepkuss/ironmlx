# 版本与发布流程

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

未来 stable tag 使用 `vX.Y.Z`，并必须与 `VERSION` 一致。DMG、App About、CLI
`--version`、`healthz.version`、release tag 和 release notes 应引用同一产品版本。

当前 development preview 使用独立的 `preview-YYYYMMDD-shortSHA` 命名空间，
不会冒充 stable semantic-version tag。

## 当前发布硬门禁

`scripts/release-legal-gate.sh` 在打包和 GitHub preview workflow 中执行。当前
`IRONMLX_PUBLIC_DISTRIBUTION_READY=false`，因此 public binary 分发必然失败。

P0-8B 完成后，只有同时满足以下条件才能由单独评审显式开启：

- `THIRD_PARTY_NOTICES.md` 存在且非空；
- `third-party-inventory.json` 存在且非空；
- `THIRD_PARTY_LICENSES/` 至少包含一份非空第三方许可证文本；
- `SBOM.cdx.json` 存在且非空；
- 材料已按最终闭源 App 的依赖与分发方式完成法律/合规复核；
- `scripts/release-config.sh` 中的 distribution-ready 标志经授权改为 `true`。

项目 `LICENSE`、`NOTICE` 与确定性生成的 `SBOM.cdx.json` 也必须进入发布材料。门禁不要求或暗示采用任何第一方开源许可证；第一方授权与版权策略由未来发布
决策单独确定。

第三方材料由 P0-8A 的锁定工程流程生成，更新与验证方式见
[第三方依赖与许可证材料](third-party-materials.md)。这些材料存在并不等于完成
法律判断，也不会自动解除 public distribution 门禁。
