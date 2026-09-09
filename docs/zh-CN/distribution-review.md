# v0.1.0 分发材料审查

审查日期：2026-09-09。基线：`e1d1f99ad3926b7c17bd62ac38db170e8d28da11`，
本报告同时覆盖本次尚未提交的材料修正。审查对象是 macOS arm64 默认 Release App，
不覆盖所有 Cargo features、第三方模型或未来依赖版本。

结论：依赖材料可重现，已补齐源码获取说明及已核实的图形来源。按既定发布安排，
`IRONMLX_PUBLIC_DISTRIBUTION_READY` 保持 `false`。
真实签名、公证、安装与升级验收属于后续阶段。

## 清单与许可证核对

| 范围 | 结果 |
|---|---|
| 项目 LICENSE / NOTICE | Apache-2.0 正文；NOTICE 标明 Copyright 2026 Xin Liu，并区分第三方及模型权利 |
| Rust | 271 个依赖版本；默认 arm64 Release 图，排除 dev、保留 build 依赖 |
| Native | 5 项：固定 MLX fork/JACCL、metal-cpp、fmt、nlohmann/json、gguflib |
| Swift | Sparkle 2.9.6、ZIPFoundation 0.9.20；保留 Sparkle 内嵌第三方声明 |
| 外部图形 | Hermes、oh-my-pi、Hugging Face、ModelScope 四项，来源、修改说明及许可证哈希已记录 |
| SBOM | CycloneDX 1.6，共 282 个组件；132 份许可证/声明文本；重新生成一致 |

Rust 实际采用的许可证分支覆盖 MIT、Apache-2.0、BSD-3-Clause、ISC、Zlib、MIT-0、
Unicode-3.0、CDLA-Permissive-2.0、MPL-2.0。`about.toml` 的允许列表不等于实际
采用的许可证：当前清单没有选择 LGPL/GPL/AGPL。复合 AND 条件继续保留，例如
encoding_rs、matchit、ring、unicode-ident，不将其简化为单一 MIT。

本次运行 `verify-third-party-materials.sh` 从实际锁定输入重建材料并逐字比较；
`verify-distribution-materials.sh` 验证 SBOM 可重现。该证据支持材料一致性，
不等同于对全部源码的穷尽版权溯源。

## 已补齐的缺项

### MPL 源码获取说明

`option-ext 0.2.0` 使用 MPL-2.0。原材料只有许可证和仓库链接；现由生成器在
随包 Notices 中明确说明源码仍受 MPL-2.0 约束，并提供精确版本源码下载：

[option-ext 0.2.0 源码归档](https://static.crates.io/crates/option-ext/option-ext-0.2.0.crate)

本次实际下载 7,345 字节，SHA-256 为
`04744f49eae99ab78e0d5c0b603ab218f515ea8cfe5a456d7629ad883a3b6e7d`，
与 Cargo.lock 完全相同。未配置覆盖该依赖的 Cargo patch。
后续若变为非 crates.io 的 MPL 来源，生成器会要求重新审查，避免发布错误源码链接。
依据：[MPL 3.2](https://www.mozilla.org/en-US/MPL/2.0/)、
[Mozilla FAQ Q8](https://www.mozilla.org/en-US/MPL/2.0/FAQ/)。

### 未纳入清单的内嵌图标

按权利人要求保留全部图标及现有界面。已找到 LobeHub 图标库的对应实现：
ModelScope 的四组路径完全匹配；Hugging Face 背景路径匹配，眼睛路径经简化，
省略嘴和手等细节。核对版本为 `a94750e3f5f8fc33757b839d85030e742284e43a`。
库的 MIT 许可证及 Copyright (c) 2023 LobeHub 已随包保留。
两项图形加入 `compliance/bundled-assets.json`，记录修改说明，并验证 HTML 内全部
对应图形的路径/颜色哈希及出现次数（HF 两处、ModelScope 三处）。

品牌使用依据：[Hugging Face 官方品牌页](https://huggingface.co/brand)明确提供项目
使用的品牌资产；[ModelScope 官方 logos 仓库](https://modelscope.cn/models/modelscope/logos)
的介绍提供项目使用方式，其官方 API 元数据标注 Apache License 2.0。
本次将图标限于标识下载来源，不宣称合作或背书，也不把商标权当作 MIT 许可授予的权利。
这补齐了可核对的工程出处和使用依据。

## 模型分发边界

项目文档明确：模型权重由用户从上游另行下载，项目许可证不替代模型条款；
技术支持不构成商用或再分发许可。现有 App/ZIP/DMG 打包路径调用模型文件排除检查，
归档测试验证其调用与内容。文件扩展名检查只能证明其检测范围，不能证明所有
可能形式的权重都已排除，也不替代最终候选包检查。本次没有分发模型。

## 最终发布安排

项目自有代码继续适用现有 Apache-2.0 许可证，本次不新增所有权审批要求。
按项目负责人要求，`IRONMLX_PUBLIC_DISTRIBUTION_READY` 保持 `false`，
在全部任务最终验收完成、即将公开发布时再开启。该安排不阻塞本阶段材料审查完成；
真实签名、公证、安装与升级验收仍需在后续阶段完成。

## 本次审查快照

| 文件 | SHA-256 |
|---|---|
| Cargo.lock | `606b42c2e1f8f510c612ee69673bea8dc1033c2bf063e9e292d2c21dad5bef2b` |
| third-party-inventory.json | `ba966fd1c1c81bc2ea85dbfa30675dde8cf0c0cb38fe78b6a8e200e04463fe63` |
| SBOM.cdx.json | `585f4e8955bb4b968a72dec93a306e355ce09f8cecddf6c5089f911a6c05eea7` |
| THIRD_PARTY_NOTICES.md | `89522531122eb471899f6235e5ec7a993442acf6689ea00428e67d206a807458` |

任何依赖、移植文件或品牌素材变更都需要复核相应项；本快照不覆盖后续变化。
