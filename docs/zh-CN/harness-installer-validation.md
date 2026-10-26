# Harness 可选安装与菜单入口验证

验证日期：2026-10-26。环境：Apple Silicon、macOS 26.6.2。

实现位于独立 worktree 的 `feat/harness-installer-and-chat` 分支，基于 `feat/dsh-desktop-one-click-setup` 的 `661e180de48fc0a412e8c23e6a354b0125e1ec62`。未修改原分支的工作区，未发布 Release。

## 已交付行为

- `.pkg` 始终显示组件选择：IronMLX 必选，官方 DeepSeek Harness Desktop 默认勾选、可取消。
- 已安装的可识别兼容或更新版本复用；默认位置的可识别旧版本升级；未知应用及其他位置的不兼容副本保留并跳过。用户可取消 Harness 组件。安装阶段不配置模型和凭据。
- 官方 Harness `0.2.1-rc.2` arm64 ZIP 的 URL、SHA-256、bundle ID 和发布团队固定在 `scripts/harness-desktop-source.json`。下载后先校验，保留其签名和全部上游资源。
- 菜单新增五种语言的“与 DSH 对话”及“打开 DeepSeek Harness”悬浮提示。缺安装、服务、对话模型或配置时给出本地化引导；已有运行实例优先，按明确路径打开；不自动改写配置。
- Agent 页面资源未修改，现有一键配置保留。新的原生路由仅负责进入现有官方桌面版配置卡片。
- RC、Stable 流程新增组件准备、PKG 构建及内容校验、Installer 签名和公证、校验清单及发布下载验证。DMG/ZIP 和 Sparkle 仍只交付 IronMLX。

## 自动化结果

| 检查 | 结果 |
| --- | --- |
| 相关 Swift 测试 | 注册 49 项：48 项通过，1 项真实环境测试默认跳过；该项另行开启后通过 |
| Swift Release 构建 | 通过 |
| Harness 安装器测试 | 7 项通过：状态矩阵、原生 Installer 选项评估、取消选择、payload 比对、目标写入前复查、下载哈希及符号链接边界 |
| 发布失败边界及签名测试 | 11 项通过；Apple/GitHub 操作使用模拟，未真实发布 |
| ZIP/DMG 归档及 PKG 校验清单测试 | 7 项通过 |
| Release notes 测试 | 3 项通过 |
| 最终真实 PKG 解包比对 | 通过：两个应用的完整清单、脚本、条件选择、来源及许可声明、五种语言资源 |
| YAML、JavaScript、Shell、diff 检查 | 通过 |
| 版本一致性、分发材料和 CycloneDX SBOM 检查 | 通过，SBOM 320 个组件；可选 Harness 的来源记录单独随安装器定义交付 |
| 官网下载选择 | 优先 PKG，旧 Release 无 PKG 时回退 DMG，断言通过 |

可重复执行的主要命令：

```bash
swift test --package-path ironmlx-app --filter 'Harness|harness|MenuBar|DashboardWindow|AppLaunch'
IRONMLX_TEST_HARNESS_CHAT_LIVE=1 swift test --package-path ironmlx-app \
  --filter harnessChatOpensExistingConfiguredDesktopAgainstLiveLocalService
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s scripts/tests -p test_harness_installer.py -v
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s scripts/tests -p test_stable_release.py -v
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s scripts/tests -p test_release_archives.py -v
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s scripts/tests -p test_render_release_notes.py -v
```

本地证据日志保留在 `.build/validation/harness-installer/`。

## 原生界面与真实对话

已打开实际生成的安装器，检查组件页面：IronMLX 的复选框选中且禁用；Harness 默认选中且可取消。此机已有官方 `0.2.1-rc.2`，Harness 大小为 Zero KB、无活动安装 payload；取消后界面显示 Skip。未点击最终安装操作，没有覆盖系统应用。

真实启动测试使用本机现有服务和既有 Harness 配置，以生产入口相同的启动服务打开官方桌面应用，没有写入配置。随后通过官方窗口新建验证会话，确认所选模型为 `mlx-community/Qwen3.5-4B-MLX-4bit`：

1. 要求仅返回 `IRONMLX_OK`，界面显示完成并返回该值。
2. 要求使用 read 工具读取本次创建的验证文件。执行记录显示“已读取文件”及准确路径，最终返回文件中的 `IRONMLX_DSH_TOOL_OK`。

验证文件已清理；会话 `Confirm IRONMLX_OK` 保留作体验验证记录。五种菜单文案和 tooltip 通过原生 NSMenuItem 属性断言验证。

## 后续验收安排

本次先提交并推送功能分支，正式签名、公证、Gatekeeper 安装评估及全新系统安装／升级／卸载验证，待后续合并到 `dev` 后补齐。这些项目仍为待验证，不作为本次已通过项。

现有发布工作流的 `publish=false` 只构建及验证内容；`publish=true` 会签名、公证并继续公开发布。需要先验收正式签名产物时，应使用本地签名脚本，或另行增加只签名和保留产物的工作流模式。

## 开发产物与边界

本地开发包位于 `.build/installer-candidate/IronMLX.pkg`，SHA-256：

```text
1476446fd209053f7815949bca200f5201fc178f5dd6b979fc4c4653f90a20f0
```

该包使用本次重新构建的 Swift Release 可执行文件，以及基线已有的 Rust 运行时。IronMLX 应用采用临时 ad-hoc 签名；Harness 保持官方签名。PKG 未进行正式签名或 Apple 公证，不是正式发布产物。

正式发布还需按发布文档配置 `Developer ID Installer` 身份及对应 P12，执行真实签名、公证、staple 和 Gatekeeper 安装评估。本次未进行全新 macOS 虚拟机中的实际安装、升级和卸载，也未在系统目录执行开发包安装；原生选项评估及临时目录测试不能替代这些验证。移位应用的 locator 返回值已覆盖状态矩阵，尚未在干净系统中实测所有自定义安装位置。

扩大运行整个 Swift 测试集时，本分支出现 `freed pointer was not the last allocation` / signal 6。原始基线分支干净工作区运行整个测试集同样出现该 allocator 崩溃；单独执行所涉及的日志导出测试在两边均通过。本次相关测试集通过，未声称完整 Swift 测试集通过，也未修改无关运行时测试。
