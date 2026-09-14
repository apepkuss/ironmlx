# RC 签名与公证

RC 使用不可变 vX.Y.Z-rc.N tag；产品版本保持 X.Y.Z。tag push 或 publish=false 仅构建、验证，无 Apple 凭据，不签名公证、不公开资产。仓库变量 IRONMLX_UPDATE_PUBLIC_ED_KEY 必须配置。

publish=true 使用 stable-release Environment 中现有的 Developer ID、公证和 Sparkle 凭据。通过分发授权门禁后传递精确提交的候选包，App 内到外签名、公证 Accepted、stapling；打包后 DMG 同样签名、公证、stapling 并重新核对校验和。

签名保留 release-candidate 分发和更新通道，feed 为 updates/release-candidate.xml，不写 stable.xml。Release 先创建草稿，上传并下载核对完整资产集合，再公开为 prerelease，make_latest=false；公开下载复核后更新 RC feed。

本改动不创建 tag，不开启公开分发门禁，也不公开 Release/feed。无需新 tag 的本地回归通过模拟 Apple/GitHub 验证失败边界；不代表新工作流已经在 GitHub 实际运行。正式运行前仍需补齐 Environment 凭据。
