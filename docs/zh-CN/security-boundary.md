# 网络与图片安全边界

[English](../security-boundary.md)

IronMLX 以 `local` 模式启动时，只接受 loopback `--host`，并在该地址提供普通
HTTP，外部主机无法访问此监听器。

`lan` 模式会在一个明确选择且当前有效的局域网 IP 上增加监听器。该监听器始终使用
HTTPS，并要求所有路由（健康检查、推理、模型管理和管理路由）都带有
`Authorization: Bearer <API Key>`。通配、组播、未指定和 loopback 地址都会被拒绝；
loopback 监听器仍供本地 App 与 Dashboard 使用。

## LAN 客户端加入

1. 在 Dashboard 设置中选择 **LAN (HTTPS + API Key)** 和一个具体网络接口。
2. 保存设置。App 会生成 API key、本地 CA，以及 SAN 为所选 IP 的服务器证书。
3. 使用原生 **Copy API Key**，立即将其放入 LAN 客户端的密钥存储。Dashboard 不会
   接收密钥；若剪贴板值未被改变，系统会在 60 秒后清除。
4. 使用 **Copy CA Certificate**，在客户端安装或明确指定该 CA。不要关闭 TLS 校验。
5. 使用显示的 `https://<selected-ip>:<port>/v1` endpoint，并且只在
   `Authorization` header 中发送 API key。绝不要放入 URL 或查询参数。

**Rotate** 会在激活前生成并复制新密钥，重启后端，再废弃旧 Keychain 项。若重启
失败，会恢复原配置和凭据。

App 的普通配置只保存凭据标识和证书指纹。API key 与 TLS 私钥以 generic-password
数据保存在用户默认 macOS Keychain 中。后端只通过 stdin 接收 API-key SHA-256 摘要
和内存 TLS 材料；密钥、bootstrap payload 不会进入参数、环境变量、Dashboard 数据、
日志或故障记录。

## 图片输入

OpenAI 兼容 API 仅接受严格的 `data:image/jpeg;base64,...`、PNG 或 WebP 值；
Anthropic 兼容 API 接受同样的受控 base64 source。IronMLX 永远不会抓取 HTTP/HTTPS
图片 URL。跨机器客户端必须自行读取图片并上传内容。

资源会在昂贵的预处理前限制：请求体 32 MiB、文本 2 MiB、每次最多 8 张图片、每张
解码 10 MiB、总解码 24 MiB、宽高 8192 像素、单张像素 16,777,216、总像素 33,554,432、
单张 decoder 分配 96 MiB，以及最多 2 个并发预处理任务。

## 稳定安全错误码

| HTTP | 错误码 | 含义 |
| ---: | --- | --- |
| 401 | `auth_invalid` | 缺少、格式错误或不正确的 Bearer API key |
| 413 | `request_body_too_large` | 请求体超过 32 MiB |
| 400 | `image_remote_url_forbidden` | 提交了 HTTP/HTTPS 图片 URL |
| 400 | `image_data_url_invalid` | data URL 或 base64 格式错误 |
| 400 | `image_media_type_unsupported` | 不是 JPEG/PNG/WebP，或声明类型与检测类型不同 |
| 413 | `image_encoded_too_large` | base64 值超过解码前上限 |
| 413 | `image_decoded_too_large` | 单张解码图片超过 10 MiB |
| 413 | `image_total_decoded_too_large` | 图片总解码大小超过 24 MiB |
| 413 | `image_count_exceeded` | 图片数量超过 8 张 |
| 413 | `image_dimensions_exceeded` | 宽或高超过 8192 像素 |
| 413 | `image_pixel_budget_exceeded` | 单张图片超过像素预算 |
| 413 | `image_total_pixel_budget_exceeded` | 图片总像素超过预算 |
| 413 | `text_content_too_large` | 请求文本超过 2 MiB |
| 400 | `image_decode_failed` | 图片无法在安全限制内解码 |

LAN 认证包裹整个 router，新推理或管理 endpoint 默认受到保护，不依赖逐路由白名单。
