# 动态 EnginePool

[English](../engine-pool.md)

面向需要在同一服务中按请求 model 分派模型的 CLI 用户。此 manifest 属于 `ironmlx serve`；App 通过 Dashboard 管理自己的模型。已有 App 后端运行时，不要另启第二个后端。

## Manifest

使用 `ironmlx serve --model-manifest` 指定配置文件：

```json
{
  "default_model": "qwen-main",
  "max_loaded_models": 2,
  "models": [
    {
      "id": "qwen-main",
      "path": "/models/qwen-main",
      "load_policy": "preload",
      "scheduler_profile": "/profiles/qwen-main.json"
    },
    {
      "id": "qwen-mtp",
      "path": "/models/qwen-mtp",
      "load_policy": "lazy",
      "mtp_model_dir": "/models/qwen-mtp-head",
      "mtp_draft_tokens": 3
    },
    {
      "id": "disabled-exp",
      "path": "/models/disabled-exp",
      "load_policy": "disabled"
    }
  ]
}
```

| 字段 | 含义 |
| --- | --- |
| `default_model` | 请求未指定 model 时使用的模型；仅一个模型启用时可省略 |
| `max_loaded_models` | 加载引擎上限，至少 1 且不少于 preload 数量 |
| `models[].id` | 稳定公开 ID，也是分页前缀缓存命名空间 |
| `models[].path` | 本地模型目录 |
| `models[].load_policy` | preload、lazy、disabled；默认 lazy |
| `models[].scheduler_profile` | 覆盖该模型的全局 scheduler profile |
| `models[].mtp_model_dir` / `mtp_draft_tokens` | 逐模型 MTP 设置；使用 manifest 时不接受全局 MTP 参数 |

启用模型必须有受支持的配置与目录，不兼容架构会在启动时失败，而不是等首次请求。
支持的标识包括 `qwen3_5`、`qwen3_5_moe`、`gemma4`、`gemma4_unified`、`glm4_moe_lite`、`llama`、`minicpmv4_6` 和 `diffusion_gemma`；具体模型限制见[支持模型](supported-models.md)。

## 加载与卸载

preload 在启动时加载，失败会阻止启动；lazy 在首次请求或控制 API 调用时加载；disabled 不参与路由。
lazy 加载失败后保持 failed，普通请求直接失败，需通过 load API 显式重试。
达到加载上限时，优先卸载最久未使用且不在处理请求的 lazy 模型；preload 模型固定保留。

## HTTP API

| 路由 | 用途 |
| --- | --- |
| `GET /v1/models` | 已启用模型及运行状态 |
| `GET /healthz` | 池状态、加载数量、逐模型状态及已加载引擎健康信息 |
| `POST /v1/models/:model_id/load` | 显式加载 lazy 模型或重试失败加载 |
| `POST /v1/models/:model_id/unload` | 卸载未使用且非 preload 的模型 |

状态包括 unloaded、loading、loaded、failed、disabled，分别对应未加载、加载中、已加载、失败和禁用。

## 示例

```bash
ironmlx serve \
  --model-manifest /path/to/models.json \
  --host 127.0.0.1 \
  --port 8080 \
  --paged-prefix-cache-dir ~/.ironmlx/cache/paged_prefix_cache
```

```bash
curl -s http://127.0.0.1:8080/v1/models
curl -s -X POST http://127.0.0.1:8080/v1/models/qwen-mtp/load
curl -s -X POST http://127.0.0.1:8080/v1/models/qwen-mtp/unload
```
