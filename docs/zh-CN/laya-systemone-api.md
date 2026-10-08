# System One API

[English](../laya-systemone-api.md) · [API 参考](api-reference.md) · [服务与管理 API](service-api.md)

IronMLX App 支持 `aac6fef/laya-multilingual-mlx` 的下载、加载、卸载与 API 服务。
该模型处理选择（choice）、评分（score）和真假概率（noul）问题，不生成聊天文本。

本文定义 Laya 的选择、评分与真假概率接口，并在后文提供 [App 配置](#使用步骤)与[独立 CLI](#可选的独立-cli)说明。App 与 EnginePool 的公共访问约定见[服务与管理 API](service-api.md)；独立 CLI 使用本页规定的认证方式。

## API 端点

- `POST /v1/systemone`：接受下方的[请求](#请求)，返回兼容 TypeSafe 的
  `model`、`answers` 和 `usage` 字段，详见[响应](#响应)。
- `GET /v1/models`：返回 `{"models":[{"name", "description", "release_date"}]}`。
  App 模式列出已注册的决策模型，同时保留 OpenAI 的 `object`/`data` 字段；
  独立服务模式列出已加载的 Laya 模型。`release_date` 为该检查点的
  Hugging Face 仓库创建日期（`2026-09-19`）。

`jev-latest` 等 TypeSafe 模型名不是 Laya 的别名。有认证的服务中，缺失或无效的
Bearer 凭据返回 `401`。JSON、模型 ID、问题格式或选项 token 预算错误返回
`422`；推理错误返回 `500`，工作线程停止返回 `503`。App 管理的服务还会在
模型加载失败或队列已满时返回 `503`，请求超时返回 `504`。模型加载错误同时
会显示在 App 中。

## 请求

```json
{
  "model": "aac6fef/laya-multilingual-mlx",
  "state": {"message": "发票被重复扣款，请退款。"},
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which team should handle this?",
      "criteria": {"billing": "refunds", "technical": "bugs"}
    },
    "refund": {
      "type": "noul",
      "instructions": "Is a refund requested?"
    },
    "urgency": {
      "type": "score",
      "instructions": "How urgent is this?",
      "criteria": ["can wait", "soon", "today"]
    }
  }
}
```

- `model` 标识实际加载的检查点。任何 Jev 别名都不会映射到 Laya。
- `state` 可以是字符串、JSON 对象或 JSON 数组。所有问题针对同一份 state
  求值。问题键不能为空，原样作为答案键返回。
- `choice.criteria` 是有序映射，包含 1–255 个不同的非空标签，说明可为字符串、对象、数组或 `null`。
  模型按请求中的顺序对标签评分。
- `score.criteria` 是包含 2–10 项说明的有序数组，说明可为字符串、对象或数组，索引从零开始。
  结构化说明在响应的 legend 中渲染为 JSON 字符串，因此 `legend`
  始终符合 TypeSafe 的 `map<string, string>` 结构。
- `noul.criteria` 可以提供 `true` 和 `false` 的说明；返回值为 P(true)。
- `instructions` 为必填项，可以是字符串、对象或数组。运行时在分词前
  以确定的方式渲染结构化值。

### 输入限制

`questions` 不能为空。问题与选项前缀的目标预算为 256 token，每个选项最多
48 token，必要时进一步降低该上限。如果处理后的选项仍无法放入总计
1,024 token 的上下文，请求会被拒绝。`state` 从右侧截断；选项不会被静默丢弃。

## 响应

```json
{
  "model": "aac6fef/laya-multilingual-mlx",
  "answers": {
    "department": {
      "type": "choice", "choice": "billing", "confidence": 0.5635,
      "probabilities": {"billing": 0.91, "technical": 0.09}
    },
    "refund": {"type": "noul", "noul": 0.94},
    "urgency": {
      "type": "score", "score": 1.32, "confidence": 0.1108,
      "legend": {"0": "can wait", "1": "soon", "2": "today"},
      "probabilities": {"0": 0.12, "1": 0.44, "2": 0.44}
    }
  },
  "usage": {"input_tokens": 327, "output_tokens": 0}
}
```

上述数值仅用于展示结构，并非实际测得的模型输出。`input_tokens` 统计所有
编码后的问题与 state 序列，包括重复的 state token；本模型不生成文本，
因此 `output_tokens` 为零。choice 和 score 的 confidence 使用参考运行时的
归一化熵计算；score 是从零开始的评分标准索引的期望值。
noul 在兼容 TypeSafe 的响应中仅返回概率。Laya action head 及诊断输出
保留在内部。

## 官方 SDK 客户端

Python 与 JavaScript 官方 SDK 示例见[英文 API 文档](../laya-systemone-api.md#official-sdk-clients)。

## 使用步骤

1. 打开 **模型 → 模型下载 → Hugging Face**，搜索并下载 `aac6fef/laya-multilingual-mlx`。
2. 在 **模型管理** 中找到 `DECISION` 类型的模型并点击 **加载**。如果当前运行 DFlash2
   独占模型，先在 App 中卸载该模型。
3. 点击该行的齿轮按钮，打开 **模型参数设置**，查看模型别名、`DECISION` 类型和只读的
   Context Size。上下文长度来自模型配置，包含任务说明、选项和待判断内容。
   服务地址可在 **状态** 页面查看。
4. 在外部应用中配置这些地址和精确的模型 ID，即可调用 `POST /v1/systemone`。

默认本机端点为 `http://127.0.0.1:9068/v1/systemone`。TypeSafe 官方 SDK 的基础地址为
`http://127.0.0.1:9068`，不包含 `/v1`。本机访问无需 API Key；要求填写密钥的 SDK
可使用 `local`。若需其他设备访问，在 App 的 **设置** 中配置局域网模式，使用页面显示的
HTTPS 地址和 Bearer API Key。API 服务沿用 App 的网络与认证设置，无需执行终端命令。

加载、卸载、固定模型和重启恢复都使用 App 的现有模型管理机制。卸载时会等待已经进入推理的
请求结束。按需加载和空闲回收遵循模型池策略。此模型不使用聊天采样、流式输出、MTP 或 KV cache。

## 推理设置

只读基础信息下方提供以下设置：

| 设置 | 默认值 | 作用 |
| --- | --- | --- |
| 计算精度 | FP16 | 可选 FP16 或 FP32。权重在内存中转换后参与计算，下载的 FP16 权重文件不变。FP32 占用更多内存。 |
| 问题批大小 | 16 | 每次前向计算处理 1–256 个问题，数值越大内存占用越高。这是单次请求中的问题批大小，不是 API 并发数。 |
| 问题前缀缓存 | 关闭 | 复用 CPU 分词和问题前缀准备结果，最多保留 128 个前缀。不缓存待判断内容或决策结果。 |

默认折叠的**高级设置**包含：

| 设置 | 默认值 | 作用 |
| --- | --- | --- |
| 计算设备 | 自动 | 优先使用可用 GPU，也可显式选择 GPU 或 CPU 推理。 |
| 编译优化 | 关闭 | 编译前向计算图。新输入形状有首次编译开销；速度收益取决于工作负载。 |
| 序列长度对齐 | 关闭 | 开启后按 1–1024 的指定倍数补齐（建议 16），不超过模型上下文上限；补齐部分不计入输入用量。 |
| 问题与选项预算 | 只读 | 从检查点读取 `head_max_len`，占用总上下文空间的一部分。 |
| 概率校准温度 | 只读 | 从检查点读取各任务及选项数量分组的有效校准值，与推理一致限制在 0.5–5.0；不是采样温度。 |

保存后，已加载的模型会在当前请求完成后重载。设置也适用于后续加载与 App 重启恢复。
Context Size 由模型定义，只读且不可修改。

模型管理的加载／注册请求可携带 `decision` 对象：
`{"dtype":"float16","batch_size":16,"cache_prompts":false,"device":"auto","compile":false,"pad_to_multiple":null}`。已加载模型信息会返回实际配置。
这些字段属于模型设置，不属于 System One 推理请求；独立 CLI 使用默认值。

## 状态与决策指标

决策模型加载后，**状态**页面按照决策运行时展示请求活动，不沿用因果模型的文本生成指标：

- **处理中**表示至少有一个决策请求正在执行或排队。
- **刚刚完成**会在最近一次请求结束后短暂保留，使短请求不会在 Dashboard 两次刷新之间消失。
- **空闲**表示当前没有正在执行、排队或刚刚完成的请求。

决策性能区域提供以下指标：

| 指标 | 含义 |
| --- | --- |
| 已完成 | 当前模型实例加载以来成功完成的请求数。 |
| 错误 | 当前模型实例加载以来失败的请求数，包括校验、队列、超时和推理错误。 |
| P50 延迟 | 最近 60 秒窗口内成功请求的端到端延迟中位数，单位为毫秒。 |
| 输入吞吐 | 同一窗口内各成功请求输入 Token 速率的中位数，单位为 Token/秒。 |
| 问题吞吐 | 同一窗口内各成功请求问题处理速率的中位数，单位为问题/秒。 |

在成功请求产生样本前，后三项近期指标显示 `—`；60 秒窗口内不再有样本时也会恢复为
`—`。已完成和错误数在当前模型加载周期内持续累计。卸载后重新加载模型或重启后端，
会开始新的指标统计周期。

App 集成也可以通过 `GET /admin/api/models/loaded` 读取相同数据。已加载的决策模型包含
`runtime_kind: "decision"` 和 `decision_metrics` 对象。字段契约见
[服务与管理 API](service-api.md#app-决策运行时指标)。

## 可选的独立 CLI

开发者仍可使用独立的 `ironmlx serve-systemone` 命令；它默认使用端口 `8767`，
并始终要求 `IRONMLX_SYSTEMONE_API_KEY`。普通 App 用户不需要这一步。
