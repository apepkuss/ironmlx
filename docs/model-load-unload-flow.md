# 模型加载/卸载流程说明

## 加载模型流程

```mermaid
flowchart TD
    A["用户点击 [Load]"] --> B["按钮变为 Loading... (disabled)"]
    B --> C["JS → WKScriptMessageHandler → Rust loadModel handler"]
    C --> D["估算模型大小\n扫描 ~/.ironmlx/models/ 下的文件"]
    D --> E["获取当前 GPU 可用内存\nGET /health → memory.active_mb"]
    E --> F{内存是否充足?}

    F -->|充足| G["POST /v1/models/load"]
    G --> H["EnginePool.load_model()"]
    H --> H1["1. 从磁盘读取模型权重 + tokenizer + config"]
    H1 --> H2["2. 创建推理引擎 EngineCore + KV Cache"]
    H2 --> H3["3. 启动独立推理线程"]
    H3 --> H4["4. 注册到 engines HashMap"]
    H4 --> I["Rust → evaluateJavaScript → JS onModelLoaded"]
    I --> J["__LOADED_MODELS__.add(modelId)\nStatus → Loaded, 按钮 → Unload"]

    F -->|不足| K["返回 warning 给 JS"]
    K --> L["弹出 confirm 警告对话框"]
    L -->|用户取消| M["恢复按钮状态"]
    L -->|用户继续| N["发送 forceLoadModel\n跳过检测直接加载"]
    N --> G
```

## 卸载模型流程

```mermaid
flowchart TD
    A["用户点击 [Unload]"] --> B["按钮变为 Unloading... (disabled)"]
    B --> C["JS → WKScriptMessageHandler → Rust unloadModel handler"]
    C --> D["POST /v1/models/unload"]
    D --> E["EnginePool.unload_model()"]
    E --> E1["1. 从 engines HashMap 移除该模型"]
    E1 --> E2["2. 发送 shutdown 命令给推理引擎线程"]
    E2 --> E3["3. 释放 GPU 内存（模型权重 + KV Cache）"]
    E3 --> E4["4. 如果是默认模型，清除 default_model"]
    E4 --> F["Rust → evaluateJavaScript → JS onModelUnloaded"]
    F --> G["__LOADED_MODELS__.delete(modelId)\nStatus → Ready, 按钮 → Load"]
```

## 初始化同步

```mermaid
flowchart LR
    A["页面加载"] --> B["syncLoadedModels()"]
    B --> C["Rust: GET /v1/models"]
    C --> D["获取已加载模型 ID 列表"]
    D --> E["同步到 __LOADED_MODELS__"]
    E --> F["表格渲染时正确显示状态"]
```

## 关键代码路径

| 组件 | 文件 | 说明 |
|------|------|------|
| 前端 JS | `ironmlx-app/src/dashboard2.html` | toggleModelLoad, loadModel, unloadModel, syncLoadedModels |
| Rust Bridge | `ironmlx-app/src/web_dashboard.rs` | loadModel/forceLoadModel/unloadModel/syncLoadedModels 消息处理 |
| 后端 API | `ironmlx/src/api/models.rs` | POST /v1/models/load, POST /v1/models/unload |
| 引擎池 | `ironmlx/src/engine_pool.rs` | EnginePool.load_model(), EnginePool.unload_model() |

## GPU 内存检测

加载前会进行内存预检：

1. **估算模型大小** — 扫描 `~/.ironmlx/models/` 下模型目录的文件总大小
2. **获取可用内存** — 通过 `/health` API 获取当前 `active_mb`，用总内存减去活跃内存
3. **比较** — 如果模型大小 > 可用内存，弹出警告（但不阻止加载）

注意：Apple Silicon 的统一内存管理较灵活，估算仅作参考。
