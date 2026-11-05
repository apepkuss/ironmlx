# 离线音频资源与数值对齐工具

[English](README.md)

这些开发工具用于准备 IndexTTS 资源，以及生成 IndexTTS 和 Breeze TTS 2 的
数值对齐基线。Rust 库和服务端都不会启动 Python。请提供已有的本地资源；
工具不会自行下载。两种模型应使用下述各自的独立依赖环境。

## IndexTTS 资源转换

IndexTTS 使用 Python 3.11 独立环境及 `requirements.txt`。
源快照标识固定在 `../resources/indextts25/sources.json` 中。

```sh
python convert_indextts25.py --snapshot "$SNAPSHOT" \
  --campplus "$CAMPPLUS" --w2v "$W2V" --output "$DERIVED"
python -m unittest discover -s . -p test_conversion.py -v
```

`DERIVED` 必须尚不存在。转换会检查全部输入哈希，不做类型转换地序列化
941 个张量，并逐字节比较恢复后的张量。浮点正负零和整数批次计数均保留；
原始 PyTorch 张量布局保持不变。统计量保留方差而不是标准差。
全部 CAMPPlus 批次计数均保留，并明确标记为推理时不用。
快照中的 w2v-BERT 权重直接复用，不复制或下载。

输出包含两个 safetensors 文件、复制的声明和配置，以及记录工具、源文件、
输出哈希、schema 和键映射的 manifest。同级锁用于串行化配合该锁的写入；
文件先暂存并校验，再通过目录原子重命名发布。转换失败会清理本次暂存目录和锁。
请勿修改不可变快照，也不要替换已有派生目录。Rust 加载器检查编译时固定的
输出标识，包括声明和配置；修改 manifest 不能授权不同的权重。

## IndexTTS App 准备配置

App 不依赖 Python 来准备这些资源。内置的纯数据配方把每个输出张量映射到
哈希固定的 PyTorch ZIP 存储成员，不解释 pickle。源归档和重建的 safetensors
必须匹配 Rust 加载器检查的标识。App 配方不内嵌模型权重。

重新生成并验证离线产物后，使用仅依赖 Python 标准库的工具更新或检查配方：

```sh
python generate_app_resources.py --snapshot "$SNAPSHOT" --campplus "$CAMPPLUS" --derived "$DERIVED"
python generate_app_resources.py --snapshot "$SNAPSHOT" --campplus "$CAMPPLUS" --derived "$DERIVED" --check
```

生成文件位于 App 的 `Resources/indextts25-preparation.json`。
App 测试将其中的资源标识与原生库配置对照；显式启用的真实资源测试会通过
哈希检查每个重建产物的完整字节。

## IndexTTS 数值基线生成

使用 `vanch007/mlx-indextts2` 的干净 checkout，固定提交
`a7666367b8551656a2029ad75f259cb5e4936b3b`，并将其路径设为 `REFERENCE`。
真实语音、大型基线、环境和进程报告应存放在受版本控制的源码之外。

```sh
python reference_features.py --reference "$REFERENCE" --snapshot "$SNAPSHOT" \
  --w2v "$W2V" --audio "$SPEECH_FILE" --output "$FEATURES"
MLX_ENABLE_TF32=0 python reference_encoders.py --reference "$REFERENCE" \
  --snapshot "$SNAPSHOT" --w2v "$W2V" --campplus "$CAMPPLUS" \
  --features "$FEATURES" --case speech_1s --output "$OUTPUT_DIRECTORY/encoders-fp32-1s.safetensors"
```

另以 `speech_odd` / `encoders-fp32-odd.safetensors` 和
`speech_5s` / `encoders-fp32-5s.safetensors` 重复编码器命令。
PyTorch 编码器运行在 CPU 上；MLX 条件编码关闭 TF32。
参考工具写出第 17 层隐藏状态、归一化语义特征、风格、情绪、GPT 条件及
长度调节器输出，不运行语音合成。

重新生成受版本控制的合成回归基线时，将 `--audio` 替换为 `--synthetic`，
并在替换前检查 safetensors 和 JSON。合成输入是 chirp 加带 seed 的噪声，
不包含真实语音数据。`generate_feature_windows.py --output PATH` 使用
PyTorch 2.10.0 重新生成固定 float32 Hann 和 Povey 系数；保留这些系数可避免
主机数学库差异经对数 mel 特征放大。

`generate_resample_kernel.py --output PATH` 重新生成通用参考转换使用的
16 kHz 到 22.05 kHz 紧凑 float32 sinc 核。其他采样率组合在运行时使用同一
float32 算法计算。稀疏核仅省略被截断的支持区尾部，即绝对值小于 1e-15 的系数。

## Breeze 网络数值对齐

`reference_breeze.py` 接收完整的本地快照，以及 Blaizzy/mlx-audio 固定提交
`b1db0f21c1767b40b3496927f12df67495f0088b` 中的 `breeze_tts.py` 和 `config.py`。
运行前会检查源文件的 SHA-256 标识。使用 Python MLX 0.32.4、mlx-lm 0.28.2
及 transformers 4.57.3；它们只用于离线验证，不是推理运行时依赖。

```sh
MLX_ENABLE_TF32=0 python reference_breeze.py \
  --snapshot /absolute/path/to/snapshot \
  --source /absolute/path/to/breeze_tts.py \
  --config-source /absolute/path/to/config.py \
  --output /absolute/path/to/network-reference.safetensors
```

BF16 和 MXFP8 分别生成基线。模型张量和生成音频应放在受版本控制的基线之外。

### Breeze codec 数值对齐

`reference_breeze_codec.py` 检查参考音频编码、完整波形解码和保留状态的流式
codec 解码。使用同一固定 mlx-audio 提交中的 `speech_tokenizer.py`，并检查其
SHA-256 标识。除 MLX 0.32.4 和 mlx-lm 0.28.2 外，环境还需 mlx-audio 的
Mimi 和 tokenizer 配置模块。IndexTTS 的 `requirements.txt` 不适用于 Breeze 环境。

```sh
MLX_ENABLE_TF32=0 python reference_breeze_codec.py \
  /absolute/path/to/snapshot /absolute/path/to/codec-reference.safetensors \
  --source /absolute/path/to/speech_tokenizer.py --frames 80 --chunk-frames 4
```

默认基线将 80 个音频码帧分成 20 块，跨越解码器的 72 帧注意力窗口。
设置 `IRONMLX_BREEZE_SNAPSHOT` 和 `IRONMLX_BREEZE_CODEC_REFERENCE` 后，
运行原生数值比较：

```sh
MLX_ENABLE_TF32=0 cargo test -p ironmlx-audio --lib breeze_tts::codec::tests::real_codec_numerical_parity -- --ignored --nocapture
```
