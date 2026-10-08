#!/usr/bin/env python3
"""Regenerate audio goldens using pinned upstream code, without IronMLX inference.

Requires Python 3.11, mlx==0.32.3, numpy==1.26.4, tokenizers==0.23.2 and
Pillow==12.3.0. Source files and checkpoints must already exist locally.
See ironmlx-lm/tests/fixtures/embedding_gemma2/README.md for commands and provenance.
"""
import argparse
import ast
from collections.abc import Sequence
import dataclasses
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import sys
import types
import warnings
import wave

MLX_VLM_REVISION = "3d87e88402f307efbf68e568971aa887ee7d9ed0"
TRANSFORMERS_REVISION = "cb33194ad6152bd9fad6305378d92db385dd7b32"
FIXTURES = Path(__file__).resolve().parents[1] / "ironmlx-lm/tests/fixtures/embedding_gemma2"


def upstream_modules(root):
    """Import the original model files, stubbing unrelated package dependencies.

    Only BaseModelConfig's dataclass field filtering is supplied here. The audio
    encoder, language encoder, pooling and projector are executed from upstream.
    """
    import mlx.core as mx
    import mlx.nn as nn

    for name in ["mlx_vlm", "mlx_vlm.models", "mlx_vlm.models.gemma4", "mlx_vlm.models.embedding_gemma2"]:
        module = types.ModuleType(name)
        module.__path__ = [str(root / name.replace(".", "/"))]
        sys.modules[name] = module
    base = types.ModuleType("mlx_vlm.models.base")

    class BaseModelConfig:
        @classmethod
        def from_dict(cls, values):
            fields = {field.name for field in dataclasses.fields(cls)}
            return cls(**{key: value for key, value in values.items() if key in fields})

    base.BaseModelConfig = BaseModelConfig
    sys.modules[base.__name__] = base
    language = types.ModuleType("mlx_vlm.models.gemma4.language")
    language.__dict__.update(mx=mx, nn=nn)
    source = (root / "mlx_vlm/models/gemma4/language.py").read_text()
    exec(source[source.index("class RMSNormNoScale"):source.index("class RMSNormZeroShift")], language.__dict__)
    sys.modules[language.__name__] = language
    projector = types.ModuleType("mlx_vlm.models.gemma4.gemma4")
    projector.__dict__.update(mx=mx, nn=nn, RMSNormNoScale=language.RMSNormNoScale)
    source = (root / "mlx_vlm/models/gemma4/gemma4.py").read_text()
    exec(source[source.index("class MultimodalEmbedder"):source.index("class Model(")], projector.__dict__)
    sys.modules[projector.__name__] = projector
    from mlx_vlm.models.embedding_gemma2.config import ModelConfig
    from mlx_vlm.models.embedding_gemma2.embedding_gemma2 import Model
    return ModelConfig, Model


def upstream_extractor(root):
    """Execute the official extractor and helpers; omit its transport imports."""
    import numpy as np
    namespace = dict(np=np, math=math, warnings=warnings, Sequence=Sequence)
    helpers = {"hertz_to_mel", "mel_to_hertz", "_create_triangular_filter_bank", "mel_filter_bank", "window_function"}
    path = root / "src/transformers/audio_utils.py"
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in helpers]
    exec(compile(tree, str(path), "exec"), namespace)

    class SequenceFeatureExtractor:
        def __init__(self, **values):
            self.__dict__.update(values)

    namespace.update(SequenceFeatureExtractor=SequenceFeatureExtractor, PaddingStrategy=object,
                     TensorType=object, BatchFeature=object)
    path = root / "src/transformers/models/gemma4/feature_extraction_gemma4.py"
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace["Gemma4AudioFeatureExtractor"]()


def features(extractor, path):
    import numpy as np
    with wave.open(str(path)) as handle:
        if (handle.getframerate(), handle.getnchannels(), handle.getsampwidth()) != (16000, 1, 2):
            raise ValueError(f"{path}: reference fixtures must be mono 16 kHz PCM16")
        audio = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2").astype(np.float32) / 32768
    padding = (-len(audio)) % 128
    samples = np.pad(audio, (0, padding))
    mask = np.pad(np.ones(len(audio), dtype=np.int64), (0, padding))
    mel, mask = extractor._extract_spectrogram(samples[None, :], mask)
    return mel.astype(np.float32) * mask[:, None], mask


def encode(ModelConfig, Model, extractor, path, inputs):
    import mlx.core as mx
    import mlx.nn as nn
    from tokenizers import Tokenizer
    config = json.loads((path / "config.json").read_text())
    config["vision_config"] = None  # These goldens do not contain image inputs.
    model = Model(ModelConfig.from_dict(config))
    weights = {}
    for file in sorted(path.glob("*.safetensors")):
        weights.update(mx.load(str(file)))
    weights = model.sanitize(weights)
    nn.quantize(model, group_size=64, bits=4, class_predicate=lambda name, module: name + ".scales" in weights)
    model.load_weights(list(weights.items()), strict=True)
    model.eval()
    mx.eval(model.parameters())
    tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
    outputs = []
    for case in inputs:
        prompt, media, masks = "", [], []
        for part in case["content"]:
            if "text" in part:
                prompt += part["text"]
            else:
                mel, mask = features(extractor, FIXTURES / "audio" / part["audio"])
                count = int(mask[::4].sum())
                prompt += "<|audio>" + "<|audio|>" * count + "<audio|>"
                media.append(mx.array(mel[None]))
                masks.append(mx.array(mask[None]))
        ids = tokenizer.encode(prompt, add_special_tokens=True).ids
        if not media:
            output = model(mx.array([ids])).text_embeds
        elif len(media) == 1:
            output = model(mx.array([ids]), input_features=media[0], input_features_mask=masks[0]).text_embeds
        else:
            # The public upstream call accepts one clip; ordered multiple clips
            # use its original get_audio_features, scatter, text model and pool.
            from mlx_vlm.models.pooling import mean_pooling, normalize_embeddings
            tokens = mx.array([ids])
            embeddings = model.language_model.embed_tokens(mx.where(tokens == config["audio_token_id"], 0, tokens))
            embeddings *= mx.array(512 ** 0.5, dtype=embeddings.dtype)
            projected = mx.concatenate([model.get_audio_features(f, m) for f, m in zip(media, masks)], axis=0)
            embeddings = model._scatter(embeddings, tokens, config["audio_token_id"], projected)
            mask = mx.ones((1, len(ids)))
            output = normalize_embeddings(mean_pooling(model.language_model(embeddings, mask), mask))
        mx.eval(output)
        outputs.append(dict(tokens=len(ids), embedding=output[0].tolist()))
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlx-vlm-dir", required=True, type=Path)
    parser.add_argument("--transformers-dir", required=True, type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--precision", choices=["bf16", "4bit"])
    parser.add_argument("--features-only", action="store_true")
    parser.add_argument("--check", action="store_true", help="verify existing goldens without rewriting")
    args = parser.parse_args()
    if not args.features_only and (args.model_dir is None or args.precision is None):
        parser.error("model-dir and precision are required unless features-only")
    for package, expected in {"mlx": "0.32.3", "numpy": "1.26.4", "tokenizers": "0.23.2", "Pillow": "12.3.0"}.items():
        if version(package) != expected:
            parser.error(f"{package} must be {expected}")
    sources = json.loads((FIXTURES / "audio-source-hashes.json").read_text())
    for family, files in sources.items():
        root = args.mlx_vlm_dir if family == "mlx_vlm" else args.transformers_dir
        for name, expected in files.items():
            actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
            if actual != expected:
                parser.error(f"{family}/{name} differs from the pinned upstream source")
    document = json.loads((FIXTURES / "audio-reference.json").read_text())
    if document["mlx_vlm_revision"] != MLX_VLM_REVISION or document["transformers_revision"] != TRANSFORMERS_REVISION:
        parser.error("fixture source revisions do not match this generator")
    for name, expected in document["audio_sha256"].items():
        if hashlib.sha256((FIXTURES / "audio" / name).read_bytes()).hexdigest() != expected:
            parser.error(f"audio fixture hash changed: {name}")
    extractor = upstream_extractor(args.transformers_dir)
    records = []
    for name in ["sunny.wav", "tone.wav", "silence.wav", "short.wav", "boundary.wav"]:
        mel, mask = features(extractor, FIXTURES / "audio" / name)
        rows = sorted({0, len(mask) // 2, len(mask) - 1})
        records.append(dict(audio=name, frames=len(mask), valid_frames=int(mask.sum()), tokens=int(mask[::4].sum()),
                            rows=[dict(frame=i, values=mel[i].tolist()) for i in rows]))
    feature_path = FIXTURES / "audio-features-reference.json"
    if args.check:
        if records != json.loads(feature_path.read_text()):
            parser.error("official audio feature goldens have changed")
    else:
        feature_path.write_text(json.dumps(records, indent=2) + "\n")
    if not args.features_only:
        config = json.loads((args.model_dir / "config.json").read_text())
        quantized = config.get("quantization") or config.get("quantization_config")
        if bool(quantized) != (args.precision == "4bit"):
            parser.error("checkpoint precision does not match precision argument")
        ModelConfig, Model = upstream_modules(args.mlx_vlm_dir)
        output = encode(ModelConfig, Model, extractor, args.model_dir, document["inputs"])
        if args.check:
            import numpy as np
            for actual, expected in zip(output, document[args.precision], strict=True):
                if actual["tokens"] != expected["tokens"] or not np.allclose(actual["embedding"], expected["embedding"], rtol=0, atol=1e-7):
                    parser.error("upstream audio embedding goldens have changed")
        else:
            document[args.precision] = output
            (FIXTURES / "audio-reference.json").write_text(json.dumps(document, indent=2) + "\n")
    print("Pinned upstream audio references verified" if args.check else "Pinned upstream audio references regenerated")


if __name__ == "__main__":
    main()
