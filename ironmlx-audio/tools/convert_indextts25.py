#!/usr/bin/env python3
"""Convert pinned IndexTTS auxiliary resources offline, without changing snapshots.

Python/PyTorch are conversion tools only; inference loads the resulting safetensors.
The output directory must not exist. Files are verified before atomic publication.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import tempfile

import torch
from safetensors.torch import load_file, save_file

PROFILE = Path(__file__).resolve().parents[1] / "resources" / "indextts25"
RECIPE = "indextts25-native-resources-v1"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path, name):
    return {"path": name, "bytes": path.stat().st_size, "sha256": sha256(path)}


def check_file(path, expected):
    actual = file_record(path, expected["path"])
    if any(actual[key] != expected[key] for key in ("bytes", "sha256")):
        raise ValueError(f"resource mismatch: {path}")
    return actual


def schema(tensors):
    dtypes = {torch.float32: "F32", torch.int64: "I64"}
    return {name: {"dtype": dtypes[value.dtype], "shape": list(value.shape)}
            for name, value in sorted(tensors.items())}


def validate_tensors(tensors, expected):
    if schema(tensors) != expected:
        raise ValueError("tensor names, shapes or dtypes differ from the fixed schema")
    for name, value in tensors.items():
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"non-finite tensor: {name}")
        if name.endswith("running_var") and (value < 0).any():
            raise ValueError(f"negative batch normalization variance: {name}")
    if "w2v_var" in tensors and (tensors["w2v_var"] <= 0).any():
        raise ValueError("w2v variance must be strictly positive")


def write_component(directory, name, tensors, expected):
    tensors = {key: value.detach().cpu().contiguous() for key, value in tensors.items()}
    validate_tensors(tensors, expected)
    path = directory / name
    # A single metadata entry avoids map-order-dependent header bytes.
    save_file(tensors, str(path), metadata={"format": "pt"})
    restored = load_file(str(path), device="cpu")
    validate_tensors(restored, expected)
    for key, original in tensors.items():
        # Byte comparison also preserves signed zero and every integer bit.
        if original.numpy().tobytes() != restored[key].numpy().tobytes():
            raise ValueError(f"conversion changed tensor bits: {name}:{key}")
    record = file_record(path, name)
    return {"name": record.pop("path"), **record, "tensors": expected}


def convert(snapshot, campplus, w2v, output):
    profile = json.loads((PROFILE / "sources.json").read_text())
    sources = {"snapshot": (snapshot, profile["source"])}
    for role, root, repository in [
        ("campplus", campplus, "funasr/campplus"),
        ("w2v", w2v, "facebook/w2v-bert-2.0"),
    ]:
        spec = next(item for item in profile["auxiliary_sources"]
                    if item["repository"] == repository)
        sources[role] = (root, spec)
    names = {
        "snapshot": ["feat1.pt", "feat2.pt", "wav2vec2bert_stats.pt",
                     "model.safetensors", "LICENSE", "README.md"],
        "campplus": [item["path"] for item in sources["campplus"][1]["files"]],
        "w2v": [item["path"] for item in sources["w2v"][1]["files"]],
    }
    inputs = []
    for role, filenames in names.items():
        root, spec = sources[role]
        for name in filenames:
            expected = next(item for item in spec["files"] if item["path"] == name)
            inputs.append({"role": role, **check_file(root / name, expected)})
    input_digest = hashlib.sha256(json.dumps(inputs, sort_keys=True,
                                           separators=(",", ":")).encode()).hexdigest()
    if output.exists():
        raise FileExistsError(f"output must not exist: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Cooperative lock prevents concurrent converters publishing the same destination.
    lock = output.with_name(output.name + ".conversion-lock")
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    stage = None
    try:
        os.close(fd)
        stage = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
        stats = torch.load(snapshot / "wav2vec2bert_stats.pt", map_location="cpu",
                           weights_only=True)
        if set(stats) != {"mean", "var"}:
            raise ValueError("unexpected w2v statistics keys")
        auxiliary = {
            "speaker_matrix": torch.load(snapshot / "feat1.pt", map_location="cpu",
                                         weights_only=True),
            "emotion_matrix": torch.load(snapshot / "feat2.pt", map_location="cpu",
                                         weights_only=True),
            "w2v_mean": stats["mean"], "w2v_var": stats["var"],
        }
        camp = torch.load(campplus / "campplus_cn_common.bin", map_location="cpu",
                          weights_only=True)
        components = {}
        for name, tensors, schema_name in [
            ("auxiliary.safetensors", auxiliary, "auxiliary.schema.json"),
            ("campplus.safetensors", camp, "campplus.schema.json"),
        ]:
            expected = json.loads((PROFILE / schema_name).read_text())
            components[name] = write_component(stage, name, tensors, expected)
        files = []
        for role, filename, dest in [
            ("snapshot", "LICENSE", "notices/main/LICENSE"),
            ("snapshot", "README.md", "notices/main/README.md"),
            ("campplus", "README.md", "notices/campplus/README.md"),
            ("campplus", "config.yaml", "notices/campplus/config.yaml"),
            ("campplus", "configuration.json", "notices/campplus/configuration.json"),
            ("w2v", "README.md", "notices/w2v-bert/README.md"),
            ("w2v", "config.json", "w2v-bert/config.json"),
            ("w2v", "preprocessor_config.json", "w2v-bert/preprocessor_config.json"),
        ]:
            target = stage / dest
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(sources[role][0] / filename, target)
            record = file_record(target, dest)
            original = next(item for item in inputs
                            if item["role"] == role and item["path"] == filename)
            if any(record[key] != original[key] for key in ("bytes", "sha256")):
                raise ValueError(f"copy changed resource: {dest}")
            files.append(record)
        manifest = {
            "schema_version": 1, "family": "indextts2_5", "recipe": RECIPE,
            "source_revision": profile["source"]["revision"],
            "input_digest": input_digest, "inputs": inputs,
            "tool": {"sha256": sha256(Path(__file__)), "python": platform.python_version(),
                     "torch": importlib.metadata.version("torch"),
                     "safetensors": importlib.metadata.version("safetensors"),
                     "profile_sha256": sha256(PROFILE / "sources.json")},
            "components": components, "files": files,
            "reuse_weight": {"role": "snapshot", "path": "model.safetensors",
                             **{key: profile["auxiliary_sources"][0]["reuse_weight"][key]
                                for key in ("sha256", "bytes")}},
            "mapping": {"speaker_matrix": "snapshot:feat1.pt",
                        "emotion_matrix": "snapshot:feat2.pt",
                        "w2v_mean": "snapshot:wav2vec2bert_stats.pt:mean",
                        "w2v_var": "snapshot:wav2vec2bert_stats.pt:var",
                        "campplus.safetensors": "identity; all tensor names and layouts preserved"},
            "allowed_unused": {"campplus.safetensors": sorted(
                key for key in camp if key.endswith(".num_batches_tracked"))},
        }
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        # Detect source changes during conversion before publishing a complete directory.
        for item in inputs:
            check_file(sources[item["role"]][0] / item["path"], item)
        for path in stage.rglob("*"):
            if path.is_file():
                with path.open("rb") as handle:
                    os.fsync(handle.fileno())
        if output.exists():
            raise FileExistsError(f"output already exists: {output}")
        stage.rename(output)
        stage = None
        print(json.dumps({"output": str(output), "input_digest": input_digest,
                          "tensors": sum(len(x["tensors"]) for x in components.values())}))
    finally:
        if stage is not None:
            shutil.rmtree(stage)
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("snapshot", "campplus", "w2v", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    convert(args.snapshot.resolve(), args.campplus.resolve(), args.w2v.resolve(),
            args.output.resolve())


if __name__ == "__main__":
    main()
