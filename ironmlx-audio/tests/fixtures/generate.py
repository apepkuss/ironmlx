"""Regenerate independent reference fixtures; never imported by production code."""

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import types

HERE = Path(__file__).resolve().parent


def require_versions(packages):
    pins = dict(
        line.split("==")
        for line in (HERE / "requirements.txt").read_text().splitlines()
        if "==" in line
    )
    for package in packages:
        actual = importlib.metadata.version(package)
        if actual != pins[package]:
            raise ValueError(f"{package}: expected {pins[package]}, got {actual}")
    return {package: pins[package] for package in packages}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")


def signal_fixtures(output):
    require_versions(["torch", "torchaudio", "numpy", "soundfile"])
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio

    cases = []
    for source, target, length in [
        (8000, 16000, 37), (44100, 16000, 239), (48000, 22050, 237),
        (22050, 16000, 131), (96000, 16000, 195), (16000, 22050, 79),
        (8001, 22050, 61),
    ]:
        x = np.array([
            math.sin(i * 0.43) * 0.3 + math.cos(i * 0.09) * 0.2
            for i in range(length)
        ], dtype=np.float32)
        y = torchaudio.functional.resample(torch.from_numpy(x), source, target).numpy()
        cases.append(dict(source_rate=source, target_rate=target,
                          input=x.tolist(), output=y.tolist()))
    write_json(output / "resample.json", {
        "reference": "torchaudio 2.10.0 default sinc_interp_hann lowpass_filter_width=6 rolloff=0.99",
        "cases": cases,
    })
    x = np.array([int(12000 * math.sin(i * 0.1)) for i in range(8000)], dtype=np.int16)
    for subtype in ["PCM_16", "PCM_24", "PCM_32", "FLOAT"]:
        sf.write(output / f"reference-{subtype}.wav", x.astype(np.float32) / 32768,
                 8000, subtype=subtype)
    sf.write(output / "reference.flac", x, 8000, subtype="PCM_16")
    sf.write(output / "reference.mp3", x.astype(np.float32) / 32768, 8000, format="MP3")


def text_fixtures(output, reference, snapshot):
    versions = require_versions(["tiktoken", "wetext", "fugashi", "unidic-lite", "kaldifst"])
    require_versions(["contractions", "PyYAML"])
    sources = json.loads((HERE / "reference-sources.json").read_text())
    for name, digest in sources["files"].items():
        if hashlib.sha256((reference / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"reference source mismatch: {name}")
    # Avoid importing the upstream package initializer and its inference dependencies.
    package = types.ModuleType("mlx_indextts")
    package.__path__ = [str(reference)]
    sys.modules["mlx_indextts"] = package
    from mlx_indextts.normalizer_v25 import (
        IndexTTS25TextFrontend, IndexTTS25TextNormalizer, OptionalNemoNormalizer,
    )
    from mlx_indextts.normalize import TextNormalizer

    def no_nemo(language):
        raise ImportError("NeMo is disabled by the fixed reference profile")

    frontend = IndexTTS25TextFrontend(snapshot, normalizer=IndexTTS25TextNormalizer(
        zh_en_normalizer=TextNormalizer(enable_glossary=False),
        nemo_normalizer=OptionalNemoNormalizer(factory=no_nemo),
    ))
    cases = []
    for case in json.loads((HERE / "text.json").read_text())["cases"]:
        prepared = frontend.prepare(case["input"], language=case["requested_language"])
        result = dataclasses.asdict(prepared)
        result.update(input=case["input"], requested_language=case["requested_language"])
        result["canonical_token_ids"] = [
            [0, *[token for token in sequence if token not in (0, 1)], 1]
            for sequence in prepared.token_ids
        ]
        cases.append(result)
    write_json(output / "text.json", {
        "reference_revision": sources["revision"], "versions": versions, "cases": cases,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["signal", "text"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, help="Fixed upstream mlx_indextts source directory")
    parser.add_argument("--snapshot", type=Path, help="Pinned IndexTTS 2.5 snapshot directory")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.kind == "signal":
        signal_fixtures(args.output)
    elif args.reference is None or args.snapshot is None:
        parser.error("text generation requires --reference and --snapshot")
    else:
        text_fixtures(args.output, args.reference.resolve(), args.snapshot.resolve())


if __name__ == "__main__":
    main()
