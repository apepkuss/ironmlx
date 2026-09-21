#!/usr/bin/env python3
"""Generate the App's pinned, data-only preparation recipe from verified fixtures.

Developer utility only. The App copies ZIP storage members without interpreting
pickle or executing Python. Every input and reconstructed output has a fixed hash.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import zipfile

PROFILE = Path(__file__).resolve().parents[1] / "resources/indextts25"
OUTPUT = PROFILE.parents[2] / "ironmlx-app/Sources/IronMLXAppCore/Resources/indextts25-preparation.json"


def checked(path, spec):
    data = path.read_bytes()
    if len(data) != spec["bytes"] or hashlib.sha256(data).hexdigest() != spec["sha256"]:
        raise ValueError(f"unverified input: {path}")
    return data


def generate(snapshot, campplus, derived):
    sources = json.loads((PROFILE / "sources.json").read_text())
    manifest = json.loads((PROFILE / "derived.json").read_text())
    stores = {}
    for role, root, name in [("snapshot", snapshot, "feat1.pt"),
                             ("snapshot", snapshot, "feat2.pt"),
                             ("snapshot", snapshot, "wav2vec2bert_stats.pt"),
                             ("campplus", campplus, "campplus_cn_common.bin")]:
        spec = next(x for x in manifest["inputs"] if x["role"] == role and x["path"] == name)
        checked(root / name, spec)
        with zipfile.ZipFile(root / name) as archive:
            for member in sorted(archive.namelist()):
                if "/data/" not in member:
                    continue
                data = archive.read(member)
                # Duplicate storages with identical bytes are interchangeable.
                stores.setdefault((len(data), hashlib.sha256(data).hexdigest()),
                                  {"role": role, "archive": name, "member": member})
    components = []
    for name, spec in sorted(manifest["components"].items()):
        data = checked(derived / name, spec)
        length = int.from_bytes(data[:8], "little")
        header = data[:8 + length]
        tensors = json.loads(header[8:])
        schema = json.loads((PROFILE / name.replace(".safetensors", ".schema.json")).read_text())
        manifest["components"][name]["tensors"] = schema
        blocks = []
        rebuilt = bytearray(header)
        for tensor, desc in tensors.items():
            if tensor == "__metadata__":
                continue
            start, end = desc["data_offsets"]
            if start != len(rebuilt) - len(header):
                raise ValueError("noncontiguous safetensors layout")
            block = data[len(header) + start:len(header) + end]
            source = stores[(len(block), hashlib.sha256(block).hexdigest())]
            if schema[tensor] != {k: desc[k] for k in ("dtype", "shape")}:
                raise ValueError(f"schema mismatch: {tensor}")
            blocks.append({**source, "tensor": tensor, "bytes": len(block)})
            rebuilt.extend(block)
        assert rebuilt == data
        components.append({"path": name, "header": base64.b64encode(header).decode(), "blocks": blocks})
    return {"version": 1, "sources": sources, "manifest": manifest, "components": components}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("snapshot", "campplus", "derived"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    content = json.dumps(generate(args.snapshot, args.campplus, args.derived),
                         ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if args.check:
        if args.output.read_text() != content:
            raise SystemExit("App resource recipe has drifted")
    else:
        args.output.write_text(content)


if __name__ == "__main__":
    main()
