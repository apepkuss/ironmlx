#!/usr/bin/env python
"""P6.3c Gate 4: semantic functional correctness on 4 test images.

Starts an ironmlx HTTP server, queries it with each image at temperature=0
and enable_thinking=false, applies per-image pass criteria from the P6.3 spec.

Usage:
    MLX_DIR=$HOME/.local/mlx \
    QWEN35_MODEL=/path/to/Qwen3.5-4B-MLX-4bit \
    ~/.venvs/mlxvlm-ref/bin/python item3_semantic_check.py \
        --out /path/to/p6_3c_semantic_report.md
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[4]
FIXTURE_DIR = REPO_ROOT / "ironmlx/tests/fixtures/p6_qwen35_vl"
PROMPT = "Describe this image in detail. If there are multiple people or objects, count them."

IMAGES = [
    {
        "name": "coco_cats",
        "path": FIXTURE_DIR / "coco_sample.jpg",
        "criteria": {
            "type": "key_facts",
            "facts": [
                ["two cats", "2 cats", "two tabby", "two kittens"],
                ["green collar", "collar"],
                ["remote", "remotes"],
            ],
            "min_pass": 2,
        },
    },
    {
        "name": "scene_room",
        "path": Path("/tmp/p6vl_test_imgs/scene.jpg"),
        "criteria": {
            "type": "forbid_keywords",
            "forbid": ["side-by-side", "side by side", "stereoscopic", "composite",
                       "duplicated", "duplicate", "mirrored", "mirror image", "stitched"],
        },
    },
    {
        "name": "counting_kids",
        "path": Path("/tmp/p6vl_test_imgs/counting.jpg"),
        "criteria": {
            "type": "count_in_range",
            "range": [10, 16],
        },
    },
    {
        "name": "text_stop",
        "path": Path("/tmp/p6vl_test_imgs/text.jpg"),
        "criteria": {
            "type": "inversion_keyword",
            "keywords": ["upside down", "upside-down", "rotated 180",
                         "rotated by 180", "POTS", "flipped"],
        },
    },
]


def wait_for_port(port: int, timeout_s: int = 120) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=2):
                return True
        except (ConnectionRefusedError, OSError):
            time.sleep(2)
    return False


def query(port: int, image_path: Path) -> tuple[str, str]:
    """Returns (response_text, finish_reason)."""
    b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    payload = {
        "model": "qwen3_5",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        }],
        "max_tokens": 400,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": False,
    }
    r = requests.post(f"http://127.0.0.1:{port}/v1/chat/completions",
                      json=payload, timeout=600)
    r.raise_for_status()
    body = r.json()
    return body["choices"][0]["message"]["content"], body["choices"][0]["finish_reason"]


def evaluate(text: str, criteria: dict) -> tuple[bool, str]:
    """Returns (passed, note)."""
    t = text.lower()
    if criteria["type"] == "key_facts":
        hits = 0
        details = []
        for fact_synonyms in criteria["facts"]:
            matched = next((s for s in fact_synonyms if s.lower() in t), None)
            if matched is not None:
                hits += 1
                details.append(f"✓ {matched}")
            else:
                details.append(f"✗ {fact_synonyms[0]}")
        return hits >= criteria["min_pass"], f"{hits}/{len(criteria['facts'])} ({'; '.join(details)})"
    if criteria["type"] == "forbid_keywords":
        for fk in criteria["forbid"]:
            if fk.lower() in t:
                return False, f"contains forbidden term '{fk}'"
        return True, "no forbidden keywords"
    if criteria["type"] == "count_in_range":
        nums = [int(n) for n in re.findall(r"\b(\d+)\b", text)]
        lo, hi = criteria["range"]
        ok_nums = [n for n in nums if lo <= n <= hi]
        return bool(ok_nums), f"numbers in response: {nums}; in [{lo},{hi}]: {ok_nums}"
    if criteria["type"] == "inversion_keyword":
        for kw in criteria["keywords"]:
            if kw.lower() in t:
                return True, f"matched '{kw}'"
        return False, "no inversion keyword"
    raise ValueError(f"unknown criteria type {criteria['type']}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--port", type=int, default=8082)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    model_dir = os.environ.get("QWEN35_MODEL")
    mlx_dir = os.environ.get("MLX_DIR")
    if not model_dir or not mlx_dir:
        print("ERROR: set MLX_DIR and QWEN35_MODEL", file=sys.stderr)
        return 1

    # Kill any leftover server first
    subprocess.run(["pkill", "-KILL", "-f", "ironmlx serve"], check=False)
    time.sleep(2)

    # Start ironmlx server
    server_log = open("/tmp/p6_3c_server.log", "w")
    env = dict(os.environ)
    env["MLX_DIR"] = mlx_dir
    server = subprocess.Popen(
        [str(REPO_ROOT / "target/release/ironmlx"), "serve",
         "--model", model_dir,
         "--host", "127.0.0.1",
         "--port", str(args.port)],
        env=env, stdout=server_log, stderr=subprocess.STDOUT,
    )
    try:
        if not wait_for_port(args.port, timeout_s=180):
            print("ERROR: server failed to start; see /tmp/p6_3c_server.log", file=sys.stderr)
            return 2

        results = []
        for spec in IMAGES:
            print(f"[item3] querying {spec['name']}...")
            text, finish = query(args.port, spec["path"])
            passed, note = evaluate(text, spec["criteria"])
            results.append({
                "name": spec["name"],
                "criteria_type": spec["criteria"]["type"],
                "passed": passed,
                "note": note,
                "finish_reason": finish,
                "response": text,
            })
            print(f"  → {'PASS' if passed else 'FAIL'}: {note}")
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
        server_log.close()

    n_pass = sum(1 for r in results if r["passed"])
    lines = [
        "# P6.3c Semantic Verification (Gate 4)",
        "",
        f"- Images tested: {len(results)}",
        f"- Passed: {n_pass} / {len(results)}",
        f"- Gate 4 threshold: ≥ 3 / 4 → **{'PASS' if n_pass >= 3 else 'FAIL'}**",
        "",
        "## Per-image",
        "",
    ]
    for r in results:
        lines.append(f"### {r['name']} — {'✅ PASS' if r['passed'] else '❌ FAIL'}")
        lines.append("")
        lines.append(f"- criterion: `{r['criteria_type']}`")
        lines.append(f"- finish_reason: `{r['finish_reason']}`")
        lines.append(f"- note: {r['note']}")
        lines.append("")
        lines.append("Response:")
        lines.append("```")
        lines.append(r["response"])
        lines.append("```")
        lines.append("")
    args.out.write_text("\n".join(lines))
    print(f"[item3] report → {args.out}")
    return 0 if n_pass >= 3 else 1


if __name__ == "__main__":
    sys.exit(main())
