#!/usr/bin/env python
"""P6.6 Gate 4: 2-image semantic correctness.

Starts the ironmlx HTTP server, queries it with ONE chat request that
includes BOTH images as image_url parts, and verifies the response
text contains key facts from each image.

Per-image criteria: >= 2 / 3 keys must match per image.

Usage:
    MLX_DIR=$HOME/.local/mlx \\
    QWEN35_MODEL=/path/to/model \\
    ~/.venvs/mlxvlm-ref/bin/python p6_6_semantic_check.py \\
        --out /path/to/p6_6_semantic_report.md
"""
from __future__ import annotations

import argparse
import base64
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[4]
FIXTURE_DIR = REPO_ROOT / "ironmlx/tests/fixtures/p6_qwen35_vl"
MULTI_DIR = FIXTURE_DIR / "multi_image"
PROMPT = (
    "There are two images. Describe each one separately. "
    "For each image, mention the key objects and what is happening."
)

# Per-image keys based on the actual fixture content:
#   image_0: kitchen with hanging copper pots/pans + person in apron + dough prep
#   image_1: NYC street scene with construction scaffolding + pedestrians
KEYS_PER_IMAGE = {
    0: [
        ["kitchen", "cooking", "culinary"],
        ["pot", "pots", "pan", "pans", "cookware", "copper"],
        ["person", "people", "chef", "cook", "apron", "man", "woman"],
    ],
    1: [
        ["street", "sidewalk", "city", "urban", "outdoor", "outdoors"],
        ["construction", "scaffolding", "scaffold", "barrier", "fence"],
        ["person", "people", "pedestrian", "walking", "woman", "man", "men"],
    ],
}
MIN_KEYS_PER_IMAGE = 2  # >= 2 / 3 per image


def wait_for_port(port: int, timeout_s: int = 180) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=2):
                return True
        except (ConnectionRefusedError, OSError):
            time.sleep(2)
    return False


def evaluate_per_image(text: str) -> dict:
    t = text.lower()
    per_image_results = {}
    for i, key_groups in KEYS_PER_IMAGE.items():
        hits = []
        for synonyms in key_groups:
            matched = next((s for s in synonyms if s.lower() in t), None)
            hits.append(matched)
        n_hit = sum(1 for h in hits if h is not None)
        per_image_results[i] = {
            "n_hit": n_hit,
            "n_total": len(key_groups),
            "hits": hits,
            "passed": n_hit >= MIN_KEYS_PER_IMAGE,
        }
    return per_image_results


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

    subprocess.run(["pkill", "-KILL", "-f", "ironmlx serve"], check=False)
    time.sleep(2)

    log_path = Path("/tmp/p6_6_server.log")
    server_log = log_path.open("w")
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
            print(f"ERROR: server failed to start; see {log_path}", file=sys.stderr)
            return 2

        b0 = base64.b64encode((MULTI_DIR / "image_0.jpg").read_bytes()).decode("ascii")
        b1 = base64.b64encode((MULTI_DIR / "image_1.jpg").read_bytes()).decode("ascii")
        payload = {
            "model": "qwen3_5",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b0}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b1}"}},
                ],
            }],
            "max_tokens": 600,
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
            "stream": False,
        }
        r = requests.post(f"http://127.0.0.1:{args.port}/v1/chat/completions",
                          json=payload, timeout=900)
        r.raise_for_status()
        body = r.json()
        text = body["choices"][0]["message"]["content"]
        finish = body["choices"][0]["finish_reason"]

        per_image = evaluate_per_image(text)
        passed = all(v["passed"] for v in per_image.values())

        lines = ["# P6.6 Multi-Image Semantic Verification (Gate 4)", "",
                 f"- Finish reason: `{finish}`",
                 f"- Overall verdict: **{'PASS' if passed else 'FAIL'}**",
                 ""]
        for i, res in per_image.items():
            mark = "PASS" if res['passed'] else "FAIL"
            lines.append(f"## image_{i} - {mark}")
            lines.append("")
            lines.append(f"- {res['n_hit']} / {res['n_total']} keys found")
            for synonyms, hit in zip(KEYS_PER_IMAGE[i], res["hits"]):
                mark2 = "ok" if hit else "miss"
                detail = f"matched {hit}" if hit else "missing"
                lines.append(f"  - [{mark2}] `{synonyms[0]}` ({detail})")
            lines.append("")
        lines.append("## Response")
        lines.append("")
        lines.append("```")
        lines.append(text)
        lines.append("```")
        args.out.write_text("\n".join(lines))
        print(f"[p6_6_semantic_check] {'PASS' if passed else 'FAIL'}; report -> {args.out}")
        return 0 if passed else 1
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
        server_log.close()


if __name__ == "__main__":
    sys.exit(main())
