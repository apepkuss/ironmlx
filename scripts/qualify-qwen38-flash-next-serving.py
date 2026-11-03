#!/usr/bin/env python3
"""Qualify the actual HTTP scheduler, independent of native graph fixtures.

Run against an isolated Flash server, then against its DFlash2 counterpart
with --baseline pointing to the first JSON. No model files are modified.
"""

import argparse
import base64
import concurrent.futures
import json
import time
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:19083")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument(
        "--images",
        action="store_true",
        help="Qualify mixed text, image and multi-image requests",
    )
    args = parser.parse_args()
    ready_deadline = time.monotonic() + 30
    while True:
        try:
            with urllib.request.urlopen(args.url + "/healthz", timeout=3) as response:
                assert json.load(response)["status"] == "healthy"
            break
        except OSError:
            if time.monotonic() >= ready_deadline:
                raise
            time.sleep(0.2)
    prompts = [
        "请只回答：一加一等于几？",
        "Explain recursion in two sentences.",
        "Write a Python function that adds two numbers.",
        "请用一句话介绍你自己。",
    ]

    if args.images:
        root = (
            Path(__file__).resolve().parents[1]
            / "ironmlx-lm/tests/fixtures/embedding_gemma2/images"
        )

        def image(name):
            return dict(
                type="image_url",
                image_url=dict(
                    url="data:image/png;base64,"
                    + base64.b64encode((root / name).read_bytes()).decode()
                ),
            )

        prompts = [
            prompts[0],
            [
                image("red-circle.png"),
                dict(type="text", text="Describe the image briefly."),
            ],
            [
                image("blue-square.png"),
                image("green-triangle.png"),
                dict(type="text", text="Compare the two pictures briefly."),
            ],
            prompts[2],
        ]

    def request(prompt, *, stream=False, max_tokens=None):
        payload = dict(
            model="flash-qualification",
            messages=[dict(role="user", content=prompt)],
            temperature=0,
            max_tokens=args.max_tokens if max_tokens is None else max_tokens,
            stream=stream,
            chat_template_kwargs=dict(enable_thinking=False),
        )
        return urllib.request.Request(
            args.url + "/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )

    def complete(prompt):
        started = time.monotonic()
        with urllib.request.urlopen(request(prompt), timeout=300) as response:
            result = json.load(response)
        choice = result["choices"][0]
        return dict(
            message=choice["message"],
            finish_reason=choice["finish_reason"],
            completion_tokens=result["usage"]["completion_tokens"],
            seconds=time.monotonic() - started,
        )

    def comparable(rows):
        return [{k: v for k, v in row.items() if k != "seconds"} for row in rows]

    baseline = [complete(prompt) for prompt in prompts]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        batch = list(executor.map(complete, prompts))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(dict(baseline=baseline, batch=batch), ensure_ascii=False, indent=2)
        + "\n"
    )
    assert comparable(batch) == comparable(baseline), (
        "B4 differs from independent B1 requests"
    )
    if args.baseline:
        other = json.loads(args.baseline.read_text())
        assert comparable(batch) == comparable(other["batch"]), (
            "DFlash2 differs from ordinary HTTP output"
        )

    # Close a live SSE request after a content event, then immediately reuse
    # the actor alongside another request. Neither cancellation nor a reused
    # slot may leak recurrent, PLE or QSA history to the next request.
    with urllib.request.urlopen(
        request(
            prompts[1] if args.images else "Count slowly from one to one hundred.",
            stream=True,
            max_tokens=512,
        ),
        timeout=300,
    ) as response:
        for raw in response:
            if raw.startswith(b"data: {"):
                event = json.loads(raw[6:])
                if any(
                    choice.get("delta", {}).get("content")
                    for choice in event.get("choices", [])
                ):
                    break
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        after_cancel = list(executor.map(complete, prompts[:2]))
    assert comparable(after_cancel) == comparable(baseline[:2]), (
        "cancellation/slot reuse changed output"
    )
    deadline = time.monotonic() + 30
    while True:
        with urllib.request.urlopen(args.url + "/healthz", timeout=10) as response:
            health = json.load(response)
        scheduler = health["scheduler"]
        if scheduler["b_active"] == 0 and scheduler["b_queued"] == 0:
            break
        assert time.monotonic() < deadline, (
            "cancelled request did not release scheduler capacity"
        )
        time.sleep(0.1)
    result = dict(
        url=args.url,
        baseline=baseline,
        batch=batch,
        after_cancel=after_cancel,
        scheduler=scheduler,
        dflash2=health.get("dflash2"),
        memory=health["memory"],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                result="pass",
                b1=4,
                b4=4,
                cancelled_and_reused=True,
                images=args.images,
                scheduler=scheduler,
                dflash2_enabled=health.get("dflash2", {}).get("enabled"),
            ),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
