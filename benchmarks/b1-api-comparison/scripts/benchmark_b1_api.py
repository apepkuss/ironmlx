#!/usr/bin/env python3
"""One fresh-server B1 session; retain raw events and failures for later analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from tokenizers import Tokenizer

WARMUPS = [
    {
        "id": "warmup-1",
        "category": "warmup",
        "prompt": "请用三句话说明良好 API 错误信息应包含哪些内容。",
    },
    {
        "id": "warmup-2",
        "category": "warmup",
        "prompt": "写一个 Python 函数，判断整数是否为偶数，并给出一个调用示例。",
    },
]


def measure(url, model, item, tokenizer, omit_effort=False):
    body = dict(
        model=model,
        messages=[dict(role="user", content=item["prompt"])],
        stream=True,
        stream_options=dict(include_usage=True),
        temperature=0,
        top_p=1,
        max_tokens=4096,
        chat_template_kwargs=dict(enable_thinking=False),
    )
    if not omit_effort:
        body["reasoning_effort"] = "none"
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    events, usage, finish, error, reasoning = [], None, None, None, []
    status = None
    done = False
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            status = response.status
            for line in response:
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    done = True
                    continue
                if not data:
                    continue
                payload = json.loads(data)
                if payload.get("error"):
                    raise ValueError(str(payload["error"]))
                if payload.get("usage"):
                    usage = payload["usage"]
                for choice in payload.get("choices", []):
                    finish = choice.get("finish_reason") or finish
                    delta = choice.get("delta") or {}
                    if delta.get("reasoning_content") or delta.get("reasoning"):
                        reasoning.append(
                            delta.get("reasoning_content") or delta["reasoning"]
                        )
                    if delta.get("content"):
                        events.append(
                            dict(t=time.perf_counter() - started, text=delta["content"])
                        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, urllib.error.HTTPError):
            status = exc.code
            error += ": " + exc.read().decode(errors="replace")
    e2e = time.perf_counter() - started
    output = "".join(e["text"] for e in events)
    common = len(tokenizer.encode(output, add_special_tokens=False).ids)
    count = (usage or {}).get("completion_tokens")
    span = events[-1]["t"] - events[0]["t"] if len(events) > 1 else 0
    valid = (
        not error
        and status == 200
        and done
        and finish == "stop"
        and not reasoning
        and "<think>" not in output
        and "</think>" not in output
        and count
        and span > 0
    )
    return dict(
        **item,
        valid=bool(valid),
        status=status,
        error=error,
        done=done,
        finish_reason=finish,
        usage=usage,
        completion_tokens=count,
        common_tokens=common,
        first_chunk_tokens=len(
            tokenizer.encode(events[0]["text"], add_special_tokens=False).ids
        )
        if events
        else 0,
        ttft_s=events[0]["t"] if events else None,
        decode_s=span,
        decode_tps=(count - 1) / span if count and span else None,
        common_decode_tps=(common - 1) / span if common > 1 and span else None,
        e2e_s=e2e,
        output=output,
        reasoning=reasoning,
        events=events,
        output_sha256=hashlib.sha256(output.encode()).hexdigest(),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:18480/v1/chat/completions")
    p.add_argument("--model", default="benchmark")
    p.add_argument("--label", required=True)
    p.add_argument("--session", type=int, default=0)
    p.add_argument("--tokenizer", required=True, type=Path)
    p.add_argument(
        "--prompts",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "fixtures/b1-api-prompts.json",
    )
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--omit-reasoning-effort", action="store_true")
    p.add_argument("--only", help="Development screen only: comma-separated prompt ids")
    args = p.parse_args()
    if args.output.exists():
        p.error(
            "output already exists; preserve earlier experiments with a new filename"
        )
    tokenizer = Tokenizer.from_file(str(args.tokenizer / "tokenizer.json"))
    prompts = json.loads(args.prompts.read_text())
    if args.only:
        prompts = [x for x in prompts if x["id"] in args.only.split(",")]
    rotation = args.session % len(prompts)
    prompts = prompts[rotation:] + prompts[:rotation]
    report = dict(
        schema=1,
        label=args.label,
        session=args.session,
        url=args.url,
        model=args.model,
        started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        prompts_sha256=hashlib.sha256(args.prompts.read_bytes()).hexdigest(),
        thermal_before=subprocess.getoutput("pmset -g therm"),
        processes_before=subprocess.getoutput("ps -axo pid,pcpu,comm"),
        runs=[],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for item in WARMUPS + prompts:
        row = measure(args.url, args.model, item, tokenizer, args.omit_reasoning_effort)
        report["runs"].append(row)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(
            json.dumps(
                {
                    k: row[k]
                    for k in (
                        "id",
                        "valid",
                        "error",
                        "completion_tokens",
                        "common_tokens",
                        "ttft_s",
                        "decode_tps",
                        "e2e_s",
                    )
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if not row["valid"]:
            raise SystemExit(
                "Invalid response retained; stop session rather than filtering it out"
            )
        time.sleep(1)
    report["thermal_after"] = subprocess.getoutput("pmset -g therm")
    report["processes_after"] = subprocess.getoutput("ps -axo pid,pcpu,comm")
    rows = [r for r in report["runs"] if r["category"] != "warmup"]
    report["summary"] = {
        metric: statistics.median(r[metric] for r in rows)
        for metric in ("ttft_s", "decode_tps", "common_decode_tps", "e2e_s")
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
