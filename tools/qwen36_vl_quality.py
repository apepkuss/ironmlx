#!/usr/bin/env python3
"""Small report-only quality harness for Qwen3.6 VL smoke evaluation."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class QualityCase:
    case_id: str
    prompt: str
    image_paths: list[Path]


@dataclass(frozen=True)
class Target:
    engine: str
    model: str
    endpoint: str


def build_cases(fixture_dir: Path) -> list[QualityCase]:
    multi_dir = fixture_dir / "multi_image"
    return [
        QualityCase(
            case_id="text_baseline",
            prompt="Write one concise sentence about why reproducible benchmarks matter.",
            image_paths=[],
        ),
        QualityCase(
            case_id="single_image_cats",
            prompt=(
                "Describe this image in one concise sentence. Mention the main animals "
                "and the furniture color."
            ),
            image_paths=[fixture_dir / "coco_sample.jpg"],
        ),
        QualityCase(
            case_id="multi_image_kitchen_street",
            prompt=(
                "You are given two images. In one sentence per image, describe the main "
                "scene in image 1 and image 2."
            ),
            image_paths=[multi_dir / "image_0.jpg", multi_dir / "image_1.jpg"],
        ),
    ]


def mime_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".webp":
        return "image/webp"
    return "application/octet-stream"


def image_data_url(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type(path)};base64,{data}"


def build_payload(
    *,
    model: str,
    prompt: str,
    image_paths: list[Path],
    max_tokens: int,
    stream: bool,
) -> dict[str, Any]:
    if image_paths:
        content: Any = [{"type": "text", "text": prompt}]
        for image_path in image_paths:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_data_url(image_path)},
                }
            )
    else:
        content = prompt

    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": stream,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def quality_flags(case_id: str, output_text: str) -> list[str]:
    stripped = output_text.strip()
    flags: list[str] = []
    if not stripped:
        flags.append("blank_output")
        return flags

    visible = re.sub(r"\s+", "", stripped)
    if len(visible) >= 12:
        most_common = max(visible.count(char) for char in set(visible))
        if most_common / len(visible) >= 0.72:
            flags.append("repetitive_output")

    lowered = stripped.lower()
    content_tokens = re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", lowered)
    if "repetitive_output" not in flags and len(content_tokens) >= 6:
        most_common_token = max(content_tokens.count(token) for token in set(content_tokens))
        if most_common_token / len(content_tokens) >= 0.6:
            flags.append("repetitive_output")

    if case_id == "single_image_cats":
        hints = ("cat", "cats", "kitten", "couch", "sofa", "pink", "猫", "沙发", "粉")
        if not any(hint in lowered for hint in hints):
            flags.append("missing_cat_or_couch_hint")
    elif case_id == "multi_image_kitchen_street":
        kitchen_hints = ("kitchen", "cook", "cooking", "pot", "stove", "pan", "厨房", "锅")
        street_hints = (
            "street",
            "sidewalk",
            "construction",
            "wall",
            "people",
            "person",
            "road",
            "街",
            "施工",
            "墙",
            "人",
        )
        if not any(hint in lowered for hint in kitchen_hints):
            flags.append("missing_kitchen_hint")
        if not any(hint in lowered for hint in street_hints):
            flags.append("missing_street_hint")

    return flags


def normalize_endpoint(endpoint: str) -> str:
    trimmed = endpoint.rstrip("/")
    if trimmed.endswith("/v1"):
        return f"{trimmed}/chat/completions"
    if trimmed.endswith("/chat/completions"):
        return trimmed
    return f"{trimmed}/v1/chat/completions"


def parse_target(raw: str) -> Target:
    parts = raw.split("=", 2)
    if len(parts) != 3 or not all(parts):
        raise argparse.ArgumentTypeError(
            "target must use engine=model=endpoint, for example "
            "ironmlx=qwen3_6_moe=http://127.0.0.1:18164/v1"
        )
    return Target(parts[0], parts[1], normalize_endpoint(parts[2]))


def extract_text(response: dict[str, Any]) -> tuple[str, str | None]:
    choices = response.get("choices") or []
    if not choices:
        return "", None
    choice = choices[0]
    message = choice.get("message") or {}
    content = message.get("content", "")
    if isinstance(content, list):
        text = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    else:
        text = str(content)
    return text, choice.get("finish_reason")


def call_completion(
    *,
    target: Target,
    case: QualityCase,
    max_tokens: int,
    timeout_sec: float,
) -> dict[str, Any]:
    payload = build_payload(
        model=target.model,
        prompt=case.prompt,
        image_paths=case.image_paths,
        max_tokens=max_tokens,
        stream=False,
    )
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        target.endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.perf_counter()
    status_code = 0
    raw_text = ""
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            status_code = response.status
            raw_text = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        status_code = exc.code
        raw_text = exc.read().decode("utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001 - this is a report-only probe.
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {
            "engine": target.engine,
            "model": target.model,
            "endpoint": target.endpoint,
            "case_id": case.case_id,
            "status_code": status_code,
            "finish_reason": None,
            "output_text": "",
            "quality_flags": [f"request_error:{type(exc).__name__}"],
            "elapsed_ms": elapsed_ms,
            "raw_response": str(exc),
        }

    elapsed_ms = (time.perf_counter() - start) * 1000
    output_text = ""
    finish_reason = None
    parse_error = None
    try:
        parsed = json.loads(raw_text)
        output_text, finish_reason = extract_text(parsed)
    except json.JSONDecodeError as exc:
        parse_error = f"json_error:{exc.msg}"

    flags = quality_flags(case.case_id, output_text)
    if status_code != 200:
        flags.insert(0, f"http_status:{status_code}")
    if parse_error is not None:
        flags.insert(0, parse_error)

    return {
        "engine": target.engine,
        "model": target.model,
        "endpoint": target.endpoint,
        "case_id": case.case_id,
        "status_code": status_code,
        "finish_reason": finish_reason,
        "output_text": output_text,
        "quality_flags": flags,
        "elapsed_ms": elapsed_ms,
        "raw_response": raw_text[:4000],
    }


def markdown_escape(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_csv(records: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "engine",
        "model",
        "case_id",
        "status_code",
        "finish_reason",
        "quality_flags",
        "elapsed_ms",
        "output_text",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    **{key: record.get(key) for key in fieldnames},
                    "quality_flags": ",".join(record.get("quality_flags", [])),
                }
            )


def write_markdown(records: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Qwen3.6 VL quality report",
        "",
        "| Engine | Case | Result | HTTP | Finish | Elapsed ms | Flags | Preview |",
        "| --- | --- | --- | ---: | --- | ---: | --- | --- |",
    ]
    for record in records:
        flags = record.get("quality_flags", [])
        result = "PASS" if record.get("status_code") == 200 and not flags else "FAIL"
        preview = re.sub(r"\s+", " ", str(record.get("output_text", ""))).strip()[:160]
        lines.append(
            "| "
            + " | ".join(
                [
                    markdown_escape(record.get("engine", "")),
                    markdown_escape(record.get("case_id", "")),
                    result,
                    markdown_escape(record.get("status_code", "")),
                    markdown_escape(record.get("finish_reason", "")),
                    f"{float(record.get('elapsed_ms', 0.0)):.1f}",
                    markdown_escape(",".join(flags)),
                    markdown_escape(preview),
                ]
            )
            + " |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def validate_cases(cases: list[QualityCase]) -> None:
    missing = [str(path) for case in cases for path in case.image_paths if not path.exists()]
    if missing:
        joined = "\n".join(missing)
        raise FileNotFoundError(f"missing fixture image(s):\n{joined}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture-dir",
        type=Path,
        default=Path("ironmlx/tests/fixtures/qwen35_vl"),
    )
    parser.add_argument(
        "--target",
        action="append",
        type=parse_target,
        required=True,
        help="engine=model=endpoint, endpoint may be base URL, /v1, or /v1/chat/completions",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/qwen36_vl_quality"))
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout-sec", type=float, default=180.0)
    args = parser.parse_args()

    cases = build_cases(args.fixture_dir)
    validate_cases(cases)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for target in args.target:
        for case in cases:
            records.append(
                call_completion(
                    target=target,
                    case=case,
                    max_tokens=args.max_tokens,
                    timeout_sec=args.timeout_sec,
                )
            )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    write_jsonl(records, args.out_dir / f"qwen36_vl_quality_{stamp}.jsonl")
    write_csv(records, args.out_dir / f"qwen36_vl_quality_{stamp}.csv")
    write_markdown(records, args.out_dir / f"qwen36_vl_quality_{stamp}.md")

    failures = sum(1 for record in records if record["status_code"] != 200 or record["quality_flags"])
    print(f"wrote {len(records)} records to {args.out_dir}; failures={failures}")
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
