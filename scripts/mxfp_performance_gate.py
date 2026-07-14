#!/usr/bin/env python3
"""Evaluate independent MXFP4 and MXFP8 affine-relative performance gates."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any


MODE_PAIRS = {
    "mxfp4": ("Qwen3.5-4B-mxfp4", "Qwen3.5-4B-MLX-4bit"),
    "mxfp8": ("Qwen3.5-4B-mxfp8", "Qwen3.5-4B-MLX-8bit"),
}
LONG_PROMPT_LENGTHS = {8192, 32768}
SCHEDULER_CONFIG_FIELDS = (
    "b_max",
    "prefill_chunk_size",
    "admission_deadline_ms",
    "admission_queue_max",
    "max_cache_cap",
    "decode_cadence_mid_chunk_cap",
)


class GateError(RuntimeError):
    pass


def optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def read_matrix_rows(matrix_dir: Path) -> list[dict[str, Any]]:
    summary = matrix_dir / "summary.csv"
    if not summary.is_file():
        raise GateError(f"missing matrix summary: {summary}")
    rows: list[dict[str, Any]] = []
    with summary.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            parsed = dict(row)
            for field in ("pp_target", "tg_target", "concurrency", "requests"):
                parsed[field] = optional_int(parsed.get(field))
            for field in (
                "e2e_s_p95",
                "tpot_ms_median",
                "itl_ms_p95",
                "tokens_per_sec",
            ):
                parsed[field] = optional_float(parsed.get(field))
            parsed["ok"] = str(parsed.get("ok", "")).lower() == "true"
            rows.append(parsed)
    return rows


def read_scheduler_configuration(run_dir: Path) -> dict[str, int]:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise GateError(f"missing scheduler_config manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = manifest.get("scheduler_config")
    if not isinstance(config, dict):
        raise GateError(f"{manifest_path}: missing scheduler_config")
    normalized: dict[str, int] = {}
    for field in SCHEDULER_CONFIG_FIELDS:
        value = config.get(field)
        if not isinstance(value, int) or value <= 0:
            raise GateError(f"{manifest_path}: invalid scheduler_config.{field}={value!r}")
        normalized[field] = value
    return normalized


def validate_scheduler_configurations(
    matrix_dirs: list[Path], strict_dir: Path
) -> dict[str, int]:
    sources = [*matrix_dirs, strict_dir]
    expected_source = sources[0]
    expected = read_scheduler_configuration(expected_source)
    for source in sources[1:]:
        actual = read_scheduler_configuration(source)
        if actual != expected:
            raise GateError(
                "scheduler configuration mismatch: "
                f"expected {expected} from {expected_source}, got {actual} from {source}"
            )
    return expected


def comparison(
    *,
    mode: str,
    metric_family: str,
    source: str,
    pp_target: int | None,
    tg_target: int | None,
    concurrency: int,
    candidate: float,
    baseline: float,
    threshold: float,
) -> dict[str, Any]:
    if candidate < 0 or baseline <= 0:
        raise GateError(
            f"invalid metric values for {mode}/{metric_family}: {candidate}, {baseline}"
        )
    ratio = candidate / baseline
    return {
        "mode": mode,
        "metric_family": metric_family,
        "source": source,
        "pp_target": pp_target,
        "tg_target": tg_target,
        "concurrency": concurrency,
        "candidate": candidate,
        "baseline": baseline,
        "ratio": ratio,
        "threshold": threshold,
        "passed": ratio <= threshold,
    }


def row_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("category"),
        row.get("pp_target"),
        row.get("tg_target"),
        row.get("concurrency"),
    )


def matrix_comparisons(
    matrix_dirs: list[Path],
    mode: str,
    candidate_label: str,
    baseline_label: str,
    threshold: float,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for matrix_dir in matrix_dirs:
        rows = read_matrix_rows(matrix_dir)
        candidates = {
            row_key(row): row
            for row in rows
            if row["model"] == candidate_label and row["ok"]
        }
        baselines = {
            row_key(row): row
            for row in rows
            if row["model"] == baseline_label and row["ok"]
        }
        if not candidates or not baselines:
            raise GateError(
                f"{matrix_dir}: missing rows for {candidate_label} or {baseline_label}"
            )
        for key, candidate_row in sorted(candidates.items(), key=lambda item: str(item[0])):
            baseline_row = baselines.get(key)
            if baseline_row is None:
                raise GateError(f"{matrix_dir}: missing matching baseline row for {key}")
            category, pp_target, tg_target, concurrency = key
            metric_specs: list[tuple[str, str]] = []
            if category in {"http_e2e", "long_context"} and concurrency == 1:
                metric_specs.append(("sequential_tpot", "tpot_ms_median"))
            if category == "long_context" and concurrency == 1:
                metric_specs.append(("long_e2e_c1", "e2e_s_p95"))
            if (
                category == "concurrent"
                and concurrency == 8
                and pp_target in LONG_PROMPT_LENGTHS
            ):
                metric_specs.extend(
                    [("long_e2e_c8", "e2e_s_p95"), ("long_itl_c8", "itl_ms_p95")]
                )
            for metric_family, field in metric_specs:
                candidate_value = candidate_row.get(field)
                baseline_value = baseline_row.get(field)
                if candidate_value is None or baseline_value is None:
                    raise GateError(f"{matrix_dir}: missing {field} for {key}")
                results.append(
                    comparison(
                        mode=mode,
                        metric_family=metric_family,
                        source=str(matrix_dir),
                        pp_target=pp_target,
                        tg_target=tg_target,
                        concurrency=concurrency,
                        candidate=candidate_value,
                        baseline=baseline_value,
                        threshold=threshold,
                    )
                )
    if not results:
        raise GateError(f"no matrix comparisons produced for {mode}")
    return results


def strict_record_path(strict_dir: Path, record: dict[str, Any]) -> tuple[Path, bool]:
    if "benchmarks" in record:
        benchmark = record["benchmarks"].get("c8") or {}
        validation_ok = bool(benchmark.get("validation", {}).get("ok"))
        raw_path = benchmark.get("benchmark_json")
    else:
        validation_ok = bool(record.get("validation", {}).get("ok"))
        raw_path = record.get("benchmark_json")
    if not raw_path:
        raise GateError(f"{record.get('label')}: missing c8 strict decode artifact")
    path = Path(raw_path)
    if not path.is_absolute():
        path = strict_dir / path
    return path, validation_ok


def validate_strict_payload(payload: dict[str, Any], label: str) -> dict[str, Any]:
    cells = payload.get("cells") or []
    raw_runs = payload.get("raw_runs") or []
    if len(cells) != 1:
        raise GateError(f"{label}: expected one strict decode cell")
    if not raw_runs:
        raise GateError(f"{label}: strict decode completed no requests")
    cell = cells[0]
    max_tokens = optional_int(cell.get("tg_target"))
    concurrent = optional_int(cell.get("concurrent"))
    if max_tokens is None or concurrent is None:
        raise GateError(f"{label}: strict decode cell is missing TG or concurrency")
    invalid = [
        index
        for index, row in enumerate(raw_runs)
        if row.get("finish_reason") != "length"
        or row.get("completion_tokens_server") != max_tokens
    ]
    if invalid:
        raise GateError(
            f"{label}: strict decode raw data is not full-length for requests {invalid}"
        )
    workers = {row.get("worker_id") for row in raw_runs}
    if workers != set(range(concurrent)):
        raise GateError(f"{label}: strict decode raw data lacks full worker coverage")
    return cell


def strict_decode_comparison(
    strict_dir: Path,
    mode: str,
    candidate_label: str,
    baseline_label: str,
    threshold: float,
) -> dict[str, Any]:
    manifest_path = strict_dir / "manifest.json"
    if not manifest_path.is_file():
        raise GateError(f"missing strict decode manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = {record["label"]: record for record in manifest.get("models", [])}
    values: dict[str, tuple[dict[str, Any], Path]] = {}
    for label in (candidate_label, baseline_label):
        record = records.get(label)
        if record is None:
            raise GateError(f"strict decode manifest missing model {label}")
        raw_path, validation_ok = strict_record_path(strict_dir, record)
        if not validation_ok:
            raise GateError(f"strict decode validation failed for {label}")
        if not raw_path.is_file():
            raise GateError(f"missing strict decode JSON for {label}: {raw_path}")
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
        values[label] = (validate_strict_payload(payload, label), raw_path)

    candidate_cell, candidate_path = values[candidate_label]
    baseline_cell, _ = values[baseline_label]
    candidate_value = optional_float((candidate_cell.get("itl_ms") or {}).get("p95"))
    baseline_value = optional_float((baseline_cell.get("itl_ms") or {}).get("p95"))
    if candidate_value is None or baseline_value is None:
        raise GateError(f"strict decode ITL p95 missing for {mode}")
    return comparison(
        mode=mode,
        metric_family="strict_decode_itl_c8",
        source=str(candidate_path.parent),
        pp_target=optional_int(candidate_cell.get("pp_target")),
        tg_target=optional_int(candidate_cell.get("tg_target")),
        concurrency=8,
        candidate=candidate_value,
        baseline=baseline_value,
        threshold=threshold,
    )


def memory_comparison(
    matrix_dirs: list[Path], mode: str, candidate_label: str, baseline_label: str
) -> dict[str, Any]:
    for matrix_dir in reversed(matrix_dirs):
        candidate_path = matrix_dir / candidate_label / "health_before.json"
        baseline_path = matrix_dir / baseline_label / "health_before.json"
        if candidate_path.is_file() and baseline_path.is_file():
            candidate = json.loads(candidate_path.read_text(encoding="utf-8"))["memory"][
                "mlx_active_bytes"
            ]
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))["memory"][
                "mlx_active_bytes"
            ]
            return {
                "mode": mode,
                "source": str(matrix_dir),
                "candidate_active_bytes": candidate,
                "baseline_active_bytes": baseline,
                "ratio": candidate / baseline,
            }
    raise GateError(f"missing health memory artifacts for {mode}")


def write_summary(run_dir: Path, report: dict[str, Any]) -> None:
    lines = [
        "# MXFP Performance Gate",
        "",
        f"- Overall status: `{report['overall_status']}`",
        f"- Regression threshold: `{report['threshold']:.2f}x` affine baseline",
        f"- Fixed scheduler configuration: `{report['scheduler_config']}`",
        "- Short c=8 ITL uses fixed-prompt full-length decode; synthetic short-prompt cells are excluded because model-specific EOS creates unequal load.",
        "",
        "| mode | metric | PP | TG | C | candidate | affine | ratio | status |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in report["comparisons"]:
        lines.append(
            "| {mode} | {metric} | {pp} | {tg} | {concurrency} | {candidate:.3f} | {baseline:.3f} | {ratio:.3f} | {status} |".format(
                mode=item["mode"],
                metric=item["metric_family"],
                pp=item["pp_target"] if item["pp_target"] is not None else "",
                tg=item["tg_target"] if item["tg_target"] is not None else "",
                concurrency=item["concurrency"],
                candidate=item["candidate"],
                baseline=item["baseline"],
                ratio=item["ratio"],
                status="passed" if item["passed"] else "failed",
            )
        )
    lines.extend(
        [
            "",
            "## Model Memory",
            "",
            "| mode | candidate active bytes | affine active bytes | ratio |",
            "|---|---:|---:|---:|",
        ]
    )
    for item in report["memory"]:
        lines.append(
            f"| {item['mode']} | {item['candidate_active_bytes']} | {item['baseline_active_bytes']} | {item['ratio']:.3f} |"
        )
    lines.extend(["", f"- Machine-readable gate: `{run_dir / 'gate.json'}`", ""])
    (run_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", action="append", type=Path, required=True)
    parser.add_argument("--strict-decode", type=Path, required=True)
    parser.add_argument(
        "--out-root", type=Path, default=Path("reports/mxfp-performance")
    )
    parser.add_argument("--threshold", type=float, default=1.25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.threshold <= 0:
        raise GateError("--threshold must be > 0")
    repo = Path(__file__).resolve().parents[1]
    matrix_dirs = [path.resolve() for path in args.matrix]
    strict_dir = args.strict_decode.resolve()
    out_root = args.out_root if args.out_root.is_absolute() else repo / args.out_root
    run_dir = out_root.resolve() / time.strftime("%Y-%m-%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir.parent / "latest.txt").write_text(f"{run_dir}\n", encoding="utf-8")

    comparisons: list[dict[str, Any]] = []
    memory: list[dict[str, Any]] = []
    scheduler_config = validate_scheduler_configurations(matrix_dirs, strict_dir)
    for mode, (candidate, baseline) in MODE_PAIRS.items():
        comparisons.extend(
            matrix_comparisons(
                matrix_dirs, mode, candidate, baseline, threshold=args.threshold
            )
        )
        comparisons.append(
            strict_decode_comparison(
                strict_dir, mode, candidate, baseline, threshold=args.threshold
            )
        )
        memory.append(memory_comparison(matrix_dirs, mode, candidate, baseline))

    report = {
        "generated_at_unix": int(time.time()),
        "threshold": args.threshold,
        "matrix_dirs": [str(path) for path in matrix_dirs],
        "strict_decode_dir": str(strict_dir),
        "scheduler_config": scheduler_config,
        "comparisons": comparisons,
        "memory": memory,
        "mode_status": {
            mode: (
                "passed"
                if all(item["passed"] for item in comparisons if item["mode"] == mode)
                else "failed"
            )
            for mode in MODE_PAIRS
        },
    }
    report["overall_status"] = (
        "passed" if all(item["passed"] for item in comparisons) else "failed"
    )
    (run_dir / "gate.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_summary(run_dir, report)
    print(run_dir)
    return 0 if report["overall_status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
