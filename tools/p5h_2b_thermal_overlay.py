"""P5h+2.b T3 — powermetrics thermal overlay joiner.

Per spec § 5.4: parses powermetrics JSON output (--samplers smc,gpu_power,thermal
--format json) and joins to iron-bench per-run timestamps (run_start_unix_ns /
run_end_unix_ns columns from --capture-run-timestamps). Outputs per-run
thermal alignment + outlier-correlation summary.

CLI:
    python tools/p5h_2b_thermal_overlay.py \\
        --powermetrics-json /tmp/p5h+2-b-t3-thermal-t1-{exp}.json \\
        --cell-dir /tmp/p5h+2-b-t1-{exp}-r1-pp128 \\
        --out-json /tmp/p5h+2-b-t3-overlay-t1-{exp}-r1-pp128.json
"""

from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from statistics import median

OUTLIER_THRESHOLD_PCT = 10.0  # match T0 threshold


def parse_powermetrics_samples(json_path: Path) -> list[dict]:
    """Parse powermetrics JSON output. Each sample has timestamp_ms and thermal/gpu/fan fields.
    Powermetrics emits one JSON object per sample with timestamp in ms-since-epoch
    (or a separate `timestamp` field — implementer verifies actual schema at runtime
    by checking the first few samples)."""
    samples = []
    text = json_path.read_text(errors="replace")
    for line in text.splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            samples.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    if samples:
        return samples
    decoder = json.JSONDecoder()
    idx = 0
    while idx < len(text):
        while idx < len(text) and text[idx].isspace():
            idx += 1
        if idx >= len(text):
            break
        try:
            obj, end = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            idx += 1
            continue
        if isinstance(obj, dict):
            samples.append(obj)
        elif isinstance(obj, list):
            samples.extend(x for x in obj if isinstance(x, dict))
        idx = end
    return samples


def join_overlay(powermetrics_samples: list[dict], cell_dir: Path) -> dict:
    bench_path = cell_dir / "bench.csv"
    meta = json.loads((cell_dir / "meta.json").read_text())
    with bench_path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    required_cols = {"run_start_unix_ns", "run_end_unix_ns"}
    missing = required_cols - set(rows[0].keys()) if rows else required_cols
    if not rows or missing:
        raise SystemExit(
            f"{cell_dir}: bench.csv missing columns {sorted(missing)} "
            "(requires --capture-run-timestamps from iron-bench)"
        )
    # Find cell median for outlier threshold
    pp_tps_list = [float(r["pp_tps"]) for r in rows]
    cell_median = median(pp_tps_list)
    overlay = []
    for row in rows:
        start_ns = int(row["run_start_unix_ns"])
        end_ns = int(row["run_end_unix_ns"])
        # Find samples in [start_ns, end_ns]
        start_ms = start_ns // 1_000_000
        end_ms = end_ns // 1_000_000
        # powermetrics timestamp field name is implementer-verified;
        # typical: 'timestamp' (Unix ms) or 'sample_time_ms'.
        # Fallback: iterate samples and infer field name from first sample.
        ts_field = _infer_timestamp_field(powermetrics_samples)
        in_window = [
            s
            for s in powermetrics_samples
            if start_ms <= int(s.get(ts_field) or 0) <= end_ms
        ]
        pp_tps = float(row["pp_tps"])
        deviation_pct = (
            abs(pp_tps - cell_median) / cell_median * 100 if cell_median > 0 else 0
        )
        is_outlier = deviation_pct > OUTLIER_THRESHOLD_PCT
        thermal_summary = _summarize_thermal(in_window) if in_window else None
        overlay.append(
            {
                "run_idx": int(row["run_idx"]),
                "pp_tps": pp_tps,
                "is_outlier": is_outlier,
                "thermal_samples_in_window": len(in_window),
                "thermal_summary": thermal_summary,
            }
        )
    # Correlation: do outlier runs coincide with thermal spikes?
    outlier_thermal_max = [
        o["thermal_summary"]["max_gpu_die_c"]
        for o in overlay
        if o["is_outlier"]
        and o["thermal_summary"] is not None
        and o["thermal_summary"]["max_gpu_die_c"] is not None
    ]
    nonoutlier_thermal_max = [
        o["thermal_summary"]["max_gpu_die_c"]
        for o in overlay
        if not o["is_outlier"]
        and o["thermal_summary"] is not None
        and o["thermal_summary"]["max_gpu_die_c"] is not None
    ]
    correlation = "unknown"
    if outlier_thermal_max and nonoutlier_thermal_max:
        avg_out = sum(outlier_thermal_max) / len(outlier_thermal_max)
        avg_norm = sum(nonoutlier_thermal_max) / len(nonoutlier_thermal_max)
        if avg_out > avg_norm * 1.05:
            correlation = "outliers_run_hot"
        elif avg_out < avg_norm * 0.95:
            correlation = "outliers_run_cool"
        else:
            correlation = "no_thermal_correlation"
    return {
        "cell": str(cell_dir),
        "server_lifecycle": meta.get("server_lifecycle"),
        "logging_mode": meta.get("logging_mode"),
        "n_overlay_runs": len(overlay),
        "correlation": correlation,
        "overlay": overlay,
    }


def _infer_timestamp_field(samples: list[dict]) -> str:
    if not samples:
        return "timestamp"
    candidates = ["timestamp", "sample_time_ms", "timestamp_ms", "time_ms"]
    for c in candidates:
        if c in samples[0]:
            return c
    raise SystemExit(
        f"cannot infer powermetrics timestamp field; first sample keys: {list(samples[0].keys())}"
    )


def _summarize_thermal(samples: list[dict]) -> dict:
    # Powermetrics schema: thermal data inside e.g. samples[i]['thermal_pressure'] or
    # samples[i]['gpu']['die_temperature_c']. Implementer verifies at runtime.
    # Conservative fallback: extract any numeric field with 'temp' or 'die' in name.
    gpu_die_temps: list[float] = []
    for s in samples:
        # Walk dict for any *temp* or *die* numeric value
        def _walk(d: object, out: list[float]) -> None:
            if isinstance(d, dict):
                for k, v in d.items():
                    if isinstance(v, (int, float)) and any(
                        kw in k.lower() for kw in ("temp", "die")
                    ):
                        out.append(float(v))
                    elif isinstance(v, (dict, list)):
                        _walk(v, out)
            elif isinstance(d, list):
                for item in d:
                    _walk(item, out)

        _walk(s, gpu_die_temps)
    if not gpu_die_temps:
        return {"max_gpu_die_c": None, "n_temp_samples": 0}
    return {
        "max_gpu_die_c": max(gpu_die_temps),
        "min_gpu_die_c": min(gpu_die_temps),
        "n_temp_samples": len(gpu_die_temps),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--powermetrics-json", type=Path, required=True)
    p.add_argument("--cell-dir", type=Path, required=True)
    p.add_argument("--out-json", type=Path, required=True)
    args = p.parse_args()
    samples = parse_powermetrics_samples(args.powermetrics_json)
    if not samples:
        result = {
            "cell": str(args.cell_dir),
            "correlation": "unavailable",
            "note": "powermetrics JSON parsed 0 samples",
        }
    else:
        result = join_overlay(samples, args.cell_dir)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2))
    print(f"Wrote {args.out_json}")
    print(f"  correlation: {result.get('correlation')}")


if __name__ == "__main__":
    main()
