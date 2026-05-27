#!/usr/bin/env python3
"""Summarize a Gemma4 VL profile report directory into Markdown."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def latest_report(root: Path) -> Path:
    base = root / "reports" / "gemma4-vl-profile"
    candidates = sorted(p for p in base.iterdir() if p.is_dir())
    if not candidates:
        raise SystemExit(f"no Gemma4 VL profile reports under {base}")
    return candidates[-1]


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def ms(value: float) -> str:
    return f"{value:.3f}"


def write_table(lines: list[str], headers: list[str], rows: list[list[str]]) -> None:
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    report = args.report.resolve() if args.report else latest_report(repo)
    summary_path = report / "summary.tsv"
    metrics_path = report / "metrics.tsv"
    chunks_path = report / "chunks.tsv"
    if not summary_path.exists() or not metrics_path.exists() or not chunks_path.exists():
        raise SystemExit(f"missing summary.tsv, metrics.tsv, or chunks.tsv in {report}")

    summary = read_tsv(summary_path)
    metrics = read_tsv(metrics_path)
    chunks = read_tsv(chunks_path)

    by_run: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in metrics:
        by_run[(row["case"], row["chunk_size"])].append(row)

    lines: list[str] = []
    lines.append("# Gemma4 VL Profile Baseline")
    lines.append("")
    lines.append(f"Report: `{report}`")
    lines.append("")

    summary_rows = []
    for row in summary:
        summary_rows.append(
            [
                row["case"],
                row["chunk_size"],
                row["output_sha256"][:12],
                row["hidden_chunks"],
                row["slice_projects"],
                f'{row["dedup_unique"]}/{row["dedup_images"]}',
                row["dedup_duplicates"],
            ]
        )
    write_table(
        lines,
        [
            "case",
            "chunk",
            "output_sha",
            "hidden_chunks",
            "slice_projects",
            "dedup_unique/images",
            "dedup_duplicates",
        ],
        summary_rows,
    )

    lines.append("## Top Metrics By Run")
    lines.append("")
    for key in sorted(by_run):
        case, chunk = key
        totals: dict[str, float] = defaultdict(float)
        counts: dict[str, int] = defaultdict(int)
        for row in by_run[key]:
            metric = row["metric"]
            try:
                value = float(row["value_ms"])
            except ValueError:
                continue
            totals[metric] += value
            counts[metric] += 1
        top = sorted(totals.items(), key=lambda item: item[1], reverse=True)[:12]
        lines.append(f"### {case} chunk={chunk}")
        lines.append("")
        write_table(
            lines,
            ["metric", "sum_ms", "count"],
            [[metric, ms(total), str(counts[metric])] for metric, total in top],
        )

    lines.append("## Chunk Composition")
    lines.append("")
    write_table(
        lines,
        [
            "case",
            "chunk",
            "path",
            "range",
            "seq",
            "image",
            "text",
            "runs",
            "lead/trail",
            "image_rows",
            "last",
        ],
        [
            [
                row["case"],
                row["chunk_size"],
                row["path"],
                f'{row["chunk_start"]}-{row["chunk_end"]}',
                row["seq"],
                row["image_tokens"],
                row["text_tokens"],
                row["image_runs"],
                f'{row["leading_image_tokens"]}/{row["trailing_image_tokens"]}',
                f'{row["image_rows_start"]}-{row["image_rows_end"]}',
                row["is_last"],
            ]
            for row in chunks
        ],
    )

    layer_rows = [row for row in metrics if row.get("layer_idx", "-") != "-"]
    if layer_rows:
        lines.append("## Layer Profile")
        lines.append("")
        layer_totals: dict[tuple[str, str, str, str, str], float] = defaultdict(float)
        layer_counts: dict[tuple[str, str, str, str, str], int] = defaultdict(int)
        for row in layer_rows:
            key = (
                row["case"],
                row["chunk_size"],
                row["layer_idx"],
                row["layer_kind"],
                row["metric"],
            )
            try:
                value = float(row["value_ms"])
            except ValueError:
                continue
            layer_totals[key] += value
            layer_counts[key] += 1

        top_layers = sorted(layer_totals.items(), key=lambda item: item[1], reverse=True)[:30]
        write_table(
            lines,
            ["case", "chunk", "layer", "kind", "metric", "sum_ms", "count"],
            [
                [case, chunk, layer, kind, metric, ms(total), str(layer_counts[key])]
                for key, total in top_layers
                for case, chunk, layer, kind, metric in [key]
            ],
        )

    out = report / "baseline.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"baseline: {out}")


if __name__ == "__main__":
    main()
