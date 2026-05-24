"""Pytest for tools/p5h_2b_t0_outlier_source.py (P5h+2.b T0)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TOOLS_DIR))

from p5h_2b_t0_outlier_source import (  # noqa: E402
    DEFAULT_PRODUCTION_WARMUP,
    decompose_cell,
)


def _write_cell(
    tmp_path: Path,
    pp: int,
    mode: str,
    bench_csv: str,
    server_log: str,
    meta: dict | None = None,
) -> Path:
    cell = tmp_path / f"r1-pp{pp}-{mode}"
    cell.mkdir()
    (cell / "bench.csv").write_text(bench_csv)
    (cell / "server.log").write_text(server_log)
    if meta is None:
        meta = {"mode": mode, "pp": pp}
    (cell / "meta.json").write_text(json.dumps(meta))
    return cell


def test_decompose_probe_request_id_join(tmp_path):
    bench = (
        "target,pp_target,tg_target,run_idx,ttft_ms,tg_tps,tpot_ms,pp_tps,e2e_s,prompt_tokens_local,prompt_tokens_server,completion_tokens_server,cached_tokens,finish_reason,request_id\n"
        "x,128,1,0,100.0,1.0,1.0,1000.0,1.0,128,140,1,0,length,req-a\n"
        "x,128,1,1,100.0,1.0,1.0,1000.0,1.0,128,140,1,0,length,req-b\n"
        "x,128,1,2,100.0,1.0,1.0,1000.0,1.0,128,140,1,0,length,req-c\n"
        "x,128,1,3,100.0,1.0,1.0,1000.0,1.0,128,140,1,0,length,req-d\n"
        "x,128,1,4,100.0,1.0,1.0,1000.0,1.0,128,140,1,0,length,req-e\n"
        "x,128,1,5,100.0,1.0,1.0,1000.0,1.0,128,140,1,0,length,req-f\n"
        "x,128,1,6,100.0,1.0,1.0,500.0,1.0,128,140,1,0,length,req-g\n"
    )
    # Server log: 7 root spans matching req-a..req-g; req-g abnormally slow
    log_lines = []
    for i, (rid, dur_us) in enumerate(
        [
            ("req-a", 90000.0),
            ("req-b", 90000.0),
            ("req-c", 90000.0),
            ("req-d", 90000.0),
            ("req-e", 90000.0),
            ("req-f", 90000.0),
            ("req-g", 180000.0),  # server-side slowdown
        ]
    ):
        start_ns = 1000000000 + i * 200000000
        end_ns = int(start_ns + dur_us * 1000)
        log_lines.append(
            f"[p5h-profile] request_id={rid} routing_path=scheduler prompt_tokens=140 "
            f"seq=0 layer_idx=-1 chunk_idx=null span_id=1 parent_span_id=null "
            f"span_name=root parent_span=null start_ns={start_ns} end_ns={end_ns} "
            f"mode=off span_kind=tree\n"
        )
    cell = _write_cell(
        tmp_path,
        128,
        "probe",
        bench,
        "".join(log_lines),
        meta={"mode": "probe", "warmup_count": 0},
    )
    v = decompose_cell(cell)
    assert v.verdict == "server_side"
    # req-g outlier; its root_us should be ~180k (server-slow) not the median ~90k
    outliers = [r for r in v.runs if r.is_outlier]
    assert len(outliers) == 1


def test_decompose_production_warmup_aware_ordinal(tmp_path):
    bench = (
        "target,pp_target,tg_target,run_idx,ttft_ms,tg_tps,tpot_ms,pp_tps,e2e_s,prompt_tokens_local,prompt_tokens_server,completion_tokens_server,cached_tokens,finish_reason\n"
        "x,128,1,0,100.0,1.0,1.0,1000.0,1.0,128,140,1,0,length\n"
        "x,128,1,1,100.0,1.0,1.0,1000.0,1.0,128,140,1,0,length\n"
    )
    # 1 warmup + 2 measured = 3 server roots.
    # request_id must be non-empty so parse_line (\S+) can match it.
    # Production mode does not have request_id in bench.csv; server always logs real UUIDs.
    log_lines = []
    for i in range(3):
        start_ns = 1000000000 + i * 200000000
        end_ns = start_ns + 100_000_000  # 100ms inclusive
        rid = f"fake-uuid-{i:04d}"
        log_lines.append(
            f"[p5h-profile] request_id={rid} routing_path=scheduler prompt_tokens=140 "
            f"seq=0 layer_idx=-1 chunk_idx=null span_id=1 parent_span_id=null "
            f"span_name=root parent_span=null start_ns={start_ns} end_ns={end_ns} "
            f"mode=off span_kind=tree\n"
        )
    cell = _write_cell(tmp_path, 128, "production", bench, "".join(log_lines))
    v = decompose_cell(cell)
    assert v.warmup_count == DEFAULT_PRODUCTION_WARMUP
    assert v.legacy_warmup_inferred is True
    assert len(v.runs) == 2  # measured rows only after dropping warmup


def test_decompose_inconclusive_on_root_count_mismatch(tmp_path):
    bench = (
        "target,pp_target,tg_target,run_idx,ttft_ms,tg_tps,tpot_ms,pp_tps,e2e_s,prompt_tokens_local,prompt_tokens_server,completion_tokens_server,cached_tokens,finish_reason\n"
        "x,128,1,0,100.0,1.0,1.0,1000.0,1.0,128,140,1,0,length\n"
        "x,128,1,1,100.0,1.0,1.0,1000.0,1.0,128,140,1,0,length\n"
    )
    # 1 warmup + 2 measured = 3 expected; provide only 2 → inconclusive.
    # request_id must be non-empty so parse_line (\S+) can parse the line.
    log_lines = []
    for i in range(2):
        log_lines.append(
            f"[p5h-profile] request_id=fake-uuid-{i:04d} routing_path=scheduler prompt_tokens=140 "
            "seq=0 layer_idx=-1 chunk_idx=null span_id=1 parent_span_id=null "
            "span_name=root parent_span=null start_ns=1000 end_ns=101000 "
            "mode=off span_kind=tree\n"
        )
    cell = _write_cell(tmp_path, 128, "production", bench, "".join(log_lines))
    v = decompose_cell(cell)
    assert v.verdict == "inconclusive"
    assert "root count" in v.note


def test_decompose_inconclusive_on_prompt_token_mismatch(tmp_path):
    bench = (
        "target,pp_target,tg_target,run_idx,ttft_ms,tg_tps,tpot_ms,pp_tps,e2e_s,prompt_tokens_local,prompt_tokens_server,completion_tokens_server,cached_tokens,finish_reason\n"
        "x,128,1,0,100.0,1.0,1.0,1000.0,1.0,128,140,1,0,length\n"
    )
    # 1 warmup (prompt_tokens=140 OK) + 1 measured (prompt_tokens=524, wrong → mismatch).
    # request_id must be non-empty so parse_line (\S+) can parse the line.
    log = (
        "[p5h-profile] request_id=fake-warmup-0000 routing_path=scheduler prompt_tokens=140 "
        "seq=0 layer_idx=-1 chunk_idx=null span_id=1 parent_span_id=null "
        "span_name=root parent_span=null start_ns=1000 end_ns=101000 "
        "mode=off span_kind=tree\n"
        "[p5h-profile] request_id=fake-mismatch-0001 routing_path=scheduler prompt_tokens=524 "
        "seq=0 layer_idx=-1 chunk_idx=null span_id=1 parent_span_id=null "
        "span_name=root parent_span=null start_ns=2000 end_ns=102000 "
        "mode=off span_kind=tree\n"
    )
    cell = _write_cell(tmp_path, 128, "production", bench, log)
    v = decompose_cell(cell)
    assert v.verdict == "inconclusive"
    assert "prompt_tokens mismatch" in v.note
