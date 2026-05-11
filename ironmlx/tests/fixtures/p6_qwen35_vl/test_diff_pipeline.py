"""Unit tests for diff_pipeline.py.

Run from within the cxx-mlx repo:
    ~/.venvs/mlxvlm-ref/bin/python -m pytest \
        ironmlx/tests/fixtures/p6_qwen35_vl/test_diff_pipeline.py -v
"""
import json
import os
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent))
import diff_pipeline


def _save(tmp: Path, name: str, arr) -> None:
    if isinstance(arr, np.ndarray):
        arr = mx.array(arr)
    mx.eval(arr)
    mx.save_safetensors(str(tmp / f"{name}.safetensors"), {"tensor": arr})


def test_diff_stats_identical():
    a = np.zeros((4, 4), dtype=np.float32)
    b = np.zeros((4, 4), dtype=np.float32)
    s = diff_pipeline.diff_stats(a, b)
    assert s["max"] == 0.0
    assert s["mean"] == 0.0
    assert s["count_above_1e-3"] == 0
    assert s["total"] == 16


def test_diff_stats_offset():
    a = np.zeros((4, 4), dtype=np.float32)
    b = np.full((4, 4), 0.5, dtype=np.float32)
    s = diff_pipeline.diff_stats(a, b)
    assert s["max"] == pytest.approx(0.5)
    assert s["mean"] == pytest.approx(0.5)
    assert s["count_above_1e-3"] == 16
    assert s["count_above_1e-2"] == 16
    assert s["count_above_1e-1"] == 16


def test_diff_stats_single_outlier():
    a = np.zeros((10,), dtype=np.float32)
    b = np.zeros((10,), dtype=np.float32)
    b[3] = 0.85
    s = diff_pipeline.diff_stats(a, b)
    assert s["max"] == pytest.approx(0.85)
    assert s["count_above_1e-1"] == 1
    assert s["count_above_1e-2"] == 1


def test_pair_files_skips_unpaired_with_warning(capsys):
    with tempfile.TemporaryDirectory() as tmp:
        py_dir = Path(tmp) / "py"
        rust_dir = Path(tmp) / "rust"
        py_dir.mkdir()
        rust_dir.mkdir()
        _save(py_dir, "00_pixel_values", np.zeros((2, 2), dtype=np.float32))
        _save(py_dir, "01_patch_embed_out", np.zeros((2, 2), dtype=np.float32))
        _save(rust_dir, "01_patch_embed_out", np.zeros((2, 2), dtype=np.float32))
        # No matching 00 in rust → should be skipped with warning
        pairs, unpaired = diff_pipeline.pair_files(py_dir, rust_dir)
        assert len(pairs) == 1
        assert pairs[0][0] == "01_patch_embed_out"
        assert "00_pixel_values" in unpaired["py_only"]


def test_pair_files_shape_mismatch_raises():
    with tempfile.TemporaryDirectory() as tmp:
        py_dir = Path(tmp) / "py"
        rust_dir = Path(tmp) / "rust"
        py_dir.mkdir()
        rust_dir.mkdir()
        _save(py_dir, "01_foo", np.zeros((4, 4), dtype=np.float32))
        _save(rust_dir, "01_foo", np.zeros((4, 8), dtype=np.float32))
        with pytest.raises(ValueError, match="shape mismatch"):
            diff_pipeline.diff_pair(py_dir / "01_foo.safetensors",
                                    rust_dir / "01_foo.safetensors")


def test_top_outliers_returns_top_n():
    a = np.zeros((10,), dtype=np.float32)
    b = np.array([0.0, 0.1, 0.3, 0.5, 0.0, 0.2, 0.4, 0.0, 0.6, 0.0],
                 dtype=np.float32)
    out = diff_pipeline.top_outliers(a, b, n=3)
    assert len(out) == 3
    # Sorted descending by abs diff
    assert out[0]["idx"] == 8 and out[0]["diff"] == pytest.approx(0.6)
    assert out[1]["idx"] == 3 and out[1]["diff"] == pytest.approx(0.5)
    assert out[2]["idx"] == 6 and out[2]["diff"] == pytest.approx(0.4)


def test_render_report_contains_required_sections(tmp_path):
    rows = [
        {"name": "01_patch_embed_out", "shape": [1200, 1024],
         "max": 0.001, "mean": 0.0001, "p99": 0.001,
         "count_above_1e-3": 12, "count_above_1e-2": 0, "count_above_1e-1": 0,
         "total": 1228800},
        {"name": "10_block_05_out", "shape": [1200, 1024],
         "max": 0.156, "mean": 0.01, "p99": 0.1,
         "count_above_1e-3": 50000, "count_above_1e-2": 100, "count_above_1e-1": 5,
         "total": 1228800},
    ]
    text = diff_pipeline.render_report(rows, rupture="10_block_05_out", top_outliers=[])
    assert "## Summary" in text
    assert "## Per-tensor table" in text
    assert "10_block_05_out" in text
