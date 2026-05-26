"""Smoke pytest for tools/p5h_2b_protocol_experiment.py P5h+2.d cooldown
pass-through. Verifies --inter-run-cooldown-secs is propagated to the
P5I_C_INTER_RUN_COOLDOWN_SECS env var the capture harness reads, and Rule D
server-log scan hard-fails ERROR lines while surfacing non-allow-listed WARNs.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

TOOLS_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TOOLS_DIR))

import p5h_2b_protocol_experiment as drv  # noqa: E402


def _base_argv() -> list[str]:
    return [
        "p5h_2b_protocol_experiment.py",
        "--phase", "t1",
        "--exp-id", "smoke",
        "--server-lifecycle", "same_spawn_per_pp",
        "--pp-order", "128",
        "--logging-mode", "quiet_acceptance",
        "--mode", "production",
        "--repeats", "1",
        "--pps", "128",
        "--runs-per-pp", "128:15",
        "--model-dir", "/tmp/model",
        "--mlx-dir", "/tmp/mlx",
        "--out-base", "/tmp/p5h-2b-smoke",
        "--skip-envelope",
    ]


def test_cooldown_env_var_propagated_when_flag_set():
    """If --inter-run-cooldown-secs is set, the env passed to cargo test
    must include P5I_C_INTER_RUN_COOLDOWN_SECS=<N>."""
    captured_env: dict[str, str] = {}

    def fake_run(cmd, cwd, env, stdout, stderr, check):  # noqa: ARG001
        captured_env.update(env)
        result = MagicMock()
        result.returncode = 1  # force run_one_repeat to raise before relocate
        return result

    argv = _base_argv() + ["--inter-run-cooldown-secs", "60"]
    with patch.object(sys, "argv", argv), patch.object(
        drv.subprocess, "run", side_effect=fake_run
    ), patch("builtins.open"):
        try:
            drv.main()
        except SystemExit:
            pass

    assert captured_env.get("P5I_C_INTER_RUN_COOLDOWN_SECS") == "60", (
        f"expected env P5I_C_INTER_RUN_COOLDOWN_SECS=60; "
        f"got {captured_env.get('P5I_C_INTER_RUN_COOLDOWN_SECS')!r}"
    )


def test_cooldown_env_var_absent_when_flag_not_set():
    """Default 0 cooldown: env must NOT have P5I_C_INTER_RUN_COOLDOWN_SECS
    (preserves harness byte-identity for legacy invocations)."""
    captured_env: dict[str, str] = {}

    def fake_run(cmd, cwd, env, stdout, stderr, check):  # noqa: ARG001
        captured_env.update(env)
        result = MagicMock()
        result.returncode = 1
        return result

    argv = _base_argv()
    with patch.object(sys, "argv", argv), patch.object(
        drv.subprocess, "run", side_effect=fake_run
    ), patch("builtins.open"):
        try:
            drv.main()
        except SystemExit:
            pass

    assert "P5I_C_INTER_RUN_COOLDOWN_SECS" not in captured_env, (
        f"expected no P5I_C_INTER_RUN_COOLDOWN_SECS env when flag absent; "
        f"got {captured_env.get('P5I_C_INTER_RUN_COOLDOWN_SECS')!r}"
    )


def test_scan_server_log_hard_fails_any_error(tmp_path: Path):
    log = tmp_path / "server.log"
    log.write_text(
        "2026-05-25T00:00:00Z INFO ok\n"
        "2026-05-25T00:00:01Z ERROR non-scheduler failure\n"
    )
    with pytest.raises(SystemExit) as exc:
        drv.scan_server_log(log, allow_server_errors=False)
    assert "ERROR lines detected" in str(exc.value)


def test_scan_server_log_marks_non_allowlisted_warn(tmp_path: Path):
    log = tmp_path / "server.log"
    log.write_text(
        "2026-05-25T00:00:00Z WARN unexpected thermal warning\n"
        "2026-05-25T00:00:01Z WARN [tracing] initialization warning\n"
    )
    scan = drv.scan_server_log(log, allow_server_errors=False)
    assert scan["error_count"] == 0
    assert scan["allowlisted_warn_count"] == 1
    assert scan["non_allowlisted_warn_count"] == 1
    assert "unexpected thermal warning" in scan["non_allowlisted_warn_lines"][0]


def test_scan_server_log_does_not_false_positive_on_message_body(tmp_path: Path):
    """Tracing-format level field anchoring: substring 'ERROR' or 'WARN' in the
    MESSAGE body (not as the level token) must NOT trigger classification.

    Regression guard for code-quality reviewer I1 — without level-field
    anchoring, the 'no ERROR detected' INFO line below would cause a sweep
    to abort spuriously.
    """
    log = tmp_path / "server.log"
    log.write_text(
        "2026-05-25T00:00:00Z  INFO target: no ERROR detected this cycle\n"
        "2026-05-25T00:00:01Z  INFO target: downward_WARN_compatible flag set\n"
        "2026-05-25T00:00:02Z  WARN target: real warning to allow-list logic\n"
        "2026-05-25T00:00:03Z ERROR target: real error\n"
    )
    # `allow_server_errors=True` so SystemExit does not fire mid-test; we are
    # asserting classification, not termination.
    scan = drv.scan_server_log(log, allow_server_errors=True)
    assert scan["error_count"] == 1, (
        f"expected exactly 1 ERROR (line 4); got {scan['error_count']}"
    )
    assert scan["non_allowlisted_warn_count"] + scan["allowlisted_warn_count"] == 1, (
        f"expected exactly 1 WARN classified (line 3); got "
        f"{scan['non_allowlisted_warn_count']} non-allowlisted + "
        f"{scan['allowlisted_warn_count']} allowlisted"
    )


def test_preheat_pp_list_env_var_propagated_when_flag_set():
    """--preheat-pp-list 'CSV' propagates to P5I_C_PREHEAT_PP_LIST env var
    passed to the cargo test subprocess."""
    captured_env: dict[str, str] = {}

    def fake_run(cmd, cwd, env, stdout, stderr, check):  # noqa: ARG001
        captured_env.update(env)
        result = MagicMock()
        result.returncode = 1  # force run_one_repeat to raise before relocate
        return result

    argv = _base_argv() + ["--preheat-pp-list", "512,{pp}"]
    with patch.object(sys, "argv", argv), patch.object(
        drv.subprocess, "run", side_effect=fake_run
    ), patch("builtins.open"):
        try:
            drv.main()
        except SystemExit:
            pass

    assert captured_env.get("P5I_C_PREHEAT_PP_LIST") == "512,{pp}", (
        f"expected env P5I_C_PREHEAT_PP_LIST='512,{{pp}}'; "
        f"got {captured_env.get('P5I_C_PREHEAT_PP_LIST')!r}"
    )


def test_preheat_pp_list_env_var_absent_when_flag_not_set():
    """Default (no --preheat-pp-list): env MUST NOT have P5I_C_PREHEAT_PP_LIST
    (preserves harness baseline behavior for legacy invocations)."""
    captured_env: dict[str, str] = {}

    def fake_run(cmd, cwd, env, stdout, stderr, check):  # noqa: ARG001
        captured_env.update(env)
        result = MagicMock()
        result.returncode = 1
        return result

    argv = _base_argv()
    with patch.object(sys, "argv", argv), patch.object(
        drv.subprocess, "run", side_effect=fake_run
    ), patch("builtins.open"):
        try:
            drv.main()
        except SystemExit:
            pass

    assert "P5I_C_PREHEAT_PP_LIST" not in captured_env, (
        f"expected no P5I_C_PREHEAT_PP_LIST env when flag absent; "
        f"got {captured_env.get('P5I_C_PREHEAT_PP_LIST')!r}"
    )


def test_nonce_seed_env_var_propagated_when_flag_set():
    """--nonce-seed N propagates to P5I_C_NONCE_SEED for reproducible
    iron-bench prompt nonce sequences in T2.A acceptance sweeps."""
    captured_env: dict[str, str] = {}

    def fake_run(cmd, cwd, env, stdout, stderr, check):  # noqa: ARG001
        captured_env.update(env)
        result = MagicMock()
        result.returncode = 1
        return result

    argv = _base_argv() + ["--nonce-seed", "20260526"]
    with patch.object(sys, "argv", argv), patch.object(
        drv.subprocess, "run", side_effect=fake_run
    ), patch("builtins.open"):
        try:
            drv.main()
        except SystemExit:
            pass

    assert captured_env.get("P5I_C_NONCE_SEED") == "20260526", (
        f"expected env P5I_C_NONCE_SEED=20260526; "
        f"got {captured_env.get('P5I_C_NONCE_SEED')!r}"
    )


def test_nonce_seed_env_var_absent_when_flag_not_set():
    """Default unset nonce seed: env must NOT have P5I_C_NONCE_SEED so legacy
    time-based nonce generation remains the default behavior."""
    captured_env: dict[str, str] = {}

    def fake_run(cmd, cwd, env, stdout, stderr, check):  # noqa: ARG001
        captured_env.update(env)
        result = MagicMock()
        result.returncode = 1
        return result

    argv = _base_argv()
    with patch.object(sys, "argv", argv), patch.object(
        drv.subprocess, "run", side_effect=fake_run
    ), patch("builtins.open"):
        try:
            drv.main()
        except SystemExit:
            pass

    assert "P5I_C_NONCE_SEED" not in captured_env, (
        f"expected no P5I_C_NONCE_SEED env when flag absent; "
        f"got {captured_env.get('P5I_C_NONCE_SEED')!r}"
    )
