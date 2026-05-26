# P5h+2.e PP=128 ironmlx-specific within-CI Residual Investigation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Investigate PP=128 ironmlx-specific within-CI 4-5% residual (confirmed IRONMLX-SPECIFIC by P5h+2.d δ) via equal-budget same-shape preheat protocol; produce evidence + recommendation that either backfills Phase 0 § 7 #4 PASS or triggers T2 H_small_batch investigation.

**Architecture:** T0 ships harness `P5I_C_PREHEAT_PP_LIST` env var + `{pp}` token substitution + driver pass-through + audit metadata + defensive validation. T1 runs 6-cell (PP=128 + PP=512 × 3 repeats) equal-budget 550+550 same-shape preheat sweep + computes pp_tps envelope per spec § 3.1. T2 (gated on T1 weak/FAIL + Boss approval) adds iron-bench `--nonce-seed` + opt-in MoE expert occupancy logging with T2.A/T2.B acceptance/diagnostic split. T3 single commit packages all infra + tests + docs + Phase 0 backfill + Phase 1 spec § 2.3 protocol coupling per spec § 9.4 single-commit policy.

**Tech Stack:** Rust (capture harness `monolithic_preheat` + iron-bench Args), Python (`tools/p5h_2b_protocol_experiment.py` driver + `tools/p5i_c_pp_tps_envelope.py` aggregator), pytest, cargo test.

**Spec:** `docs/superpowers/specs/2026-05-26-ironmlx-p5h+2-e-pp128-investigation-design.md` (commit `dbcb03f`).

**Predecessor close-outs:** `docs/p5h+2-d-close-out.md` § 6 (P5h+2.e direction binding); `docs/p5i-c-phase-0-close-out.md` § 1 #4 (STILL FAIL/DEFERRED awaiting this phase).

**Branch baseline:** Current HEAD includes the approved spec commit `dbcb03f` on top of `110a181`. Start T0 from that HEAD on `ironmlx-p5h+2-e-pp128-investigation`; do not rebase/drop the spec commit during implementation.

**Single-commit discipline (spec § 9.4):** T0-T2 produce WIP only. T3 makes ONE commit attaching all changes + close-out doc + Phase 0 + Phase 1 spec backfill. Each non-T3 task ends with "Stop and report DONE; DO NOT commit".

---

## File Structure

| Path | Role | Touched in task |
|---|---|---|
| `ironmlx/tests/p5i_c_phase_0_capture.rs` | Capture harness — `monolithic_preheat` env var + `{pp}` substitution + cycle + audit; main reads `P5I_C_PREHEAT_PP_LIST`; lifecycle handlers thread pp_list_template; meta.json audit fields | T0 (modify) |
| `tools/p5h_2b_protocol_experiment.py` | Driver — `--preheat-pp-list` CLI arg + env pass-through | T0 (modify) |
| `tools/p5h_aggregator/tests/test_p5h_2b_protocol_experiment.py` | Driver smoke pytest — env pass-through verification | T0 (modify, append) |
| `iron-bench/src/main.rs` | iron-bench Args — `--nonce-seed N` CLI flag (T2 only) | T2 (modify) |
| `iron-bench/src/runner.rs` | iron-bench nonce semantics: `N ^ (run_idx << 8)` instead of `nonce_seed() ^ (run_idx << 8)` when `--nonce-seed` set (T2 only) | T2 (modify) |
| `iron-bench/tests/nonce_seed.rs` (NEW; T2 only) | iron-bench nonce-seed CLI parse smoke test; exact sequence semantics live in `runner.rs` unit tests | T2 (create) |
| `ironmlx/src/models/qwen3_5_moe/sparse_moe.rs` | MoE expert occupancy diagnostic log gated by `IRONMLX_EXPERT_OCCUPANCY_LOG=1` (T2 only) | T2 (modify) |
| `docs/superpowers/plans/2026-05-26-ironmlx-p5h+2-e-pp128-investigation.md` | This implementation plan; must be included in the T3 single commit if not already committed | T3 (add/modify) |
| `docs/p5h+2-e-close-out.md` (NEW) | T3 close-out per spec § 3.2 outcome | T3 (create) |
| `docs/p5i-c-phase-0-close-out.md` | Phase 0 § 1 #4 backfill per spec § 8 (actual envelope numbers as evidence, NOT just narrative) | T3 (modify) |
| `docs/p5i-c-phase-0-ranking-snapshot.md` | Preamble update with P5h+2.e closure pointer | T3 (modify) |
| `docs/superpowers/specs/2026-05-25-ironmlx-p5i-c-phase-1-gather-qmm-gate-up-design.md` | § 2.3 measurement protocol binding update — reference P5h+2.e-resolved protocol | T3 (modify; only if T1 or T2 PASS) |

Output (host, NOT committed):
- `/tmp/p5h+2-e-t1-e-t1-r${R}-pp${PP}/{bench.csv,server.log,meta.json,server_log_scan.json}` (T1 cells; driver appends `-{exp_id}-rN-ppP` to `--out-base`)
- `/tmp/p5h+2-e-t1-pp${PP}-envelope.json` (per-PP envelope JSON)
- `/tmp/p5h+2-e-t1-verdict.md` (T1 acceptance verdict working notes)
- `/tmp/p5h+2-e-t2a-e-t2a-r${R}-pp${PP}/{bench.csv,server.log,meta.json,server_log_scan.json}` + `/tmp/p5h+2-e-t2b-e-t2b-r${R}-pp${PP}/{bench.csv,server.log,meta.json,server_log_scan.json}` (T2 only, gated)

---

## Predeclared exclusion rules (lock BEFORE T1 begins; spec § 6)

- **Rule B** (drop first 1-2 cold-start runs): OFF by default; allowed only as a predeclared, tested tool option implemented BEFORE the sweep starts, applied uniformly to all cells, recorded in envelope JSON + cell `meta.json`. Manual trim FORBIDDEN.
- **Rule C** (conditional drop last N): REMOVED.
- **Rule D**: any server.log ERROR line → cell FAILS hard-stop (inherits from `tools/p5h_2b_protocol_experiment.py` scan_server_log already shipped in P5h+2.d); non-allow-listed WARN → human review.
- **Rule E** (post-hoc trim): REMOVED. Codex round-1 explicit prohibition.

T1 acceptance gate is the existing `tools/p5i_c_pp_tps_envelope.py` all-runs envelope (no Rule B trim activation in this plan).

---

## Task 0: harness preheat protocol + driver pass-through + audit metadata

**Files:**
- Modify: `ironmlx/tests/p5i_c_phase_0_capture.rs`:
  - Doc header (~line 22-35): add `P5I_C_PREHEAT_PP_LIST` env var documentation
  - Add `DEFAULT_PREHEAT_PP_LIST` constant (~line 56 area where DEFAULTs live)
  - Add `parse_preheat_pp_list(template: &str, measured_pp: i32) -> Vec<i32>` function with defensive validation
  - Modify `fn monolithic_preheat` (line 324) signature to accept `pp_list: &[i32]` instead of hardcoded "512"; loop over PPs
  - Modify 3 lifecycle handler call sites (lines 546, 654, 773) to substitute `{pp}` per measured PP and pass `pp_list`
  - Modify lifecycle handler signatures (lines 524, 624, 743) to take `preheat_pp_list_template: &str`
  - Modify main (line 824 area) to read `P5I_C_PREHEAT_PP_LIST` env var
  - Modify meta.json emit (search `meta.json`) to add audit fields `preheat_pp_list_effective` + `preheat_runs_per_shape` + `preheat_total_runs_effective`
- Modify: `tools/p5h_2b_protocol_experiment.py`:
  - Add `--preheat-pp-list` CLI arg (after `--preheat-runs` at line 264)
  - Add env pass-through in `run_one_repeat` beside the existing `P5I_C_*` environment assignments
- Modify: `tools/p5h_aggregator/tests/test_p5h_2b_protocol_experiment.py`:
  - Append 2 smoke pytests verifying `--preheat-pp-list "512,{pp}"` propagates to `P5I_C_PREHEAT_PP_LIST` and absent flag does not set the env var

### T0.A — Rust harness: env var reader + parser + defensive validation

- [ ] **Step A1: Add documentation in file doc-comment header**

In `ironmlx/tests/p5i_c_phase_0_capture.rs`, in the top doc-comment block (~line 22-35 where other `P5I_C_*` env vars are listed), append after the `P5I_C_PREHEAT_RUNS` line:

```rust
//!   * `P5I_C_PREHEAT_PP_LIST` — comma-separated PP list for monolithic
//!     preheat with `{pp}` token substituted to the measured PP per cell.
//!     Default `"512"` (legacy single-shape behavior; backward-compatible).
//!     P5h+2.e T1 sets `"512,{pp}"` for equal-budget same-shape preheat.
```

- [ ] **Step A2: Add DEFAULT constant**

In the same file, near the `DEFAULT_PREHEAT_*` constants (search `DEFAULT_PREHEAT_RUNS`), append:

```rust
const DEFAULT_PREHEAT_PP_LIST: &str = "512";
```

- [ ] **Step A3: Add parse_preheat_pp_list function with defensive validation**

In the same file, near `fn env_or` (line 62) or near other parser helpers, add:

```rust
/// Parse the `P5I_C_PREHEAT_PP_LIST` template into a concrete PP list for
/// the given `measured_pp`. Substitutes the literal `{pp}` token (case-
/// sensitive) with the measured PP integer.
///
/// Defensive validation per P5h+2.e spec § 2.2: rejects empty input,
/// empty entries, non-positive PPs, and any unresolved `{pp}` token
/// (only the exact literal `{pp}` is substituted; any other brace-form
/// is a typo and we fail loudly).
fn parse_preheat_pp_list(template: &str, measured_pp: i32) -> Vec<i32> {
    let raw = template.trim();
    if raw.is_empty() {
        panic!("P5I_C_PREHEAT_PP_LIST is empty");
    }
    let mut out: Vec<i32> = Vec::new();
    for entry in raw.split(',') {
        let entry = entry.trim();
        if entry.is_empty() {
            panic!("P5I_C_PREHEAT_PP_LIST contains an empty entry: {raw:?}");
        }
        let resolved: String = if entry == "{pp}" {
            measured_pp.to_string()
        } else {
            // Reject any other brace-form so typos like `{Pp}` or `${pp}`
            // fail loudly instead of silently producing garbage.
            if entry.contains('{') || entry.contains('}') {
                panic!(
                    "P5I_C_PREHEAT_PP_LIST entry {entry:?} contains an unresolved \
                     brace token; only the literal `{{pp}}` is substituted"
                );
            }
            entry.to_string()
        };
        let pp: i32 = resolved
            .parse()
            .unwrap_or_else(|e| panic!("P5I_C_PREHEAT_PP_LIST entry {resolved:?} parse: {e}"));
        if pp <= 0 {
            panic!("P5I_C_PREHEAT_PP_LIST entry {pp} must be > 0");
        }
        out.push(pp);
    }
    if out.is_empty() {
        panic!("P5I_C_PREHEAT_PP_LIST parsed to zero entries: {raw:?}");
    }
    out
}
```

- [ ] **Step A4: Add 4 unit tests for parse_preheat_pp_list**

In the same file, at the existing `#[cfg(test)] mod tests` block (search for it; if it doesn't exist, add a new one at the bottom of the file), append:

```rust
#[cfg(test)]
mod tests_preheat_pp_list {
    use super::parse_preheat_pp_list;

    #[test]
    fn substitutes_pp_token() {
        assert_eq!(parse_preheat_pp_list("512,{pp}", 128), vec![512, 128]);
        assert_eq!(parse_preheat_pp_list("512,{pp}", 512), vec![512, 512]);
    }

    #[test]
    fn parses_pure_static_list() {
        assert_eq!(parse_preheat_pp_list("512", 128), vec![512]);
        assert_eq!(parse_preheat_pp_list("128,256,512", 128), vec![128, 256, 512]);
    }

    #[test]
    #[should_panic(expected = "empty")]
    fn rejects_empty_template() {
        let _ = parse_preheat_pp_list("", 128);
    }

    #[test]
    #[should_panic(expected = "unresolved brace token")]
    fn rejects_unrecognized_brace_token() {
        // `{Pp}` is a typo; only literal `{pp}` is substituted
        let _ = parse_preheat_pp_list("512,{Pp}", 128);
    }
}
```

- [ ] **Step A5: Run parser unit tests**

```bash
cargo test --release -p ironmlx --features p5h-profile --test p5i_c_phase_0_capture tests_preheat_pp_list
```

Expected: 4 tests run; A1+A2 PASS (parser body just compiled in), A3+A4 PASS (panics caught). All 4 should PASS after Step A3 because `parse_preheat_pp_list` body is complete.

(TDD spirit: parser is small + pure, so tests double as spec.)

### T0.B — Rust harness: thread pp_list through monolithic_preheat + 3 lifecycle handlers

- [ ] **Step B1: Modify `monolithic_preheat` signature + body**

In `ironmlx/tests/p5i_c_phase_0_capture.rs`, replace the existing `fn monolithic_preheat` (line 324) with:

```rust
fn monolithic_preheat(
    model_dir: &str,
    preheat_seconds: u64,
    preheat_runs: usize,
    preheat_pp_list: &[i32],
) -> std::io::Result<u64> {
    if preheat_pp_list.is_empty() {
        return Err(std::io::Error::other(
            "monolithic_preheat: preheat_pp_list is empty (defensive guard)",
        ));
    }
    let start = Instant::now();
    let pp_list_csv = preheat_pp_list
        .iter()
        .map(|pp| pp.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let out = Command::new("cargo")
        .args([
            "run",
            "--release",
            "-p",
            "iron-bench",
            "--",
            "--target",
            &format!("preheat=http://127.0.0.1:{PORT}"),
            "--model",
            &iron_bench_model_token(),
            "--model-dir",
            model_dir,
            "--prompt-len",
            &pp_list_csv,
            "--max-tokens",
            "1",
            "--runs",
            &preheat_runs.to_string(),
            "--warmup",
            "0",
            "--format",
            "csv",
        ])
        .output()?;
    let wall_s = start.elapsed().as_secs();
    if !out.status.success() {
        return Err(std::io::Error::other(format!(
            "preheat non-success: stderr={}",
            String::from_utf8_lossy(&out.stderr)
        )));
    }
    if wall_s < preheat_seconds {
        eprintln!(
            "[p5i-c WARN] preheat wall {wall_s}s < target {preheat_seconds}s; \
             consider bumping P5I_C_PREHEAT_RUNS (current {preheat_runs} per PP, \
             PP list: {pp_list_csv})"
        );
    }
    Ok(wall_s)
}
```

Note: `iron-bench --prompt-len 512,128 --runs 550` cycles 550 runs per PP per its existing `for pp in &args.prompt_len { for (target_name, target_url) in &args.target { runner::run_cell(...) }}` loop. Total runs = `preheat_runs × pp_list.len()` (e.g., 550 × 2 = 1100 for P5h+2.e T1).

- [ ] **Step B2: Modify 3 lifecycle handler signatures + call sites**

In the same file, locate the 3 lifecycle handlers:
- `fn run_phase0_current(...)` (signature line ~520; monolithic_preheat call line 546)
- `fn run_same_spawn_cross_pp(...)` (signature line ~620; call line 654)
- `fn run_same_spawn_per_pp(...)` (signature line ~740; call line 773)

For each, add `preheat_pp_list_template: &str` parameter (after `preheat_runs: usize`).

For each call site, replace `monolithic_preheat(model_dir, preheat_seconds, preheat_runs)` with the appropriate per-call pp_list resolution:

For `run_phase0_current` (no measured PP at preheat time; uses first PP from `pp_order` as `measured_pp` for `{pp}` substitution — legacy mode, may not even use `{pp}` in template):

```rust
    let measured_pp_for_substitution = pp_order.first().copied().unwrap_or(512);
    let preheat_pp_list = parse_preheat_pp_list(
        preheat_pp_list_template,
        measured_pp_for_substitution,
    );
    let preheat_wall = match monolithic_preheat(
        model_dir,
        preheat_seconds,
        preheat_runs,
        &preheat_pp_list,
    ) {
```

For `run_same_spawn_cross_pp` (preheat runs ONCE before all PP cells; same legacy treatment — first PP for substitution):

```rust
    let measured_pp_for_substitution = pp_order.first().copied().unwrap_or(512);
    let preheat_pp_list = parse_preheat_pp_list(
        preheat_pp_list_template,
        measured_pp_for_substitution,
    );
    let preheat_wall = match monolithic_preheat(
        model_dir,
        preheat_seconds,
        preheat_runs,
        &preheat_pp_list,
    ) {
```

For `run_same_spawn_per_pp` (preheat runs PER PP; this is the lifecycle P5h+2.e T1 uses; `{pp}` substitution uses the loop's current PP):

In this handler, the existing per-PP `for pp in pp_order { ... }` loop at line ~770. The call at line 773 is inside that loop. Replace with:

```rust
        let preheat_pp_list = parse_preheat_pp_list(preheat_pp_list_template, pp);
        let preheat_wall = match monolithic_preheat(
            model_dir,
            preheat_seconds,
            preheat_runs,
            &preheat_pp_list,
        ) {
```

(Use the loop variable `pp` for `{pp}` substitution.)

- [ ] **Step B3: Modify main to read env var + thread to handlers**

In the same file, locate the main entry (line ~820 area) where `preheat_runs` is read. After the `preheat_runs` read, add:

```rust
    let preheat_pp_list_template = env_or("P5I_C_PREHEAT_PP_LIST", DEFAULT_PREHEAT_PP_LIST);
```

Then update the 3 lifecycle handler call sites (lines ~847, ~859, ~871 area) to pass `&preheat_pp_list_template` as the new parameter.

- [ ] **Step B4: Audit metadata in meta.json — preheat_pp_list_effective + preheat_runs_per_shape + preheat_total_runs_effective**

`meta.json` is built by the manual `format!` in `write_cell_meta`, not by `serde_json`. Update `write_cell_meta` to receive the effective preheat list and run count. Add these two parameters immediately before `ts: CellTimestamps`:

```rust
    preheat_pp_list: &[i32],
    preheat_runs: usize,
```

After the existing `pp_order_s` computation, insert:

```rust
    let preheat_pp_list_json = format!(
        "[{}]",
        preheat_pp_list
            .iter()
            .map(|p| p.to_string())
            .collect::<Vec<_>>()
            .join(",")
    );
    let preheat_total_runs_effective = preheat_runs * preheat_pp_list.len();
```

Then add these fields inside the existing JSON string, before `"port"`:

```rust
         \"preheat_pp_list_effective\": {preheat_pp_list_json},\n  \
         \"preheat_runs_per_shape\": {preheat_runs},\n  \
         \"preheat_total_runs_effective\": {preheat_total_runs_effective},\n  \
```

Update all three lifecycle handlers to pass the effective `preheat_pp_list` from Step B2 and `preheat_runs` into every `write_cell_meta(...)` call. For `same_spawn_cross_pp`, every PP cell gets the same effective preheat list because the preheat is shared across PP cells; for `same_spawn_per_pp`, each PP cell gets the loop-local substituted list.

- [ ] **Step B5: Run cargo gates**

```bash
cargo fmt
cargo +nightly fmt --all -- --check
cargo +nightly clippy -p ironmlx --features p5h-profile -- -D warnings
cargo build --release --tests -p ironmlx --features p5h-profile
```

Expected: zero clippy warnings on ironmlx scope; release-tests build OK.

- [ ] **Step B6: Run capture-harness test compilation list (no GPU execution)**

```bash
cargo test --release -p ironmlx --features p5h-profile --test p5i_c_phase_0_capture -- --list
```

Expected: tests list including the 4 new `tests_preheat_pp_list::*` entries from T0.A.

- [ ] **Step B7: Re-run T0.A unit tests after wiring**

```bash
cargo test --release -p ironmlx --features p5h-profile --test p5i_c_phase_0_capture tests_preheat_pp_list
```

Expected: all 4 tests PASS.

### T0.C — Python driver pass-through + smoke pytest (TDD)

- [ ] **Step C1: Append failing smoke pytest**

In `tools/p5h_aggregator/tests/test_p5h_2b_protocol_experiment.py`, append at end of file:

```python


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
```

- [ ] **Step C2: Run pytests to confirm failure**

```bash
uv run pytest tools/p5h_aggregator/tests/test_p5h_2b_protocol_experiment.py -v -k "preheat_pp_list"
```

Expected: 2 tests FAIL — first with "unrecognized arguments: --preheat-pp-list"; second may pass (env doesn't include the var) but the first is the active failure.

- [ ] **Step C3: Add argparse arg + env propagation**

In `tools/p5h_2b_protocol_experiment.py`:

(a) After the existing `--preheat-runs` argparse `add_argument` call (line 264), add:

```python
    p.add_argument(
        "--preheat-pp-list",
        type=str,
        default=None,
        help="Comma-separated PP list for monolithic preheat with '{pp}' token "
        "substituted to the measured PP per cell. Default unset = harness uses "
        "DEFAULT_PREHEAT_PP_LIST. P5h+2.e T1 sets '512,{pp}' for equal-budget "
        "same-shape preheat. Per P5h+2.e spec § 2.2.",
    )
```

(b) In `def run_one_repeat(args, repeat)`, AFTER the existing `if args.inter_run_cooldown_secs > 0:` block (around line 138-139), add:

```python
    if args.preheat_pp_list is not None:
        env["P5I_C_PREHEAT_PP_LIST"] = args.preheat_pp_list
```

Conditional (NOT unconditional) preserves byte-identity for legacy invocations.

- [ ] **Step C4: Run pytests to confirm pass**

```bash
uv run pytest tools/p5h_aggregator/tests/test_p5h_2b_protocol_experiment.py -v
```

Expected: ALL pytests PASS (7 prior + 2 new = 9 total).

- [ ] **Step C5: Stop and report DONE — do NOT commit**

Per spec § 9.4 single-commit policy. Report DONE to controller; controller dispatches T1 sweep.

---

## Task 1: equal-budget same-shape preheat sweep (6 cells) + Acceptance gate analysis

**Files:**
- No new code; uses T0 plumbing
- Output (host): `/tmp/p5h+2-e-t1-e-t1-r${R}-pp${PP}/{bench.csv,server.log,meta.json,server_log_scan.json}` (6 cells) + `/tmp/p5h+2-e-t1-pp${PP}-envelope.json` (per-PP envelope) + `/tmp/p5h+2-e-t1-verdict.md`

### T1.A — Pre-flight

- [ ] **Step A1: Verify model + MLX paths**

```bash
SNAP=/Users/xin/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/1e20fd8d42056f870933bf98ca6211024744f7ec
ls "$SNAP/config.json" && echo "MoE model OK"
ls "$HOME/.local/mlx" && echo "MLX_DIR OK"
```

If either fails, BLOCK; ask Boss for current model + MLX_DIR.

- [ ] **Step A2: Clean stale T1 outputs**

```bash
rm -rf /tmp/p5h+2-e-t1-* 2>/dev/null
echo "cleaned"
```

### T1.B — Launch + wait sweep

- [ ] **Step B1: Compute + print wall estimate, ask Boss confirm**

Per spec § 4.3: 6 cells × ~38 min/cell ≈ 3.8 hr + driver overhead 30 min ≈ **~4 hr GPU wall** (within spec § 7.3 8 hr T1 cap). Print this to Boss; await explicit confirm before B2 launch.

- [ ] **Step B2: Launch T1 sweep in background**

```bash
SNAP=/Users/xin/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/1e20fd8d42056f870933bf98ca6211024744f7ec
MLX_DIR=$HOME/.local/mlx
(
  uv run python tools/p5h_2b_protocol_experiment.py \
      --phase t4 --exp-id e-t1 \
      --server-lifecycle same_spawn_per_pp \
      --pp-order 128,512 \
      --logging-mode quiet_acceptance \
      --mode production \
      --repeats 3 --pps 128,512 \
      --runs-per-pp '128:15,512:15' \
      --preheat-seconds 300 --preheat-runs 550 \
      --preheat-pp-list '512,{pp}' \
      --inter-run-cooldown-secs 120 \
      --model-dir "$SNAP" --mlx-dir "$MLX_DIR" \
      --out-base /tmp/p5h+2-e-t1
  status=$?
  echo "$status" > /tmp/p5h+2-e-t1-sweep.exit
  exit "$status"
) > /tmp/p5h+2-e-t1-sweep.log 2>&1 &
echo $! > /tmp/p5h+2-e-t1-sweep.pid
echo "T1 sweep launched; pid=$(cat /tmp/p5h+2-e-t1-sweep.pid)"
sleep 3
kill -0 $(cat /tmp/p5h+2-e-t1-sweep.pid) 2>/dev/null && echo "alive after 3s" || echo "DEAD; see log"
```

- [ ] **Step B3: Wait for sweep completion via shell polling**

```bash
PID=$(cat /tmp/p5h+2-e-t1-sweep.pid)
while kill -0 "$PID" 2>/dev/null; do
  date
  tail -20 /tmp/p5h+2-e-t1-sweep.log
  sleep 1800
done
EXIT=$(cat /tmp/p5h+2-e-t1-sweep.exit 2>/dev/null || echo missing)
echo "T1 sweep exit=$EXIT"
test "$EXIT" = "0"
```

Expected: exit `0`. If exit is missing/non-zero, inspect `/tmp/p5h+2-e-t1-sweep.log` and block before B4.

- [ ] **Step B4: Verify 6 cells present + Rule D scan clean**

```bash
ls -d /tmp/p5h+2-e-t1-e-t1-r*-pp*/ | wc -l   # expect 6
TOTAL_ERR=0
for d in /tmp/p5h+2-e-t1-e-t1-r*-pp*/; do
  if [ -f "$d/server_log_scan.json" ]; then
    n=$(uv run python -c "import json; print(json.load(open('$d/server_log_scan.json'))['error_count'])")
    TOTAL_ERR=$((TOTAL_ERR + n))
  else
    echo "MISSING: $d/server_log_scan.json"
  fi
done
echo "Rule D total ERROR across 6 cells: $TOTAL_ERR"
```

Expected: 6/6 cells; TOTAL_ERR = 0. If TOTAL_ERR > 0 → BLOCK (Rule D hard-stop violated; investigate before proceeding).

### T1.C — Acceptance gate

- [ ] **Step C1: Compute per-PP envelope JSONs**

```bash
for pp in 128 512; do
  uv run python tools/p5i_c_pp_tps_envelope.py \
    --pp $pp \
    --expected-runs 15 \
    --repeat-csv /tmp/p5h+2-e-t1-e-t1-r1-pp${pp}/bench.csv \
    --repeat-csv /tmp/p5h+2-e-t1-e-t1-r2-pp${pp}/bench.csv \
    --repeat-csv /tmp/p5h+2-e-t1-e-t1-r3-pp${pp}/bench.csv \
    --out-json /tmp/p5h+2-e-t1-pp${pp}-envelope.json
done
```

- [ ] **Step C2: Extract verdict per spec § 3.2**

```bash
uv run python <<'PY' | tee /tmp/p5h+2-e-t1-verdict.md
import json
results = {}
passes = {}
for pp in (128, 512):
    d = json.load(open(f"/tmp/p5h+2-e-t1-pp{pp}-envelope.json"))
    iron = d['ironmlx']
    results[pp] = iron['final_uncertainty_envelope_pct']
    passes[pp] = iron['verdict'] == 'PASS'
    print(f"PP={pp}: envelope={iron['final_uncertainty_envelope_pct']:.3f}% "
          f"(within_max={iron['within_sweep_ci95_max_pct']:.3f}%, "
          f"between_half={iron['between_sweep_half_range_pct']:.3f}%) "
          f"target={iron['target_pct']:.1f}% "
          f"policy={iron['target_policy']} "
          f"medians={[round(m,1) for m in iron['medians']]} "
          f"verdict={iron['verdict']}")
e128, e512 = results[128], results[512]
if passes[128] and passes[512]:
    verdict = "STRONG_PASS"
elif e128 <= 3.0 and e512 <= 3.0:
    verdict = "WEAK"
else:
    verdict = "FAIL"
print(f"\nT1 verdict: {verdict} (e128={e128:.3f}%, e512={e512:.3f}%)")
PY
```

Per spec § 3.2 mapping:
- **STRONG_PASS** (A1 + A2 both PASS under their per-PP acceptance target): proceed to T3 close-out Strong PASS
- **WEAK** (one PP exceeds its per-PP target but neither PP > 3%): NO Phase 0 backfill; Boss + Codex decide T2 expansion
- **FAIL** (one PP > 3%): NO Phase 0 backfill; T2 recommended but Boss approval REQUIRED for new GPU work

- [ ] **Step C3: Stop and report DONE + VERDICT — do NOT commit**

Report to controller (and Boss): T1 verdict + per-PP envelope numbers + per-cell medians. Controller dispatches T2 (if Weak/FAIL + Boss approves) OR T3 close-out (if Strong PASS, OR after T2 if applicable).

---

## Task 2: H_small_batch — iron-bench `--nonce-seed` + MoE expert occupancy (GATED on T1 weak/FAIL + Boss approval)

**Gating reminder:** If T1 verdict = STRONG_PASS, SKIP this entire task. If Weak/FAIL but Boss does NOT approve T2: SKIP. Proceed directly to T3 close-out.

**Files:**
- Modify: `iron-bench/src/main.rs` — Args struct `nonce_seed: Option<u64>` field
- Modify: `iron-bench/src/runner.rs` — replace `nonce_seed()` call at line 91 with `args-derived` value when `--nonce-seed` set; preserve `(run_idx << 8)` xor semantics
- Modify: `iron-bench/src/main.rs` v1 dispatch — pass `args.nonce_seed` through
- Create: `iron-bench/tests/nonce_seed.rs` — integration smoke test confirming clap accepts `--nonce-seed N`
- Modify: `iron-bench/src/runner.rs` test module — unit tests confirming exact `Some(N)` nonce sequence semantics (`N ^ (run_idx << 8)`)
- Modify: `ironmlx/src/models/qwen3_5_moe/sparse_moe.rs` — opt-in MoE expert occupancy logging gated by `IRONMLX_EXPERT_OCCUPANCY_LOG=1` env var (read once per process via `OnceLock`); emit summary stats per `routing_sort_pack` substep call
- Modify: `tools/p5h_2b_protocol_experiment.py` — add `--nonce-seed N` CLI + env pass-through (`P5I_C_NONCE_SEED`)
- Modify: `ironmlx/tests/p5i_c_phase_0_capture.rs` `build_iron_args` — push `--nonce-seed` from `P5I_C_NONCE_SEED` env var (mirror inter-run-cooldown-secs pattern)

### T2.A — iron-bench `--nonce-seed N` (TDD)

- [ ] **Step A1: Write failing tests**

Create `iron-bench/tests/nonce_seed.rs` for the CLI parse surface:

```rust
//! Verifies clap accepts the production `--nonce-seed N` flag. Exact nonce
//! sequence semantics are covered by runner unit tests in `src/runner.rs`.

use std::process::Command;

#[test]
fn nonce_seed_accepts_flag() {
    let bin = env!("CARGO_BIN_EXE_iron-bench");
    let out = Command::new(bin)
        .args([
            "--target",
            "bogus=http://127.0.0.1:1",
            "--model",
            "x",
            "--model-dir",
            "/tmp/nonexistent",
            "--prompt-len",
            "16",
            "--max-tokens",
            "1",
            "--runs",
            "1",
            "--warmup",
            "0",
            "--nonce-seed",
            "42",
            "--format",
            "csv",
        ])
        .output()
        .expect("iron-bench spawn");
    let code = out.status.code().unwrap_or(-1);
    assert_ne!(
        code, 2,
        "clap arg parse rejected --nonce-seed flag: stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
}
```

- In `iron-bench/src/runner.rs`, add this test module near the bottom of the file (after the nonce helper area is fine). It intentionally references helpers that do not exist yet; Step A2 should fail until Step A4 adds them.

```rust
#[cfg(test)]
mod tests_nonce_seed {
    use super::{measured_nonce, warmup_nonce};

    #[test]
    fn fixed_seed_measured_nonce_sequence_keeps_run_variation() {
        let seed = Some(20260526_u64);
        assert_eq!(measured_nonce(seed, 0), 20260526);
        assert_eq!(measured_nonce(seed, 1), 20260526 ^ (1_u64 << 8));
        assert_eq!(measured_nonce(seed, 14), 20260526 ^ (14_u64 << 8));
    }

    #[test]
    fn fixed_seed_warmup_nonce_sequence_uses_warmup_index() {
        let seed = Some(42_u64);
        assert_eq!(warmup_nonce(seed, 0), 42);
        assert_eq!(warmup_nonce(seed, 1), 43);
        assert_eq!(warmup_nonce(seed, 7), 45);
    }
}
```

- [ ] **Step A2: Run test to confirm failure**

```bash
cargo test --release -p iron-bench --test nonce_seed
cargo test --release -p iron-bench tests_nonce_seed
```

Expected: integration test FAILS with clap error code 2 (flag not yet defined), and runner unit tests fail to compile because `measured_nonce` / `warmup_nonce` are not defined yet.

- [ ] **Step A3: Add Args field**

In `iron-bench/src/main.rs`, after the `inter_run_cooldown_secs` field (added in P5h+2.d T0), append:

```rust
    /// Fixed base nonce seed for sequential (v1) mode. When set, the per-run
    /// nonce = `nonce_seed ^ (run_idx << 8)`, giving a reproducible nonce
    /// SEQUENCE across iron-bench invocations with the same seed; per-run
    /// variation still applies so each measured run within a sweep gets a
    /// distinct synthesized prompt. When absent, the legacy time-based seed
    /// is used. Per P5h+2.e spec § 5.1.
    #[arg(long)]
    pub nonce_seed: Option<u64>,
```

- [ ] **Step A4: Add nonce helpers and plumb arg through `run_cell` signature**

In `iron-bench/src/runner.rs`, locate the `pub async fn run_cell(...)` signature. Add `nonce_seed_override: Option<u64>,` after `inter_run_cooldown_secs: u64,`. Inside the function, replace:

```rust
        let nonce = nonce_seed() ^ (w as u64);
```

with:

```rust
        let nonce = warmup_nonce(nonce_seed_override, w);
```

Then replace the measured loop nonce:

```rust
        let nonce = nonce_seed() ^ ((i as u64) << 8);
```

with:

```rust
        let nonce = measured_nonce(nonce_seed_override, i);
```

Add these helpers near the existing private `fn nonce_seed() -> u64`:

```rust
fn warmup_nonce(nonce_seed_override: Option<u64>, warmup_idx: usize) -> u64 {
    nonce_seed_override.unwrap_or_else(nonce_seed) ^ (warmup_idx as u64)
}

fn measured_nonce(nonce_seed_override: Option<u64>, run_idx: usize) -> u64 {
    nonce_seed_override.unwrap_or_else(nonce_seed) ^ ((run_idx as u64) << 8)
}
```

This preserves legacy default behavior because `None` still calls `nonce_seed()` at each warmup/measured run. With `Some(N)`, the base seed is fixed and the exact sequence is reproducible.

- [ ] **Step A5: Update v1 dispatch call site**

In `iron-bench/src/main.rs` v1 dispatch, add `args.nonce_seed,` to the `runner::run_cell(...)` call after `args.inter_run_cooldown_secs,`.

- [ ] **Step A6: Run tests**

```bash
cargo test --release -p iron-bench
```

Expected: all iron-bench tests PASS (including new nonce_seed test + existing inter_run_cooldown_secs + smoke + unit tests).

- [ ] **Step A7: cargo fmt + clippy gates**

```bash
cargo fmt
cargo +nightly fmt --all -- --check
cargo +nightly clippy -p iron-bench --all-features -- -D warnings
cargo build --release -p iron-bench
```

Expected: zero warnings; build OK.

### T2.B — sparse_moe.rs opt-in MoE expert occupancy logging

- [ ] **Step B1: Add env var read + summary-stats helper**

In `ironmlx/src/models/qwen3_5_moe/sparse_moe.rs`, add this helper near `SORTED_ROUTING_MIN_BS_K` so the env var is read once per process:

```rust
fn expert_occupancy_log_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("IRONMLX_EXPERT_OCCUPANCY_LOG")
            .ok()
            .as_deref()
            == Some("1")
    })
}
```

Then locate the `routing_sort_pack` substep emission (inside `try_with_p5h_span_from_current_trace("routing_sort_pack", ...)` closure). Just before the substep returns its `(sorted_x_4d, sorted_topk_2d, sort_perm)` tuple, add:

```rust
                    // P5h+2.e T2.B: opt-in MoE expert occupancy summary stats per spec § 5.2.
                    // Diagnostic-only; do not enable in T2.A acceptance sweep cells because
                    // `to_vec` materializes routing indices and can perturb the hot path.
                    if expert_occupancy_log_enabled() {
                        // sorted_topk_2d shape: [BS*k, 1]; flatten to Vec<u32> of expert IDs.
                        // `Array::to_vec` is the repository-supported host extraction API.
                        let topk_flat: Vec<u32> = sorted_topk_2d
                            .to_vec::<u32>()
                            .context("SparseMoeBlock: occupancy sorted_topk_2d to_vec")?;
                        let mut counts = std::collections::HashMap::<u32, u32>::new();
                        for e in &topk_flat {
                            *counts.entry(*e).or_insert(0) += 1;
                        }
                        let nonempty = counts.len();
                        let mut sorted_counts: Vec<u32> = counts.values().copied().collect();
                        sorted_counts.sort_unstable();
                        let max_bucket = sorted_counts.last().copied().unwrap_or(0);
                        let p95_bucket = if sorted_counts.is_empty() {
                            0
                        } else {
                            let idx = (sorted_counts.len() * 95 / 100)
                                .min(sorted_counts.len() - 1);
                            sorted_counts[idx]
                        };
                        let total: f64 = topk_flat.len() as f64;
                        let entropy: f64 = counts
                            .values()
                            .map(|&c| {
                                let p = c as f64 / total;
                                if p > 0.0 { -p * p.log2() } else { 0.0 }
                            })
                            .sum();
                        // top-5 expert IDs sorted by count desc; hash for cross-run dispatch consistency.
                        let mut by_count: Vec<(u32, u32)> = counts.into_iter().collect();
                        by_count.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
                        let top5: Vec<u32> = by_count.iter().take(5).map(|(e, _)| *e).collect();
                        let mut hasher = std::collections::hash_map::DefaultHasher::new();
                        std::hash::Hash::hash(&top5, &mut hasher);
                        let top_expert_hash = std::hash::Hasher::finish(&hasher);
                        tracing::info!(
                            target: "moe_expert_occupancy",
                            "[p5h+2-e moe_occupancy] layer={layer_idx} \
                             nonempty_experts={nonempty} max_bucket={max_bucket} \
                             p95_bucket={p95_bucket} entropy={entropy:.4} \
                             top_expert_hash={top_expert_hash:016x}"
                        );
                    }
```

(`layer_idx` is already in scope inside the substep closure.)

- [ ] **Step B2: cargo gates + verify substep compiles + clippy clean**

```bash
cargo fmt
cargo +nightly fmt --all -- --check
cargo +nightly clippy -p ironmlx --features p5h-profile -- -D warnings
cargo build --release --features p5h-profile -p ironmlx
```

Expected: zero warnings; build OK.

### T2.C — driver pass-through (--nonce-seed) + harness env-var pass-through

- [ ] **Step C1: harness `build_iron_args` pushes --nonce-seed from env**

In `ironmlx/tests/p5i_c_phase_0_capture.rs` `fn build_iron_args` (line ~437), after the existing inter-run-cooldown push (added in P5h+2.d T0.B), add:

```rust
    let nonce_seed = env_or("P5I_C_NONCE_SEED", "");
    if !nonce_seed.is_empty() {
        iron_args.push("--nonce-seed".into());
        iron_args.push(nonce_seed);
    }
```

- [ ] **Step C2: harness doc-comment header update**

Add to doc-comment env var listing:

```rust
//!   * `P5I_C_NONCE_SEED` — iron-bench `--nonce-seed N` for reproducible
//!     nonce sequences (P5h+2.e T2.A acceptance sweep). Default unset =
//!     legacy time-based nonce.
```

- [ ] **Step C3: driver argparse + env pass-through**

In `tools/p5h_2b_protocol_experiment.py`:

(a) After `--preheat-pp-list` argparse arg (T0.C above), add:

```python
    p.add_argument(
        "--nonce-seed",
        type=int,
        default=None,
        help="iron-bench --nonce-seed N for reproducible nonce sequences "
        "(P5h+2.e T2.A acceptance sweep). Default unset = legacy time-based. "
        "Per P5h+2.e spec § 5.1.",
    )
```

(b) In `run_one_repeat`, after the `--preheat-pp-list` env push (T0.C above), add:

```python
    if args.nonce_seed is not None:
        env["P5I_C_NONCE_SEED"] = str(args.nonce_seed)
```

- [ ] **Step C4: cargo gates + pytest**

```bash
cargo fmt
cargo +nightly fmt --all -- --check
cargo +nightly clippy -p ironmlx --features p5h-profile -- -D warnings
uv run pytest tools/p5h_aggregator/tests/test_p5h_2b_protocol_experiment.py -v
```

Expected: zero clippy warnings; pytests still PASS.

### T2.D — T2.A acceptance sweep (no occupancy logging)

- [ ] **Step D1: Launch T2.A sweep with --nonce-seed but NO occupancy log**

```bash
SNAP=/Users/xin/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/1e20fd8d42056f870933bf98ca6211024744f7ec
MLX_DIR=$HOME/.local/mlx
(
  uv run python tools/p5h_2b_protocol_experiment.py \
      --phase t4 --exp-id e-t2a \
      --server-lifecycle same_spawn_per_pp \
      --pp-order 128,512 \
      --logging-mode quiet_acceptance \
      --mode production \
      --repeats 3 --pps 128,512 \
      --runs-per-pp '128:15,512:15' \
      --preheat-seconds 300 --preheat-runs 550 \
      --preheat-pp-list '512,{pp}' \
      --inter-run-cooldown-secs 120 \
      --nonce-seed 20260526 \
      --model-dir "$SNAP" --mlx-dir "$MLX_DIR" \
      --out-base /tmp/p5h+2-e-t2a
  status=$?
  echo "$status" > /tmp/p5h+2-e-t2a-sweep.exit
  exit "$status"
) > /tmp/p5h+2-e-t2a-sweep.log 2>&1 &
echo $! > /tmp/p5h+2-e-t2a-sweep.pid
echo "T2.A launched; pid=$(cat /tmp/p5h+2-e-t2a-sweep.pid)"
sleep 3
kill -0 $(cat /tmp/p5h+2-e-t2a-sweep.pid) 2>/dev/null && echo "alive" || echo "DEAD"
```

Wall: same as T1 ~4 hr.

- [ ] **Step D2: Wait via shell polling + Rule D scan + envelope compute**

```bash
PID=$(cat /tmp/p5h+2-e-t2a-sweep.pid)
while kill -0 "$PID" 2>/dev/null; do
  date
  tail -20 /tmp/p5h+2-e-t2a-sweep.log
  sleep 1800
done
EXIT=$(cat /tmp/p5h+2-e-t2a-sweep.exit 2>/dev/null || echo missing)
echo "T2.A sweep exit=$EXIT"
test "$EXIT" = "0"

ls -d /tmp/p5h+2-e-t2a-e-t2a-r*-pp*/ | wc -l
TOTAL_ERR=0
for d in /tmp/p5h+2-e-t2a-e-t2a-r*-pp*/; do
  if [ -f "$d/server_log_scan.json" ]; then
    n=$(uv run python -c "import json; print(json.load(open('$d/server_log_scan.json'))['error_count'])")
    TOTAL_ERR=$((TOTAL_ERR + n))
  else
    echo "MISSING: $d/server_log_scan.json"
  fi
done
echo "Rule D total ERROR across T2.A cells: $TOTAL_ERR"
test "$TOTAL_ERR" = "0"

for pp in 128 512; do
  uv run python tools/p5i_c_pp_tps_envelope.py \
    --pp $pp \
    --expected-runs 15 \
    --repeat-csv /tmp/p5h+2-e-t2a-e-t2a-r1-pp${pp}/bench.csv \
    --repeat-csv /tmp/p5h+2-e-t2a-e-t2a-r2-pp${pp}/bench.csv \
    --repeat-csv /tmp/p5h+2-e-t2a-e-t2a-r3-pp${pp}/bench.csv \
    --out-json /tmp/p5h+2-e-t2a-pp${pp}-envelope.json
done

uv run python <<'PY' | tee /tmp/p5h+2-e-t2a-verdict.md
import json
results = {}
passes = {}
for pp in (128, 512):
    d = json.load(open(f"/tmp/p5h+2-e-t2a-pp{pp}-envelope.json"))
    iron = d["ironmlx"]
    results[pp] = iron["final_uncertainty_envelope_pct"]
    passes[pp] = iron["verdict"] == "PASS"
    print(f"PP={pp}: envelope={iron['final_uncertainty_envelope_pct']:.3f}% "
          f"(within_max={iron['within_sweep_ci95_max_pct']:.3f}%, "
          f"between_half={iron['between_sweep_half_range_pct']:.3f}%) "
          f"target={iron['target_pct']:.1f}% "
          f"policy={iron['target_policy']} "
          f"medians={[round(m,1) for m in iron['medians']]} "
          f"verdict={iron['verdict']}")
e128, e512 = results[128], results[512]
if passes[128] and passes[512]:
    verdict = "STRONG_PASS"
elif e128 <= 3.0 and e512 <= 3.0:
    verdict = "WEAK"
else:
    verdict = "FAIL"
print(f"\nT2.A verdict: {verdict} (e128={e128:.3f}%, e512={e512:.3f}%)")
PY
```

Expected: 6/6 cells, Rule D ERROR total `0`, and `/tmp/p5h+2-e-t2a-verdict.md` written.

### T2.E — T2.B diagnostic occupancy capture (short; opt-in)

- [ ] **Step E1: Run T2.B diagnostic sweep with occupancy logging enabled**

```bash
SNAP=/Users/xin/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/1e20fd8d42056f870933bf98ca6211024744f7ec
MLX_DIR=$HOME/.local/mlx
# Smaller config to stay within 30min budget per spec § 5.3:
# 1 repeat × 2 PPs × 15 runs cd=0s × 5-min preheat = ~12 min per repeat = ~12 min total
(
  IRONMLX_EXPERT_OCCUPANCY_LOG=1 uv run python tools/p5h_2b_protocol_experiment.py \
      --phase t4 --exp-id e-t2b \
      --server-lifecycle same_spawn_per_pp \
      --pp-order 128,512 \
      --logging-mode default_profile \
      --mode production \
      --repeats 1 --pps 128,512 \
      --runs-per-pp '128:15,512:15' \
      --preheat-seconds 300 --preheat-runs 550 \
      --preheat-pp-list '512,{pp}' \
      --inter-run-cooldown-secs 0 \
      --nonce-seed 20260526 \
      --skip-envelope \
      --model-dir "$SNAP" --mlx-dir "$MLX_DIR" \
      --out-base /tmp/p5h+2-e-t2b
  status=$?
  echo "$status" > /tmp/p5h+2-e-t2b-sweep.exit
  exit "$status"
) > /tmp/p5h+2-e-t2b-sweep.log 2>&1 &
echo $! > /tmp/p5h+2-e-t2b-sweep.pid
```

Note `IRONMLX_EXPERT_OCCUPANCY_LOG=1` env passed to the driver process → harness inherits it via `os.environ.copy()` in `run_one_repeat`. `logging-mode default_profile` (NOT quiet_acceptance) so `tracing::info!` from `moe_expert_occupancy` target appears in `server.log`.

- [ ] **Step E2: Wait and extract occupancy summaries**

```bash
PID=$(cat /tmp/p5h+2-e-t2b-sweep.pid)
while kill -0 "$PID" 2>/dev/null; do
  date
  tail -20 /tmp/p5h+2-e-t2b-sweep.log
  sleep 600
done
EXIT=$(cat /tmp/p5h+2-e-t2b-sweep.exit 2>/dev/null || echo missing)
echo "T2.B sweep exit=$EXIT"
test "$EXIT" = "0"

uv run python <<'PY'
import json, re, statistics
from pathlib import Path

pattern = re.compile(
    r"layer=(?P<layer>-?\d+) nonempty_experts=(?P<nonempty>\d+) "
    r"max_bucket=(?P<max_bucket>\d+) p95_bucket=(?P<p95_bucket>\d+) "
    r"entropy=(?P<entropy>[0-9.]+) top_expert_hash=(?P<top_hash>[0-9a-f]+)"
)
cells = {}
for d in sorted(Path("/tmp").glob("p5h+2-e-t2b-e-t2b-r1-pp*")):
    rows = []
    log_path = d / "server.log"
    for line in log_path.read_text(errors="replace").splitlines():
        if "p5h+2-e moe_occupancy" not in line:
            continue
        m = pattern.search(line)
        if m:
            row = m.groupdict()
            rows.append({
                "layer": int(row["layer"]),
                "nonempty_experts": int(row["nonempty"]),
                "max_bucket": int(row["max_bucket"]),
                "p95_bucket": int(row["p95_bucket"]),
                "entropy": float(row["entropy"]),
                "top_expert_hash": row["top_hash"],
            })
    entropies = [r["entropy"] for r in rows]
    hashes = sorted({r["top_expert_hash"] for r in rows})
    cells[d.name] = {
        "line_count": len(rows),
        "entropy_mean": statistics.fmean(entropies) if entropies else None,
        "entropy_stdev": statistics.pstdev(entropies) if len(entropies) > 1 else 0.0,
        "top_expert_hash_unique_count": len(hashes),
        "top_expert_hash_sample": hashes[:10],
    }

out = {"cells": cells}
Path("/tmp/p5h+2-e-t2b-occupancy-summary.json").write_text(json.dumps(out, indent=2))
lines = ["# P5h+2.e T2.B Occupancy Analysis", ""]
for cell, summary in cells.items():
    lines.append(
        f"- {cell}: lines={summary['line_count']} "
        f"entropy_mean={summary['entropy_mean']} "
        f"entropy_stdev={summary['entropy_stdev']:.6f} "
        f"top_hash_unique={summary['top_expert_hash_unique_count']}"
    )
Path("/tmp/p5h+2-e-t2b-occupancy-analysis.md").write_text("\n".join(lines) + "\n")
print(json.dumps(out, indent=2))
PY
```

Expected: `/tmp/p5h+2-e-t2b-occupancy-summary.json` and `/tmp/p5h+2-e-t2b-occupancy-analysis.md` exist. This satisfies spec § 5.2 "aggregator parses diagnostic lines and emits per-cell occupancy summary JSON" without using occupancy data in any acceptance verdict.

### T2.F — T2 verdict + interpretation

- [ ] **Step F1: T2.A envelope verdict (PASS/FAIL per spec § 3.2 same gate as T1)**

Apply same outcome mapping (Strong_PASS / Weak / FAIL) to T2.A envelope data. T2.B occupancy data is informational, NOT used for verdict.

- [ ] **Step F2: Stop and report DONE + VERDICT + occupancy summary — do NOT commit**

Report to controller: T2 verdict, per-PP envelope numbers, top-line occupancy observations (e.g., "top-5 expert hash consistent across r1 PP=128 runs: YES/NO").

---

## Task 3: Close-out — single commit packaging all infra + tests + docs + Phase 0 backfill + Phase 1 spec coupling

**Files:**
- Add/modify: `docs/superpowers/plans/2026-05-26-ironmlx-p5h+2-e-pp128-investigation.md`
- Create: `docs/p5h+2-e-close-out.md`
- Modify: `docs/p5i-c-phase-0-close-out.md` (§ 1 #4 + § 6 + § 9 backfill per spec § 8)
- Modify: `docs/p5i-c-phase-0-ranking-snapshot.md` (preamble update with closure pointer)
- Modify: `docs/superpowers/specs/2026-05-25-ironmlx-p5i-c-phase-1-gather-qmm-gate-up-design.md` (§ 2.3 protocol coupling per spec § 9; ONLY if T1 or T2 PASSED)
- All WIP from T0-T2: harness + driver + tests + (T2 only) iron-bench Args + runner + sparse_moe.rs

### T3.A — Backfill docs per outcome (spec § 3.2 + § 8 mapping)

- [ ] **Step A1: Read all verdict files**

```bash
cat /tmp/p5h+2-e-t1-verdict.md 2>/dev/null
[ -f /tmp/p5h+2-e-t2a-verdict.md ] && cat /tmp/p5h+2-e-t2a-verdict.md
[ -f /tmp/p5h+2-e-t2b-occupancy-analysis.md ] && cat /tmp/p5h+2-e-t2b-occupancy-analysis.md
```

- [ ] **Step A2: Write `docs/p5h+2-e-close-out.md`**

Template (substitute `<value>` placeholders with actual verdict data):

```markdown
# P5h+2.e — PP=128 ironmlx-specific within-CI Residual Investigation: Close-out (<Strong PASS | Weak | FAIL | T2-PASS>)

**Status:** `<verdict mapped to spec § 1.3 outcome matrix>`.
**Date:** 2026-05-26 (or actual close date).
**Branch:** `ironmlx-p5h+2-e-pp128-investigation` HEAD `<T3 commit SHA>`.

## § 1 T1 — equal-budget same-shape preheat sweep verdict

Per spec § 3.1 acceptance gate (A1 + A2 both PASS under their per-PP acceptance target):

| PP | envelope | target | target_policy | within-CI max | between-half | medians (r1/r2/r3) | A1/A2 verdict |
|---|---|---|---|---|---|---|---|
| 128 | <X>% | 2.5% | `small_pp_acceptance_threshold` | <X>% | <X>% | <m1>/<m2>/<m3> | <PASS|FAIL> |
| 512 | <X>% | 2.0% | `standard_acceptance_threshold` | <X>% | <X>% | <m1>/<m2>/<m3> | <PASS|FAIL> |

T1 outcome per spec § 3.2: `<Strong PASS | Weak | FAIL>`. <Rule D scan total ERROR = 0 across 6 cells; non-allow-listed WARN: <count>.>

## § 2 T2 — H_small_batch (if triggered)

<If T1 STRONG_PASS: "T2 not triggered; H1.c preheat-topology hypothesis confirmed via equal-budget protocol.">
<If T1 Weak/FAIL + Boss approved + T2 ran: T2.A envelope table + verdict; T2.B occupancy summary observations.>
<If T2 ran but T1 is later reclassified PASS under `small-PP acceptance threshold`: T2 is recorded as diagnostic/secondary evidence and MUST NOT become a future protocol requirement unless it is the only passing path.>

## § 3 Mechanism conclusion

<If H1.c PASS (T1 STRONG_PASS):>
"Measurement protocol stabilization" — ironmlx PP=128 shape-warmup sensitivity confirmed; working around via equal-budget same-shape preheat. Deferred deeper "why ironmlx needs same-shape preheat while omlx does not": noted as open + tagged for Phase 1 or separate ironmlx-internals investigation; NOT blocking P5h+2 chain closure (per spec § 10).

<If T2 PASS (after T1 weak/FAIL):>
"Fixed prompt-sequence measurement stabilization" — acceptance depends on reproducible nonce sequence. ironmlx PP=128 has prompt-variation sensitivity at small batch; documented as known characteristic.

<If both FAIL:>
PP=128 within-CI 4-5% residual persists with both H1.c protocol fix AND H_small_batch nonce pinning. Escalate to H2 (MLX state-decay; fresh-spawn-per-run control) in successor mini-phase.

## § 4 Phase 0 § 7 #4 backfill action

`docs/p5i-c-phase-0-close-out.md` § 1 #4 row updated per spec § 8 binding:
<If PASS: "PASS — restored via P5h+2.e equal-budget same-shape preheat protocol; PP=128 envelope <X>% / PP=512 <Y>% as evidence.">
<If FAIL: "STILL FAIL/DEFERRED; P5h+2.e attempt closed FAIL/escalate to H2.">

## § 5 Phase 1 spec § 2.3 protocol coupling

<If PASS:> `docs/superpowers/specs/2026-05-25-ironmlx-p5i-c-phase-1-gather-qmm-gate-up-design.md` § 2.3 updated to reference P5h+2.e-resolved protocol (equal-budget same-shape preheat; if T2 path also `--nonce-seed` setting).
<If FAIL:> No Phase 1 spec update; Phase 1 implementation REMAINS BLOCKED.

## § 6 Reusable infrastructure shipped

| Code | Path | Tests |
|---|---|---|
| Harness `P5I_C_PREHEAT_PP_LIST` env var + `{pp}` token substitution + defensive validation + meta.json audit fields | `ironmlx/tests/p5i_c_phase_0_capture.rs` | `tests_preheat_pp_list` (4 unit tests) |
| Driver `--preheat-pp-list` CLI + env pass-through | `tools/p5h_2b_protocol_experiment.py` | `test_preheat_pp_list_*` (2 smoke pytests) |
| (T2 only) iron-bench `--nonce-seed N` production CLI flag | `iron-bench/src/main.rs` + `iron-bench/src/runner.rs` | `iron-bench/tests/nonce_seed.rs` CLI smoke + `runner.rs` nonce-sequence unit tests |
| (T2 only) Driver `--nonce-seed N` + `P5I_C_NONCE_SEED` env pass-through | `tools/p5h_2b_protocol_experiment.py` + harness `build_iron_args` | — |
| (T2 only) MoE expert occupancy diagnostic logging gated by `IRONMLX_EXPERT_OCCUPANCY_LOG=1` | `ironmlx/src/models/qwen3_5_moe/sparse_moe.rs` `routing_sort_pack` substep | — (diagnostic only) |

## § 7 Wall summary

| Bucket | Cap (spec § 7.3) | Actual |
|---|---|---|
| GPU wall (T1 + T2 if triggered) | 8 hr / 15 hr (T2 path) | <X> hr |
| Docs/analysis wall | 3 hr | <X> hr |
| **Total** | **11 hr / 15 hr** | **<X> hr** |

## § 8 References

- Spec: `docs/superpowers/specs/2026-05-26-ironmlx-p5h+2-e-pp128-investigation-design.md` (commit `dbcb03f`)
- Plan: `docs/superpowers/plans/2026-05-26-ironmlx-p5h+2-e-pp128-investigation.md` (this T3 predecessor)
- Predecessor: `docs/p5h+2-d-close-out.md` § 6, `docs/p5i-c-phase-0-close-out.md` § 1 #4
- Codex review chain (gitignored): `reports/p5h+2-e-brainstorm-codex-questions.md`
- Memory: new `[project-p5h-2e-findings]` (written by this close-out)
```

- [ ] **Step A3: Update `docs/p5i-c-phase-0-close-out.md` § 1 #4 row per spec § 8**

Append to existing § 1 #4 row text. Per outcome:

- If T1 STRONG_PASS or T2 PASS: append `**2026-05-26 P5h+2.e update**: PASS — restored via P5h+2.e <equal-budget same-shape preheat | T2 fixed nonce-sequence> protocol. PP=128 envelope <X>% vs small-PP acceptance threshold 2.5%; PP=512 envelope <Y>% vs standard threshold 2.0% (≥3 fresh-spawn repeats, cd=120s). Criterion #4 PASS. See `docs/p5h+2-e-close-out.md`.`
- If T1 FAIL + (no T2 OR T2 FAIL): append `**2026-05-26 P5h+2.e update**: STILL FAIL/DEFERRED — H1.c <protocol> attempt closed FAIL; <T2 H_small_batch if run> also FAILED. Escalate H2 MLX state-decay to successor mini-phase. See `docs/p5h+2-e-close-out.md`.`

Also update § 6 + § 9 with P5h+2.e closure narrative + (if PASS) Phase 1 unblock note OR (if FAIL) successor-phase pointer.

- [ ] **Step A4: Update `docs/p5i-c-phase-0-ranking-snapshot.md` preamble**

Append `**2026-05-26 P5h+2.e update**:` sentence summarizing closure + criterion #4 status.

- [ ] **Step A5: (PASS path only) Update Phase 1 spec § 2.3**

ONLY if T1 or T2 PASSED. In `docs/superpowers/specs/2026-05-25-ironmlx-p5i-c-phase-1-gather-qmm-gate-up-design.md` § 2.3 measurement protocol binding paragraph, append:

```markdown
**P5h+2.e-resolved protocol (2026-05-26 binding):** measurement uses equal-budget same-shape preheat (`P5I_C_PREHEAT_PP_LIST="512,{pp}"` + `P5I_C_PREHEAT_RUNS=550`) per `docs/p5h+2-e-close-out.md`. <If T2 path: "Additionally, `--nonce-seed N` for reproducible nonce sequences is required."> Mismatched protocol (e.g., legacy single-shape 1100-run preheat) MUST NOT be used as primary evidence.
```

- [ ] **Step A6: Write memory entry**

Create `/Users/xin/.claude/projects/-Users-xin-workspace-ironmlx-backend/memory/project_p5h_2e_findings.md`:

```markdown
---
name: project-p5h-2e-findings
description: P5h+2.e PP=128 ironmlx-specific within-CI residual investigation — <outcome>; H1.c <confirmed|rejected> via equal-budget same-shape preheat; <T2 H_small_batch if run summary>; Phase 0 § 7 #4 <PASS backfill | STILL FAIL/DEFERRED>; Phase 1 implementation <unblocked | REMAINS BLOCKED>
metadata:
  type: project
---

P5h+2.e closed <DATE> as <Strong PASS | Weak | FAIL | T2-PASS> per spec § 3.2.

**T1 verdict**: PP=128 envelope <X>% / PP=512 envelope <Y>% under equal-budget 550+550 same-shape preheat at cd=120s on MoE Qwen3.5-35B-A3B-4bit.

<T2 verdict if run>

**Mechanism conclusion**: <"measurement protocol stabilization" / "fixed prompt-sequence stabilization" / "H1.c+H_small_batch both reject; H2 candidate next">

**Phase 0 § 7 #4 backfill**: <PASS with evidence | STILL FAIL/DEFERRED>

**Phase 1 implementation**: <UNBLOCKED per spec § 6 G1 | REMAINS BLOCKED>

**Reusable infra**: harness P5I_C_PREHEAT_PP_LIST + `{pp}` substitution + defensive validation + meta.json audit fields; (T2 only) iron-bench --nonce-seed + sparse_moe.rs IRONMLX_EXPERT_OCCUPANCY_LOG.

Links: [[project-p5h-2d-findings]] (predecessor; identified PP=128 as ironmlx-specific via δ), [[project-p5i-c-phase-0-findings]] (Phase 0 envelope gate this unblocks), [[project-p5h-t3-findings]] (MoE substep dominance context).
```

Append one-line entry to `MEMORY.md`:

```
- [P5h+2.e PP=128 ironmlx-specific residual <outcome>](project_p5h_2e_findings.md) — <one-line summary>
```

### T3.B — cargo + pytest regression gates

- [ ] **Step B1: Full cargo gates**

```bash
cargo fmt
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace -- -D warnings
cargo build --release
cargo test --release -p iron-bench
cargo test --release -p ironmlx --features p5h-profile --test p5i_c_phase_0_capture -- --list
```

Expected: full workspace clippy has zero warnings; release build OK; iron-bench tests PASS; capture harness compiles/listable. Because this phase changes Rust, do not carve out pre-existing clippy warnings in the final gate.

- [ ] **Step B2: pytest gate**

```bash
uv run pytest tools/p5h_aggregator/tests/test_p5h_2b_protocol_experiment.py tools/p5h_aggregator/tests/test_p5i_c_pp_tps_envelope.py tools/p5h_aggregator/tests/test_p5h_2d_thermal_experiment.py -v
```

Expected: all P5h+2.d-touched + P5h+2.e-touched pytests PASS (no regression).

### T3.C — Single commit

- [ ] **Step C1: Stage all changes**

```bash
git add docs/superpowers/plans/2026-05-26-ironmlx-p5h+2-e-pp128-investigation.md \
  ironmlx/tests/p5i_c_phase_0_capture.rs \
  tools/p5h_2b_protocol_experiment.py \
  tools/p5h_aggregator/tests/test_p5h_2b_protocol_experiment.py \
  docs/p5h+2-e-close-out.md \
  docs/p5i-c-phase-0-close-out.md \
  docs/p5i-c-phase-0-ranking-snapshot.md
# T2-only files (add if T2 ran):
[ -f iron-bench/tests/nonce_seed.rs ] && git add \
  iron-bench/src/main.rs iron-bench/src/runner.rs \
  iron-bench/tests/nonce_seed.rs \
  ironmlx/src/models/qwen3_5_moe/sparse_moe.rs
# PASS path only: Phase 1 spec coupling update
git diff --quiet -- docs/superpowers/specs/2026-05-25-ironmlx-p5i-c-phase-1-gather-qmm-gate-up-design.md || \
  git add docs/superpowers/specs/2026-05-25-ironmlx-p5i-c-phase-1-gather-qmm-gate-up-design.md
git status --short
```

- [ ] **Step C2: Create single commit**

```bash
git commit -m "$(cat <<'EOF'
feat(p5h+2-e): PP=128 ironmlx-specific within-CI investigation — <outcome>; Phase 0 § 7 #4 <PASS backfill | STILL FAIL/DEFERRED>

Phase outcome per spec § 3.2: <Strong PASS | Weak | FAIL | T2-PASS>.

T1 equal-budget same-shape preheat sweep verdict:
- PP=128 envelope <X>% (within <Y>%, between <Z>%) <PASS|FAIL>
- PP=512 envelope <X>% (within <Y>%, between <Z>%) <PASS|FAIL>
- Rule D scan: 0 ERROR across 6 cells

<If T2 ran: T2 verdict block>

Mechanism: <"measurement protocol stabilization" | "fixed prompt-sequence stabilization" | "H1.c+H_small_batch reject; H2 candidate">

Phase 0 § 7 #4 backfill: <details with envelope numbers as evidence>
Phase 1 implementation: <UNBLOCKED | REMAINS BLOCKED>

Reusable infra:
- ironmlx/tests/p5i_c_phase_0_capture.rs P5I_C_PREHEAT_PP_LIST env
  var + {pp} substitution + defensive validation + meta.json audit
  fields (preheat_pp_list_effective / preheat_runs_per_shape /
  preheat_total_runs_effective) per spec § 4.1
- tools/p5h_2b_protocol_experiment.py --preheat-pp-list CLI +
  P5I_C_PREHEAT_PP_LIST env pass-through; 2 smoke pytests
- (T2 only) iron-bench --nonce-seed N production CLI flag with
  N ^ (run_idx << 8) semantics; CLI smoke + runner unit tests
- (T2 only) sparse_moe.rs IRONMLX_EXPERT_OCCUPANCY_LOG=1 opt-in
  diagnostic logging in routing_sort_pack (NOT in p5h SpanFields;
  emitted to server.log via tracing::info!)

Predeclared exclusion rules locked before T1: Rule B OFF by default
(tool-option-only, predeclared); Rule C removed; Rule D ERROR=0
hard-stop; Rule E removed (no post-hoc rules).

T2.A/T2.B split (spec § 5.2 critical): T2.A acceptance sweep uses
--nonce-seed without occupancy logging (avoid hot-path perturbation);
T2.B short diagnostic capture enables occupancy log; envelope ONLY
from T2.A.

Wall: GPU <X>hr (cap 8/15hr); docs <X>hr (cap 3hr); total <X>hr
(cap 11/15hr per spec § 7.3).

Spec: docs/superpowers/specs/2026-05-26-ironmlx-p5h+2-e-pp128-
investigation-design.md (dbcb03f)
Plan: docs/superpowers/plans/2026-05-26-ironmlx-p5h+2-e-pp128-
investigation.md
Close-out: docs/p5h+2-e-close-out.md
Codex review chain: reports/p5h+2-e-* (gitignored per
[feedback-no-reports-commit])

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
git status
git log --oneline -3
```

Expected: clean commit; working tree clean; new commit at HEAD. **REPLACE `<X>` / `<outcome>` placeholders with actual verdict values BEFORE committing.**

- [ ] **Step C3: Report DONE — Boss handles git push manually**

Per Boss's prior pattern: do NOT push. Report to controller; Boss decides when to push.

---

## Self-review check (controller before T3 dispatch)

1. **Spec coverage**: § 2 protocol → T0.A-B + T1.B2 launch; § 3.1 gate → T1.C compute; § 3.2 expansion rules → T1.C2 classifier + T2 gating; § 4 T1 sweep → T1.B-C; § 5 T2 → T2.A-F; § 5.2 T2.A/T2.B split + occupancy summary JSON → T2.D + T2.E; § 6 exclusion rules → top of plan; § 7.1 task split → 4 tasks (T0/T1/T2/T3); § 8 Phase 0 backfill → T3.A3; § 9 Phase 1 spec coupling → T3.A5; § 10 close-out narrative language → T3.A2 template `<value>` substitutions; § 11 out-of-scope → respected; § 12 risks → mitigations noted inline (R7 → T2.A/T2.B split per § 5.2). AGENTS Rust verification requirement → T3.B full workspace fmt/clippy/build gate.

2. **Placeholder scan**: `<value>` placeholders in T3.A2 close-out template + T3.C2 commit message are INTENTIONAL (implementer substitutes based on actual T1/T2 verdict). Self-review § 7 in close-out template wall numbers also intentional substitution. No accidental unresolved placeholders.

3. **Type consistency**: `P5I_C_PREHEAT_PP_LIST` (Rust env var) ↔ `--preheat-pp-list` (Python argparse arg) ↔ `args.preheat_pp_list` (Python) — names consistent. `nonce_seed` (Rust Args field) ↔ `--nonce-seed N` (clap) ↔ `P5I_C_NONCE_SEED` (env var) ↔ `args.nonce_seed` (Python) — consistent. `IRONMLX_EXPERT_OCCUPANCY_LOG` (env var) ↔ `[p5h+2-e moe_occupancy]` log prefix ↔ `/tmp/p5h+2-e-t2b-occupancy-summary.json` — consistent.

No spec coverage gaps. Placeholder-template instructions explicit.

---

## Execution Handoff

Plan saved to `docs/superpowers/plans/2026-05-26-ironmlx-p5h+2-e-pp128-investigation.md`.

**Two execution options:**

1. **Subagent-Driven (recommended)** — controller dispatches fresh subagent per task with full task text + context; two-stage review (spec compliance then code quality) between tasks; T3 commits all WIP.
2. **Inline Execution** — controller executes inline via superpowers:executing-plans; checkpoint reviews per task.

Boss chooses.
