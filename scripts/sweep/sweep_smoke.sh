#!/bin/bash
# B1-p2 smoke gate (~5-15 min) — fast feedback for every code-change cycle.
#
# Always:
#   1. `cargo test --lib` (lib unit tests, ~1 min)
#
# Then either:
#   (a) Suites picked automatically by `git diff` vs the base ref
#       (default base = HEAD~1; override with `--base <ref>`).
#   (b) Explicit suite list via `--suites s1 s2 ...` (skips auto-pick).
#
# Suite identifiers accept two forms:
#   - "b1_p2_3b_2_scheduler_actor"                  → runs all #[ignore] tests in that file
#   - "b1_p2_4_batched_vl::mid_admit_vl_during_text_decode"  → just that single test
#
# Exit 0 if all pass; non-zero (1 + suite index) if any fails.
#
# Usage:
#   ./scripts/sweep/sweep_smoke.sh                       # auto-pick vs HEAD~1
#   ./scripts/sweep/sweep_smoke.sh --base main           # auto-pick vs main
#   ./scripts/sweep/sweep_smoke.sh --suites b1_p2_4_batched_vl b1_p2_3b_2_scheduler_actor
#   ./scripts/sweep/sweep_smoke.sh --suites b1_p2_4_batched_vl::mid_admit_vl_during_text_decode

set -u

export MLX_DIR="${MLX_DIR:-$HOME/.local/mlx}"
export QWEN35_MODEL="${QWEN35_MODEL:-$(ls -d "$HOME"/.ironmlx/models/models--*Qwen3.5-4B-MLX-4bit*/snapshots/*/ 2>/dev/null | head -1)}"

if [ -z "$QWEN35_MODEL" ]; then
  echo "[smoke] ERROR: QWEN35_MODEL not set and no Qwen3.5-4B-MLX-4bit fixture under ~/.ironmlx/models/"
  exit 2
fi

# Default: auto-pick. Parse args.
BASE_REF="HEAD~1"
EXPLICIT_SUITES=()
MODE="auto"

while [ $# -gt 0 ]; do
  case "$1" in
    --base)
      BASE_REF="$2"
      shift 2
      ;;
    --suites)
      MODE="explicit"
      shift
      while [ $# -gt 0 ]; do
        EXPLICIT_SUITES+=("$1")
        shift
      done
      ;;
    -h|--help)
      sed -n '1,30p' "$0"
      exit 0
      ;;
    *)
      echo "[smoke] unknown arg: $1"
      exit 2
      ;;
  esac
done

REPORT="/tmp/sweep_smoke_$(date +%s).log"
: > "$REPORT"

log() {
  echo "$@" | tee -a "$REPORT"
}

log "=== smoke gate — $(date) ==="
log ""

# Step 1: lib tests (always).
LIB_T0=$(date +%s)
log "[$(date +%H:%M:%S)] lib tests ..."
LIB_OUT=$(cargo +stable test --release --lib -p ironmlx 2>&1)
LIB_RC=$?
LIB_T1=$(date +%s)
LIB_ELAPSED=$((LIB_T1 - LIB_T0))
if [ "$LIB_RC" -ne 0 ]; then
  log "  FAIL lib tests in ${LIB_ELAPSED}s"
  echo "$LIB_OUT" | tail -30 | tee -a "$REPORT"
  log ""
  log "=== smoke FAILED (lib) ==="
  exit 1
fi
log "  PASS lib tests (${LIB_ELAPSED}s) — $(echo "$LIB_OUT" | grep -E '^test result:' | tail -1)"
log ""

# Step 2: pick integration suites.
if [ "$MODE" = "explicit" ]; then
  SUITES=("${EXPLICIT_SUITES[@]}")
  log "[smoke] explicit suites: ${SUITES[*]}"
else
  # Auto-pick by git diff scope.
  CHANGED=$(git diff --name-only "$BASE_REF" 2>/dev/null || git status --porcelain | awk '{print $NF}')
  log "[smoke] auto-pick vs $BASE_REF — changed files:"
  if [ -z "$CHANGED" ]; then
    log "  (none — falling back to default smoke suite)"
  else
    echo "$CHANGED" | sed 's/^/    /' | tee -a "$REPORT"
  fi

  declare -A PATTERNS=(
    [scheduler_or_actor]='core/scheduler\.rs|core/server/scheduler_actor'
    [admit_mid]='admit_mid|AdmitMidHandle'
    [http_server]='core/server/(openai|anthropic|chat_format|mod)|cli/serve'
    [generate]='core/generate\.rs|GenerationStream'
    [vision]='models/(vision|qwen3_5/(cross_modal|image_processor)|qwen3_5_moe)'
    [cache]='core/cache/'
    [model_core]='models/(qwen3_5|qwen3_5_moe)/(model|text_model|config)\.rs'
  )
  declare -A SUITES_FOR=(
    [scheduler_or_actor]="b1_p2_3b_2_scheduler_actor b1_p2_3c_3_continuous_batching"
    [admit_mid]="b1_p2_4_batched_vl::mid_admit_vl_during_text_decode"
    [http_server]="p4_http_smoke"
    [generate]="b1_p2_2_batched_decode"
    [vision]="p6_qwen35_vl_logits_match"
    [cache]="b1_p2_1_batched_prefill"
    [model_core]="b1_p2_2_batched_decode b1_p2_4_batched_vl::batched_vl_b2_full_vl_bit_id"
  )

  declare -A PICKED=()
  for key in "${!PATTERNS[@]}"; do
    pat="${PATTERNS[$key]}"
    if echo "$CHANGED" | grep -qE "$pat"; then
      for s in ${SUITES_FOR[$key]}; do
        PICKED[$s]=1
      done
    fi
  done

  SUITES=()
  for s in "${!PICKED[@]}"; do
    SUITES+=("$s")
  done

  if [ ${#SUITES[@]} -eq 0 ]; then
    log "[smoke] no pattern matched — running default smoke suite b1_p2_3b_2_scheduler_actor"
    SUITES=("b1_p2_3b_2_scheduler_actor")
  fi
  log "[smoke] picked suites: ${SUITES[*]}"
fi
log ""

# Step 3: run each picked suite (or specific test) sequentially.
SUITE_IDX=0
for tag in "${SUITES[@]}"; do
  SUITE_IDX=$((SUITE_IDX + 1))
  if [[ "$tag" == *::* ]]; then
    suite="${tag%%::*}"
    test_name="${tag##*::}"
    label="$suite::$test_name"
    CARGO_TEST_ARGS=("--test" "$suite" "$test_name")
  else
    suite="$tag"
    label="$suite"
    CARGO_TEST_ARGS=("--test" "$suite")
  fi

  T0=$(date +%s)
  log "[$(date +%H:%M:%S)] ($SUITE_IDX/${#SUITES[@]}) running $label ..."
  OUT=$(cargo +stable test --release "${CARGO_TEST_ARGS[@]}" -- --ignored --test-threads=1 2>&1)
  RC=$?
  T1=$(date +%s)
  ELAPSED=$((T1 - T0))
  RESULT_LINE=$(echo "$OUT" | grep -E '^test result:' | tail -1)
  if [ "$RC" -eq 0 ]; then
    log "  PASS ${label} (${ELAPSED}s) — $RESULT_LINE"
  else
    log "  FAIL ${label} (${ELAPSED}s, rc=$RC) — $RESULT_LINE"
    log "  last 30 lines:"
    echo "$OUT" | tail -30 | sed 's/^/    /' | tee -a "$REPORT"
    log ""
    log "=== smoke FAILED at $label ==="
    exit $((SUITE_IDX + 1))
  fi
done

log ""
log "=== smoke ALL PASS — report: $REPORT ==="
