#!/bin/bash
# B1-p2 full regression sweep (~3-4 h) — pre-merge / pre-release gate.
# 15 suites covering the B1-p2 batched-serving stack end-to-end against
# the real Qwen3.5-4B-MLX-4bit fixture.
#
# Failure does NOT abort: every suite is logged so the closure has the
# full picture. Use `sweep_smoke.sh` / `sweep_scoped.sh` for shorter
# dev-cycle gates.

set -u

export MLX_DIR="${MLX_DIR:-$HOME/.local/mlx}"
export QWEN35_MODEL="${QWEN35_MODEL:-$(ls -d "$HOME"/.ironmlx/models/models--*Qwen3.5-4B-MLX-4bit*/snapshots/*/ 2>/dev/null | head -1)}"

if [ -z "$QWEN35_MODEL" ]; then
  echo "[full] ERROR: QWEN35_MODEL not set and no Qwen3.5-4B-MLX-4bit fixture under ~/.ironmlx/models/"
  exit 2
fi

SUITES=(
  "b1_p2_1_batched_prefill"
  "b1_p2_2_batched_decode"
  "b1_p2_3a_scheduler_skeleton"
  "b1_p2_3b_1_scheduler_step"
  "b1_p2_3b_2_scheduler_actor"
  "b1_p2_3b_3_admission_window"
  "b1_p2_3b_4_anthropic_actor"
  "b1_p2_3c_1_per_row_offset"
  "b1_p2_3c_2_scheduler_decode_mask"
  "b1_p2_3c_3_continuous_batching"
  "b1_p2_3d_admission_queue"
  "b1_p2_4_batched_vl"
  "b1_p2_3f_cache_cap"
  "p6_qwen35_vl_logits_match"
  "p4_http_smoke"
)

# Append b1_p2_3c_plus_chunked_admit_mid if it exists (3c+ branch only).
if [ -f "ironmlx/tests/b1_p2_3c_plus_chunked_admit_mid.rs" ]; then
  SUITES+=("b1_p2_3c_plus_chunked_admit_mid")
fi

REPORT="/tmp/sweep_full_$(date +%s).log"
: > "$REPORT"
log() { echo "$@" | tee -a "$REPORT"; }

log "=== full regression sweep — $(date) ==="
log "suites: ${#SUITES[@]}"
log ""

TOTAL_T0=$(date +%s)
PASS_COUNT=0
FAIL_LIST=()

for s in "${SUITES[@]}"; do
  T0=$(date +%s)
  log "[$(date +%H:%M:%S)] running $s ..."
  OUT=$(cargo +stable test --release --test "$s" -- --ignored --test-threads=1 2>&1)
  RC=$?
  T1=$(date +%s)
  ELAPSED=$((T1 - T0))
  RESULT_LINE=$(echo "$OUT" | grep -E '^test result:' | tail -1)
  if [ "$RC" -eq 0 ]; then
    log "  PASS (${ELAPSED}s) — $RESULT_LINE"
    PASS_COUNT=$((PASS_COUNT + 1))
  else
    log "  FAIL (${ELAPSED}s, rc=$RC) — $RESULT_LINE"
    log "  last 30 lines:"
    echo "$OUT" | tail -30 | sed 's/^/    /' | tee -a "$REPORT"
    FAIL_LIST+=("$s")
  fi
done

TOTAL_T1=$(date +%s)
TOTAL_ELAPSED=$((TOTAL_T1 - TOTAL_T0))
log ""
log "=== full sweep done — ${PASS_COUNT}/${#SUITES[@]} PASS in $((TOTAL_ELAPSED / 60))m $((TOTAL_ELAPSED % 60))s ==="
log "report: $REPORT"
if [ ${#FAIL_LIST[@]} -gt 0 ]; then
  log "FAILED: ${FAIL_LIST[*]}"
  exit 1
fi
log "ALL PASS"
