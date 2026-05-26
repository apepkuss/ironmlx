#!/bin/bash
# B1-p2 scoped sweep (~30-60 min) — pre-merge / pre-close-out gate that
# runs the suites most likely to catch regressions for a given change set.
#
# Pick by `git diff` vs base ref (default HEAD~1). Mapping is wider than
# `sweep_smoke.sh`: includes adjacent suites (e.g. Scheduler change pulls
# in 3a/3b/3c/3d/3f), not just the most-relevant 1-2.
#
# For full 16-suite regression use `sweep_full.sh` (~3-4 h).
#
# Usage:
#   ./scripts/sweep/sweep_scoped.sh                # auto-pick vs HEAD~1
#   ./scripts/sweep/sweep_scoped.sh --base main    # auto-pick vs main
#   ./scripts/sweep/sweep_scoped.sh --area scheduler # force "scheduler" area
#   ./scripts/sweep/sweep_scoped.sh --area vl
#   ./scripts/sweep/sweep_scoped.sh --area http
#   ./scripts/sweep/sweep_scoped.sh --area all     # equivalent to sweep_full

set -u

export MLX_DIR="${MLX_DIR:-$HOME/.local/mlx}"
export QWEN35_MODEL="${QWEN35_MODEL:-$(ls -d "$HOME"/.ironmlx/models/models--*Qwen3.5-4B-MLX-4bit*/snapshots/*/ 2>/dev/null | head -1)}"

if [ -z "$QWEN35_MODEL" ]; then
  echo "[scoped] ERROR: QWEN35_MODEL not set and no Qwen3.5-4B-MLX-4bit fixture under ~/.ironmlx/models/"
  exit 2
fi

BASE_REF="HEAD~1"
FORCED_AREA=""

while [ $# -gt 0 ]; do
  case "$1" in
    --base) BASE_REF="$2"; shift 2 ;;
    --area) FORCED_AREA="$2"; shift 2 ;;
    -h|--help) sed -n '1,25p' "$0"; exit 0 ;;
    *) echo "[scoped] unknown arg: $1"; exit 2 ;;
  esac
done

# Suite areas. Each area is a tested-together cluster.
SUITES_SCHEDULER=(
  "b1_p2_3a_scheduler_skeleton"
  "b1_p2_3b_1_scheduler_step"
  "b1_p2_3b_2_scheduler_actor"
  "b1_p2_3b_3_admission_window"
  "b1_p2_3b_4_anthropic_actor"
  "b1_p2_3c_1_per_row_offset"
  "b1_p2_3c_2_scheduler_decode_mask"
  "b1_p2_3c_3_continuous_batching"
  "b1_p2_3d_admission_queue"
  "b1_p2_3f_cache_cap"
)
SUITES_VL=(
  "b1_p2_4_batched_vl"
  "p6_qwen35_vl_logits_match"
)
SUITES_HTTP=(
  "p4_http_smoke"
  "b1_p2_3b_4_anthropic_actor"
  "b1_p2_3d_admission_queue"
)
SUITES_DECODE=(
  "b1_p2_1_batched_prefill"
  "b1_p2_2_batched_decode"
  "b1_p2_3b_1_scheduler_step"
)
SUITES_ALL=(
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

declare -A PATTERNS=(
  [scheduler]='core/scheduler\.rs|core/server/scheduler_actor|admit_mid|AdmitMidHandle'
  [vl]='models/(vision|qwen3_5/(cross_modal|image_processor)|qwen3_5_moe)|batched_vl'
  [http]='core/server/(openai|anthropic|chat_format|mod)\.rs|cli/serve'
  [decode]='core/generate\.rs|GenerationStream|core/cache/|models/(qwen3_5|qwen3_5_moe)/(model|text_model)\.rs'
)

declare -A AREA_TO_SUITES_VAR=(
  [scheduler]=SUITES_SCHEDULER
  [vl]=SUITES_VL
  [http]=SUITES_HTTP
  [decode]=SUITES_DECODE
)

REPORT="/tmp/sweep_scoped_$(date +%s).log"
: > "$REPORT"
log() { echo "$@" | tee -a "$REPORT"; }

log "=== scoped sweep — $(date) ==="
log ""

# Determine areas.
declare -A AREAS=()
if [ "$FORCED_AREA" = "all" ]; then
  SUITES=("${SUITES_ALL[@]}")
  log "[scoped] forced --area all (15 suites)"
elif [ -n "$FORCED_AREA" ]; then
  case "$FORCED_AREA" in
    scheduler|vl|http|decode) AREAS["$FORCED_AREA"]=1 ;;
    *) log "[scoped] unknown area '$FORCED_AREA' (choices: scheduler vl http decode all)"; exit 2 ;;
  esac
  log "[scoped] forced area: $FORCED_AREA"
else
  CHANGED=$(git diff --name-only "$BASE_REF" 2>/dev/null || git status --porcelain | awk '{print $NF}')
  log "[scoped] auto-pick vs $BASE_REF — changed files:"
  if [ -z "$CHANGED" ]; then
    log "  (none — defaulting to scheduler area)"
    AREAS[scheduler]=1
  else
    echo "$CHANGED" | sed 's/^/    /' | tee -a "$REPORT"
    for area in "${!PATTERNS[@]}"; do
      if echo "$CHANGED" | grep -qE "${PATTERNS[$area]}"; then
        AREAS[$area]=1
      fi
    done
    if [ ${#AREAS[@]} -eq 0 ]; then
      log "[scoped] no area matched — defaulting to scheduler"
      AREAS[scheduler]=1
    fi
  fi
fi

# Resolve suites (unique) for the picked areas.
if [ "$FORCED_AREA" != "all" ]; then
  declare -A PICKED=()
  for area in "${!AREAS[@]}"; do
    var_name="${AREA_TO_SUITES_VAR[$area]}"
    eval "list=(\"\${${var_name}[@]}\")"
    for s in "${list[@]}"; do
      PICKED[$s]=1
    done
  done
  SUITES=()
  for s in "${!PICKED[@]}"; do
    SUITES+=("$s")
  done
fi

log ""
log "[scoped] running ${#SUITES[@]} suite(s): ${SUITES[*]}"
log ""

TOTAL_T0=$(date +%s)
FAILED=()
SUITE_IDX=0
for s in "${SUITES[@]}"; do
  SUITE_IDX=$((SUITE_IDX + 1))
  T0=$(date +%s)
  log "[$(date +%H:%M:%S)] ($SUITE_IDX/${#SUITES[@]}) $s ..."
  OUT=$(cargo +stable test --release --test "$s" -- --ignored --test-threads=1 2>&1)
  RC=$?
  T1=$(date +%s)
  ELAPSED=$((T1 - T0))
  RESULT_LINE=$(echo "$OUT" | grep -E '^test result:' | tail -1)
  if [ "$RC" -eq 0 ]; then
    log "  PASS (${ELAPSED}s) — $RESULT_LINE"
  else
    log "  FAIL (${ELAPSED}s, rc=$RC) — $RESULT_LINE"
    log "  last 30 lines:"
    echo "$OUT" | tail -30 | sed 's/^/    /' | tee -a "$REPORT"
    FAILED+=("$s")
  fi
done

TOTAL_T1=$(date +%s)
TOTAL_ELAPSED=$((TOTAL_T1 - TOTAL_T0))
log ""
log "=== scoped done — ${#SUITES[@]} suite(s) in $((TOTAL_ELAPSED / 60))m $((TOTAL_ELAPSED % 60))s ==="
log "report: $REPORT"
if [ ${#FAILED[@]} -gt 0 ]; then
  log "FAILED: ${FAILED[*]}"
  exit 1
fi
log "ALL PASS"
