#!/usr/bin/env bash
set -euo pipefail
OUT_DIR="/Users/xin/workspace/ironmlx-backend-mtp-phase3-performance/docs/benchmarks/mtp-phase3-performance/2026-06-07-141108"
BIN="/Users/xin/workspace/ironmlx-backend-mtp-phase3-performance/target/release/ironmlx-core-bench"
PROMPT="$OUT_DIR/fixed_prompt.txt"
COMMON=(--prompt-file "$PROMPT" --mode scheduler-text --max-tokens 64 --runs 5 --warmup-runs 1 --prefill-chunk-size 2048 --b-max 1)
Q35_BASE="/Users/xin/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3"
Q35_MTP="/Users/xin/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MTP-4bit/snapshots/ab6f59bc6627196c611ab8851638651078170485"
Q27_BASE="/Users/xin/.ironmlx/models/models--mlx-community--Qwen3.6-27B-4bit/snapshots/c000ac2c2057d94be3fa931000c31723aac53282"
Q27_MTP="/Users/xin/.ironmlx/models/models--mlx-community--Qwen3.6-27B-MTP-4bit/snapshots/83795d546e9d328160e593fb0bf10b2bf2fe637e"
Q35A_BASE="/Users/xin/.ironmlx/models/models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots/38740b847e4cb78f352aba30aa41c76e08e6eb46"
Q35A_MTP="/Users/xin/.ironmlx/models/models--mlx-community--Qwen3.6-35B-A3B-MTP-4bit/snapshots/0295b81421bf4d0fccca9a7c0fcfb1418dda3516"
run_case() {
  local name="$1" model="$2" mtp="${3:-}" draft="${4:-}"
  if [[ -n "$mtp" ]]; then
    "$BIN" --model "$model" "${COMMON[@]}" --mtp-model-dir "$mtp" --mtp-draft-tokens "$draft" --out "$OUT_DIR/${name}.json"
  else
    "$BIN" --model "$model" "${COMMON[@]}" --out "$OUT_DIR/${name}.json"
  fi
}
run_case qwen35_4b_baseline "$Q35_BASE"
run_case qwen35_4b_mtp_d1 "$Q35_BASE" "$Q35_MTP" 1
run_case qwen35_4b_mtp_d2 "$Q35_BASE" "$Q35_MTP" 2
run_case qwen35_4b_mtp_d4 "$Q35_BASE" "$Q35_MTP" 4
run_case qwen36_27b_baseline "$Q27_BASE"
run_case qwen36_27b_mtp_d1 "$Q27_BASE" "$Q27_MTP" 1
run_case qwen36_27b_mtp_d2 "$Q27_BASE" "$Q27_MTP" 2
run_case qwen36_27b_mtp_d4 "$Q27_BASE" "$Q27_MTP" 4
run_case qwen36_35b_a3b_baseline "$Q35A_BASE"
run_case qwen36_35b_a3b_mtp_d1 "$Q35A_BASE" "$Q35A_MTP" 1
run_case qwen36_35b_a3b_mtp_d2 "$Q35A_BASE" "$Q35A_MTP" 2
run_case qwen36_35b_a3b_mtp_d4 "$Q35A_BASE" "$Q35A_MTP" 4
