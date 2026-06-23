# MTP P2 Acceptance Hardening

## Goal

Make Qwen3.5/Qwen3.6 MTP support easier to validate and operate after merge:

- add real-checkpoint smoke coverage for the supported text/VL entry points;
- expose enough server-side MTP counters to distinguish active MTP, fallback, and draft acceptance;
- document the exact acceptance matrix and commands.

## Scope

1. Extend ignored real-checkpoint CLI smoke tests:
   - Qwen3.5 4B text + MTP;
   - Qwen3.6 27B text + MTP;
   - Qwen3.6 35B-A3B text + MTP;
   - Qwen3.5 VL + MTP;
   - Qwen3.6 35B-A3B VL + MTP.
2. Extend `/healthz.mtp` with additive fields:
   - `fallback_prefill_count`;
   - `drafted_tokens`;
   - `accepted_draft_tokens`.
3. Keep MTP as a startup-level server configuration. Do not add per-request API parameters.
4. Document usage-layer acceptance commands, environment variables, and expected `healthz` behavior.

## Verification

- `MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --lib actor_mtp_mode -- --nocapture`
- `MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --lib health_collector_mtp -- --nocapture`
- `MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --test cli_generate_mtp_e2e -- --list`
- Rust required checks from `AGENTS.md`.
