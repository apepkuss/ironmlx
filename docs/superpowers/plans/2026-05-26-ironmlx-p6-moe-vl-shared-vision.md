# P6 MoE-VL Shared Vision Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract the existing Qwen3.5 vision tower implementation into a shared model-neutral module so dense and MoE models can both use it without cross-model imports.

**Architecture:** Move `models/qwen3_5/vision` to `models/vision`, re-export the same public `VisionTower` API, and update dense `Qwen35Model` to import from the shared module. Keep `VisionConfig` in `qwen3_5::config` for this step and re-export it from `models::vision` as a type alias to avoid behavior changes.

**Tech Stack:** Rust 2024, existing `mlx` wrappers, `cargo fmt`, nightly rustfmt check, clippy, release build. Python/uv is not required for this extraction; uv will be useful later for mlx-vlm reference fixture work.

---

## File Structure

| File | Responsibility | Change |
| --- | --- | --- |
| `ironmlx/src/models/mod.rs` | Model module registry and re-exports | Add `pub mod vision`; add a structural unit test for shared path visibility |
| `ironmlx/src/models/vision/*` | Shared Qwen3.5/Qwen3.5-MoE vision tower implementation | Move from `qwen3_5/vision` |
| `ironmlx/src/models/qwen3_5/mod.rs` | Dense model module exports | Remove local `vision` module declaration |
| `ironmlx/src/models/qwen3_5/model.rs` | Dense top-level model | Import `VisionTower` from `crate::models::vision` |
| `ironmlx/src/models/vision/mod.rs` | Shared vision module root | Import/re-export `VisionConfig` via `crate::models::qwen3_5::VisionConfig` |
| `docs/superpowers/specs/2026-05-26-ironmlx-p6-moe-vl-research.md` | Research notes | Update decision status from open to accepted |

## Task 1: Add Structural RED Test

**Files:**
- Modify: `ironmlx/src/models/mod.rs`

- [x] **Step 1: Write the failing test**

Add this test module at the bottom of `ironmlx/src/models/mod.rs`:

```rust
#[cfg(test)]
mod tests {
    #[test]
    fn shared_vision_module_exports_vision_tower() {
        fn assert_type<T>() {}
        assert_type::<super::vision::VisionTower>();
    }
}
```

- [x] **Step 2: Run test to verify RED**

Run:

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --lib models::tests::shared_vision_module_exports_vision_tower
```

Expected: compile failure with `could not find vision in super` or equivalent missing module error.

## Task 2: Move Vision Module To Shared Location

**Files:**
- Move: `ironmlx/src/models/qwen3_5/vision` -> `ironmlx/src/models/vision`
- Modify: `ironmlx/src/models/mod.rs`
- Modify: `ironmlx/src/models/qwen3_5/mod.rs`
- Modify: `ironmlx/src/models/qwen3_5/model.rs`
- Modify: `ironmlx/src/models/vision/mod.rs`

- [x] **Step 1: Move files with git**

Run:

```bash
git mv ironmlx/src/models/qwen3_5/vision ironmlx/src/models/vision
```

- [x] **Step 2: Register shared module**

In `ironmlx/src/models/mod.rs`, add:

```rust
pub mod vision;
```

- [x] **Step 3: Remove dense-local module declaration**

In `ironmlx/src/models/qwen3_5/mod.rs`, remove:

```rust
pub mod vision;
```

- [x] **Step 4: Update dense model import**

In `ironmlx/src/models/qwen3_5/model.rs`, replace:

```rust
use super::vision::VisionTower;
```

with:

```rust
use crate::models::vision::VisionTower;
```

- [x] **Step 5: Update shared vision config import**

In `ironmlx/src/models/vision/mod.rs`, replace:

```rust
use crate::models::qwen3_5::VisionConfig;
```

with:

```rust
pub use crate::models::qwen3_5::VisionConfig;
```

- [x] **Step 6: Run RED test to verify GREEN**

Run:

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --lib models::tests::shared_vision_module_exports_vision_tower
```

Expected: PASS.

## Task 3: Update Documentation And Broader Verification

**Files:**
- Modify: `docs/superpowers/specs/2026-05-26-ironmlx-p6-moe-vl-research.md`

- [x] **Step 1: Update research decision**

Change the open decision for module ownership to say Boss accepted shared extraction first.

- [x] **Step 2: Run focused Rust checks**

Run:

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --lib models::vision
```

Expected: shared vision tests should pass. Also check source references with
`rg "qwen3_5::vision|models/qwen3_5/vision" ironmlx/src` and confirm there are
no active source references to the old module path.

Observed:

- `models::vision` passed with 12 tests passed and 1 ignored.
- `MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --test p6_vision_dump --test p6_6_multi_image_dump --no-run`: passed.
- Active non-doc source references to `qwen3_5::vision` / `models/qwen3_5/vision`: none.

- [x] **Step 3: Run required Rust gates**

Run:

```bash
cargo fmt
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace -- -D warnings
cargo build --release
```

Expected: all commands exit 0. If a failure is unrelated to the extraction, record the exact failure and stop for Boss direction.

Observed:

- `cargo fmt`: passed.
- `cargo +nightly fmt --all -- --check`: passed.
- `cargo +nightly clippy --all-features --workspace -- -D warnings`: failed because `mlx-sys` requires `MLX_DIR`; rerun as `MLX_DIR=$HOME/.local/mlx cargo +nightly clippy --all-features --workspace -- -D warnings`: passed.
- `cargo build --release`: failed because `mlx-sys` requires `MLX_DIR`; rerun as `MLX_DIR=$HOME/.local/mlx cargo build --release`: passed.
