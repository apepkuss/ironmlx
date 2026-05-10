//! Qwen3.5 vision tower (24-layer ViT) — see
//! `docs/superpowers/specs/2026-05-10-p6-vl-design.md` §4.2-4.5.

pub mod block;
pub mod merger;
pub mod patch_embed;

// VisionTower struct + forward 在 Task 12 填。
