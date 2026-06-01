//! MiniCPM-V-4.6 P2a ACCEPTANCE GATE: single-image VL logits parity vs mlx-vlm.
//!
//! This drives the FULL `MiniCpmV46Model` VL forward (SigLIP vision tower →
//! cross-modal scatter into `image_token_id` positions → Qwen3.5 text backbone →
//! last-token logits) on a fixed single-image prompt and compares the last-token
//! logits against the mlx-vlm full-model reference.
//!
//! P1 (`minicpmv46_vision_parity.rs`) proved the SigLIP vision embeds are
//! bit-exact; the text-only logits test (`minicpmv46_text_logits_match.rs`)
//! proved the text path is bit-exact. This test closes the loop: it verifies the
//! cross-modal scatter (`replace_image_tokens`) places the vision rows at the
//! correct `image_token_id` (248056) positions, that the image-token count
//! matches the vision-embed row count, and that SEQUENTIAL position ids are used
//! (MiniCPM-V uses `arange` positions even WITH images — verified against
//! mlx-vlm `_set_position_state`).
//!
//! Fixtures are produced by
//! `tests/fixtures/minicpmv46_vl/gen_single_image_logits.py` (gitignored .npy):
//!   * `expected_input_ids_img.npy`       — int32 [S] ids the model consumes
//!   * `input_pixel_values.npy`           — f32 [1, 14, n*14, 3] HWC pixels
//!   * `input_grid.npy`                    — int32 [gh, gw]
//!   * `expected_single_image_logits.npy` — f32 [vocab] full-model last logits
//!
//! Run with:
//! ```text
//! source ~/.local/mlx/mlx-env.sh
//! MINICPMV46_MODEL=/path/to/MiniCPM-V-4.6-4bit/snapshots/<sha> \
//!   cargo test --release -p ironmlx --test minicpmv46_single_image_parity -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use mlx::{ops, Array, Dtype};

use ironmlx::core::generate::build_position_ids;
use ironmlx::core::Loader;
use ironmlx::models::minicpmv4_6::model::MiniCpmV46Model;

const FIXTURE_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/minicpmv46_vl");
const PATCH: i32 = 14;

fn load_npy(name: &str) -> Array {
    let p = format!("{FIXTURE_DIR}/{name}");
    mlx::io::load_npy(&p).unwrap_or_else(|e| {
        panic!("failed to load {p} — run gen_single_image_logits.py first: {e}")
    })
}

fn checkpoint_dir() -> PathBuf {
    let env = std::env::var("MINICPMV46_MODEL").expect(
        "MINICPMV46_MODEL env var must point to the MiniCPM-V-4.6-4bit snapshot dir (#[ignore] test)",
    );
    PathBuf::from(env)
}

fn to_f32_vec(a: &Array) -> Vec<f32> {
    ops::cast::astype(a, Dtype::Float32)
        .expect("astype f32")
        .to_vec()
        .expect("to_vec")
}

/// Worst-element absolute deviation between two arrays (cast to f32).
fn max_abs_diff(a: &Array, b: &Array) -> f32 {
    let av = to_f32_vec(a);
    let bv = to_f32_vec(b);
    assert_eq!(
        av.len(),
        bv.len(),
        "size mismatch: {} vs {}",
        av.len(),
        bv.len()
    );
    av.iter()
        .zip(bv.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

/// Greedy argmax over a 1-D logits Array (cast to fp32 first).
fn greedy_argmax(arr: &Array) -> usize {
    let v = to_f32_vec(arr);
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap()
}

/// Top-k token ids by logit value (descending), for distribution-shape comparison.
fn top_k(arr: &Array, k: usize) -> Vec<usize> {
    let v = to_f32_vec(arr);
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[b].partial_cmp(&v[a]).unwrap_or(std::cmp::Ordering::Equal));
    idx.truncate(k);
    idx
}

/// Absolute logit difference at a specific token index.
fn diff_at(a: &Array, b: &Array, i: usize) -> f32 {
    let av = to_f32_vec(a);
    let bv = to_f32_vec(b);
    (av[i] - bv[i]).abs()
}

#[test]
#[ignore = "requires MINICPMV46_MODEL env var pointing to a real 4-bit checkpoint"]
fn minicpmv46_single_image_parity() {
    let model_dir = checkpoint_dir();
    // open_multimodal retains vision_tower.* / vit_merger.* / merger.* keys so the
    // SigLIP vision tower is loaded.
    let loader = Loader::open_multimodal(&model_dir).expect("Loader::open_multimodal");
    let model = MiniCpmV46Model::from_loader(&loader).expect("MiniCpmV46Model::from_loader");

    // --- ids: the EXACT token ids the mlx-vlm model consumed (image placeholder
    // already expanded to [im_start] + [248056]*N + [im_end]). ---------------
    let ids: Vec<i32> = load_npy("expected_input_ids_img.npy")
        .to_vec()
        .expect("input ids to_vec");
    let s = ids.len() as i32;
    let input_ids: Array = (ids.as_slice(), &[1_i32, s][..])
        .try_into()
        .expect("input_ids");

    // --- grid: [gh, gw] ------------------------------------------------------
    let grid: Vec<i32> = load_npy("input_grid.npy").to_vec().expect("grid to_vec");
    assert_eq!(grid.len(), 2, "input_grid.npy must be [grid_h, grid_w]");
    let (gh, gw) = (grid[0], grid[1]);

    // --- pixels: [1, 14, n*14, 3] f32 → bf16 (vision-tower precision) --------
    let pix_f32 = load_npy("input_pixel_values.npy");
    let d_borrow = pix_f32.shape();
    let d = d_borrow.as_slice();
    assert_eq!(
        d.len(),
        4,
        "input_pixel_values must be [1, 14, n*14, 3], got {d:?}"
    );
    assert_eq!(d[0], 1, "batch dim must be 1, got {d:?}");
    assert_eq!(
        d[1], PATCH,
        "packed height must be patch={PATCH}, got {d:?}"
    );
    assert_eq!(d[3], 3, "channel dim must be 3, got {d:?}");
    let n = d[2] / PATCH;
    assert_eq!(
        n,
        gh * gw,
        "packed patch count n={n} must equal gh*gw={}",
        gh * gw
    );
    let pix = ops::cast::astype(&pix_f32, Dtype::Bfloat16).expect("cast pixels to bf16");

    // --- vision embeds: [N, 1024] (one image → 1-element slice, t=1) --------
    let ve = model
        .compute_vision_embeds(&[pix], &[(1, gh, gw)], ())
        .expect("compute_vision_embeds");
    let ve_rows = ve.shape().as_slice()[0];
    let img_token_count = ids
        .iter()
        .filter(|&&id| id == model.image_token_id())
        .count() as i32;
    assert_eq!(
        ve_rows, img_token_count,
        "vision-embed rows {ve_rows} must equal image-token count {img_token_count} in ids \
         (scatter would mismatch otherwise)"
    );
    // Structural sanity: 16× spatial downsample → (gh/4)*(gw/4) embed rows.
    assert_eq!(
        ve_rows,
        (gh / 4) * (gw / 4),
        "vision-embed rows {ve_rows} must equal (gh/4)*(gw/4)={}",
        (gh / 4) * (gw / 4)
    );

    // --- SEQUENTIAL position ids (MiniCPM-V uses arange even WITH images) ----
    let position_ids = build_position_ids(0, s).expect("position_ids");

    // --- full VL forward → [1, 1, vocab] last-token logits ------------------
    let mut cache = model
        .make_cache(/* batch */ 1, s + 1, Dtype::Bfloat16)
        .expect("make_cache");
    let logits = model
        .forward_vl_chunk(
            &input_ids,
            &position_ids,
            Some(&[s]),
            None,
            Some(&mut cache),
            Some(&ve),
            model.image_token_id(),
            (),
        )
        .expect("forward_vl_chunk");
    let vocab = logits.shape().as_slice()[2];
    let last_flat = logits.reshape((vocab,)).expect("reshape");

    let expected = load_npy("expected_single_image_logits.npy");
    assert_eq!(
        last_flat.shape().as_slice().last().copied(),
        expected.shape().as_slice().last().copied(),
        "vocab dim must match"
    );

    let argmax_rust = greedy_argmax(&last_flat);
    let argmax_ref = greedy_argmax(&expected);
    let top5_rust = top_k(&last_flat, 5);
    let top5_ref = top_k(&expected, 5);
    let err = max_abs_diff(&last_flat, &expected);
    let diff_at_argmax = diff_at(&last_flat, &expected, argmax_ref);
    println!(
        "single_image_parity: grid=({gh},{gw}) N={img_token_count} S={s} vocab={vocab} \
         argmax rust={argmax_rust} ref={argmax_ref} max_abs={err:.4} \
         diff@argmax={diff_at_argmax:.4} top5_rust={top5_rust:?} top5_ref={top5_ref:?}"
    );

    // Primary correctness gate — what actually determines generated output:
    //   1. greedy argmax token matches exactly, and
    //   2. the whole top-5 head of the distribution matches as a set.
    // A scatter bug (wrong vision-row placement / count) or a position-id bug
    // (non-sequential) would corrupt the head and flip these.
    assert_eq!(
        argmax_rust, argmax_ref,
        "greedy argmax mismatch — ironmlx={argmax_rust}, mlx-vlm={argmax_ref}",
    );
    let set_rust: std::collections::BTreeSet<usize> = top5_rust.iter().copied().collect();
    let set_ref: std::collections::BTreeSet<usize> = top5_ref.iter().copied().collect();
    assert_eq!(
        set_rust, set_ref,
        "top-5 token set mismatch — ironmlx={top5_rust:?}, mlx-vlm={top5_ref:?}",
    );

    // Structural-sanity guard on the full-vocab worst-element deviation.
    //
    // Bound = 1.0 (locked from observed max_abs = 0.50, winning-token logit
    // bit-identical at diff@argmax = 0.0). Rationale: the text-only test
    // (minicpmv46_text_logits_match) established a ~0.53 far-tail noise floor for
    // this 4-bit-LM / bf16 hybrid backbone under an independent quantized-matmul
    // accumulation order (ironmlx self_qmm/gather vs mlx quantized_matmul), and
    // locked its guard at 1.0 (~1.9x that floor). The VL path adds one extra
    // source of bf16 jitter — the cross-modal scatter rounds the f32→bf16 vision
    // rows (`replace_image_tokens` builds the scatter buffer in f32 then casts to
    // the bf16 embed dtype) vs mlx-vlm's `.at[].add` delta — yet the observed
    // far-tail deviation here (0.50) lands AT/UNDER the text floor: the SigLIP
    // tower is bit-exact (P1), the scatter is an exact selection (not arithmetic),
    // and the winning logit is bit-identical, so the VL path adds no measurable
    // deviation beyond the text noise floor. We keep the same 1.0 guard as the
    // text test — ~2x the observed 0.50 — which trips hard on the multi-unit
    // deviation a real structural bug (wrong scatter positions / wrong count /
    // non-sequential positions) would produce (any of those also flips the
    // argmax + top-5 gates above first). The observed value is printed; if it
    // ever creeps above 1.0, investigate before loosening.
    assert!(
        err < 1.0,
        "max abs logits diff = {err} > 1.0 (structural bug suspected)",
    );
}
