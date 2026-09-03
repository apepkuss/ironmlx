//! MiniCPM-V-4.6 P3 Task 5 FULL-VLM ACCEPTANCE GATE: multi-slice + multi-image
//! end-to-end parity vs mlx-vlm.
//!
//! This is the capstone of the MiniCPM-V-4.6 VL integration. P1
//! (`minicpmv46_vision_parity`) proved the SigLIP vision embeds are bit-exact;
//! P2a (`minicpmv46_single_image_parity`) proved the single-image (no-slice) VL
//! logits match; P3 Task 2/3 (`minicpmv46_multislice_preprocess_parity`) proved
//! per-slice preprocess + the sliced-prompt placeholder convention match
//! mlx-vlm. This test closes the loop on the FULL sliced + multi-image pipeline:
//! slice ordering (source first, then refine patches row-major; image-major
//! across images) → per-slice vision encode + concat → cross-modal scatter into
//! the `<|image_pad|>` (248056) runs → Qwen3.5 text backbone → last-token logits.
//!
//! Four checks:
//!   (a) preprocess parity   — `preprocess_sliced_with_grid(coco, 9)` slice
//!       count + per-slice grids + per-slice pixel max_abs < 0.05 vs the fixture.
//!   (b) prompt-ids parity   — `preprocess_sliced_to_parts(coco, 4).placeholder`
//!       encodes to exactly the fixture's image-region span (the `<image>` +
//!       `<slice>` runs).
//!   (c) single-image-sliced logits — feed the fixture sliced ids + ALL slice
//!       pixel_values (source + patches, in order) + grids through
//!       `compute_vision_embeds` + `forward_vl_chunk` (SEQUENTIAL positions);
//!       assert argmax + top-5 set + max_abs < 1.0 vs `expected_sliced_logits`.
//!   (d) multi-image logits  — the 2-image variant of (c) (both images' slices
//!       concatenated image-major).
//!
//! Fixtures (gitignored; regenerate via
//! `tests/fixtures/minicpmv46_vl/gen_multislice.py`):
//!   Single-sliced: `expected_input_ids_sliced.npy`, `sliced_count.npy`,
//!     `sliced_grids.npy`, `sliced_pixels_{i}.npy`, `expected_sliced_logits.npy`.
//!   Two-image:     `expected_input_ids_2img.npy`, `2img_count.npy`,
//!     `2img_grids.npy`, `2img_pixels_{i}.npy`, `expected_2img_logits.npy`.
//!
//! `#[ignore]` + fixture-gated.
//!
//! Run:
//! ```text
//! source ~/.local/mlx/mlx-env.sh && \
//!   MINICPMV46_MODEL=/path/to/MiniCPM-V-4.6-4bit/snapshots/<sha> \
//!   cargo test --release -p ironmlx --test minicpmv46_multislice_parity -- --ignored --nocapture
//! ```

use std::collections::BTreeSet;

use mlx::{ops, Array, Dtype};

use ironmlx::core::generate::build_position_ids;
use ironmlx::core::{Loader, Tokenizer};
use ironmlx::models::minicpmv4_6::image_processor::{preprocess_sliced_with_grid, MAX_SLICE_NUMS};
use ironmlx::models::minicpmv4_6::{model::MiniCpmV46Model, preprocess_sliced_to_parts};

mod common;
use common::minicpmv46_parity::{
    checkpoint_dir, diff_at, greedy_argmax, load_npy_in, max_abs_diff, to_f32_vec, top_k,
    FIXTURE_DIR_VL,
};

const PATCH: i32 = 14;
/// MiniCPM-V-4.6 effective vision downsample (2×2 VitMerger × 2×2 Merger).
const SPATIAL_MERGE_SIZE: i32 = 4;

// MiniCPM-V-4.6 image-placeholder special token ids (from the checkpoint
// tokenizer; verified against mlx-vlm's `tokenizer.{im,slice}_{start,end}_id`).
const IM_START_ID: i32 = 248078; // <image>
const IM_END_ID: i32 = 248079; // </image>
const SLICE_START_ID: i32 = 248088; // <slice>
const SLICE_END_ID: i32 = 248089; // </slice>

fn load_npy(name: &str) -> Array {
    load_npy_in(FIXTURE_DIR_VL, name)
}

fn coco_bytes() -> Vec<u8> {
    let p = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/qwen35_vl/coco_sample.jpg"
    );
    std::fs::read(p).expect("read coco_sample.jpg")
}

/// Image-region span `[start, end)` of an id sequence: from the first
/// `<image>`/`<slice>` start token to (inclusive) the last `</image>`/`</slice>`
/// end token. Spans every `<|image_pad|>` run plus the start/end markers and any
/// inter-row newlines — exactly the contiguous block the placeholder string
/// builds. Panics if no image markers are present.
fn image_region(ids: &[i32]) -> (usize, usize) {
    let start = ids
        .iter()
        .position(|&id| id == IM_START_ID || id == SLICE_START_ID)
        .expect("no <image>/<slice> start marker in ids");
    let end = ids
        .iter()
        .rposition(|&id| id == IM_END_ID || id == SLICE_END_ID)
        .expect("no </image>/</slice> end marker in ids");
    assert!(
        end >= start,
        "malformed image region: end {end} < start {start}"
    );
    (start, end + 1)
}

/// Load a `[count]` int32 npy as a single `usize`.
fn load_count(name: &str) -> usize {
    let v: Vec<i32> = load_npy(name).to_vec().expect("count to_vec");
    assert_eq!(v.len(), 1, "{name} must be [1]");
    v[0] as usize
}

/// (a) Per-slice preprocess parity for `coco_sample.jpg` (slicing enabled):
/// slice count + per-slice grids + per-slice pixel byte-closeness vs the
/// fixture, using the `sliced_*` fixture set.
#[test]
#[ignore = "fixture-gated MiniCPM-V-4.6 multi-slice preprocess parity"]
fn multislice_preprocess_parity() {
    let bytes = coco_bytes();
    let exp_count = load_count("sliced_count.npy");

    let grids_arr = load_npy("sliced_grids.npy");
    assert_eq!(
        grids_arr.shape().to_vec(),
        vec![exp_count as i32, 2],
        "sliced_grids.npy must be [count, 2]"
    );
    let grids: Vec<i32> = grids_arr.to_vec().expect("grids to_vec");

    let (slices, best_grid) =
        preprocess_sliced_with_grid(&bytes, MAX_SLICE_NUMS).expect("preprocess_sliced_with_grid");
    println!(
        "(a) slice count ours={} expected={exp_count} best_grid={best_grid:?}",
        slices.len()
    );
    assert_eq!(
        slices.len(),
        exp_count,
        "slice count mismatch (1 source + gx*gy patches)"
    );

    let mut worst = 0.0_f32;
    for (i, (pixel_values, gh, gw)) in slices.iter().enumerate() {
        let (exp_gh, exp_gw) = (grids[i * 2], grids[i * 2 + 1]);
        assert_eq!(
            (*gh, *gw),
            (exp_gh, exp_gw),
            "slice {i} grid mismatch (slice/refine resize math off)"
        );
        let exp_arr = load_npy(&format!("sliced_pixels_{i}.npy"));
        assert_eq!(
            pixel_values.shape().to_vec(),
            exp_arr.shape().to_vec(),
            "slice {i} pixel_values shape mismatch"
        );
        let max_abs = max_abs_diff(pixel_values, &exp_arr);
        println!("(a) slice {i}: grid=({gh},{gw}) pixel max_abs={max_abs}");
        worst = worst.max(max_abs);
    }
    // Same JPEG-decode floor (≤3/255 normalized = 0.0235) as the single-image +
    // P3-T2 preprocess parity tests; 0.05 is the spec-locked ceiling.
    assert!(
        worst < 0.05,
        "worst pixel max_abs {worst} exceeds 0.05 (JPEG decode / slice-resize / pack mismatch)"
    );
}

/// (b) Prompt-ids parity: the sliced placeholder string ironmlx builds encodes
/// to exactly the fixture sliced prompt's image-region span (the `<image>` +
/// `<slice>` runs). This validates the source/slice token counts + the
/// `<image>`/`<slice>`/newline placement convention against mlx-vlm's
/// `_build_placeholder_ids_for_image`.
#[test]
#[ignore = "fixture-gated MiniCPM-V-4.6 sliced prompt-ids parity"]
fn multislice_prompt_ids_parity() {
    let model_dir = checkpoint_dir();
    let loader = Loader::open(&model_dir).expect("Loader::open");
    let tokenizer = Tokenizer::from_loader(&loader).expect("Tokenizer::from_loader");

    let parts =
        preprocess_sliced_to_parts(&coco_bytes(), SPATIAL_MERGE_SIZE).expect("sliced_to_parts");
    let ours: Vec<i32> = tokenizer
        .encode(&parts.placeholder, false)
        .expect("encode placeholder")
        .into_iter()
        .map(|id| id as i32)
        .collect();

    let exp_ids: Vec<i32> = load_npy("expected_input_ids_sliced.npy")
        .to_vec()
        .expect("ids to_vec");
    let (start, end) = image_region(&exp_ids);
    let exp_region = &exp_ids[start..end];

    println!(
        "(b) placeholder encoded len={} fixture image-region len={} [start={start} end={end}]",
        ours.len(),
        exp_region.len()
    );
    assert_eq!(
        ours.as_slice(),
        exp_region,
        "sliced placeholder ids mismatch fixture image-region span"
    );
}

/// Run a full sliced/multi-image VL forward over a fixture scenario and assert
/// last-token logits parity (argmax + top-5 set + max_abs bound).
///
/// `tag` selects the fixture set (`sliced` or `2img`); `image_count` is the
/// number of source images (1 or 2) used only for the diagnostic print.
fn run_vl_logits_scenario(tag: &str, image_count: usize) {
    let model_dir = checkpoint_dir();
    let loader = Loader::open_multimodal(&model_dir).expect("Loader::open_multimodal");
    let model = MiniCpmV46Model::from_loader(&loader).expect("MiniCpmV46Model::from_loader");

    // ids: the EXACT token ids mlx-vlm consumed (placeholders already expanded).
    let ids: Vec<i32> = load_npy(&format!("expected_input_ids_{tag}.npy"))
        .to_vec()
        .expect("ids to_vec");
    let s = ids.len() as i32;
    let input_ids: Array = (ids.as_slice(), &[1_i32, s][..])
        .try_into()
        .expect("input_ids");

    // Per-slice grids [count, 2] (image-major: image0 source+patches, then image1).
    let count = load_count(&format!("{tag}_count.npy"));
    let grids_arr = load_npy(&format!("{tag}_grids.npy"));
    assert_eq!(
        grids_arr.shape().to_vec(),
        vec![count as i32, 2],
        "{tag}_grids.npy must be [count, 2]"
    );
    let grids: Vec<i32> = grids_arr.to_vec().expect("grids to_vec");

    // Load every slice's pixel tensor (image-major order) → bf16 (vision precision).
    let mut all_pixels: Vec<Array> = Vec::with_capacity(count);
    let mut all_grids: Vec<(i32, i32, i32)> = Vec::with_capacity(count);
    let mut total_rows: i32 = 0;
    for i in 0..count {
        let (gh, gw) = (grids[i * 2], grids[i * 2 + 1]);
        let pix_f32 = load_npy(&format!("{tag}_pixels_{i}.npy"));
        let d = pix_f32.shape().to_vec();
        assert_eq!(
            d.len(),
            4,
            "{tag} slice {i}: pixels must be [1, 14, n*14, 3]"
        );
        assert_eq!(d[0], 1, "{tag} slice {i}: batch dim must be 1");
        assert_eq!(
            d[1], PATCH,
            "{tag} slice {i}: packed height must be {PATCH}"
        );
        assert_eq!(d[3], 3, "{tag} slice {i}: channel dim must be 3");
        let n = d[2] / PATCH;
        assert_eq!(n, gh * gw, "{tag} slice {i}: n={n} != gh*gw={}", gh * gw);
        all_pixels.push(ops::cast::astype(&pix_f32, Dtype::Bfloat16).expect("cast bf16"));
        all_grids.push((1, gh, gw));
        total_rows += (gh / SPATIAL_MERGE_SIZE) * (gw / SPATIAL_MERGE_SIZE);
    }

    // Vision embeds: per-slice SigLIP encode + concat (image-major) → [N_total, 1024].
    let ve = model
        .compute_vision_embeds(&all_pixels, &all_grids, ())
        .expect("compute_vision_embeds");
    let ve_rows = ve.shape().as_slice()[0];
    let img_token_count = ids
        .iter()
        .filter(|&&id| id == model.image_token_id())
        .count() as i32;
    assert_eq!(
        ve_rows, total_rows,
        "{tag}: vision-embed rows {ve_rows} != sum (gh/4)*(gw/4) = {total_rows}"
    );
    assert_eq!(
        ve_rows, img_token_count,
        "{tag}: vision-embed rows {ve_rows} != image-token count {img_token_count} in ids \
         (scatter would mismatch otherwise)"
    );

    // SEQUENTIAL position ids (MiniCPM-V uses arange even WITH images).
    let position_ids = build_position_ids(0, s).expect("position_ids");

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
    // → [1, 1, vocab] last-token logits
    let vocab = logits.shape().as_slice()[2];
    let last_flat = logits.reshape((vocab,)).expect("reshape");

    let expected = load_npy(&format!("expected_{tag}_logits.npy"));
    assert_eq!(
        last_flat.shape().as_slice().last().copied(),
        expected.shape().as_slice().last().copied(),
        "{tag}: vocab dim must match"
    );

    let argmax_rust = greedy_argmax(&last_flat);
    let argmax_ref = greedy_argmax(&expected);
    let top5_rust = top_k(&last_flat, 5);
    let top5_ref = top_k(&expected, 5);
    let err = max_abs_diff(&last_flat, &expected);
    let diff_at_argmax = diff_at(&last_flat, &expected, argmax_ref);
    println!(
        "({tag}) images={image_count} slices={count} N={img_token_count} S={s} vocab={vocab} \
         argmax rust={argmax_rust} ref={argmax_ref} max_abs={err:.4} \
         diff@argmax={diff_at_argmax:.4} top5_rust={top5_rust:?} top5_ref={top5_ref:?}"
    );

    // Primary correctness gate: argmax + top-5 set. A slice/pixel ORDERING bug,
    // an image-token-count mismatch, or a scatter bug would corrupt the head and
    // flip these (and the count asserts above trip first).
    assert_eq!(
        argmax_rust, argmax_ref,
        "{tag}: greedy argmax mismatch — ironmlx={argmax_rust}, mlx-vlm={argmax_ref}",
    );

    // Tie-aware top-5 set check. The strict top-5 SET must match, EXCEPT a token
    // at the rank-5 boundary may be swapped for another token whose REFERENCE
    // logit equals (within the bf16 noise floor) the reference's 5th-ranked
    // logit — i.e. a genuine logit-tie crossing the top-5/top-6 boundary, where
    // the rank-5 slot is decided by arbitrary argsort tie-breaking, not by any
    // real distributional difference.
    //
    // This is exactly the benign bf16 near-tie P2b documented. In the 2-image
    // scenario the reference itself has TWO tokens (5044, 3437) at bit-identical
    // logit 23.625 (gap 0.000000) sitting at ranks 5 and 6; ironmlx's tiny jitter
    // (max_abs 0.31, well under the ~0.5 floor) nudges 5044 above 3437 so it lands
    // at rank 5 while mlx-vlm's argsort placed 3437 there. The first 4 head tokens
    // are bit-identical and the argmax matches exactly. A real structural bug
    // (wrong slice/pixel order, wrong vision-row count, broken scatter) does NOT
    // produce a boundary tie — it reshuffles the whole head and trips the argmax /
    // top-4 / max_abs gates. The count asserts above also trip first.
    let set_rust: BTreeSet<usize> = top5_rust.iter().copied().collect();
    let set_ref: BTreeSet<usize> = top5_ref.iter().copied().collect();
    if set_rust != set_ref {
        let exp_v = to_f32_vec(&expected);
        // Reference logit at the rank-5 boundary (the 5th-highest).
        // TIE_TOL = 0.0625: a conservative tolerance (≈ half a bf16 ULP in the
        // [16,32) exponent band where the observed rank-5 tie at ~23.625 lives;
        // one full ULP there is 0.125) for accepting a boundary token swap ONLY
        // against a reference tie partner at the same logit.
        let rank5_ref_logit = top5_ref
            .iter()
            .map(|&t| exp_v[t])
            .fold(f32::INFINITY, f32::min);
        const TIE_TOL: f32 = 0.0625;
        let only_rust: Vec<usize> = set_rust.difference(&set_ref).copied().collect();
        let only_ref: Vec<usize> = set_ref.difference(&set_rust).copied().collect();
        // Every token that differs must be a boundary token: its reference logit
        // equals the rank-5 reference logit (within one bf16 ULP). This holds for
        // both the dropped ref-side token (it WAS at rank 5) and the swapped-in
        // rust-side token (a tie partner sitting at rank 6 in the reference).
        let all_boundary_ties = only_rust
            .iter()
            .chain(only_ref.iter())
            .all(|&t| (exp_v[t] - rank5_ref_logit).abs() <= TIE_TOL);
        assert!(
            all_boundary_ties,
            "{tag}: top-5 token set mismatch beyond a rank-5 logit tie — \
             ironmlx={top5_rust:?}, mlx-vlm={top5_ref:?}; rank5_ref_logit={rank5_ref_logit:.4}, \
             only_rust={only_rust:?} only_ref={only_ref:?}",
        );
        println!(
            "({tag}) benign rank-5 logit-tie: only_rust={only_rust:?} only_ref={only_ref:?} \
             all at reference logit ~{rank5_ref_logit:.4} (top-4 head + argmax exact)"
        );
    }

    // Structural-sanity guard on the full-vocab worst-element deviation. Same
    // 1.0 bound as the single-image VL test (`minicpmv46_single_image_parity`):
    // the SigLIP tower is bit-exact (P1), the scatter is an exact selection, and
    // the text backbone's far-tail 4-bit/bf16 noise floor is ~0.53. The sliced /
    // multi-image path adds more vision rows (more f32→bf16 scatter rounding) but
    // no new arithmetic source, so the deviation stays in the same band. 1.0 is
    // ~2x the observed floor and trips hard on the multi-unit deviation a real
    // structural bug would produce. The observed value is printed; if it ever
    // creeps above 1.0, investigate before loosening.
    assert!(
        err < 1.0,
        "{tag}: max abs logits diff = {err} > 1.0 (structural bug suspected)",
    );
}

/// (c) Single-image-sliced e2e logits parity (coco → grid (2,1) → 3 slices).
#[test]
#[ignore = "requires MINICPMV46_MODEL env var pointing to a real 4-bit checkpoint"]
fn multislice_single_image_logits_parity() {
    run_vl_logits_scenario("sliced", 1);
}

/// (d) Multi-image (2-image) e2e logits parity (coco + image_0, image-major).
#[test]
#[ignore = "requires MINICPMV46_MODEL env var pointing to a real 4-bit checkpoint"]
fn multislice_two_image_logits_parity() {
    run_vl_logits_scenario("2img", 2);
}
