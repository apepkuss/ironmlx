//! Cross-modal token routing — replace image_token_id placeholders in
//! text_embeds with vision_embeds. See spec §4.6.

use anyhow::{anyhow, Result};
use mlx::{ops, Array, Dtype};

/// Replace `text_embeds[b, s, :]` with `vision_embeds[k, :]` where
/// `input_ids[b, s] == image_token_id`.
///
/// # Arguments
/// - `text_embeds` — shape `[B, S, hidden]`; B must equal 1 for P6.
/// - `input_ids`   — shape `[B, S]`; int32.
/// - `vision_embeds` — shape `[N_img, hidden]`; must have the same dtype as
///   `text_embeds`.
/// - `image_token_id` — the special token id that marks image positions.
///
/// # Errors
/// Returns an error if the number of image-token positions in `input_ids` does
/// not equal the number of rows in `vision_embeds` (programming error, not a
/// runtime mismatch that should panic).
pub fn replace_image_tokens(
    text_embeds: &Array,   // [B, S, hidden]
    input_ids: &Array,     // [B, S]
    vision_embeds: &Array, // [N_img, hidden]
    image_token_id: i32,
) -> Result<Array> {
    // --- shape extraction ---------------------------------------------------
    let te_shape = text_embeds.shape();
    let te_dims = te_shape.as_slice();
    if te_dims.len() != 3 {
        return Err(anyhow!(
            "text_embeds must be 3-D [B, S, hidden], got {te_dims:?}"
        ));
    }
    let (b, s, hidden) = (te_dims[0], te_dims[1], te_dims[2]);
    if b != 1 {
        // Deliberate gate — multi-image / multi-sequence batching is a future
        // expansion (see audit ref B1). Return Err rather than assert! so the
        // production HTTP path surfaces it as a clean 4xx.
        return Err(anyhow!(
            "replace_image_tokens currently supports B=1 only (got B={b}); \
             multi-image batched VL is not yet implemented"
        ));
    }

    let ve_shape = vision_embeds.shape();
    let ve_dims = ve_shape.as_slice();
    if ve_dims.len() != 2 {
        return Err(anyhow!(
            "vision_embeds must be 2-D [N_img, hidden], got {ve_dims:?}"
        ));
    }
    let (n_img, v_hidden) = (ve_dims[0], ve_dims[1]);
    if hidden != v_hidden {
        return Err(anyhow!(
            "hidden dim mismatch: text={hidden} vision={v_hidden}"
        ));
    }

    // --- read input_ids to host (int32 cast first) --------------------------
    let ids_i32 = ops::astype(input_ids, Dtype::Int32)?;
    let ids_flat: Vec<i32> = ids_i32
        .reshape(&[b * s][..])?
        .to_vec::<i32>()
        .map_err(|e| anyhow!("to_vec input_ids: {e}"))?;

    // --- read vision_embeds to host (via f32) --------------------------------
    let ve_f32 = ops::astype(vision_embeds, Dtype::Float32)?;
    let ve_flat: Vec<f32> = ve_f32
        .to_vec::<f32>()
        .map_err(|e| anyhow!("to_vec vision_embeds: {e}"))?;

    // Verify count of image positions == n_img
    let img_count = ids_flat.iter().filter(|&&id| id == image_token_id).count();
    if img_count != n_img as usize {
        return Err(anyhow!(
            "input_ids has {img_count} image tokens but vision_embeds has {n_img} rows"
        ));
    }

    // --- build vision_at_text on host ----------------------------------------
    // Output layout: flat [B * S * hidden] f32, row-major.
    // Image positions filled with vision_embeds[k, :]; text positions filled with
    // zeros (will be discarded by mx::where selection — not multiplied in).
    let total = (b * s * hidden) as usize;
    let mut vat = vec![0.0_f32; total];

    let hidden_usize = hidden as usize;
    let mut k: usize = 0;
    for (pos, &token_id) in ids_flat.iter().enumerate() {
        if token_id == image_token_id {
            let src_start = k * hidden_usize;
            let dst_start = pos * hidden_usize;
            vat[dst_start..dst_start + hidden_usize]
                .copy_from_slice(&ve_flat[src_start..src_start + hidden_usize]);
            k += 1;
        }
    }
    // Algorithmic invariant: loop above iterates exactly img_count == n_img
    // times by the pre-check at line 67. Asserted via debug_assert only.
    debug_assert_eq!(k, n_img as usize);

    // --- ship vision_at_text back to device ---------------------------------
    let vat_arr: Array = (vat.as_slice(), &[b, s, hidden][..])
        .try_into()
        .map_err(|e| anyhow!("vision_at_text array construction: {e}"))?;
    // cast to text_embeds dtype (e.g. bf16)
    let vat_arr = ops::astype(&vat_arr, text_embeds.dtype())?;

    // --- build bool mask [B, S, 1] for mx::where ----------------------------
    // true where input_ids[b, s] == image_token_id, else false.
    // Broadcast over hidden dim via mx::where's broadcasting semantics.
    let mask_bool: Vec<bool> = ids_flat.iter().map(|&id| id == image_token_id).collect();
    let mask_arr: Array = (mask_bool.as_slice(), &[b, s, 1][..])
        .try_into()
        .map_err(|e| anyhow!("bool mask array construction: {e}"))?;

    // --- select with mx::where (exact, no multiply) -------------------------
    // Matches Python's `mx.where(special_image_mask[..., None], image_embeds, text_embeds)`
    // which performs an exact selection with no arithmetic, preserving bf16 precision.
    let out = ops::where_(&mask_arr, &vat_arr, text_embeds)?;
    Ok(out)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use mlx::{ops, Array, Dtype};

    const IMAGE_TOKEN_ID: i32 = 248056;

    #[test]
    fn replaces_image_placeholders() {
        // text_embeds [1, 5, 4]: positions [text, IMG, IMG, IMG, text]
        // input_ids [1, 5]: [100, 248056, 248056, 248056, 200]
        // vision_embeds [3, 4] for 3 image tokens

        // ops::full requires &Array for vals — construct a scalar Array first
        let fill_one: Array = (&[1.0_f32][..], &[][..]).try_into().unwrap();
        let text_embeds = ops::full(&[1_i32, 5, 4][..], &fill_one, Dtype::Bfloat16).unwrap();

        let input_ids: Array = (
            &[100_i32, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, 200][..],
            &[1_i32, 5][..],
        )
            .try_into()
            .unwrap();

        let vision_embeds: Array = (&[7.0_f32; 12][..], &[3_i32, 4][..]).try_into().unwrap();
        let v_bf16 = ops::astype(&vision_embeds, Dtype::Bfloat16).unwrap();

        let merged =
            replace_image_tokens(&text_embeds, &input_ids, &v_bf16, IMAGE_TOKEN_ID).unwrap();

        let v: Vec<f32> = ops::astype(&merged, Dtype::Float32)
            .unwrap()
            .to_vec()
            .unwrap();

        // pos 0, 4 (text): 1.0 each
        assert_eq!(v[0], 1.0);
        assert_eq!(v[4 * 4], 1.0);
        // pos 1, 2, 3 (image): 7.0 each
        assert_eq!(v[1 * 4], 7.0);
        assert_eq!(v[2 * 4], 7.0);
        assert_eq!(v[3 * 4], 7.0);
    }

    #[test]
    fn count_mismatch_returns_error() {
        // input_ids has 2 image tokens but vision_embeds has 3 rows → error
        let fill_one: Array = (&[1.0_f32][..], &[][..]).try_into().unwrap();
        let text_embeds = ops::full(&[1_i32, 4, 4][..], &fill_one, Dtype::Bfloat16).unwrap();
        let input_ids: Array = (
            &[100_i32, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, 200][..],
            &[1_i32, 4][..],
        )
            .try_into()
            .unwrap();
        // 3 rows in vision_embeds but only 2 image positions
        let ve: Array = (&[7.0_f32; 12][..], &[3_i32, 4][..]).try_into().unwrap();
        let ve_bf16 = ops::astype(&ve, Dtype::Bfloat16).unwrap();
        let result = replace_image_tokens(&text_embeds, &input_ids, &ve_bf16, IMAGE_TOKEN_ID);
        assert!(result.is_err(), "should fail with count mismatch");
    }

    #[test]
    fn no_image_tokens_passthrough() {
        // No image tokens: output should equal text_embeds
        let fill_one: Array = (&[1.0_f32][..], &[][..]).try_into().unwrap();
        let text_embeds = ops::full(&[1_i32, 3, 4][..], &fill_one, Dtype::Float32).unwrap();
        let input_ids: Array = (&[10_i32, 20, 30][..], &[1_i32, 3][..]).try_into().unwrap();
        // vision_embeds: 0 rows
        let ve: Array = (&[0.0_f32; 0][..], &[0_i32, 4][..]).try_into().unwrap();
        let out = replace_image_tokens(&text_embeds, &input_ids, &ve, IMAGE_TOKEN_ID).unwrap();
        let v: Vec<f32> = out.to_vec().unwrap();
        assert!(v.iter().all(|&x| x == 1.0), "expected all 1.0, got {v:?}");
    }
}
