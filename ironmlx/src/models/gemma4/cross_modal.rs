use anyhow::{anyhow, Result};
use mlx::{ops, Array, Dtype};

pub fn replace_image_tokens(
    text_embeds: &Array,
    input_ids: &Array,
    vision_embeds: &Array,
    image_token_id: i32,
) -> Result<Array> {
    let te_shape = text_embeds.shape();
    let te_dims = te_shape.as_slice();
    if te_dims.len() != 3 {
        return Err(anyhow!("text_embeds must be 3-D [B,S,H], got {te_dims:?}"));
    }
    let (b, s, hidden) = (te_dims[0], te_dims[1], te_dims[2]);
    let ve_shape = vision_embeds.shape();
    let ve_dims = ve_shape.as_slice();
    if ve_dims.len() != 2 {
        return Err(anyhow!("vision_embeds must be 2-D [N,H], got {ve_dims:?}"));
    }
    let (n_img, v_hidden) = (ve_dims[0], ve_dims[1]);
    if hidden != v_hidden {
        return Err(anyhow!(
            "hidden dim mismatch: text={hidden} vision={v_hidden}"
        ));
    }

    let ids_i32 = ops::astype(input_ids, Dtype::Int32)?;
    let ids_flat: Vec<i32> = ids_i32.reshape(&[b * s][..])?.to_vec::<i32>()?;
    let img_count = ids_flat.iter().filter(|&&id| id == image_token_id).count();
    if img_count != n_img as usize {
        return Err(anyhow!(
            "input_ids has {img_count} image tokens but vision_embeds has {n_img} rows"
        ));
    }

    let ve_f32 = ops::astype(vision_embeds, Dtype::Float32)?;
    let ve_flat: Vec<f32> = ve_f32.to_vec::<f32>()?;
    let hidden_usize = hidden as usize;
    let mut routed = vec![0.0_f32; (b * s * hidden) as usize];
    let mut k = 0usize;
    for (pos, &token_id) in ids_flat.iter().enumerate() {
        if token_id == image_token_id {
            let src = k * hidden_usize;
            let dst = pos * hidden_usize;
            routed[dst..dst + hidden_usize].copy_from_slice(&ve_flat[src..src + hidden_usize]);
            k += 1;
        }
    }

    let routed: Array = (routed.as_slice(), &[b, s, hidden][..]).try_into()?;
    let routed = ops::astype(&routed, text_embeds.dtype())?;
    let mask: Vec<bool> = ids_flat.iter().map(|&id| id == image_token_id).collect();
    let mask: Array = (mask.as_slice(), &[b, s, 1][..]).try_into()?;
    Ok(ops::where_(&mask, &routed, text_embeds)?)
}
