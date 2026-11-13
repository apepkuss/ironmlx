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

/// Where the images sit in a `[B, S]` batch of token ids: for every row, the
/// `[start, end)` ranges of consecutive image tokens. One range is one image
/// block: a model with bidirectional vision attention lets its tokens see
/// each other in both directions, so a block has to go through the model in
/// a single forward call.
pub fn image_token_blocks(input_ids: &Array, image_token_id: i32) -> Result<Vec<Vec<(i32, i32)>>> {
    let shape = input_ids.shape();
    let dims = shape.as_slice();
    if dims.len() != 2 {
        return Err(anyhow!("image_token_blocks expects [B,S], got {dims:?}"));
    }
    let (b, s) = (dims[0] as usize, dims[1] as usize);
    let ids: Vec<i32> = ops::astype(input_ids, Dtype::Int32)?
        .reshape(&[(b * s) as i32][..])?
        .to_vec::<i32>()?;
    Ok(ids
        .chunks(s.max(1))
        .map(|row| image_token_blocks_in_row(row, image_token_id))
        .collect())
}

pub fn image_token_blocks_in_row(row: &[i32], image_token_id: i32) -> Vec<(i32, i32)> {
    let mut blocks = Vec::new();
    let mut start = None;
    for (pos, &id) in row.iter().enumerate() {
        match (id == image_token_id, start) {
            (true, None) => start = Some(pos),
            (false, Some(begin)) => {
                blocks.push((begin as i32, pos as i32));
                start = None;
            }
            _ => {}
        }
    }
    if let Some(begin) = start {
        blocks.push((begin as i32, row.len() as i32));
    }
    blocks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_token_blocks_are_the_runs_of_image_tokens_in_each_row() {
        const IMG: i32 = 9;
        assert_eq!(
            image_token_blocks_in_row(&[1, IMG, IMG, 2, IMG, 3, IMG, IMG], IMG),
            vec![(1, 3), (4, 5), (6, 8)]
        );
        assert!(image_token_blocks_in_row(&[1, 2, 3], IMG).is_empty());

        let ids: Array = (&[IMG, IMG, 1, 2, 3, IMG][..], &[2_i32, 3][..])
            .try_into()
            .unwrap();
        assert_eq!(
            image_token_blocks(&ids, IMG).unwrap(),
            vec![vec![(0, 2)], vec![(2, 3)]]
        );
    }
}
