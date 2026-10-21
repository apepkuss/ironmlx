//! Prefix-cache identity of a request beyond its token ids, shared by the
//! scheduler's paged prefix cache and the DFlash2 prefix cache.
//!
//! Two different images of one grid expand to the same image-pad tokens, so
//! an image prompt's cached prefix is keyed by the images it covers as well.
//! A prefix ending at a boundary covers the images whose pad run starts before
//! it: a later turn that appends an image still matches the earlier turns'
//! prefixes. Image prompts never share entries with text prompts (their
//! prefill runs another forward path, so the values may differ in the last
//! bits).

use std::sync::Arc;

use anyhow::Result;
use mlx::{Array, Dtype};

pub(crate) fn update_prefix_fingerprint_hash(hash: &mut u64, bytes: &[u8]) {
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
    for &byte in bytes {
        *hash ^= u64::from(byte);
        *hash = hash.wrapping_mul(FNV_PRIME);
    }
}

pub(crate) fn update_prefix_fingerprint_i32(hash: &mut u64, value: i32) {
    update_prefix_fingerprint_hash(hash, &value.to_le_bytes());
}

pub(crate) fn update_prefix_fingerprint_usize(hash: &mut u64, value: usize) {
    update_prefix_fingerprint_hash(hash, &value.to_le_bytes());
}

pub(crate) fn update_prefix_fingerprint_str(hash: &mut u64, value: &str) {
    update_prefix_fingerprint_usize(hash, value.len());
    update_prefix_fingerprint_hash(hash, value.as_bytes());
}

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;

/// Prefix identity of one request: none for a text prompt; for an image
/// prompt, a digest per image (grid and pixel values) and, when every image
/// expands to one pad run of its grid's token count, where those runs start.
#[derive(Debug, Clone, Default)]
pub(crate) struct PrefixFingerprint {
    vl: Option<Arc<VlPrefixIdentity>>,
}

#[derive(Debug)]
struct VlPrefixIdentity {
    header: u64,
    image_digests: Vec<u64>,
    /// `(start, len)` of every image's pad run, in image order; `None` when
    /// images and runs do not pair up (the whole image set keys every
    /// boundary then).
    runs: Option<Vec<(usize, usize)>>,
}

/// Images of an image prompt not entirely inside a restored prefix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct UncachedImages {
    /// Index of the first image whose pad run ends past the prefix.
    pub(crate) first: usize,
    /// Leading vision rows of that image already inside the prefix.
    pub(crate) rows_inside: usize,
}

impl PrefixFingerprint {
    pub(crate) fn text() -> Self {
        Self::default()
    }

    /// The identity of a request; text when it carries no images.
    pub(crate) fn for_request(
        prompt_ids: &[u32],
        pixel_values: Option<&[Array]>,
        image_grid_thw: Option<&[(i32, i32, i32)]>,
        image_token_id: i32,
        image_spatial_merge_size: i32,
    ) -> Result<Self> {
        let Some(pixel_values) = pixel_values else {
            return Ok(Self::text());
        };
        let grids = image_grid_thw.ok_or_else(|| {
            anyhow::anyhow!(
                "prefix cache VL fingerprint: pixel_values present but grid_thw is None"
            )
        })?;
        let mut header = FNV_OFFSET;
        update_prefix_fingerprint_str(&mut header, "ironmlx-vl-prefix-v2");
        update_prefix_fingerprint_i32(&mut header, image_token_id);
        update_prefix_fingerprint_i32(&mut header, image_spatial_merge_size);

        // One digest per image when the pixel arrays come one per image,
        // otherwise one digest over every array.
        let per_image = pixel_values.len() == grids.len();
        let mut image_digests = Vec::with_capacity(grids.len());
        if per_image {
            for (pixels, &grid) in pixel_values.iter().zip(grids) {
                image_digests.push(image_digest(&[pixels], &[grid])?);
            }
        } else {
            let arrays = pixel_values.iter().collect::<Vec<_>>();
            image_digests.push(image_digest(&arrays, grids)?);
        }
        let runs = per_image
            .then(|| image_runs(prompt_ids, grids, image_token_id, image_spatial_merge_size))
            .flatten();
        Ok(Self {
            vl: Some(Arc::new(VlPrefixIdentity {
                header,
                image_digests,
                runs,
            })),
        })
    }

    pub(crate) fn is_vl(&self) -> bool {
        self.vl.is_some()
    }

    /// Fingerprint of a prefix entry of `cached_len` tokens: `None` for a
    /// text prompt; for an image prompt, the images that prefix covers.
    pub(crate) fn at(&self, cached_len: usize) -> Option<String> {
        let vl = self.vl.as_ref()?;
        let mut hash = vl.header;
        match &vl.runs {
            Some(runs) => {
                let covered = runs
                    .iter()
                    .filter(|&&(start, _)| start < cached_len)
                    .count();
                update_prefix_fingerprint_str(&mut hash, "covered");
                update_prefix_fingerprint_usize(&mut hash, covered);
                for digest in &vl.image_digests[..covered] {
                    update_prefix_fingerprint_hash(&mut hash, &digest.to_le_bytes());
                }
            }
            None => {
                update_prefix_fingerprint_str(&mut hash, "all");
                update_prefix_fingerprint_usize(&mut hash, vl.image_digests.len());
                for digest in &vl.image_digests {
                    update_prefix_fingerprint_hash(&mut hash, &digest.to_le_bytes());
                }
            }
        }
        Some(format!("vl2:{hash:016x}"))
    }

    /// The images a prefill resuming after `cached_len` restored tokens still
    /// encodes; `None` when every image must be encoded (text prompts, or
    /// images that do not pair up with their pad runs).
    pub(crate) fn uncached_images(&self, cached_len: usize) -> Option<UncachedImages> {
        let runs = self.vl.as_ref()?.runs.as_ref()?;
        let first = runs
            .iter()
            .position(|&(start, len)| start + len > cached_len)
            .unwrap_or(runs.len());
        let rows_inside = runs
            .get(first)
            .map_or(0, |&(start, _)| cached_len.saturating_sub(start));
        Some(UncachedImages { first, rows_inside })
    }
}

/// Digest of the grids and pixel values of one image (or of every image).
fn image_digest(pixel_values: &[&Array], grids: &[(i32, i32, i32)]) -> Result<u64> {
    let mut hash = FNV_OFFSET;
    update_prefix_fingerprint_usize(&mut hash, grids.len());
    for &(t, h, w) in grids {
        update_prefix_fingerprint_i32(&mut hash, t);
        update_prefix_fingerprint_i32(&mut hash, h);
        update_prefix_fingerprint_i32(&mut hash, w);
    }
    update_prefix_fingerprint_usize(&mut hash, pixel_values.len());
    for pixels in pixel_values {
        update_prefix_fingerprint_str(&mut hash, &pixels.dtype().to_string());
        let shape = pixels.shape();
        update_prefix_fingerprint_usize(&mut hash, shape.as_slice().len());
        for &dim in shape.as_slice() {
            update_prefix_fingerprint_i32(&mut hash, dim);
        }
        let values = mlx::ops::astype(pixels, Dtype::Float32)?.to_vec::<f32>()?;
        update_prefix_fingerprint_usize(&mut hash, values.len());
        for value in values {
            update_prefix_fingerprint_hash(&mut hash, &value.to_bits().to_le_bytes());
        }
    }
    Ok(hash)
}

/// `(start, len)` of each image's pad run when the prompt has exactly one run
/// per image and each run has its grid's merged token count.
fn image_runs(
    prompt_ids: &[u32],
    grids: &[(i32, i32, i32)],
    image_token_id: i32,
    merge: i32,
) -> Option<Vec<(usize, usize)>> {
    let pad = u32::try_from(image_token_id).ok()?;
    if merge <= 0 {
        return None;
    }
    let mut runs = Vec::with_capacity(grids.len());
    let mut index = 0;
    while index < prompt_ids.len() {
        if prompt_ids[index] != pad {
            index += 1;
            continue;
        }
        let start = index;
        while index < prompt_ids.len() && prompt_ids[index] == pad {
            index += 1;
        }
        runs.push((start, index - start));
    }
    let pairs = runs.len() == grids.len()
        && runs.iter().zip(grids).all(|(&(_, len), &(t, h, w))| {
            usize::try_from(t * (h / merge) * (w / merge)).ok() == Some(len)
        });
    pairs.then_some(runs)
}

#[cfg(test)]
mod tests {
    use super::*;

    const PAD: u32 = 248_056;

    fn pixels(value: f32) -> Array {
        (&[value; 8][..], &[4_i32, 2][..]).try_into().unwrap()
    }

    /// `text, image(1,4,4) -> 4 pads, text, image(1,4,4) -> 4 pads, text`.
    fn two_image_prompt() -> Vec<u32> {
        let mut ids = vec![1, 2];
        ids.extend([PAD; 4]);
        ids.extend([3, 4]);
        ids.extend([PAD; 4]);
        ids.push(5);
        ids
    }

    fn identity(ids: &[u32], images: &[f32]) -> PrefixFingerprint {
        let arrays: Vec<Array> = images.iter().map(|&value| pixels(value)).collect();
        let grids = vec![(1, 4, 4); images.len()];
        PrefixFingerprint::for_request(ids, Some(&arrays), Some(&grids), PAD as i32, 2).unwrap()
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn boundaries_key_only_the_images_they_cover() {
        let ids = two_image_prompt();
        let a = identity(&ids, &[0.1, 0.2]);
        let b = identity(&ids, &[0.1, 0.3]);
        // Before the second image's run (starts at 8): same first image.
        assert_eq!(a.at(8), b.at(8));
        assert_eq!(a.at(2), b.at(2));
        // From its first pad on, the second image tells them apart.
        assert_ne!(a.at(9), b.at(9));
        assert_ne!(a.at(ids.len()), b.at(ids.len()));
        // Same grid, different content: never equal once covered.
        let c = identity(&ids, &[0.4, 0.2]);
        assert_ne!(a.at(3), c.at(3));
        // An image prompt never shares a key with a text prompt.
        assert!(a.at(1).is_some());
        assert_eq!(PrefixFingerprint::text().at(1), None);
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn a_later_turn_with_one_more_image_keeps_the_earlier_prefix_keys() {
        let first: Vec<u32> = two_image_prompt()[..7].to_vec(); // one image
        let turn1 = identity(&first, &[0.1]);
        let turn2 = identity(&two_image_prompt(), &[0.1, 0.2]);
        assert_eq!(turn1.at(first.len()), turn2.at(first.len()));
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn resumed_prefill_encodes_only_images_past_the_prefix() {
        let ids = two_image_prompt();
        let fp = identity(&ids, &[0.1, 0.2]);
        let at = |len| fp.uncached_images(len).unwrap();
        assert_eq!(
            at(0),
            UncachedImages {
                first: 0,
                rows_inside: 0
            }
        );
        assert_eq!(
            at(4),
            UncachedImages {
                first: 0,
                rows_inside: 2
            }
        );
        assert_eq!(
            at(6),
            UncachedImages {
                first: 1,
                rows_inside: 0
            }
        );
        assert_eq!(
            at(10),
            UncachedImages {
                first: 1,
                rows_inside: 2
            }
        );
        assert_eq!(
            at(ids.len()),
            UncachedImages {
                first: 2,
                rows_inside: 0
            }
        );
        assert_eq!(PrefixFingerprint::text().uncached_images(3), None);
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn unpaired_images_key_every_boundary_by_the_whole_set() {
        // Two pad runs for one image: no per-image pairing.
        let mut ids = vec![1];
        ids.extend([PAD; 2]);
        ids.push(2);
        ids.extend([PAD; 2]);
        let a = identity(&ids, &[0.1]);
        assert_eq!(a.uncached_images(4), None);
        assert_eq!(a.at(1), a.at(ids.len()));
        assert_ne!(a.at(1), identity(&ids, &[0.2]).at(1));
    }
}
