//! Shared, refcounted full-attention K/V segments for the ordinary prefix
//! cache. A segment holds the K/V of one block-aligned token range for every
//! shareable layer (paged or TurboQuant); entries reference segments instead
//! of owning copies of their whole prefix.

use std::fmt;
use std::sync::Arc;

use mlx::ops::indexing::slice_strided_on;
use mlx::ops::shape::concatenate_on;
use mlx::Array;

use crate::core::cache::prefix_payload::{
    PagedPrefixEntry, PagedPrefixKeySpec, PrefixLayerPayload, PrefixTensorSpec,
};

/// Random identity of a segment. Lineages refer to segments by id only.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct PrefixSegmentId(pub u128);

impl PrefixSegmentId {
    pub fn new() -> Self {
        Self(uuid::Uuid::new_v4().as_u128())
    }
}

impl Default for PrefixSegmentId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for PrefixSegmentId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:032x}", self.0)
    }
}

/// Identifies which entries may share segments: same model, block size, KV
/// profile and per-layer tensor geometry (everything except lengths).
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct PrefixSegmentLayout(pub String);

const LAYOUT_DOMAIN: &str = "ironmlx-prefix-segment-layout-v1";

fn fnv1a_hex(value: &str) -> String {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in value.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

/// Stable dtype names: the layout names a directory on disk, so it must not
/// depend on `Debug` output.
fn dtype_name(dtype: mlx::Dtype) -> std::borrow::Cow<'static, str> {
    use mlx::Dtype;
    let name = match dtype {
        Dtype::Bool => "bool",
        Dtype::Uint8 => "u8",
        Dtype::Uint16 => "u16",
        Dtype::Uint32 => "u32",
        Dtype::Uint64 => "u64",
        Dtype::Int8 => "i8",
        Dtype::Int16 => "i16",
        Dtype::Int32 => "i32",
        Dtype::Int64 => "i64",
        Dtype::Float16 => "f16",
        Dtype::Float32 => "f32",
        Dtype::Float64 => "f64",
        Dtype::Bfloat16 => "bf16",
        Dtype::Complex64 => "c64",
        // A dtype added later is named by its stable MLX code.
        other => return format!("dtype{}", other as i32).into(),
    };
    name.into()
}

fn layer_kind_name(kind: crate::core::cache::prefix_payload::PrefixLayerKind) -> &'static str {
    use crate::core::cache::prefix_payload::PrefixLayerKind;
    match kind {
        PrefixLayerKind::FullDense => "full_dense",
        PrefixLayerKind::FullPaged => "full_paged",
        PrefixLayerKind::FullTurboQuantPacked => "full_turboquant_packed",
        PrefixLayerKind::Linear => "linear",
        PrefixLayerKind::Mla => "mla",
    }
}

fn entry_kind_name(kind: crate::core::cache::prefix_payload::PrefixEntryKind) -> &'static str {
    use crate::core::cache::prefix_payload::PrefixEntryKind;
    match kind {
        PrefixEntryKind::WholePrefix => "whole_prefix",
        PrefixEntryKind::ImmutableBlock => "immutable_block",
    }
}

fn tensor_geometry(spec: &PrefixTensorSpec, length_dim: Option<usize>) -> String {
    let dims: Vec<String> = spec
        .shape
        .iter()
        .enumerate()
        .map(|(i, d)| {
            if Some(i) == length_dim {
                "*".to_owned()
            } else {
                d.to_string()
            }
        })
        .collect();
    format!("{}[{}]", dtype_name(spec.dtype), dims.join(","))
}

/// Layout of `spec`: every field except `token_ids`, `cached_len`,
/// `fingerprint` and the tensors' length dimension. It names the SSD
/// directory of the entries that may share segments, so it is built from
/// stable names only.
pub fn prefix_segment_layout(spec: &PagedPrefixKeySpec) -> PrefixSegmentLayout {
    use crate::core::cache::prefix_payload::PrefixLayerKind;
    let mut text = format!(
        "{}|{}|{}|{}",
        entry_kind_name(spec.entry_kind),
        spec.model_id,
        spec.block_size,
        spec.kv_cache_profile.as_deref().unwrap_or("-")
    );
    for layer in &spec.main_layers {
        // Paged tensors are [pages, H, block, D]; the page count is the
        // length dimension. Dense, TurboQuant and MLA tensors carry the
        // token count on axis 2. Linear states have no length dimension.
        let length_dim = match layer.kind {
            PrefixLayerKind::FullPaged => Some(0),
            PrefixLayerKind::Linear => None,
            _ => Some(2),
        };
        text.push_str(&format!("|{}", layer_kind_name(layer.kind)));
        for tensor in &layer.tensors {
            text.push(':');
            text.push_str(&tensor_geometry(tensor, length_dim));
        }
    }
    for layer in &spec.mtp_layers {
        text.push_str(&format!(
            "|mtp:{}:{}",
            tensor_geometry(&layer.k, Some(2)),
            tensor_geometry(&layer.v, Some(2))
        ));
    }
    for (name, hidden) in [
        ("mtp_hidden", spec.mtp_last_hidden.as_ref()),
        ("drafter_hidden", spec.gemma4_drafter_last_hidden.as_ref()),
    ] {
        match hidden {
            Some(t) => text.push_str(&format!("|{name}:{}", tensor_geometry(t, None))),
            None => text.push_str(&format!("|{name}:-")),
        }
    }
    PrefixSegmentLayout(fnv1a_hex(&format!("{LAYOUT_DOMAIN}\n{text}")))
}

#[derive(Debug, Clone)]
pub enum PrefixSegmentLayer {
    /// `[pages, H, block, D]` pages covering the segment's token range.
    Paged { k: Array, v: Array },
    TurboQuant {
        k_packed: Array,
        k_norms: Array,
        v_packed: Array,
        v_norms: Array,
    },
}

impl PrefixSegmentLayer {
    fn arrays(&self) -> Vec<&Array> {
        match self {
            Self::Paged { k, v } => vec![k, v],
            Self::TurboQuant {
                k_packed,
                k_norms,
                v_packed,
                v_norms,
            } => vec![k_packed, k_norms, v_packed, v_norms],
        }
    }
}

fn array_bytes(a: &Array) -> usize {
    a.size().saturating_mul(a.dtype().byte_size())
}

/// K/V of `[start, stop)` for every shareable layer. `usize` is the main
/// layer index.
#[derive(Debug)]
pub struct PrefixKvSegment {
    pub id: PrefixSegmentId,
    pub layout: PrefixSegmentLayout,
    pub start: i32,
    pub stop: i32,
    pub layers: Vec<(usize, PrefixSegmentLayer)>,
}

// SAFETY: invariant: a segment's arrays are evaluated before the segment
// enters the hot prefix cache (enforced by `PrefixLruCache::insert_segmented`)
// and are never mutated afterwards. Sharing an evaluated array through `&` only
// reads shape/dtype/data or builds new graphs from it (`clone`, slicing,
// concatenation), which copy MLX's `shared_ptr<ArrayDesc>` with an atomic
// refcount; none of the const methods that mutate an `ArrayDesc` (status,
// event, lazy-to-available) runs on an available array. Segments built by
// `export_prefix_segment_for_row` are evaluated already.
unsafe impl Sync for PrefixKvSegment {}

impl PrefixKvSegment {
    /// Evaluate every array (blocking).
    pub fn eval(&self) -> anyhow::Result<()> {
        let arrays: Vec<&Array> = self.layers.iter().flat_map(|(_, l)| l.arrays()).collect();
        mlx::transforms::eval(&arrays)?;
        Ok(())
    }

    pub fn payload_bytes(&self) -> usize {
        self.layers
            .iter()
            .flat_map(|(_, layer)| layer.arrays())
            .fold(0usize, |bytes, a| bytes.saturating_add(array_bytes(a)))
    }

    fn layer(&self, index: usize) -> Option<&PrefixSegmentLayer> {
        self.layers
            .iter()
            .find(|(i, _)| *i == index)
            .map(|(_, l)| l)
    }
}

/// Uses `[segment.start, stop)`; `stop <= segment.stop` and block aligned.
#[derive(Debug, Clone)]
pub struct PrefixSegmentRef {
    pub segment: Arc<PrefixKvSegment>,
    pub stop: i32,
}

/// One segment a row's K/V came from: the row uses `[start, stop)` of the
/// segment `id`, which covers `[start, segment_stop)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefixLineageRef {
    pub id: PrefixSegmentId,
    pub start: i32,
    pub stop: i32,
    pub segment_stop: i32,
}

impl PrefixLineageRef {
    /// Whether the row holds the segment's whole range, so the row alone
    /// can rebuild the segment.
    pub fn is_whole(&self) -> bool {
        self.stop == self.segment_stop
    }
}

/// Ordered segment ids covering a token prefix; holds no arrays.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefixLineage {
    pub layout: PrefixSegmentLayout,
    pub refs: Vec<PrefixLineageRef>,
}

impl PrefixLineage {
    pub fn covered_len(&self) -> i32 {
        self.refs.last().map_or(0, |r| r.stop)
    }

    /// Cut to `floor(len / block_size) * block_size`, dropping empty refs.
    pub fn truncated(&self, len: i32, block_size: i32) -> PrefixLineage {
        let limit = if block_size > 0 {
            len.max(0) / block_size * block_size
        } else {
            0
        };
        let refs = self
            .refs
            .iter()
            .filter_map(|&r| {
                let stop = r.stop.min(limit);
                (stop > r.start).then_some(PrefixLineageRef { stop, ..r })
            })
            .collect();
        PrefixLineage {
            layout: self.layout.clone(),
            refs,
        }
    }
}

/// An entry whose block-aligned head lives in shared segments. `private`
/// holds the tail `[aligned_len, cached_len)` of the shareable layers (paged:
/// whole pages including padding; TurboQuant: `[1,H,tail,.]`) and everything
/// else exactly as an unshared entry would.
#[derive(Debug, Clone)]
pub struct SegmentedPrefixEntry {
    pub segments: Vec<PrefixSegmentRef>,
    pub aligned_len: i32,
    pub private: PagedPrefixEntry,
}

fn slice_axis(a: &Array, axis: usize, start: i32, stop: i32) -> anyhow::Result<Array> {
    let shape = a.shape();
    let dims = shape.as_slice();
    let mut begin = vec![0_i32; dims.len()];
    let mut end = dims.to_vec();
    begin[axis] = start;
    end[axis] = stop;
    Ok(slice_strided_on(
        a,
        begin,
        end,
        vec![1_i32; dims.len()],
        (),
    )?)
}

/// Position of the token axis (paged: page axis) used for splitting.
fn split_axis(is_paged: bool) -> usize {
    if is_paged {
        0
    } else {
        2
    }
}

fn axis_len(a: &Array, axis: usize) -> i32 {
    a.shape().as_slice()[axis]
}

/// Rebuild the unshared per-layer payloads: segment slices in order, then the
/// private tail, concatenated. Zero-copy when one whole segment is the entire
/// layer.
pub fn materialize_prefix_entry(
    entry: &SegmentedPrefixEntry,
    cached_len: i32,
    block_size: i32,
) -> anyhow::Result<PagedPrefixEntry> {
    if block_size <= 0 || entry.aligned_len % block_size != 0 {
        anyhow::bail!("materialize_prefix_entry: aligned_len must be a multiple of block_size");
    }
    let mut pos = 0;
    for r in &entry.segments {
        if r.segment.start != pos || r.stop > r.segment.stop || r.stop <= r.segment.start {
            anyhow::bail!(
                "materialize_prefix_entry: segment ref [{}, {}) of segment [{}, {}) does not continue at {pos}",
                r.segment.start,
                r.stop,
                r.segment.start,
                r.segment.stop
            );
        }
        if r.stop % block_size != 0 {
            anyhow::bail!(
                "materialize_prefix_entry: segment stop {} not block aligned",
                r.stop
            );
        }
        pos = r.stop;
    }
    if pos != entry.aligned_len {
        anyhow::bail!(
            "materialize_prefix_entry: segments cover {pos} != aligned_len {}",
            entry.aligned_len
        );
    }
    if cached_len < entry.aligned_len {
        anyhow::bail!(
            "materialize_prefix_entry: cached_len {cached_len} < aligned_len {}",
            entry.aligned_len
        );
    }

    let mut out = entry.private.clone();
    for (idx, payload) in out.main_layers.iter_mut().enumerate() {
        match payload {
            PrefixLayerPayload::FullPaged { k_pages, v_pages } => {
                let axis = split_axis(true);
                let mut ks = Vec::new();
                let mut vs = Vec::new();
                for r in &entry.segments {
                    let Some(PrefixSegmentLayer::Paged { k, v }) = r.segment.layer(idx) else {
                        anyhow::bail!("materialize_prefix_entry: segment lacks paged layer {idx}");
                    };
                    let pages = (r.stop - r.segment.start) / block_size;
                    if pages == axis_len(k, axis) {
                        ks.push(k.clone());
                        vs.push(v.clone());
                    } else {
                        ks.push(slice_axis(k, axis, 0, pages)?);
                        vs.push(slice_axis(v, axis, 0, pages)?);
                    }
                }
                if axis_len(k_pages, axis) > 0 {
                    ks.push(k_pages.clone());
                    vs.push(v_pages.clone());
                }
                if entry.segments.is_empty() && ks.len() == 1 {
                    continue;
                }
                *k_pages = concat_axis(ks, axis)?;
                *v_pages = concat_axis(vs, axis)?;
            }
            PrefixLayerPayload::FullTurboQuantPacked {
                k_packed,
                k_norms,
                v_packed,
                v_norms,
            } => {
                let axis = split_axis(false);
                let mut parts: [Vec<Array>; 4] = Default::default();
                for r in &entry.segments {
                    let Some(PrefixSegmentLayer::TurboQuant {
                        k_packed: sk,
                        k_norms: skn,
                        v_packed: sv,
                        v_norms: svn,
                    }) = r.segment.layer(idx)
                    else {
                        anyhow::bail!(
                            "materialize_prefix_entry: segment lacks TurboQuant layer {idx}"
                        );
                    };
                    let len = r.stop - r.segment.start;
                    for (slot, a) in [sk, skn, sv, svn].into_iter().enumerate() {
                        if len == axis_len(a, axis) {
                            parts[slot].push(a.clone());
                        } else {
                            parts[slot].push(slice_axis(a, axis, 0, len)?);
                        }
                    }
                }
                if axis_len(k_packed, axis) > 0 {
                    for (slot, a) in [&*k_packed, &*k_norms, &*v_packed, &*v_norms]
                        .into_iter()
                        .enumerate()
                    {
                        parts[slot].push(a.clone());
                    }
                }
                if entry.segments.is_empty() {
                    continue;
                }
                let [kp, kn, vp, vn] = parts;
                *k_packed = concat_axis(kp, axis)?;
                *k_norms = concat_axis(kn, axis)?;
                *v_packed = concat_axis(vp, axis)?;
                *v_norms = concat_axis(vn, axis)?;
            }
            _ => {}
        }
    }
    Ok(out)
}

fn concat_axis(mut parts: Vec<Array>, axis: usize) -> anyhow::Result<Array> {
    if parts.len() == 1 {
        return Ok(parts.remove(0));
    }
    let refs: Vec<&Array> = parts.iter().collect();
    Ok(concatenate_on(&refs, axis as i32, ())?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::cache::{
        export_prefix_segment_for_row, prefix_entry_for_row, KVCache, LayerCache, TurboQuantKVBits,
    };
    use mlx::Dtype;

    const BS: i32 = 128;
    const HEADS: i32 = 2;
    const DIM: i32 = 8;

    fn tokens(n: i32, salt: f32) -> Array {
        let data: Vec<f32> = (0..(HEADS * n * DIM))
            .map(|i| ((i as f32) * 0.013 + salt).sin())
            .collect();
        (data.as_slice(), (1_i32, HEADS, n, DIM))
            .try_into()
            .unwrap()
    }

    fn filled(kind: &str, n: i32) -> Vec<LayerCache> {
        (0..2)
            .map(|layer| {
                let base = KVCache::new(1, HEADS, DIM, DIM, Dtype::Float32, 512).with_step(128);
                let mut kv = match kind {
                    "paged" => base.with_paged(BS, 8).expect("paged"),
                    _ => base
                        .with_turboquant(TurboQuantKVBits::K3V4)
                        .expect("turboquant"),
                };
                kv.update_and_fetch(
                    &tokens(n, layer as f32),
                    &tokens(n, 3.0 + layer as f32),
                    &[n],
                )
                .expect("fill");
                LayerCache::Full(kv)
            })
            .collect()
    }

    /// Append `n` more tokens to every layer.
    fn extend(caches: &mut [LayerCache], n: i32, salt: f32) {
        for (layer, cache) in caches.iter_mut().enumerate() {
            let LayerCache::Full(kv) = cache else {
                unreachable!()
            };
            kv.update_and_fetch(
                &tokens(n, salt + layer as f32),
                &tokens(n, salt + 3.0 + layer as f32),
                &[n],
            )
            .expect("extend");
        }
    }

    fn layout() -> PrefixSegmentLayout {
        PrefixSegmentLayout("test".into())
    }

    fn export(
        caches: &[LayerCache],
        from_len: i32,
    ) -> (Option<PrefixKvSegment>, PagedPrefixEntry, i32) {
        export_prefix_segment_for_row(caches, 0, from_len, BS, layout())
            .unwrap()
            .unwrap()
    }

    fn flat(a: &Array) -> (Vec<i32>, Vec<f32>, Vec<u32>) {
        a.eval().unwrap();
        let shape = a.shape().as_slice().to_vec();
        match a.dtype() {
            Dtype::Uint32 => (shape, vec![], a.to_vec::<u32>().unwrap()),
            _ => (shape, a.to_vec::<f32>().unwrap(), vec![]),
        }
    }

    fn payload_arrays(entry: &PagedPrefixEntry) -> Vec<Array> {
        entry
            .main_layers
            .iter()
            .flat_map(|l| match l {
                PrefixLayerPayload::FullPaged { k_pages, v_pages } => {
                    vec![k_pages.clone(), v_pages.clone()]
                }
                PrefixLayerPayload::FullTurboQuantPacked {
                    k_packed,
                    k_norms,
                    v_packed,
                    v_norms,
                } => vec![
                    k_packed.clone(),
                    k_norms.clone(),
                    v_packed.clone(),
                    v_norms.clone(),
                ],
                other => panic!("unexpected layer {other:?}"),
            })
            .collect()
    }

    /// A first save at BS (segment `[0, BS)`, empty tail), then a second
    /// save at `2*BS + tail` that references the first segment and adds
    /// `[BS, 2*BS)` plus the tail: materializing equals the whole export.
    fn assert_matches_whole_copy(kind: &str, tail: i32) {
        let mut caches = filled(kind, BS);
        let (seg1, private1, len1) = export(&caches, 0);
        assert_eq!(len1, BS);
        let seg1 = seg1.expect("first segment");
        assert_eq!((seg1.start, seg1.stop), (0, BS));
        assert!(payload_arrays(&private1)
            .iter()
            .all(|a| a.shape().as_slice().contains(&0)));

        extend(&mut caches, BS + tail, 7.0);
        let n = 2 * BS + tail;
        let (seg2, private, len) = export(&caches, BS);
        assert_eq!(len, n);
        let seg2 = seg2.expect("second segment");
        assert_eq!((seg2.start, seg2.stop), (BS, 2 * BS));
        let entry = SegmentedPrefixEntry {
            segments: vec![
                PrefixSegmentRef {
                    segment: Arc::new(seg1),
                    stop: BS,
                },
                PrefixSegmentRef {
                    segment: Arc::new(seg2),
                    stop: 2 * BS,
                },
            ],
            aligned_len: 2 * BS,
            private,
        };
        let rebuilt = materialize_prefix_entry(&entry, n, BS).unwrap();

        let (whole, whole_len) = prefix_entry_for_row(&caches, 0).unwrap().unwrap();
        assert_eq!(whole_len, n);
        let (a, b) = (payload_arrays(&whole), payload_arrays(&rebuilt));
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(&b) {
            assert_eq!(flat(x), flat(y), "{kind} tail={tail}");
        }
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn materialize_matches_whole_copy_for_paged_layers() {
        for tail in [0, 1, 127] {
            assert_matches_whole_copy("paged", tail);
        }
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn materialize_matches_whole_copy_for_turboquant_layers() {
        for tail in [0, 1, 127] {
            assert_matches_whole_copy("turboquant", tail);
        }
    }

    fn all_arrays(seg: &PrefixKvSegment, private: &PagedPrefixEntry) -> Vec<Array> {
        seg.layers
            .iter()
            .flat_map(|(_, l)| l.arrays())
            .cloned()
            .chain(payload_arrays(private))
            .collect()
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn exported_segment_and_tail_survive_the_live_cache() {
        for kind in ["paged", "turboquant"] {
            let mut caches = filled(kind, 300);
            let (seg, private, _) = export(&caches, 0);
            let seg = seg.expect("segment");
            assert_eq!((seg.start, seg.stop), (0, 256));
            let before: Vec<_> = all_arrays(&seg, &private).iter().map(flat).collect();

            for cache in &mut caches {
                if let LayerCache::Full(kv) = cache {
                    kv.reset();
                }
            }
            drop(caches);
            mlx::transforms::clear_cache();

            let after: Vec<_> = all_arrays(&seg, &private).iter().map(flat).collect();
            assert_eq!(before, after, "{kind}");
            let expected: usize = seg
                .layers
                .iter()
                .flat_map(|(_, l)| l.arrays())
                .map(|a| a.size() * a.dtype().byte_size())
                .sum();
            assert_eq!(seg.payload_bytes(), expected);
        }

        let caches = filled("turboquant", 300);
        let (seg, private, _) = export(&caches, 0);
        let seg = seg.expect("segment");
        let heads = HEADS as usize;
        for (_, layer) in &seg.layers {
            let PrefixSegmentLayer::TurboQuant {
                k_packed, k_norms, ..
            } = layer
            else {
                panic!("expected TurboQuant layer");
            };
            assert_eq!(array_bytes(k_norms), heads * 256 * 4);
            assert_eq!(
                array_bytes(k_packed),
                heads * 256 * k_packed.shape().as_slice()[3] as usize * 4
            );
        }
        for layer in &private.main_layers {
            let PrefixLayerPayload::FullTurboQuantPacked { k_norms, .. } = layer else {
                panic!("expected TurboQuant tail");
            };
            assert_eq!(k_norms.shape().as_slice(), &[1, HEADS, 300 - 256]);
        }
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn paged_tail_export_is_the_partial_last_page() {
        let caches = filled("paged", 200);
        let (whole, _) = prefix_entry_for_row(&caches, 0).unwrap().unwrap();
        let (seg, tail, len) = export(&caches, BS);
        assert_eq!(len, 200);
        assert!(seg.is_none(), "[BS, aligned) is empty");
        for (w, t) in whole.main_layers.iter().zip(&tail.main_layers) {
            let (
                PrefixLayerPayload::FullPaged {
                    k_pages: wk,
                    v_pages: wv,
                },
                PrefixLayerPayload::FullPaged {
                    k_pages: tk,
                    v_pages: tv,
                },
            ) = (w, t)
            else {
                panic!("expected paged layers");
            };
            assert_eq!(tk.shape().as_slice(), &[1, HEADS, BS, DIM]);
            assert_eq!(tv.shape().as_slice(), &[1, HEADS, BS, DIM]);
            assert_eq!(flat(tk), flat(&slice_axis(wk, 0, 1, 2).unwrap()));
            assert_eq!(flat(tv), flat(&slice_axis(wv, 0, 1, 2).unwrap()));
        }
        // `from_len` past the aligned length or off a block boundary.
        assert!(export_prefix_segment_for_row(&caches, 0, 2 * BS, BS, layout()).is_err());
        assert!(export_prefix_segment_for_row(&caches, 0, 1, BS, layout()).is_err());
        assert!(export_prefix_segment_for_row(&caches, 0, 0, 64, layout()).is_err());
    }

    #[test]
    fn segment_layout_is_stable_across_builds() {
        use crate::core::cache::prefix_payload::{
            PagedPrefixKeySpec, PrefixEntryKind, PrefixLayerKind, PrefixLayerSpec,
        };
        let tensor = |dtype, shape: Vec<i32>| PrefixTensorSpec { dtype, shape };
        let spec = PagedPrefixKeySpec {
            entry_kind: PrefixEntryKind::WholePrefix,
            model_id: "qwen3-test".to_owned(),
            token_ids: vec![1, 2, 3],
            cached_len: 3,
            fingerprint: None,
            block_size: 128,
            kv_cache_profile: Some("turboquant-k3v4".to_owned()),
            main_layers: vec![
                PrefixLayerSpec {
                    kind: PrefixLayerKind::FullPaged,
                    tensors: vec![
                        tensor(Dtype::Bfloat16, vec![1, 2, 128, 8]),
                        tensor(Dtype::Bfloat16, vec![1, 2, 128, 8]),
                    ],
                },
                PrefixLayerSpec {
                    kind: PrefixLayerKind::Linear,
                    tensors: vec![
                        tensor(Dtype::Float32, vec![1, 3, 16]),
                        tensor(Dtype::Float32, vec![1, 2, 4, 4]),
                    ],
                },
            ],
            mtp_layers: vec![],
            mtp_last_hidden: None,
            gemma4_drafter_last_hidden: None,
        };
        // Changing this breaks every `wp8/<layout>` directory on disk.
        assert_eq!(prefix_segment_layout(&spec).0, "69023322dabb0d31");
    }

    #[test]
    fn lineage_truncates_to_block_boundary() {
        let (a, b) = (PrefixSegmentId(1), PrefixSegmentId(2));
        let r = |id, start, stop| PrefixLineageRef {
            id,
            start,
            stop,
            segment_stop: stop,
        };
        let lineage = PrefixLineage {
            layout: layout(),
            refs: vec![r(a, 0, 256), r(b, 256, 512)],
        };
        let cut = lineage.truncated(300, 128);
        assert_eq!(cut.refs, vec![r(a, 0, 256)]);
        assert!(cut.refs[0].is_whole());
        let cut = lineage.truncated(400, 128);
        assert_eq!(
            cut.refs[1],
            PrefixLineageRef {
                id: b,
                start: 256,
                stop: 384,
                segment_stop: 512
            }
        );
        assert!(!cut.refs[1].is_whole());
        assert_eq!(cut.covered_len(), 384);
        assert_eq!(lineage.truncated(300, 128).covered_len(), 256);
        assert_eq!(lineage.covered_len(), 512);
        assert_eq!(
            PrefixLineage {
                layout: layout(),
                refs: vec![]
            }
            .covered_len(),
            0
        );
    }
}
