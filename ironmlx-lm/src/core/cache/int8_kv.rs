//! 8-bit affine K/V storage (groups of 64 along the head dimension, as MLX's
//! `quantize` lays them out) for one full-history attention layer of a single
//! row. Reads that need dense K/V get them dequantized; single-query decode
//! reads the packed buffers directly.
use mlx::ops::indexing::{slice_strided_on, slice_update_on};
use mlx::ops::shape::concatenate_on;
use mlx::{Array, Dtype, StreamOrDevice};

use crate::core::cache::TurboQuantPrefixLayer;
use crate::Result;

pub const INT8_KV_GROUP: i32 = 64;
const BITS: i32 = 8;

/// The packed form of one side (K or V): `[1, H, capacity, D / 4]` `uint32`
/// words and `[1, H, capacity, D / 64]` scales and biases.
#[derive(Clone)]
pub struct Int8KvSide {
    pub words: Array,
    pub scales: Array,
    pub biases: Array,
}

#[derive(Clone)]
pub struct Int8KvStorage {
    heads: i32,
    head_dim: i32,
    v_head_dim: i32,
    keys: Option<Int8KvSide>,
    values: Option<Int8KvSide>,
}

impl Int8KvStorage {
    pub fn supports(head_dim: i32, v_head_dim: i32) -> bool {
        head_dim > 0
            && v_head_dim > 0
            && head_dim % INT8_KV_GROUP == 0
            && v_head_dim % INT8_KV_GROUP == 0
    }

    pub fn new(heads: i32, head_dim: i32, v_head_dim: i32) -> Self {
        Self {
            heads,
            head_dim,
            v_head_dim,
            keys: None,
            values: None,
        }
    }

    pub fn capacity(&self) -> i32 {
        self.keys.as_ref().map_or(0, |side| side.words.shape_at(2))
    }

    /// The packed K and V buffers, valid below the cache's offset.
    pub fn sides(&self) -> Option<(&Int8KvSide, &Int8KvSide)> {
        Some((self.keys.as_ref()?, self.values.as_ref()?))
    }

    pub fn arrays(&self) -> Vec<&Array> {
        let mut out = Vec::new();
        for side in self.keys.iter().chain(self.values.iter()) {
            out.extend([&side.words, &side.scales, &side.biases]);
        }
        out
    }

    pub fn clear(&mut self) {
        self.keys = None;
        self.values = None;
    }

    fn empty_side(&self, capacity: i32, dim: i32, target: StreamOrDevice) -> Result<Int8KvSide> {
        Ok(Int8KvSide {
            words: Array::zeros_on((1, self.heads, capacity, dim / 4), Dtype::Uint32, target)?,
            scales: Array::zeros_on(
                (1, self.heads, capacity, dim / INT8_KV_GROUP),
                Dtype::Bfloat16,
                target,
            )?,
            biases: Array::zeros_on(
                (1, self.heads, capacity, dim / INT8_KV_GROUP),
                Dtype::Bfloat16,
                target,
            )?,
        })
    }

    fn grow_side(
        old: Option<&Int8KvSide>,
        kept: i32,
        fresh: Int8KvSide,
        target: StreamOrDevice,
    ) -> Result<Int8KvSide> {
        let Some(old) = old.filter(|_| kept > 0) else {
            return Ok(fresh);
        };
        let join = |old: &Array, fresh: &Array| -> Result<Array> {
            let s = old.shape();
            let dims = s.as_slice();
            let head = slice_strided_on(
                old,
                [0_i32, 0, 0, 0],
                [dims[0], dims[1], kept, dims[3]],
                [1_i32, 1, 1, 1],
                target,
            )?;
            let f = fresh.shape();
            let fd = f.as_slice();
            let tail = slice_strided_on(
                fresh,
                [0_i32, 0, kept, 0],
                [fd[0], fd[1], fd[2], fd[3]],
                [1_i32, 1, 1, 1],
                target,
            )?;
            Ok(concatenate_on(&[&head, &tail], 2, target)?)
        };
        Ok(Int8KvSide {
            words: join(&old.words, &fresh.words)?,
            scales: join(&old.scales, &fresh.scales)?,
            biases: join(&old.biases, &fresh.biases)?,
        })
    }

    fn grow_to(&mut self, capacity: i32, kept: i32, target: StreamOrDevice) -> Result<()> {
        let keys = self.empty_side(capacity, self.head_dim, target)?;
        let values = self.empty_side(capacity, self.v_head_dim, target)?;
        self.keys = Some(Self::grow_side(self.keys.as_ref(), kept, keys, target)?);
        self.values = Some(Self::grow_side(self.values.as_ref(), kept, values, target)?);
        Ok(())
    }

    fn write_side(
        side: &Int8KvSide,
        x: &Array,
        offset: i32,
        target: StreamOrDevice,
    ) -> Result<Int8KvSide> {
        let parts = mlx::quantization::quantize_on(
            x,
            Some(INT8_KV_GROUP),
            Some(BITS),
            "affine",
            None,
            target,
        )?;
        let [words, scales, biases]: [Array; 3] = parts
            .try_into()
            .map_err(|_| anyhow::anyhow!("affine quantize returned an unexpected part count"))?;
        let put = |dst: &Array, src: &Array| -> Result<Array> {
            let s = src.shape();
            let d = s.as_slice();
            Ok(slice_update_on(
                dst,
                src,
                [0_i32, 0, offset, 0],
                [d[0], d[1], offset + d[2], d[3]],
                [1_i32, 1, 1, 1],
                target,
            )?)
        };
        Ok(Int8KvSide {
            words: put(&side.words, &words)?,
            scales: put(&side.scales, &scales.astype_on(Dtype::Bfloat16, target)?)?,
            biases: put(&side.biases, &biases.astype_on(Dtype::Bfloat16, target)?)?,
        })
    }

    fn dequantize_side(
        side: &Int8KvSide,
        len: i32,
        dtype: Dtype,
        target: StreamOrDevice,
    ) -> Result<Array> {
        // MLX's dequantize reads its inputs as row-contiguous.
        let head = |a: &Array| -> Result<Array> {
            let s = a.shape();
            let d = s.as_slice();
            let head = slice_strided_on(
                a,
                [0_i32, 0, 0, 0],
                [d[0], d[1], len, d[3]],
                [1_i32, 1, 1, 1],
                target,
            )?;
            Ok(if len == d[2] {
                head
            } else {
                head.contiguous_on(false, target)?
            })
        };
        Ok(mlx::quantization::dequantize_on(
            &head(&side.words)?,
            &head(&side.scales)?,
            Some(&head(&side.biases)?),
            Some(INT8_KV_GROUP),
            Some(BITS),
            "affine",
            None,
            Some(dtype),
            target,
        )?)
    }

    /// The first `len` positions of K and V dequantized to `dtype`.
    pub fn dequantized_on(
        &self,
        len: i32,
        dtype: Dtype,
        target: StreamOrDevice,
    ) -> Result<(Array, Array)> {
        let (keys, values) = self
            .sides()
            .ok_or_else(|| anyhow::anyhow!("8-bit K/V storage holds nothing yet"))?;
        Ok((
            Self::dequantize_side(keys, len, dtype, target)?,
            Self::dequantize_side(values, len, dtype, target)?,
        ))
    }

    /// Width of one side's packed `uint32` words, and of its scales and
    /// biases joined along the last axis, for a prefix payload.
    pub fn packed_prefix_dims(&self) -> [i32; 4] {
        [
            self.head_dim / 4,
            2 * self.head_dim / INT8_KV_GROUP,
            self.v_head_dim / 4,
            2 * self.v_head_dim / INT8_KV_GROUP,
        ]
    }

    /// Positions `[start, stop)` (below `len`, the cache's offset) as a
    /// prefix payload that owns its buffers: each side's words, and its
    /// scales and biases joined along the last axis. Saving the quantized
    /// data as it is keeps a restore bit-identical to what was saved.
    pub fn owned_packed_range_on(
        &self,
        len: i32,
        start: i32,
        stop: i32,
        target: StreamOrDevice,
    ) -> Result<TurboQuantPrefixLayer> {
        anyhow::ensure!(
            0 <= start && start <= stop && stop <= len && len <= self.capacity().max(0),
            "8-bit K/V range [{start}, {stop}) outside [0, {len})"
        );
        let [kw, ksb, vw, vsb] = self.packed_prefix_dims();
        if start == stop {
            let empty =
                |dim: i32, dtype: Dtype| Array::zeros_on((1, self.heads, 0, dim), dtype, target);
            return Ok(TurboQuantPrefixLayer {
                k_packed: empty(kw, Dtype::Uint32)?,
                k_norms: empty(ksb, Dtype::Bfloat16)?,
                v_packed: empty(vw, Dtype::Uint32)?,
                v_norms: empty(vsb, Dtype::Bfloat16)?,
            });
        }
        let (keys, values) = self
            .sides()
            .ok_or_else(|| anyhow::anyhow!("8-bit K/V storage holds nothing yet"))?;
        let cut = |a: &Array| -> Result<Array> {
            let s = a.shape();
            let d = s.as_slice();
            Ok(slice_strided_on(
                a,
                [0_i32, 0, start, 0],
                [d[0], d[1], stop, d[3]],
                [1_i32, 1, 1, 1],
                target,
            )?)
        };
        let words = |side: &Int8KvSide| -> Result<Array> {
            super::paged_kv::owned_copy_on(&cut(&side.words)?, target)
        };
        let scales_biases = |side: &Int8KvSide| -> Result<Array> {
            Ok(concatenate_on(
                &[&cut(&side.scales)?, &cut(&side.biases)?],
                3,
                target,
            )?)
        };
        Ok(TurboQuantPrefixLayer {
            k_packed: words(keys)?,
            k_norms: scales_biases(keys)?,
            v_packed: words(values)?,
            v_norms: scales_biases(values)?,
        })
    }

    /// Positions `[0, len)` in the prefix payload form: views of the words,
    /// and each side's scales and biases joined (a small copy).
    pub fn packed_prefix_view_on(
        &self,
        len: i32,
        target: StreamOrDevice,
    ) -> Result<TurboQuantPrefixLayer> {
        if len == 0 {
            return self.owned_packed_range_on(0, 0, 0, target);
        }
        anyhow::ensure!(
            0 < len && len <= self.capacity(),
            "8-bit K/V prefix of {len} positions outside [0, {}]",
            self.capacity()
        );
        let (keys, values) = self
            .sides()
            .ok_or_else(|| anyhow::anyhow!("8-bit K/V storage holds nothing yet"))?;
        let head = |a: &Array| -> Result<Array> {
            let s = a.shape();
            let d = s.as_slice();
            Ok(slice_strided_on(
                a,
                [0_i32, 0, 0, 0],
                [d[0], d[1], len, d[3]],
                [1_i32, 1, 1, 1],
                target,
            )?)
        };
        let scales_biases = |side: &Int8KvSide| -> Result<Array> {
            Ok(concatenate_on(
                &[&head(&side.scales)?, &head(&side.biases)?],
                3,
                target,
            )?)
        };
        Ok(TurboQuantPrefixLayer {
            k_packed: head(&keys.words)?,
            k_norms: scales_biases(keys)?,
            v_packed: head(&values.words)?,
            v_norms: scales_biases(values)?,
        })
    }

    /// Write a prefix saved by [`Self::owned_packed_range_on`], held as
    /// consecutive parts along the token axis, at positions `[0,
    /// cached_len)`, growing the buffers in steps of `step` up to `cap`.
    /// The data are written as saved, not quantized again.
    pub fn restore_packed_parts_on(
        &mut self,
        parts: &[TurboQuantPrefixLayer],
        cached_len: i32,
        step: i32,
        cap: i32,
        target: StreamOrDevice,
    ) -> Result<()> {
        anyhow::ensure!(
            0 <= cached_len && cached_len <= cap,
            "8-bit K/V prefix of {cached_len} positions outside [0, {cap}]"
        );
        let [kw, ksb, vw, vsb] = self.packed_prefix_dims();
        let mut total = 0;
        for part in parts {
            let len = part.k_packed.shape_at(2);
            for (array, dim, dtype) in [
                (&part.k_packed, kw, Dtype::Uint32),
                (&part.k_norms, ksb, Dtype::Bfloat16),
                (&part.v_packed, vw, Dtype::Uint32),
                (&part.v_norms, vsb, Dtype::Bfloat16),
            ] {
                anyhow::ensure!(
                    array.dtype() == dtype && array.shape().as_slice() == [1, self.heads, len, dim],
                    "8-bit K/V prefix part {:?} {:?} incompatible with [1,{},{len},{dim}] {dtype:?}",
                    array.shape().as_slice(),
                    array.dtype(),
                    self.heads
                );
            }
            total += len;
        }
        anyhow::ensure!(
            total == cached_len,
            "8-bit K/V prefix parts hold {total} positions != {cached_len}"
        );
        if cached_len == 0 {
            return Ok(());
        }
        if cached_len > self.capacity() || self.keys.is_none() {
            let step = step.max(1);
            let capacity = ((cached_len + step - 1) / step * step).min(cap);
            self.grow_to(capacity, 0, target)?;
        }
        let put = |dst: &Array, src: &Array, off: i32| -> Result<Array> {
            let s = src.shape();
            let d = s.as_slice();
            Ok(slice_update_on(
                dst,
                src,
                [0_i32, 0, off, 0],
                [d[0], d[1], off + d[2], d[3]],
                [1_i32, 1, 1, 1],
                target,
            )?)
        };
        let half = |a: &Array, second: bool| -> Result<Array> {
            let s = a.shape();
            let d = s.as_slice();
            let mid = d[3] / 2;
            let (lo, hi) = if second { (mid, d[3]) } else { (0, mid) };
            Ok(slice_strided_on(
                a,
                [0_i32, 0, 0, lo],
                [d[0], d[1], d[2], hi],
                [1_i32, 1, 1, 1],
                target,
            )?)
        };
        let write =
            |side: &Int8KvSide, words: &Array, sb: &Array, off: i32| -> Result<Int8KvSide> {
                Ok(Int8KvSide {
                    words: put(&side.words, words, off)?,
                    scales: put(&side.scales, &half(sb, false)?, off)?,
                    biases: put(&side.biases, &half(sb, true)?, off)?,
                })
            };
        let mut off = 0;
        for part in parts {
            let len = part.k_packed.shape_at(2);
            if len == 0 {
                continue;
            }
            let keys = self.keys.as_ref().expect("allocated above");
            let values = self.values.as_ref().expect("allocated above");
            let keys = write(keys, &part.k_packed, &part.k_norms, off)?;
            let values = write(values, &part.v_packed, &part.v_norms, off)?;
            self.keys = Some(keys);
            self.values = Some(values);
            off += len;
        }
        // Slice updates from strided sources can leave the buffers in a
        // layout MLX's dequantize does not read correctly.
        for side in [&mut self.keys, &mut self.values].into_iter().flatten() {
            side.words = side.words.contiguous_on(false, target)?;
            side.scales = side.scales.contiguous_on(false, target)?;
            side.biases = side.biases.contiguous_on(false, target)?;
        }
        Ok(())
    }

    /// Quantize and append `k`, `v` (`[1, H, S, D]`) at `offset`, growing
    /// the buffers in steps of `step` up to `cap`. Returns the whole history
    /// dequantized to `dtype`; graphs that only read the packed buffers leave
    /// it unevaluated.
    #[allow(clippy::too_many_arguments)]
    pub fn update_and_fetch_on(
        &mut self,
        k: &Array,
        v: &Array,
        offset: i32,
        len: i32,
        step: i32,
        cap: i32,
        dtype: Dtype,
        target: StreamOrDevice,
    ) -> Result<(Array, Array)> {
        anyhow::ensure!(
            k.shape_at(0) == 1 && v.shape_at(0) == 1,
            "8-bit K/V storage holds one row"
        );
        let end = offset + len;
        anyhow::ensure!(end <= cap, "KVCache cap {cap} exceeded: {offset} + {len}");
        if end == 0 && self.keys.is_none() {
            return Ok((
                Array::zeros_on((1, self.heads, 0, self.head_dim), dtype, target)?,
                Array::zeros_on((1, self.heads, 0, self.v_head_dim), dtype, target)?,
            ));
        }
        if end > self.capacity() || self.keys.is_none() {
            let step = step.max(1);
            let capacity = ((end + step - 1) / step * step).min(cap);
            self.grow_to(capacity, offset, target)?;
        }
        if len > 0 {
            let take = |x: &Array| -> Result<Array> {
                if x.shape_at(2) == len {
                    return Ok(x.clone());
                }
                let s = x.shape();
                let d = s.as_slice();
                Ok(slice_strided_on(
                    x,
                    [0_i32, 0, 0, 0],
                    [d[0], d[1], len, d[3]],
                    [1_i32, 1, 1, 1],
                    target,
                )?)
            };
            let keys = self.keys.as_ref().expect("allocated above");
            let values = self.values.as_ref().expect("allocated above");
            let keys = Self::write_side(keys, &take(k)?, offset, target)?;
            let values = Self::write_side(values, &take(v)?, offset, target)?;
            self.keys = Some(keys);
            self.values = Some(values);
        }
        let (keys, values) = self.sides().expect("allocated above");
        Ok((
            Self::dequantize_side(keys, end, dtype, target)?,
            Self::dequantize_side(values, end, dtype, target)?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use crate::core::cache::KVCache;
    use mlx::{Array, Dtype};

    fn normal(shape: &[i32], seed: u64) -> Array {
        mlx::random::normal()
            .shape(shape)
            .key(&mlx::random::key(seed).unwrap())
            .sample()
            .unwrap()
            .astype(Dtype::Bfloat16)
            .unwrap()
    }

    fn cache() -> KVCache {
        let mut cache = KVCache::new(1, 2, 128, 128, Dtype::Bfloat16, 1024).with_step(64);
        assert!(cache.enable_int8());
        cache
    }

    fn same(a: &Array, b: &Array) -> bool {
        a.shape() == b.shape()
            && a.not_equal(b)
                .unwrap()
                .astype(Dtype::Int32)
                .unwrap()
                .sum(mlx::ops::All, false)
                .unwrap()
                .item::<i32>()
                .unwrap()
                == 0
    }

    fn slice(a: &Array, start: i32, stop: i32) -> Array {
        let s = a.shape();
        let d = s.as_slice();
        mlx::ops::indexing::slice(a, &[0, 0, start, 0], &[d[0], d[1], stop, d[3]]).unwrap()
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn appends_in_pieces_and_across_growth_match_one_append() {
        let (k, v) = (normal(&[1, 2, 150, 128], 1), normal(&[1, 2, 150, 128], 2));
        let mut whole = cache();
        let (wk, wv) = whole.update_and_fetch(&k, &v, &[150]).unwrap();
        let mut pieces = cache();
        // Single-token appends cross the 64-position growth steps.
        for (start, stop) in [(0, 30), (30, 31), (31, 100), (100, 101), (101, 150)] {
            pieces
                .update_and_fetch(
                    &slice(&k, start, stop),
                    &slice(&v, start, stop),
                    &[stop - start],
                )
                .unwrap();
        }
        let (pk, pv) = pieces
            .update_and_fetch(&slice(&k, 0, 0), &slice(&v, 0, 0), &[0])
            .unwrap();
        assert_eq!(pieces.offsets(), &[150]);
        assert!(same(&wk, &pk) && same(&wv, &pv));
        // Each key is quantized on its own, so the packed data agree too.
        let (a, b) = (whole.int8().unwrap(), pieces.int8().unwrap());
        let (ak, _) = a.sides().unwrap();
        let (bk, _) = b.sides().unwrap();
        assert!(same(&slice(&ak.words, 0, 150), &slice(&bk.words, 0, 150)));
        assert!(same(&slice(&ak.scales, 0, 150), &slice(&bk.scales, 0, 150)));
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn rollback_then_append_overwrites_the_rejected_keys() {
        let (k, v) = (normal(&[1, 2, 40, 128], 3), normal(&[1, 2, 40, 128], 4));
        let mut cache = cache();
        cache
            .update_and_fetch(&slice(&k, 0, 32), &slice(&v, 0, 32), &[32])
            .unwrap();
        let snapshot = cache.snapshot();
        assert!(!snapshot.restores_dense_storage());
        let (rejected_k, rejected_v) = (normal(&[1, 2, 4, 128], 5), normal(&[1, 2, 4, 128], 6));
        cache
            .update_and_fetch(&rejected_k, &rejected_v, &[4])
            .unwrap();
        cache.restore(&snapshot).unwrap();
        assert_eq!(cache.offsets(), &[32]);
        let (got_k, got_v) = cache
            .update_and_fetch(&slice(&k, 32, 40), &slice(&v, 32, 40), &[8])
            .unwrap();
        let mut fresh = self::cache();
        let (want_k, want_v) = fresh.update_and_fetch(&k, &v, &[40]).unwrap();
        assert!(same(&got_k, &want_k) && same(&got_v, &want_v));
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn dense_only_paths_refuse_eight_bit_storage() {
        let mut cache = cache();
        let (k, v) = (normal(&[1, 2, 8, 128], 7), normal(&[1, 2, 8, 128], 8));
        cache.update_and_fetch(&k, &v, &[8]).unwrap();
        assert!(!cache.is_plain_dense());
        assert!(cache.dense_prefix_layer_for_row_on(0, ()).is_err());
        assert!(cache.owned_dense_range_for_row_on(0, 0, 8, ()).is_err());
        assert_eq!(
            cache.prefix_cache_profile().as_deref(),
            Some("int8-affine-g64")
        );
        // Only one row holding its whole history, before any write.
        assert!(!KVCache::new(2, 2, 128, 128, Dtype::Bfloat16, 64).enable_int8());
        assert!(!KVCache::new(1, 2, 128, 128, Dtype::Bfloat16, 64)
            .with_window(16, 4)
            .enable_int8());
        assert!(!KVCache::new(1, 2, 96, 96, Dtype::Bfloat16, 64).enable_int8());
    }
    #[test]
    #[serial_test::serial(mlx_metal)]
    fn adoption_keeps_eight_bit_rows_and_dequantizes_into_dense_batches() {
        let (k, v) = (normal(&[1, 2, 20, 128], 9), normal(&[1, 2, 20, 128], 10));
        let mut src = cache();
        let (want_k, want_v) = src.update_and_fetch(&k, &v, &[20]).unwrap();
        let mut dst = cache();
        dst.adopt_row_from(&src, 0, 0).unwrap();
        assert_eq!(dst.offsets(), &[20]);
        let (got_k, got_v) = dst
            .update_and_fetch(&slice(&k, 0, 0), &slice(&v, 0, 0), &[0])
            .unwrap();
        assert!(same(&got_k, &want_k) && same(&got_v, &want_v));
        let mut batch = KVCache::new(2, 2, 128, 128, Dtype::Bfloat16, 1024);
        batch.adopt_row_from(&src, 1, 0).unwrap();
        assert_eq!(batch.offsets(), &[0, 20]);
        let (dense_k, _) = batch
            .dense_prefix_layer_for_row_on(1, ())
            .map(|(k, v, _)| (k, v))
            .unwrap();
        assert!(same(&dense_k, &want_k));
    }
}
