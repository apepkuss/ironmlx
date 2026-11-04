//! Sparse QSA history with immutable SSD pages and bounded selected-page
//! gathers. Indexer summaries and recurrent/PLE state remain in RAM.
use crate::core::active_pages::{prefetch, ActivePage, ActivePageFile, ActivePagesPool};
use anyhow::{ensure, Context, Result};
use mlx::ops::shape::{concatenate_on, contiguous_on, stack_on, transpose_axes_on};
use mlx::{Array, Dtype, StreamOrDevice};
use std::{collections::BTreeSet, sync::Arc};

#[derive(Clone)]
pub(crate) struct PagedQsa {
    pool: Arc<ActivePagesPool>,
    file: Arc<ActivePageFile>,
    pages: Arc<Vec<Arc<ActivePage>>>,
    tail: Arc<Vec<u8>>,
    tokens: usize,
    ratio: usize,
    heads: i32,
    dim: i32,
}
impl PagedQsa {
    pub fn new(pool: Arc<ActivePagesPool>, heads: i32, dim: i32, ratio: i32) -> Result<Self> {
        ensure!(
            heads > 0 && dim > 0 && ratio > 0,
            "invalid active QSA geometry"
        );
        let bytes = (heads as usize)
            .checked_mul(dim as usize)
            .and_then(|n| n.checked_mul(4))
            .and_then(|n| n.checked_mul(ratio as usize))
            .context("active QSA page overflow")?;
        let file = pool.file(bytes)?;
        Ok(Self {
            pool,
            file,
            pages: Arc::default(),
            tail: Arc::default(),
            tokens: 0,
            ratio: ratio as usize,
            heads,
            dim,
        })
    }
    fn token_bytes(&self) -> usize {
        self.heads as usize * self.dim as usize * 4
    }
    fn page_bytes(&self) -> usize {
        self.ratio * self.token_bytes()
    }
    pub fn append(&mut self, keys: &Array, values: &Array, target: StreamOrDevice) -> Result<()> {
        let shape = keys.shape();
        let dims = shape.as_slice();
        ensure!(
            dims.len() == 4
                && dims[0] == 1
                && dims[1] == self.heads
                && dims[2] > 0
                && dims[3] == self.dim
                && keys.dtype() == Dtype::Bfloat16
                && values.shape() == shape
                && values.dtype() == keys.dtype(),
            "invalid active QSA append"
        );
        let k = transpose_axes_on(keys, [0, 2, 1, 3], target)?;
        let v = transpose_axes_on(values, [0, 2, 1, 3], target)?;
        let packed = contiguous_on(&stack_on(&[&k, &v], 2, target)?, false, target)?;
        packed.eval()?;
        let mut bytes = vec![0; dims[2] as usize * self.token_bytes()];
        packed.read_bytes_into(0, &mut bytes)?;
        let mut combined = Vec::with_capacity(self.tail.len() + bytes.len());
        combined.extend_from_slice(&self.tail);
        combined.extend(bytes);
        // Do not mutate the logical generation until every write succeeds.
        let payloads: Vec<_> = combined.chunks_exact(self.page_bytes()).collect();
        let mut written = Vec::with_capacity(payloads.len());
        let pages_per_batch = self.pool.config.pack_bytes / self.page_bytes();
        for batch in payloads.chunks(pages_per_batch) {
            // A large admitted prefill can exceed the gather payload ceiling.
            // Keep each write bounded and publish only after all batches succeed.
            written.extend(self.file.write_batch(batch)?);
        }
        let tail = combined
            .chunks_exact(self.page_bytes())
            .remainder()
            .to_vec();
        if !written.is_empty() {
            Arc::make_mut(&mut self.pages).extend(written);
        }
        self.tail = Arc::new(tail);
        self.tokens += dims[2] as usize;
        Ok(())
    }
    pub fn trim(&mut self, tokens: usize) -> Result<()> {
        ensure!(tokens <= self.tokens, "active QSA rollback exceeds history");
        let complete = tokens / self.ratio;
        let remainder = tokens % self.ratio;
        let tail = if remainder == 0 {
            Vec::new()
        } else if complete == self.pages.len() {
            self.tail[..remainder * self.token_bytes()].to_vec()
        } else {
            self.pages[complete].read()?[..remainder * self.token_bytes()].to_vec()
        };
        if complete < self.pages.len() {
            Arc::make_mut(&mut self.pages).truncate(complete);
        }
        self.tail = Arc::new(tail);
        self.tokens = tokens;
        Ok(())
    }
    fn arrays(&self, pages: &[usize], target: StreamOrDevice) -> Result<(Array, Array)> {
        ensure!(
            !pages.is_empty()
                && pages
                    .len()
                    .checked_mul(self.page_bytes())
                    .is_some_and(|n| n <= self.pool.config.pack_bytes),
            "active QSA gather exceeds RAM budget"
        );
        ensure!(
            pages.iter().all(|&p| p < self.tokens.div_ceil(self.ratio)),
            "active QSA page out of range"
        );
        let disk: Vec<_> = pages
            .iter()
            .filter_map(|&p| self.pages.get(p).cloned())
            .collect();
        let loaded = prefetch(&disk)?;
        let mut loaded = loaded.into_iter();
        let mut bytes = Vec::with_capacity(pages.len() * self.page_bytes());
        for &p in pages {
            if p < self.pages.len() {
                bytes.extend_from_slice(
                    &loaded
                        .next()
                        .context("active QSA missing prefetched page")?,
                );
            } else {
                bytes.extend_from_slice(&self.tail);
                bytes.resize(bytes.len() + self.page_bytes() - self.tail.len(), 0);
            }
        }
        let tokens = i32::try_from(pages.len() * self.ratio)?;
        let packed =
            Array::from_bytes(&bytes, &[tokens, 2, self.heads, self.dim], Dtype::Bfloat16)?;
        let get = |which| -> Result<Array> {
            let array = mlx::ops::indexing::slice_strided_on(
                &packed,
                [0, which, 0, 0],
                [tokens, which + 1, self.heads, self.dim],
                [1, 1, 1, 1],
                target,
            )?
            .reshape_on([1, tokens, self.heads, self.dim], target)?;
            Ok(transpose_axes_on(&array, [0, 2, 1, 3], target)?)
        };
        Ok((get(0)?, get(1)?))
    }
    pub fn materialize(&self, target: StreamOrDevice) -> Result<(Array, Array)> {
        let (k, v) = self.arrays(
            &(0..self.tokens.div_ceil(self.ratio)).collect::<Vec<_>>(),
            target,
        )?;
        let cut = |a: &Array| -> Result<Array> {
            Ok(mlx::ops::indexing::slice_strided_on(
                a,
                [0, 0, 0, 0],
                [1, self.heads, self.tokens as i32, self.dim],
                [1, 1, 1, 1],
                target,
            )?)
        };
        Ok((cut(&k)?, cut(&v)?))
    }
    pub fn attention(
        &self,
        q: &Array,
        selected: Option<&Array>,
        start: i32,
        scale: f32,
        target: StreamOrDevice,
    ) -> Result<Array> {
        let seq = q.shape().as_slice()[2] as usize;
        let (selection, picks) = if let Some(selected) = selected {
            (
                selected.to_vec::<i32>()?,
                selected.shape().as_slice()[2] as usize,
            )
        } else {
            (Vec::new(), 0)
        };
        let tiles = plan(
            &selection,
            picks,
            seq,
            start as usize,
            self.ratio,
            self.pool.config.pack_bytes / self.page_bytes(),
        )?;
        let mut outputs = Vec::new();
        for tile in tiles {
            let (k, v) = self.arrays(&tile.pages, target)?;
            let queries = mlx::ops::indexing::slice_strided_on(
                q,
                [0, 0, tile.start as i32, 0],
                [1, q.shape().as_slice()[1], tile.end as i32, self.dim],
                [1, 1, 1, 1],
                target,
            )?;
            let output = if seq == 1 {
                if self.tokens >= 4096 {
                    super::indexed_sdpa::decode(
                        &queries,
                        &k,
                        &v,
                        &tile.pages,
                        self.tokens,
                        self.ratio,
                        scale,
                        target,
                    )?
                } else {
                    // A rollback across the tier boundary can temporarily fall
                    // below native two-pass dispatch. This bounded history uses
                    // the original native shape and mask, never renumbered keys.
                    let (keys, values) = self.materialize(target)?;
                    let mut mask = vec![false; self.tokens];
                    for &page in &tile.pages {
                        let start = page * self.ratio;
                        let end = (start + self.ratio).min(self.tokens);
                        mask[start..end].fill(true);
                    }
                    let mask = Array::try_from((mask.as_slice(), [1, 1, 1, self.tokens as i32]))?;
                    mlx::fast::scaled_dot_product_attention_on(
                        &queries,
                        &keys,
                        &values,
                        scale,
                        "",
                        if selected.is_some() {
                            Some(&mask)
                        } else {
                            None
                        },
                        None,
                        target,
                    )?
                }
            } else {
                let selected = Array::try_from((
                    tile.selected.as_slice(),
                    [1, (tile.end - tile.start) as i32, tile.picks as i32],
                ))?;
                let ends =
                    Array::try_from((tile.ends.as_slice(), [1, (tile.end - tile.start) as i32]))?;
                super::qsa_kernel::prefill_on(
                    &queries,
                    &k,
                    &v,
                    &selected,
                    &ends,
                    scale,
                    self.ratio as i32,
                    target,
                )?
            };
            // Release packed history before packing another tile; otherwise a
            // lazy output list would retain every tile's selected KV buffers.
            output.eval()?;
            outputs.push(output);
        }
        Ok(concatenate_on(
            &outputs.iter().collect::<Vec<_>>(),
            2,
            target,
        )?)
    }
}

struct Tile {
    start: usize,
    end: usize,
    pages: Vec<usize>,
    selected: Vec<i32>,
    ends: Vec<i32>,
    picks: usize,
}
fn plan(
    selected: &[i32],
    picks: usize,
    seq: usize,
    start: usize,
    ratio: usize,
    max_pages: usize,
) -> Result<Vec<Tile>> {
    ensure!(
        ratio > 0 && seq > 0 && max_pages > 0 && selected.len() == seq * picks,
        "invalid active QSA selection layout"
    );
    let mut tiles = Vec::new();
    let mut first = 0;
    while first < seq {
        let mut union = BTreeSet::new();
        let mut last = first;
        while last < seq {
            let end = start + last + 1;
            let complete = end / ratio;
            let mut pages: Vec<usize> = if picks == 0 {
                (0..complete).collect()
            } else {
                selected[last * picks..(last + 1) * picks]
                    .iter()
                    .filter_map(|&b| (b >= 0 && (b as usize) < complete).then_some(b as usize))
                    .collect()
            };
            if !end.is_multiple_of(ratio) {
                pages.push(complete);
            }
            let added: BTreeSet<_> = pages.into_iter().filter(|p| !union.contains(p)).collect();
            if union.len() + added.len() > max_pages {
                ensure!(
                    last > first,
                    "active QSA gather budget cannot fit one query"
                );
                break;
            }
            union.extend(added);
            last += 1;
        }
        let pages: Vec<_> = union.into_iter().collect();
        ensure!(!pages.is_empty(), "active QSA selected no history");
        let output_picks = if picks == 0 {
            (start + last).div_ceil(ratio).max(1)
        } else {
            picks
        };
        let mut mapped = Vec::new();
        let mut ends = Vec::new();
        for row in first..last {
            let end = start + row + 1;
            let complete = end / ratio;
            let selected: Vec<i32> = if picks == 0 {
                std::iter::repeat_n(-1, output_picks - complete)
                    .chain(0..complete as i32)
                    .collect()
            } else {
                selected[row * picks..(row + 1) * picks].to_vec()
            };
            for b in selected {
                mapped.push(if b < 0 || b as usize >= complete {
                    -1
                } else {
                    pages
                        .binary_search(&(b as usize))
                        .map_err(|_| anyhow::anyhow!("active QSA lost selected block"))?
                        as i32
                });
            }
            let tail = pages.partition_point(|&p| p < complete);
            ends.push(i32::try_from(tail * ratio + end % ratio)?);
        }
        tiles.push(Tile {
            start: first,
            end: last,
            pages,
            selected: mapped,
            ends,
            picks: output_picks,
        });
        first = last;
    }
    Ok(tiles)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires actual Apple GPU; bounded multi-batch writes and Q1 prefill tiles"]
    fn large_append_q1_tiles_and_partial_write_failure_are_transactional() {
        let root = std::env::temp_dir().join(format!("qsa-batched-{}", uuid::Uuid::new_v4()));
        let cfg = crate::core::active_pages::ActivePagesConfig {
            root: root.clone(),
            cache_bytes: 8192,
            pack_bytes: 8 << 20,
            disk_bytes: 32 << 20,
            min_free_bytes: 0,
        };
        let pool = ActivePagesPool::new(cfg.clone()).unwrap();
        let seq = 6147;
        let data =
            mlx::ops::constructors::arange(0., (2 * seq * 256) as f64, 1., Dtype::Float32).unwrap();
        let k = mlx::ops::unary::sin(&data)
            .unwrap()
            .reshape([1, 2, seq, 256])
            .unwrap()
            .astype(Dtype::Bfloat16)
            .unwrap();
        let v = mlx::ops::unary::cos(&data)
            .unwrap()
            .reshape([1, 2, seq, 256])
            .unwrap()
            .astype(Dtype::Bfloat16)
            .unwrap();
        let mut history = PagedQsa::new(pool.clone(), 2, 256, 4).unwrap();
        history.append(&k, &v, ().into()).unwrap();
        assert_eq!(history.tokens, seq as usize);
        let picks: Vec<i32> = (0..1024).collect();
        let selected = Array::try_from((picks.as_slice(), [1, 2, 512])).unwrap();
        let tiles = plan(&picks, 512, 2, 6145, 4, 1024).unwrap();
        assert_eq!(tiles.len(), 2);
        assert!(tiles.iter().all(|t| t.end - t.start == 1));
        let q = Array::try_from((vec![1f32; 24 * 2 * 256].as_slice(), [1, 24, 2, 256]))
            .unwrap()
            .astype(Dtype::Bfloat16)
            .unwrap();
        let ends = Array::try_from((&[6146i32, 6147][..], [1, 2])).unwrap();
        let expected = super::super::qsa_kernel::prefill_on(
            &q,
            &k,
            &v,
            &selected,
            &ends,
            1. / 16.,
            4,
            ().into(),
        )
        .unwrap();
        let actual = history
            .attention(&q, Some(&selected), 6145, 1. / 16., ().into())
            .unwrap();
        assert_eq!(
            expected.to_vec::<half::bf16>().unwrap(),
            actual.to_vec::<half::bf16>().unwrap()
        );
        history.trim(4093).unwrap();
        let (a, b) = history.materialize(().into()).unwrap();
        let cut = |x: &Array, end| x.slice([0, 0, 0, 0], [1, 2, end, 256]).unwrap();
        assert_eq!(
            a.to_vec::<half::bf16>().unwrap(),
            cut(&k, 4093).to_vec::<half::bf16>().unwrap()
        );
        assert_eq!(
            b.to_vec::<half::bf16>().unwrap(),
            cut(&v, 4093).to_vec::<half::bf16>().unwrap()
        );
        drop(history);
        assert_eq!(pool.stats.snapshot().disk_bytes, 0);
        drop(pool);
        let pool = ActivePagesPool::new(crate::core::active_pages::ActivePagesConfig {
            disk_bytes: 1024 * (8192 + 40),
            ..cfg
        })
        .unwrap();
        let mut history = PagedQsa::new(pool.clone(), 2, 256, 4).unwrap();
        assert!(history.append(&k, &v, ().into()).is_err());
        assert_eq!(history.tokens, 0);
        assert!(history.pages.is_empty() && history.tail.is_empty());
        assert_eq!(pool.stats.snapshot().live_pages, 0);
        history.append(&cut(&k, 4), &cut(&v, 4), ().into()).unwrap();
        let (a, b) = history.materialize(().into()).unwrap();
        assert_eq!(
            a.to_vec::<half::bf16>().unwrap(),
            cut(&k, 4).to_vec::<half::bf16>().unwrap()
        );
        assert_eq!(
            b.to_vec::<half::bf16>().unwrap(),
            cut(&v, 4).to_vec::<half::bf16>().unwrap()
        );
        drop(history);
        assert_eq!(pool.stats.snapshot().live_files, 0);
        assert_eq!(pool.stats.snapshot().disk_bytes, 0);
        drop(pool);
        std::fs::remove_dir(root).unwrap();
    }
    #[test]
    fn packed_causal_positions_preserve_sparse_selection_and_partial_tails() {
        let t = plan(&[0, 3, 0, 4], 2, 2, 18, 4, 8).unwrap();
        assert_eq!(t.len(), 1);
        assert_eq!(t[0].pages, [0, 3, 4]);
        assert_eq!(t[0].selected, [0, 1, 0, 2]);
        assert_eq!(t[0].ends, [11, 12]);
    }
    #[test]
    fn tiles_bound_union_and_keep_causal_sentinels() {
        let t = plan(&[-1, 0, 1, 2], 2, 2, 11, 4, 3).unwrap();
        assert_eq!(t.len(), 2);
        assert_eq!(t[0].selected, [-1, 0]);
        assert_eq!(t[1].pages, [1, 2, 3]);
    }
}
