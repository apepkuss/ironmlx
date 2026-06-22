use mlx::ops::indexing::{slice_strided_on, slice_update_on};
use mlx::ops::shape::concatenate_on;
use mlx::{Array, Dtype, StreamOrDevice};

use crate::Result;

/// Fixed-size paged K/V storage for full-attention decode.
///
/// Physical storage is `[page, Hkv, block_size, D]`. Each batch row owns a
/// logical block table mapping token blocks to physical pages. Multi-token
/// prefill appends into pages and then materializes the dense prefix for the
/// existing SDPA path; single-token decode appends and dispatches the paged
/// attention kernel directly.
pub struct PagedKVCache {
    k_pages: Option<Array>,
    v_pages: Option<Array>,
    block_tables: Vec<Vec<i32>>,
    free_pages: Vec<i32>,
    page_ref_counts: Vec<i32>,
    allocated_pages: i32,
    page_capacity: i32,
    max_pages: i32,
    max_blocks_per_row: i32,
    batch: i32,
    n_kv_heads: i32,
    head_dim: i32,
    v_head_dim: i32,
    dtype: Dtype,
    cap: i32,
    block_size: i32,
    page_grow_step: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PageRun {
    src_start: i32,
    dst_start: i32,
    len: i32,
}

fn contiguous_page_runs(dst_pages: &[i32]) -> Vec<PageRun> {
    if dst_pages.is_empty() {
        return Vec::new();
    }

    let mut runs = Vec::new();
    let mut src_start = 0_i32;
    let mut dst_start = dst_pages[0];
    let mut len = 1_i32;
    for (idx, &page) in dst_pages.iter().enumerate().skip(1) {
        if page == dst_start + len {
            len += 1;
            continue;
        }
        runs.push(PageRun {
            src_start,
            dst_start,
            len,
        });
        src_start = idx as i32;
        dst_start = page;
        len = 1;
    }
    runs.push(PageRun {
        src_start,
        dst_start,
        len,
    });
    runs
}

impl PagedKVCache {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        batch: i32,
        n_kv_heads: i32,
        head_dim: i32,
        v_head_dim: i32,
        dtype: Dtype,
        cap: i32,
        block_size: i32,
        max_pages: i32,
    ) -> Result<Self> {
        if batch <= 0 {
            anyhow::bail!("PagedKVCache::new: batch must be > 0, got {batch}");
        }
        if n_kv_heads <= 0 {
            anyhow::bail!("PagedKVCache::new: n_kv_heads must be > 0, got {n_kv_heads}");
        }
        if head_dim <= 0 || v_head_dim <= 0 {
            anyhow::bail!(
                "PagedKVCache::new: head dims must be > 0, got key={head_dim} value={v_head_dim}"
            );
        }
        if head_dim != v_head_dim {
            anyhow::bail!(
                "PagedKVCache::new: paged decode requires value head_dim == key head_dim, got {v_head_dim} != {head_dim}"
            );
        }
        if cap < 0 {
            anyhow::bail!("PagedKVCache::new: cap must be >= 0, got {cap}");
        }
        if block_size <= 0 {
            anyhow::bail!("PagedKVCache::new: block_size must be > 0, got {block_size}");
        }
        if max_pages <= 0 {
            anyhow::bail!("PagedKVCache::new: max_pages must be > 0, got {max_pages}");
        }
        let max_blocks_per_row = ceil_div(cap, block_size).max(1);
        let block_tables = vec![vec![-1; max_blocks_per_row as usize]; batch as usize];
        Ok(Self {
            k_pages: None,
            v_pages: None,
            block_tables,
            free_pages: Vec::new(),
            page_ref_counts: Vec::new(),
            allocated_pages: 0,
            page_capacity: 0,
            max_pages,
            max_blocks_per_row,
            batch,
            n_kv_heads,
            head_dim,
            v_head_dim,
            dtype,
            cap,
            block_size,
            page_grow_step: 64,
        })
    }

    pub fn allocated_pages(&self) -> i32 {
        self.allocated_pages
    }

    pub fn block_size(&self) -> i32 {
        self.block_size
    }

    pub fn max_pages(&self) -> i32 {
        self.max_pages
    }

    pub fn block_table_row(&self, row: usize) -> &[i32] {
        &self.block_tables[row]
    }

    pub fn k_pages(&self) -> Option<&Array> {
        self.k_pages.as_ref()
    }

    pub fn v_pages(&self) -> Option<&Array> {
        self.v_pages.as_ref()
    }

    pub fn capacity(&self) -> i32 {
        self.cap
    }

    pub fn batch(&self) -> i32 {
        self.batch
    }

    pub fn n_kv_heads(&self) -> i32 {
        self.n_kv_heads
    }

    pub fn head_dim(&self) -> i32 {
        self.head_dim
    }

    pub fn v_head_dim(&self) -> i32 {
        self.v_head_dim
    }

    pub fn dtype(&self) -> Dtype {
        self.dtype
    }

    pub fn grow_cap(&mut self, new_cap: i32) {
        if new_cap <= self.cap {
            return;
        }
        self.cap = new_cap;
        let new_blocks = ceil_div(new_cap, self.block_size).max(1);
        if new_blocks > self.max_blocks_per_row {
            for row in &mut self.block_tables {
                row.resize(new_blocks as usize, -1);
            }
            self.max_blocks_per_row = new_blocks;
        }
    }

    pub fn clear(&mut self) {
        for row in &mut self.block_tables {
            row.fill(-1);
        }
        self.free_pages.clear();
        self.page_ref_counts.fill(0);
        self.allocated_pages = 0;
    }

    pub fn restore_offsets(
        &mut self,
        current_offsets: &mut [i32],
        new_offsets: &[i32],
    ) -> Result<()> {
        self.validate_offsets_shape_and_cap(new_offsets)?;
        self.validate_offsets_allocated(new_offsets)?;
        for (row, (&old_off, &new_off)) in current_offsets.iter().zip(new_offsets).enumerate() {
            if new_off < old_off {
                self.release_row_pages_from(row, ceil_div(new_off, self.block_size));
            }
        }
        current_offsets.clone_from_slice(new_offsets);
        Ok(())
    }

    pub fn update_and_fetch_on(
        &mut self,
        k: &Array,
        v: &Array,
        offsets: &mut [i32],
        per_row_lens: &[i32],
        target: impl Into<StreamOrDevice>,
    ) -> Result<(Array, Array)> {
        let target = target.into();
        if per_row_lens.iter().all(|&n| n == 0) {
            self.validate_update_inputs(k, v, offsets, per_row_lens)?;
            let empty_k = Array::zeros_on(
                (self.batch, self.n_kv_heads, 0_i32, self.head_dim),
                self.dtype,
                target,
            )?;
            let empty_v = Array::zeros_on(
                (self.batch, self.n_kv_heads, 0_i32, self.v_head_dim),
                self.dtype,
                target,
            )?;
            return Ok((empty_k, empty_v));
        }
        self.append_on(k, v, offsets, per_row_lens, target)?;
        let max_off = offsets.iter().copied().max().unwrap_or(0);
        self.materialize_prefix_on(offsets, max_off, target)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn update_and_attend_decode_on(
        &mut self,
        queries: &Array,
        k: &Array,
        v: &Array,
        offsets: &mut [i32],
        per_row_lens: &[i32],
        scale: f32,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        let target = target.into();
        self.validate_decode_inputs(queries, k, v, offsets, per_row_lens)?;
        self.append_on(k, v, offsets, per_row_lens, target)?;
        let max_blocks = offsets
            .iter()
            .map(|&off| ceil_div(off, self.block_size))
            .max()
            .unwrap_or(0)
            .max(1);
        let block_table = self.block_table_array(max_blocks)?;
        let lengths_vec = offsets.to_vec();
        let lengths: Array = (lengths_vec.as_slice(), &[self.batch][..]).try_into()?;
        let k_pages = self.k_pages.as_ref().ok_or_else(|| {
            anyhow::anyhow!("PagedKVCache::update_and_attend_decode_on: K pages are unallocated")
        })?;
        let v_pages = self.v_pages.as_ref().ok_or_else(|| {
            anyhow::anyhow!("PagedKVCache::update_and_attend_decode_on: V pages are unallocated")
        })?;
        Ok(mlx::fast::paged_scaled_dot_product_attention_decode_on(
            queries,
            k_pages,
            v_pages,
            &block_table,
            &lengths,
            scale,
            self.block_size,
            target,
        )?)
    }

    pub fn materialize_prefix_on(
        &self,
        offsets: &[i32],
        max_len: i32,
        target: impl Into<StreamOrDevice>,
    ) -> Result<(Array, Array)> {
        let target = target.into();
        self.validate_offsets_shape_and_cap(offsets)?;
        if max_len < 0 || max_len > self.cap {
            anyhow::bail!(
                "PagedKVCache::materialize_prefix_on: max_len {max_len} outside [0, {}]",
                self.cap
            );
        }
        if offsets.iter().any(|&off| off > max_len) {
            anyhow::bail!(
                "PagedKVCache::materialize_prefix_on: max_len {max_len} is smaller than at least one row offset"
            );
        }
        if max_len == 0 {
            let empty_k = Array::zeros_on(
                (self.batch, self.n_kv_heads, 0_i32, self.head_dim),
                self.dtype,
                target,
            )?;
            let empty_v = Array::zeros_on(
                (self.batch, self.n_kv_heads, 0_i32, self.v_head_dim),
                self.dtype,
                target,
            )?;
            return Ok((empty_k, empty_v));
        }
        self.validate_offsets_allocated(offsets)?;

        let k_pages = self.k_pages.as_ref().ok_or_else(|| {
            anyhow::anyhow!("PagedKVCache::materialize_prefix_on: K pages are unallocated")
        })?;
        let v_pages = self.v_pages.as_ref().ok_or_else(|| {
            anyhow::anyhow!("PagedKVCache::materialize_prefix_on: V pages are unallocated")
        })?;
        let mut dense_k = Array::zeros_on(
            (self.batch, self.n_kv_heads, max_len, self.head_dim),
            self.dtype,
            target,
        )?;
        let mut dense_v = Array::zeros_on(
            (self.batch, self.n_kv_heads, max_len, self.v_head_dim),
            self.dtype,
            target,
        )?;

        for (row_usize, &row_len) in offsets.iter().enumerate() {
            let row = row_usize as i32;
            let blocks = ceil_div(row_len, self.block_size);
            for block_col in 0..blocks {
                let logical_start = block_col * self.block_size;
                let take = (row_len - logical_start).min(self.block_size);
                let page = self.block_tables[row_usize][block_col as usize];
                if page < 0 {
                    anyhow::bail!(
                        "PagedKVCache::materialize_prefix_on: missing page for row {row_usize} block {block_col}"
                    );
                }
                let k_page = slice_strided_on(
                    k_pages,
                    [page, 0, 0, 0],
                    [page + 1, self.n_kv_heads, take, self.head_dim],
                    [1_i32, 1, 1, 1],
                    target,
                )?;
                let v_page = slice_strided_on(
                    v_pages,
                    [page, 0, 0, 0],
                    [page + 1, self.n_kv_heads, take, self.v_head_dim],
                    [1_i32, 1, 1, 1],
                    target,
                )?;
                dense_k = slice_update_on(
                    &dense_k,
                    &k_page,
                    [row, 0, logical_start, 0],
                    [
                        row + 1,
                        self.n_kv_heads,
                        logical_start + take,
                        self.head_dim,
                    ],
                    [1_i32, 1, 1, 1],
                    target,
                )?;
                dense_v = slice_update_on(
                    &dense_v,
                    &v_page,
                    [row, 0, logical_start, 0],
                    [
                        row + 1,
                        self.n_kv_heads,
                        logical_start + take,
                        self.v_head_dim,
                    ],
                    [1_i32, 1, 1, 1],
                    target,
                )?;
            }
        }

        Ok((dense_k, dense_v))
    }

    pub fn prefix_pages_for_row_on(
        &self,
        offsets: &[i32],
        row: usize,
        target: impl Into<StreamOrDevice>,
    ) -> Result<(Array, Array)> {
        let target = target.into();
        self.validate_offsets_allocated(offsets)?;
        if row >= self.batch as usize {
            anyhow::bail!(
                "PagedKVCache::prefix_pages_for_row_on: row {row} >= batch {}",
                self.batch
            );
        }
        let row_len = offsets[row];
        let blocks = ceil_div(row_len, self.block_size);
        if blocks == 0 {
            let empty_k = Array::zeros_on(
                (0_i32, self.n_kv_heads, self.block_size, self.head_dim),
                self.dtype,
                target,
            )?;
            let empty_v = Array::zeros_on(
                (0_i32, self.n_kv_heads, self.block_size, self.v_head_dim),
                self.dtype,
                target,
            )?;
            return Ok((empty_k, empty_v));
        }
        let k_pages = self.k_pages.as_ref().ok_or_else(|| {
            anyhow::anyhow!("PagedKVCache::prefix_pages_for_row_on: K pages are unallocated")
        })?;
        let v_pages = self.v_pages.as_ref().ok_or_else(|| {
            anyhow::anyhow!("PagedKVCache::prefix_pages_for_row_on: V pages are unallocated")
        })?;

        let mut k_parts = Vec::with_capacity(blocks as usize);
        let mut v_parts = Vec::with_capacity(blocks as usize);
        for block_col in 0..blocks {
            let page = self.block_tables[row][block_col as usize];
            if page < 0 {
                anyhow::bail!(
                    "PagedKVCache::prefix_pages_for_row_on: missing page row {row} block {block_col}"
                );
            }
            k_parts.push(slice_strided_on(
                k_pages,
                [page, 0, 0, 0],
                [page + 1, self.n_kv_heads, self.block_size, self.head_dim],
                [1_i32, 1, 1, 1],
                target,
            )?);
            v_parts.push(slice_strided_on(
                v_pages,
                [page, 0, 0, 0],
                [page + 1, self.n_kv_heads, self.block_size, self.v_head_dim],
                [1_i32, 1, 1, 1],
                target,
            )?);
        }

        if k_parts.len() == 1 {
            Ok((k_parts.remove(0), v_parts.remove(0)))
        } else {
            let k_refs = k_parts.iter().collect::<Vec<_>>();
            let v_refs = v_parts.iter().collect::<Vec<_>>();
            Ok((
                concatenate_on(&k_refs, 0, target)?,
                concatenate_on(&v_refs, 0, target)?,
            ))
        }
    }

    pub fn restore_prefix_pages_for_row_on(
        &mut self,
        k_pages_src: &Array,
        v_pages_src: &Array,
        offsets: &mut [i32],
        row: usize,
        prefix_len: i32,
        target: impl Into<StreamOrDevice>,
    ) -> Result<()> {
        self.restore_prefix_pages_for_rows_on(
            k_pages_src,
            v_pages_src,
            offsets,
            &[row],
            prefix_len,
            target,
        )
    }

    pub fn restore_prefix_pages_for_rows_on(
        &mut self,
        k_pages_src: &Array,
        v_pages_src: &Array,
        offsets: &mut [i32],
        rows: &[usize],
        prefix_len: i32,
        target: impl Into<StreamOrDevice>,
    ) -> Result<()> {
        let target = target.into();
        self.validate_offsets_shape_and_cap(offsets)?;
        if rows.is_empty() {
            return Ok(());
        }
        let mut seen_rows = vec![false; self.batch as usize];
        for &row in rows {
            if row >= self.batch as usize {
                anyhow::bail!(
                    "PagedKVCache::restore_prefix_pages_for_rows_on: row {row} >= batch {}",
                    self.batch
                );
            }
            if seen_rows[row] {
                anyhow::bail!(
                    "PagedKVCache::restore_prefix_pages_for_rows_on: duplicate row {row}"
                );
            }
            seen_rows[row] = true;
        }
        if prefix_len < 0 || prefix_len > self.cap {
            anyhow::bail!(
                "PagedKVCache::restore_prefix_pages_for_rows_on: prefix_len {prefix_len} outside [0, {}]",
                self.cap
            );
        }
        self.validate_prefix_pages("K", k_pages_src, self.head_dim)?;
        self.validate_prefix_pages("V", v_pages_src, self.v_head_dim)?;
        let needed_blocks = ceil_div(prefix_len, self.block_size);
        let available_blocks = k_pages_src.shape().as_slice()[0];
        if needed_blocks > available_blocks {
            anyhow::bail!(
                "PagedKVCache::restore_prefix_pages_for_rows_on: prefix_len {prefix_len} needs {needed_blocks} pages but source has {available_blocks}"
            );
        }

        for &row in rows {
            self.release_row_pages(row);
            offsets[row] = 0;
        }
        if needed_blocks == 0 {
            return Ok(());
        }

        let tail_private = rows.len() > 1 && prefix_len % self.block_size != 0;
        let shared_blocks = if tail_private {
            needed_blocks - 1
        } else {
            needed_blocks
        };
        let total_pages = shared_blocks + if tail_private { rows.len() as i32 } else { 0 };
        if total_pages > self.max_pages {
            anyhow::bail!(
                "PagedKVCache::restore_prefix_pages_for_rows_on: restoring {total_pages} pages exceeds max_pages {}",
                self.max_pages
            );
        }

        let start = std::time::Instant::now();
        if self.has_live_pages() {
            self.copy_prefix_pages_once_for_rows_on(
                k_pages_src,
                v_pages_src,
                rows,
                shared_blocks,
                tail_private,
                target,
            )?;
        } else {
            self.install_prefix_pages_direct_on(
                k_pages_src,
                v_pages_src,
                rows,
                shared_blocks,
                tail_private,
                target,
            )?;
        }
        for &row in rows {
            offsets[row] = prefix_len;
        }
        if std::env::var_os("IRONMLX_PAGED_PREFIX_RESTORE_PROFILE").is_some() {
            tracing::info!(
                "[paged-prefix-profile] event=batch_restore_install rows={} prefix_len={} source_pages={} cache_pages={} shared_pages={} private_tail_pages={} elapsed_ms={:.3}",
                rows.len(),
                prefix_len,
                needed_blocks,
                total_pages,
                shared_blocks,
                if tail_private { rows.len() } else { 0 },
                start.elapsed().as_secs_f64() * 1000.0
            );
        }
        Ok(())
    }

    fn install_prefix_pages_direct_on(
        &mut self,
        k_pages_src: &Array,
        v_pages_src: &Array,
        rows: &[usize],
        shared_blocks: i32,
        tail_private: bool,
        target: StreamOrDevice,
    ) -> Result<()> {
        let private_tail_pages = if tail_private { rows.len() as i32 } else { 0 };
        let total_pages = shared_blocks + private_tail_pages;
        self.free_pages.clear();
        self.allocated_pages = total_pages;
        self.page_capacity = total_pages;
        self.page_ref_counts.clear();
        self.page_ref_counts.resize(total_pages as usize, 0);
        self.k_pages = Some(self.build_prefix_page_tensor_on(
            k_pages_src,
            shared_blocks,
            private_tail_pages,
            self.head_dim,
            target,
        )?);
        self.v_pages = Some(self.build_prefix_page_tensor_on(
            v_pages_src,
            shared_blocks,
            private_tail_pages,
            self.v_head_dim,
            target,
        )?);

        for page in 0..shared_blocks {
            self.page_ref_counts[page as usize] = rows.len() as i32;
        }
        for (idx, &row) in rows.iter().enumerate() {
            for block_col in 0..shared_blocks {
                self.block_tables[row][block_col as usize] = block_col;
            }
            if tail_private {
                let page = shared_blocks + idx as i32;
                self.page_ref_counts[page as usize] = 1;
                self.block_tables[row][shared_blocks as usize] = page;
            }
        }
        Ok(())
    }

    fn build_prefix_page_tensor_on(
        &self,
        pages_src: &Array,
        shared_blocks: i32,
        private_tail_pages: i32,
        dim: i32,
        target: StreamOrDevice,
    ) -> Result<Array> {
        let mut parts = Vec::new();
        let source_pages = pages_src.shape().as_slice()[0];
        if shared_blocks > 0 {
            if private_tail_pages == 0 && shared_blocks == source_pages {
                parts.push(pages_src.clone());
            } else {
                parts.push(slice_strided_on(
                    pages_src,
                    [0_i32, 0, 0, 0],
                    [shared_blocks, self.n_kv_heads, self.block_size, dim],
                    [1_i32, 1, 1, 1],
                    target,
                )?);
            }
        }
        if private_tail_pages > 0 {
            let tail = slice_strided_on(
                pages_src,
                [shared_blocks, 0, 0, 0],
                [shared_blocks + 1, self.n_kv_heads, self.block_size, dim],
                [1_i32, 1, 1, 1],
                target,
            )?;
            for _ in 0..private_tail_pages {
                parts.push(tail.clone());
            }
        }

        if parts.len() == 1 {
            Ok(parts.remove(0))
        } else {
            let refs = parts.iter().collect::<Vec<_>>();
            Ok(concatenate_on(&refs, 0, target)?)
        }
    }

    fn copy_prefix_pages_once_for_rows_on(
        &mut self,
        k_pages_src: &Array,
        v_pages_src: &Array,
        rows: &[usize],
        shared_blocks: i32,
        tail_private: bool,
        target: StreamOrDevice,
    ) -> Result<()> {
        let mut shared_pages = Vec::with_capacity(shared_blocks as usize);
        for _ in 0..shared_blocks {
            shared_pages.push(self.allocate_page(target)?);
        }
        for &page in &shared_pages {
            self.set_page_ref_count(page, rows.len() as i32);
        }
        self.copy_prefix_page_runs_on(k_pages_src, v_pages_src, &shared_pages, target)?;

        let mut private_tail_pages = Vec::with_capacity(if tail_private { rows.len() } else { 0 });
        if tail_private {
            for _ in rows {
                let page = self.allocate_page(target)?;
                self.copy_single_prefix_page_on(
                    k_pages_src,
                    v_pages_src,
                    shared_blocks,
                    page,
                    target,
                )?;
                private_tail_pages.push(page);
            }
        }

        for (idx, &row) in rows.iter().enumerate() {
            for (block_col, &page) in shared_pages.iter().enumerate() {
                self.block_tables[row][block_col] = page;
            }
            if tail_private {
                self.block_tables[row][shared_blocks as usize] = private_tail_pages[idx];
            }
        }
        Ok(())
    }

    fn copy_prefix_page_runs_on(
        &mut self,
        k_pages_src: &Array,
        v_pages_src: &Array,
        dst_pages: &[i32],
        target: StreamOrDevice,
    ) -> Result<()> {
        for run in contiguous_page_runs(dst_pages) {
            let k_page_run = slice_strided_on(
                k_pages_src,
                [run.src_start, 0, 0, 0],
                [
                    run.src_start + run.len,
                    self.n_kv_heads,
                    self.block_size,
                    self.head_dim,
                ],
                [1_i32, 1, 1, 1],
                target,
            )?;
            let v_page_run = slice_strided_on(
                v_pages_src,
                [run.src_start, 0, 0, 0],
                [
                    run.src_start + run.len,
                    self.n_kv_heads,
                    self.block_size,
                    self.v_head_dim,
                ],
                [1_i32, 1, 1, 1],
                target,
            )?;
            let dst_k_pages = self
                .k_pages
                .as_ref()
                .expect("copy_prefix_page_runs_on allocated K storage");
            let dst_v_pages = self
                .v_pages
                .as_ref()
                .expect("copy_prefix_page_runs_on allocated V storage");
            self.k_pages = Some(slice_update_on(
                dst_k_pages,
                &k_page_run,
                [run.dst_start, 0, 0, 0],
                [
                    run.dst_start + run.len,
                    self.n_kv_heads,
                    self.block_size,
                    self.head_dim,
                ],
                [1_i32, 1, 1, 1],
                target,
            )?);
            self.v_pages = Some(slice_update_on(
                dst_v_pages,
                &v_page_run,
                [run.dst_start, 0, 0, 0],
                [
                    run.dst_start + run.len,
                    self.n_kv_heads,
                    self.block_size,
                    self.v_head_dim,
                ],
                [1_i32, 1, 1, 1],
                target,
            )?);
        }
        Ok(())
    }

    fn copy_single_prefix_page_on(
        &mut self,
        k_pages_src: &Array,
        v_pages_src: &Array,
        src_page: i32,
        dst_page: i32,
        target: StreamOrDevice,
    ) -> Result<()> {
        let k_page = slice_strided_on(
            k_pages_src,
            [src_page, 0, 0, 0],
            [
                src_page + 1,
                self.n_kv_heads,
                self.block_size,
                self.head_dim,
            ],
            [1_i32, 1, 1, 1],
            target,
        )?;
        let v_page = slice_strided_on(
            v_pages_src,
            [src_page, 0, 0, 0],
            [
                src_page + 1,
                self.n_kv_heads,
                self.block_size,
                self.v_head_dim,
            ],
            [1_i32, 1, 1, 1],
            target,
        )?;
        let k_pages = self
            .k_pages
            .as_ref()
            .expect("copy_single_prefix_page_on allocated K storage");
        let v_pages = self
            .v_pages
            .as_ref()
            .expect("copy_single_prefix_page_on allocated V storage");
        self.k_pages = Some(slice_update_on(
            k_pages,
            &k_page,
            [dst_page, 0, 0, 0],
            [
                dst_page + 1,
                self.n_kv_heads,
                self.block_size,
                self.head_dim,
            ],
            [1_i32, 1, 1, 1],
            target,
        )?);
        self.v_pages = Some(slice_update_on(
            v_pages,
            &v_page,
            [dst_page, 0, 0, 0],
            [
                dst_page + 1,
                self.n_kv_heads,
                self.block_size,
                self.v_head_dim,
            ],
            [1_i32, 1, 1, 1],
            target,
        )?);
        Ok(())
    }

    pub fn adopt_row_from(
        &mut self,
        src: &PagedKVCache,
        dst_offsets: &mut [i32],
        src_offsets: &[i32],
        dst_row: usize,
        src_row: usize,
    ) -> Result<()> {
        self.adopt_row_from_on(src, dst_offsets, src_offsets, dst_row, src_row, ())
    }

    pub fn adopt_row_from_on(
        &mut self,
        src: &PagedKVCache,
        dst_offsets: &mut [i32],
        src_offsets: &[i32],
        dst_row: usize,
        src_row: usize,
        target: impl Into<StreamOrDevice>,
    ) -> Result<()> {
        let target = target.into();
        self.validate_same_layout(src)?;
        if dst_row >= self.batch as usize {
            anyhow::bail!(
                "PagedKVCache::adopt_row_from: dst_row {dst_row} >= batch {}",
                self.batch
            );
        }
        if src_row >= src.batch as usize {
            anyhow::bail!(
                "PagedKVCache::adopt_row_from: src_row {src_row} >= src batch {}",
                src.batch
            );
        }
        if dst_offsets.len() != self.batch as usize {
            anyhow::bail!(
                "PagedKVCache::adopt_row_from: dst_offsets.len()={} != batch {}",
                dst_offsets.len(),
                self.batch
            );
        }
        if src_offsets.len() != src.batch as usize {
            anyhow::bail!(
                "PagedKVCache::adopt_row_from: src_offsets.len()={} != src batch {}",
                src_offsets.len(),
                src.batch
            );
        }
        let src_off = src_offsets[src_row];
        if src_off < 0 || src_off > self.cap {
            anyhow::bail!(
                "PagedKVCache::adopt_row_from: src offset {src_off} outside destination cap {}",
                self.cap
            );
        }
        src.validate_offsets_allocated(src_offsets)?;

        self.release_row_pages(dst_row);
        if src_off == 0 {
            dst_offsets[dst_row] = 0;
            return Ok(());
        }

        let src_k_pages = src.k_pages.as_ref().ok_or_else(|| {
            anyhow::anyhow!("PagedKVCache::adopt_row_from: src K pages are unallocated")
        })?;
        let src_v_pages = src.v_pages.as_ref().ok_or_else(|| {
            anyhow::anyhow!("PagedKVCache::adopt_row_from: src V pages are unallocated")
        })?;
        let blocks = ceil_div(src_off, self.block_size);
        for block_col in 0..blocks {
            let src_page = src.block_tables[src_row][block_col as usize];
            if src_page < 0 {
                anyhow::bail!(
                    "PagedKVCache::adopt_row_from: missing src page row {src_row} block {block_col}"
                );
            }
            let dst_page = self.allocate_page(target)?;
            let k_page = slice_strided_on(
                src_k_pages,
                [src_page, 0, 0, 0],
                [
                    src_page + 1,
                    self.n_kv_heads,
                    self.block_size,
                    self.head_dim,
                ],
                [1_i32, 1, 1, 1],
                target,
            )?;
            let v_page = slice_strided_on(
                src_v_pages,
                [src_page, 0, 0, 0],
                [
                    src_page + 1,
                    self.n_kv_heads,
                    self.block_size,
                    self.v_head_dim,
                ],
                [1_i32, 1, 1, 1],
                target,
            )?;
            let k_pages = self
                .k_pages
                .as_ref()
                .expect("allocate_page created destination K storage");
            let v_pages = self
                .v_pages
                .as_ref()
                .expect("allocate_page created destination V storage");
            self.k_pages = Some(slice_update_on(
                k_pages,
                &k_page,
                [dst_page, 0, 0, 0],
                [
                    dst_page + 1,
                    self.n_kv_heads,
                    self.block_size,
                    self.head_dim,
                ],
                [1_i32, 1, 1, 1],
                target,
            )?);
            self.v_pages = Some(slice_update_on(
                v_pages,
                &v_page,
                [dst_page, 0, 0, 0],
                [
                    dst_page + 1,
                    self.n_kv_heads,
                    self.block_size,
                    self.v_head_dim,
                ],
                [1_i32, 1, 1, 1],
                target,
            )?);
            self.block_tables[dst_row][block_col as usize] = dst_page;
        }
        dst_offsets[dst_row] = src_off;
        Ok(())
    }

    fn append_on(
        &mut self,
        k: &Array,
        v: &Array,
        offsets: &mut [i32],
        per_row_lens: &[i32],
        target: StreamOrDevice,
    ) -> Result<()> {
        self.validate_update_inputs(k, v, offsets, per_row_lens)?;
        let k_seq = k.shape().as_slice()[2];
        for (row_usize, &n) in per_row_lens.iter().enumerate() {
            if n == 0 {
                continue;
            }
            let row = row_usize as i32;
            let mut logical_pos = offsets[row_usize];
            let mut src_pos = 0_i32;
            while src_pos < n {
                let block_col = logical_pos / self.block_size;
                let in_block = logical_pos % self.block_size;
                let take = (n - src_pos).min(self.block_size - in_block);
                if block_col >= self.max_blocks_per_row {
                    anyhow::bail!(
                        "PagedKVCache::append_on: row {row_usize} block {block_col} exceeds table width {}",
                        self.max_blocks_per_row
                    );
                }
                if self.block_tables[row_usize][block_col as usize] < 0 {
                    let page = self.allocate_page(target)?;
                    self.block_tables[row_usize][block_col as usize] = page;
                }
                let mut page = self.block_tables[row_usize][block_col as usize];
                if self.page_ref_count(page) > 1 {
                    let private_page = self.clone_page_for_write_on(page, target)?;
                    self.release_page_ref(page);
                    page = private_page;
                    self.block_tables[row_usize][block_col as usize] = private_page;
                }
                let k_part = slice_strided_on(
                    k,
                    [row, 0, src_pos, 0],
                    [row + 1, self.n_kv_heads, src_pos + take, self.head_dim],
                    [1_i32, 1, 1, 1],
                    target,
                )?;
                let v_part = slice_strided_on(
                    v,
                    [row, 0, src_pos, 0],
                    [row + 1, self.n_kv_heads, src_pos + take, self.v_head_dim],
                    [1_i32, 1, 1, 1],
                    target,
                )?;
                let k_pages = self
                    .k_pages
                    .as_ref()
                    .expect("allocate_page created K storage");
                let v_pages = self
                    .v_pages
                    .as_ref()
                    .expect("allocate_page created V storage");
                self.k_pages = Some(slice_update_on(
                    k_pages,
                    &k_part,
                    [page, 0, in_block, 0],
                    [page + 1, self.n_kv_heads, in_block + take, self.head_dim],
                    [1_i32, 1, 1, 1],
                    target,
                )?);
                self.v_pages = Some(slice_update_on(
                    v_pages,
                    &v_part,
                    [page, 0, in_block, 0],
                    [page + 1, self.n_kv_heads, in_block + take, self.v_head_dim],
                    [1_i32, 1, 1, 1],
                    target,
                )?);
                logical_pos += take;
                src_pos += take;
            }
            debug_assert!(src_pos <= k_seq);
            offsets[row_usize] += n;
        }
        Ok(())
    }

    fn validate_update_inputs(
        &self,
        k: &Array,
        v: &Array,
        offsets: &[i32],
        per_row_lens: &[i32],
    ) -> Result<()> {
        self.validate_offsets_shape_and_cap(offsets)?;
        if per_row_lens.len() != self.batch as usize {
            anyhow::bail!(
                "PagedKVCache::append_on: per_row_lens.len()={} != batch {}",
                per_row_lens.len(),
                self.batch
            );
        }
        if k.dtype() != self.dtype || v.dtype() != self.dtype {
            anyhow::bail!(
                "PagedKVCache::append_on: dtype mismatch cache={:?} k={:?} v={:?}",
                self.dtype,
                k.dtype(),
                v.dtype()
            );
        }
        let k_shape = k.shape();
        let v_shape = v.shape();
        let k_dims = k_shape.as_slice();
        let v_dims = v_shape.as_slice();
        if k_dims.len() != 4 || v_dims.len() != 4 {
            anyhow::bail!(
                "PagedKVCache::append_on: expected rank-4 K/V, got k={} v={}",
                k_dims.len(),
                v_dims.len()
            );
        }
        if k_dims[0] != self.batch
            || k_dims[1] != self.n_kv_heads
            || k_dims[3] != self.head_dim
            || v_dims[0] != self.batch
            || v_dims[1] != self.n_kv_heads
            || v_dims[2] != k_dims[2]
            || v_dims[3] != self.v_head_dim
        {
            anyhow::bail!(
                "PagedKVCache::append_on: shape mismatch k={k_dims:?} v={v_dims:?} cache=[{}, {}, _, {}]",
                self.batch,
                self.n_kv_heads,
                self.head_dim
            );
        }
        let k_seq = k_dims[2];
        for (i, (&off, &n)) in offsets.iter().zip(per_row_lens).enumerate() {
            if n < 0 {
                anyhow::bail!("PagedKVCache::append_on: per_row_lens[{i}]={n} must be >= 0");
            }
            if n > k_seq {
                anyhow::bail!(
                    "PagedKVCache::append_on: per_row_lens[{i}]={n} > k.shape()[2]={k_seq}"
                );
            }
            if off + n > self.cap {
                anyhow::bail!(
                    "PagedKVCache::append_on: row {i} cap {} exceeded by offset {off} + new {n}",
                    self.cap
                );
            }
        }
        Ok(())
    }

    fn validate_decode_inputs(
        &self,
        queries: &Array,
        k: &Array,
        v: &Array,
        offsets: &[i32],
        per_row_lens: &[i32],
    ) -> Result<()> {
        self.validate_update_inputs(k, v, offsets, per_row_lens)?;
        if per_row_lens.iter().any(|&n| n != 1) {
            anyhow::bail!("PagedKVCache::decode: every row must append exactly one token");
        }
        let q_shape = queries.shape();
        let q_dims = q_shape.as_slice();
        if q_dims.len() != 4
            || q_dims[0] != self.batch
            || q_dims[2] != 1
            || q_dims[3] != self.head_dim
            || q_dims[1] % self.n_kv_heads != 0
        {
            anyhow::bail!(
                "PagedKVCache::decode: query shape mismatch q={q_dims:?} batch={} hkv={} d={}",
                self.batch,
                self.n_kv_heads,
                self.head_dim
            );
        }
        Ok(())
    }

    fn validate_offsets_shape_and_cap(&self, offsets: &[i32]) -> Result<()> {
        if offsets.len() != self.batch as usize {
            anyhow::bail!(
                "PagedKVCache: offsets.len()={} != batch {}",
                offsets.len(),
                self.batch
            );
        }
        for (i, &off) in offsets.iter().enumerate() {
            if off < 0 || off > self.cap {
                anyhow::bail!("PagedKVCache: offsets[{i}]={off} outside [0, {}]", self.cap);
            }
        }
        Ok(())
    }

    fn validate_offsets_allocated(&self, offsets: &[i32]) -> Result<()> {
        self.validate_offsets_shape_and_cap(offsets)?;
        for (row, &off) in offsets.iter().enumerate() {
            let blocks = ceil_div(off, self.block_size);
            if blocks > self.max_blocks_per_row {
                anyhow::bail!(
                    "PagedKVCache: row {row} needs {blocks} blocks, table has {}",
                    self.max_blocks_per_row
                );
            }
            for block_col in 0..blocks {
                let page = self.block_tables[row][block_col as usize];
                if page < 0 || page >= self.allocated_pages {
                    anyhow::bail!(
                        "PagedKVCache: missing page row {row} block {block_col} page {page}"
                    );
                }
                if self.page_ref_count(page) <= 0 {
                    anyhow::bail!(
                        "PagedKVCache: row {row} block {block_col} references released page {page}"
                    );
                }
            }
        }
        Ok(())
    }

    fn validate_same_layout(&self, src: &PagedKVCache) -> Result<()> {
        if self.n_kv_heads != src.n_kv_heads
            || self.head_dim != src.head_dim
            || self.v_head_dim != src.v_head_dim
            || self.dtype != src.dtype
            || self.block_size != src.block_size
        {
            anyhow::bail!(
                "PagedKVCache::adopt_row_from: layout mismatch dst=({},{},{},{:?},block={}) src=({},{},{},{:?},block={})",
                self.n_kv_heads,
                self.head_dim,
                self.v_head_dim,
                self.dtype,
                self.block_size,
                src.n_kv_heads,
                src.head_dim,
                src.v_head_dim,
                src.dtype,
                src.block_size
            );
        }
        Ok(())
    }

    fn validate_prefix_pages(&self, name: &str, pages: &Array, head_dim: i32) -> Result<()> {
        if pages.dtype() != self.dtype {
            anyhow::bail!(
                "PagedKVCache::restore_prefix_pages_for_row_on: {name} dtype {} != cache dtype {}",
                pages.dtype(),
                self.dtype
            );
        }
        let shape = pages.shape();
        let dims = shape.as_slice();
        if dims.len() != 4
            || dims[1] != self.n_kv_heads
            || dims[2] != self.block_size
            || dims[3] != head_dim
        {
            anyhow::bail!(
                "PagedKVCache::restore_prefix_pages_for_row_on: {name} pages shape {:?} incompatible with [pages,{},{},{}]",
                dims,
                self.n_kv_heads,
                self.block_size,
                head_dim
            );
        }
        Ok(())
    }

    fn block_table_array(&self, max_blocks: i32) -> Result<Array> {
        if max_blocks <= 0 || max_blocks > self.max_blocks_per_row {
            anyhow::bail!(
                "PagedKVCache::block_table_array: max_blocks {max_blocks} outside [1, {}]",
                self.max_blocks_per_row
            );
        }
        let mut flat = vec![-1_i32; (self.batch * max_blocks) as usize];
        for row in 0..self.batch as usize {
            for block in 0..max_blocks as usize {
                flat[row * max_blocks as usize + block] = self.block_tables[row][block];
            }
        }
        let arr: Array = (flat.as_slice(), &[self.batch, max_blocks][..]).try_into()?;
        Ok(arr)
    }

    fn has_live_pages(&self) -> bool {
        self.page_ref_counts
            .iter()
            .take(self.allocated_pages as usize)
            .any(|&count| count > 0)
    }

    fn page_ref_count(&self, page: i32) -> i32 {
        self.page_ref_counts
            .get(page as usize)
            .copied()
            .unwrap_or(0)
    }

    fn set_page_ref_count(&mut self, page: i32, count: i32) {
        debug_assert!(page >= 0);
        debug_assert!(count >= 0);
        let needed = page as usize + 1;
        if self.page_ref_counts.len() < needed {
            self.page_ref_counts.resize(needed, 0);
        }
        self.page_ref_counts[page as usize] = count;
    }

    fn release_page_ref(&mut self, page: i32) {
        debug_assert!(page >= 0);
        let Some(count) = self.page_ref_counts.get_mut(page as usize) else {
            debug_assert!(false, "released page {page} without refcount slot");
            return;
        };
        debug_assert!(*count > 0, "released page {page} with refcount {count}");
        if *count <= 0 {
            return;
        }
        *count -= 1;
        if *count == 0 {
            self.free_pages.push(page);
        }
    }

    fn allocate_page(&mut self, target: StreamOrDevice) -> Result<i32> {
        let page = if let Some(page) = self.free_pages.pop() {
            page
        } else {
            if self.allocated_pages >= self.max_pages {
                anyhow::bail!(
                    "PagedKVCache::allocate_page: max_pages {} exhausted",
                    self.max_pages
                );
            }
            let page = self.allocated_pages;
            self.allocated_pages += 1;
            page
        };
        self.ensure_page_capacity(page + 1, target)?;
        self.set_page_ref_count(page, 1);
        Ok(page)
    }

    fn ensure_page_capacity(&mut self, needed: i32, target: StreamOrDevice) -> Result<()> {
        if needed <= self.page_capacity {
            return Ok(());
        }
        if needed > self.max_pages {
            anyhow::bail!(
                "PagedKVCache::ensure_page_capacity: needed {needed} > max_pages {}",
                self.max_pages
            );
        }
        let target_capacity = round_up(needed, self.page_grow_step).min(self.max_pages);
        let new_k = self.grow_page_tensor(
            self.k_pages.as_ref(),
            target_capacity,
            self.head_dim,
            target,
        )?;
        let new_v = self.grow_page_tensor(
            self.v_pages.as_ref(),
            target_capacity,
            self.v_head_dim,
            target,
        )?;
        self.k_pages = Some(new_k);
        self.v_pages = Some(new_v);
        self.page_capacity = target_capacity;
        if self.page_ref_counts.len() < target_capacity as usize {
            self.page_ref_counts.resize(target_capacity as usize, 0);
        }
        Ok(())
    }

    fn clone_page_for_write_on(&mut self, src_page: i32, target: StreamOrDevice) -> Result<i32> {
        let dst_page = self.allocate_page(target)?;
        let k_pages = self.k_pages.as_ref().ok_or_else(|| {
            anyhow::anyhow!("PagedKVCache::clone_page_for_write_on: K pages are unallocated")
        })?;
        let v_pages = self.v_pages.as_ref().ok_or_else(|| {
            anyhow::anyhow!("PagedKVCache::clone_page_for_write_on: V pages are unallocated")
        })?;
        let k_page = slice_strided_on(
            k_pages,
            [src_page, 0, 0, 0],
            [
                src_page + 1,
                self.n_kv_heads,
                self.block_size,
                self.head_dim,
            ],
            [1_i32, 1, 1, 1],
            target,
        )?;
        let v_page = slice_strided_on(
            v_pages,
            [src_page, 0, 0, 0],
            [
                src_page + 1,
                self.n_kv_heads,
                self.block_size,
                self.v_head_dim,
            ],
            [1_i32, 1, 1, 1],
            target,
        )?;
        let k_pages = self
            .k_pages
            .as_ref()
            .expect("clone_page_for_write_on kept K storage allocated");
        let v_pages = self
            .v_pages
            .as_ref()
            .expect("clone_page_for_write_on kept V storage allocated");
        self.k_pages = Some(slice_update_on(
            k_pages,
            &k_page,
            [dst_page, 0, 0, 0],
            [
                dst_page + 1,
                self.n_kv_heads,
                self.block_size,
                self.head_dim,
            ],
            [1_i32, 1, 1, 1],
            target,
        )?);
        self.v_pages = Some(slice_update_on(
            v_pages,
            &v_page,
            [dst_page, 0, 0, 0],
            [
                dst_page + 1,
                self.n_kv_heads,
                self.block_size,
                self.v_head_dim,
            ],
            [1_i32, 1, 1, 1],
            target,
        )?);
        Ok(dst_page)
    }

    fn grow_page_tensor(
        &self,
        old: Option<&Array>,
        new_capacity: i32,
        dim: i32,
        target: StreamOrDevice,
    ) -> Result<Array> {
        match (old, self.allocated_pages.min(self.page_capacity)) {
            (None, _) | (Some(_), 0) => Ok(Array::zeros_on(
                (new_capacity, self.n_kv_heads, self.block_size, dim),
                self.dtype,
                target,
            )?),
            (Some(old), keep_pages) => {
                let kept = slice_strided_on(
                    old,
                    [0_i32, 0, 0, 0],
                    [keep_pages, self.n_kv_heads, self.block_size, dim],
                    [1_i32, 1, 1, 1],
                    target,
                )?;
                let tail = Array::zeros_on(
                    (
                        new_capacity - keep_pages,
                        self.n_kv_heads,
                        self.block_size,
                        dim,
                    ),
                    self.dtype,
                    target,
                )?;
                Ok(concatenate_on(&[&kept, &tail], 0, target)?)
            }
        }
    }

    fn release_row_pages(&mut self, row: usize) {
        self.release_row_pages_from(row, 0);
    }

    fn release_row_pages_from(&mut self, row: usize, start_block: i32) {
        for block in (start_block as usize..self.block_tables[row].len()).rev() {
            let page = self.block_tables[row][block];
            if page >= 0 {
                self.release_page_ref(page);
                self.block_tables[row][block] = -1;
            }
        }
    }
}

fn ceil_div(n: i32, d: i32) -> i32 {
    if n <= 0 {
        0
    } else {
        (n + d - 1) / d
    }
}

fn round_up(n: i32, step: i32) -> i32 {
    if n <= 0 {
        0
    } else {
        ((n + step - 1) / step) * step
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::generate::build_per_row_decode_mask;
    use mlx::{Array, Dtype};

    fn assert_close(actual: &[f32], expected: &[f32], tol: f32) {
        assert_eq!(actual.len(), expected.len());
        for (idx, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() <= tol,
                "idx={idx} actual={a} expected={e} diff={}",
                (a - e).abs()
            );
        }
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn paged_kv_append_materializes_across_page_boundaries() {
        let mut cache =
            PagedKVCache::new(1, 1, 2, 2, Dtype::Float32, 8, 2, 4).expect("paged cache");
        let k_data: Vec<f32> = (0..10).map(|i| i as f32 + 1.0).collect();
        let v_data: Vec<f32> = k_data.iter().map(|x| x * 10.0).collect();
        let k: Array = (k_data.as_slice(), (1_i32, 1_i32, 5_i32, 2_i32))
            .try_into()
            .unwrap();
        let v: Array = (v_data.as_slice(), (1_i32, 1_i32, 5_i32, 2_i32))
            .try_into()
            .unwrap();
        let mut offsets = vec![0_i32];

        let (k_read, v_read) = cache
            .update_and_fetch_on(&k, &v, &mut offsets, &[5], ())
            .expect("append");

        assert_eq!(offsets, vec![5]);
        assert_eq!(cache.allocated_pages(), 3);
        assert_eq!(cache.block_table_row(0), &[0, 1, 2, -1]);
        assert_eq!(k_read.shape().as_slice(), &[1, 1, 5, 2]);
        assert_eq!(v_read.shape().as_slice(), &[1, 1, 5, 2]);
        assert_eq!(k_read.to_vec::<f32>().unwrap(), k_data);
        assert_eq!(v_read.to_vec::<f32>().unwrap(), v_data);
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn paged_kv_adopt_row_copies_prefix_into_new_pages() {
        let mut src = PagedKVCache::new(1, 1, 2, 2, Dtype::Float32, 8, 2, 4).expect("src cache");
        let k_data = vec![7.0_f32; 6];
        let v_data = vec![70.0_f32; 6];
        let k: Array = (k_data.as_slice(), (1_i32, 1_i32, 3_i32, 2_i32))
            .try_into()
            .unwrap();
        let v: Array = (v_data.as_slice(), (1_i32, 1_i32, 3_i32, 2_i32))
            .try_into()
            .unwrap();
        let mut src_offsets = vec![0_i32];
        src.update_and_fetch_on(&k, &v, &mut src_offsets, &[3], ())
            .expect("src append");

        let mut dst = PagedKVCache::new(2, 1, 2, 2, Dtype::Float32, 8, 2, 8).expect("dst cache");
        let mut dst_offsets = vec![0_i32, 0_i32];
        dst.adopt_row_from(&src, &mut dst_offsets, &src_offsets, 1, 0)
            .expect("adopt row");

        assert_eq!(dst_offsets, vec![0, 3]);
        let (k_read, v_read) = dst
            .materialize_prefix_on(&dst_offsets, 3, ())
            .expect("read");
        let k_vec = k_read.to_vec::<f32>().unwrap();
        let v_vec = v_read.to_vec::<f32>().unwrap();
        assert_eq!(k_vec[0], 0.0);
        assert_eq!(v_vec[0], 0.0);
        assert_eq!(k_vec[6], 7.0);
        assert_eq!(v_vec[6], 70.0);
        assert_eq!(k_vec[11], 7.0);
        assert_eq!(v_vec[11], 70.0);
    }

    #[test]
    fn paged_kv_restore_groups_contiguous_destination_pages() {
        let runs = contiguous_page_runs(&[7, 8, 9, 3, 4, 11]);
        assert_eq!(
            runs,
            vec![
                PageRun {
                    src_start: 0,
                    dst_start: 7,
                    len: 3,
                },
                PageRun {
                    src_start: 3,
                    dst_start: 3,
                    len: 2,
                },
                PageRun {
                    src_start: 5,
                    dst_start: 11,
                    len: 1,
                },
            ]
        );
    }

    #[test]
    fn paged_kv_restore_prefix_pages_for_rows_restores_same_prefix_to_multiple_rows() {
        let mut paged =
            PagedKVCache::new(3, 1, 2, 2, Dtype::Float32, 8, 2, 8).expect("paged cache");
        let k_src: Array = (
            &[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0][..],
            (2_i32, 1_i32, 2_i32, 2_i32),
        )
            .try_into()
            .unwrap();
        let v_src: Array = (
            &[10.0_f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0][..],
            (2_i32, 1_i32, 2_i32, 2_i32),
        )
            .try_into()
            .unwrap();
        let mut offsets = vec![0_i32, 0_i32, 0_i32];

        paged
            .restore_prefix_pages_for_rows_on(&k_src, &v_src, &mut offsets, &[0, 2], 3, ())
            .expect("restore rows");

        assert_eq!(offsets, vec![3, 0, 3]);
        assert_eq!(paged.allocated_pages(), 3);
        assert_eq!(paged.block_table_row(0), &[0, 1, -1, -1]);
        assert_eq!(paged.block_table_row(2), &[0, 2, -1, -1]);
        let (k_read, v_read) = paged
            .materialize_prefix_on(&offsets, 3, ())
            .expect("read restored rows");
        let k_vec = k_read.to_vec::<f32>().unwrap();
        let v_vec = v_read.to_vec::<f32>().unwrap();
        assert_eq!(&k_vec[0..6], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(&v_vec[0..6], &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
        assert_eq!(&k_vec[6..12], &[0.0; 6]);
        assert_eq!(&v_vec[6..12], &[0.0; 6]);
        assert_eq!(&k_vec[12..18], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(&v_vec[12..18], &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);

        let k_new: Array = (
            &[101.0_f32, 102.0, 0.0, 0.0, 201.0, 202.0][..],
            (3_i32, 1_i32, 1_i32, 2_i32),
        )
            .try_into()
            .unwrap();
        let v_new: Array = (
            &[1001.0_f32, 1002.0, 0.0, 0.0, 2001.0, 2002.0][..],
            (3_i32, 1_i32, 1_i32, 2_i32),
        )
            .try_into()
            .unwrap();
        paged
            .update_and_fetch_on(&k_new, &v_new, &mut offsets, &[1, 0, 1], ())
            .expect("append distinct row tails");

        assert_eq!(offsets, vec![4, 0, 4]);
        let (k_after, v_after) = paged
            .materialize_prefix_on(&offsets, 4, ())
            .expect("read appended rows");
        let k_after = k_after.to_vec::<f32>().unwrap();
        let v_after = v_after.to_vec::<f32>().unwrap();
        assert_eq!(
            &k_after[0..8],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 101.0, 102.0]
        );
        assert_eq!(
            &v_after[0..8],
            &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 1001.0, 1002.0]
        );
        assert_eq!(
            &k_after[16..24],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 201.0, 202.0]
        );
        assert_eq!(
            &v_after[16..24],
            &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 2001.0, 2002.0]
        );
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn paged_kv_decode_attention_matches_dense_sdpa_for_ragged_batch() {
        let mut paged =
            PagedKVCache::new(2, 1, 4, 4, Dtype::Float32, 8, 2, 8).expect("paged cache");
        let prefix_k_data: Vec<f32> = (0..(2 * 1 * 4 * 4))
            .map(|i| ((i % 19) as f32 - 9.0) * 0.03)
            .collect();
        let prefix_v_data: Vec<f32> = (0..(2 * 1 * 4 * 4))
            .map(|i| ((i % 23) as f32 - 11.0) * 0.025)
            .collect();
        let prefix_k: Array = (prefix_k_data.as_slice(), (2_i32, 1_i32, 4_i32, 4_i32))
            .try_into()
            .unwrap();
        let prefix_v: Array = (prefix_v_data.as_slice(), (2_i32, 1_i32, 4_i32, 4_i32))
            .try_into()
            .unwrap();
        let mut offsets = vec![0_i32, 0_i32];
        paged
            .update_and_fetch_on(&prefix_k, &prefix_v, &mut offsets, &[4, 2], ())
            .expect("prefix append");

        let q_data: Vec<f32> = (0..(2 * 2 * 4))
            .map(|i| ((i % 17) as f32 - 8.0) * 0.02)
            .collect();
        let step_k_data: Vec<f32> = (0..(2 * 1 * 1 * 4))
            .map(|i| ((i % 13) as f32 - 6.0) * 0.04)
            .collect();
        let step_v_data: Vec<f32> = (0..(2 * 1 * 1 * 4))
            .map(|i| ((i % 11) as f32 - 5.0) * 0.05)
            .collect();
        let q: Array = (q_data.as_slice(), (2_i32, 2_i32, 1_i32, 4_i32))
            .try_into()
            .unwrap();
        let step_k: Array = (step_k_data.as_slice(), (2_i32, 1_i32, 1_i32, 4_i32))
            .try_into()
            .unwrap();
        let step_v: Array = (step_v_data.as_slice(), (2_i32, 1_i32, 1_i32, 4_i32))
            .try_into()
            .unwrap();
        let scale = 0.5_f32;

        let actual = paged
            .update_and_attend_decode_on(&q, &step_k, &step_v, &mut offsets, &[1, 1], scale, ())
            .expect("paged decode");
        assert_eq!(offsets, vec![5, 3]);

        let (k_ref, v_ref) = paged.materialize_prefix_on(&offsets, 5, ()).expect("ref");
        let mask = build_per_row_decode_mask(&offsets, 5, Dtype::Float32).expect("mask");
        let expected = mlx::fast::scaled_dot_product_attention(
            &q,
            &k_ref,
            &v_ref,
            scale,
            "",
            Some(&mask),
            None,
        )
        .expect("dense sdpa");

        assert_eq!(actual.shape().as_slice(), &[2, 2, 1, 4]);
        assert_close(
            &actual.to_vec::<f32>().unwrap(),
            &expected.to_vec::<f32>().unwrap(),
            1.0e-4,
        );
    }
}
