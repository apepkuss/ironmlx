use std::collections::{HashMap, HashSet, VecDeque};
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::Context;
use mlx::ops::indexing::{slice_strided_on, slice_update_on};
use mlx::ops::shape::concatenate_on;
use mlx::{Array, Dtype, StreamOrDevice};

use crate::Result;

const STREAMING_PREFILL_QUERY_CHUNK_TOKENS: i32 = 128;

/// Fixed-size paged K/V storage for full-attention decode.
///
/// Physical storage is `[page, Hkv, block_size, D]`. Stable owners retain
/// logical block tables across compact execution-row rebuilds; batch rows are
/// temporary views of those tables. Multi-token prefill appends into pages and
/// then materializes the dense prefix for the existing SDPA path; single-token
/// decode appends and dispatches the paged attention kernel directly.
pub struct PagedKVCache {
    k_pages: Option<Array>,
    v_pages: Option<Array>,
    block_tables: Vec<Vec<i32>>,
    execution_owners: Vec<PagedKvBlockOwner>,
    owned_block_tables: HashMap<PagedKvBlockOwner, PagedKvOwnedBlockTable>,
    next_transient_owner: u64,
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
    hot_cold: Option<PagedKvHotColdTiering>,
    observability: PagedKvObservability,
}

/// Stable owner of a paged K/V block table.
///
/// Request owners survive compact execution-row rebuilds. Transient owners are
/// local to standalone or temporary caches and never cross a physical pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PagedKvBlockOwner {
    Request(u64),
    Transient(u64),
}

#[derive(Debug, Clone)]
struct PagedKvOwnedBlockTable {
    blocks: Vec<i32>,
    offset: i32,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PagedKvPhysicalStats {
    pub physical_pages_total: u64,
    pub physical_pages_free: u64,
    pub physical_pages_referenced: u64,
    pub shared_physical_pages: u64,
    pub shared_page_references: u64,
    pub max_page_refcount: u64,
    pub request_owned_tables: u64,
    pub transient_owned_tables: u64,
    pub orphan_pages: u64,
    pub cow_page_copies: u64,
    pub adopt_page_copies: u64,
    pub owner_releases: u64,
}

#[derive(Debug, Clone, Copy, Default)]
struct PagedKvObservability {
    cow_page_copies: u64,
    adopt_page_copies: u64,
    owner_releases: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PageRun {
    src_start: i32,
    dst_start: i32,
    len: i32,
}

#[derive(Debug, Clone, Copy)]
struct StreamingDecodeRowCtx {
    row: usize,
    row_len: i32,
    scale: f32,
    q_per_kv: i32,
    output_dtype: Dtype,
    target: StreamOrDevice,
}

#[derive(Debug, Clone, Copy)]
struct StreamingPrefillRowCtx {
    row: usize,
    old_len: i32,
    row_len: i32,
    query_len: i32,
    kv_len: i32,
    scale: f32,
    q_per_kv: i32,
    output_dtype: Dtype,
    target: StreamOrDevice,
}

#[derive(Debug, Clone, Copy)]
struct PrefillChunkRange {
    start: i32,
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
        let execution_owners = (0..batch)
            .map(|row| PagedKvBlockOwner::Transient(row as u64))
            .collect::<Vec<_>>();
        let owned_block_tables = execution_owners
            .iter()
            .copied()
            .map(|owner| {
                (
                    owner,
                    PagedKvOwnedBlockTable {
                        blocks: vec![-1; max_blocks_per_row as usize],
                        offset: 0,
                    },
                )
            })
            .collect();
        Ok(Self {
            k_pages: None,
            v_pages: None,
            block_tables,
            execution_owners,
            owned_block_tables,
            next_transient_owner: batch as u64,
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
            hot_cold: None,
            observability: PagedKvObservability::default(),
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

    pub fn execution_owners(&self) -> &[PagedKvBlockOwner] {
        &self.execution_owners
    }

    pub(super) fn validate_storage_reuse_from(
        &self,
        src: &PagedKVCache,
        owners: &[PagedKvBlockOwner],
    ) -> Result<()> {
        anyhow::ensure!(
            !owners.is_empty(),
            "PagedKVCache::validate_storage_reuse_from: owners cannot be empty"
        );
        anyhow::ensure!(
            owners.iter().copied().collect::<HashSet<_>>().len() == owners.len(),
            "PagedKVCache::validate_storage_reuse_from: duplicate owner"
        );
        anyhow::ensure!(
            self.n_kv_heads == src.n_kv_heads
                && self.head_dim == src.head_dim
                && self.v_head_dim == src.v_head_dim
                && self.dtype == src.dtype
                && self.block_size == src.block_size
                && self.max_pages == src.max_pages
                && self.hot_cold.is_some() == src.hot_cold.is_some(),
            "PagedKVCache::validate_storage_reuse_from: physical layout mismatch"
        );
        src.validate_owner_invariants()
    }

    pub fn physical_stats(&self) -> PagedKvPhysicalStats {
        let mut owner_refs = vec![0_u64; self.allocated_pages as usize];
        let mut request_owned_tables = 0_u64;
        let mut transient_owned_tables = 0_u64;
        for (owner, table) in &self.owned_block_tables {
            match owner {
                PagedKvBlockOwner::Request(_) => request_owned_tables += 1,
                PagedKvBlockOwner::Transient(_) => transient_owned_tables += 1,
            }
            for &page in &table.blocks {
                if let Some(count) = usize::try_from(page)
                    .ok()
                    .and_then(|page| owner_refs.get_mut(page))
                {
                    *count += 1;
                }
            }
        }

        let mut stats = PagedKvPhysicalStats {
            physical_pages_total: self.allocated_pages.max(0) as u64,
            request_owned_tables,
            transient_owned_tables,
            cow_page_copies: self.observability.cow_page_copies,
            adopt_page_copies: self.observability.adopt_page_copies,
            owner_releases: self.observability.owner_releases,
            ..PagedKvPhysicalStats::default()
        };
        for (page, &ref_count) in self
            .page_ref_counts
            .iter()
            .take(self.allocated_pages as usize)
            .enumerate()
        {
            if ref_count <= 0 {
                stats.physical_pages_free += 1;
                continue;
            }
            stats.physical_pages_referenced += 1;
            stats.max_page_refcount = stats.max_page_refcount.max(ref_count as u64);
            if ref_count > 1 {
                stats.shared_physical_pages += 1;
                stats.shared_page_references += (ref_count - 1) as u64;
            }
            if owner_refs.get(page).copied().unwrap_or(0) == 0 {
                stats.orphan_pages += 1;
            }
        }
        stats
    }

    pub fn validate_owner_invariants(&self) -> Result<()> {
        anyhow::ensure!(
            self.execution_owners.len() == self.batch as usize,
            "PagedKVCache owner invariant: execution owner count {} != batch {}",
            self.execution_owners.len(),
            self.batch
        );
        anyhow::ensure!(
            self.block_tables.len() == self.batch as usize,
            "PagedKVCache owner invariant: execution table count {} != batch {}",
            self.block_tables.len(),
            self.batch
        );

        let mut counted_refs = vec![0_i32; self.allocated_pages as usize];
        for (owner, table) in &self.owned_block_tables {
            anyhow::ensure!(
                table.blocks.len() == self.max_blocks_per_row as usize,
                "PagedKVCache owner invariant: owner {owner:?} table width {} != {}",
                table.blocks.len(),
                self.max_blocks_per_row
            );
            anyhow::ensure!(
                table.offset >= 0 && table.offset <= self.cap,
                "PagedKVCache owner invariant: owner {owner:?} offset {} outside [0, {}]",
                table.offset,
                self.cap
            );
            for &page in &table.blocks {
                if page < 0 {
                    continue;
                }
                let page = usize::try_from(page).expect("non-negative page fits usize");
                anyhow::ensure!(
                    page < counted_refs.len(),
                    "PagedKVCache owner invariant: owner {owner:?} references page {page} >= allocated pages {}",
                    self.allocated_pages
                );
                counted_refs[page] += 1;
            }
        }
        for (page, (&counted, &stored)) in counted_refs
            .iter()
            .zip(self.page_ref_counts.iter())
            .enumerate()
        {
            anyhow::ensure!(
                counted == stored,
                "PagedKVCache owner invariant: page {page} counted refs {counted} != stored refs {stored}"
            );
        }
        for (row, &owner) in self.execution_owners.iter().enumerate() {
            let table = self.owned_block_tables.get(&owner).ok_or_else(|| {
                anyhow::anyhow!(
                    "PagedKVCache owner invariant: execution row {row} owner {owner:?} absent"
                )
            })?;
            anyhow::ensure!(
                self.block_tables[row] == table.blocks,
                "PagedKVCache owner invariant: execution row {row} diverged from owner {owner:?}"
            );
        }
        Ok(())
    }

    pub fn k_pages(&self) -> Option<&Array> {
        self.k_pages.as_ref()
    }

    pub fn v_pages(&self) -> Option<&Array> {
        self.v_pages.as_ref()
    }

    pub fn enable_hot_cold_tiering(&mut self, config: PagedKvHotColdConfig) -> Result<()> {
        anyhow::ensure!(
            self.allocated_pages == 0 && !self.has_live_pages(),
            "PagedKVCache::enable_hot_cold_tiering: cache must be empty"
        );
        self.hot_cold = Some(PagedKvHotColdTiering::new(config)?);
        self.k_pages = None;
        self.v_pages = None;
        self.page_capacity = 0;
        Ok(())
    }

    pub fn hot_cold_summary(&self) -> Option<PagedKvHotColdSummary> {
        self.hot_cold.as_ref().map(PagedKvHotColdTiering::summary)
    }

    pub fn shrink_hot_window_on(
        &mut self,
        offsets: &[i32],
        hot_window_pages: i32,
        target: impl Into<StreamOrDevice>,
    ) -> Result<usize> {
        anyhow::ensure!(
            hot_window_pages > 0,
            "hot window must retain at least one page"
        );
        let Some(tiering) = self.hot_cold.as_mut() else {
            return Ok(0);
        };
        tiering.hot_window_pages = tiering.hot_window_pages.min(hot_window_pages);
        let before = tiering
            .slot_to_logical
            .iter()
            .filter(|page| page.is_some())
            .count();
        self.enforce_hot_window(offsets, target.into())?;
        let after = self.hot_cold.as_ref().map_or(0, |tiering| {
            tiering
                .slot_to_logical
                .iter()
                .filter(|page| page.is_some())
                .count()
        });
        Ok(before.saturating_sub(after))
    }

    pub fn restore_configured_hot_window(&mut self) -> bool {
        let Some(tiering) = self.hot_cold.as_mut() else {
            return false;
        };
        let changed = tiering.hot_window_pages != tiering.configured_hot_window_pages;
        tiering.hot_window_pages = tiering.configured_hot_window_pages;
        changed
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
            for table in self.owned_block_tables.values_mut() {
                table.blocks.resize(new_blocks as usize, -1);
            }
            self.max_blocks_per_row = new_blocks;
        }
    }

    /// Rebind compact execution rows to stable owners without copying K/V pages.
    pub fn bind_execution_rows(
        &mut self,
        owners: &[PagedKvBlockOwner],
        offsets: &mut Vec<i32>,
    ) -> Result<()> {
        anyhow::ensure!(
            !owners.is_empty(),
            "PagedKVCache::bind_execution_rows: owners cannot be empty"
        );
        let unique = owners.iter().copied().collect::<HashSet<_>>();
        anyhow::ensure!(
            unique.len() == owners.len(),
            "PagedKVCache::bind_execution_rows: duplicate owner"
        );
        self.commit_execution_rows(offsets)?;

        self.batch = i32::try_from(owners.len())
            .map_err(|_| anyhow::anyhow!("PagedKVCache::bind_execution_rows: batch overflow"))?;
        self.execution_owners.clear();
        self.execution_owners.extend_from_slice(owners);
        self.block_tables.clear();
        offsets.clear();
        for &owner in owners {
            let table =
                self.owned_block_tables
                    .entry(owner)
                    .or_insert_with(|| PagedKvOwnedBlockTable {
                        blocks: vec![-1; self.max_blocks_per_row as usize],
                        offset: 0,
                    });
            self.block_tables.push(table.blocks.clone());
            offsets.push(table.offset);
        }
        self.remove_unbound_empty_transient_owners();
        self.validate_owner_invariants()
    }

    /// Release a stable owner's physical page references. Repeated release is
    /// a no-op so finish, cancel, retry, and admission rollback share one path.
    pub fn release_owner(&mut self, owner: PagedKvBlockOwner, offsets: &mut [i32]) -> Result<bool> {
        anyhow::ensure!(
            offsets.len() == self.batch as usize,
            "PagedKVCache::release_owner: offsets.len()={} != batch {}",
            offsets.len(),
            self.batch
        );
        self.commit_execution_rows(offsets)?;
        let Some(table) = self.owned_block_tables.remove(&owner) else {
            return Ok(false);
        };
        for page in table.blocks.into_iter().rev().filter(|&page| page >= 0) {
            self.release_page_ref(page);
        }
        for (row, execution_owner) in self.execution_owners.iter().enumerate() {
            if *execution_owner == owner {
                self.block_tables[row].fill(-1);
                offsets[row] = 0;
            }
        }
        let released_rows = self
            .execution_owners
            .iter()
            .enumerate()
            .filter_map(|(row, &execution_owner)| (execution_owner == owner).then_some(row))
            .collect::<Vec<_>>();
        for row in released_rows {
            let transient = PagedKvBlockOwner::Transient(self.next_transient_owner);
            self.next_transient_owner = self.next_transient_owner.saturating_add(1);
            self.execution_owners[row] = transient;
            self.owned_block_tables.insert(
                transient,
                PagedKvOwnedBlockTable {
                    blocks: vec![-1; self.max_blocks_per_row as usize],
                    offset: 0,
                },
            );
        }
        self.observability.owner_releases = self.observability.owner_releases.saturating_add(1);
        self.validate_owner_invariants()?;
        Ok(true)
    }

    pub(super) fn commit_execution_rows(&mut self, offsets: &[i32]) -> Result<()> {
        anyhow::ensure!(
            offsets.len() == self.batch as usize,
            "PagedKVCache::commit_execution_rows: offsets.len()={} != batch {}",
            offsets.len(),
            self.batch
        );
        for (row, (&owner, &offset)) in self.execution_owners.iter().zip(offsets.iter()).enumerate()
        {
            let table = self
                .owned_block_tables
                .get_mut(&owner)
                .ok_or_else(|| anyhow::anyhow!("execution owner {owner:?} absent"))?;
            table.blocks.clone_from(&self.block_tables[row]);
            table.offset = offset;
        }
        Ok(())
    }

    fn remove_unbound_empty_transient_owners(&mut self) {
        let bound = self
            .execution_owners
            .iter()
            .copied()
            .collect::<HashSet<_>>();
        self.owned_block_tables.retain(|owner, table| {
            !matches!(owner, PagedKvBlockOwner::Transient(_))
                || bound.contains(owner)
                || table.offset != 0
                || table.blocks.iter().any(|&page| page >= 0)
        });
    }

    pub fn clear(&mut self) {
        for row in &mut self.block_tables {
            row.fill(-1);
        }
        self.free_pages.clear();
        self.page_ref_counts.fill(0);
        self.allocated_pages = 0;
        for table in self.owned_block_tables.values_mut() {
            table.blocks.fill(-1);
            table.offset = 0;
        }
        if let Some(hot_cold) = &mut self.hot_cold {
            let _ = fs::remove_dir_all(&hot_cold.cache_dir);
            let _ = fs::create_dir_all(&hot_cold.cache_dir);
            hot_cold.reset_runtime_state();
            self.k_pages = None;
            self.v_pages = None;
            self.page_capacity = 0;
        }
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
        self.commit_execution_rows(current_offsets)?;
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
    pub fn update_and_attend_prefill_on(
        &mut self,
        queries: &Array,
        k: &Array,
        v: &Array,
        offsets: &mut [i32],
        per_row_lens: &[i32],
        scale: f32,
        mask_arr: Option<&Array>,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        let target = target.into();
        self.validate_prefill_attention_inputs(queries, k, v, offsets, per_row_lens, mask_arr)?;
        let old_offsets = offsets.to_vec();
        self.append_on_inner(k, v, offsets, per_row_lens, target, false)?;
        let out = self.streaming_prefill_attention_on(
            queries,
            &old_offsets,
            offsets,
            scale,
            mask_arr,
            target,
        )?;
        self.enforce_hot_window_if_over_resident_budget(offsets, target)?;
        Ok(out)
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
        self.append_on_inner(k, v, offsets, per_row_lens, target, false)?;
        if self.needs_hot_cold_streaming(offsets) {
            if self.try_stage_hot_cold_context_for_paged_decode(offsets, target)? {
                let out = self.paged_decode_attention_on(queries, offsets, scale, target)?;
                self.enforce_hot_window_if_over_resident_budget(offsets, target)?;
                return Ok(out);
            }
            return self.streaming_decode_attention_on(queries, offsets, scale, target);
        }
        self.paged_decode_attention_on(queries, offsets, scale, target)
    }

    fn paged_decode_attention_on(
        &self,
        queries: &Array,
        offsets: &[i32],
        scale: f32,
        target: StreamOrDevice,
    ) -> Result<Array> {
        let max_blocks = offsets
            .iter()
            .map(|&off| ceil_div(off, self.block_size))
            .max()
            .unwrap_or(0)
            .max(1);
        let block_table = if self.hot_cold.is_some() {
            self.resident_block_table_array(max_blocks)?
        } else {
            self.block_table_array(max_blocks)?
        };
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

    fn streaming_prefill_attention_on(
        &mut self,
        queries: &Array,
        old_offsets: &[i32],
        offsets: &[i32],
        scale: f32,
        mask_arr: Option<&Array>,
        target: StreamOrDevice,
    ) -> Result<Array> {
        let q_shape = queries.shape();
        let q_dims = q_shape.as_slice();
        let q_heads = q_dims[1];
        let query_len = q_dims[2];
        let q_per_kv = q_heads / self.n_kv_heads;
        anyhow::ensure!(
            q_per_kv > 0 && q_heads % self.n_kv_heads == 0,
            "PagedKVCache::streaming_prefill_attention_on: invalid GQA layout"
        );
        let max_kv_len = match mask_arr {
            Some(mask) => mask.shape().as_slice()[3],
            None => offsets.iter().copied().max().unwrap_or(0),
        };

        let mut outputs = Vec::with_capacity(self.batch as usize);
        for row in 0..self.batch as usize {
            let q_row = slice_strided_on(
                queries,
                [row as i32, 0, 0, 0],
                [row as i32 + 1, q_heads, query_len, self.head_dim],
                [1_i32, 1, 1, 1],
                target,
            )?;
            let ctx = StreamingPrefillRowCtx {
                row,
                old_len: old_offsets[row],
                row_len: offsets[row],
                query_len,
                kv_len: max_kv_len,
                scale,
                q_per_kv,
                output_dtype: queries.dtype(),
                target,
            };
            outputs.push(self.streaming_prefill_row_on(&q_row, ctx, mask_arr)?);
        }
        let refs = outputs.iter().collect::<Vec<_>>();
        Ok(concatenate_on(&refs, 0, target)?)
    }

    fn streaming_prefill_row_on(
        &mut self,
        q_row: &Array,
        ctx: StreamingPrefillRowCtx,
        mask_arr: Option<&Array>,
    ) -> Result<Array> {
        struct QueryChunkState {
            start: i32,
            len: i32,
            q: Array,
            state: Option<(Array, Array, Array)>,
        }

        anyhow::ensure!(
            ctx.query_len > 0,
            "PagedKVCache::streaming_prefill_row_on: row {} has empty query",
            ctx.row
        );
        anyhow::ensure!(
            ctx.kv_len > 0,
            "PagedKVCache::streaming_prefill_row_on: row {} has empty KV",
            ctx.row
        );
        let target = ctx.target;
        let q_shape = q_row.shape();
        let q_heads = q_shape.as_slice()[1];
        let scale_arr: Array = (&[ctx.scale][..], ()).try_into()?;
        let neg_large: Array = (&[-1.0e30_f32][..], ()).try_into()?;
        let valid_floor: Array = (&[-1.0e20_f32][..], ()).try_into()?;
        let mut q_chunks = Vec::new();
        let mut q_start = 0_i32;
        while q_start < ctx.query_len {
            let q_take = (ctx.query_len - q_start).min(STREAMING_PREFILL_QUERY_CHUNK_TOKENS);
            let q = slice_strided_on(
                q_row,
                [0_i32, 0, q_start, 0],
                [1_i32, q_heads, q_start + q_take, self.head_dim],
                [1_i32, 1, 1, 1],
                target,
            )?
            .astype_on(Dtype::Float32, target)?;
            q_chunks.push(QueryChunkState {
                start: q_start,
                len: q_take,
                q,
                state: None,
            });
            q_start += q_take;
        }

        let mut block_col = 0_i32;
        let blocks = ceil_div(ctx.kv_len, self.block_size);
        let chunk_pages = self
            .hot_cold
            .as_ref()
            .map(|tiering| tiering.chunk_pages.max(1))
            .unwrap_or(1);
        while block_col < blocks {
            let count = (blocks - block_col).min(chunk_pages);
            let key_start = block_col * self.block_size;
            let (k_chunk, v_chunk) = self.row_chunk_with_virtual_zeros_on(
                ctx.row,
                ctx.row_len,
                ctx.kv_len,
                block_col,
                count,
                target,
            )?;
            let k_chunk = k_chunk.astype_on(Dtype::Float32, target)?;
            let v_chunk = v_chunk.astype_on(Dtype::Float32, target)?;
            let chunk_len = k_chunk.shape().as_slice()[2];
            let k_heads = if ctx.q_per_kv == 1 {
                k_chunk
            } else {
                k_chunk.repeat_on(ctx.q_per_kv, 1, target)?
            };
            let v_heads = if ctx.q_per_kv == 1 {
                v_chunk
            } else {
                v_chunk.repeat_on(ctx.q_per_kv, 1, target)?
            };
            let k_t = k_heads.transpose_axes_on([0_i32, 1, 3, 2], target)?;
            for q_chunk in &mut q_chunks {
                if mask_arr.is_none() && key_start > ctx.old_len + q_chunk.start + q_chunk.len - 1 {
                    continue;
                }
                let logits = q_chunk
                    .q
                    .matmul_on(&k_t, target)?
                    .try_mul_on(&scale_arr, target)?;
                let (masked_logits, valid_mask) = match mask_arr {
                    Some(mask) => {
                        let mask_chunk = self.prefill_mask_chunk_on(
                            mask,
                            ctx.row,
                            PrefillChunkRange {
                                start: q_chunk.start,
                                len: q_chunk.len,
                            },
                            PrefillChunkRange {
                                start: key_start,
                                len: chunk_len,
                            },
                            target,
                        )?;
                        let mask_chunk = mask_chunk.astype_on(Dtype::Float32, target)?;
                        let valid = mask_chunk.greater_on(&valid_floor, target)?;
                        let logits_with_mask = logits.try_add_on(&mask_chunk, target)?;
                        (
                            mlx::ops::indexing::where_on(
                                &valid,
                                &logits_with_mask,
                                &neg_large,
                                target,
                            )?,
                            valid,
                        )
                    }
                    None => {
                        let valid = standard_prefill_valid_mask_on(
                            ctx.old_len,
                            ctx.row_len,
                            q_chunk.start,
                            q_chunk.len,
                            key_start,
                            chunk_len,
                            target,
                        )?;
                        (
                            mlx::ops::indexing::where_on(&valid, &logits, &neg_large, target)?,
                            valid,
                        )
                    }
                };
                let chunk_max = masked_logits.max_on(-1_i32, true, target)?;
                let valid_float = valid_mask.astype_on(Dtype::Float32, target)?;
                let weights = masked_logits
                    .try_sub_on(&chunk_max, target)?
                    .exp_on(target)?
                    .try_mul_on(&valid_float, target)?;
                let chunk_den = weights.sum_on(-1_i32, true, target)?;
                let chunk_num = weights.matmul_on(&v_heads, target)?;

                q_chunk.state = Some(match q_chunk.state.take() {
                    None => (chunk_max, chunk_den, chunk_num),
                    Some((prev_max, prev_den, prev_num)) => {
                        let new_max = prev_max.maximum_on(&chunk_max, target)?;
                        let prev_scale = prev_max.try_sub_on(&new_max, target)?.exp_on(target)?;
                        let chunk_scale = chunk_max.try_sub_on(&new_max, target)?.exp_on(target)?;
                        let den = prev_den
                            .try_mul_on(&prev_scale, target)?
                            .try_add_on(&chunk_den.try_mul_on(&chunk_scale, target)?, target)?;
                        let num = prev_num
                            .try_mul_on(&prev_scale, target)?
                            .try_add_on(&chunk_num.try_mul_on(&chunk_scale, target)?, target)?;
                        (new_max, den, num)
                    }
                });
            }
            block_col += count;
        }

        let mut q_outputs = Vec::with_capacity(q_chunks.len());
        for mut q_chunk in q_chunks {
            let (_, den, num) = q_chunk.state.take().ok_or_else(|| {
                anyhow::anyhow!("PagedKVCache::streaming_prefill_row_on: no chunks")
            })?;
            q_outputs.push(
                num.try_div_on(&den, target)?
                    .astype_on(ctx.output_dtype, target)?,
            );
        }
        if q_outputs.len() == 1 {
            Ok(q_outputs.remove(0))
        } else {
            let refs = q_outputs.iter().collect::<Vec<_>>();
            Ok(concatenate_on(&refs, 2, target)?)
        }
    }

    fn streaming_decode_attention_on(
        &mut self,
        queries: &Array,
        offsets: &[i32],
        scale: f32,
        target: StreamOrDevice,
    ) -> Result<Array> {
        let q_shape = queries.shape();
        let q_dims = q_shape.as_slice();
        let q_heads = q_dims[1];
        let q_per_kv = q_heads / self.n_kv_heads;
        anyhow::ensure!(
            q_per_kv > 0 && q_heads % self.n_kv_heads == 0,
            "PagedKVCache::streaming_decode_attention_on: invalid GQA layout"
        );

        let mut outputs = Vec::with_capacity(self.batch as usize);
        for (row, &row_len) in offsets.iter().enumerate().take(self.batch as usize) {
            let q_row = slice_strided_on(
                queries,
                [row as i32, 0, 0, 0],
                [row as i32 + 1, q_heads, 1_i32, self.head_dim],
                [1_i32, 1, 1, 1],
                target,
            )?;
            let row_ctx = StreamingDecodeRowCtx {
                row,
                row_len,
                scale,
                q_per_kv,
                output_dtype: queries.dtype(),
                target,
            };
            outputs.push(self.streaming_decode_row_on(&q_row, row_ctx)?);
        }
        self.enforce_hot_window_if_over_resident_budget(offsets, target)?;
        let refs = outputs.iter().collect::<Vec<_>>();
        Ok(concatenate_on(&refs, 0, target)?)
    }

    fn streaming_decode_row_on(
        &mut self,
        q_row: &Array,
        ctx: StreamingDecodeRowCtx,
    ) -> Result<Array> {
        anyhow::ensure!(
            ctx.row_len > 0,
            "PagedKVCache::streaming_decode_row_on: row {} has empty KV",
            ctx.row
        );
        let target = ctx.target;
        let blocks = ceil_div(ctx.row_len, self.block_size);
        let chunk_pages = self
            .hot_cold
            .as_ref()
            .map(|tiering| tiering.chunk_pages.max(1))
            .unwrap_or(1);
        let q = q_row.astype_on(Dtype::Float32, target)?;
        let scale_arr: Array = (&[ctx.scale][..], ()).try_into()?;
        let mut state: Option<(Array, Array, Array)> = None;
        let mut block_col = 0_i32;
        while block_col < blocks {
            let count = (blocks - block_col).min(chunk_pages);
            let (k_chunk, v_chunk) =
                self.row_chunk_on(ctx.row, ctx.row_len, block_col, count, target)?;
            let k_chunk = k_chunk.astype_on(Dtype::Float32, target)?;
            let v_chunk = v_chunk.astype_on(Dtype::Float32, target)?;
            let k_heads = if ctx.q_per_kv == 1 {
                k_chunk
            } else {
                k_chunk.repeat_on(ctx.q_per_kv, 1, target)?
            };
            let v_heads = if ctx.q_per_kv == 1 {
                v_chunk
            } else {
                v_chunk.repeat_on(ctx.q_per_kv, 1, target)?
            };
            let k_t = k_heads.transpose_axes_on([0_i32, 1, 3, 2], target)?;
            let logits = q.matmul_on(&k_t, target)?.try_mul_on(&scale_arr, target)?;
            let chunk_max = logits.max_on(-1_i32, true, target)?;
            let weights = logits.try_sub_on(&chunk_max, target)?.exp_on(target)?;
            let chunk_den = weights.sum_on(-1_i32, true, target)?;
            let chunk_num = weights.matmul_on(&v_heads, target)?;

            state = Some(match state {
                None => (chunk_max, chunk_den, chunk_num),
                Some((prev_max, prev_den, prev_num)) => {
                    let new_max = prev_max.maximum_on(&chunk_max, target)?;
                    let prev_scale = prev_max.try_sub_on(&new_max, target)?.exp_on(target)?;
                    let chunk_scale = chunk_max.try_sub_on(&new_max, target)?.exp_on(target)?;
                    let den = prev_den
                        .try_mul_on(&prev_scale, target)?
                        .try_add_on(&chunk_den.try_mul_on(&chunk_scale, target)?, target)?;
                    let num = prev_num
                        .try_mul_on(&prev_scale, target)?
                        .try_add_on(&chunk_num.try_mul_on(&chunk_scale, target)?, target)?;
                    (new_max, den, num)
                }
            });
            block_col += count;
        }
        let (_, den, num) = state
            .ok_or_else(|| anyhow::anyhow!("PagedKVCache::streaming_decode_row_on: no chunks"))?;
        Ok(num
            .try_div_on(&den, target)?
            .astype_on(ctx.output_dtype, target)?)
    }

    fn row_chunk_on(
        &mut self,
        row: usize,
        row_len: i32,
        start_block: i32,
        block_count: i32,
        target: StreamOrDevice,
    ) -> Result<(Array, Array)> {
        let mut k_parts = Vec::with_capacity(block_count as usize);
        let mut v_parts = Vec::with_capacity(block_count as usize);
        for offset in 0..block_count {
            let block_col = start_block + offset;
            let logical_start = block_col * self.block_size;
            let take = (row_len - logical_start).min(self.block_size);
            let page = self.block_tables[row][block_col as usize];
            if page < 0 {
                anyhow::bail!(
                    "PagedKVCache::row_chunk_on: missing page row {row} block {block_col}"
                );
            }
            let (k_page, v_page) = self.page_slice_streaming_on(page, take, target)?;
            k_parts.push(k_page);
            v_parts.push(v_page);
        }
        if k_parts.len() == 1 {
            Ok((k_parts.remove(0), v_parts.remove(0)))
        } else {
            let k_refs = k_parts.iter().collect::<Vec<_>>();
            let v_refs = v_parts.iter().collect::<Vec<_>>();
            Ok((
                concatenate_on(&k_refs, 2, target)?,
                concatenate_on(&v_refs, 2, target)?,
            ))
        }
    }

    fn row_chunk_with_virtual_zeros_on(
        &mut self,
        row: usize,
        row_len: i32,
        kv_len: i32,
        start_block: i32,
        block_count: i32,
        target: StreamOrDevice,
    ) -> Result<(Array, Array)> {
        let mut k_parts = Vec::with_capacity(block_count as usize);
        let mut v_parts = Vec::with_capacity(block_count as usize);
        for offset in 0..block_count {
            let block_col = start_block + offset;
            let logical_start = block_col * self.block_size;
            let take = (kv_len - logical_start).min(self.block_size);
            if take <= 0 {
                continue;
            }
            let actual_take = if logical_start < row_len {
                (row_len - logical_start).min(take)
            } else {
                0
            };
            let mut k_piece: Option<Array> = None;
            let mut v_piece: Option<Array> = None;
            if actual_take > 0 {
                let page = self.block_tables[row][block_col as usize];
                if page < 0 {
                    anyhow::bail!(
                        "PagedKVCache::row_chunk_with_virtual_zeros_on: missing page row {row} block {block_col}"
                    );
                }
                let (k_page, v_page) = self.page_slice_streaming_on(page, actual_take, target)?;
                k_piece = Some(k_page);
                v_piece = Some(v_page);
            }
            if actual_take < take {
                let zero_k = Array::zeros_on(
                    (1_i32, self.n_kv_heads, take - actual_take, self.head_dim),
                    self.dtype,
                    target,
                )?;
                let zero_v = Array::zeros_on(
                    (1_i32, self.n_kv_heads, take - actual_take, self.v_head_dim),
                    self.dtype,
                    target,
                )?;
                match (k_piece.take(), v_piece.take()) {
                    (Some(k_actual), Some(v_actual)) => {
                        k_piece = Some(concatenate_on(&[&k_actual, &zero_k], 2, target)?);
                        v_piece = Some(concatenate_on(&[&v_actual, &zero_v], 2, target)?);
                    }
                    (None, None) => {
                        k_piece = Some(zero_k);
                        v_piece = Some(zero_v);
                    }
                    _ => unreachable!("K/V virtual zero pieces are constructed together"),
                }
            }
            k_parts.push(k_piece.expect("chunk K piece"));
            v_parts.push(v_piece.expect("chunk V piece"));
        }
        if k_parts.len() == 1 {
            Ok((k_parts.remove(0), v_parts.remove(0)))
        } else {
            let k_refs = k_parts.iter().collect::<Vec<_>>();
            let v_refs = v_parts.iter().collect::<Vec<_>>();
            Ok((
                concatenate_on(&k_refs, 2, target)?,
                concatenate_on(&v_refs, 2, target)?,
            ))
        }
    }

    fn prefill_mask_chunk_on(
        &self,
        mask: &Array,
        row: usize,
        query: PrefillChunkRange,
        key: PrefillChunkRange,
        target: StreamOrDevice,
    ) -> Result<Array> {
        let mask_shape = mask.shape();
        let mask_dims = mask_shape.as_slice();
        let mask_row = if mask_dims[0] == 1 { 0 } else { row as i32 };
        let mask_heads = mask_dims[1];
        Ok(slice_strided_on(
            mask,
            [mask_row, 0, query.start, key.start],
            [
                mask_row + 1,
                mask_heads,
                query.start + query.len,
                key.start + key.len,
            ],
            [1_i32, 1, 1, 1],
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
                let (k_page, v_page) = self.page_slice_on(page, take, target)?;
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
        let mut k_parts = Vec::with_capacity(blocks as usize);
        let mut v_parts = Vec::with_capacity(blocks as usize);
        for block_col in 0..blocks {
            let page = self.block_tables[row][block_col as usize];
            if page < 0 {
                anyhow::bail!(
                    "PagedKVCache::prefix_pages_for_row_on: missing page row {row} block {block_col}"
                );
            }
            let (k_page, v_page) = self.page_slice_on(page, self.block_size, target)?;
            k_parts.push(k_page);
            v_parts.push(v_page);
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
            self.commit_execution_rows(offsets)?;
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
        if self.can_direct_install_prefix_pages(total_pages) {
            self.install_prefix_pages_direct_on(
                k_pages_src,
                v_pages_src,
                rows,
                shared_blocks,
                tail_private,
                target,
            )?;
        } else if self.can_direct_install_hot_cold_prefix_pages(total_pages) {
            self.install_prefix_pages_hot_cold_direct_on(
                k_pages_src,
                v_pages_src,
                rows,
                shared_blocks,
                tail_private,
                target,
            )?;
        } else {
            self.copy_prefix_pages_once_for_rows_on(
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
        self.commit_execution_rows(offsets)?;
        self.enforce_hot_window_if_over_resident_budget(offsets, target)?;
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

    fn can_direct_install_prefix_pages(&self, total_pages: i32) -> bool {
        total_pages >= 0 && !self.has_live_pages() && total_pages <= self.resident_slot_limit()
    }

    fn can_direct_install_hot_cold_prefix_pages(&self, total_pages: i32) -> bool {
        self.hot_cold.is_some()
            && total_pages >= 0
            && !self.has_live_pages()
            && total_pages <= self.max_pages
            && total_pages > self.resident_slot_limit()
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
        self.install_hot_cold_direct_resident_pages(total_pages);
        Ok(())
    }

    fn install_prefix_pages_hot_cold_direct_on(
        &mut self,
        k_pages_src: &Array,
        v_pages_src: &Array,
        rows: &[usize],
        shared_blocks: i32,
        tail_private: bool,
        target: StreamOrDevice,
    ) -> Result<()> {
        anyhow::ensure!(
            self.hot_cold.is_some(),
            "PagedKVCache::install_prefix_pages_hot_cold_direct_on: hot/cold tiering is disabled"
        );
        let private_tail_pages = if tail_private { rows.len() as i32 } else { 0 };
        let total_pages = shared_blocks + private_tail_pages;
        let page_source = Self::prefix_page_source_map(shared_blocks, private_tail_pages);
        let resident_pages = self.hot_cold_direct_resident_pages(rows, shared_blocks, tail_private);

        self.free_pages.clear();
        self.allocated_pages = total_pages;
        self.page_capacity = resident_pages.len() as i32;
        self.page_ref_counts.clear();
        self.page_ref_counts.resize(total_pages as usize, 0);
        self.k_pages = Some(self.build_selected_prefix_page_tensor_on(
            k_pages_src,
            &page_source,
            &resident_pages,
            self.head_dim,
            target,
        )?);
        self.v_pages = Some(self.build_selected_prefix_page_tensor_on(
            v_pages_src,
            &page_source,
            &resident_pages,
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

        self.install_hot_cold_direct_resident_subset(total_pages, &resident_pages);
        self.save_hot_cold_direct_offloaded_prefix_runs_on(
            k_pages_src,
            v_pages_src,
            &page_source,
            &resident_pages,
            target,
        )?;
        Ok(())
    }

    fn prefix_page_source_map(shared_blocks: i32, private_tail_pages: i32) -> Vec<i32> {
        let total_pages = shared_blocks + private_tail_pages;
        let mut page_source = Vec::with_capacity(total_pages as usize);
        for page in 0..shared_blocks {
            page_source.push(page);
        }
        for _ in 0..private_tail_pages {
            page_source.push(shared_blocks);
        }
        page_source
    }

    fn hot_cold_direct_resident_pages(
        &self,
        rows: &[usize],
        shared_blocks: i32,
        tail_private: bool,
    ) -> Vec<i32> {
        let Some(hot_cold) = &self.hot_cold else {
            return Vec::new();
        };
        let keep_pages_per_row = hot_cold
            .hot_window_pages
            .saturating_add(hot_cold.chunk_pages)
            .max(1);
        let row_blocks = shared_blocks + if tail_private { 1 } else { 0 };
        let keep_start = row_blocks.saturating_sub(keep_pages_per_row);
        let mut resident_pages = HashSet::new();
        for (row_idx, _) in rows.iter().enumerate() {
            for block_col in keep_start..row_blocks {
                let page = if block_col < shared_blocks {
                    block_col
                } else {
                    shared_blocks + row_idx as i32
                };
                resident_pages.insert(page);
            }
        }
        let mut resident_pages = resident_pages.into_iter().collect::<Vec<_>>();
        resident_pages.sort_unstable();
        let limit = self.resident_slot_limit() as usize;
        if resident_pages.len() > limit {
            resident_pages.sort_unstable_by(|left, right| right.cmp(left));
            resident_pages.truncate(limit);
            resident_pages.sort_unstable();
        }
        resident_pages
    }

    fn install_hot_cold_direct_resident_pages(&mut self, total_pages: i32) {
        let Some(hot_cold) = &mut self.hot_cold else {
            return;
        };
        let total_pages = total_pages.max(0) as usize;
        hot_cold.page_states = vec![Some(PagedKvPageState::Dirty); total_pages];
        hot_cold.logical_to_slot = (0..total_pages).map(|slot| Some(slot as i32)).collect();
        hot_cold.slot_to_logical = (0..total_pages).map(|page| Some(page as i32)).collect();
        hot_cold.free_slots.clear();
    }

    fn install_hot_cold_direct_resident_subset(
        &mut self,
        total_pages: i32,
        resident_pages: &[i32],
    ) {
        let Some(hot_cold) = &mut self.hot_cold else {
            return;
        };
        let total_pages = total_pages.max(0) as usize;
        hot_cold.stream_cache.clear();
        hot_cold.stream_cache_lru.clear();
        hot_cold.page_states = vec![None; total_pages];
        hot_cold.logical_to_slot = vec![None; total_pages];
        hot_cold.slot_to_logical = vec![None; resident_pages.len()];
        hot_cold.free_slots.clear();
        for (slot, &page) in resident_pages.iter().enumerate() {
            hot_cold.mark_resident(page, slot as i32, PagedKvPageState::Dirty);
        }
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

    fn build_selected_prefix_page_tensor_on(
        &self,
        pages_src: &Array,
        page_source: &[i32],
        logical_pages: &[i32],
        dim: i32,
        target: StreamOrDevice,
    ) -> Result<Array> {
        if logical_pages.is_empty() {
            return Ok(Array::zeros_on(
                (0_i32, self.n_kv_heads, self.block_size, dim),
                self.dtype,
                target,
            )?);
        }
        let mut parts = Vec::with_capacity(logical_pages.len());
        for &logical_page in logical_pages {
            let source_page = page_source
                .get(logical_page as usize)
                .copied()
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "PagedKVCache::build_selected_prefix_page_tensor_on: logical page {logical_page} has no source"
                    )
                })?;
            parts.push(slice_strided_on(
                pages_src,
                [source_page, 0, 0, 0],
                [source_page + 1, self.n_kv_heads, self.block_size, dim],
                [1_i32, 1, 1, 1],
                target,
            )?);
        }
        if parts.len() == 1 {
            Ok(parts.remove(0))
        } else {
            let refs = parts.iter().collect::<Vec<_>>();
            Ok(concatenate_on(&refs, 0, target)?)
        }
    }

    fn save_hot_cold_direct_offloaded_prefix_runs_on(
        &mut self,
        k_pages_src: &Array,
        v_pages_src: &Array,
        page_source: &[i32],
        resident_pages: &[i32],
        target: StreamOrDevice,
    ) -> Result<()> {
        let resident_pages = resident_pages.iter().copied().collect::<HashSet<_>>();
        let chunk_pages = self
            .hot_cold
            .as_ref()
            .map(|hot_cold| hot_cold.chunk_pages.max(1))
            .unwrap_or(1);
        let mut logical_page = 0_i32;
        while (logical_page as usize) < page_source.len() {
            if resident_pages.contains(&logical_page) {
                logical_page += 1;
                continue;
            }
            let run_start = logical_page;
            let source_start = page_source[run_start as usize];
            let mut run_len = 1_i32;
            while (run_start + run_len) < page_source.len() as i32
                && run_len < chunk_pages
                && !resident_pages.contains(&(run_start + run_len))
                && page_source[(run_start + run_len) as usize] == source_start + run_len
            {
                run_len += 1;
            }
            self.save_hot_cold_direct_offloaded_prefix_run_on(
                k_pages_src,
                v_pages_src,
                run_start,
                source_start,
                run_len,
                target,
            )?;
            logical_page = run_start + run_len;
        }
        Ok(())
    }

    fn save_hot_cold_direct_offloaded_prefix_run_on(
        &mut self,
        k_pages_src: &Array,
        v_pages_src: &Array,
        run_start: i32,
        source_start: i32,
        run_len: i32,
        target: StreamOrDevice,
    ) -> Result<()> {
        let path = self
            .hot_cold
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("PagedKVCache::save_hot_cold_direct_offloaded_prefix_run_on: hot/cold tiering is disabled"))?
            .segment_path(run_start, run_len);
        let k_segment = slice_strided_on(
            k_pages_src,
            [source_start, 0, 0, 0],
            [
                source_start + run_len,
                self.n_kv_heads,
                self.block_size,
                self.head_dim,
            ],
            [1_i32, 1, 1, 1],
            target,
        )?;
        let v_segment = slice_strided_on(
            v_pages_src,
            [source_start, 0, 0, 0],
            [
                source_start + run_len,
                self.n_kv_heads,
                self.block_size,
                self.v_head_dim,
            ],
            [1_i32, 1, 1, 1],
            target,
        )?;
        Self::save_page_segment_file(&path, &k_segment, &v_segment)?;
        let bytes = fs::metadata(&path)
            .map(|meta| meta.len().min(usize::MAX as u64) as usize)
            .unwrap_or(0);
        if let Some(hot_cold) = &mut self.hot_cold {
            for idx in 0..run_len {
                hot_cold.mark_offloaded(
                    run_start + idx,
                    PagedKvPageSegment::new(path.clone(), bytes, run_start, idx, run_len),
                );
                hot_cold.swap_out_count = hot_cold.swap_out_count.saturating_add(1);
            }
        }
        Ok(())
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
        if self.hot_cold.is_some() {
            for (src_page, &dst_page) in dst_pages.iter().enumerate() {
                self.copy_single_prefix_page_on(
                    k_pages_src,
                    v_pages_src,
                    src_page as i32,
                    dst_page,
                    target,
                )?;
            }
            return Ok(());
        }
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
        let dst_slot = self.ensure_page_resident_for_write(dst_page, target)?;
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
            [dst_slot, 0, 0, 0],
            [
                dst_slot + 1,
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
            [dst_slot, 0, 0, 0],
            [
                dst_slot + 1,
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
            self.commit_execution_rows(dst_offsets)?;
            return Ok(());
        }

        let blocks = ceil_div(src_off, self.block_size);
        for block_col in 0..blocks {
            let src_page = src.block_tables[src_row][block_col as usize];
            if src_page < 0 {
                anyhow::bail!(
                    "PagedKVCache::adopt_row_from: missing src page row {src_row} block {block_col}"
                );
            }
            let dst_page = self.allocate_page(target)?;
            let (k_page, v_page) = src.page_slice_on(src_page, self.block_size, target)?;
            let dst_slot = self.ensure_page_resident_for_write(dst_page, target)?;
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
                [dst_slot, 0, 0, 0],
                [
                    dst_slot + 1,
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
                [dst_slot, 0, 0, 0],
                [
                    dst_slot + 1,
                    self.n_kv_heads,
                    self.block_size,
                    self.v_head_dim,
                ],
                [1_i32, 1, 1, 1],
                target,
            )?);
            self.block_tables[dst_row][block_col as usize] = dst_page;
            self.observability.adopt_page_copies =
                self.observability.adopt_page_copies.saturating_add(1);
        }
        dst_offsets[dst_row] = src_off;
        self.commit_execution_rows(dst_offsets)?;
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
        self.append_on_inner(k, v, offsets, per_row_lens, target, true)
    }

    fn append_on_inner(
        &mut self,
        k: &Array,
        v: &Array,
        offsets: &mut [i32],
        per_row_lens: &[i32],
        target: StreamOrDevice,
        enforce_hot_window: bool,
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
                let slot = self.ensure_page_resident_for_write(page, target)?;
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
                    [slot, 0, in_block, 0],
                    [slot + 1, self.n_kv_heads, in_block + take, self.head_dim],
                    [1_i32, 1, 1, 1],
                    target,
                )?);
                self.v_pages = Some(slice_update_on(
                    v_pages,
                    &v_part,
                    [slot, 0, in_block, 0],
                    [slot + 1, self.n_kv_heads, in_block + take, self.v_head_dim],
                    [1_i32, 1, 1, 1],
                    target,
                )?);
                logical_pos += take;
                src_pos += take;
            }
            debug_assert!(src_pos <= k_seq);
            offsets[row_usize] += n;
        }
        if enforce_hot_window {
            self.enforce_hot_window(offsets, target)?;
        }
        self.commit_execution_rows(offsets)?;
        Ok(())
    }

    pub(crate) fn hot_cold_needs_streaming_after(
        &self,
        offsets: &[i32],
        per_row_lens: &[i32],
    ) -> bool {
        let Some(hot_window_pages) = self
            .hot_cold
            .as_ref()
            .map(|hot_cold| hot_cold.hot_window_pages)
        else {
            return false;
        };
        offsets
            .iter()
            .zip(per_row_lens.iter())
            .any(|(&off, &n)| ceil_div(off + n, self.block_size) > hot_window_pages)
    }

    fn needs_hot_cold_streaming(&self, offsets: &[i32]) -> bool {
        let Some(hot_window_pages) = self
            .hot_cold
            .as_ref()
            .map(|hot_cold| hot_cold.hot_window_pages)
        else {
            return false;
        };
        offsets
            .iter()
            .any(|&row_len| ceil_div(row_len, self.block_size) > hot_window_pages)
    }

    fn enforce_hot_window(&mut self, offsets: &[i32], target: StreamOrDevice) -> Result<()> {
        let Some(hot_window_pages) = self
            .hot_cold
            .as_ref()
            .map(|hot_cold| hot_cold.hot_window_pages)
        else {
            return Ok(());
        };
        let mut protected = HashSet::new();
        for (row, &row_len) in offsets.iter().enumerate() {
            let blocks = ceil_div(row_len, self.block_size);
            if blocks == 0 {
                continue;
            }
            let keep_start = (blocks - hot_window_pages).max(0);
            for block in keep_start..blocks {
                let page = self.block_tables[row][block as usize];
                if page >= 0 && self.page_ref_count(page) > 0 {
                    protected.insert(page);
                }
            }
        }

        let resident_pages = self
            .hot_cold
            .as_ref()
            .map(|hot_cold| {
                hot_cold
                    .slot_to_logical
                    .iter()
                    .enumerate()
                    .filter_map(|(slot, page)| page.map(|page| (page, slot as i32)))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        let victims = resident_pages
            .into_iter()
            .filter(|(page, _)| !protected.contains(page) && self.page_ref_count(*page) > 0)
            .collect::<Vec<_>>();
        for run in self.group_contiguous_resident_runs(victims) {
            self.offload_resident_page_run(&run, target)?;
        }
        Ok(())
    }

    fn enforce_hot_window_if_over_resident_budget(
        &mut self,
        offsets: &[i32],
        target: StreamOrDevice,
    ) -> Result<()> {
        let Some(hot_cold) = &self.hot_cold else {
            return Ok(());
        };
        let resident_pages = hot_cold
            .slot_to_logical
            .iter()
            .filter(|page| page.is_some())
            .count();
        if resident_pages as i32 <= self.resident_slot_limit() {
            return Ok(());
        }
        self.enforce_hot_window(offsets, target)
    }

    fn validate_prefill_attention_inputs(
        &self,
        queries: &Array,
        k: &Array,
        v: &Array,
        offsets: &[i32],
        per_row_lens: &[i32],
        mask_arr: Option<&Array>,
    ) -> Result<()> {
        self.validate_update_inputs(k, v, offsets, per_row_lens)?;
        if self.v_head_dim != self.head_dim {
            anyhow::bail!(
                "PagedKVCache::prefill: value head dim {} != key/query head dim {}",
                self.v_head_dim,
                self.head_dim
            );
        }
        let q_shape = queries.shape();
        let q_dims = q_shape.as_slice();
        let k_shape = k.shape();
        let k_dims = k_shape.as_slice();
        if q_dims.len() != 4 {
            anyhow::bail!(
                "PagedKVCache::prefill: expected rank-4 Q, got {}",
                q_dims.len()
            );
        }
        if q_dims[0] != self.batch
            || q_dims[2] != k_dims[2]
            || q_dims[3] != self.head_dim
            || q_dims[1] % self.n_kv_heads != 0
        {
            anyhow::bail!(
                "PagedKVCache::prefill: query shape mismatch q={q_dims:?} batch={} hkv={} d={} k_seq={}",
                self.batch,
                self.n_kv_heads,
                self.head_dim,
                k_dims[2]
            );
        }
        if per_row_lens.iter().any(|&n| n <= 0) {
            anyhow::bail!("PagedKVCache::prefill: every row must append at least one token");
        }
        let max_off_after = offsets
            .iter()
            .zip(per_row_lens.iter())
            .map(|(o, n)| o + n)
            .max()
            .unwrap_or(0);
        match mask_arr {
            Some(mask) => {
                let mask_shape = mask.shape();
                let mask_dims = mask_shape.as_slice();
                if mask_dims.len() != 4
                    || !(mask_dims[0] == 1 || mask_dims[0] == self.batch)
                    || !(mask_dims[1] == 1 || mask_dims[1] == q_dims[1])
                    || mask_dims[2] != q_dims[2]
                    || mask_dims[3] != max_off_after
                {
                    anyhow::bail!(
                        "PagedKVCache::prefill: mask shape {mask_dims:?} is incompatible with q={q_dims:?} max_kv={max_off_after}"
                    );
                }
            }
            None => {
                if per_row_lens.iter().any(|&n| n != q_dims[2]) {
                    anyhow::bail!(
                        "PagedKVCache::prefill: ragged prefill without explicit mask is unsupported"
                    );
                }
            }
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

    fn resident_block_table_array(&self, max_blocks: i32) -> Result<Array> {
        if max_blocks <= 0 || max_blocks > self.max_blocks_per_row {
            anyhow::bail!(
                "PagedKVCache::resident_block_table_array: max_blocks {max_blocks} outside [1, {}]",
                self.max_blocks_per_row
            );
        }
        let mut flat = vec![-1_i32; (self.batch * max_blocks) as usize];
        for row in 0..self.batch as usize {
            for block in 0..max_blocks as usize {
                let page = self.block_tables[row][block];
                if page < 0 {
                    continue;
                }
                flat[row * max_blocks as usize + block] =
                    self.resident_slot_for_read(page)?.ok_or_else(|| {
                        anyhow::anyhow!(
                            "PagedKVCache::resident_block_table_array: page {page} is not resident"
                        )
                    })?;
            }
        }
        let arr: Array = (flat.as_slice(), &[self.batch, max_blocks][..]).try_into()?;
        Ok(arr)
    }

    fn referenced_pages_for_offsets(&self, offsets: &[i32]) -> Result<Vec<i32>> {
        self.validate_offsets_allocated(offsets)?;
        let mut pages = HashSet::new();
        for (row, &row_len) in offsets.iter().enumerate().take(self.batch as usize) {
            let blocks = ceil_div(row_len, self.block_size);
            for block_col in 0..blocks {
                let page = self.block_tables[row][block_col as usize];
                if page < 0 {
                    anyhow::bail!(
                        "PagedKVCache::referenced_pages_for_offsets: missing page row {row} block {block_col}"
                    );
                }
                pages.insert(page);
            }
        }
        let mut pages = pages.into_iter().collect::<Vec<_>>();
        pages.sort_unstable();
        Ok(pages)
    }

    fn try_stage_hot_cold_context_for_paged_decode(
        &mut self,
        offsets: &[i32],
        target: StreamOrDevice,
    ) -> Result<bool> {
        if self.hot_cold.is_none() {
            return Ok(false);
        }
        let pages = self.referenced_pages_for_offsets(offsets)?;
        if pages.len() as i32 > self.resident_slot_limit() {
            return Ok(false);
        }
        let protected = pages.iter().copied().collect::<HashSet<_>>();
        for page in pages {
            self.ensure_page_resident_for_read(page, &protected, target)?;
        }
        Ok(true)
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
            if self.hot_cold.is_some() {
                let _ = self.remove_hot_cold_page(page);
            }
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
        if self.hot_cold.is_some() {
            self.ensure_page_resident_for_write(page, target)?;
        } else {
            self.ensure_page_capacity(page + 1, target)?;
        }
        self.set_page_ref_count(page, 1);
        Ok(page)
    }

    fn remove_hot_cold_page(&mut self, page: i32) -> Result<()> {
        let Some(hot_cold) = &mut self.hot_cold else {
            return Ok(());
        };
        hot_cold.ensure_page(page);
        hot_cold.invalidate_stream_cache_page(page);
        if let Some(slot) = hot_cold.logical_to_slot[page as usize].take() {
            hot_cold.mark_slot_free(slot);
        }
        if let Some(state) = hot_cold.page_states[page as usize].take() {
            if let Some((path, _, _, _, _)) = state.cloned_segment_info() {
                Self::remove_offloaded_segment_if_unreferenced(hot_cold, page, &path)?;
            }
        }
        Ok(())
    }

    fn ensure_resident_slot_capacity(
        &mut self,
        needed_slots: i32,
        target: StreamOrDevice,
    ) -> Result<()> {
        if needed_slots <= self.page_capacity {
            return Ok(());
        }
        let old_capacity = self.page_capacity;
        let max_resident = self.resident_slot_limit();
        let next = if old_capacity == 0 {
            needed_slots
        } else {
            (old_capacity * 2).max(needed_slots)
        };
        let target_capacity = next.min(max_resident);
        anyhow::ensure!(
            needed_slots <= target_capacity,
            "PagedKVCache::ensure_resident_slot_capacity: resident slot budget exhausted"
        );

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
        if let Some(hot_cold) = &mut self.hot_cold {
            hot_cold
                .slot_to_logical
                .resize(target_capacity as usize, None);
            for slot in (old_capacity..target_capacity).rev() {
                hot_cold.free_slots.push(slot);
            }
        }
        Ok(())
    }

    fn ensure_page_resident_for_write(&mut self, page: i32, target: StreamOrDevice) -> Result<i32> {
        if self.hot_cold.is_none() {
            self.ensure_page_capacity(page + 1, target)?;
            return Ok(page);
        }
        if let Some(slot) = self
            .hot_cold
            .as_ref()
            .and_then(|hot_cold| hot_cold.slot_for(page))
        {
            if let Some(hot_cold) = &mut self.hot_cold {
                hot_cold.ensure_page(page);
                let previous_state = hot_cold.page_states[page as usize].take();
                if let Some(state) = previous_state {
                    if let Some((path, _, _, _, _)) = state.cloned_segment_info() {
                        Self::remove_offloaded_segment_if_unreferenced(hot_cold, page, &path)?;
                    }
                }
                hot_cold.invalidate_stream_cache_page(page);
                hot_cold.page_states[page as usize] = Some(PagedKvPageState::Dirty);
            }
            return Ok(slot);
        }

        let slot = self.allocate_resident_slot(page, target)?;
        let offloaded_segment = self
            .hot_cold
            .as_ref()
            .and_then(|hot_cold| hot_cold.page_states.get(page as usize))
            .and_then(|state| state.as_ref())
            .and_then(|state| match state {
                PagedKvPageState::Offloaded {
                    path,
                    bytes,
                    start_page,
                    page_index,
                    page_count,
                } => Some((path.clone(), *bytes, *start_page, *page_index, *page_count)),
                _ => None,
            });

        if let Some((path, bytes, start_page, page_index, page_count)) = offloaded_segment {
            let segment =
                PagedKvPageSegment::new(path.clone(), bytes, start_page, page_index, page_count);
            if let Some(hot_cold) = &mut self.hot_cold {
                hot_cold.mark_loading(page, segment.clone());
            }
            let load_result = Self::load_page_segment_file(&path)
                .with_context(|| format!("load active KV page {}", path.display()));
            let (k_segment, v_segment) = match load_result {
                Ok(segment_pair) => segment_pair,
                Err(err) => {
                    if let Some(hot_cold) = &mut self.hot_cold {
                        let bytes = fs::metadata(&path)
                            .map(|meta| meta.len().min(usize::MAX as u64) as usize)
                            .unwrap_or(0);
                        hot_cold.mark_offloaded(
                            page,
                            PagedKvPageSegment::new(
                                path, bytes, start_page, page_index, page_count,
                            ),
                        );
                        hot_cold.mark_slot_free(slot);
                    }
                    return Err(err);
                }
            };
            let k_page = slice_strided_on(
                &k_segment,
                [page_index, 0, 0, 0],
                [
                    page_index + 1,
                    self.n_kv_heads,
                    self.block_size,
                    self.head_dim,
                ],
                [1_i32, 1, 1, 1],
                target,
            )?;
            let v_page = slice_strided_on(
                &v_segment,
                [page_index, 0, 0, 0],
                [
                    page_index + 1,
                    self.n_kv_heads,
                    self.block_size,
                    self.v_head_dim,
                ],
                [1_i32, 1, 1, 1],
                target,
            )?;
            self.write_full_page_to_slot(slot, &k_page, &v_page, target)?;
            if let Some(hot_cold) = &mut self.hot_cold {
                hot_cold.invalidate_stream_cache_page(page);
                hot_cold.swap_in_count = hot_cold.swap_in_count.saturating_add(1);
                Self::remove_offloaded_segment_if_unreferenced(hot_cold, page, &path)?;
            }
        }

        if let Some(hot_cold) = &mut self.hot_cold {
            hot_cold.mark_resident(page, slot, PagedKvPageState::Dirty);
        }
        Ok(slot)
    }

    fn ensure_page_resident_for_read(
        &mut self,
        page: i32,
        protected: &HashSet<i32>,
        target: StreamOrDevice,
    ) -> Result<i32> {
        if self.hot_cold.is_none() {
            self.ensure_page_capacity(page + 1, target)?;
            return Ok(page);
        }
        if let Some(slot) = self
            .hot_cold
            .as_ref()
            .and_then(|hot_cold| hot_cold.slot_for(page))
        {
            return Ok(slot);
        }
        let segment = self
            .hot_cold
            .as_ref()
            .and_then(|hot_cold| hot_cold.page_states.get(page as usize))
            .and_then(|state| state.as_ref())
            .and_then(PagedKvPageState::cloned_segment_info)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "PagedKVCache::ensure_page_resident_for_read: page {page} is missing"
                )
            })?;
        self.stage_offloaded_segment_for_read(segment, protected, target)?;
        self.hot_cold
            .as_ref()
            .and_then(|hot_cold| hot_cold.slot_for(page))
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "PagedKVCache::ensure_page_resident_for_read: page {page} was not staged"
                )
            })
    }

    fn stage_offloaded_segment_for_read(
        &mut self,
        segment: (PathBuf, usize, i32, i32, i32),
        protected: &HashSet<i32>,
        target: StreamOrDevice,
    ) -> Result<()> {
        let (path, bytes, start_page, _page_index, page_count) = segment;
        let (k_segment, v_segment) = Self::load_page_segment_file(&path)
            .with_context(|| format!("stage active KV page segment {}", path.display()))?;
        let mut staged = 0_u64;
        for idx in 0..page_count {
            let logical_page = start_page + idx;
            if !protected.contains(&logical_page) {
                continue;
            }
            if self
                .hot_cold
                .as_ref()
                .and_then(|hot_cold| hot_cold.slot_for(logical_page))
                .is_some()
            {
                continue;
            }
            let still_same_segment = self
                .hot_cold
                .as_ref()
                .and_then(|hot_cold| hot_cold.page_states.get(logical_page as usize))
                .and_then(|state| state.as_ref())
                .and_then(PagedKvPageState::segment_info)
                .is_some_and(
                    |(other_path, _bytes, other_start, other_index, other_count)| {
                        other_path == &path
                            && other_start == start_page
                            && other_index == idx
                            && other_count == page_count
                    },
                );
            if !still_same_segment {
                continue;
            }
            let slot = self.allocate_resident_slot_protected(protected, target)?;
            let k_page = slice_strided_on(
                &k_segment,
                [idx, 0, 0, 0],
                [idx + 1, self.n_kv_heads, self.block_size, self.head_dim],
                [1_i32, 1, 1, 1],
                target,
            )?;
            let v_page = slice_strided_on(
                &v_segment,
                [idx, 0, 0, 0],
                [idx + 1, self.n_kv_heads, self.block_size, self.v_head_dim],
                [1_i32, 1, 1, 1],
                target,
            )?;
            self.write_full_page_to_slot(slot, &k_page, &v_page, target)?;
            if let Some(hot_cold) = &mut self.hot_cold {
                hot_cold.invalidate_stream_cache_page(logical_page);
                hot_cold.mark_resident_clean(
                    logical_page,
                    slot,
                    PagedKvPageSegment::new(path.clone(), bytes, start_page, idx, page_count),
                );
            }
            staged = staged.saturating_add(1);
        }
        if staged > 0 {
            if let Some(hot_cold) = &mut self.hot_cold {
                hot_cold.swap_in_count = hot_cold.swap_in_count.saturating_add(staged);
            }
        }
        Ok(())
    }

    fn resident_slot_for_read(&self, page: i32) -> Result<Option<i32>> {
        if let Some(hot_cold) = &self.hot_cold {
            return Ok(hot_cold.slot_for(page));
        }
        if page < 0 || page >= self.page_capacity {
            anyhow::bail!("PagedKVCache: page {page} outside resident capacity");
        }
        Ok(Some(page))
    }

    fn allocate_resident_slot(
        &mut self,
        protected_page: i32,
        target: StreamOrDevice,
    ) -> Result<i32> {
        self.allocate_resident_slot_protected(&HashSet::from([protected_page]), target)
    }

    fn allocate_resident_slot_protected(
        &mut self,
        protected: &HashSet<i32>,
        target: StreamOrDevice,
    ) -> Result<i32> {
        if self.hot_cold.is_none() {
            let page = protected
                .iter()
                .next()
                .copied()
                .ok_or_else(|| anyhow::anyhow!("PagedKVCache: empty protected set"))?;
            return Ok(page);
        }
        if let Some(slot) = self
            .hot_cold
            .as_mut()
            .and_then(|hot_cold| hot_cold.free_slots.pop())
        {
            return Ok(slot);
        }

        let needed = self.page_capacity + 1;
        let can_grow = self.hot_cold.is_some() && needed <= self.resident_slot_limit();
        if can_grow {
            self.ensure_resident_slot_capacity(needed, target)?;
            if let Some(slot) = self
                .hot_cold
                .as_mut()
                .and_then(|hot_cold| hot_cold.free_slots.pop())
            {
                return Ok(slot);
            }
        }

        let victim_run = self.select_resident_victim_run(protected)?;
        self.offload_resident_page_run(&victim_run, target)?;
        self.hot_cold
            .as_mut()
            .and_then(|hot_cold| hot_cold.free_slots.pop())
            .ok_or_else(|| anyhow::anyhow!("PagedKVCache: victim eviction freed no resident slot"))
    }

    fn resident_slot_limit(&self) -> i32 {
        self.hot_cold
            .as_ref()
            .map(|tiering| {
                tiering
                    .hot_window_pages
                    .saturating_add(tiering.chunk_pages)
                    .saturating_mul(self.batch)
            })
            .unwrap_or(self.max_pages)
            .min(self.max_pages)
            .max(1)
    }

    fn select_resident_victim_run(&self, protected: &HashSet<i32>) -> Result<Vec<(i32, i32)>> {
        let Some(hot_cold) = &self.hot_cold else {
            anyhow::bail!("PagedKVCache::select_resident_victim_run: hot/cold tiering is disabled");
        };
        let victims = hot_cold
            .slot_to_logical
            .iter()
            .enumerate()
            .filter_map(|(slot, page)| page.map(|page| (page, slot as i32)))
            .filter(|(page, _)| !protected.contains(page) && self.page_ref_count(*page) > 0)
            .collect::<Vec<_>>();
        self.group_resident_victims(victims)
    }

    fn group_resident_victims(&self, mut victims: Vec<(i32, i32)>) -> Result<Vec<(i32, i32)>> {
        victims.sort_by_key(|(page, _)| *page);
        let Some(&(first_page, _)) = victims.first() else {
            anyhow::bail!("PagedKVCache: no resident page can be evicted");
        };
        let chunk_pages = self
            .hot_cold
            .as_ref()
            .map(|tiering| tiering.chunk_pages.max(1))
            .unwrap_or(1);
        let mut run = Vec::new();
        for (expected_page, (page, slot)) in (first_page..).zip(victims) {
            if page != expected_page || run.len() >= chunk_pages as usize {
                break;
            }
            run.push((page, slot));
        }
        Ok(run)
    }

    fn group_contiguous_resident_runs(&self, mut pages: Vec<(i32, i32)>) -> Vec<Vec<(i32, i32)>> {
        pages.sort_by_key(|(page, _)| *page);
        let chunk_pages = self
            .hot_cold
            .as_ref()
            .map(|tiering| tiering.chunk_pages.max(1) as usize)
            .unwrap_or(1);
        let mut runs = Vec::new();
        let mut current = Vec::new();
        let mut expected_page: Option<i32> = None;
        for (page, slot) in pages {
            let starts_new_run = current.len() >= chunk_pages
                || expected_page
                    .map(|expected| page != expected)
                    .unwrap_or(false);
            if starts_new_run && !current.is_empty() {
                runs.push(std::mem::take(&mut current));
            }
            current.push((page, slot));
            expected_page = Some(page + 1);
        }
        if !current.is_empty() {
            runs.push(current);
        }
        runs
    }

    fn offload_resident_page_run(
        &mut self,
        pages: &[(i32, i32)],
        target: StreamOrDevice,
    ) -> Result<()> {
        let mut dirty_run = Vec::new();
        for &(page, slot) in pages {
            if self.free_resident_clean_page(page, slot)? {
                if !dirty_run.is_empty() {
                    self.offload_dirty_resident_page_run(&dirty_run, target)?;
                    dirty_run.clear();
                }
            } else {
                dirty_run.push((page, slot));
            }
        }
        if !dirty_run.is_empty() {
            self.offload_dirty_resident_page_run(&dirty_run, target)?;
        }
        Ok(())
    }

    fn free_resident_clean_page(&mut self, page: i32, slot: i32) -> Result<bool> {
        let Some(hot_cold) = &mut self.hot_cold else {
            return Ok(false);
        };
        let Some(PagedKvPageState::ResidentClean {
            path,
            bytes,
            start_page,
            page_index,
            page_count,
        }) = hot_cold
            .page_states
            .get_mut(page as usize)
            .and_then(Option::take)
        else {
            return Ok(false);
        };
        hot_cold.mark_slot_free(slot);
        hot_cold.mark_offloaded(
            page,
            PagedKvPageSegment::new(path, bytes, start_page, page_index, page_count),
        );
        Ok(true)
    }

    fn offload_dirty_resident_page_run(
        &mut self,
        pages: &[(i32, i32)],
        target: StreamOrDevice,
    ) -> Result<()> {
        anyhow::ensure!(
            !pages.is_empty(),
            "PagedKVCache::offload_resident_page_run: empty page run"
        );
        let start_page = pages[0].0;
        let path = self
            .hot_cold
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("PagedKVCache::offload_resident_page: disabled"))?
            .segment_path(start_page, pages.len() as i32);
        let k_pages = self.k_pages.as_ref().ok_or_else(|| {
            anyhow::anyhow!("PagedKVCache::offload_resident_page_run: K pages are unallocated")
        })?;
        let v_pages = self.v_pages.as_ref().ok_or_else(|| {
            anyhow::anyhow!("PagedKVCache::offload_resident_page_run: V pages are unallocated")
        })?;
        let mut k_parts = Vec::with_capacity(pages.len());
        let mut v_parts = Vec::with_capacity(pages.len());
        for &(_, slot) in pages {
            k_parts.push(slice_strided_on(
                k_pages,
                [slot, 0, 0, 0],
                [slot + 1, self.n_kv_heads, self.block_size, self.head_dim],
                [1_i32, 1, 1, 1],
                target,
            )?);
            v_parts.push(slice_strided_on(
                v_pages,
                [slot, 0, 0, 0],
                [slot + 1, self.n_kv_heads, self.block_size, self.v_head_dim],
                [1_i32, 1, 1, 1],
                target,
            )?);
        }
        let k_refs = k_parts.iter().collect::<Vec<_>>();
        let v_refs = v_parts.iter().collect::<Vec<_>>();
        let k_segment = concatenate_on(&k_refs, 0, target)?;
        let v_segment = concatenate_on(&v_refs, 0, target)?;
        Self::save_page_segment_file(&path, &k_segment, &v_segment)?;
        let bytes = fs::metadata(&path)
            .map(|meta| meta.len().min(usize::MAX as u64) as usize)
            .unwrap_or(0);
        if let Some(hot_cold) = &mut self.hot_cold {
            for (idx, &(page, slot)) in pages.iter().enumerate() {
                hot_cold.invalidate_stream_cache_page(page);
                hot_cold.mark_offloaded(
                    page,
                    PagedKvPageSegment::new(
                        path.clone(),
                        bytes,
                        start_page,
                        idx as i32,
                        pages.len() as i32,
                    ),
                );
                hot_cold.mark_slot_free(slot);
                hot_cold.swap_out_count = hot_cold.swap_out_count.saturating_add(1);
            }
        }
        Ok(())
    }

    fn save_page_segment_file(path: &Path, k_pages: &Array, v_pages: &Array) -> Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("create active KV page dir {}", parent.display()))?;
        }
        let stem = path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or("page");
        let tmp_path = path.with_file_name(format!(
            "{stem}.tmp-{}.safetensors",
            uuid::Uuid::new_v4().simple()
        ));
        let mut tensors = HashMap::new();
        tensors.insert("k".to_owned(), k_pages.clone());
        tensors.insert("v".to_owned(), v_pages.clone());
        let mut metadata = HashMap::new();
        metadata.insert(
            "ironmlx.active_kv.schema".to_owned(),
            "hot_cold_segment_v1".to_owned(),
        );
        let tmp = tmp_path.to_string_lossy().into_owned();
        mlx::io::save_safetensors(&tmp, &tensors, &metadata)
            .with_context(|| format!("save active KV page {}", tmp_path.display()))?;
        fs::rename(&tmp_path, path).with_context(|| {
            format!(
                "install active KV page {} -> {}",
                tmp_path.display(),
                path.display()
            )
        })?;
        Ok(())
    }

    fn load_page_segment_file(path: &Path) -> Result<(Array, Array)> {
        let path_str = path.to_string_lossy().into_owned();
        let (mut tensors, _) = mlx::io::load_safetensors(&path_str)
            .with_context(|| format!("load active KV page {path_str}"))?;
        let k = tensors
            .remove("k")
            .ok_or_else(|| anyhow::anyhow!("active KV page {} missing tensor k", path.display()))?;
        let v = tensors
            .remove("v")
            .ok_or_else(|| anyhow::anyhow!("active KV page {} missing tensor v", path.display()))?;
        Ok((k, v))
    }

    fn remove_offloaded_segment_if_unreferenced(
        hot_cold: &PagedKvHotColdTiering,
        page: i32,
        path: &Path,
    ) -> Result<()> {
        let still_referenced = hot_cold.page_states.iter().enumerate().any(|(idx, state)| {
            idx != page as usize
                && matches!(
                    state,
                    Some(PagedKvPageState::ResidentClean {
                        path: other_path,
                        ..
                    })
                    | Some(PagedKvPageState::Offloaded {
                        path: other_path,
                        ..
                    }) | Some(PagedKvPageState::Loading {
                        path: other_path,
                        ..
                    }) if other_path == path
                )
        });
        if still_referenced {
            return Ok(());
        }
        match fs::remove_file(path) {
            Ok(()) => {}
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
            Err(err) => {
                return Err(err).with_context(|| {
                    format!("remove offloaded active KV page {}", path.display())
                });
            }
        }
        Ok(())
    }

    fn write_full_page_to_slot(
        &mut self,
        slot: i32,
        k_page: &Array,
        v_page: &Array,
        target: StreamOrDevice,
    ) -> Result<()> {
        let k_pages = self.k_pages.as_ref().ok_or_else(|| {
            anyhow::anyhow!("PagedKVCache::write_full_page_to_slot: K pages are unallocated")
        })?;
        let v_pages = self.v_pages.as_ref().ok_or_else(|| {
            anyhow::anyhow!("PagedKVCache::write_full_page_to_slot: V pages are unallocated")
        })?;
        self.k_pages = Some(slice_update_on(
            k_pages,
            k_page,
            [slot, 0, 0, 0],
            [slot + 1, self.n_kv_heads, self.block_size, self.head_dim],
            [1_i32, 1, 1, 1],
            target,
        )?);
        self.v_pages = Some(slice_update_on(
            v_pages,
            v_page,
            [slot, 0, 0, 0],
            [slot + 1, self.n_kv_heads, self.block_size, self.v_head_dim],
            [1_i32, 1, 1, 1],
            target,
        )?);
        Ok(())
    }

    fn page_slice_on(
        &self,
        page: i32,
        take: i32,
        target: StreamOrDevice,
    ) -> Result<(Array, Array)> {
        anyhow::ensure!(
            take >= 0 && take <= self.block_size,
            "PagedKVCache::page_slice_on: take {take} outside [0, {}]",
            self.block_size
        );
        if take == 0 {
            let k = Array::zeros_on(
                (1_i32, self.n_kv_heads, 0_i32, self.head_dim),
                self.dtype,
                target,
            )?;
            let v = Array::zeros_on(
                (1_i32, self.n_kv_heads, 0_i32, self.v_head_dim),
                self.dtype,
                target,
            )?;
            return Ok((k, v));
        }
        if let Some(slot) = self.resident_slot_for_read(page)? {
            let k_pages = self.k_pages.as_ref().ok_or_else(|| {
                anyhow::anyhow!("PagedKVCache::page_slice_on: K pages are unallocated")
            })?;
            let v_pages = self.v_pages.as_ref().ok_or_else(|| {
                anyhow::anyhow!("PagedKVCache::page_slice_on: V pages are unallocated")
            })?;
            let k = slice_strided_on(
                k_pages,
                [slot, 0, 0, 0],
                [slot + 1, self.n_kv_heads, take, self.head_dim],
                [1_i32, 1, 1, 1],
                target,
            )?;
            let v = slice_strided_on(
                v_pages,
                [slot, 0, 0, 0],
                [slot + 1, self.n_kv_heads, take, self.v_head_dim],
                [1_i32, 1, 1, 1],
                target,
            )?;
            return Ok((k, v));
        }

        let path = self
            .hot_cold
            .as_ref()
            .and_then(|hot_cold| hot_cold.page_states.get(page as usize))
            .and_then(|state| state.as_ref())
            .and_then(|state| match state {
                PagedKvPageState::Offloaded {
                    path, page_index, ..
                } => Some((path.clone(), *page_index)),
                _ => None,
            })
            .ok_or_else(|| {
                anyhow::anyhow!("PagedKVCache::page_slice_on: page {page} is missing")
            })?;
        let (path, page_index) = path;
        let (k_segment, v_segment) = Self::load_page_segment_file(&path)
            .with_context(|| format!("load offloaded active KV page {}", path.display()))?;
        let k = slice_strided_on(
            &k_segment,
            [page_index, 0, 0, 0],
            [page_index + 1, self.n_kv_heads, take, self.head_dim],
            [1_i32, 1, 1, 1],
            target,
        )?;
        let v = slice_strided_on(
            &v_segment,
            [page_index, 0, 0, 0],
            [page_index + 1, self.n_kv_heads, take, self.v_head_dim],
            [1_i32, 1, 1, 1],
            target,
        )?;
        Ok((k, v))
    }

    fn page_slice_streaming_on(
        &mut self,
        page: i32,
        take: i32,
        target: StreamOrDevice,
    ) -> Result<(Array, Array)> {
        anyhow::ensure!(
            take >= 0 && take <= self.block_size,
            "PagedKVCache::page_slice_streaming_on: take {take} outside [0, {}]",
            self.block_size
        );
        if take == 0 {
            let k = Array::zeros_on(
                (1_i32, self.n_kv_heads, 0_i32, self.head_dim),
                self.dtype,
                target,
            )?;
            let v = Array::zeros_on(
                (1_i32, self.n_kv_heads, 0_i32, self.v_head_dim),
                self.dtype,
                target,
            )?;
            return Ok((k, v));
        }
        if let Some(slot) = self.resident_slot_for_read(page)? {
            let k_pages = self.k_pages.as_ref().ok_or_else(|| {
                anyhow::anyhow!("PagedKVCache::page_slice_streaming_on: K pages are unallocated")
            })?;
            let v_pages = self.v_pages.as_ref().ok_or_else(|| {
                anyhow::anyhow!("PagedKVCache::page_slice_streaming_on: V pages are unallocated")
            })?;
            let k = slice_strided_on(
                k_pages,
                [slot, 0, 0, 0],
                [slot + 1, self.n_kv_heads, take, self.head_dim],
                [1_i32, 1, 1, 1],
                target,
            )?;
            let v = slice_strided_on(
                v_pages,
                [slot, 0, 0, 0],
                [slot + 1, self.n_kv_heads, take, self.v_head_dim],
                [1_i32, 1, 1, 1],
                target,
            )?;
            return Ok((k, v));
        }

        let path = self
            .hot_cold
            .as_ref()
            .and_then(|hot_cold| hot_cold.page_states.get(page as usize))
            .and_then(|state| state.as_ref())
            .and_then(|state| match state {
                PagedKvPageState::Offloaded {
                    path,
                    start_page,
                    page_index,
                    page_count,
                    ..
                } => Some((path.clone(), *start_page, *page_index, *page_count)),
                _ => None,
            })
            .ok_or_else(|| {
                anyhow::anyhow!("PagedKVCache::page_slice_streaming_on: page {page} is missing")
            })?;
        let (path, start_page, page_index, page_count) = path;
        if let Some((k_page, v_page)) = self
            .hot_cold
            .as_mut()
            .and_then(|hot_cold| hot_cold.cached_stream_page(page))
        {
            let k = slice_strided_on(
                &k_page,
                [0_i32, 0, 0, 0],
                [1_i32, self.n_kv_heads, take, self.head_dim],
                [1_i32, 1, 1, 1],
                target,
            )?;
            let v = slice_strided_on(
                &v_page,
                [0_i32, 0, 0, 0],
                [1_i32, self.n_kv_heads, take, self.v_head_dim],
                [1_i32, 1, 1, 1],
                target,
            )?;
            return Ok((k, v));
        }
        let (k_segment, v_segment) = Self::load_page_segment_file(&path)
            .with_context(|| format!("stream offloaded active KV page {}", path.display()))?;
        let k = slice_strided_on(
            &k_segment,
            [page_index, 0, 0, 0],
            [page_index + 1, self.n_kv_heads, take, self.head_dim],
            [1_i32, 1, 1, 1],
            target,
        )?;
        let v = slice_strided_on(
            &v_segment,
            [page_index, 0, 0, 0],
            [page_index + 1, self.n_kv_heads, take, self.v_head_dim],
            [1_i32, 1, 1, 1],
            target,
        )?;
        let mut segment_cache_entries = Vec::new();
        if let Some(hot_cold) = self.hot_cold.as_ref() {
            for idx in 0..page_count {
                let logical_page = start_page + idx;
                let still_same_segment = hot_cold
                    .page_states
                    .get(logical_page as usize)
                    .and_then(|state| state.as_ref())
                    .is_some_and(|state| match state {
                        PagedKvPageState::Offloaded {
                            path: other_path,
                            start_page: other_start,
                            page_index: other_index,
                            page_count: other_count,
                            ..
                        } => {
                            other_path == &path
                                && *other_start == start_page
                                && *other_index == idx
                                && *other_count == page_count
                        }
                        _ => false,
                    });
                if still_same_segment {
                    let k_page = slice_strided_on(
                        &k_segment,
                        [idx, 0, 0, 0],
                        [idx + 1, self.n_kv_heads, self.block_size, self.head_dim],
                        [1_i32, 1, 1, 1],
                        target,
                    )?;
                    let v_page = slice_strided_on(
                        &v_segment,
                        [idx, 0, 0, 0],
                        [idx + 1, self.n_kv_heads, self.block_size, self.v_head_dim],
                        [1_i32, 1, 1, 1],
                        target,
                    )?;
                    segment_cache_entries.push((logical_page, k_page, v_page));
                }
            }
        }
        if let Some(hot_cold) = self.hot_cold.as_mut() {
            for (logical_page, k_page, v_page) in segment_cache_entries {
                hot_cold.cache_stream_page(logical_page, &k_page, &v_page);
            }
            hot_cold.stream_read_count = hot_cold.stream_read_count.saturating_add(1);
        }
        Ok((k, v))
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
        let (k_page, v_page) = self.page_slice_on(src_page, self.block_size, target)?;
        let dst_slot = self.ensure_page_resident_for_write(dst_page, target)?;
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
            [dst_slot, 0, 0, 0],
            [
                dst_slot + 1,
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
            [dst_slot, 0, 0, 0],
            [
                dst_slot + 1,
                self.n_kv_heads,
                self.block_size,
                self.v_head_dim,
            ],
            [1_i32, 1, 1, 1],
            target,
        )?);
        self.observability.cow_page_copies = self.observability.cow_page_copies.saturating_add(1);
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

fn standard_prefill_valid_mask_on(
    old_len: i32,
    row_len: i32,
    q_start: i32,
    q_take: i32,
    key_start: i32,
    key_take: i32,
    _target: StreamOrDevice,
) -> Result<Array> {
    let mut flat = vec![false; (q_take * key_take) as usize];
    for q in 0..q_take {
        let q_abs = old_len + q_start + q;
        let real_query = q_abs < row_len;
        for k in 0..key_take {
            let k_abs = key_start + k;
            let allow = if real_query {
                k_abs < row_len && k_abs <= q_abs
            } else {
                k_abs == q_abs
            };
            flat[(q * key_take + k) as usize] = allow;
        }
    }
    let mask: Array = (&flat[..], &[1_i32, 1_i32, q_take, key_take][..]).try_into()?;
    Ok(mask)
}

impl Drop for PagedKVCache {
    fn drop(&mut self) {
        if let Some(hot_cold) = &self.hot_cold {
            let _ = fs::remove_dir_all(&hot_cold.cache_dir);
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PagedKvHotColdConfig {
    pub root: PathBuf,
    pub hot_window_pages: i32,
    pub chunk_pages: i32,
    pub stream_cache_pages: usize,
}

impl PagedKvHotColdConfig {
    pub fn new(root: impl Into<PathBuf>, hot_window_pages: i32, chunk_pages: i32) -> Result<Self> {
        anyhow::ensure!(
            hot_window_pages > 0,
            "PagedKvHotColdConfig::new: hot_window_pages must be > 0"
        );
        anyhow::ensure!(
            chunk_pages > 0,
            "PagedKvHotColdConfig::new: chunk_pages must be > 0"
        );
        Ok(Self {
            root: root.into(),
            hot_window_pages,
            chunk_pages,
            stream_cache_pages: default_stream_cache_pages(hot_window_pages, chunk_pages),
        })
    }
}

fn default_stream_cache_pages(hot_window_pages: i32, chunk_pages: i32) -> usize {
    let base = hot_window_pages.max(chunk_pages).max(1);
    base.saturating_mul(16).clamp(1, 512) as usize
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PagedKvHotColdSummary {
    pub hot_window_pages: i32,
    pub configured_hot_window_pages: i32,
    pub resident_pages: usize,
    pub offloaded_pages: usize,
    pub loading_pages: usize,
    pub dirty_pages: usize,
    pub offloaded_bytes: usize,
    pub swap_out_count: u64,
    pub swap_in_count: u64,
    pub stream_read_count: u64,
    pub storage_dir: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum PagedKvPageState {
    ResidentClean {
        path: PathBuf,
        bytes: usize,
        start_page: i32,
        page_index: i32,
        page_count: i32,
    },
    Offloaded {
        path: PathBuf,
        bytes: usize,
        start_page: i32,
        page_index: i32,
        page_count: i32,
    },
    Loading {
        path: PathBuf,
        bytes: usize,
        start_page: i32,
        page_index: i32,
        page_count: i32,
    },
    Dirty,
}

#[derive(Debug, Clone)]
struct PagedKvPageSegment {
    path: PathBuf,
    bytes: usize,
    start_page: i32,
    page_index: i32,
    page_count: i32,
}

impl PagedKvPageSegment {
    fn new(path: PathBuf, bytes: usize, start_page: i32, page_index: i32, page_count: i32) -> Self {
        Self {
            path,
            bytes,
            start_page,
            page_index,
            page_count,
        }
    }
}

impl PagedKvPageState {
    fn segment_info(&self) -> Option<(&PathBuf, usize, i32, i32, i32)> {
        match self {
            Self::ResidentClean {
                path,
                bytes,
                start_page,
                page_index,
                page_count,
            }
            | Self::Offloaded {
                path,
                bytes,
                start_page,
                page_index,
                page_count,
            }
            | Self::Loading {
                path,
                bytes,
                start_page,
                page_index,
                page_count,
            } => Some((path, *bytes, *start_page, *page_index, *page_count)),
            Self::Dirty => None,
        }
    }

    fn cloned_segment_info(&self) -> Option<(PathBuf, usize, i32, i32, i32)> {
        self.segment_info()
            .map(|(path, bytes, start_page, page_index, page_count)| {
                (path.clone(), bytes, start_page, page_index, page_count)
            })
    }
}

#[derive(Debug)]
struct PagedKvHotColdTiering {
    cache_dir: PathBuf,
    hot_window_pages: i32,
    configured_hot_window_pages: i32,
    chunk_pages: i32,
    stream_cache_pages: usize,
    stream_cache: HashMap<i32, (Array, Array)>,
    stream_cache_lru: VecDeque<i32>,
    page_states: Vec<Option<PagedKvPageState>>,
    logical_to_slot: Vec<Option<i32>>,
    slot_to_logical: Vec<Option<i32>>,
    free_slots: Vec<i32>,
    swap_out_count: u64,
    swap_in_count: u64,
    stream_read_count: u64,
}

impl PagedKvHotColdTiering {
    fn new(config: PagedKvHotColdConfig) -> Result<Self> {
        let cache_dir = config
            .root
            .join(format!("cache-{}", uuid::Uuid::new_v4().simple()));
        fs::create_dir_all(&cache_dir)
            .with_context(|| format!("create active KV page cache dir {}", cache_dir.display()))?;
        Ok(Self {
            cache_dir,
            hot_window_pages: config.hot_window_pages,
            configured_hot_window_pages: config.hot_window_pages,
            chunk_pages: config.chunk_pages,
            stream_cache_pages: config.stream_cache_pages,
            stream_cache: HashMap::new(),
            stream_cache_lru: VecDeque::new(),
            page_states: Vec::new(),
            logical_to_slot: Vec::new(),
            slot_to_logical: Vec::new(),
            free_slots: Vec::new(),
            swap_out_count: 0,
            swap_in_count: 0,
            stream_read_count: 0,
        })
    }

    fn ensure_page(&mut self, page: i32) {
        let needed = page as usize + 1;
        if self.page_states.len() < needed {
            self.page_states.resize(needed, None);
        }
        if self.logical_to_slot.len() < needed {
            self.logical_to_slot.resize(needed, None);
        }
    }

    fn segment_path(&self, start_page: i32, page_count: i32) -> PathBuf {
        self.cache_dir.join(format!(
            "pages-{start_page}-{page_count}-{}.safetensors",
            uuid::Uuid::new_v4().simple()
        ))
    }

    fn slot_for(&self, page: i32) -> Option<i32> {
        self.logical_to_slot.get(page as usize).copied().flatten()
    }

    fn cached_stream_page(&mut self, page: i32) -> Option<(Array, Array)> {
        let pair = self.stream_cache.get(&page).cloned()?;
        self.touch_stream_cache_page(page);
        Some(pair)
    }

    fn cache_stream_page(&mut self, page: i32, k_page: &Array, v_page: &Array) {
        if self.stream_cache_pages == 0 {
            return;
        }
        if !self.stream_cache.contains_key(&page) {
            self.stream_cache_lru.push_back(page);
        }
        self.stream_cache
            .insert(page, (k_page.clone(), v_page.clone()));
        self.touch_stream_cache_page(page);
        while self.stream_cache.len() > self.stream_cache_pages {
            let Some(victim) = self.stream_cache_lru.pop_front() else {
                break;
            };
            if victim != page {
                self.stream_cache.remove(&victim);
            }
        }
    }

    fn invalidate_stream_cache_page(&mut self, page: i32) {
        self.stream_cache.remove(&page);
        self.stream_cache_lru.retain(|&cached| cached != page);
    }

    fn touch_stream_cache_page(&mut self, page: i32) {
        self.stream_cache_lru.retain(|&cached| cached != page);
        self.stream_cache_lru.push_back(page);
    }

    fn mark_resident(&mut self, page: i32, slot: i32, state: PagedKvPageState) {
        self.ensure_page(page);
        let needed = slot as usize + 1;
        if self.slot_to_logical.len() < needed {
            self.slot_to_logical.resize(needed, None);
        }
        if let Some(previous_page) = self.slot_to_logical[slot as usize] {
            if previous_page != page && (previous_page as usize) < self.logical_to_slot.len() {
                self.logical_to_slot[previous_page as usize] = None;
            }
        }
        self.free_slots.retain(|&free_slot| free_slot != slot);
        self.logical_to_slot[page as usize] = Some(slot);
        self.slot_to_logical[slot as usize] = Some(page);
        self.page_states[page as usize] = Some(state);
    }

    fn mark_offloaded(&mut self, page: i32, segment: PagedKvPageSegment) {
        let PagedKvPageSegment {
            path,
            bytes,
            start_page,
            page_index,
            page_count,
        } = segment;
        self.ensure_page(page);
        self.logical_to_slot[page as usize] = None;
        self.page_states[page as usize] = Some(PagedKvPageState::Offloaded {
            path,
            bytes,
            start_page,
            page_index,
            page_count,
        });
    }

    fn mark_resident_clean(&mut self, page: i32, slot: i32, segment: PagedKvPageSegment) {
        let PagedKvPageSegment {
            path,
            bytes,
            start_page,
            page_index,
            page_count,
        } = segment;
        self.mark_resident(
            page,
            slot,
            PagedKvPageState::ResidentClean {
                path,
                bytes,
                start_page,
                page_index,
                page_count,
            },
        );
    }

    fn mark_loading(&mut self, page: i32, segment: PagedKvPageSegment) {
        let PagedKvPageSegment {
            path,
            bytes,
            start_page,
            page_index,
            page_count,
        } = segment;
        self.ensure_page(page);
        self.logical_to_slot[page as usize] = None;
        self.page_states[page as usize] = Some(PagedKvPageState::Loading {
            path,
            bytes,
            start_page,
            page_index,
            page_count,
        });
    }

    fn mark_slot_free(&mut self, slot: i32) {
        if let Some(entry) = self.slot_to_logical.get_mut(slot as usize) {
            *entry = None;
        }
        if !self.free_slots.contains(&slot) {
            self.free_slots.push(slot);
        }
    }

    fn reset_runtime_state(&mut self) {
        self.stream_cache.clear();
        self.stream_cache_lru.clear();
        self.page_states.clear();
        self.logical_to_slot.clear();
        self.slot_to_logical.clear();
        self.free_slots.clear();
        self.swap_out_count = 0;
        self.swap_in_count = 0;
        self.stream_read_count = 0;
    }

    fn summary(&self) -> PagedKvHotColdSummary {
        let mut summary = PagedKvHotColdSummary {
            hot_window_pages: self.hot_window_pages,
            configured_hot_window_pages: self.configured_hot_window_pages,
            resident_pages: 0,
            offloaded_pages: 0,
            loading_pages: 0,
            dirty_pages: 0,
            offloaded_bytes: 0,
            swap_out_count: self.swap_out_count,
            swap_in_count: self.swap_in_count,
            stream_read_count: self.stream_read_count,
            storage_dir: self.cache_dir.clone(),
        };
        let mut counted_paths = HashSet::new();
        for state in self.page_states.iter().flatten() {
            match state {
                PagedKvPageState::ResidentClean { path, bytes, .. } => {
                    summary.resident_pages += 1;
                    if counted_paths.insert(path.clone()) {
                        summary.offloaded_bytes = summary.offloaded_bytes.saturating_add(*bytes);
                    }
                }
                PagedKvPageState::Dirty => {
                    summary.resident_pages += 1;
                    summary.dirty_pages += 1;
                }
                PagedKvPageState::Offloaded { path, bytes, .. } => {
                    summary.offloaded_pages += 1;
                    if counted_paths.insert(path.clone()) {
                        summary.offloaded_bytes = summary.offloaded_bytes.saturating_add(*bytes);
                    }
                }
                PagedKvPageState::Loading { path, bytes, .. } => {
                    summary.loading_pages += 1;
                    if counted_paths.insert(path.clone()) {
                        summary.offloaded_bytes = summary.offloaded_bytes.saturating_add(*bytes);
                    }
                }
            }
        }
        summary
    }
}

impl Drop for PagedKvHotColdTiering {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.cache_dir);
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
    use crate::core::generate::{build_batch_attention_mask, build_per_row_decode_mask};
    use mlx::{Array, Dtype};
    use std::{
        fs,
        time::{SystemTime, UNIX_EPOCH},
    };

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

    fn unique_test_dir(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock before epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("ironmlx-{label}-{}-{nanos}", std::process::id()))
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
        let stats = dst.physical_stats();
        assert_eq!(stats.adopt_page_copies, 2);
        assert_eq!(stats.orphan_pages, 0);
        dst.validate_owner_invariants().expect("owner invariants");
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn paged_kv_request_owners_survive_execution_row_rebind_and_release() {
        let mut cache =
            PagedKVCache::new(2, 1, 2, 2, Dtype::Float32, 8, 2, 8).expect("paged cache");
        let mut offsets = vec![0_i32, 0_i32];
        cache
            .bind_execution_rows(
                &[
                    PagedKvBlockOwner::Request(11),
                    PagedKvBlockOwner::Request(22),
                ],
                &mut offsets,
            )
            .expect("bind request owners");

        let k_data = [1.0_f32, 2.0, 3.0, 4.0, 11.0, 12.0, 13.0, 14.0];
        let v_data = [10.0_f32, 20.0, 30.0, 40.0, 110.0, 120.0, 130.0, 140.0];
        let k: Array = (&k_data[..], (2_i32, 1_i32, 2_i32, 2_i32))
            .try_into()
            .unwrap();
        let v: Array = (&v_data[..], (2_i32, 1_i32, 2_i32, 2_i32))
            .try_into()
            .unwrap();
        cache
            .update_and_fetch_on(&k, &v, &mut offsets, &[2, 2], ())
            .expect("append request rows");
        let request_11_page = cache.block_table_row(0)[0];
        let request_22_page = cache.block_table_row(1)[0];

        cache
            .bind_execution_rows(&[PagedKvBlockOwner::Request(22)], &mut offsets)
            .expect("compact to request 22");
        assert_eq!(offsets, vec![2]);
        assert_eq!(cache.block_table_row(0)[0], request_22_page);
        let compact_stats = cache.physical_stats();
        assert_eq!(compact_stats.request_owned_tables, 2);
        assert_eq!(compact_stats.physical_pages_referenced, 2);
        assert_eq!(compact_stats.adopt_page_copies, 0);
        assert_eq!(compact_stats.orphan_pages, 0);

        cache
            .bind_execution_rows(
                &[
                    PagedKvBlockOwner::Request(11),
                    PagedKvBlockOwner::Request(22),
                ],
                &mut offsets,
            )
            .expect("expand request views");
        assert_eq!(offsets, vec![2, 2]);
        assert_eq!(cache.block_table_row(0)[0], request_11_page);
        assert_eq!(cache.block_table_row(1)[0], request_22_page);

        assert!(cache
            .release_owner(PagedKvBlockOwner::Request(11), &mut offsets)
            .expect("release request 11"));
        assert!(!cache
            .release_owner(PagedKvBlockOwner::Request(11), &mut offsets)
            .expect("repeat release request 11"));
        assert_eq!(offsets, vec![0, 2]);
        let released_stats = cache.physical_stats();
        assert_eq!(released_stats.request_owned_tables, 1);
        assert_eq!(released_stats.physical_pages_referenced, 1);
        assert_eq!(released_stats.physical_pages_free, 1);
        assert_eq!(released_stats.owner_releases, 1);
        assert_eq!(released_stats.orphan_pages, 0);
        cache.validate_owner_invariants().expect("owner invariants");
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
    fn paged_kv_hot_cold_allows_direct_prefix_restore_when_empty_and_budget_fits() {
        let root = unique_test_dir("paged-kv-hot-cold-direct-restore");
        let mut empty =
            PagedKVCache::new(1, 1, 2, 2, Dtype::Float32, 8, 2, 8).expect("empty cache");
        empty
            .enable_hot_cold_tiering(
                PagedKvHotColdConfig::new(root.join("empty"), 4, 1).expect("hot/cold config"),
            )
            .expect("enable hot/cold tiering");
        assert!(empty.can_direct_install_prefix_pages(2));
        assert!(empty.can_direct_install_prefix_pages(5));
        assert!(!empty.can_direct_install_prefix_pages(6));

        let mut live = PagedKVCache::new(1, 1, 2, 2, Dtype::Float32, 8, 2, 8).expect("live cache");
        live.enable_hot_cold_tiering(
            PagedKvHotColdConfig::new(root.join("live"), 4, 1).expect("hot/cold config"),
        )
        .expect("enable hot/cold tiering");
        let k: Array = (&[1.0_f32, 2.0, 3.0, 4.0][..], (1_i32, 1_i32, 2_i32, 2_i32))
            .try_into()
            .unwrap();
        let v: Array = (
            &[10.0_f32, 20.0, 30.0, 40.0][..],
            (1_i32, 1_i32, 2_i32, 2_i32),
        )
            .try_into()
            .unwrap();
        let mut offsets = vec![0_i32];
        live.update_and_fetch_on(&k, &v, &mut offsets, &[2], ())
            .expect("append live page");
        assert!(!live.can_direct_install_prefix_pages(1));

        fs::remove_dir_all(&root).expect("remove test hot/cold root");
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn paged_kv_hot_cold_direct_prefix_restore_installs_cold_pages_without_swap_in() {
        let root = unique_test_dir("paged-kv-hot-cold-direct-cold-restore");
        let mut paged =
            PagedKVCache::new(1, 1, 2, 2, Dtype::Float32, 12, 2, 16).expect("paged cache");
        paged
            .enable_hot_cold_tiering(
                PagedKvHotColdConfig::new(&root, 1, 1).expect("hot/cold config"),
            )
            .expect("enable hot/cold tiering");

        let k_data: Vec<f32> = (0..(5 * 2 * 2))
            .map(|i| ((i % 17) as f32 - 8.0) * 0.013)
            .collect();
        let v_data: Vec<f32> = (0..(5 * 2 * 2))
            .map(|i| ((i % 19) as f32 - 9.0) * 0.017)
            .collect();
        let k_pages: Array = (k_data.as_slice(), (5_i32, 1_i32, 2_i32, 2_i32))
            .try_into()
            .unwrap();
        let v_pages: Array = (v_data.as_slice(), (5_i32, 1_i32, 2_i32, 2_i32))
            .try_into()
            .unwrap();
        let mut offsets = vec![0_i32];

        paged
            .restore_prefix_pages_for_row_on(&k_pages, &v_pages, &mut offsets, 0, 10, ())
            .expect("restore prefix into cold hot/cold cache");

        let summary = paged.hot_cold_summary().expect("hot/cold summary");
        assert_eq!(offsets, vec![10]);
        assert_eq!(summary.resident_pages, 2);
        assert_eq!(summary.offloaded_pages, 3);
        assert_eq!(
            summary.swap_in_count, 0,
            "direct cold prefix restore should not load offloaded pages back into resident slots"
        );
        assert!(
            summary.swap_out_count <= summary.offloaded_pages as u64,
            "direct cold prefix restore should not evict the same logical prefix repeatedly: {summary:?}"
        );

        let (k_ref, v_ref) = paged.materialize_prefix_on(&offsets, 10, ()).expect("ref");
        assert_close(
            &k_ref.to_vec::<f32>().unwrap(),
            &k_pages.to_vec::<f32>().unwrap(),
            1.0e-6,
        );
        assert_close(
            &v_ref.to_vec::<f32>().unwrap(),
            &v_pages.to_vec::<f32>().unwrap(),
            1.0e-6,
        );

        drop(paged);
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn paged_kv_hot_cold_direct_prefix_restore_preserves_shared_and_private_tail_pages() {
        let root = unique_test_dir("paged-kv-hot-cold-direct-cold-restore-tail");
        let mut paged =
            PagedKVCache::new(2, 1, 2, 2, Dtype::Float32, 12, 2, 16).expect("paged cache");
        paged
            .enable_hot_cold_tiering(
                PagedKvHotColdConfig::new(&root, 1, 1).expect("hot/cold config"),
            )
            .expect("enable hot/cold tiering");

        let k_data: Vec<f32> = (0..(5 * 2 * 2)).map(|i| i as f32 * 0.031).collect();
        let v_data: Vec<f32> = (0..(5 * 2 * 2)).map(|i| i as f32 * 0.047).collect();
        let k_pages: Array = (k_data.as_slice(), (5_i32, 1_i32, 2_i32, 2_i32))
            .try_into()
            .unwrap();
        let v_pages: Array = (v_data.as_slice(), (5_i32, 1_i32, 2_i32, 2_i32))
            .try_into()
            .unwrap();
        let mut offsets = vec![0_i32, 0_i32];

        paged
            .restore_prefix_pages_for_rows_on(&k_pages, &v_pages, &mut offsets, &[0, 1], 9, ())
            .expect("restore shared prefix with private tails");

        let summary = paged.hot_cold_summary().expect("hot/cold summary");
        assert_eq!(offsets, vec![9, 9]);
        assert_eq!(paged.block_table_row(0)[..5], [0, 1, 2, 3, 4]);
        assert_eq!(paged.block_table_row(1)[..5], [0, 1, 2, 3, 5]);
        assert!(
            summary.offloaded_pages > 0,
            "cold direct restore should keep older shared prefix pages offloaded: {summary:?}"
        );
        assert_eq!(
            summary.swap_in_count, 0,
            "direct cold restore should not stage cold shared prefix pages through resident slots: {summary:?}"
        );

        let (k_ref, v_ref) = paged
            .materialize_prefix_on(&offsets, 9, ())
            .expect("materialize restored rows");
        let expected_k = &k_data[..18];
        let expected_v = &v_data[..18];
        let actual_k = k_ref.to_vec::<f32>().unwrap();
        let actual_v = v_ref.to_vec::<f32>().unwrap();
        assert_close(&actual_k[..18], expected_k, 1.0e-6);
        assert_close(&actual_k[18..36], expected_k, 1.0e-6);
        assert_close(&actual_v[..18], expected_v, 1.0e-6);
        assert_close(&actual_v[18..36], expected_v, 1.0e-6);

        drop(paged);
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn paged_kv_hot_cold_drop_removes_segment_cache_dir() {
        let root = unique_test_dir("paged-kv-hot-cold-drop-cleanup");
        let storage_dir = {
            let mut paged =
                PagedKVCache::new(1, 1, 4, 4, Dtype::Float32, 8, 2, 8).expect("paged cache");
            paged
                .enable_hot_cold_tiering(
                    PagedKvHotColdConfig::new(&root, 1, 1).expect("hot/cold config"),
                )
                .expect("enable hot/cold tiering");
            let storage_dir = paged
                .hot_cold_summary()
                .expect("hot/cold summary")
                .storage_dir;
            assert!(storage_dir.exists());
            storage_dir
        };

        assert!(
            !storage_dir.exists(),
            "dropping a hot/cold cache should remove its temporary segment directory: {}",
            storage_dir.display()
        );
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn paged_kv_pressure_shrink_demotes_resident_pages_idempotently() {
        let root = unique_test_dir("paged-kv-pressure-shrink");
        let mut paged =
            PagedKVCache::new(1, 1, 2, 2, Dtype::Float32, 8, 2, 8).expect("paged cache");
        paged
            .enable_hot_cold_tiering(
                PagedKvHotColdConfig::new(&root, 4, 1).expect("hot/cold config"),
            )
            .expect("enable hot/cold tiering");
        let data = vec![1.0_f32; 16];
        let k: Array = (data.as_slice(), (1_i32, 1_i32, 8_i32, 2_i32))
            .try_into()
            .unwrap();
        let v = k.clone();
        let mut offsets = vec![0_i32];
        paged
            .update_and_fetch_on(&k, &v, &mut offsets, &[8], ())
            .expect("populate hot cache");
        assert_eq!(paged.hot_cold_summary().unwrap().resident_pages, 4);

        let reclaimed = paged
            .shrink_hot_window_on(&offsets, 1, ())
            .expect("pressure shrink");
        assert_eq!(reclaimed, 3);
        let summary = paged.hot_cold_summary().unwrap();
        assert_eq!(summary.resident_pages, 1);
        assert_eq!(summary.offloaded_pages, 3);
        assert_eq!(summary.hot_window_pages, 1);
        assert_eq!(summary.configured_hot_window_pages, 4);
        assert_eq!(
            paged
                .shrink_hot_window_on(&offsets, 1, ())
                .expect("repeated pressure shrink"),
            0
        );
        assert!(paged.restore_configured_hot_window());
        assert_eq!(paged.hot_cold_summary().unwrap().hot_window_pages, 4);
        assert!(!paged.restore_configured_hot_window());

        drop(paged);
        fs::remove_dir_all(root).ok();
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn paged_kv_hot_cold_clear_resets_stream_cache_and_counters() {
        let root = unique_test_dir("paged-kv-hot-cold-clear-reset");
        let mut paged =
            PagedKVCache::new(1, 1, 4, 4, Dtype::Float32, 8, 2, 8).expect("paged cache");
        paged
            .enable_hot_cold_tiering(
                PagedKvHotColdConfig::new(&root, 1, 1).expect("hot/cold config"),
            )
            .expect("enable hot/cold tiering");

        let q_data: Vec<f32> = (0..(2 * 5 * 4))
            .map(|i| ((i % 19) as f32 - 9.0) * 0.017)
            .collect();
        let k_data: Vec<f32> = (0..(5 * 4))
            .map(|i| ((i % 23) as f32 - 11.0) * 0.021)
            .collect();
        let v_data: Vec<f32> = (0..(5 * 4))
            .map(|i| ((i % 29) as f32 - 14.0) * 0.019)
            .collect();
        let q: Array = (q_data.as_slice(), (1_i32, 2_i32, 5_i32, 4_i32))
            .try_into()
            .unwrap();
        let k: Array = (k_data.as_slice(), (1_i32, 1_i32, 5_i32, 4_i32))
            .try_into()
            .unwrap();
        let v: Array = (v_data.as_slice(), (1_i32, 1_i32, 5_i32, 4_i32))
            .try_into()
            .unwrap();
        let mut offsets = vec![0_i32];
        paged
            .update_and_attend_prefill_on(&q, &k, &v, &mut offsets, &[5], 0.5, None, ())
            .expect("hot/cold streaming prefill");
        let before_clear = paged.hot_cold_summary().expect("hot/cold summary");
        assert!(before_clear.swap_out_count > 0);
        assert!(before_clear.stream_read_count > 0);
        assert!(!paged
            .hot_cold
            .as_ref()
            .expect("hot/cold tiering")
            .stream_cache
            .is_empty());

        paged.clear();

        let after_clear = paged.hot_cold_summary().expect("hot/cold summary");
        assert_eq!(after_clear.resident_pages, 0);
        assert_eq!(after_clear.offloaded_pages, 0);
        assert_eq!(after_clear.loading_pages, 0);
        assert_eq!(after_clear.dirty_pages, 0);
        assert_eq!(after_clear.offloaded_bytes, 0);
        assert_eq!(after_clear.swap_out_count, 0);
        assert_eq!(after_clear.swap_in_count, 0);
        assert_eq!(after_clear.stream_read_count, 0);
        let hot_cold = paged.hot_cold.as_ref().expect("hot/cold tiering");
        assert!(hot_cold.stream_cache.is_empty());
        assert!(hot_cold.stream_cache_lru.is_empty());

        drop(paged);
        fs::remove_dir_all(&root).ok();
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

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn paged_kv_hot_cold_streaming_decode_streams_cold_pages_without_resident_swap_in() {
        let root = unique_test_dir("paged-kv-hot-cold");
        let mut paged =
            PagedKVCache::new(2, 1, 4, 4, Dtype::Float32, 8, 2, 8).expect("paged cache");
        paged
            .enable_hot_cold_tiering(
                PagedKvHotColdConfig::new(&root, 1, 1).expect("hot/cold config"),
            )
            .expect("enable hot/cold tiering");

        let prefix_k_data: Vec<f32> = (0..(2 * 1 * 4 * 4))
            .map(|i| ((i % 29) as f32 - 14.0) * 0.027)
            .collect();
        let prefix_v_data: Vec<f32> = (0..(2 * 1 * 4 * 4))
            .map(|i| ((i % 31) as f32 - 15.0) * 0.021)
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
        let prefix_summary = paged.hot_cold_summary().expect("hot/cold summary");
        assert!(
            prefix_summary.offloaded_pages > 0,
            "expected prefix append to offload at least one page: {prefix_summary:?}"
        );
        assert!(
            prefix_summary.resident_pages <= 2,
            "hot window should bound resident pages: {prefix_summary:?}"
        );

        let q_data: Vec<f32> = (0..(2 * 2 * 4))
            .map(|i| ((i % 17) as f32 - 8.0) * 0.019)
            .collect();
        let step_k_data: Vec<f32> = (0..(2 * 1 * 1 * 4))
            .map(|i| ((i % 13) as f32 - 6.0) * 0.037)
            .collect();
        let step_v_data: Vec<f32> = (0..(2 * 1 * 1 * 4))
            .map(|i| ((i % 11) as f32 - 5.0) * 0.043)
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
            .expect("hot/cold streaming decode");
        assert_eq!(offsets, vec![5, 3]);
        let decode_summary = paged.hot_cold_summary().expect("hot/cold summary");
        assert!(
            decode_summary.offloaded_pages >= prefix_summary.offloaded_pages,
            "decode should retain cold pages offloaded: {decode_summary:?}"
        );
        assert!(
            decode_summary.resident_pages <= 4,
            "decode should respect hot window plus staging budget: {decode_summary:?}"
        );
        assert_eq!(
            decode_summary.swap_in_count, prefix_summary.swap_in_count,
            "streaming decode should read immutable cold pages without promoting them to resident slots"
        );
        assert!(
            decode_summary.stream_read_count > prefix_summary.stream_read_count,
            "decode should account read-only cold page streaming: before={prefix_summary:?} after={decode_summary:?}"
        );

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
            1.0e-3,
        );

        drop(paged);
        fs::remove_dir_all(&root).expect("remove test hot/cold root");
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn paged_kv_hot_cold_streaming_decode_reuses_cached_cold_pages() {
        let root = unique_test_dir("paged-kv-hot-cold-stream-cache");
        let mut paged =
            PagedKVCache::new(1, 1, 4, 4, Dtype::Float32, 8, 2, 16).expect("paged cache");
        paged
            .enable_hot_cold_tiering(
                PagedKvHotColdConfig::new(&root, 1, 1).expect("hot/cold config"),
            )
            .expect("enable hot/cold tiering");

        let prefix_k_data: Vec<f32> = (0..(1 * 1 * 6 * 4))
            .map(|i| ((i % 29) as f32 - 14.0) * 0.027)
            .collect();
        let prefix_v_data: Vec<f32> = (0..(1 * 1 * 6 * 4))
            .map(|i| ((i % 31) as f32 - 15.0) * 0.021)
            .collect();
        let prefix_k: Array = (prefix_k_data.as_slice(), (1_i32, 1_i32, 6_i32, 4_i32))
            .try_into()
            .unwrap();
        let prefix_v: Array = (prefix_v_data.as_slice(), (1_i32, 1_i32, 6_i32, 4_i32))
            .try_into()
            .unwrap();
        let mut offsets = vec![0_i32];
        paged
            .update_and_fetch_on(&prefix_k, &prefix_v, &mut offsets, &[6], ())
            .expect("prefix append");

        let mut stream_counts = Vec::new();
        for step in 0..2 {
            let q_data: Vec<f32> = (0..(1 * 2 * 1 * 4))
                .map(|i| (((i + step * 7) % 17) as f32 - 8.0) * 0.019)
                .collect();
            let step_k_data: Vec<f32> = (0..(1 * 1 * 1 * 4))
                .map(|i| (((i + step * 5) % 13) as f32 - 6.0) * 0.037)
                .collect();
            let step_v_data: Vec<f32> = (0..(1 * 1 * 1 * 4))
                .map(|i| (((i + step * 3) % 11) as f32 - 5.0) * 0.043)
                .collect();
            let q: Array = (q_data.as_slice(), (1_i32, 2_i32, 1_i32, 4_i32))
                .try_into()
                .unwrap();
            let step_k: Array = (step_k_data.as_slice(), (1_i32, 1_i32, 1_i32, 4_i32))
                .try_into()
                .unwrap();
            let step_v: Array = (step_v_data.as_slice(), (1_i32, 1_i32, 1_i32, 4_i32))
                .try_into()
                .unwrap();
            let actual = paged
                .update_and_attend_decode_on(&q, &step_k, &step_v, &mut offsets, &[1], 0.5, ())
                .expect("hot/cold streaming decode");
            let (k_ref, v_ref) = paged
                .materialize_prefix_on(&offsets, offsets[0], ())
                .expect("ref");
            let mask =
                build_per_row_decode_mask(&offsets, offsets[0], Dtype::Float32).expect("mask");
            let expected = mlx::fast::scaled_dot_product_attention(
                &q,
                &k_ref,
                &v_ref,
                0.5,
                "",
                Some(&mask),
                None,
            )
            .expect("dense sdpa");
            assert_close(
                &actual.to_vec::<f32>().unwrap(),
                &expected.to_vec::<f32>().unwrap(),
                1.0e-4,
            );
            stream_counts.push(
                paged
                    .hot_cold_summary()
                    .expect("hot/cold summary")
                    .stream_read_count,
            );
        }

        let [after_first_streams, after_second_streams]: [u64; 2] =
            stream_counts.try_into().expect("two stream counts");
        assert!(
            after_first_streams > 0,
            "first decode should stream cold pages"
        );
        assert!(
            after_second_streams <= after_first_streams + 1,
            "second decode may read the page newly offloaded after the first step, but should reuse older cached immutable cold pages: first={after_first_streams} second={after_second_streams}"
        );

        drop(paged);
        fs::remove_dir_all(&root).expect("remove test hot/cold root");
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn paged_kv_hot_cold_streaming_decode_loads_offloaded_segments_once() {
        let root = unique_test_dir("paged-kv-hot-cold-segment-cache");
        let mut paged =
            PagedKVCache::new(1, 1, 4, 4, Dtype::Float32, 12, 2, 16).expect("paged cache");
        paged
            .enable_hot_cold_tiering(
                PagedKvHotColdConfig::new(&root, 1, 4).expect("hot/cold config"),
            )
            .expect("enable hot/cold tiering");

        let prefix_len = 10_i32;
        let prefix_k_data: Vec<f32> = (0..(prefix_len * 4))
            .map(|i| ((i % 29) as f32 - 14.0) * 0.027)
            .collect();
        let prefix_v_data: Vec<f32> = (0..(prefix_len * 4))
            .map(|i| ((i % 31) as f32 - 15.0) * 0.021)
            .collect();
        let prefix_k: Array = (prefix_k_data.as_slice(), (1_i32, 1_i32, prefix_len, 4_i32))
            .try_into()
            .unwrap();
        let prefix_v: Array = (prefix_v_data.as_slice(), (1_i32, 1_i32, prefix_len, 4_i32))
            .try_into()
            .unwrap();
        let mut offsets = vec![0_i32];
        paged
            .update_and_fetch_on(&prefix_k, &prefix_v, &mut offsets, &[prefix_len], ())
            .expect("prefix append");
        let prefix_summary = paged.hot_cold_summary().expect("hot/cold summary");
        assert_eq!(
            prefix_summary.offloaded_pages, 4,
            "prefix should offload the four cold pages as one chunk-sized run: {prefix_summary:?}"
        );

        let q_data: Vec<f32> = (0..(1 * 2 * 1 * 4))
            .map(|i| ((i % 17) as f32 - 8.0) * 0.019)
            .collect();
        let step_k_data: Vec<f32> = (0..(1 * 1 * 1 * 4))
            .map(|i| ((i % 13) as f32 - 6.0) * 0.037)
            .collect();
        let step_v_data: Vec<f32> = (0..(1 * 1 * 1 * 4))
            .map(|i| ((i % 11) as f32 - 5.0) * 0.043)
            .collect();
        let q: Array = (q_data.as_slice(), (1_i32, 2_i32, 1_i32, 4_i32))
            .try_into()
            .unwrap();
        let step_k: Array = (step_k_data.as_slice(), (1_i32, 1_i32, 1_i32, 4_i32))
            .try_into()
            .unwrap();
        let step_v: Array = (step_v_data.as_slice(), (1_i32, 1_i32, 1_i32, 4_i32))
            .try_into()
            .unwrap();
        let actual = paged
            .update_and_attend_decode_on(&q, &step_k, &step_v, &mut offsets, &[1], 0.5, ())
            .expect("hot/cold streaming decode");
        let summary = paged.hot_cold_summary().expect("hot/cold summary");
        assert_eq!(
            summary.stream_read_count, 1,
            "chunk-sized offloaded runs should be loaded once and sliced from memory: {summary:?}"
        );

        let (k_ref, v_ref) = paged
            .materialize_prefix_on(&offsets, offsets[0], ())
            .expect("ref");
        let mask = build_per_row_decode_mask(&offsets, offsets[0], Dtype::Float32).expect("mask");
        let expected =
            mlx::fast::scaled_dot_product_attention(&q, &k_ref, &v_ref, 0.5, "", Some(&mask), None)
                .expect("dense sdpa");
        assert_close(
            &actual.to_vec::<f32>().unwrap(),
            &expected.to_vec::<f32>().unwrap(),
            1.0e-4,
        );

        drop(paged);
        fs::remove_dir_all(&root).expect("remove test hot/cold root");
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn paged_kv_hot_cold_decode_stages_context_to_paged_kernel_when_budget_allows() {
        let root = unique_test_dir("paged-kv-hot-cold-staged-paged");
        let mut paged =
            PagedKVCache::new(1, 1, 4, 4, Dtype::Float32, 12, 2, 16).expect("paged cache");
        paged
            .enable_hot_cold_tiering(
                PagedKvHotColdConfig::new(&root, 1, 8).expect("hot/cold config"),
            )
            .expect("enable hot/cold tiering");

        let prefix_len = 10_i32;
        let prefix_k_data: Vec<f32> = (0..(prefix_len * 4))
            .map(|i| ((i % 29) as f32 - 14.0) * 0.027)
            .collect();
        let prefix_v_data: Vec<f32> = (0..(prefix_len * 4))
            .map(|i| ((i % 31) as f32 - 15.0) * 0.021)
            .collect();
        let prefix_k: Array = (prefix_k_data.as_slice(), (1_i32, 1_i32, prefix_len, 4_i32))
            .try_into()
            .unwrap();
        let prefix_v: Array = (prefix_v_data.as_slice(), (1_i32, 1_i32, prefix_len, 4_i32))
            .try_into()
            .unwrap();
        let mut offsets = vec![0_i32];
        paged
            .update_and_fetch_on(&prefix_k, &prefix_v, &mut offsets, &[prefix_len], ())
            .expect("prefix append");
        let prefix_summary = paged.hot_cold_summary().expect("hot/cold summary");
        assert!(
            prefix_summary.offloaded_pages > 0,
            "prefix should create cold pages: {prefix_summary:?}"
        );

        let q_data: Vec<f32> = (0..(1 * 2 * 1 * 4))
            .map(|i| ((i % 17) as f32 - 8.0) * 0.019)
            .collect();
        let step_k_data: Vec<f32> = (0..(1 * 1 * 1 * 4))
            .map(|i| ((i % 13) as f32 - 6.0) * 0.037)
            .collect();
        let step_v_data: Vec<f32> = (0..(1 * 1 * 1 * 4))
            .map(|i| ((i % 11) as f32 - 5.0) * 0.043)
            .collect();
        let q: Array = (q_data.as_slice(), (1_i32, 2_i32, 1_i32, 4_i32))
            .try_into()
            .unwrap();
        let step_k: Array = (step_k_data.as_slice(), (1_i32, 1_i32, 1_i32, 4_i32))
            .try_into()
            .unwrap();
        let step_v: Array = (step_v_data.as_slice(), (1_i32, 1_i32, 1_i32, 4_i32))
            .try_into()
            .unwrap();

        let actual = paged
            .update_and_attend_decode_on(&q, &step_k, &step_v, &mut offsets, &[1], 0.5, ())
            .expect("staged paged decode");
        let decode_summary = paged.hot_cold_summary().expect("hot/cold summary");
        assert_eq!(
            decode_summary.stream_read_count, prefix_summary.stream_read_count,
            "when staging budget covers the decode context, hot/cold should use paged decode instead of streaming row chunks: before={prefix_summary:?} after={decode_summary:?}"
        );

        let (k_ref, v_ref) = paged
            .materialize_prefix_on(&offsets, offsets[0], ())
            .expect("ref");
        let mask = build_per_row_decode_mask(&offsets, offsets[0], Dtype::Float32).expect("mask");
        let expected =
            mlx::fast::scaled_dot_product_attention(&q, &k_ref, &v_ref, 0.5, "", Some(&mask), None)
                .expect("dense sdpa");
        assert_close(
            &actual.to_vec::<f32>().unwrap(),
            &expected.to_vec::<f32>().unwrap(),
            1.0e-4,
        );

        drop(paged);
        fs::remove_dir_all(&root).expect("remove test hot/cold root");
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn paged_kv_hot_cold_decode_keeps_staged_clean_pages_when_budget_allows() {
        let root = unique_test_dir("paged-kv-hot-cold-staged-retention");
        let mut paged =
            PagedKVCache::new(1, 1, 4, 4, Dtype::Float32, 16, 2, 16).expect("paged cache");
        paged
            .enable_hot_cold_tiering(
                PagedKvHotColdConfig::new(&root, 1, 8).expect("hot/cold config"),
            )
            .expect("enable hot/cold tiering");

        let prefix_len = 10_i32;
        let prefix_k_data: Vec<f32> = (0..(prefix_len * 4))
            .map(|i| ((i % 29) as f32 - 14.0) * 0.027)
            .collect();
        let prefix_v_data: Vec<f32> = (0..(prefix_len * 4))
            .map(|i| ((i % 31) as f32 - 15.0) * 0.021)
            .collect();
        let prefix_k: Array = (prefix_k_data.as_slice(), (1_i32, 1_i32, prefix_len, 4_i32))
            .try_into()
            .unwrap();
        let prefix_v: Array = (prefix_v_data.as_slice(), (1_i32, 1_i32, prefix_len, 4_i32))
            .try_into()
            .unwrap();
        let mut offsets = vec![0_i32];
        paged
            .update_and_fetch_on(&prefix_k, &prefix_v, &mut offsets, &[prefix_len], ())
            .expect("prefix append");

        let q_data: Vec<f32> = (0..(1 * 2 * 1 * 4))
            .map(|i| ((i % 17) as f32 - 8.0) * 0.019)
            .collect();
        let q: Array = (q_data.as_slice(), (1_i32, 2_i32, 1_i32, 4_i32))
            .try_into()
            .unwrap();

        for step in 0..2 {
            let step_k_data: Vec<f32> = (0..4)
                .map(|i| ((i + step * 4) as f32 % 13.0 - 6.0) * 0.037)
                .collect();
            let step_v_data: Vec<f32> = (0..4)
                .map(|i| ((i + step * 4) as f32 % 11.0 - 5.0) * 0.043)
                .collect();
            let step_k: Array = (step_k_data.as_slice(), (1_i32, 1_i32, 1_i32, 4_i32))
                .try_into()
                .unwrap();
            let step_v: Array = (step_v_data.as_slice(), (1_i32, 1_i32, 1_i32, 4_i32))
                .try_into()
                .unwrap();
            paged
                .update_and_attend_decode_on(&q, &step_k, &step_v, &mut offsets, &[1], 0.5, ())
                .expect("staged paged decode");
        }

        let after_first_two_decodes = paged.hot_cold_summary().expect("hot/cold summary");
        let swap_in_after_two_decodes = after_first_two_decodes.swap_in_count;

        let step_k: Array = (
            [0.031_f32, -0.017, 0.011, 0.023].as_slice(),
            (1_i32, 1_i32, 1_i32, 4_i32),
        )
            .try_into()
            .unwrap();
        let step_v: Array = (
            [0.019_f32, 0.007, -0.029, 0.013].as_slice(),
            (1_i32, 1_i32, 1_i32, 4_i32),
        )
            .try_into()
            .unwrap();
        paged
            .update_and_attend_decode_on(&q, &step_k, &step_v, &mut offsets, &[1], 0.5, ())
            .expect("staged paged decode");

        let after_third_decode = paged.hot_cold_summary().expect("hot/cold summary");
        assert_eq!(
            after_third_decode.swap_in_count, swap_in_after_two_decodes,
            "budget-sized staged clean pages should remain resident across decode steps instead of being reloaded every token: before={after_first_two_decodes:?} after={after_third_decode:?}"
        );
        assert_eq!(
            after_third_decode.stream_read_count, 0,
            "budget-sized staged decode should keep using paged kernel rather than streaming chunks"
        );

        drop(paged);
        fs::remove_dir_all(&root).expect("remove test hot/cold root");
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn paged_kv_hot_cold_decode_uses_resident_paged_kernel_when_window_covers_context() {
        let root = unique_test_dir("paged-kv-hot-cold-resident");
        let mut paged =
            PagedKVCache::new(2, 1, 4, 4, Dtype::Float32, 8, 2, 8).expect("paged cache");
        paged
            .enable_hot_cold_tiering(
                PagedKvHotColdConfig::new(&root, 4, 1).expect("hot/cold config"),
            )
            .expect("enable hot/cold tiering");

        let prefix_k_data: Vec<f32> = (0..(2 * 1 * 4 * 4))
            .map(|i| ((i % 29) as f32 - 14.0) * 0.027)
            .collect();
        let prefix_v_data: Vec<f32> = (0..(2 * 1 * 4 * 4))
            .map(|i| ((i % 31) as f32 - 15.0) * 0.021)
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
        let prefix_summary = paged.hot_cold_summary().expect("hot/cold summary");
        assert_eq!(prefix_summary.offloaded_pages, 0);
        assert_eq!(prefix_summary.stream_read_count, 0);

        let q_data: Vec<f32> = (0..(2 * 2 * 4))
            .map(|i| ((i % 17) as f32 - 8.0) * 0.019)
            .collect();
        let step_k_data: Vec<f32> = (0..(2 * 1 * 1 * 4))
            .map(|i| ((i % 13) as f32 - 6.0) * 0.037)
            .collect();
        let step_v_data: Vec<f32> = (0..(2 * 1 * 1 * 4))
            .map(|i| ((i % 11) as f32 - 5.0) * 0.043)
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

        let actual = paged
            .update_and_attend_decode_on(&q, &step_k, &step_v, &mut offsets, &[1, 1], 0.5, ())
            .expect("resident paged decode");
        assert_eq!(offsets, vec![5, 3]);
        let decode_summary = paged.hot_cold_summary().expect("hot/cold summary");
        assert_eq!(decode_summary.offloaded_pages, 0);
        assert_eq!(
            decode_summary.stream_read_count, 0,
            "resident context should use the normal paged decode kernel instead of streaming"
        );

        let (k_ref, v_ref) = paged.materialize_prefix_on(&offsets, 5, ()).expect("ref");
        let mask = build_per_row_decode_mask(&offsets, 5, Dtype::Float32).expect("mask");
        let expected =
            mlx::fast::scaled_dot_product_attention(&q, &k_ref, &v_ref, 0.5, "", Some(&mask), None)
                .expect("dense sdpa");

        assert_eq!(actual.shape().as_slice(), &[2, 2, 1, 4]);
        assert_close(
            &actual.to_vec::<f32>().unwrap(),
            &expected.to_vec::<f32>().unwrap(),
            1.0e-4,
        );

        drop(paged);
        fs::remove_dir_all(&root).expect("remove test hot/cold root");
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn paged_kv_hot_cold_prefill_streaming_matches_dense_causal_sdpa() {
        let root = unique_test_dir("paged-kv-hot-cold-prefill");
        let mut paged =
            PagedKVCache::new(1, 1, 4, 4, Dtype::Float32, 8, 2, 8).expect("paged cache");
        paged
            .enable_hot_cold_tiering(
                PagedKvHotColdConfig::new(&root, 1, 1).expect("hot/cold config"),
            )
            .expect("enable hot/cold tiering");

        let q_data: Vec<f32> = (0..(2 * 5 * 4))
            .map(|i| ((i % 19) as f32 - 9.0) * 0.017)
            .collect();
        let k_data: Vec<f32> = (0..(5 * 4))
            .map(|i| ((i % 23) as f32 - 11.0) * 0.021)
            .collect();
        let v_data: Vec<f32> = (0..(5 * 4))
            .map(|i| ((i % 29) as f32 - 14.0) * 0.019)
            .collect();
        let q: Array = (q_data.as_slice(), (1_i32, 2_i32, 5_i32, 4_i32))
            .try_into()
            .unwrap();
        let k: Array = (k_data.as_slice(), (1_i32, 1_i32, 5_i32, 4_i32))
            .try_into()
            .unwrap();
        let v: Array = (v_data.as_slice(), (1_i32, 1_i32, 5_i32, 4_i32))
            .try_into()
            .unwrap();
        let mut offsets = vec![0_i32];
        let scale = 0.5_f32;

        let actual = paged
            .update_and_attend_prefill_on(&q, &k, &v, &mut offsets, &[5], scale, None, ())
            .expect("hot/cold streaming prefill");
        assert_eq!(offsets, vec![5]);
        let summary = paged.hot_cold_summary().expect("hot/cold summary");
        assert!(
            summary.offloaded_pages > 0,
            "prefill should offload cold pages: {summary:?}"
        );
        assert!(
            summary.stream_read_count > 0,
            "prefill should stream cold pages: {summary:?}"
        );

        let (k_ref, v_ref) = paged.materialize_prefix_on(&offsets, 5, ()).expect("ref");
        let expected = mlx::fast::scaled_dot_product_attention(
            &q, &k_ref, &v_ref, scale, "causal", None, None,
        )
        .expect("dense sdpa");
        assert_eq!(actual.shape().as_slice(), &[1, 2, 5, 4]);
        assert_close(
            &actual.to_vec::<f32>().unwrap(),
            &expected.to_vec::<f32>().unwrap(),
            1.0e-4,
        );

        drop(paged);
        fs::remove_dir_all(&root).expect("remove test hot/cold root");
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn paged_kv_hot_cold_prefill_keeps_resident_when_budget_allows() {
        let root = unique_test_dir("paged-kv-hot-cold-prefill-resident-budget");
        let mut paged =
            PagedKVCache::new(1, 1, 4, 4, Dtype::Float32, 8, 2, 8).expect("paged cache");
        paged
            .enable_hot_cold_tiering(
                PagedKvHotColdConfig::new(&root, 1, 7).expect("hot/cold config"),
            )
            .expect("enable hot/cold tiering");

        let q_data: Vec<f32> = (0..(2 * 5 * 4))
            .map(|i| ((i % 19) as f32 - 9.0) * 0.017)
            .collect();
        let k_data: Vec<f32> = (0..(5 * 4))
            .map(|i| ((i % 23) as f32 - 11.0) * 0.021)
            .collect();
        let v_data: Vec<f32> = (0..(5 * 4))
            .map(|i| ((i % 29) as f32 - 14.0) * 0.019)
            .collect();
        let q: Array = (q_data.as_slice(), (1_i32, 2_i32, 5_i32, 4_i32))
            .try_into()
            .unwrap();
        let k: Array = (k_data.as_slice(), (1_i32, 1_i32, 5_i32, 4_i32))
            .try_into()
            .unwrap();
        let v: Array = (v_data.as_slice(), (1_i32, 1_i32, 5_i32, 4_i32))
            .try_into()
            .unwrap();
        let mut offsets = vec![0_i32];
        let scale = 0.5_f32;

        let actual = paged
            .update_and_attend_prefill_on(&q, &k, &v, &mut offsets, &[5], scale, None, ())
            .expect("hot/cold prefill");
        assert_eq!(offsets, vec![5]);

        let summary = paged.hot_cold_summary().expect("hot/cold summary");
        assert_eq!(
            summary.offloaded_pages, 0,
            "prefill should keep all resident pages when the staging budget covers the context: {summary:?}"
        );
        assert_eq!(
            summary.swap_out_count, 0,
            "prefill should not write cold pages to SSD when resident budget is sufficient: {summary:?}"
        );
        assert_eq!(
            summary.stream_read_count, 0,
            "resident prefill should not read cold pages from SSD: {summary:?}"
        );

        let (k_ref, v_ref) = paged.materialize_prefix_on(&offsets, 5, ()).expect("ref");
        let expected = mlx::fast::scaled_dot_product_attention(
            &q, &k_ref, &v_ref, scale, "causal", None, None,
        )
        .expect("dense sdpa");
        assert_eq!(actual.shape().as_slice(), &[1, 2, 5, 4]);
        assert_close(
            &actual.to_vec::<f32>().unwrap(),
            &expected.to_vec::<f32>().unwrap(),
            1.0e-4,
        );

        drop(paged);
        fs::remove_dir_all(&root).expect("remove test hot/cold root");
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn paged_kv_hot_cold_prefill_streaming_reads_cold_pages_once_per_row() {
        let root = unique_test_dir("paged-kv-hot-cold-prefill-read-amp");
        let mut paged =
            PagedKVCache::new(1, 1, 4, 4, Dtype::Float32, 132, 4, 64).expect("paged cache");
        paged
            .enable_hot_cold_tiering(
                PagedKvHotColdConfig::new(&root, 2, 4).expect("hot/cold config"),
            )
            .expect("enable hot/cold tiering");

        let q_len = 130_i32;
        let q_data: Vec<f32> = (0..(2 * q_len * 4))
            .map(|i| ((i % 31) as f32 - 15.0) * 0.009)
            .collect();
        let k_data: Vec<f32> = (0..(q_len * 4))
            .map(|i| ((i % 37) as f32 - 18.0) * 0.011)
            .collect();
        let v_data: Vec<f32> = (0..(q_len * 4))
            .map(|i| ((i % 41) as f32 - 20.0) * 0.013)
            .collect();
        let q: Array = (q_data.as_slice(), (1_i32, 2_i32, q_len, 4_i32))
            .try_into()
            .unwrap();
        let k: Array = (k_data.as_slice(), (1_i32, 1_i32, q_len, 4_i32))
            .try_into()
            .unwrap();
        let v: Array = (v_data.as_slice(), (1_i32, 1_i32, q_len, 4_i32))
            .try_into()
            .unwrap();
        let mut offsets = vec![0_i32];
        let scale = 0.5_f32;

        let actual = paged
            .update_and_attend_prefill_on(&q, &k, &v, &mut offsets, &[q_len], scale, None, ())
            .expect("hot/cold streaming prefill");
        assert_eq!(offsets, vec![q_len]);

        let summary = paged.hot_cold_summary().expect("hot/cold summary");
        assert!(
            summary.offloaded_pages > 0,
            "prefill should offload cold pages: {summary:?}"
        );
        assert!(
            summary.stream_read_count <= (summary.offloaded_pages as u64) + 1,
            "streaming prefill should not reread cold pages for each query chunk: {summary:?}"
        );

        let (k_ref, v_ref) = paged
            .materialize_prefix_on(&offsets, q_len, ())
            .expect("ref");
        let expected = mlx::fast::scaled_dot_product_attention(
            &q, &k_ref, &v_ref, scale, "causal", None, None,
        )
        .expect("dense sdpa");
        assert_eq!(actual.shape().as_slice(), &[1, 2, q_len, 4]);
        assert_close(
            &actual.to_vec::<f32>().unwrap(),
            &expected.to_vec::<f32>().unwrap(),
            1.0e-3,
        );

        drop(paged);
        fs::remove_dir_all(&root).expect("remove test hot/cold root");
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn paged_kv_hot_cold_prefill_streaming_matches_dense_batched_mask_sdpa() {
        let root = unique_test_dir("paged-kv-hot-cold-prefill-batch");
        let mut paged =
            PagedKVCache::new(2, 1, 4, 4, Dtype::Float32, 8, 2, 8).expect("paged cache");
        paged
            .enable_hot_cold_tiering(
                PagedKvHotColdConfig::new(&root, 1, 1).expect("hot/cold config"),
            )
            .expect("enable hot/cold tiering");

        let q_data: Vec<f32> = (0..(2 * 2 * 4 * 4))
            .map(|i| ((i % 31) as f32 - 15.0) * 0.013)
            .collect();
        let k_data: Vec<f32> = (0..(2 * 4 * 4))
            .map(|i| ((i % 37) as f32 - 18.0) * 0.011)
            .collect();
        let v_data: Vec<f32> = (0..(2 * 4 * 4))
            .map(|i| ((i % 41) as f32 - 20.0) * 0.015)
            .collect();
        let q: Array = (q_data.as_slice(), (2_i32, 2_i32, 4_i32, 4_i32))
            .try_into()
            .unwrap();
        let k: Array = (k_data.as_slice(), (2_i32, 1_i32, 4_i32, 4_i32))
            .try_into()
            .unwrap();
        let v: Array = (v_data.as_slice(), (2_i32, 1_i32, 4_i32, 4_i32))
            .try_into()
            .unwrap();
        let mask = build_batch_attention_mask(&[2, 4], 4, Dtype::Float32).expect("mask");
        let mut offsets = vec![0_i32, 0_i32];
        let scale = 0.5_f32;

        let actual = paged
            .update_and_attend_prefill_on(&q, &k, &v, &mut offsets, &[2, 4], scale, Some(&mask), ())
            .expect("hot/cold batched streaming prefill");
        assert_eq!(offsets, vec![2, 4]);

        let (k_ref, v_ref) = paged.materialize_prefix_on(&offsets, 4, ()).expect("ref");
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
        assert_eq!(actual.shape().as_slice(), &[2, 2, 4, 4]);
        assert_close(
            &actual.to_vec::<f32>().unwrap(),
            &expected.to_vec::<f32>().unwrap(),
            1.0e-4,
        );

        drop(paged);
        fs::remove_dir_all(&root).expect("remove test hot/cold root");
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn paged_kv_hot_cold_repeated_decode_matches_dense_sdpa() {
        let root = unique_test_dir("paged-kv-hot-cold-repeated");
        let mut paged =
            PagedKVCache::new(3, 2, 4, 4, Dtype::Float32, 24, 2, 48).expect("paged cache");
        paged
            .enable_hot_cold_tiering(
                PagedKvHotColdConfig::new(&root, 2, 2).expect("hot/cold config"),
            )
            .expect("enable hot/cold tiering");

        let prefix_k_data: Vec<f32> = (0..(3 * 2 * 5 * 4))
            .map(|i| ((i % 37) as f32 - 18.0) * 0.017)
            .collect();
        let prefix_v_data: Vec<f32> = (0..(3 * 2 * 5 * 4))
            .map(|i| ((i % 41) as f32 - 20.0) * 0.013)
            .collect();
        let prefix_k: Array = (prefix_k_data.as_slice(), (3_i32, 2_i32, 5_i32, 4_i32))
            .try_into()
            .unwrap();
        let prefix_v: Array = (prefix_v_data.as_slice(), (3_i32, 2_i32, 5_i32, 4_i32))
            .try_into()
            .unwrap();
        let mut offsets = vec![0_i32, 0_i32, 0_i32];
        paged
            .update_and_fetch_on(&prefix_k, &prefix_v, &mut offsets, &[5, 3, 4], ())
            .expect("prefix append");

        for step in 0..6 {
            let q_data: Vec<f32> = (0..(3 * 4 * 4))
                .map(|i| (((i + step * 11) % 43) as f32 - 21.0) * 0.011)
                .collect();
            let step_k_data: Vec<f32> = (0..(3 * 2 * 4))
                .map(|i| (((i + step * 7) % 31) as f32 - 15.0) * 0.019)
                .collect();
            let step_v_data: Vec<f32> = (0..(3 * 2 * 4))
                .map(|i| (((i + step * 5) % 29) as f32 - 14.0) * 0.023)
                .collect();
            let q: Array = (q_data.as_slice(), (3_i32, 4_i32, 1_i32, 4_i32))
                .try_into()
                .unwrap();
            let step_k: Array = (step_k_data.as_slice(), (3_i32, 2_i32, 1_i32, 4_i32))
                .try_into()
                .unwrap();
            let step_v: Array = (step_v_data.as_slice(), (3_i32, 2_i32, 1_i32, 4_i32))
                .try_into()
                .unwrap();
            let scale = 0.5_f32;

            let actual = paged
                .update_and_attend_decode_on(
                    &q,
                    &step_k,
                    &step_v,
                    &mut offsets,
                    &[1, 1, 1],
                    scale,
                    (),
                )
                .expect("hot/cold streaming decode");
            let max_len = offsets.iter().copied().max().expect("offsets");
            let (k_ref, v_ref) = paged
                .materialize_prefix_on(&offsets, max_len, ())
                .expect("ref");
            let mask = build_per_row_decode_mask(&offsets, max_len, Dtype::Float32).expect("mask");
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
            assert_eq!(actual.shape().as_slice(), &[3, 4, 1, 4]);
            assert_close(
                &actual.to_vec::<f32>().unwrap(),
                &expected.to_vec::<f32>().unwrap(),
                1.0e-4,
            );

            let summary = paged.hot_cold_summary().expect("hot/cold summary");
            assert!(
                summary.resident_pages <= 12,
                "decode step {step} should respect the hot window plus staging budget: {summary:?}"
            );
        }

        drop(paged);
        fs::remove_dir_all(&root).expect("remove test hot/cold root");
    }
}
