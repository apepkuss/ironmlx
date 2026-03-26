//! Paged KV Cache — fixed-size pages managed via page table.
//!
//! Each page holds `PAGE_SIZE` token positions of KV data for one layer.
//! A page table maps logical sequence positions to physical page indices.
//! The paged attention kernel reads KV data by following page table lookups.

use crate::array::Array;
use crate::dtype::Dtype;
use crate::error::Result;
use crate::ops;
use crate::stream::Stream;

/// Number of token positions per page.
pub const PAGE_SIZE: usize = 256;

/// Physical page pool — pre-allocated GPU memory for KV cache pages.
///
/// Each page stores KV data for `PAGE_SIZE` tokens across all layers:
/// `[num_layers, 2, n_kv_heads, PAGE_SIZE, head_dim]`
///
/// The pool tracks which pages are free vs allocated.
pub struct PagePool {
    /// Total number of physical pages.
    pub num_pages: usize,
    /// Per-layer K pages: `[num_pages, n_kv_heads, PAGE_SIZE, head_dim]`
    pub k_pages: Vec<Array>,
    /// Per-layer V pages: `[num_pages, n_kv_heads, PAGE_SIZE, head_dim]`
    pub v_pages: Vec<Array>,
    /// Free page indices (stack).
    free_list: Vec<usize>,
    /// Configuration.
    pub num_layers: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub dtype: Dtype,
}

impl PagePool {
    /// Create a new page pool with pre-allocated pages.
    pub fn new(
        num_pages: usize,
        num_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        dtype: Dtype,
    ) -> Result<Self> {
        let stream = Stream::new(&crate::device::Device::gpu());
        let page_shape = &[
            num_pages as i32,
            n_kv_heads as i32,
            PAGE_SIZE as i32,
            head_dim as i32,
        ];

        let mut k_pages = Vec::with_capacity(num_layers);
        let mut v_pages = Vec::with_capacity(num_layers);

        for _ in 0..num_layers {
            let k = ops::zeros(page_shape, dtype, &stream)?;
            let v = ops::zeros(page_shape, dtype, &stream)?;
            // Eval to actually allocate GPU memory
            k.eval()?;
            v.eval()?;
            k_pages.push(k);
            v_pages.push(v);
        }

        let free_list: Vec<usize> = (0..num_pages).rev().collect();

        Ok(Self {
            num_pages,
            k_pages,
            v_pages,
            free_list,
            num_layers,
            n_kv_heads,
            head_dim,
            dtype,
        })
    }

    /// Allocate a page. Returns physical page index, or None if pool is exhausted.
    pub fn alloc(&mut self) -> Option<usize> {
        self.free_list.pop()
    }

    /// Free a page back to the pool.
    pub fn free(&mut self, page_idx: usize) {
        self.free_list.push(page_idx);
    }

    /// Number of free pages.
    pub fn free_count(&self) -> usize {
        self.free_list.len()
    }

    /// Bytes per page (for reporting).
    pub fn page_bytes(&self) -> usize {
        self.num_layers * 2 * self.n_kv_heads * PAGE_SIZE * self.head_dim * self.dtype.size_of()
    }
}

/// Per-sequence page table — maps logical token positions to physical pages.
#[derive(Default)]
pub struct PageTable {
    /// Physical page indices in order. `pages[i]` holds tokens `[i*PAGE_SIZE .. (i+1)*PAGE_SIZE)`.
    pub pages: Vec<usize>,
    /// Number of tokens actually written.
    pub len: usize,
}

impl PageTable {
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of pages allocated for this sequence.
    pub fn num_pages(&self) -> usize {
        self.pages.len()
    }

    /// Ensure capacity for `new_len` tokens, allocating pages from pool as needed.
    /// Returns Ok(()) or Err if pool is exhausted.
    pub fn ensure_capacity(&mut self, new_len: usize, pool: &mut PagePool) -> Result<()> {
        let pages_needed = new_len.div_ceil(PAGE_SIZE);
        while self.pages.len() < pages_needed {
            let page_idx = pool.alloc().ok_or_else(|| {
                crate::error::Error::Mlx("page pool exhausted".to_string())
            })?;
            self.pages.push(page_idx);
        }
        Ok(())
    }

    /// Release all pages back to the pool.
    pub fn release_all(&mut self, pool: &mut PagePool) {
        for &page_idx in &self.pages {
            pool.free(page_idx);
        }
        self.pages.clear();
        self.len = 0;
    }

    /// Build a flat page table array for the kernel: [num_pages] of i32 physical indices.
    pub fn to_array(&self) -> Array {
        let indices: Vec<i32> = self.pages.iter().map(|&p| p as i32).collect();
        Array::from_slice_i32(&indices)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn page_pool_alloc_free() {
        let pool = PagePool::new(4, 2, 4, 64, Dtype::Float16).unwrap();
        assert_eq!(pool.free_count(), 4);

        let mut pool = pool;
        let p0 = pool.alloc().unwrap();
        let p1 = pool.alloc().unwrap();
        assert_eq!(pool.free_count(), 2);
        assert_ne!(p0, p1);

        pool.free(p0);
        assert_eq!(pool.free_count(), 3);
    }

    #[test]
    fn page_table_ensure_capacity() {
        let mut pool = PagePool::new(4, 2, 4, 64, Dtype::Float16).unwrap();
        let mut pt = PageTable::new();

        pt.ensure_capacity(100, &mut pool).unwrap(); // 1 page (256 tokens)
        assert_eq!(pt.num_pages(), 1);

        pt.ensure_capacity(300, &mut pool).unwrap(); // 2 pages
        assert_eq!(pt.num_pages(), 2);

        pt.ensure_capacity(256, &mut pool).unwrap(); // still 2 pages (already have capacity)
        assert_eq!(pt.num_pages(), 2);

        pt.release_all(&mut pool);
        assert_eq!(pt.num_pages(), 0);
        assert_eq!(pool.free_count(), 4);
    }
}
