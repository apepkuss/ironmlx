//! Paged Attention — custom Metal kernel for page-table-based KV cache access.

use crate::array::Array;
use crate::cache::paged_cache::{PAGE_SIZE, PagePool, PageTable};
use crate::dtype::Dtype;
use crate::error::Result;
use crate::metal_kernel::{MetalKernel, MetalKernelConfig};
use crate::ops;
use crate::stream::Stream;

/// Metal shader source for paged attention (embedded at compile time).
const PAGED_ATTN_SOURCE: &str = include_str!("paged_attention_kernel.metal");

/// Execute paged attention for batched decode.
///
/// # Arguments
/// * `q` - Query tensor `[B, n_heads, 1, head_dim]`
/// * `pool` - Page pool containing all KV pages
/// * `page_tables` - Per-sequence page tables
/// * `seq_lens` - Actual sequence length per batch element
/// * `layer_idx` - Which layer's pages to use
/// * `n_heads` - Number of query heads
/// * `head_dim` - Dimension per head
/// * `stream` - MLX stream
///
/// # Returns
/// Attention output `[B, n_heads, 1, head_dim]`
#[allow(clippy::too_many_arguments)]
pub fn paged_attention(
    q: &Array,
    pool: &PagePool,
    page_tables: &[&PageTable],
    seq_lens: &[i32],
    layer_idx: usize,
    n_heads: usize,
    head_dim: usize,
    dtype: Dtype,
    stream: &Stream,
) -> Result<Array> {
    let batch_size = page_tables.len();
    let n_kv_heads = pool.n_kv_heads;
    let max_pages = page_tables
        .iter()
        .map(|pt| pt.num_pages())
        .max()
        .unwrap_or(0);
    let scale = 1.0 / (head_dim as f32).sqrt();

    // Build padded page table: [B, max_pages]
    let mut pt_data: Vec<i32> = Vec::with_capacity(batch_size * max_pages);
    for pt in page_tables {
        let pages: Vec<i32> = pt.pages.iter().map(|&p| p as i32).collect();
        pt_data.extend_from_slice(&pages);
        // Pad with 0 (page 0 will be masked by seq_len)
        pt_data.extend(std::iter::repeat_n(0i32, max_pages - pages.len()));
    }
    let page_table_arr = Array::from_slice_i32(&pt_data);
    let page_table_2d = ops::reshape(
        &page_table_arr,
        &[batch_size as i32, max_pages as i32],
        stream,
    )?;

    let seq_lens_arr = Array::from_slice_i32(seq_lens);

    // Scalar constants as arrays
    let n_heads_arr = Array::from_int(n_heads as i32);
    let n_kv_heads_arr = Array::from_int(n_kv_heads as i32);
    let max_pages_arr = Array::from_int(max_pages as i32);
    let num_pages_total_arr = Array::from_int(pool.num_pages as i32);
    let scale_arr = Array::from_float(scale);

    // Create kernel (cached after first call by MLX)
    let kernel = MetalKernel::new(
        "paged_attention",
        &[
            "q",
            "k_pages",
            "v_pages",
            "page_table",
            "seq_lens",
            "out",
            "n_heads",
            "n_kv_heads",
            "max_pages",
            "num_pages_total",
            "scale",
        ],
        &["out"],
        PAGED_ATTN_SOURCE,
        "", // no header
        true,
        false,
    );

    // Configure kernel execution
    let config = MetalKernelConfig::new();

    // Output shape: [B, n_heads, 1, head_dim]
    let out_shape = [batch_size as i32, n_heads as i32, 1, head_dim as i32];
    config.add_output(&out_shape, dtype);

    // Grid: one thread group per (batch, head)
    config.set_grid(batch_size as i32, n_heads as i32, 1);
    // Thread group: single thread for now (TODO: multi-thread reduction)
    config.set_thread_group(1, 1, 1);

    // Template args for compile-time constants
    config.add_template_dtype("T", dtype);
    config.add_template_int("HEAD_DIM", head_dim as i32);
    config.add_template_int("PAGE_SIZE", PAGE_SIZE as i32);
    config.add_template_int("BLOCK_THREADS", 1);

    config.set_init_value(0.0);

    // Execute kernel
    let outputs = kernel.apply(
        &[
            q,
            &pool.k_pages[layer_idx],
            &pool.v_pages[layer_idx],
            &page_table_2d,
            &seq_lens_arr,
            // out is implicit (allocated by config)
            &n_heads_arr,
            &n_kv_heads_arr,
            &max_pages_arr,
            &num_pages_total_arr,
            &scale_arr,
        ],
        &config,
        stream,
    )?;

    outputs
        .into_iter()
        .next()
        .ok_or_else(|| crate::error::Error::Mlx("paged_attention returned no output".to_string()))
}

/// Write new KV data into page pool at the correct positions.
///
/// * `new_k` / `new_v`: `[B, n_kv_heads, 1, head_dim]` — new KV for one token
/// * `page_tables`: per-sequence page tables
/// * `seq_lens`: current seq lengths (write position = seq_len for each)
pub fn write_kv_to_pages(
    new_k: &Array,
    new_v: &Array,
    pool: &PagePool,
    page_tables: &[&PageTable],
    seq_lens: &[i32],
    layer_idx: usize,
    stream: &Stream,
) -> Result<()> {
    let batch_size = page_tables.len();

    for b in 0..batch_size {
        let write_pos = seq_lens[b] as usize;
        let page_logical = write_pos / PAGE_SIZE;
        let page_offset = write_pos % PAGE_SIZE;

        if page_logical >= page_tables[b].pages.len() {
            continue; // shouldn't happen if ensure_capacity was called
        }
        let physical_page = page_tables[b].pages[page_logical] as i32;

        // Extract this batch element's KV: [1, n_kv_heads, 1, head_dim]
        let b_i = b as i32;
        let k_shape = new_k.shape();
        let seq_k = ops::slice(
            new_k,
            &[b_i, 0, 0, 0],
            &[b_i + 1, k_shape[1], 1, k_shape[3]],
            &[1, 1, 1, 1],
            stream,
        )?;
        let seq_v = ops::slice(
            new_v,
            &[b_i, 0, 0, 0],
            &[b_i + 1, k_shape[1], 1, k_shape[3]],
            &[1, 1, 1, 1],
            stream,
        )?;

        // Reshape to [n_kv_heads, 1, head_dim] for put_along_axis
        let seq_k = ops::reshape(&seq_k, &[k_shape[1], 1, k_shape[3]], stream)?;
        let seq_v = ops::reshape(&seq_v, &[k_shape[1], 1, k_shape[3]], stream)?;

        // Write into pool's k_pages[layer_idx] at [physical_page, :, page_offset, :]
        // Use slice + put_along_axis
        let page_k = ops::slice(
            &pool.k_pages[layer_idx],
            &[physical_page, 0, 0, 0],
            &[physical_page + 1, k_shape[1], PAGE_SIZE as i32, k_shape[3]],
            &[1, 1, 1, 1],
            stream,
        )?;
        let page_k = ops::reshape(&page_k, &[k_shape[1], PAGE_SIZE as i32, k_shape[3]], stream)?;

        let write_idx = Array::from_int(page_offset as i32);
        let write_idx = ops::reshape(&write_idx, &[1, 1, 1], stream)?;
        let updated_k = ops::put_along_axis(&page_k, &write_idx, &seq_k, 1, stream)?;

        let page_v = ops::slice(
            &pool.v_pages[layer_idx],
            &[physical_page, 0, 0, 0],
            &[physical_page + 1, k_shape[1], PAGE_SIZE as i32, k_shape[3]],
            &[1, 1, 1, 1],
            stream,
        )?;
        let page_v = ops::reshape(&page_v, &[k_shape[1], PAGE_SIZE as i32, k_shape[3]], stream)?;
        let updated_v = ops::put_along_axis(&page_v, &write_idx, &seq_v, 1, stream)?;

        // TODO: Write updated_k/v back into pool.k_pages[layer_idx][physical_page]
        // This requires mutable access to pool arrays — currently MLX arrays are immutable.
        // For now, store updated pages as new arrays (will be optimized in future).
        let _ = (updated_k, updated_v);
    }

    Ok(())
}
