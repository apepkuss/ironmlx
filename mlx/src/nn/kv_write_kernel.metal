// KV Write kernel body for MLX fast::metal_kernel
//
// Writes new KV data for B sequences into the page pool at specified positions.
//
// Compile-time constants: N_KV_HEADS_VAL, HEAD_DIM_VAL, PAGE_SIZE_VAL
// Template: T (dtype)
// Inputs: new_kv [B, n_kv_heads, 1, head_dim], write_positions [B], kv_pool [num_pages, n_kv_heads, PAGE_SIZE, head_dim]
// Output: kv_pool_out (same shape as kv_pool, with new values written)
//
// Grid: (B * N_KV_HEADS * HEAD_DIM, 1, 1)

const int n_kv_heads = N_KV_HEADS_VAL;
const int HEAD_DIM = HEAD_DIM_VAL;
const int PAGE_SIZE = PAGE_SIZE_VAL;

uint gid = thread_position_in_grid.x;
int total_per_batch = n_kv_heads * HEAD_DIM;
int batch_idx = gid / total_per_batch;
int remainder = gid % total_per_batch;
int kv_head = remainder / HEAD_DIM;
int d = remainder % HEAD_DIM;

// Read write position (global flat index in pool)
int write_pos = write_positions[batch_idx];
int page_idx = write_pos / PAGE_SIZE;
int page_offset = write_pos % PAGE_SIZE;

// Source: new_kv[batch_idx, kv_head, 0, d]
int src_offset = (batch_idx * n_kv_heads + kv_head) * HEAD_DIM + d;
T val = new_kv[src_offset];

// Destination: kv_pool[page_idx, kv_head, page_offset, d]
int dst_offset = ((page_idx * n_kv_heads + kv_head) * PAGE_SIZE + page_offset) * HEAD_DIM + d;

// Copy entire pool, then overwrite the target position
// Note: kv_pool_out is initialized as a copy of kv_pool by MLX (init_value not used for this)
// We just write the new value at the target position
kv_pool_out[dst_offset] = val;
