// Fused Paged Attention + KV Write kernel body
//
// Compile-time constants (string-replaced):
//   N_HEADS, N_KV_HEADS, HEAD_DIM, PAGE_SIZE_C, NUM_PAGES, SCALE_F
//
// Template: T (dtype, e.g. bfloat16_t)
//
// Inputs:
//   q:           [B * n_heads * HEAD_DIM]  flattened query
//   new_k:       [B * n_kv_heads * HEAD_DIM]  new key for this token
//   new_v:       [B * n_kv_heads * HEAD_DIM]  new value for this token
//   k_pool:      [NUM_PAGES * n_kv_heads * PAGE_SIZE * HEAD_DIM]  current K pool
//   v_pool:      [NUM_PAGES * n_kv_heads * PAGE_SIZE * HEAD_DIM]  current V pool
//   page_table:  [B * max_pages]  physical page indices
//   seq_lens:    [B]  actual sequence length per batch
//   write_pos:   [B]  global write position (page*PAGE_SIZE+offset) per batch
//   max_pages_c: [1]  max pages per sequence (for page_table indexing)
//
// Outputs:
//   attn_out:    [B * n_heads * HEAD_DIM]  attention output
//   k_pool_out:  [same as k_pool]  updated K pool
//   v_pool_out:  [same as v_pool]  updated V pool
//
// Grid: (B * N_HEADS, 1, 1) — one thread per (batch, head) pair
// Each thread:
//   Phase 1: write new KV to pool (cooperative across heads sharing same kv_head)
//   Phase 2: compute attention over seq_len positions via page_table
//   Phase 3: write attn_out

const int n_heads = /*##N_HEADS##*/1;
const int n_kv_heads = /*##N_KV_HEADS##*/1;
const int hd = /*##HEAD_DIM##*/128;
const int page_sz = /*##PAGE_SIZE##*/256;
const int num_pages = /*##NUM_PAGES##*/8;
const float scale_val = /*##SCALE##*/0.08838835;
const int pool_stride = n_kv_heads * page_sz * hd; // stride per page in pool

uint gid = thread_position_in_grid.x;
int batch_idx = gid / n_heads;
int head_idx = gid % n_heads;
int heads_per_kv = n_heads / n_kv_heads;
int kv_head_idx = head_idx / heads_per_kv;

int sl = seq_lens[batch_idx];
int wp = write_pos[batch_idx];
int mp = max_pages_c;

// ── Phase 1: Copy pool + write new KV ──
// Only the first head per kv_head group does the write (avoid duplicate writes)
if (head_idx % heads_per_kv == 0) {
    // Copy entire k_pool and v_pool to output
    // This is done cooperatively: each (batch, kv_head=0) thread copies a slice
    // For simplicity, only batch_idx==0, head_idx==0 copies the full pool
    // (This is inefficient but correct — TODO: parallelize with more threads)
    if (batch_idx == 0 && kv_head_idx == 0) {
        int pool_size = num_pages * pool_stride;
        for (int i = 0; i < pool_size; i++) {
            k_pool_out[i] = k_pool[i];
            v_pool_out[i] = v_pool[i];
        }
    }

    // Memory barrier to ensure copy is complete before writes
    threadgroup_barrier(metal::mem_flags::mem_device);

    // Write new K and V at write_pos for this batch element
    int page_idx = wp / page_sz;
    int page_off = wp % page_sz;
    int dst_base = (page_idx * n_kv_heads + kv_head_idx) * page_sz * hd + page_off * hd;
    int src_base = (batch_idx * n_kv_heads + kv_head_idx) * hd;

    for (int d = 0; d < hd; d++) {
        k_pool_out[dst_base + d] = new_k[src_base + d];
        v_pool_out[dst_base + d] = new_v[src_base + d];
    }
}

// Barrier to ensure all KV writes are visible before attention reads
threadgroup_barrier(metal::mem_flags::mem_device);

// ── Phase 2: Paged Attention ──
int q_base = (batch_idx * n_heads + head_idx) * hd;
int pt_base = batch_idx * mp;
int new_sl = sl + 1; // include the newly written token

float max_score = -INFINITY;
float sum_exp = 0.0;
float acc[/*##HEAD_DIM##*/128]; // compile-time constant via string replacement
for (int d = 0; d < hd; d++) acc[d] = 0.0;

for (int pos = 0; pos < new_sl; pos++) {
    int p_logical = pos / page_sz;
    int p_offset = pos % page_sz;
    int phys_page = page_table[pt_base + p_logical];
    int kv_base = (phys_page * n_kv_heads + kv_head_idx) * page_sz * hd + p_offset * hd;

    // dot(q, k) * scale
    float score = 0.0;
    for (int d = 0; d < hd; d++) {
        score += float(q[q_base + d]) * float(k_pool_out[kv_base + d]);
    }
    score *= scale_val;

    // Online softmax
    if (score > max_score) {
        float corr = exp(max_score - score);
        sum_exp = sum_exp * corr + 1.0;
        for (int d = 0; d < hd; d++) acc[d] *= corr;
        max_score = score;
    } else {
        sum_exp += exp(score - max_score);
    }

    // Weighted V accumulation
    float w = exp(score - max_score);
    for (int d = 0; d < hd; d++) {
        acc[d] += w * float(v_pool_out[kv_base + d]);
    }
}

// ── Phase 3: Write output ──
int out_base = (batch_idx * n_heads + head_idx) * hd;
float inv = (sum_exp > 0.0) ? (1.0 / sum_exp) : 0.0;
for (int d = 0; d < hd; d++) {
    attn_out[out_base + d] = T(acc[d] * inv);
}
