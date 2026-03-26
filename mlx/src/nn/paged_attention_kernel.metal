// Paged Attention kernel body for MLX fast::metal_kernel
//
// Compile-time constants (injected via string replacement):
//   N_HEADS_VAL, N_KV_HEADS_VAL, MAX_PAGES_VAL, NUM_PAGES_VAL,
//   SCALE_VAL, HEAD_DIM_VAL, PAGE_SIZE_VAL
//
// Template: T (dtype)
// Inputs: q, k_pages, v_pages, page_table, seq_lens
// Output: out
// Grid: (B * n_heads, 1, 1)

const int n_heads = N_HEADS_VAL;
const int n_kv_heads = N_KV_HEADS_VAL;
const int max_pages = MAX_PAGES_VAL;
const float scale = SCALE_VAL;
const int HEAD_DIM = HEAD_DIM_VAL;
const int PAGE_SIZE = PAGE_SIZE_VAL;

uint gid = thread_position_in_grid.x;
int batch_idx = gid / n_heads;
int head_idx = gid % n_heads;
int kv_head_idx = head_idx / (n_heads / n_kv_heads);

int seq_len_val = seq_lens[batch_idx];
if (seq_len_val == 0) return;

// Query offset: q is [B, n_heads, 1, HEAD_DIM] flattened
int q_offset = (batch_idx * n_heads + head_idx) * HEAD_DIM;

// Online softmax attention
float max_score = -INFINITY;
float sum_exp = 0.0;

// Use threadgroup memory for accumulator if HEAD_DIM <= 256
float acc[HEAD_DIM_VAL];
for (int d = 0; d < HEAD_DIM; d++) acc[d] = 0.0;

int pt_offset = batch_idx * max_pages;

for (int pos = 0; pos < seq_len_val; pos++) {
    int page_logical = pos / PAGE_SIZE;
    int page_offset = pos % PAGE_SIZE;
    int physical_page = page_table[pt_offset + page_logical];

    // k_pages: [num_pages, n_kv_heads, PAGE_SIZE, HEAD_DIM] flattened
    int k_off = ((physical_page * n_kv_heads + kv_head_idx) * PAGE_SIZE + page_offset) * HEAD_DIM;

    // dot(q, k) * scale
    float score = 0.0;
    for (int d = 0; d < HEAD_DIM; d++) {
        score += float(q[q_offset + d]) * float(k_pages[k_off + d]);
    }
    score *= scale;

    // Online softmax update
    if (score > max_score) {
        float correction = exp(max_score - score);
        sum_exp = sum_exp * correction + 1.0;
        for (int d = 0; d < HEAD_DIM; d++) {
            acc[d] *= correction;
        }
        max_score = score;
    } else {
        sum_exp += exp(score - max_score);
    }

    // Accumulate weighted V
    float w = exp(score - max_score);
    int v_off = ((physical_page * n_kv_heads + kv_head_idx) * PAGE_SIZE + page_offset) * HEAD_DIM;
    for (int d = 0; d < HEAD_DIM; d++) {
        acc[d] += w * float(v_pages[v_off + d]);
    }
}

// Write normalized output
int out_offset = (batch_idx * n_heads + head_idx) * HEAD_DIM;
float inv_sum = (sum_exp > 0.0) ? (1.0 / sum_exp) : 0.0;
for (int d = 0; d < HEAD_DIM; d++) {
    out[out_offset + d] = T(acc[d] * inv_sum);
}
