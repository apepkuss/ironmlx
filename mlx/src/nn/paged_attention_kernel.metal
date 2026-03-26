// Paged Attention Metal Kernel for ironmlx
//
// Performs scaled dot-product attention where KV cache is stored in
// fixed-size pages referenced by a page table.
//
// Inputs:
//   q:           [B, n_heads, 1, head_dim]  — query for current token
//   k_pages:     [num_pages, n_kv_heads, PAGE_SIZE, head_dim] — all K pages
//   v_pages:     [num_pages, n_kv_heads, PAGE_SIZE, head_dim] — all V pages
//   page_table:  [B, max_pages_per_seq] — physical page indices per sequence
//   seq_lens:    [B] — actual sequence length per batch element
//   scale:       scalar — 1/sqrt(head_dim)
//
// Output:
//   out:         [B, n_heads, 1, head_dim] — attention output
//
// Each thread group handles one (batch, head) pair.
// Within the group, threads iterate over pages and compute attention scores.

template <typename T, int HEAD_DIM, int PAGE_SIZE, int BLOCK_THREADS>
[[kernel]] void paged_attention(
    const device T* q          [[buffer(0)]],
    const device T* k_pages    [[buffer(1)]],
    const device T* v_pages    [[buffer(2)]],
    const device int* page_table [[buffer(3)]],
    const device int* seq_lens [[buffer(4)]],
    device T* out              [[buffer(5)]],
    constant int& n_heads      [[buffer(6)]],
    constant int& n_kv_heads   [[buffer(7)]],
    constant int& max_pages    [[buffer(8)]],
    constant int& num_pages_total [[buffer(9)]],
    constant float& scale      [[buffer(10)]],
    uint3 tid                  [[thread_position_in_threadgroup]],
    uint3 gid                  [[threadgroup_position_in_grid]]
) {
    const int batch_idx = gid.x;
    const int head_idx = gid.y;
    const int kv_head_idx = head_idx / (n_heads / n_kv_heads);
    const int seq_len = seq_lens[batch_idx];
    const int thread_id = tid.x;

    if (seq_len == 0) return;

    // Load query vector into registers
    // q layout: [B, n_heads, 1, HEAD_DIM]
    const int q_offset = ((batch_idx * n_heads + head_idx) * 1) * HEAD_DIM;
    float q_reg[HEAD_DIM];
    for (int d = 0; d < HEAD_DIM; d++) {
        q_reg[d] = float(q[q_offset + d]);
    }

    // Phase 1: Compute attention scores across all pages
    // Each thread handles a subset of positions
    const int num_pages_seq = (seq_len + PAGE_SIZE - 1) / PAGE_SIZE;
    const int pt_offset = batch_idx * max_pages;

    // Accumulate softmax numerator and denominator
    float max_score = -INFINITY;
    float sum_exp = 0.0;
    float acc[HEAD_DIM];
    for (int d = 0; d < HEAD_DIM; d++) acc[d] = 0.0;

    // Iterate over all valid positions
    for (int pos = thread_id; pos < seq_len; pos += BLOCK_THREADS) {
        int page_idx_logical = pos / PAGE_SIZE;
        int page_offset = pos % PAGE_SIZE;
        int physical_page = page_table[pt_offset + page_idx_logical];

        // k_pages layout: [num_pages, n_kv_heads, PAGE_SIZE, HEAD_DIM]
        int k_offset = ((physical_page * n_kv_heads + kv_head_idx) * PAGE_SIZE + page_offset) * HEAD_DIM;

        // Compute dot product: q . k
        float score = 0.0;
        for (int d = 0; d < HEAD_DIM; d++) {
            score += q_reg[d] * float(k_pages[k_offset + d]);
        }
        score *= scale;

        // Online softmax update
        if (score > max_score) {
            float old_max = max_score;
            max_score = score;
            float correction = exp(old_max - max_score);
            sum_exp = sum_exp * correction + exp(score - max_score);
            for (int d = 0; d < HEAD_DIM; d++) {
                acc[d] = acc[d] * correction;
            }
        } else {
            sum_exp += exp(score - max_score);
        }

        // Accumulate weighted V
        float w = exp(score - max_score);
        int v_offset = ((physical_page * n_kv_heads + kv_head_idx) * PAGE_SIZE + page_offset) * HEAD_DIM;
        for (int d = 0; d < HEAD_DIM; d++) {
            acc[d] += w * float(v_pages[v_offset + d]);
        }
    }

    // Phase 2: Reduce across threads in the group (if BLOCK_THREADS > 1)
    // For simplicity, using single-thread per (batch, head) for now
    // TODO: threadgroup reduction for BLOCK_THREADS > 1

    // Write output
    if (thread_id == 0) {
        int out_offset = ((batch_idx * n_heads + head_idx) * 1) * HEAD_DIM;
        float inv_sum = (sum_exp > 0.0) ? (1.0 / sum_exp) : 0.0;
        for (int d = 0; d < HEAD_DIM; d++) {
            out[out_offset + d] = T(acc[d] * inv_sum);
        }
    }
}
