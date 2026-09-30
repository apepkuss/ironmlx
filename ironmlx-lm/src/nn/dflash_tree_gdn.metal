// Adapted from TensorFold v0.3.6.2 lane_tree.py. See LICENSE.tensorfold.
auto hv_idx = thread_position_in_grid.z % Hv;
auto hk_idx = hv_idx / (Hv / Hk);
constexpr int n_per_t = Dk / 32;
auto dk_idx = thread_position_in_threadgroup.x;
auto dv_idx = thread_position_in_grid.y;
auto i_state = state_in + (hv_idx * Dv + dv_idx) * Dk;
float s0[n_per_t];
for (int i=0;i<n_per_t;++i) s0[i]=float(i_state[n_per_t*dk_idx+i]);
float states[16][n_per_t];
for (int node=0;node<nodes[0];++node) {
  const int parent=parents[node];
  float state[n_per_t];
  for(int i=0;i<n_per_t;++i) state[i]=parent<0?s0[i]:states[parent][i];
  auto q_=q+(node*Hk+hk_idx)*Dk;
  auto k_=k+(node*Hk+hk_idx)*Dk;
  auto v_=v+(node*Hv+hv_idx)*Dv;
  const float g_=float(g[node*Hv+hv_idx]), beta_=float(beta[node*Hv+hv_idx]);
  float kv_mem=0.0f;
  for(int i=0;i<n_per_t;++i) {
    auto idx=n_per_t*dk_idx+i;
    state[i]=state[i]*g_;
    kv_mem+=state[i]*k_[idx];
  }
  kv_mem=simd_sum(kv_mem);
  auto delta=(v_[dv_idx]-kv_mem)*beta_;
  float out=0.0f;
  for(int i=0;i<n_per_t;++i) {
    auto idx=n_per_t*dk_idx+i;
    state[i]=state[i]+k_[idx]*delta;
    out+=state[i]*q_[idx];
  }
  out=simd_sum(out);
  if(thread_index_in_simdgroup==0) y[(node*Hv+hv_idx)*Dv+dv_idx]=static_cast<InT>(out);
  for(int i=0;i<n_per_t;++i) states[node][i]=state[i];
}
