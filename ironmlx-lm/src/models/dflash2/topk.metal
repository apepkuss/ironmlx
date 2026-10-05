// Radix threshold adapted from TensorFold v0.3.6.2 engine/topk.py (MIT).
// Unlike its bounded tie buffer, ties here are selected over the entire row.
constexpr uint TPG=1024, MAXK=64;
const uint row=threadgroup_position_in_grid.x, tid=thread_position_in_threadgroup.x;
const int V=dims[0], K=dims[1];
const device ushort* x=(const device ushort*)X+(int64_t)row*V;
threadgroup atomic_uint hist[256], count;
threadgroup uint found[4], selected_key[MAXK], selected_idx[MAXK], minima[32];
#define SKEY(b) (((b & 0x7FFFu)==0u) ? 0x8000u : ((b & 0x7FFFu)>0x7F80u) ? 0u : ((b & 0x8000u) ? (~b & 0xFFFFu) : (b | 0x8000u)))
for(uint i=tid;i<256;i+=TPG) atomic_store_explicit(&hist[i],0u,memory_order_relaxed);
threadgroup_barrier(mem_flags::mem_threadgroup);
for(int i=tid;i<V;i+=TPG) atomic_fetch_add_explicit(&hist[SKEY(uint(x[i]))>>8],1u,memory_order_relaxed);
threadgroup_barrier(mem_flags::mem_threadgroup);
if(tid==0) {
  uint above=0; int b=255;
  for(;b>=0;--b) { uint n=atomic_load_explicit(&hist[b],memory_order_relaxed); if(above+n>=uint(K))break; above+=n; }
  found[0]=uint(max(b,0)); found[2]=above;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
const uint hi=found[0];
for(uint i=tid;i<256;i+=TPG) atomic_store_explicit(&hist[i],0u,memory_order_relaxed);
threadgroup_barrier(mem_flags::mem_threadgroup);
for(int i=tid;i<V;i+=TPG) { uint key=SKEY(uint(x[i])); if((key>>8)==hi) atomic_fetch_add_explicit(&hist[key&255u],1u,memory_order_relaxed); }
threadgroup_barrier(mem_flags::mem_threadgroup);
if(tid==0) {
  uint above=found[2]; int b=255;
  for(;b>=0;--b) { uint n=atomic_load_explicit(&hist[b],memory_order_relaxed); if(above+n>=uint(K))break; above+=n; }
  found[1]=(hi<<8)|uint(max(b,0)); found[2]=above; found[3]=uint(K)-above;
  atomic_store_explicit(&count,0u,memory_order_relaxed);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
const uint threshold=found[1];
for(int i=tid;i<V;i+=TPG) {
  uint key=SKEY(uint(x[i]));
  if(key>threshold) { uint at=atomic_fetch_add_explicit(&count,1u,memory_order_relaxed); if(at<MAXK) {selected_key[at]=key;selected_idx[at]=uint(i);} }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
// Ordered global minima make arbitrarily many equal logits deterministic.
for(uint j=0;j<found[3];++j) {
  uint best=0xFFFFFFFFu;
  uint last=j ? selected_idx[found[2]+j-1] : 0u;
  for(int i=tid;i<V;i+=TPG) if((j==0 || uint(i)>last) && SKEY(uint(x[i]))==threshold) best=min(best,uint(i));
  best=simd_min(best);
  if(thread_index_in_simdgroup==0) minima[simdgroup_index_in_threadgroup]=best;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if(tid==0) {
    uint best=0xFFFFFFFFu; for(uint s=0;s<32;++s) best=min(best,minima[s]);
    selected_idx[found[2]+j]=best; selected_key[found[2]+j]=threshold;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}
if(tid==0) {
  for(int a=1;a<K;++a) {
    uint key=selected_key[a], id=selected_idx[a]; int b=a-1;
    while(b>=0 && (selected_key[b]<key || (selected_key[b]==key && selected_idx[b]>id))) {selected_key[b+1]=selected_key[b];selected_idx[b+1]=selected_idx[b];--b;}
    selected_key[b+1]=key;selected_idx[b+1]=id;
  }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for(int i=tid;i<K;i+=TPG) IDX[row*K+i]=selected_idx[i];
