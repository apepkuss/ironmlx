#include "cxx_mlx_shim/memory.h"
#include "cxx_mlx_shim/shim_helpers.h"
#include "mlx-sys/src/bridge/memory.rs.h"

#include <algorithm>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <unordered_set>
#include <variant>

#if __has_include(<Metal/Metal.hpp>)
#include <dispatch/dispatch.h>
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/resident.h"
#endif

namespace cxx_mlx {

struct ScopedResidency::Impl {
  std::vector<MlxArray> weights;
  std::size_t bytes{0};
#if __has_include(<Metal/Metal.hpp>)
  std::vector<NS::SharedPtr<MTL::ResidencySet>> sets;
  std::vector<NS::SharedPtr<MTL::CommandQueue>> queues;
  bool active{false};
  bool retiring{false};
  // The first request runs unlocked; no other caller may start or end it.
  bool first_request{false};
  void detach() {
    for (auto& queue : queues)
      for (auto& set : sets) queue->removeResidencySet(set.get());
    queues.clear();
  }
  void activate(bool enabled) {
    auto pool = mlx::core::metal::new_scoped_memory_pool();
    enabled = enabled && !retiring;
    if (!enabled) detach();
    if (first_request || enabled == active) return;
    for (auto& set : sets) {
      if (enabled) set->requestResidency();
      else set->endResidency();
    }
    active = enabled;
  }
  ~Impl() { activate(false); }
#endif
};

namespace {
struct Registry {
  std::mutex mutex;
  std::unordered_set<ScopedResidency::Impl*> leases;
  std::size_t bytes{0};
  bool enabled{true};
  bool external_wired{false};
#if __has_include(<Metal/Metal.hpp>)
  bool os_normal{true};
  dispatch_source_t pressure_source{nullptr};
  dispatch_source_t renewal_source{nullptr};
  bool renewal_running{false};
#endif
};
Registry& registry() {
  static auto* r = new Registry;
  return *r;
}
#if __has_include(<Metal/Metal.hpp>)
bool allowed(const Registry& r) {
  return r.enabled && r.os_normal && !r.external_wired &&
      mlx::core::metal::device(mlx::core::Device::gpu).residency_sets().num_sets() < 32;
}
void pressure_changed(void* context) {
  auto& r = *static_cast<Registry*>(context);
  const auto pressure = dispatch_source_get_data(r.pressure_source);
  std::lock_guard<std::mutex> lock(r.mutex);
  if (pressure & (DISPATCH_MEMORYPRESSURE_WARN | DISPATCH_MEMORYPRESSURE_CRITICAL))
    r.os_normal = false;
  else if (pressure & DISPATCH_MEMORYPRESSURE_NORMAL)
    r.os_normal = true;
  for (auto* lease : r.leases) lease->activate(allowed(r));
}
// macOS drops a set's residency a few seconds after the GPU goes idle, even
// with a standing request; the next submission then re-wires every weight in
// kernel time. Renewing active requests keeps them resident without GPU work.
constexpr std::uint64_t kRenewalIntervalNs = 500 * NSEC_PER_MSEC;
void renew(void* context) {
  auto& r = *static_cast<Registry*>(context);
  auto pool = mlx::core::metal::new_scoped_memory_pool();
  // Measured at tens of microseconds once resident; up to ~0.3 s once while a
  // new 78 GB lease is still settling during model load.
  std::lock_guard<std::mutex> lock(r.mutex);
  for (auto* lease : r.leases) {
    if (!lease->active || lease->first_request || lease->retiring) continue;
    for (auto& set : lease->sets) set->requestResidency();
  }
}
void ensure_monitors(Registry& r) {
  if (!r.pressure_source) {
    auto source = dispatch_source_create(DISPATCH_SOURCE_TYPE_MEMORYPRESSURE, 0,
        DISPATCH_MEMORYPRESSURE_NORMAL | DISPATCH_MEMORYPRESSURE_WARN |
            DISPATCH_MEMORYPRESSURE_CRITICAL,
        dispatch_get_global_queue(QOS_CLASS_UTILITY, 0));
    if (!source) throw std::runtime_error("weight residency pressure monitor unavailable");
    // Process-wide, with no model/storage capture and no dedicated poll thread.
    r.pressure_source = source;
    dispatch_set_context(source, &r);
    dispatch_source_set_event_handler_f(source, pressure_changed);
    dispatch_resume(source);
  }
  if (!r.renewal_source) {
    // Created suspended; sync_renewal runs it only while a lease exists.
    auto source = dispatch_source_create(DISPATCH_SOURCE_TYPE_TIMER, 0, 0,
        dispatch_get_global_queue(QOS_CLASS_UTILITY, 0));
    if (!source) throw std::runtime_error("weight residency renewal unavailable");
    dispatch_set_context(source, &r);
    dispatch_source_set_event_handler_f(source, renew);
    dispatch_source_set_timer(source, dispatch_time(DISPATCH_TIME_NOW, kRenewalIntervalNs),
        kRenewalIntervalNs, kRenewalIntervalNs / 5);
    r.renewal_source = source;
  }
}
// Caller holds the registry lock.
void sync_renewal(Registry& r) {
  const bool wanted = r.renewal_source && !r.leases.empty();
  if (wanted == r.renewal_running) return;
  if (wanted) dispatch_resume(r.renewal_source);
  else dispatch_suspend(r.renewal_source);
  r.renewal_running = wanted;
}
#endif
}  // namespace

ScopedResidency::ScopedResidency(std::unique_ptr<Impl> p) : impl(std::move(p)) {}
ScopedResidency::~ScopedResidency() {
  auto& r = registry();
  {
#if __has_include(<Metal/Metal.hpp>)
    auto pool = mlx::core::metal::new_scoped_memory_pool();
#endif
    std::vector<MlxArray> weights;
    {
      std::lock_guard<std::mutex> lock(r.mutex);
#if __has_include(<Metal/Metal.hpp>)
      impl->activate(false);
      impl->retiring = true;
      impl->sets.clear();
#endif
      weights.swap(impl->weights);
    }
    // Data destruction may call the allocator. Keep the retiring reservation
    // visible, but release array handles outside the registry lock.
    weights.clear();
  }
  std::lock_guard<std::mutex> lock(r.mutex);
  if (r.leases.erase(impl.get())) r.bytes -= impl->bytes;
#if __has_include(<Metal/Metal.hpp>)
  sync_renewal(r);
#endif
}

std::unique_ptr<ScopedResidency> make_scoped_residency(
    rust::Slice<const MlxArray* const> arrays, std::size_t budget_bytes) {
#if __has_include(<Metal/Metal.hpp>)
  if (__builtin_available(macOS 15, *)) {
    auto pool = mlx::core::metal::new_scoped_memory_pool();
    auto& device = mlx::core::metal::device(mlx::core::Device::gpu);
    auto* metal = device.mtl_device();
    if (!metal->supportsFamily(MTL::GPUFamilyMetal3))
      throw std::runtime_error("scoped residency requires Metal 3");
    auto impl = std::make_unique<ScopedResidency::Impl>();
    std::unordered_set<const MTL::Allocation*> unique;
    std::vector<const MTL::Allocation*> allocations;
    for (const auto* a : arrays) {
      if (!a || a->size() == 0 || !a->is_available())
        throw std::invalid_argument("scoped residency requires evaluated weights");
      auto* buffer = static_cast<MTL::Buffer*>(a->buffer().ptr());
      if (!buffer) throw std::invalid_argument("weight has no Metal storage");
      // A shared heap may also contain KV or another model's buffers.
      const MTL::Allocation* allocation = buffer;
      if (unique.insert(allocation).second) {
        const auto bytes = allocation->allocatedSize();
        if (bytes > budget_bytes - impl->bytes)
          throw std::runtime_error("scoped weight residency exceeds model budget");
        impl->bytes += bytes;
        allocations.push_back(allocation);
      }
      impl->weights.push_back(*a);
    }
    if (allocations.empty()) throw std::invalid_argument("empty weight residency");
    const auto recommended = static_cast<std::size_t>(metal->recommendedMaxWorkingSetSize());
    const auto physical = std::get<std::size_t>(
        mlx::core::device_info(mlx::core::Device::gpu).at("memory_size"));
    constexpr std::size_t gib = std::size_t(1) << 30;
    const auto ceiling = std::min(recommended > 2*gib ? recommended-2*gib : 0,
                                  physical > 8*gib ? physical-8*gib : 0);
    auto& r = registry();
    std::unique_lock<std::mutex> lock(r.mutex);
    ensure_monitors(r);
    const auto other = device.residency_sets().wired_size();
    const auto used = r.bytes > ceiling ? ceiling : r.bytes;
    if (other > ceiling-used || impl->bytes > ceiling-used-other)
      throw std::runtime_error("scoped weight residency has no process headroom");
    const auto cap = std::max<std::size_t>(64 << 20, recommended/20);
    std::size_t in_set = 0;
    for (const auto* allocation : allocations) {
      if (impl->sets.empty() || (in_set + allocation->allocatedSize() > cap)) {
        auto descriptor = NS::TransferPtr(MTL::ResidencySetDescriptor::alloc()->init());
        NS::Error* error = nullptr;
        auto set = NS::TransferPtr(metal->newResidencySet(descriptor.get(), &error));
        if (!set) throw std::runtime_error("Metal weight residency creation failed");
        impl->sets.push_back(std::move(set));
        in_set = 0;
      }
      impl->sets.back()->addAllocation(allocation);
      in_set += allocation->allocatedSize();
    }
    for (auto& set : impl->sets) set->commit();
    auto* identity = impl.get();
    // Reserve first so a concurrent lease sees these bytes. requestResidency
    // takes seconds for a large model; holding the registry through it would
    // stall health reads and every memory-governor sample behind them.
    r.leases.insert(identity);
    r.bytes += identity->bytes;
    sync_renewal(r);
    if (allowed(r)) {
      identity->first_request = true;
      lock.unlock();
      for (auto& set : identity->sets) set->requestResidency();
      lock.lock();
      identity->first_request = false;
      identity->active = true;
      // Policy may have changed during the request; settle before publishing.
      identity->activate(allowed(r));
    }
    // Register before constructing the owner, whose destructor takes this lock.
    // No owning ScopedResidency object exists if allocation/registration fails.
    try {
      return std::make_unique<ScopedResidency>(std::move(impl));
    } catch (...) {
      r.leases.erase(identity);
      r.bytes -= identity->bytes;
      sync_renewal(r);
      throw;
    }
  }
#endif
  throw std::runtime_error("scoped weight residency requires macOS 15 Metal");
}

std::size_t scoped_residency_bytes(const ScopedResidency& lease) {
  return lease.impl->bytes;
}

ScopedResidencyStats scoped_residency_stats() {
  auto& r = registry();
  std::lock_guard<std::mutex> lock(r.mutex);
  std::size_t active = 0;
  std::size_t attachments = 0;
#if __has_include(<Metal/Metal.hpp>)
  for (const auto* lease : r.leases) {
    if (lease->active) active += lease->bytes;
    attachments += lease->queues.size();
  }
#endif
  return {r.leases.size(), r.bytes, active, attachments};
}

void set_scoped_residency_enabled(bool enabled) {
  auto& r = registry();
  std::lock_guard<std::mutex> lock(r.mutex);
#if __has_include(<Metal/Metal.hpp>)
  if (r.enabled != enabled) {
    r.enabled = enabled;
    for (auto* lease : r.leases) lease->activate(allowed(r));
  }
#endif
  r.enabled = enabled;
}

bool scoped_residency_external_wired(bool enabled) {
  auto& r = registry();
  std::lock_guard<std::mutex> lock(r.mutex);
  const auto previous = r.external_wired;
  r.external_wired = enabled;
#if __has_include(<Metal/Metal.hpp>)
  for (auto* lease : r.leases) lease->activate(allowed(r));
#endif
  return previous;
}

bool scoped_residency_bind(const ScopedResidency& lease, bool has_target,
                          bool is_device_only, std::uint8_t device_type,
                          std::int32_t stream_index) {
#if __has_include(<Metal/Metal.hpp>)
  auto pool = mlx::core::metal::new_scoped_memory_pool();
  auto target = mlx::core::to_stream(helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index));
  if (target.device != mlx::core::Device::gpu) return false;
  auto* queue = mlx::core::metal::get_command_encoder(target)
                    .get_command_buffer()->commandQueue();
  auto& r = registry();
  std::lock_guard<std::mutex> lock(r.mutex);
  if (!allowed(r)) return false;
  auto& impl = *lease.impl;
  if (impl.retiring) return false;
  for (auto& existing : impl.queues) if (existing.get() == queue) return true;
  // MLX's sets and all model-owned sets share Metal's per-queue limit.
  auto count = mlx::core::metal::device(target.device).residency_sets().num_sets();
  for (auto* other : r.leases)
    for (auto& existing : other->queues)
      if (existing.get() == queue) count += other->sets.size();
  if (count + impl.sets.size() > 32 || impl.queues.size() >= 16) {
    impl.activate(false);
    return false;
  }
  impl.activate(true);
  impl.queues.push_back(NS::RetainPtr(queue));
  for (auto& set : impl.sets) queue->addResidencySet(set.get());
  return true;
#else
  return false;
#endif
}

}  // namespace cxx_mlx
