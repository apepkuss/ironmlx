#include "cxx_mlx_shim/experiment_config.h"

#include <cstdlib>
#include <cstring>
#include <mutex>
#include <optional>

namespace cxx_mlx {
namespace {

std::mutex& config_mutex() {
  static std::mutex mutex;
  return mutex;
}

struct LibrarySetting {
  std::optional<std::string> configured;
  bool resolved = false;
};

LibrarySetting& qmm_setting() {
  static LibrarySetting setting;
  return setting;
}

LibrarySetting& nax_setting() {
  static LibrarySetting setting;
  return setting;
}

struct FlagSetting {
  std::optional<bool> configured;
  bool resolved = false;
};

FlagSetting& softmax_setting() {
  static FlagSetting setting;
  return setting;
}

std::string env_or_empty(const char* name) {
  const auto value = std::getenv(name);
  return value ? std::string(value) : std::string();
}

bool configure(LibrarySetting& setting, rust::Str path) {
  std::lock_guard<std::mutex> lock(config_mutex());
  if (setting.resolved) return false;
  setting.configured = std::string(path);
  return true;
}

std::string resolve(LibrarySetting& setting, const char* env) {
  std::lock_guard<std::mutex> lock(config_mutex());
  setting.resolved = true;
  return setting.configured ? *setting.configured : env_or_empty(env);
}

}  // namespace

bool set_prefill_qmm_mtile_library(rust::Str path) {
  return configure(qmm_setting(), path);
}

bool set_prefill_d256_nax_library(rust::Str path) {
  return configure(nax_setting(), path);
}

bool set_prefill_masked_softmax(bool enabled) {
  std::lock_guard<std::mutex> lock(config_mutex());
  auto& setting = softmax_setting();
  if (setting.resolved) return false;
  setting.configured = enabled;
  return true;
}

const std::string& prefill_qmm_mtile_library() {
  static const std::string value =
      resolve(qmm_setting(), "IRONMLX_EXPERIMENTAL_PREFILL_QMM_MTILE_METALLIB");
  return value;
}

const std::string& prefill_d256_nax_library() {
  static const std::string value =
      resolve(nax_setting(), "IRONMLX_EXPERIMENTAL_PREFILL_D256_NAX_METALLIB");
  return value;
}

bool prefill_masked_softmax_requested() {
  static const bool value = [] {
    std::lock_guard<std::mutex> lock(config_mutex());
    auto& setting = softmax_setting();
    setting.resolved = true;
    if (setting.configured) return *setting.configured;
    const auto env = std::getenv("IRONMLX_EXPERIMENTAL_PREFILL_MASKED_SOFTMAX");
    return env != nullptr && std::strcmp(env, "1") == 0;
  }();
  return value;
}

}  // namespace cxx_mlx
