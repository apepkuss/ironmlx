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

FlagSetting& qmm_any_shape_setting() {
  static FlagSetting setting;
  return setting;
}

bool resolve_flag(FlagSetting& setting, const char* env) {
  std::lock_guard<std::mutex> lock(config_mutex());
  setting.resolved = true;
  if (setting.configured) return *setting.configured;
  const auto value = std::getenv(env);
  return value != nullptr && std::strcmp(value, "1") == 0;
}

struct IntSetting {
  std::optional<int32_t> configured;
  bool resolved = false;
};

IntSetting& nax_min_keys_setting() {
  static IntSetting setting;
  return setting;
}

IntSetting& decode_min_keys_setting() {
  static IntSetting setting;
  return setting;
}

bool configure_int(IntSetting& setting, int32_t value) {
  std::lock_guard<std::mutex> lock(config_mutex());
  if (setting.resolved) return false;
  setting.configured = value;
  return true;
}

int32_t resolve_int(IntSetting& setting, const char* env) {
  std::lock_guard<std::mutex> lock(config_mutex());
  setting.resolved = true;
  if (setting.configured) return *setting.configured;
  const auto value = std::getenv(env);
  return value != nullptr ? std::atoi(value) : 0;
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

bool set_prefill_qmm_mtile_any_shape(bool enabled) {
  std::lock_guard<std::mutex> lock(config_mutex());
  auto& setting = qmm_any_shape_setting();
  if (setting.resolved) return false;
  setting.configured = enabled;
  return true;
}

bool set_prefill_d256_nax_min_keys(int32_t keys) {
  return configure_int(nax_min_keys_setting(), keys);
}

bool set_decode_d256_nax_min_keys(int32_t keys) {
  return configure_int(decode_min_keys_setting(), keys);
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

bool prefill_qmm_mtile_any_shape() {
  static const bool value =
      resolve_flag(qmm_any_shape_setting(), "IRONMLX_EXPERIMENTAL_PREFILL_QMM_MTILE_ANY_SHAPE");
  return value;
}

int32_t prefill_d256_nax_min_keys() {
  static const int32_t value =
      resolve_int(nax_min_keys_setting(), "IRONMLX_EXPERIMENTAL_PREFILL_D256_NAX_MIN_KEYS");
  return value;
}

int32_t decode_d256_nax_min_keys() {
  static const int32_t value =
      resolve_int(decode_min_keys_setting(), "IRONMLX_EXPERIMENTAL_DECODE_D256_NAX_MIN_KEYS");
  return value;
}

}  // namespace cxx_mlx
