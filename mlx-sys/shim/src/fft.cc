#include "cxx_mlx_shim/fft.h"
#include "cxx_mlx_shim/shim_helpers.h"
#include "mlx/fft.h"
#include <stdexcept>
namespace cxx_mlx {
std::unique_ptr<MlxArray> ops_real_fft(const MlxArray& input, int32_t n, int32_t axis, bool inverse, uint8_t norm,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = helpers::decode_stream_or_device(has_target, is_device_only, device_type, stream_index);
  if (norm > 2) throw std::invalid_argument("invalid FFT normalization");
  auto normalization = norm == 0 ? mlx::core::fft::FFTNorm::Backward :
      norm == 1 ? mlx::core::fft::FFTNorm::Ortho : mlx::core::fft::FFTNorm::Forward;
  return std::make_unique<MlxArray>(inverse ? mlx::core::fft::irfft(input, n, axis, normalization, target)
      : mlx::core::fft::rfft(input, n, axis, normalization, target));
}
}
