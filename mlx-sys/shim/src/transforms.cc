#include "cxx_mlx_shim/transforms.h"

#include "mlx/array.h"
#include "mlx/transforms.h"

namespace cxx_mlx {

void eval_one(const MlxArray& a) {
  // mlx::core::eval takes std::vector<array> by value. The array copy ctor
  // is cheap because internal storage is refcounted.
  mlx::core::eval(std::vector<mlx::core::array>{a});
}

void array_wait(const MlxArray& a) {
  // array::wait() is non-const (sets status, detaches event), but Array
  // copies share array_desc_, so wait()ing on a copy mutates the same
  // underlying state. Mirror eval_one's pattern.
  auto copy = a;
  copy.wait();
}

}  // namespace cxx_mlx
