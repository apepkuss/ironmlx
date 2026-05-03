#include "cxx_mlx_shim/transforms.h"

#include "mlx/array.h"
#include "mlx/transforms.h"

namespace cxx_mlx {

void eval_one(const MlxArray& a) {
  // mlx::core::eval takes std::vector<array> by value. The array copy ctor
  // is cheap because internal storage is refcounted.
  mlx::core::eval(std::vector<mlx::core::array>{a});
}

}  // namespace cxx_mlx
