//! Element-wise unary ops.
//!
//! All return `Result<Array>` because dtype mismatches (e.g. `sqrt` on integer
//! types) raise MLX exceptions that we surface as `Error::Mlx`.

use crate::{Array, Error, Result};

/// Macro to define a unary op delegating to a single shim function.
macro_rules! unary_op {
    ($name:ident, $shim:ident, $doc:literal) => {
        #[doc = $doc]
        pub fn $name(a: &Array) -> Result<Array> {
            let inner = mlx_sys::array::ffi::$shim(a.as_inner()).map_err(Error::from)?;
            Ok(Array::from_inner(inner))
        }
    };
}

unary_op!(exp, array_exp, "Element-wise natural exponential.");
unary_op!(log, array_log, "Element-wise natural logarithm.");
unary_op!(sqrt, array_sqrt, "Element-wise square root.");
unary_op!(tanh, array_tanh, "Element-wise hyperbolic tangent.");
unary_op!(
    sigmoid,
    array_sigmoid,
    "Element-wise sigmoid (1 / (1 + exp(-x)))."
);
unary_op!(square, array_square, "Element-wise x^2.");
unary_op!(
    rsqrt,
    array_rsqrt,
    "Element-wise 1/sqrt(x). Used in attention scaling."
);
unary_op!(erf, array_erf, "Element-wise error function. Used in GELU.");
unary_op!(reciprocal, array_reciprocal, "Element-wise 1/x.");
