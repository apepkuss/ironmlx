//! Operator trait impls (`Add`/`Sub`/`Mul`/`Div`/`Neg`) for `Array`.
//!
//! All trait methods return `Result<Array>` (`type Output = Result<Array>`)
//! because broadcasting validation, dtype mismatch, or MLX-side errors all
//! surface immediately. User code threads `?` through expressions:
//!
//! ```ignore
//! let y = (&a + &b)?.matmul(&w)?;
//! ```
//!
//! The `forward_ref_binop!` macro generates the 3 variant impls (`Array op
//! Array`, `Array op &Array`, `&Array op Array`) by delegating to the
//! `&Array op &Array` impl (which holds the actual logic).

use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::{ops, Array, Result};

/// Generate the 3 by-value/by-ref variant impls for a binary operator.
///
/// Pattern: write the canonical `impl Trait<&Array> for &Array` body once;
/// the macro forwards `Array` operands to `&Array` via `&self` / `&other`.
macro_rules! forward_ref_binop {
    ($trait:ident, $method:ident) => {
        impl std::ops::$trait<Array> for &Array {
            type Output = Result<Array>;
            fn $method(self, other: Array) -> Self::Output {
                std::ops::$trait::$method(self, &other)
            }
        }
        impl std::ops::$trait<&Array> for Array {
            type Output = Result<Array>;
            fn $method(self, other: &Array) -> Self::Output {
                std::ops::$trait::$method(&self, other)
            }
        }
        impl std::ops::$trait<Array> for Array {
            type Output = Result<Array>;
            fn $method(self, other: Array) -> Self::Output {
                std::ops::$trait::$method(&self, &other)
            }
        }
    };
}

// === Add ===

impl Add<&Array> for &Array {
    type Output = Result<Array>;
    fn add(self, other: &Array) -> Self::Output {
        ops::add(self, other)
    }
}
forward_ref_binop!(Add, add);

// === Sub / Mul / Div ===

impl Sub<&Array> for &Array {
    type Output = Result<Array>;
    fn sub(self, other: &Array) -> Self::Output { ops::subtract(self, other) }
}
forward_ref_binop!(Sub, sub);

impl Mul<&Array> for &Array {
    type Output = Result<Array>;
    fn mul(self, other: &Array) -> Self::Output { ops::multiply(self, other) }
}
forward_ref_binop!(Mul, mul);

impl Div<&Array> for &Array {
    type Output = Result<Array>;
    fn div(self, other: &Array) -> Self::Output { ops::divide(self, other) }
}
forward_ref_binop!(Div, div);

// === Neg ===

impl Neg for &Array {
    type Output = Result<Array>;
    fn neg(self) -> Self::Output { ops::negative(self) }
}

impl Neg for Array {
    type Output = Result<Array>;
    fn neg(self) -> Self::Output { ops::negative(&self) }
}

// === Scalar RHS ===

use crate::Element;

/// Generate `impl Trait<T: Element> for &Array` and `for Array` by constructing
/// a 1-element scalar `Array` from the RHS scalar and delegating to the
/// `&Array op &Array` impl above.
///
/// Spec A4: this avoids 50 per-dtype scalar shim functions. Cost is a small
/// per-call allocation; broadcasting in MLX makes the actual op cheap.
macro_rules! impl_scalar_rhs {
    ($trait:ident, $method:ident) => {
        impl<T: Element> std::ops::$trait<T> for &Array {
            type Output = Result<Array>;
            fn $method(self, rhs: T) -> Self::Output {
                let scalar = Array::from_slice(&[rhs], &[])?;
                std::ops::$trait::$method(self, &scalar)
            }
        }
        impl<T: Element> std::ops::$trait<T> for Array {
            type Output = Result<Array>;
            fn $method(self, rhs: T) -> Self::Output {
                let scalar = Array::from_slice(&[rhs], &[])?;
                std::ops::$trait::$method(&self, &scalar)
            }
        }
    };
}

impl_scalar_rhs!(Add, add);
impl_scalar_rhs!(Sub, sub);
impl_scalar_rhs!(Mul, mul);
impl_scalar_rhs!(Div, div);
