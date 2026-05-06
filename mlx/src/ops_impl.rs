//! Operator trait impls (`Add`/`Sub`/`Mul`/`Div`/`Neg`) for `Array`.
//!
//! These return `Array` directly and **panic on error** with a formatted
//! message. Use the free functions in [`crate::ops::binary`] (or the
//! `Array::add` / `Array::sub` / `Array::mul` / `Array::div` methods, when
//! present) to get a `Result<Array>` back.
//!
//! All four reference combinations are supported: `&Array op &Array`,
//! `&Array op Array`, `Array op &Array`, `Array op Array`. The owned-form
//! impls delegate to `&Array op &Array`.
//!
//! Scalar RHS support is provided via `&Array op T: Element` and
//! `Array op T: Element`; the scalar is wrapped into a 1-element scalar
//! `Array` and broadcast.

use crate::{ops, Array, Element};

// === binary, &Array op &Array (canonical) ===

impl<'b> std::ops::Add<&'b Array> for &Array {
    type Output = Array;
    fn add(self, rhs: &'b Array) -> Array {
        ops::binary::add(self, rhs).unwrap_or_else(|e| panic!("Array + Array failed: {e}"))
    }
}

impl<'b> std::ops::Sub<&'b Array> for &Array {
    type Output = Array;
    fn sub(self, rhs: &'b Array) -> Array {
        ops::binary::subtract(self, rhs).unwrap_or_else(|e| panic!("Array - Array failed: {e}"))
    }
}

impl<'b> std::ops::Mul<&'b Array> for &Array {
    type Output = Array;
    fn mul(self, rhs: &'b Array) -> Array {
        ops::binary::multiply(self, rhs).unwrap_or_else(|e| panic!("Array * Array failed: {e}"))
    }
}

impl<'b> std::ops::Div<&'b Array> for &Array {
    type Output = Array;
    fn div(self, rhs: &'b Array) -> Array {
        ops::binary::divide(self, rhs).unwrap_or_else(|e| panic!("Array / Array failed: {e}"))
    }
}

// === binary, owned variants — delegate to canonical &op& ===

macro_rules! forward_owned_binop {
    ($trait:ident, $method:ident) => {
        impl<'a> std::ops::$trait<Array> for &Array {
            type Output = Array;
            fn $method(self, rhs: Array) -> Array {
                std::ops::$trait::$method(self, &rhs)
            }
        }
        impl<'a> std::ops::$trait<&'a Array> for Array {
            type Output = Array;
            fn $method(self, rhs: &'a Array) -> Array {
                std::ops::$trait::$method(&self, rhs)
            }
        }
        impl std::ops::$trait<Array> for Array {
            type Output = Array;
            fn $method(self, rhs: Array) -> Array {
                std::ops::$trait::$method(&self, &rhs)
            }
        }
    };
}

forward_owned_binop!(Add, add);
forward_owned_binop!(Sub, sub);
forward_owned_binop!(Mul, mul);
forward_owned_binop!(Div, div);

// === Neg ===

impl std::ops::Neg for &Array {
    type Output = Array;
    fn neg(self) -> Array {
        ops::binary::negative(self).unwrap_or_else(|e| panic!("-Array failed: {e}"))
    }
}

impl std::ops::Neg for Array {
    type Output = Array;
    fn neg(self) -> Array {
        -&self
    }
}

// === Scalar RHS ===
//
// `&Array op T` and `Array op T` for any `T: Element`, building a scalar
// 1-element Array and delegating to the `&Array op &Array` impl.

macro_rules! impl_scalar_rhs {
    ($trait:ident, $method:ident) => {
        impl<T: Element> std::ops::$trait<T> for &Array {
            type Output = Array;
            fn $method(self, rhs: T) -> Array {
                let scalar: Array = (&[rhs][..], ())
                    .try_into()
                    .unwrap_or_else(|e| panic!("scalar Array build failed: {e}"));
                std::ops::$trait::$method(self, &scalar)
            }
        }
        impl<T: Element> std::ops::$trait<T> for Array {
            type Output = Array;
            fn $method(self, rhs: T) -> Array {
                let scalar: Array = (&[rhs][..], ())
                    .try_into()
                    .unwrap_or_else(|e| panic!("scalar Array build failed: {e}"));
                std::ops::$trait::$method(&self, &scalar)
            }
        }
    };
}

impl_scalar_rhs!(Add, add);
impl_scalar_rhs!(Sub, sub);
impl_scalar_rhs!(Mul, mul);
impl_scalar_rhs!(Div, div);
