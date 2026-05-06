//! Internal declarative macro for the ops layer.
//!
//! [`op_with_stream!`] generates a pair of public functions from a single
//! definition: a default variant (no stream arg) and a `*_on` variant
//! taking `impl Into<StreamOrDevice>`. Both delegate to the same FFI
//! bridge function with 4 trailing stream params produced by
//! [`StreamOrDevice::encode`](crate::StreamOrDevice::encode).
//!
//! ## Form
//!
//! ```ignore
//! op_with_stream! {
//!     /// Doc comment for the op.
//!     pub fn add(a: &Array, b: &Array) -> Result<Array> {
//!         // optional Rust-side prologue (validation, conversions). Runs
//!         // before the FFI call. May early-return via `?` because the
//!         // surrounding fn has return type `Result<_>`.
//!         crate::broadcast::broadcast_shape(a.shape().as_slice(), b.shape().as_slice())?;
//!     } => mlx_sys::array::ffi::array_add(a.as_inner(), b.as_inner());
//! }
//! ```
//!
//! Generates:
//!
//! ```ignore
//! pub fn add(a: &Array, b: &Array) -> Result<Array> { add_on(a, b, ()) }
//! pub fn add_on(
//!     a: &Array,
//!     b: &Array,
//!     target: impl Into<StreamOrDevice>,
//! ) -> Result<Array> {
//!     // prologue here (broadcast_shape...)
//!     let (has, dev_only, dev_t, idx) = target.into().encode();
//!     let inner = mlx_sys::array::ffi::array_add(
//!         a.as_inner(), b.as_inner(),
//!         has, dev_only, dev_t, idx,
//!     ).map_err(crate::Error::from)?;
//!     Ok(crate::Array::from_inner(inner))
//! }
//! ```
//!
//! The empty-prologue case is also accepted:
//!
//! ```ignore
//! op_with_stream! {
//!     /// Element-wise negation `-a`.
//!     pub fn negative(a: &Array) -> Result<Array>
//!         => mlx_sys::array::ffi::array_negative(a.as_inner());
//! }
//! ```

/// Generates default + `_on` variants of an op (see module docs).
#[macro_export]
macro_rules! op_with_stream {
    // Form 1: with explicit prologue block (for ops doing Rust-side validation).
    (
        $(#[$attr:meta])*
        $vis:vis fn $name:ident ( $($arg:ident : $ty:ty),* $(,)? ) -> $ret:ty
            { $($prologue:tt)* }
            => $($bridge_seg:ident)::+ ( $($bridge_arg:expr),* $(,)? ) ;
    ) => {
        $crate::__paste::paste! {
            $(#[$attr])*
            $vis fn $name( $($arg: $ty),* ) -> $ret {
                [<$name _on>]( $($arg,)* () )
            }

            #[doc = concat!(
                "Stream-targeted variant of [`", stringify!($name),
                "`]. Pass `()` for the current default stream, a `Stream`, or a `Device`."
            )]
            $vis fn [<$name _on>](
                $($arg: $ty,)*
                target: impl Into<$crate::StreamOrDevice>,
            ) -> $ret {
                $($prologue)*
                let (has, dev_only, dev_t, idx) = target.into().encode();
                let inner = $($bridge_seg)::+ ( $($bridge_arg,)* has, dev_only, dev_t, idx )
                    .map_err($crate::Error::from)?;
                Ok($crate::Array::from_inner(inner))
            }
        }
    };

    // Form 2: no prologue (purely FFI forwarder).
    (
        $(#[$attr:meta])*
        $vis:vis fn $name:ident ( $($arg:ident : $ty:ty),* $(,)? ) -> $ret:ty
            => $($bridge_seg:ident)::+ ( $($bridge_arg:expr),* $(,)? ) ;
    ) => {
        $crate::op_with_stream! {
            $(#[$attr])*
            $vis fn $name ( $($arg : $ty),* ) -> $ret { } => $($bridge_seg)::+ ( $($bridge_arg),* );
        }
    };
}
