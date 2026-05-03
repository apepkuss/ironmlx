//! Dtype mirrors `mlx::core::Dtype::Val` (u8 enum over the FFI boundary).
//!
//! The numeric values must stay in sync with `mlx/dtype.h`. The C++ shim
//! does the round-trip; this enum is the Rust-side mirror.
//!
//! Compile-time invariant: `mlx-sys/shim/src/array.cc` contains a
//! `static_assert` that `Dtype::Val::float32 == 10`. If MLX ever reorders
//! the enum, that assertion fires at the C++ build step before this Rust
//! mirror has a chance to silently drift.

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dtype {
    Bool = 0,
    Uint8 = 1,
    Uint16 = 2,
    Uint32 = 3,
    Uint64 = 4,
    Int8 = 5,
    Int16 = 6,
    Int32 = 7,
    Int64 = 8,
    Float16 = 9,
    Float32 = 10,
    Float64 = 11,
    Bfloat16 = 12,
    Complex64 = 13,
}

impl Dtype {
    pub(crate) fn as_u8(self) -> u8 {
        self as u8
    }

    pub(crate) fn from_u8(v: u8) -> Result<Self, crate::Error> {
        match v {
            0 => Ok(Dtype::Bool),
            1 => Ok(Dtype::Uint8),
            2 => Ok(Dtype::Uint16),
            3 => Ok(Dtype::Uint32),
            4 => Ok(Dtype::Uint64),
            5 => Ok(Dtype::Int8),
            6 => Ok(Dtype::Int16),
            7 => Ok(Dtype::Int32),
            8 => Ok(Dtype::Int64),
            9 => Ok(Dtype::Float16),
            10 => Ok(Dtype::Float32),
            11 => Ok(Dtype::Float64),
            12 => Ok(Dtype::Bfloat16),
            13 => Ok(Dtype::Complex64),
            other => Err(crate::Error::Mlx(format!("unknown Dtype::Val={other}"))),
        }
    }
}
