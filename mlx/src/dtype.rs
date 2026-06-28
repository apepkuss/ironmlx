//! Dtype mirrors `mlx::core::Dtype::Val` (u8 enum over the FFI boundary).
//!
//! The numeric values must stay in sync with `mlx/dtype.h`. The C++ shim
//! does the round-trip; this enum is the Rust-side mirror.
//!
//! Compile-time invariant: `mlx-sys/shim/src/array.cc` contains a
//! `static_assert` that `Dtype::Val::float32 == 10`. If MLX ever reorders
//! the enum, that assertion fires at the C++ build step before this Rust
//! mirror has a chance to silently drift.

#[non_exhaustive]
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
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
    #[default]
    Float32 = 10,
    Float64 = 11,
    Bfloat16 = 12,
    Complex64 = 13,
}

impl Dtype {
    pub fn byte_size(self) -> usize {
        match self {
            Dtype::Bool | Dtype::Uint8 | Dtype::Int8 => 1,
            Dtype::Uint16 | Dtype::Int16 | Dtype::Float16 | Dtype::Bfloat16 => 2,
            Dtype::Uint32 | Dtype::Int32 | Dtype::Float32 => 4,
            Dtype::Uint64 | Dtype::Int64 | Dtype::Float64 | Dtype::Complex64 => 8,
        }
    }

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

impl std::fmt::Display for Dtype {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Dtype::Bool => "bool",
            Dtype::Uint8 => "u8",
            Dtype::Uint16 => "u16",
            Dtype::Uint32 => "u32",
            Dtype::Uint64 => "u64",
            Dtype::Int8 => "i8",
            Dtype::Int16 => "i16",
            Dtype::Int32 => "i32",
            Dtype::Int64 => "i64",
            Dtype::Float16 => "f16",
            Dtype::Float32 => "f32",
            Dtype::Float64 => "f64",
            Dtype::Bfloat16 => "bf16",
            Dtype::Complex64 => "complex64",
        };
        f.write_str(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_float32() {
        assert_eq!(Dtype::default(), Dtype::Float32);
    }

    #[test]
    fn display_format() {
        assert_eq!(format!("{}", Dtype::Float32), "f32");
        assert_eq!(format!("{}", Dtype::Float16), "f16");
        assert_eq!(format!("{}", Dtype::Bfloat16), "bf16");
        assert_eq!(format!("{}", Dtype::Float64), "f64");
        assert_eq!(format!("{}", Dtype::Bool), "bool");
        assert_eq!(format!("{}", Dtype::Int8), "i8");
        assert_eq!(format!("{}", Dtype::Int32), "i32");
        assert_eq!(format!("{}", Dtype::Uint32), "u32");
        assert_eq!(format!("{}", Dtype::Complex64), "complex64");
    }

    #[test]
    fn byte_size_matches_storage_width() {
        assert_eq!(Dtype::Bool.byte_size(), 1);
        assert_eq!(Dtype::Uint8.byte_size(), 1);
        assert_eq!(Dtype::Int8.byte_size(), 1);
        assert_eq!(Dtype::Uint16.byte_size(), 2);
        assert_eq!(Dtype::Int16.byte_size(), 2);
        assert_eq!(Dtype::Float16.byte_size(), 2);
        assert_eq!(Dtype::Bfloat16.byte_size(), 2);
        assert_eq!(Dtype::Uint32.byte_size(), 4);
        assert_eq!(Dtype::Int32.byte_size(), 4);
        assert_eq!(Dtype::Float32.byte_size(), 4);
        assert_eq!(Dtype::Uint64.byte_size(), 8);
        assert_eq!(Dtype::Int64.byte_size(), 8);
        assert_eq!(Dtype::Float64.byte_size(), 8);
        assert_eq!(Dtype::Complex64.byte_size(), 8);
    }
}
