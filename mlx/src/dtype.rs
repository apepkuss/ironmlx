use mlx_sys as sys;

/// Data type of an [`Array`](crate::Array).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dtype {
    Bool,
    Uint8,
    Uint16,
    Uint32,
    Uint64,
    Int8,
    Int16,
    Int32,
    Int64,
    Float16,
    Float32,
    Float64,
    Bfloat16,
    Complex64,
}

impl Dtype {
    pub(crate) fn from_raw(raw: sys::mlx_dtype) -> Self {
        match raw {
            sys::mlx_dtype_MLX_BOOL => Dtype::Bool,
            sys::mlx_dtype_MLX_UINT8 => Dtype::Uint8,
            sys::mlx_dtype_MLX_UINT16 => Dtype::Uint16,
            sys::mlx_dtype_MLX_UINT32 => Dtype::Uint32,
            sys::mlx_dtype_MLX_UINT64 => Dtype::Uint64,
            sys::mlx_dtype_MLX_INT8 => Dtype::Int8,
            sys::mlx_dtype_MLX_INT16 => Dtype::Int16,
            sys::mlx_dtype_MLX_INT32 => Dtype::Int32,
            sys::mlx_dtype_MLX_INT64 => Dtype::Int64,
            sys::mlx_dtype_MLX_FLOAT16 => Dtype::Float16,
            sys::mlx_dtype_MLX_FLOAT32 => Dtype::Float32,
            sys::mlx_dtype_MLX_FLOAT64 => Dtype::Float64,
            sys::mlx_dtype_MLX_BFLOAT16 => Dtype::Bfloat16,
            sys::mlx_dtype_MLX_COMPLEX64 => Dtype::Complex64,
            _ => panic!("Unknown mlx_dtype value: {}", raw),
        }
    }

    pub(crate) fn to_raw(self) -> sys::mlx_dtype {
        match self {
            Dtype::Bool => sys::mlx_dtype_MLX_BOOL,
            Dtype::Uint8 => sys::mlx_dtype_MLX_UINT8,
            Dtype::Uint16 => sys::mlx_dtype_MLX_UINT16,
            Dtype::Uint32 => sys::mlx_dtype_MLX_UINT32,
            Dtype::Uint64 => sys::mlx_dtype_MLX_UINT64,
            Dtype::Int8 => sys::mlx_dtype_MLX_INT8,
            Dtype::Int16 => sys::mlx_dtype_MLX_INT16,
            Dtype::Int32 => sys::mlx_dtype_MLX_INT32,
            Dtype::Int64 => sys::mlx_dtype_MLX_INT64,
            Dtype::Float16 => sys::mlx_dtype_MLX_FLOAT16,
            Dtype::Float32 => sys::mlx_dtype_MLX_FLOAT32,
            Dtype::Float64 => sys::mlx_dtype_MLX_FLOAT64,
            Dtype::Bfloat16 => sys::mlx_dtype_MLX_BFLOAT16,
            Dtype::Complex64 => sys::mlx_dtype_MLX_COMPLEX64,
        }
    }

    /// Size of one element in bytes.
    pub fn size_of(self) -> usize {
        match self {
            Dtype::Bool | Dtype::Uint8 | Dtype::Int8 => 1,
            Dtype::Uint16 | Dtype::Int16 | Dtype::Float16 | Dtype::Bfloat16 => 2,
            Dtype::Uint32 | Dtype::Int32 | Dtype::Float32 => 4,
            Dtype::Uint64 | Dtype::Int64 | Dtype::Float64 | Dtype::Complex64 => 8,
        }
    }
}
