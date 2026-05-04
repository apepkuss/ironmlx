use crate::Dtype;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("MLX runtime error: {0}")]
    Mlx(String),

    #[error("dtype mismatch: expected {expected:?}, got {actual:?}")]
    DtypeMismatch { expected: Dtype, actual: Dtype },

    #[error("shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<i32>,
        actual: Vec<i32>,
    },

    #[error("broadcast mismatch: lhs {lhs:?} vs rhs {rhs:?}")]
    BroadcastMismatch { lhs: Vec<i32>, rhs: Vec<i32> },
}

pub type Result<T> = std::result::Result<T, Error>;

impl From<cxx::Exception> for Error {
    fn from(e: cxx::Exception) -> Self {
        Error::Mlx(e.what().to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Dtype;

    #[test]
    fn dtype_mismatch_displays() {
        let e = Error::DtypeMismatch {
            expected: Dtype::Float32,
            actual: Dtype::Int32,
        };
        assert_eq!(e.to_string(), "dtype mismatch: expected Float32, got Int32");
    }

    #[test]
    fn shape_mismatch_displays() {
        let e = Error::ShapeMismatch {
            expected: vec![2, 3],
            actual: vec![6],
        };
        assert_eq!(e.to_string(), "shape mismatch: expected [2, 3], got [6]");
    }

    #[test]
    fn broadcast_mismatch_displays() {
        let e = Error::BroadcastMismatch {
            lhs: vec![3, 1],
            rhs: vec![2, 4],
        };
        assert_eq!(
            e.to_string(),
            "broadcast mismatch: lhs [3, 1] vs rhs [2, 4]"
        );
    }
}
