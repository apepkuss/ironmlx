use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("MLX runtime error: {0}")]
    Mlx(String),
}

pub type Result<T> = std::result::Result<T, Error>;

impl From<cxx::Exception> for Error {
    fn from(e: cxx::Exception) -> Self {
        Error::Mlx(e.what().to_owned())
    }
}
