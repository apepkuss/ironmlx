//! MLX random number generation (PRNG).
//!
//! Functional-style RNG: explicit `key(seed)` returns a PRNG key, split via
//! `split()` / `split_n()` to derive independent streams. Each distribution
//! is exposed as a builder; chain setters then call `.sample()` to materialize
//! an [`Array`](crate::Array).
//!
//! ```no_run
//! use mlx::{random, Dtype};
//!
//! # fn main() -> mlx::Result<()> {
//! let k = random::key(42)?;
//! let u = random::uniform().shape((3, 4)).dtype(Dtype::Float32).key(&k).sample()?;
//! # let _ = u;
//! # Ok(())
//! # }
//! ```

mod bernoulli;
mod bits;
mod categorical;
mod gumbel;
mod laplace;
mod multivariate_normal;
mod normal;
mod permutation;
mod randint;
mod state;
mod truncated_normal;
mod uniform;

pub use bernoulli::{bernoulli, Bernoulli};
pub use bits::{bits, Bits};
pub use categorical::{categorical, Categorical};
pub use gumbel::{gumbel, Gumbel};
pub use laplace::{laplace, Laplace};
pub use multivariate_normal::{multivariate_normal, MultivariateNormal};
pub use normal::{normal, Normal};
pub use permutation::{permutation, permutation_range, Permutation, PermutationRange};
pub use randint::{randint, RandInt};
pub use state::{key, seed, split, split_n};
pub use truncated_normal::{truncated_normal, TruncatedNormal};
pub use uniform::{uniform, Uniform};

use crate::{Array, Result};

/// Build a scalar f32 [`Array`] from `f64`. Cast happens at materialization time.
pub(crate) fn scalar_f32(v: f64) -> Result<Array> {
    Array::try_from((&[v as f32][..], ()))
}

/// Build a scalar i32 [`Array`] from `i64`. Cast happens at materialization time.
pub(crate) fn scalar_i32(v: i64) -> Result<Array> {
    Array::try_from((&[v as i32][..], ()))
}
