//! Flash-Next computation and bounded-storage building blocks.
//!
//! The execution architecture remains unregistered until its complete model,
//! request state and reference gates pass. Importing files does not certify
//! inference. See the pinned component fixtures for the reference identities.
pub mod attention;
#[cfg(unix)]
pub mod cache;
pub mod config;
#[cfg(unix)]
mod dflash2;
pub mod hyper;
pub mod math;
#[cfg(unix)]
mod model;
#[cfg(unix)]
mod paged_moe;
#[cfg(unix)]
pub mod ple;
pub mod ple_index;
#[cfg(unix)]
pub mod ple_rows;
mod qsa_kernel;
#[cfg(unix)]
pub mod text_model;
#[cfg(unix)]
mod vision;

#[cfg(unix)]
pub mod load_plan;

pub(crate) mod gdn_kernel;
