//! Bridge for `mlx::core::metal::{start,stop}_capture` (debug/profiling) and
//! `mlx::core::metal::device_info()['architecture']` (device-arch lookup
//! tables — used by ironmlx-p8a-stage9 to pick per-chip kernel tiles).
//!
//! All functions in this bridge can throw on the C++ side (driver /
//! capture-manager failure for capture; missing Metal backend or unexpected
//! variant type for architecture), so they're declared `Result<T>` per the
//! project's "shim-can-throw → cxx Result" rule (`bridge/mod.rs`).

#[cxx::bridge(namespace = "cxx_mlx")]
mod ffi_bridge {
    unsafe extern "C++" {
        include!("cxx_mlx_shim/metal.h");
        include!("cxx_mlx_shim/experiment_config.h");

        /// Open an Xcode-compatible `.gputrace` file at `path` and start
        /// capturing every Metal command submitted on the default device.
        ///
        /// Throws on driver / capture-manager failure: missing Xcode
        /// entitlement (`MTL_CAPTURE_ENABLED=1`), path not writable, capture
        /// already running, or no Metal device available.
        fn start_capture(path: &str) -> Result<()>;

        /// Finalize the in-progress capture. Throws if no capture is active.
        fn stop_capture() -> Result<()>;

        /// Configure the exact MLX metallib loaded by this process.
        fn set_metallib_path(path: &str) -> Result<()>;

        /// Return the Metal device's architecture name (e.g. `"apple_g13s"`
        /// for an M1 Pro 16-core GPU, `"apple_g15p"` for M3 Pro), as
        /// reported by `MTLDevice.architecture.name` and exposed via
        /// `mlx::core::metal::device_info()['architecture']`.
        ///
        /// Throws if the Metal backend isn't available on this system or
        /// the entry's variant is unexpectedly not a string (defensive —
        /// MLX always stores it as `std::string`).
        fn device_architecture() -> Result<String>;

        /// Whether MLX may use its NAX (neural accelerator) kernels on this
        /// device and OS (`mlx::core::metal::is_nax_available()`), the
        /// condition MLX checks before choosing one. Throws if the Metal
        /// backend isn't available.
        fn nax_available() -> Result<bool>;

        /// Configure the QMM M-tile prefill library path (empty = off).
        /// Returns false if the setting was already read.
        fn set_prefill_qmm_mtile_library(path: &str) -> bool;

        /// Configure the D256 NAX prefill attention library path (empty =
        /// off). Returns false if the setting was already read.
        fn set_prefill_d256_nax_library(path: &str) -> bool;

        /// Fewest keys at which a causal chunk of any GQA layout takes the
        /// D256 NAX attention (0: only the dense 27B layout). Returns false
        /// if the setting was already read.
        fn set_prefill_d256_nax_min_keys(keys: i32) -> bool;

        /// Fewest keys at which a query block of at most 8 rows takes the
        /// split-K D256 NAX attention (0: never). Returns false if the
        /// setting was already read.
        fn set_decode_d256_nax_min_keys(keys: i32) -> bool;

        /// The split-K D256 NAX minimum key count if that attention can run
        /// (library configured, NAX available), else 0. Resolves the
        /// settings: call only after configuring them.
        fn decode_d256_nax_active_min_keys() -> i32;

        /// Enable or disable the masked causal softmax prefill fallback.
        /// Returns false if the setting was already read.
        fn set_prefill_masked_softmax(enabled: bool) -> bool;

        /// Run the BM128 QMM for every affine4/g64 prefill chunk shape.
        /// Returns false if the setting was already read.
        fn set_prefill_qmm_mtile_any_shape(enabled: bool) -> bool;
    }
}

pub mod ffi {
    pub use super::ffi_bridge::{
        decode_d256_nax_active_min_keys, device_architecture, nax_available,
        set_decode_d256_nax_min_keys, set_metallib_path, set_prefill_d256_nax_library,
        set_prefill_d256_nax_min_keys, set_prefill_masked_softmax, set_prefill_qmm_mtile_any_shape,
        set_prefill_qmm_mtile_library, start_capture, stop_capture,
    };
}
