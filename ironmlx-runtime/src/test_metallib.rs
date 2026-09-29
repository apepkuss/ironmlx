//! Process-wide Metal library setup for tests.
//!
//! MLX initializes its Metal backend globally on first use. The Rust test
//! harness may start any test first, so per-test setup cannot reliably run
//! before another test creates an MLX array. Install the test metallib from a
//! process constructor, before libtest starts scheduling tests.

use std::path::Path;
use std::sync::OnceLock;

static CONFIGURED: OnceLock<()> = OnceLock::new();

pub(crate) fn configure_from_mlx_dir() {
    CONFIGURED.get_or_init(|| {
        let mlx_dir = std::env::var_os("MLX_DIR")
            .expect("MLX_DIR is required to run ironmlx-runtime MLX tests");
        let path = Path::new(&mlx_dir).join("lib/mlx.metallib");
        assert!(
            path.is_file(),
            "MLX test metallib does not exist: {}",
            path.display()
        );
        let path = path
            .to_str()
            .expect("MLX test metallib path must be valid UTF-8");
        mlx::metal::set_metallib_path(path).expect("configure MLX test metallib");
    });
}

#[ctor::ctor(unsafe)]
fn configure_before_test_harness() {
    configure_from_mlx_dir();
}
