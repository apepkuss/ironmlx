use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=MLX_DIR");
    println!("cargo:rerun-if-env-changed=MLX_INCLUDE_DIR");
    println!("cargo:rerun-if-env-changed=MLX_LIB_DIR");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=shim");
    println!("cargo:rerun-if-changed=src/bridge");

    // P0 only supports the MLX_DIR discovery path. P1 adds MLX_INCLUDE_DIR/MLX_LIB_DIR
    // and pkg-config fallback; P2 adds the `bundled` feature.
    let (include_dir, lib_dir) = locate_mlx();

    // Verify the MLX install looks sane before going further.
    let array_h = include_dir.join("mlx/array.h");
    if !array_h.exists() {
        panic!(
            "MLX install at {} is missing mlx/array.h — is MLX_DIR pointing at the install prefix?",
            include_dir.display()
        );
    }

    // Link search path
    println!("cargo:rustc-link-search=native={}", lib_dir.display());

    // Mandatory link: libmlx (linker picks .a or .dylib; README documents preferring static)
    println!("cargo:rustc-link-lib=mlx");

    // Link any other static archives MLX shipped alongside libmlx.a
    // (transitive deps like libjaccl.a, libfmt.a, libgguflib.a, etc., depending on MLX build config).
    link_extra_static_archives(&lib_dir);

    // macOS frameworks MLX uses
    for fw in [
        "Metal",
        "Foundation",
        "Accelerate",
        "MetalPerformanceShaders",
        "MetalPerformanceShadersGraph",
    ] {
        println!("cargo:rustc-link-lib=framework={fw}");
    }

    // C++ standard library
    println!("cargo:rustc-link-lib=c++");

    cxx_build::bridges([
        "src/bridge/array.rs",
        "src/bridge/transforms.rs",
        "src/bridge/stream.rs",
        "src/bridge/fast.rs",
        "src/bridge/io.rs",
    ])
    .file("shim/src/array.cc")
    .file("shim/src/transforms.cc")
    .file("shim/src/stream.cc")
    .file("shim/src/fast.cc")
    .file("shim/src/io.cc")
    .include("shim/include")
    .include(&include_dir)
    .std("c++20")
    .flag_if_supported("-fvisibility=hidden")
    .compile("cxx_mlx_shim");
}

fn locate_mlx() -> (PathBuf, PathBuf) {
    let mlx_dir = env::var_os("MLX_DIR").map(PathBuf::from).unwrap_or_else(|| {
        panic!(
            "MLX_DIR is not set. Build MLX first (see docs/superpowers/plans/2026-05-03-cxx-mlx-p0-scaffold.md) \
             and export MLX_DIR=<install prefix>."
        )
    });
    let include = mlx_dir.join("include");
    let lib = mlx_dir.join("lib");
    if !include.is_dir() || !lib.is_dir() {
        panic!(
            "MLX_DIR={} does not look like an MLX install prefix (missing include/ or lib/)",
            mlx_dir.display()
        );
    }
    (include, lib)
}

/// Scan `lib_dir` for `lib<name>.a` files other than `libmlx.a` (already linked above)
/// and emit `cargo:rustc-link-lib=static=<name>` for each. This catches transitive static
/// archives MLX ships (e.g., libjaccl.a) without hardcoding a list that drifts as MLX changes.
fn link_extra_static_archives(lib_dir: &std::path::Path) {
    let entries = match fs::read_dir(lib_dir) {
        Ok(e) => e,
        Err(e) => panic!("failed to read MLX lib dir {}: {e}", lib_dir.display()),
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = match name.to_str() {
            Some(s) => s,
            None => continue,
        };
        // Match libNAME.a, exclude libmlx.a (already linked) and any non-archive.
        let stem = match name.strip_prefix("lib").and_then(|s| s.strip_suffix(".a")) {
            Some(s) => s,
            None => continue,
        };
        // Skip libmlx.a (already linked above) and a degenerate `lib.a` with empty stem.
        if stem.is_empty() || stem == "mlx" {
            continue;
        }
        println!("cargo:rustc-link-lib=static={stem}");
    }
}
