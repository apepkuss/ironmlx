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
    println!("cargo:rerun-if-changed=shaders");

    compile_prefill_shaders();

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
        "src/bridge/compile.rs",
        "src/bridge/conv.rs",
        "src/bridge/fft.rs",
        "src/bridge/transforms.rs",
        "src/bridge/stream.rs",
        "src/bridge/fast.rs",
        "src/bridge/io.rs",
        "src/bridge/metal.rs",
        "src/bridge/memory.rs",
        "src/bridge/quantization.rs",
        "src/bridge/random.rs",
    ])
    .file("shim/src/array.cc")
    .file("shim/src/compile.cc")
    .file("shim/src/conv.cc")
    .file("shim/src/fft.cc")
    .file("shim/src/transforms.cc")
    .file("shim/src/stream.cc")
    .file("shim/src/experiment_config.cc")
    .file("shim/src/fast.cc")
    .file("shim/src/prefill_d256_nax.cc")
    .file("shim/src/prefill_masked_softmax.cc")
    .file("shim/src/prefill_qmm_mtile.cc")
    .file("shim/src/io.cc")
    .file("shim/src/metal.cc")
    .file("shim/src/memory.cc")
    .file("shim/src/quantization.cc")
    .file("shim/src/random.cc")
    .include("shim/include")
    .include(&include_dir)
    .include(include_dir.join("metal_cpp"))
    .std("c++20")
    .flag_if_supported("-fvisibility=hidden")
    .compile("cxx_mlx_shim");
}

/// Precompiled prefill kernels for Apple GPU generation 17+ (NAX). Built from
/// `shaders/` with the pinned MLX kernel headers in `shaders/vendor` (QMM
/// M-tile: MLX 0.32.2; D256 attention: MLX 0.32.3) using the same flags as
/// the qualified candidate, so the output is byte-identical to it. The
/// libraries are embedded in the crate (`mlx_sys::shaders`) and loaded at
/// runtime only when the device profile enables them.
fn compile_prefill_shaders() {
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR"));
    let shaders = manifest.join("shaders");
    for (name, source, mlx_headers) in [
        ("prefill_qmm_mtile", "qmm_mtile_probe.metal", "mlx-0.32.2"),
        ("prefill_d256_nax", "nax_dsplit_probe.metal", "mlx-0.32.3"),
    ] {
        let air = out_dir.join(format!("{name}.air"));
        let library = out_dir.join(format!("{name}.metallib"));
        let include = shaders.join("vendor").join(mlx_headers).join("include");
        run_xcrun(&[
            "metal".as_ref(),
            "-std=metal4.0".as_ref(),
            "-O2".as_ref(),
            "-fno-fast-math".as_ref(),
            "-mmacosx-version-min=26.2".as_ref(),
            "-I".as_ref(),
            include.as_os_str(),
            "-c".as_ref(),
            shaders.join(source).as_os_str(),
            "-o".as_ref(),
            air.as_os_str(),
        ]);
        run_xcrun(&[
            "metallib".as_ref(),
            air.as_os_str(),
            "-o".as_ref(),
            library.as_os_str(),
        ]);
    }
}

fn run_xcrun(args: &[&std::ffi::OsStr]) {
    let status = std::process::Command::new("xcrun")
        .args(args)
        .status()
        .unwrap_or_else(|error| panic!("failed to run xcrun {args:?}: {error}"));
    if !status.success() {
        panic!(
            "xcrun {args:?} failed ({status}). The Metal toolchain is required to build \
             the prefill kernels (install it with `xcodebuild -downloadComponent MetalToolchain`)."
        );
    }
}

fn locate_mlx() -> (PathBuf, PathBuf) {
    let mlx_dir = env::var_os("MLX_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            panic!(
                "MLX_DIR is not set. Build MLX first (see docs/installation.md) \
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
