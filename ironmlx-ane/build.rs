fn main() {
    println!("cargo:rerun-if-changed=native/ane.cpp");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        return;
    }
    cc::Build::new()
        .cpp(true)
        .file("native/ane.cpp")
        .flag("-std=c++17")
        .flag("-fno-objc-arc")
        .compile("ironmlx_ane_native");
    for framework in ["CoreFoundation", "IOSurface"] {
        println!("cargo:rustc-link-lib=framework={framework}");
    }
    println!("cargo:rustc-link-lib=objc");
}
