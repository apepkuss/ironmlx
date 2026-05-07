//! Compile-fail: missing .inputs(...) on dispatch builder.

use mlx::{Dtype, MetalKernel, Shape};

fn main() {
    let k = MetalKernel::builder("k")
        .inputs(&["x"])
        .outputs(&["y"])
        .source("y[0] = 1.0;")
        .build()
        .unwrap();

    // ERROR: .inputs() not called → .dispatch() not callable
    let _ = k
        .dispatch_builder()
        .output_shapes(&[Shape::from((4_i32,))])
        .output_dtypes(&[Dtype::Float32])
        .grid(4, 1, 1)
        .threadgroup(4, 1, 1)
        .dispatch();
}
