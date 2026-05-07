//! Compile-fail: missing .grid(...) on dispatch builder.

use mlx::{Array, Dtype, MetalKernel, Shape};

fn main() {
    let k = MetalKernel::builder("k")
        .inputs(&["x"])
        .outputs(&["y"])
        .source("y[0] = 1.0;")
        .build()
        .unwrap();
    let x: Array = (&[0.0_f32; 4][..], (4_i32,)).try_into().unwrap();

    let _ = k
        .dispatch_builder()
        .inputs(&[&x])
        .output_shapes(&[Shape::from((4_i32,))])
        .output_dtypes(&[Dtype::Float32])
        .threadgroup(4, 1, 1)
        .dispatch();
}
