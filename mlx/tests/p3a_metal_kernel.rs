//! Integration tests for P3a — `MetalKernel` end-to-end via the safe API.
//!
//! These tests serve as canonical usage examples for the public surface
//! exposed in T4-T5: `MetalKernel::builder(...).build()` and
//! `kernel.dispatch_builder()...dispatch()`. They run real Metal kernels
//! on the GPU, so `MLX_DIR=$HOME/.local/mlx` must be set when running.

use mlx::{Array, Dtype, MetalKernel, Shape};

#[test]
fn simple_add_kernel() {
    let kernel = MetalKernel::builder("simple_add")
        .inputs(&["x"])
        .outputs(&["y"])
        .source("uint gid = thread_position_in_grid.x; y[gid] = x[gid] + 1.0;")
        .build()
        .expect("compile");

    let x: Array = (&[1.0_f32, 2.0, 3.0, 4.0][..], (4_i32,))
        .try_into()
        .unwrap();
    let mut outputs = kernel
        .dispatch_builder()
        .inputs(&[&x])
        .output_shapes(&[Shape::from((4_i32,))])
        .output_dtypes(&[Dtype::Float32])
        .grid(4, 1, 1)
        .threadgroup(4, 1, 1)
        .dispatch()
        .expect("dispatch");

    assert_eq!(outputs.len(), 1);
    let y = outputs.take_at(0).expect("take 0");
    assert_eq!(y.shape().as_slice(), &[4]);
    assert_eq!(y.to_vec::<f32>().unwrap(), vec![2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn multi_output_kernel() {
    // Two outputs: y = x*2, z = x+10
    let kernel = MetalKernel::builder("multi_out")
        .inputs(&["x"])
        .outputs(&["y", "z"])
        .source(
            "uint gid = thread_position_in_grid.x; \
             y[gid] = x[gid] * 2.0; \
             z[gid] = x[gid] + 10.0;",
        )
        .build()
        .expect("compile");

    let x: Array = (&[1.0_f32, 2.0, 3.0, 4.0][..], (4_i32,))
        .try_into()
        .unwrap();
    let mut outputs = kernel
        .dispatch_builder()
        .inputs(&[&x])
        .output_shapes(&[Shape::from((4_i32,)), Shape::from((4_i32,))])
        .output_dtypes(&[Dtype::Float32, Dtype::Float32])
        .grid(4, 1, 1)
        .threadgroup(4, 1, 1)
        .dispatch()
        .expect("dispatch");

    assert_eq!(outputs.len(), 2);

    // Take in declared order: y first (index 0), then z (which shifts to index 0
    // after y is erased by take_at's erase-on-take semantics).
    let y = outputs.take_at(0).expect("take 0");
    assert_eq!(y.to_vec::<f32>().unwrap(), vec![2.0, 4.0, 6.0, 8.0]);

    let z = outputs.take_at(0).expect("take 1 (now at 0)");
    assert_eq!(z.to_vec::<f32>().unwrap(), vec![11.0, 12.0, 13.0, 14.0]);
}

#[test]
fn template_int_substitution() {
    // Use a template to multiply by a compile-time constant.
    let kernel = MetalKernel::builder("template_mul")
        .inputs(&["x"])
        .outputs(&["y"])
        .source("uint gid = thread_position_in_grid.x; y[gid] = x[gid] * static_cast<float>(MUL);")
        .build()
        .expect("compile");

    let x: Array = (&[1.0_f32, 2.0, 3.0][..], (3_i32,)).try_into().unwrap();
    let mut outputs = kernel
        .dispatch_builder()
        .inputs(&[&x])
        .output_shapes(&[Shape::from((3_i32,))])
        .output_dtypes(&[Dtype::Float32])
        .grid(3, 1, 1)
        .threadgroup(3, 1, 1)
        .template_int("MUL", 7)
        .dispatch()
        .expect("dispatch");

    let y = outputs.take_at(0).expect("take 0");
    assert_eq!(y.to_vec::<f32>().unwrap(), vec![7.0, 14.0, 21.0]);
}

#[test]
fn output_count_mismatch_errors() {
    // Kernel declares 1 output but dispatch passes 2 shapes — should error.
    let kernel = MetalKernel::builder("one_out")
        .inputs(&["x"])
        .outputs(&["y"])
        .source("uint gid = thread_position_in_grid.x; y[gid] = x[gid];")
        .build()
        .expect("compile");

    let x: Array = (&[0.0_f32; 4][..], (4_i32,)).try_into().unwrap();
    let r = kernel
        .dispatch_builder()
        .inputs(&[&x])
        .output_shapes(&[Shape::from((4_i32,)), Shape::from((4_i32,))]) // 2 shapes
        .output_dtypes(&[Dtype::Float32, Dtype::Float32])
        .grid(4, 1, 1)
        .threadgroup(4, 1, 1)
        .dispatch();

    assert!(r.is_err(), "expected error from count mismatch");
    let msg = format!("{}", r.err().unwrap());
    assert!(msg.contains("output_shapes count"), "msg: {msg}");
}

#[test]
fn clone_kernel_dispatches_independently() {
    let kernel = MetalKernel::builder("add_two")
        .inputs(&["x"])
        .outputs(&["y"])
        .source("uint gid = thread_position_in_grid.x; y[gid] = x[gid] + 2.0;")
        .build()
        .expect("compile");

    let kernel2 = kernel.clone();
    let x: Array = (&[1.0_f32; 4][..], (4_i32,)).try_into().unwrap();

    let mut o1 = kernel
        .dispatch_builder()
        .inputs(&[&x])
        .output_shapes(&[Shape::from((4_i32,))])
        .output_dtypes(&[Dtype::Float32])
        .grid(4, 1, 1)
        .threadgroup(4, 1, 1)
        .dispatch()
        .unwrap();

    let mut o2 = kernel2
        .dispatch_builder()
        .inputs(&[&x])
        .output_shapes(&[Shape::from((4_i32,))])
        .output_dtypes(&[Dtype::Float32])
        .grid(4, 1, 1)
        .threadgroup(4, 1, 1)
        .dispatch()
        .unwrap();

    assert_eq!(
        o1.take_at(0).unwrap().to_vec::<f32>().unwrap(),
        o2.take_at(0).unwrap().to_vec::<f32>().unwrap()
    );
}
