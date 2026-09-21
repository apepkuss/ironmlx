//! 1D convolution: `mlx::core::conv1d`.
//!
//! Input layout `[N, L, C_in]`, weight layout `[C_out, K, C_in / groups]`,
//! output `[N, L_out, C_out]` where `L_out = (L + 2*padding - dilation*(K-1) - 1) / stride + 1`.
//!
//! For depthwise convolution, set `groups = C_in == C_out`.

use crate::{Array, Error, Result, StreamOrDevice};

/// 1D convolution with default stream.
pub fn conv1d(
    input: &Array,
    weight: &Array,
    stride: i32,
    padding: i32,
    dilation: i32,
    groups: i32,
) -> Result<Array> {
    conv1d_on(input, weight, stride, padding, dilation, groups, ())
}

/// Stream-targeted 1D convolution.
#[allow(clippy::too_many_arguments)]
pub fn conv1d_on(
    input: &Array,
    weight: &Array,
    stride: i32,
    padding: i32,
    dilation: i32,
    groups: i32,
    target: impl Into<StreamOrDevice>,
) -> Result<Array> {
    let (has, dev_only, dev_t, idx) = target.into().encode();
    // SAFETY: input/weight borrows valid for the call duration.
    let inner = unsafe {
        mlx_sys::conv::ffi::ops_conv1d(
            input.as_inner(),
            weight.as_inner(),
            stride,
            padding,
            dilation,
            groups,
            has,
            dev_only,
            dev_t,
            idx,
        )
    }
    .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// 2D convolution in NHWC layout; weights are [out, height, width, in/groups].
#[allow(clippy::too_many_arguments)]
pub fn conv2d(
    input: &Array,
    weight: &Array,
    stride: (i32, i32),
    padding: (i32, i32),
    dilation: (i32, i32),
    groups: i32,
) -> Result<Array> {
    conv2d_on(input, weight, stride, padding, dilation, groups, ())
}

/// Stream-targeted conv2d.
#[allow(clippy::too_many_arguments)]
pub fn conv2d_on(
    input: &Array,
    weight: &Array,
    stride: (i32, i32),
    padding: (i32, i32),
    dilation: (i32, i32),
    groups: i32,
    target: impl Into<StreamOrDevice>,
) -> Result<Array> {
    let (has, dev_only, dev_t, idx) = target.into().encode();
    // SAFETY: input and weight are borrowed for the duration of the call.
    let inner = unsafe {
        mlx_sys::conv::ffi::ops_conv2d(
            input.as_inner(),
            weight.as_inner(),
            stride.0,
            stride.1,
            padding.0,
            padding.1,
            dilation.0,
            dilation.1,
            groups,
            has,
            dev_only,
            dev_t,
            idx,
        )
    }
    .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

/// Transposed 1D convolution in NLC layout; weights are [out, kernel, in/groups].
#[allow(clippy::too_many_arguments)]
pub fn conv_transpose1d(
    input: &Array,
    weight: &Array,
    stride: i32,
    padding: i32,
    dilation: i32,
    output_padding: i32,
    groups: i32,
) -> Result<Array> {
    conv_transpose1d_on(
        input,
        weight,
        stride,
        padding,
        dilation,
        output_padding,
        groups,
        (),
    )
}

/// Stream-targeted conv_transpose1d.
#[allow(clippy::too_many_arguments)]
pub fn conv_transpose1d_on(
    input: &Array,
    weight: &Array,
    stride: i32,
    padding: i32,
    dilation: i32,
    output_padding: i32,
    groups: i32,
    target: impl Into<StreamOrDevice>,
) -> Result<Array> {
    let (has, dev_only, dev_t, idx) = target.into().encode();
    // SAFETY: input and weight are borrowed for the duration of the call.
    let inner = unsafe {
        mlx_sys::conv::ffi::ops_conv_transpose1d(
            input.as_inner(),
            weight.as_inner(),
            stride,
            padding,
            dilation,
            output_padding,
            groups,
            has,
            dev_only,
            dev_t,
            idx,
        )
    }
    .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Dtype;

    #[test]
    fn conv1d_shape_basic() {
        // input: [1, 8, 2], weight: [4, 3, 2] (C_out=4, K=3, C_in=2), groups=1
        let input = Array::zeros((1_i32, 8, 2), Dtype::Float32).unwrap();
        let weight = Array::zeros((4_i32, 3, 2), Dtype::Float32).unwrap();
        let out = conv1d(&input, &weight, 1, 0, 1, 1).expect("conv1d");
        assert_eq!(out.shape().as_slice(), &[1, 6, 4]);
        assert_eq!(out.dtype(), Dtype::Float32);
    }

    #[test]
    fn conv1d_depthwise_shape() {
        // depthwise: groups = C_in = C_out = 6
        // input: [1, 4, 6], weight: [6, 3, 1] (C_in/groups = 1)
        let input = Array::zeros((1_i32, 4, 6), Dtype::Float32).unwrap();
        let weight = Array::zeros((6_i32, 3, 1), Dtype::Float32).unwrap();
        let out = conv1d(&input, &weight, 1, 0, 1, /* groups */ 6).expect("depthwise conv1d");
        // L_out = 4 - 3 + 1 = 2
        assert_eq!(out.shape().as_slice(), &[1, 2, 6]);
    }
}
