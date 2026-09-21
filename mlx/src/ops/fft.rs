//! Real Fourier transforms with an explicit transform length (including odd inverse lengths).
use crate::{Array, Error, Result, StreamOrDevice};
/// Scaling convention for forward and inverse Fourier transforms.
#[derive(Clone, Copy, Debug, Default)]
#[repr(u8)]
pub enum FftNorm {
    /// Forward unscaled; inverse divided by transform length.
    #[default]
    Backward = 0,
    /// Both directions divided by the square root of transform length.
    Ortho = 1,
    /// Forward divided by transform length; inverse unscaled.
    Forward = 2,
}
/// Real Fourier transform along `axis`, zero padded or truncated to `n`.
pub fn rfft(input: &Array, n: i32, axis: i32, norm: FftNorm) -> Result<Array> {
    rfft_on(input, n, axis, norm, ())
}
/// Stream-targeted `rfft`.
pub fn rfft_on(
    input: &Array,
    n: i32,
    axis: i32,
    norm: FftNorm,
    target: impl Into<StreamOrDevice>,
) -> Result<Array> {
    let (has, dev_only, dev_t, idx) = target.into().encode();
    // SAFETY: input is borrowed for the duration of the call.
    let inner = unsafe {
        mlx_sys::fft::ffi::ops_real_fft(
            input.as_inner(),
            n,
            axis,
            false,
            norm as u8,
            has,
            dev_only,
            dev_t,
            idx,
        )
    }
    .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}
/// Inverse real Fourier transform along `axis`, zero padded or truncated to `n`.
pub fn irfft(input: &Array, n: i32, axis: i32, norm: FftNorm) -> Result<Array> {
    irfft_on(input, n, axis, norm, ())
}
/// Stream-targeted `irfft`.
pub fn irfft_on(
    input: &Array,
    n: i32,
    axis: i32,
    norm: FftNorm,
    target: impl Into<StreamOrDevice>,
) -> Result<Array> {
    let (has, dev_only, dev_t, idx) = target.into().encode();
    // SAFETY: input is borrowed for the duration of the call.
    let inner = unsafe {
        mlx_sys::fft::ffi::ops_real_fft(
            input.as_inner(),
            n,
            axis,
            true,
            norm as u8,
            has,
            dev_only,
            dev_t,
            idx,
        )
    }
    .map_err(Error::from)?;
    Ok(Array::from_inner(inner))
}
