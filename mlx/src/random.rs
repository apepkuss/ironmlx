use mlx_sys as sys;

use crate::array::Array;
use crate::dtype::Dtype;
use crate::error::{check, Result};
use crate::stream::Stream;

/// Generate a PRNG key from a seed.
pub fn key(seed: u64) -> Result<Array> {
    let mut res = Array::new_empty();
    check(unsafe { sys::mlx_random_key(res.as_raw_mut(), seed) })?;
    Ok(res)
}

/// Seed the global random number generator.
pub fn seed(seed: u64) -> Result<()> {
    check(unsafe { sys::mlx_random_seed(seed) })
}

/// Split a PRNG key into two new keys.
pub fn split(key: &Array, stream: &Stream) -> Result<(Array, Array)> {
    let mut res_0 = Array::new_empty();
    let mut res_1 = Array::new_empty();
    check(unsafe {
        sys::mlx_random_split(
            res_0.as_raw_mut(),
            res_1.as_raw_mut(),
            key.as_raw(),
            stream.as_raw(),
        )
    })?;
    Ok((res_0, res_1))
}

/// Sample from a categorical distribution defined by unnormalised log-probabilities.
pub fn categorical(
    logits: &Array,
    axis: i32,
    key: Option<&Array>,
    stream: &Stream,
) -> Result<Array> {
    let mut res = Array::new_empty();
    let raw_key = key.map_or(unsafe { std::mem::zeroed() }, |k| k.as_raw());
    check(unsafe {
        sys::mlx_random_categorical(
            res.as_raw_mut(),
            logits.as_raw(),
            axis,
            raw_key,
            stream.as_raw(),
        )
    })?;
    Ok(res)
}

/// Sample from a categorical distribution with a specified number of samples.
pub fn categorical_num_samples(
    logits: &Array,
    axis: i32,
    num_samples: i32,
    key: Option<&Array>,
    stream: &Stream,
) -> Result<Array> {
    let mut res = Array::new_empty();
    let raw_key = key.map_or(unsafe { std::mem::zeroed() }, |k| k.as_raw());
    check(unsafe {
        sys::mlx_random_categorical_num_samples(
            res.as_raw_mut(),
            logits.as_raw(),
            axis,
            num_samples,
            raw_key,
            stream.as_raw(),
        )
    })?;
    Ok(res)
}

/// Generate uniformly distributed random numbers.
pub fn uniform(
    low: &Array,
    high: &Array,
    shape: &[i32],
    dtype: Dtype,
    key: Option<&Array>,
    stream: &Stream,
) -> Result<Array> {
    let mut res = Array::new_empty();
    let raw_key = key.map_or(unsafe { std::mem::zeroed() }, |k| k.as_raw());
    check(unsafe {
        sys::mlx_random_uniform(
            res.as_raw_mut(),
            low.as_raw(),
            high.as_raw(),
            shape.as_ptr(),
            shape.len(),
            dtype.to_raw(),
            raw_key,
            stream.as_raw(),
        )
    })?;
    Ok(res)
}

/// Generate normally distributed random numbers.
pub fn normal(
    shape: &[i32],
    dtype: Dtype,
    loc: f32,
    scale: f32,
    key: Option<&Array>,
    stream: &Stream,
) -> Result<Array> {
    let mut res = Array::new_empty();
    let raw_key = key.map_or(unsafe { std::mem::zeroed() }, |k| k.as_raw());
    check(unsafe {
        sys::mlx_random_normal(
            res.as_raw_mut(),
            shape.as_ptr(),
            shape.len(),
            dtype.to_raw(),
            loc,
            scale,
            raw_key,
            stream.as_raw(),
        )
    })?;
    Ok(res)
}
