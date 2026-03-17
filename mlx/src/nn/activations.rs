use crate::array::Array;
use crate::error::Result;
use crate::ops;
use crate::stream::Stream;

/// SiLU (Swish): x * sigmoid(x)
pub fn silu(x: &Array, stream: &Stream) -> Result<Array> {
    let s = ops::sigmoid(x, stream)?;
    ops::multiply(x, &s, stream)
}

/// GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
pub fn gelu(x: &Array, stream: &Stream) -> Result<Array> {
    let sqrt2 = Array::from_float(std::f32::consts::SQRT_2);
    let half = Array::from_float(0.5);
    let one = Array::from_float(1.0);

    let x_scaled = ops::divide(x, &sqrt2, stream)?;
    let erf_val = ops::erf(&x_scaled, stream)?;
    let inner = ops::add(&one, &erf_val, stream)?;
    let half_x = ops::multiply(&half, x, stream)?;
    ops::multiply(&half_x, &inner, stream)
}

/// GELU with tanh approximation (pytorch_tanh variant):
/// 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
pub fn gelu_tanh(x: &Array, stream: &Stream) -> Result<Array> {
    let coeff = Array::from_float(0.044715);
    let sqrt_2_pi = Array::from_float((2.0f32 / std::f32::consts::PI).sqrt());
    let half = Array::from_float(0.5);
    let one = Array::from_float(1.0);

    let x3 = ops::power(x, &Array::from_int(3), stream)?;
    let inner = ops::multiply(&coeff, &x3, stream)?;
    let inner = ops::add(x, &inner, stream)?;
    let inner = ops::multiply(&sqrt_2_pi, &inner, stream)?;
    let tanh_val = ops::tanh(&inner, stream)?;
    let gate = ops::add(&one, &tanh_val, stream)?;
    let half_x = ops::multiply(&half, x, stream)?;
    ops::multiply(&half_x, &gate, stream)
}
