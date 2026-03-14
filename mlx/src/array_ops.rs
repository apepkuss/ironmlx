use std::ops::{Add, Div, Mul, Sub};

use crate::array::Array;
use crate::error::Result;

fn s() -> crate::stream::Stream {
    crate::stream::Stream::new(&crate::device::Device::gpu())
}

impl Add<&Array> for &Array {
    type Output = Result<Array>;
    fn add(self, rhs: &Array) -> Result<Array> {
        crate::ops::add(self, rhs, &s())
    }
}

impl Sub<&Array> for &Array {
    type Output = Result<Array>;
    fn sub(self, rhs: &Array) -> Result<Array> {
        crate::ops::subtract(self, rhs, &s())
    }
}

impl Mul<&Array> for &Array {
    type Output = Result<Array>;
    fn mul(self, rhs: &Array) -> Result<Array> {
        crate::ops::multiply(self, rhs, &s())
    }
}

impl Div<&Array> for &Array {
    type Output = Result<Array>;
    fn div(self, rhs: &Array) -> Result<Array> {
        crate::ops::divide(self, rhs, &s())
    }
}
