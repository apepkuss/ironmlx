use crate::{AudioError, Result};
use ironmlx_core::weights::{WeightMap, WeightSource};
use mlx::{ops, Array};

pub(super) struct Weights(pub WeightMap);
impl Weights {
    pub fn get(&self, key: &str) -> Result<&Array> {
        self.0
            .tensor_opt(key)
            .ok_or_else(|| AudioError::ResourceMissing {
                component: key.into(),
            })
    }
    pub fn linear(&self, x: &Array, name: &str) -> Result<Array> {
        let weight = self.get(&format!("{name}.weight"))?;
        let y = ops::matmul(x, &weight.transpose()?)?;
        match self.0.tensor_opt(&format!("{name}.bias")) {
            Some(bias) => Ok(ops::add(&y, bias)?),
            None => Ok(y),
        }
    }
    /// MLX nn.Linear fuses the bias into matmul, preserving fp16 rounding.
    pub fn linear_fused(&self, x: &Array, name: &str) -> Result<Array> {
        let weight = self.get(&format!("{name}.weight"))?.transpose()?;
        Ok(match self.0.tensor_opt(&format!("{name}.bias")) {
            Some(bias) => ops::addmm(bias, x, &weight, 1., 1.)?,
            None => ops::matmul(x, &weight)?,
        })
    }
    pub fn norm(&self, x: &Array, name: &str, eps: f32) -> Result<Array> {
        Ok(mlx::fast::layer_norm(
            x,
            Some(self.get(&format!("{name}.weight"))?),
            Some(self.get(&format!("{name}.bias"))?),
            eps,
        )?)
    }
    // PyTorch [out,in,kernel] -> MLX [out,kernel,in]; activations remain NLC.
    pub fn conv1(
        &self,
        x: &Array,
        name: &str,
        stride: i32,
        padding: i32,
        groups: i32,
    ) -> Result<Array> {
        self.conv1_dilated(x, name, stride, padding, 1, groups)
    }
    pub fn conv1_dilated(
        &self,
        x: &Array,
        name: &str,
        stride: i32,
        padding: i32,
        dilation: i32,
        groups: i32,
    ) -> Result<Array> {
        let w = self
            .get(&format!("{name}.weight"))?
            .transpose_axes([0, 2, 1])?;
        let y = ops::conv1d(x, &w, stride, padding, dilation, groups)?;
        match self.0.tensor_opt(&format!("{name}.bias")) {
            Some(b) => Ok(ops::add(&y, b)?),
            None => Ok(y),
        }
    }
    pub fn conv2(
        &self,
        x: &Array,
        name: &str,
        stride: (i32, i32),
        padding: (i32, i32),
    ) -> Result<Array> {
        let weight = self
            .get(&format!("{name}.weight"))?
            .transpose_axes([0, 2, 3, 1])?;
        Ok(ops::conv2d(x, &weight, stride, padding, (1, 1), 1)?)
    }
    pub fn batch_norm(&self, x: &Array, name: &str) -> Result<Array> {
        let mean = self.get(&format!("{name}.running_mean"))?;
        let var = self.get(&format!("{name}.running_var"))?;
        let mut y = ops::divide(
            &ops::subtract(x, mean)?,
            &ops::sqrt(&ops::add(var, &scalar(1e-5)?)?)?,
        )?;
        if let Some(weight) = self.0.tensor_opt(&format!("{name}.weight")) {
            y = ops::add(
                &ops::multiply(&y, weight)?,
                self.get(&format!("{name}.bias"))?,
            )?;
        }
        Ok(y)
    }
}
pub(super) fn scalar(value: f32) -> Result<Array> {
    Ok(Array::try_from((&[value][..], []))?)
}
pub(super) fn scale(x: &Array, factor: f32) -> Result<Array> {
    Ok(ops::multiply(x, &scalar(factor)?)?)
}
pub(super) fn swish(x: &Array) -> Result<Array> {
    Ok(ops::multiply(x, &ops::sigmoid(x)?)?)
}
pub(super) fn relu(x: &Array) -> Result<Array> {
    Ok(ops::maximum(x, &scalar(0.)?)?)
}
pub(super) fn finite(x: &Array, name: &str) -> Result<()> {
    if !ops::all(&ops::isfinite(x)?, ops::All, false)?.item::<bool>()? {
        return Err(AudioError::InferenceFailed {
            reason: format!("non-finite {name}"),
        });
    }
    Ok(())
}
