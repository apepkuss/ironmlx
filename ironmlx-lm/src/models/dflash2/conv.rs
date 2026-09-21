use anyhow::anyhow;
use mlx::{Array, StreamOrDevice};

use crate::core::Loader;
use crate::nn::Linear;
use crate::Result;

use super::load_linear;

/// DFlash2 grouped dynamic causal convolution wrapped around attention/MLP.
pub(super) struct DFlash2GroupedConv {
    base_kernel: Array,
    kernel_projection: Linear,
    kernel_size: i32,
    group_size: i32,
    groups: i32,
}

impl DFlash2GroupedConv {
    pub(super) fn from_loader(
        loader: &Loader,
        prefix: &str,
        hidden_size: i32,
        kernel_size: i32,
        group_size: i32,
        draft_bits: Option<i32>,
    ) -> Result<Self> {
        if hidden_size <= 0 || kernel_size <= 0 || group_size <= 0 {
            return Err(anyhow!("DFlash2 grouped conv dimensions must be positive"));
        }
        if hidden_size % group_size != 0 {
            return Err(anyhow!(
                "DFlash2 grouped conv group_size {group_size} must divide hidden_size {hidden_size}"
            ));
        }
        Ok(Self {
            base_kernel: loader.tensor(&format!("{prefix}.base_kernel"))?.clone(),
            kernel_projection: load_linear(
                loader,
                &format!("{prefix}.kernel_projection"),
                draft_bits,
            )?,
            kernel_size,
            group_size,
            groups: hidden_size / group_size,
        })
    }

    #[cfg(test)]
    fn from_components(
        base_kernel: Array,
        kernel_projection: Linear,
        kernel_size: i32,
        group_size: i32,
        hidden_size: i32,
    ) -> Self {
        Self {
            base_kernel,
            kernel_projection,
            kernel_size,
            group_size,
            groups: hidden_size / group_size,
        }
    }

    pub(super) fn prepare_on(
        &self,
        hidden: &Array,
        target: StreamOrDevice,
    ) -> Result<(Array, Array)> {
        let shape = hidden.shape();
        let dims = shape.as_slice();
        if dims.len() != 3 || dims[2] != self.groups * self.group_size {
            return Err(anyhow!(
                "DFlash2 grouped conv expected [B,L,{}], got {dims:?}",
                self.groups * self.group_size
            ));
        }
        let dynamic = self
            .kernel_projection
            .forward_on(hidden, target)?
            .reshape_on(
                &[dims[0], dims[1], 2_i32, self.kernel_size, self.groups][..],
                target,
            )?;
        let input_dynamic = mlx::ops::indexing::slice_strided_on(
            &dynamic,
            &[0_i32, 0, 0, 0, 0][..],
            &[dims[0], dims[1], 1, self.kernel_size, self.groups][..],
            &[1_i32, 1, 1, 1, 1][..],
            target,
        )?
        .reshape_on((dims[0], dims[1], self.kernel_size, self.groups), target)?;
        let output_dynamic = mlx::ops::indexing::slice_strided_on(
            &dynamic,
            &[0_i32, 0, 1, 0, 0][..],
            &[dims[0], dims[1], 2, self.kernel_size, self.groups][..],
            &[1_i32, 1, 1, 1, 1][..],
            target,
        )?
        .reshape_on((dims[0], dims[1], self.kernel_size, self.groups), target)?;
        Ok((
            self.convolve_on(hidden, &input_dynamic, 0, target)?,
            output_dynamic,
        ))
    }

    pub(super) fn finish_on(
        &self,
        hidden: &Array,
        dynamic: &Array,
        target: StreamOrDevice,
    ) -> Result<Array> {
        self.convolve_on(hidden, dynamic, 1, target)
    }

    fn convolve_on(
        &self,
        hidden: &Array,
        dynamic: &Array,
        side: i32,
        target: StreamOrDevice,
    ) -> Result<Array> {
        let shape = hidden.shape();
        let dims = shape.as_slice();
        let (batch, length, hidden_size) = (dims[0], dims[1], dims[2]);
        let blocks = hidden.reshape_on((batch, length, self.groups, self.group_size), target)?;
        let mut output = Array::zeros_on(
            (batch, length, self.groups, self.group_size),
            hidden.dtype(),
            target,
        )?;
        for offset in 0..self.kernel_size {
            let values = if offset == 0 {
                blocks.clone()
            } else {
                let zeros = Array::zeros_on(
                    (batch, offset, self.groups, self.group_size),
                    hidden.dtype(),
                    target,
                )?;
                let prefix = mlx::ops::indexing::slice_strided_on(
                    &blocks,
                    &[0_i32, 0, 0, 0][..],
                    &[batch, length - offset, self.groups, self.group_size][..],
                    &[1_i32, 1, 1, 1][..],
                    target,
                )?;
                mlx::ops::shape::concatenate_on(&[&zeros, &prefix], 1, target)?
            };
            let base = mlx::ops::indexing::slice_strided_on(
                &self.base_kernel,
                &[side, offset, 0][..],
                &[side + 1, offset + 1, hidden_size][..],
                &[1_i32, 1, 1][..],
                target,
            )?
            .reshape_on((1_i32, 1, self.groups, self.group_size), target)?;
            let delta = mlx::ops::indexing::slice_strided_on(
                dynamic,
                &[0_i32, 0, offset, 0][..],
                &[batch, length, offset + 1, self.groups][..],
                &[1_i32, 1, 1, 1][..],
                target,
            )?
            .reshape_on((batch, length, self.groups, 1_i32), target)?;
            output = &output + &(&(&base + &delta) * &values);
        }
        output
            .reshape_on((batch, length, hidden_size), target)
            .map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use mlx::Dtype;
    use serial_test::serial;

    #[test]
    #[serial(mlx_metal)]
    fn grouped_dynamic_convolution_matches_reference_equations() {
        let base: Array = (
            &[1.0_f32, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5][..],
            &[2_i32, 2, 2][..],
        )
            .try_into()
            .expect("base");
        let projection = Linear::new_fp(
            Array::zeros((4_i32, 2), Dtype::Float32).expect("projection"),
            None,
        );
        let conv = DFlash2GroupedConv::from_components(base, projection, 2, 1, 2);
        let hidden: Array = (&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0][..], &[1_i32, 3, 2][..])
            .try_into()
            .expect("hidden");
        let dynamic = Array::zeros((1_i32, 3, 2, 2), Dtype::Float32).expect("dynamic");
        let got = conv
            .convolve_on(&hidden, &dynamic, 1, StreamOrDevice::default())
            .expect("convolve")
            .to_vec::<f32>()
            .expect("materialize");
        let expected = [0.5_f32, 1.0, 2.0, 3.0, 4.0, 5.0];
        for (got, expected) in got.iter().zip(expected) {
            assert_abs_diff_eq!(*got, expected, epsilon = 1e-6);
        }
    }
}
