use anyhow::anyhow;
use mlx::{Array, StreamOrDevice};

use crate::Result;

use super::config::Gemma4RopeParams;

#[derive(Clone)]
pub struct RopeOffsets {
    values: Vec<i32>,
    values_array: Array,
    non_uniform_array: Option<Array>,
}

impl RopeOffsets {
    pub fn from_values(values: Vec<i32>) -> Result<Self> {
        if values.is_empty() {
            return Err(anyhow!("Gemma4 RopeOffsets cannot be empty"));
        }
        let values_array: Array = (&values[..], &[values.len() as i32][..]).try_into()?;
        let non_uniform_array = if values.iter().all(|&v| v == values[0]) {
            None
        } else {
            Some(values_array.clone())
        };
        Ok(Self {
            values,
            values_array,
            non_uniform_array,
        })
    }

    pub fn scalar(&self) -> Option<i32> {
        if self.non_uniform_array.is_none() {
            Some(self.values[0])
        } else {
            None
        }
    }

    pub fn array(&self) -> Option<&Array> {
        self.non_uniform_array.as_ref()
    }

    pub(crate) fn values_array(&self) -> &Array {
        &self.values_array
    }

    pub fn values(&self) -> &[i32] {
        &self.values
    }
}

pub enum Gemma4Rope {
    Default {
        dims: i32,
        base: f32,
        traditional: bool,
    },
    Proportional {
        dims: i32,
        rotated_dims: i32,
        traditional: bool,
        freqs: Array,
    },
}

impl Gemma4Rope {
    pub fn new(dims: i32, traditional: bool, params: &Gemma4RopeParams) -> Result<Self> {
        match params.rope_type.as_str() {
            "default" => Ok(Self::Default {
                dims,
                base: params.rope_theta,
                traditional,
            }),
            "proportional" => {
                let rope_angles = (params.partial_rotary_factor * dims as f32 / 2.0) as i32;
                let rotated_dims = 2 * rope_angles;
                if rotated_dims <= 0 {
                    return Err(anyhow!(
                        "Gemma4 proportional RoPE rotated_dims must be > 0, got {rotated_dims}"
                    ));
                }
                let mut values = Vec::with_capacity((rotated_dims / 2) as usize);
                let mut i = 0;
                while i < rotated_dims {
                    let exponent = i as f32 / dims as f32;
                    values.push(params.factor * params.rope_theta.powf(exponent));
                    i += 2;
                }
                let freqs: Array = (&values[..], &[values.len() as i32][..]).try_into()?;
                Ok(Self::Proportional {
                    dims,
                    rotated_dims,
                    traditional,
                    freqs,
                })
            }
            other => Err(anyhow!("Gemma4: unsupported rope_type `{other}`")),
        }
    }

    pub fn apply_on(
        &self,
        x: &Array,
        offsets: &RopeOffsets,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        let target = target.into();
        match self {
            Self::Default {
                dims,
                base,
                traditional,
            } => apply_fast_rope(x, *dims, *traditional, Some(*base), None, offsets, target),
            Self::Proportional {
                dims,
                rotated_dims,
                traditional,
                freqs,
            } => apply_proportional_rope(
                x,
                *dims,
                *rotated_dims,
                *traditional,
                freqs,
                offsets,
                target,
            ),
        }
    }

    pub(crate) fn default_params(&self) -> Option<(i32, f32, bool)> {
        match self {
            Self::Default {
                dims,
                base,
                traditional,
            } => Some((*dims, *base, *traditional)),
            Self::Proportional { .. } => None,
        }
    }
}

fn apply_fast_rope(
    x: &Array,
    dims: i32,
    traditional: bool,
    base: Option<f32>,
    freqs: Option<&Array>,
    offsets: &RopeOffsets,
    target: StreamOrDevice,
) -> Result<Array> {
    if let Some(offset) = offsets.scalar() {
        Ok(mlx::fast::rope_on(
            x,
            dims,
            traditional,
            base,
            1.0,
            offset,
            freqs,
            target,
        )?)
    } else {
        Ok(mlx::fast::rope_with_array_offset_on(
            x,
            dims,
            traditional,
            base,
            1.0,
            offsets.array().expect("non-scalar offsets have array"),
            freqs,
            target,
        )?)
    }
}

fn apply_proportional_rope(
    x: &Array,
    dims: i32,
    rotated_dims: i32,
    traditional: bool,
    freqs: &Array,
    offsets: &RopeOffsets,
    target: StreamOrDevice,
) -> Result<Array> {
    if x.ndim() != 4 {
        return Err(anyhow!(
            "Gemma4 proportional RoPE expects rank-4 [B,H,S,D], got rank {}",
            x.ndim()
        ));
    }
    let last = x.shape_at(-1);
    if last < dims {
        return Err(anyhow!(
            "Gemma4 proportional RoPE last dim {last} < configured dims {dims}"
        ));
    }
    let half = dims / 2;
    let rot_half = rotated_dims / 2;

    let left_rot = slice_last_axis(x, 0, rot_half, target)?;
    let left_tail = slice_last_axis(x, rot_half, half, target)?;
    let right_rot = slice_last_axis(x, half, half + rot_half, target)?;
    let right_tail = slice_last_axis(x, half + rot_half, dims, target)?;

    let rotated = mlx::ops::shape::concatenate_on(&[&left_rot, &right_rot], 3, target)?;
    let rotated = apply_fast_rope(
        &rotated,
        rotated_dims,
        traditional,
        None,
        Some(freqs),
        offsets,
        target,
    )?;
    let parts = mlx::ops::shape::split_at_on(&rotated, &[rot_half][..], 3, target)?;
    let left = mlx::ops::shape::concatenate_on(&[&parts[0], &left_tail], 3, target)?;
    let right = mlx::ops::shape::concatenate_on(&[&parts[1], &right_tail], 3, target)?;
    let head = mlx::ops::shape::concatenate_on(&[&left, &right], 3, target)?;

    if last == dims {
        Ok(head)
    } else {
        let tail = slice_last_axis(x, dims, last, target)?;
        Ok(mlx::ops::shape::concatenate_on(&[&head, &tail], 3, target)?)
    }
}

fn slice_last_axis(x: &Array, start: i32, stop: i32, target: StreamOrDevice) -> Result<Array> {
    let shape = x.shape();
    let s = shape.as_slice();
    Ok(mlx::ops::indexing::slice_strided_on(
        x,
        &[0_i32, 0, 0, start][..],
        &[s[0], s[1], s[2], stop][..],
        &[1_i32, 1, 1, 1][..],
        target,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx::Dtype;

    #[test]
    fn proportional_rotated_dims_use_full_head_denominator() {
        let params = Gemma4RopeParams {
            partial_rotary_factor: 0.25,
            rope_theta: 1_000_000.0,
            rope_type: "proportional".to_owned(),
            factor: 1.0,
        };
        let rope = Gemma4Rope::new(512, false, &params).unwrap();
        match rope {
            Gemma4Rope::Proportional { rotated_dims, .. } => assert_eq!(rotated_dims, 128),
            _ => panic!("expected proportional"),
        }
    }

    #[test]
    fn non_uniform_offsets_build_array() {
        let offsets = RopeOffsets::from_values(vec![1, 3]).unwrap();
        assert!(offsets.scalar().is_none());
        assert!(offsets.array().is_some());
        assert_eq!(offsets.array().unwrap().dtype(), Dtype::Int32);
    }
}
