use super::config::StackConfig;
use crate::SessionControl;
use anyhow::{ensure, Result};
use ironmlx_core::{
    nn::{activations::gelu_tanh, Linear, RmsNorm},
    weights::{WeightMap, WeightSource},
};
use mlx::{ops, Array, Dtype};

pub(super) type Cache = Vec<Option<(Array, Array)>>;
pub(super) fn scalar(v: f32, dtype: Dtype) -> Result<Array> {
    Ok(Array::try_from((&[v][..], []))?.astype(dtype)?)
}
pub(super) fn scale(x: &Array, v: f32) -> Result<Array> {
    Ok(ops::multiply(x, &scalar(v, x.dtype())?)?)
}
fn norm(w: &WeightMap, p: &str, dim: i32, eps: f32, offset: bool) -> Result<RmsNorm> {
    let weight = w.tensor(&format!("{p}.weight"))?;
    ensure!(
        weight.shape().as_slice() == [dim],
        "{p}: invalid norm shape"
    );
    Ok(RmsNorm::new(
        if offset {
            ops::add(weight, &scalar(1., weight.dtype())?)?
        } else {
            weight.clone()
        },
        eps,
    ))
}
fn linear(w: &WeightMap, p: &str, input: i32, output: i32) -> Result<Linear> {
    let layer = Linear::from_loader(w, p)?;
    ensure!(
        layer.in_features() == input as usize && layer.out_features() == output as usize,
        "{p}: invalid projection shape"
    );
    Ok(layer)
}
struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    qn: Option<RmsNorm>,
    kn: Option<RmsNorm>,
    heads: i32,
    kv: i32,
    dim: i32,
    theta: f32,
    rope_scale: f32,
    freqs: Option<Array>,
    scale: f32,
}
impl Attention {
    fn load(w: &WeightMap, p: &str, c: &StackConfig, text: bool, layer: usize) -> Result<Self> {
        let d = c.head_dim;
        let (theta, rope_scale, freqs) = if text {
            let r = &c.rope_parameters[&c.layer_types[layer]];
            (
                r["rope_theta"].as_f64().unwrap_or(10000.) as f32,
                if r["rope_type"] == "linear" {
                    1. / r["factor"].as_f64().unwrap_or(1.) as f32
                } else {
                    1.
                },
                None,
            )
        } else if let Some(r) = &c.rope_scaling {
            ensure!(
                r["rope_type"] == "llama3",
                "unsupported Breeze rotary scaling"
            );
            let factor = r["factor"].as_f64().unwrap_or(1.) as f32;
            let low = r["low_freq_factor"].as_f64().unwrap_or(1.) as f32;
            let high = r["high_freq_factor"].as_f64().unwrap_or(4.) as f32;
            let old = r["original_max_position_embeddings"]
                .as_f64()
                .unwrap_or(8192.) as f32;
            ensure!(
                factor > 0. && high > low && low > 0. && old > 0.,
                "invalid Breeze llama3 frequencies"
            );
            let frequencies: Vec<f32> = (0..d / 2)
                .map(|i| {
                    let f = c.rope_theta.powf(2. * i as f32 / d as f32);
                    let wave = 2. * std::f32::consts::PI * f;
                    if wave > old / low {
                        f * factor
                    } else if wave > old / high {
                        let smooth = (old / wave - low) / (high - low);
                        f / ((1. - smooth) / factor + smooth)
                    } else {
                        f
                    }
                })
                .collect();
            (
                c.rope_theta,
                1.,
                Some(Array::try_from((frequencies.as_slice(), [d / 2]))?),
            )
        } else {
            (c.rope_theta, 1., None)
        };
        let qnorm = text || w.contains(&format!("{p}.q_norm.weight"));
        Ok(Self {
            q: linear(
                w,
                &format!("{p}.q_proj"),
                c.hidden_size,
                c.num_attention_heads * d,
            )?,
            k: linear(
                w,
                &format!("{p}.k_proj"),
                c.hidden_size,
                c.num_key_value_heads * d,
            )?,
            v: linear(
                w,
                &format!("{p}.v_proj"),
                c.hidden_size,
                c.num_key_value_heads * d,
            )?,
            o: linear(
                w,
                &format!("{p}.o_proj"),
                c.num_attention_heads * d,
                c.hidden_size,
            )?,
            qn: if qnorm {
                Some(norm(w, &format!("{p}.q_norm"), d, c.rms_norm_eps, text)?)
            } else {
                None
            },
            kn: if qnorm {
                Some(norm(w, &format!("{p}.k_norm"), d, c.rms_norm_eps, text)?)
            } else {
                None
            },
            heads: c.num_attention_heads,
            kv: c.num_key_value_heads,
            dim: d,
            theta,
            rope_scale,
            freqs,
            scale: (d as f32).sqrt().recip(),
        })
    }
    fn forward(
        &self,
        x: &Array,
        cache: &mut Option<(Array, Array)>,
        mask: Option<&Array>,
        causal: bool,
    ) -> Result<Array> {
        let shape = x.shape();
        let s = shape.as_slice()[1];
        let reshape = |v: Array, h| -> Result<Array> { Ok(v.reshape([1, s, h, self.dim])?) };
        let mut q = reshape(self.q.forward(x)?, self.heads)?;
        let mut k = reshape(self.k.forward(x)?, self.kv)?;
        if let Some(n) = &self.qn {
            q = n.forward(&q)?;
            k = self.kn.as_ref().unwrap().forward(&k)?;
        }
        q = q.transpose_axes([0, 2, 1, 3])?;
        k = k.transpose_axes([0, 2, 1, 3])?;
        let mut v = reshape(self.v.forward(x)?, self.kv)?.transpose_axes([0, 2, 1, 3])?;
        let offset = cache.as_ref().map_or(0, |(k, _)| k.shape().as_slice()[2]);
        let rope = |x: &Array| {
            mlx::fast::rope(
                x,
                self.dim,
                false,
                if self.freqs.is_some() {
                    None
                } else {
                    Some(self.theta)
                },
                self.rope_scale,
                offset,
                self.freqs.as_ref(),
            )
        };
        q = rope(&q)?;
        k = rope(&k)?;
        if let Some((a, b)) = cache {
            k = ops::concatenate(&[a, &k], 2)?;
            v = ops::concatenate(&[b, &v], 2)?;
        }
        let out = mlx::fast::scaled_dot_product_attention(
            &q,
            &k,
            &v,
            self.scale,
            if causal && s > 1 { "causal" } else { "" },
            mask,
            None,
        )?;
        if causal {
            *cache = Some((k, v));
        }
        self.o.forward(
            &out.transpose_axes([0, 2, 1, 3])?
                .reshape([1, s, self.heads * self.dim])?,
        )
    }
}
struct Layer {
    attn: Attention,
    n1: RmsNorm,
    n2: RmsNorm,
    post_attn: Option<RmsNorm>,
    post_mlp: Option<RmsNorm>,
    gate: Linear,
    up: Linear,
    down: Linear,
    text: bool,
    window: Option<i32>,
}
impl Layer {
    fn load(w: &WeightMap, p: &str, c: &StackConfig, text: bool, i: usize) -> Result<Self> {
        let h = c.hidden_size;
        let n = |name: &str| norm(w, &format!("{p}.{name}"), h, c.rms_norm_eps, text);
        Ok(Self {
            attn: Attention::load(w, &format!("{p}.self_attn"), c, text, i)?,
            n1: n(if text {
                "pre_self_attn_layernorm"
            } else {
                "input_layernorm"
            })?,
            n2: n(if text {
                "pre_feedforward_layernorm"
            } else {
                "post_attention_layernorm"
            })?,
            post_attn: if text {
                Some(n("post_self_attn_layernorm")?)
            } else {
                None
            },
            post_mlp: if text {
                Some(n("post_feedforward_layernorm")?)
            } else {
                None
            },
            gate: linear(w, &format!("{p}.mlp.gate_proj"), h, c.intermediate_size)?,
            up: linear(w, &format!("{p}.mlp.up_proj"), h, c.intermediate_size)?,
            down: linear(w, &format!("{p}.mlp.down_proj"), c.intermediate_size, h)?,
            text,
            window: if text && c.layer_types[i] == "sliding_attention" {
                c.sliding_window
            } else {
                None
            },
        })
    }
    fn forward(&self, x: &Array, cache: &mut Option<(Array, Array)>) -> Result<Array> {
        let mask = if let Some(w) = self.window {
            Some(local_mask(x.shape().as_slice()[1], w)?)
        } else {
            None
        };
        let mut a = self
            .attn
            .forward(&self.n1.forward(x)?, cache, mask.as_ref(), !self.text)?;
        if let Some(n) = &self.post_attn {
            a = n.forward(&a)?;
        }
        let x = ops::add(x, &a)?;
        let z = self.n2.forward(&x)?;
        let gate = self.gate.forward(&z)?;
        let gate = if self.text {
            gelu_tanh(&gate, ().into())?
        } else {
            ops::multiply(&gate, &gate.sigmoid()?)?
        };
        let mut a = self
            .down
            .forward(&ops::multiply(&gate, &self.up.forward(&z)?)?)?;
        if let Some(n) = &self.post_mlp {
            a = n.forward(&a)?;
        }
        Ok(ops::add(&x, &a)?)
    }
}
pub(super) fn local_mask(n: i32, w: i32) -> Result<Array> {
    let p = ops::arange(0., n as f64, 1., Dtype::Int32)?;
    let delta = ops::subtract(&p.reshape([n, 1])?, &p.reshape([1, n])?)?;
    let left = Array::try_from((&[(w + 1) / 2][..], []))?;
    let right = Array::try_from((&[-w / 2][..], []))?;
    Ok(ops::minimum(
        &ops::less(&delta, &left)?,
        &ops::greater_equal(&delta, &right)?,
    )?)
}
pub(super) struct Stack {
    layers: Vec<Layer>,
    norm: RmsNorm,
}
impl Stack {
    pub fn load(w: &WeightMap, p: &str, c: &StackConfig, text: bool) -> Result<Self> {
        Ok(Self {
            layers: (0..c.num_hidden_layers)
                .map(|i| Layer::load(w, &format!("{p}.layers.{i}"), c, text, i))
                .collect::<Result<_>>()?,
            norm: norm(w, &format!("{p}.norm"), c.hidden_size, c.rms_norm_eps, text)?,
        })
    }
    pub fn cache(&self) -> Cache {
        (0..self.layers.len()).map(|_| None).collect()
    }
    pub fn forward(
        &self,
        mut x: Array,
        cache: &mut Cache,
        control: &dyn SessionControl,
    ) -> Result<Array> {
        for (layer, cache) in self.layers.iter().zip(cache) {
            control.check()?;
            x = layer.forward(&x, cache)?;
            x.eval()?;
        }
        self.norm.forward(&x)
    }
}
