use crate::array::Array;
use crate::error::Result;
use crate::ops;
use crate::stream::Stream;
use crate::vector::VectorArray;

/// KV Cache for a single transformer layer.
pub struct KVCache {
    pub keys: Option<Array>,
    pub values: Option<Array>,
}

impl KVCache {
    pub fn new() -> Self {
        Self {
            keys: None,
            values: None,
        }
    }

    /// Current sequence length in the cache.
    pub fn seq_len(&self) -> usize {
        self.keys.as_ref().map_or(0, |k| k.shape()[2] as usize)
    }

    /// Update cache with new key/value tensors. Returns full (keys, values).
    pub fn update(
        &mut self,
        new_k: Array,
        new_v: Array,
        stream: &Stream,
    ) -> Result<(Array, Array)> {
        let (full_k, full_v) = match (&self.keys, &self.values) {
            (Some(ck), Some(cv)) => {
                let arr_k = VectorArray::from_arrays(&[ck, &new_k]);
                let arr_v = VectorArray::from_arrays(&[cv, &new_v]);
                (
                    ops::concatenate(&arr_k, 2, stream)?,
                    ops::concatenate(&arr_v, 2, stream)?,
                )
            }
            _ => (new_k, new_v),
        };
        self.keys = Some(full_k.clone());
        self.values = Some(full_v.clone());
        Ok((full_k, full_v))
    }

    /// Trim cache to max_len along the sequence dimension.
    pub fn trim(&mut self, max_len: usize, stream: &Stream) -> Result<()> {
        if let (Some(ref k), Some(ref v)) = (&self.keys, &self.values) {
            let seq = k.shape()[2];
            if seq > max_len as i32 {
                let shape = k.shape();
                self.keys = Some(ops::slice(
                    k,
                    &[0, 0, seq - max_len as i32, 0],
                    &[shape[0], shape[1], seq, shape[3]],
                    &[1, 1, 1, 1],
                    stream,
                )?);
                let vshape = v.shape();
                self.values = Some(ops::slice(
                    v,
                    &[0, 0, seq - max_len as i32, 0],
                    &[vshape[0], vshape[1], seq, vshape[3]],
                    &[1, 1, 1, 1],
                    stream,
                )?);
            }
        }
        Ok(())
    }
}

impl Default for KVCache {
    fn default() -> Self {
        Self::new()
    }
}
