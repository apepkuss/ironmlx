use crate::error::Result;

/// Wrapper around HuggingFace tokenizers.
pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
}

impl Tokenizer {
    /// Load from a tokenizer.json file.
    pub fn from_file(path: &str) -> Result<Self> {
        let inner = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| crate::error::Error::Mlx(format!("failed to load tokenizer: {}", e)))?;
        Ok(Self { inner })
    }

    /// Encode text to token IDs.
    pub fn encode(&self, text: &str) -> Result<Vec<i32>> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| crate::error::Error::Mlx(format!("encode failed: {}", e)))?;
        Ok(encoding.get_ids().iter().map(|&id| id as i32).collect())
    }

    /// Decode token IDs to text.
    pub fn decode(&self, ids: &[i32]) -> Result<String> {
        let u32_ids: Vec<u32> = ids.iter().map(|&id| id as u32).collect();
        self.inner
            .decode(&u32_ids, true)
            .map_err(|e| crate::error::Error::Mlx(format!("decode failed: {}", e)))
    }

    /// Decode a single token ID.
    pub fn decode_single(&self, id: i32) -> Result<String> {
        self.decode(&[id])
    }

    /// Get the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }
}
