use crate::error::{Error, Result};

/// Load media from various URL formats.
/// Supports: base64 data URIs, HTTP(S) URLs, local file paths.
pub fn load_media(url: &str) -> Result<Vec<u8>> {
    if url.starts_with("data:") {
        // data:image/jpeg;base64,/9j/4AAQ...
        let comma_pos = url
            .find(',')
            .ok_or_else(|| Error::Mlx("invalid data URI: missing comma".into()))?;
        let encoded = &url[comma_pos + 1..];
        use base64::Engine;
        base64::engine::general_purpose::STANDARD
            .decode(encoded)
            .map_err(|e| Error::Mlx(format!("base64 decode failed: {}", e)))
    } else if url.starts_with("http://") || url.starts_with("https://") {
        // HTTP download via curl
        let output = std::process::Command::new("curl")
            .args(["-sL", "--max-time", "30", url])
            .output()
            .map_err(|e| Error::Mlx(format!("curl failed: {}", e)))?;
        if !output.status.success() {
            return Err(Error::Mlx(format!(
                "HTTP download failed: {}",
                String::from_utf8_lossy(&output.stderr)
            )));
        }
        Ok(output.stdout)
    } else {
        // Local file path (strip file:// prefix if present)
        let path = url.strip_prefix("file://").unwrap_or(url);
        std::fs::read(path).map_err(|e| Error::Mlx(format!("failed to read file {}: {}", path, e)))
    }
}
