use crate::array::Array;
use crate::error::{Error, Result};

/// Extract frames from a video using ffmpeg CLI.
/// Returns raw PNG bytes for each extracted frame.
pub fn extract_frames(video_path: &str, max_frames: usize) -> Result<Vec<Vec<u8>>> {
    let tmp_dir = tempfile::TempDir::new()
        .map_err(|e| Error::Mlx(format!("failed to create temp dir: {}", e)))?;

    let pattern = tmp_dir.path().join("frame_%04d.png");
    let pattern_str = pattern
        .to_str()
        .ok_or_else(|| Error::Mlx("invalid temp path".into()))?;

    let max_frames_str = max_frames.to_string();

    // Use ffmpeg to extract frames
    let output = std::process::Command::new("ffmpeg")
        .args([
            "-i",
            video_path,
            "-vf",
            "fps=1", // 1 frame per second
            "-frames:v",
            &max_frames_str,
            "-y", // overwrite
            pattern_str,
        ])
        .stderr(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .output()
        .map_err(|e| {
            Error::Mlx(format!(
                "ffmpeg not found or failed: {}. Install with: brew install ffmpeg",
                e
            ))
        })?;

    if !output.status.success() {
        return Err(Error::Mlx(format!(
            "ffmpeg failed: {}",
            String::from_utf8_lossy(&output.stderr)
        )));
    }

    // Read extracted frames
    let mut frames = Vec::new();
    for i in 1..=max_frames {
        let frame_path = tmp_dir.path().join(format!("frame_{:04}.png", i));
        if frame_path.exists() {
            let data = std::fs::read(&frame_path)
                .map_err(|e| Error::Mlx(format!("failed to read frame: {}", e)))?;
            frames.push(data);
        } else {
            break; // no more frames
        }
    }

    if frames.is_empty() {
        return Err(Error::Mlx("ffmpeg extracted 0 frames".into()));
    }

    Ok(frames)
}

/// Process video frames: extract + process each frame + stack.
/// Returns pixel_values [T, C, H, W] and frame count.
pub fn process_video(
    video_path: &str,
    max_frames: usize,
    patch_size: usize,
    spatial_merge_size: usize,
) -> Result<(Array, usize)> {
    use crate::device::Device;
    use crate::ops;
    use crate::stream::Stream;
    use crate::vector::VectorArray;

    let frame_bytes = extract_frames(video_path, max_frames)?;
    let num_frames = frame_bytes.len();

    // Process each frame
    let stream = Stream::new(&Device::gpu());
    let mut frame_arrays = Vec::new();
    for data in &frame_bytes {
        let (arr, _h, _w) =
            super::image_proc::process_image_bytes(data, patch_size, spatial_merge_size)?;
        frame_arrays.push(arr);
    }

    // Stack along batch dim: each is [1, C, H, W] -> concatenate -> [T, C, H, W]
    let refs: Vec<&Array> = frame_arrays.iter().collect();
    let va = VectorArray::from_arrays(&refs);
    let stacked = ops::concatenate(&va, 0, &stream)?;

    Ok((stacked, num_frames))
}
