//! Bounded byte-based decoding and shared PCM/WAV output encoding.
use crate::{error::invalid, AudioError, AudioIo, DecodeLimits, PcmBuffer, PcmFormat, Result};
use base64::{engine::general_purpose::STANDARD, Engine};
use std::io::{Cursor, Write};
use symphonia::core::{
    audio::SampleBuffer, codecs::DecoderOptions, errors::Error as DecodeError,
    formats::FormatOptions, io::MediaSourceStream, meta::MetadataOptions, probe::Hint,
};

impl PcmFormat {
    pub fn validate(self) -> Result<()> {
        if self.sample_rate == 0 || self.channels == 0 {
            return Err(invalid(
                "pcm.format",
                "sample rate and channels must be positive",
            ));
        }
        Ok(())
    }
}
impl PcmBuffer {
    pub fn validate(&self) -> Result<()> {
        self.format.validate()?;
        if self.samples.is_empty()
            || !self
                .samples
                .len()
                .is_multiple_of(usize::from(self.format.channels))
        {
            return Err(invalid("pcm", "expected nonempty whole interleaved frames"));
        }
        if self.samples.iter().any(|x| !x.is_finite()) {
            return Err(invalid("pcm", "non-finite sample"));
        }
        Ok(())
    }
    pub fn frames(&self) -> Result<usize> {
        self.validate()?;
        Ok(self.samples.len() / usize::from(self.format.channels))
    }
}
impl Default for DecodeLimits {
    fn default() -> Self {
        Self {
            max_encoded_bytes: 16 * 1024 * 1024,
            max_pcm_bytes: 48 * 1024 * 1024,
            max_duration_seconds: 60,
            min_sample_rate: 8000,
            max_sample_rate: 96000,
            max_channels: 2,
        }
    }
}
impl DecodeLimits {
    fn sample_limit(&self, format: PcmFormat) -> Result<usize> {
        format.validate()?;
        if self.min_sample_rate == 0
            || self.min_sample_rate > self.max_sample_rate
            || self.max_channels == 0
        {
            return Err(invalid("decode_limits", "invalid format bounds"));
        }
        if !(self.min_sample_rate..=self.max_sample_rate).contains(&format.sample_rate)
            || format.channels > self.max_channels
        {
            return Err(invalid(
                "ref_audio",
                "sample rate or channel count is outside the configured bounds",
            ));
        }
        let duration = u64::from(format.sample_rate)
            .saturating_mul(u64::from(format.channels))
            .saturating_mul(u64::from(self.max_duration_seconds));
        Ok((self.max_pcm_bytes / 4).min(usize::try_from(duration).unwrap_or(usize::MAX)))
    }
}

/// Stateless decoder; callers own and budget input/output storage.
#[derive(Default)]
pub struct NativeAudioIo;
impl NativeAudioIo {
    /// Strict RFC 4648 standard Base64, with canonical padding and no whitespace.
    pub fn decode_base64(&self, text: &str, limits: &DecodeLimits) -> Result<PcmBuffer> {
        let encoded_limit = (limits.max_encoded_bytes.saturating_add(2) / 3).saturating_mul(4);
        if text.len() > encoded_limit {
            return Err(AudioError::CapacityExceeded {
                resource: "encoded audio",
            });
        }
        let bytes = STANDARD
            .decode(text)
            .map_err(|_| invalid("ref_audio", "expected canonical standard Base64"))?;
        self.decode(&bytes, limits)
    }
}
impl AudioIo for NativeAudioIo {
    fn decode(&self, bytes: &[u8], limits: &DecodeLimits) -> Result<PcmBuffer> {
        if bytes.len() > limits.max_encoded_bytes {
            return Err(AudioError::CapacityExceeded {
                resource: "encoded audio",
            });
        }
        if bytes.starts_with(b"RIFF") {
            return decode_wav(bytes, limits);
        }
        // Probe only formats in the public contract; no filename or extension hint.
        let flac = bytes.starts_with(b"fLaC");
        let mp3 = bytes.starts_with(b"ID3")
            || (bytes.len() > 1 && bytes[0] == 0xff && bytes[1] & 0xe0 == 0xe0);
        if !flac && !mp3 {
            return Err(AudioError::UnsupportedFormat);
        }
        if mp3 {
            validate_mp3_frames(bytes)?;
        }
        let source =
            MediaSourceStream::new(Box::new(Cursor::new(bytes.to_vec())), Default::default());
        let mut reader = symphonia::default::get_probe()
            .format(
                &Hint::new(),
                source,
                &FormatOptions {
                    enable_gapless: true,
                    ..Default::default()
                },
                &MetadataOptions::default(),
            )
            .map_err(|e| invalid("ref_audio", e.to_string()))?
            .format;
        let track = reader
            .default_track()
            .ok_or(AudioError::UnsupportedFormat)?;
        let track_id = track.id;
        let expected_frames = track.codec_params.n_frames;
        let mut decoder = symphonia::default::get_codecs()
            .make(&track.codec_params, &DecoderOptions { verify: true })
            .map_err(|e| invalid("ref_audio", e.to_string()))?;
        let mut pcm: Option<PcmBuffer> = None;
        loop {
            let packet = match reader.next_packet() {
                Ok(packet) => packet,
                Err(DecodeError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                    break
                }
                Err(e) => return Err(invalid("ref_audio", e.to_string())),
            };
            if packet.track_id() != track_id {
                return Err(invalid("ref_audio", "multiple audio tracks"));
            }
            let decoded = decoder
                .decode(&packet)
                .map_err(|e| invalid("ref_audio", e.to_string()))?;
            let spec = *decoded.spec();
            let format = PcmFormat {
                sample_rate: spec.rate,
                channels: spec.channels.count() as u16,
            };
            let max_samples = limits.sample_limit(format)?;
            let output = pcm.get_or_insert_with(|| PcmBuffer {
                format,
                samples: Vec::new(),
            });
            if output.format != format {
                return Err(invalid("ref_audio", "format changed within file"));
            }
            let count = decoded
                .frames()
                .checked_mul(usize::from(format.channels))
                .ok_or(AudioError::CapacityExceeded {
                    resource: "decoded audio",
                })?;
            if count > max_samples.saturating_sub(output.samples.len()) {
                return Err(AudioError::CapacityExceeded {
                    resource: "decoded audio",
                });
            }
            let mut buffer = SampleBuffer::<f32>::new(decoded.frames() as u64, spec);
            buffer.copy_interleaved_ref(decoded);
            if buffer.samples().iter().any(|x| !x.is_finite()) {
                return Err(invalid("ref_audio", "non-finite sample"));
            }
            output.samples.extend_from_slice(buffer.samples());
        }
        if decoder.finalize().verify_ok == Some(false) {
            return Err(invalid("ref_audio", "checksum mismatch"));
        }
        let pcm = pcm.ok_or_else(|| invalid("ref_audio", "no decoded samples"))?;
        pcm.validate()?;
        if flac
            && expected_frames.is_some_and(|n| {
                n != 0 && n != pcm.samples.len() as u64 / u64::from(pcm.format.channels)
            })
        {
            return Err(invalid("ref_audio", "truncated FLAC"));
        }
        Ok(pcm)
    }
    fn encode_wav(&self, pcm: &PcmBuffer) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        write_wav(pcm, &mut bytes)?;
        Ok(bytes)
    }
}
fn u16le(bytes: &[u8]) -> u16 {
    u16::from_le_bytes([bytes[0], bytes[1]])
}
fn u32le(bytes: &[u8]) -> u32 {
    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}
fn decode_wav(bytes: &[u8], limits: &DecodeLimits) -> Result<PcmBuffer> {
    if bytes.len() < 12 || &bytes[8..12] != b"WAVE" {
        return Err(AudioError::UnsupportedFormat);
    }
    let end = 8usize
        .checked_add(u32le(&bytes[4..8]) as usize)
        .ok_or_else(|| invalid("ref_audio", "invalid RIFF size"))?;
    if end != bytes.len() {
        return Err(invalid("ref_audio", "RIFF size mismatch or trailing bytes"));
    }
    let mut position = 12;
    let mut fmt = None;
    let mut data = None;
    while position < end {
        if end - position < 8 {
            return Err(invalid("ref_audio", "truncated WAV chunk"));
        }
        let kind = &bytes[position..position + 4];
        let size = u32le(&bytes[position + 4..position + 8]) as usize;
        position += 8;
        if size > end - position {
            return Err(invalid("ref_audio", "truncated WAV payload"));
        }
        let chunk = &bytes[position..position + size];
        if kind == b"fmt " && fmt.replace(chunk).is_some() {
            return Err(invalid("ref_audio", "duplicate fmt"));
        }
        if kind == b"data" && data.replace(chunk).is_some() {
            return Err(invalid("ref_audio", "duplicate data"));
        }
        position += size + size % 2;
        if position > end {
            return Err(invalid("ref_audio", "missing WAV padding"));
        }
    }
    let fmt = fmt
        .filter(|f| f.len() >= 16)
        .ok_or_else(|| invalid("ref_audio", "missing fmt"))?;
    let format = PcmFormat {
        channels: u16le(&fmt[2..4]),
        sample_rate: u32le(&fmt[4..8]),
    };
    let max_samples = limits.sample_limit(format)?;
    let codec = u16le(&fmt[..2]);
    let bits = u16le(&fmt[14..16]);
    if !matches!((codec, bits), (1, 16) | (1, 24) | (1, 32) | (3, 32)) {
        return Err(AudioError::UnsupportedFormat);
    }
    let width = usize::from(bits / 8);
    let alignment = width * usize::from(format.channels);
    if usize::from(u16le(&fmt[12..14])) != alignment
        || u64::from(u32le(&fmt[8..12])) != u64::from(format.sample_rate) * alignment as u64
    {
        return Err(invalid("ref_audio", "inconsistent WAV format"));
    }
    let data = data.ok_or_else(|| invalid("ref_audio", "missing data"))?;
    if data.len() % alignment != 0 {
        return Err(invalid("ref_audio", "partial WAV frame"));
    }
    if data.len() / width > max_samples {
        return Err(AudioError::CapacityExceeded {
            resource: "decoded audio",
        });
    }
    let samples = data
        .chunks_exact(width)
        .map(|v| match (codec, bits) {
            (3, 32) => f32::from_le_bytes(v.try_into().expect("fixed sample width")),
            (1, 16) => {
                i16::from_le_bytes(v.try_into().expect("fixed sample width")) as f32 / 32768.
            }
            (1, 24) => ((i32::from_le_bytes([v[0], v[1], v[2], 0]) << 8) >> 8) as f32 / 8388608.,
            _ => i32::from_le_bytes(v.try_into().expect("fixed sample width")) as f32 / 2147483648.,
        })
        .collect();
    let pcm = PcmBuffer { format, samples };
    pcm.validate()?;
    Ok(pcm)
}
// A demuxer's EOF can also mean an incomplete final MPEG frame. Validate the
// frame envelope separately, so a truncated MP3 cannot become successful PCM.
fn validate_mp3_frames(bytes: &[u8]) -> Result<()> {
    let mut start = 0;
    let mut end = bytes.len();
    if bytes.starts_with(b"ID3") {
        if bytes.len() < 10 || bytes[6..10].iter().any(|b| b & 0x80 != 0) {
            return Err(invalid("ref_audio", "invalid ID3 header"));
        }
        let size = bytes[6..10]
            .iter()
            .fold(0usize, |n, b| (n << 7) | usize::from(*b));
        start = 10
            + size
            + if bytes[3] == 4 && bytes[5] & 0x10 != 0 {
                10
            } else {
                0
            };
        if start > end {
            return Err(invalid("ref_audio", "truncated ID3"));
        }
    }
    if end >= 128 && &bytes[end - 128..end - 125] == b"TAG" {
        end -= 128;
    }
    let mut frames = 0;
    while start < end {
        if end - start < 4 {
            return Err(invalid("ref_audio", "truncated MPEG header"));
        }
        let b = &bytes[start..start + 4];
        let version = (b[1] >> 3) & 3;
        let layer = (b[1] >> 1) & 3;
        let bitrate_index = usize::from(b[2] >> 4);
        let rate_index = usize::from((b[2] >> 2) & 3);
        if b[0] != 0xff
            || b[1] & 0xe0 != 0xe0
            || version == 1
            || layer != 1
            || rate_index == 3
            || bitrate_index == 15
        {
            return Err(invalid("ref_audio", "invalid MP3 frame"));
        }
        if bitrate_index == 0 {
            return Err(AudioError::UnsupportedFormat);
        }
        let rate = [44100usize, 48000, 32000][rate_index]
            / match version {
                3 => 1,
                2 => 2,
                _ => 4,
            };
        let bitrate = if version == 3 {
            [
                0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320,
            ][bitrate_index]
        } else {
            [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160][bitrate_index]
        };
        let size = (if version == 3 { 144000 } else { 72000 }) * bitrate / rate
            + usize::from((b[2] >> 1) & 1);
        if size < 4 || size > end - start {
            return Err(invalid("ref_audio", "truncated MP3 frame"));
        }
        start += size;
        frames += 1;
    }
    if frames == 0 {
        return Err(invalid("ref_audio", "no MP3 frames"));
    }
    Ok(())
}
/// Quantization shared by full WAV and chunked raw PCM. Half values round away from zero.
pub fn write_pcm_s16le(pcm: &PcmBuffer, mut output: impl Write) -> Result<()> {
    pcm.validate()?;
    let mut buffer = [0u8; 8192];
    for chunk in pcm.samples.chunks(buffer.len() / 2) {
        for (index, sample) in chunk.iter().enumerate() {
            let value = (*sample * 32768.).round().clamp(-32768., 32767.) as i16;
            buffer[index * 2..index * 2 + 2].copy_from_slice(&value.to_le_bytes());
        }
        output.write_all(&buffer[..chunk.len() * 2])?;
    }
    Ok(())
}
/// Encode known-length PCM directly into bounded storage owned by the caller.
pub fn write_wav(pcm: &PcmBuffer, mut output: impl Write) -> Result<()> {
    pcm.validate()?;
    let data_size = u32::try_from(
        pcm.samples
            .len()
            .checked_mul(2)
            .ok_or(AudioError::CapacityExceeded { resource: "WAV" })?,
    )
    .map_err(|_| AudioError::CapacityExceeded { resource: "WAV" })?;
    let riff_size = data_size
        .checked_add(36)
        .ok_or(AudioError::CapacityExceeded { resource: "WAV" })?;
    let block = pcm
        .format
        .channels
        .checked_mul(2)
        .ok_or_else(|| invalid("pcm.format", "channel overflow"))?;
    let byte_rate = pcm
        .format
        .sample_rate
        .checked_mul(u32::from(block))
        .ok_or_else(|| invalid("pcm.format", "rate overflow"))?;
    output.write_all(b"RIFF")?;
    output.write_all(&riff_size.to_le_bytes())?;
    output.write_all(b"WAVEfmt ")?;
    output.write_all(&16u32.to_le_bytes())?;
    output.write_all(&1u16.to_le_bytes())?;
    output.write_all(&pcm.format.channels.to_le_bytes())?;
    output.write_all(&pcm.format.sample_rate.to_le_bytes())?;
    output.write_all(&byte_rate.to_le_bytes())?;
    output.write_all(&block.to_le_bytes())?;
    output.write_all(&16u16.to_le_bytes())?;
    output.write_all(b"data")?;
    output.write_all(&data_size.to_le_bytes())?;
    write_pcm_s16le(pcm, output)
}
