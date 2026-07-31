use std::fmt;
use std::io::Cursor;

use axum::http::StatusCode;
use base64::Engine;
use image::{DynamicImage, ImageFormat, ImageReader};

pub const MAX_IMAGE_COUNT: usize = 8;
pub const MAX_IMAGE_BYTES: usize = 10 * 1024 * 1024;
pub const MAX_TOTAL_IMAGE_BYTES: usize = 24 * 1024 * 1024;
pub const MAX_TEXT_BYTES: usize = 2 * 1024 * 1024;
pub const MAX_IMAGE_SIDE: u32 = 8192;
pub const MAX_IMAGE_PIXELS: u64 = 16_777_216;
pub const MAX_TOTAL_IMAGE_PIXELS: u64 = 33_554_432;
pub const MAX_IMAGE_DECODER_ALLOC_BYTES: u64 = 96 * 1024 * 1024;
pub const MAX_IMAGE_BASE64_BYTES: usize = 4 * MAX_IMAGE_BYTES.div_ceil(3);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SupportedImageMediaType {
    Jpeg,
    Png,
    Webp,
}

impl SupportedImageMediaType {
    pub fn parse(value: &str) -> Result<Self, ImageInputError> {
        match value {
            "image/jpeg" => Ok(Self::Jpeg),
            "image/png" => Ok(Self::Png),
            "image/webp" => Ok(Self::Webp),
            _ => Err(ImageInputError::MediaTypeUnsupported),
        }
    }

    fn format(self) -> ImageFormat {
        match self {
            Self::Jpeg => ImageFormat::Jpeg,
            Self::Png => ImageFormat::Png,
            Self::Webp => ImageFormat::WebP,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageInputError {
    RemoteUrlForbidden,
    DataUrlInvalid,
    MediaTypeUnsupported,
    ImageCountExceeded,
    EncodedTooLarge,
    DecodedTooLarge,
    TotalDecodedTooLarge,
    DimensionsExceeded,
    PixelBudgetExceeded,
    TotalPixelBudgetExceeded,
    TextTooLarge,
    DecodeFailed,
}

impl ImageInputError {
    pub fn status(self) -> StatusCode {
        match self {
            Self::EncodedTooLarge
            | Self::DecodedTooLarge
            | Self::TotalDecodedTooLarge
            | Self::ImageCountExceeded
            | Self::DimensionsExceeded
            | Self::PixelBudgetExceeded
            | Self::TotalPixelBudgetExceeded
            | Self::TextTooLarge => StatusCode::PAYLOAD_TOO_LARGE,
            _ => StatusCode::BAD_REQUEST,
        }
    }

    pub fn code(self) -> &'static str {
        match self {
            Self::RemoteUrlForbidden => "image_remote_url_forbidden",
            Self::DataUrlInvalid => "image_data_url_invalid",
            Self::MediaTypeUnsupported => "image_media_type_unsupported",
            Self::ImageCountExceeded => "image_count_exceeded",
            Self::EncodedTooLarge => "image_encoded_too_large",
            Self::DecodedTooLarge => "image_decoded_too_large",
            Self::TotalDecodedTooLarge => "image_total_decoded_too_large",
            Self::DimensionsExceeded => "image_dimensions_exceeded",
            Self::PixelBudgetExceeded => "image_pixel_budget_exceeded",
            Self::TotalPixelBudgetExceeded => "image_total_pixel_budget_exceeded",
            Self::TextTooLarge => "text_content_too_large",
            Self::DecodeFailed => "image_decode_failed",
        }
    }

    pub fn message(self) -> &'static str {
        match self {
            Self::RemoteUrlForbidden => {
                "Remote image URLs are forbidden; upload image content as base64 data."
            }
            Self::DataUrlInvalid => "The image data URL is malformed.",
            Self::MediaTypeUnsupported => "Only JPEG, PNG, and WebP images are supported.",
            Self::ImageCountExceeded => "A request may contain at most 8 images.",
            Self::EncodedTooLarge => "An encoded image exceeds the base64 size limit.",
            Self::DecodedTooLarge => "A decoded image exceeds the 10 MiB limit.",
            Self::TotalDecodedTooLarge => "Decoded images exceed the 24 MiB request limit.",
            Self::DimensionsExceeded => "An image dimension exceeds 8192 pixels.",
            Self::PixelBudgetExceeded => "An image exceeds the 16 megapixel limit.",
            Self::TotalPixelBudgetExceeded => "Images exceed the 32 megapixel request limit.",
            Self::TextTooLarge => "Text content exceeds the 2 MiB request limit.",
            Self::DecodeFailed => "The image could not be decoded safely.",
        }
    }
}

impl fmt::Display for ImageInputError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.code())
    }
}

impl std::error::Error for ImageInputError {}

#[derive(Default)]
pub struct ImageRequestBudget {
    image_count: usize,
    total_image_bytes: usize,
    total_pixels: u64,
    total_text_bytes: usize,
}

impl ImageRequestBudget {
    pub fn add_text(&mut self, text: &str) -> Result<(), ImageInputError> {
        self.total_text_bytes = self
            .total_text_bytes
            .checked_add(text.len())
            .ok_or(ImageInputError::TextTooLarge)?;
        if self.total_text_bytes > MAX_TEXT_BYTES {
            return Err(ImageInputError::TextTooLarge);
        }
        Ok(())
    }

    pub fn decode_data_url(&mut self, url: &str) -> Result<Vec<u8>, ImageInputError> {
        if url.starts_with("http://") || url.starts_with("https://") {
            return Err(ImageInputError::RemoteUrlForbidden);
        }
        let rest = url
            .strip_prefix("data:")
            .ok_or(ImageInputError::DataUrlInvalid)?;
        let (metadata, data) = rest
            .split_once(',')
            .ok_or(ImageInputError::DataUrlInvalid)?;
        let media_type = metadata
            .strip_suffix(";base64")
            .ok_or(ImageInputError::DataUrlInvalid)?;
        self.decode_base64(media_type, data)
    }

    pub fn decode_base64(
        &mut self,
        media_type: &str,
        data: &str,
    ) -> Result<Vec<u8>, ImageInputError> {
        let media_type = SupportedImageMediaType::parse(media_type)?;
        self.image_count = self
            .image_count
            .checked_add(1)
            .ok_or(ImageInputError::ImageCountExceeded)?;
        if self.image_count > MAX_IMAGE_COUNT {
            return Err(ImageInputError::ImageCountExceeded);
        }
        if data.len() > MAX_IMAGE_BASE64_BYTES {
            return Err(ImageInputError::EncodedTooLarge);
        }
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(data)
            .map_err(|_| ImageInputError::DataUrlInvalid)?;
        if bytes.len() > MAX_IMAGE_BYTES {
            return Err(ImageInputError::DecodedTooLarge);
        }
        let (width, height, actual_format) = inspect_image(&bytes)?;
        if actual_format != media_type.format() {
            return Err(ImageInputError::MediaTypeUnsupported);
        }
        if width > MAX_IMAGE_SIDE || height > MAX_IMAGE_SIDE {
            return Err(ImageInputError::DimensionsExceeded);
        }
        let pixels = u64::from(width)
            .checked_mul(u64::from(height))
            .ok_or(ImageInputError::PixelBudgetExceeded)?;
        if pixels > MAX_IMAGE_PIXELS {
            return Err(ImageInputError::PixelBudgetExceeded);
        }
        self.total_image_bytes = self
            .total_image_bytes
            .checked_add(bytes.len())
            .ok_or(ImageInputError::TotalDecodedTooLarge)?;
        if self.total_image_bytes > MAX_TOTAL_IMAGE_BYTES {
            return Err(ImageInputError::TotalDecodedTooLarge);
        }
        self.total_pixels = self
            .total_pixels
            .checked_add(pixels)
            .ok_or(ImageInputError::TotalPixelBudgetExceeded)?;
        if self.total_pixels > MAX_TOTAL_IMAGE_PIXELS {
            return Err(ImageInputError::TotalPixelBudgetExceeded);
        }
        Ok(bytes)
    }
}

fn reader(bytes: &[u8]) -> Result<ImageReader<Cursor<&[u8]>>, ImageInputError> {
    ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|_| ImageInputError::DecodeFailed)
}

fn inspect_image(bytes: &[u8]) -> Result<(u32, u32, ImageFormat), ImageInputError> {
    let reader = reader(bytes)?;
    let format = reader.format().ok_or(ImageInputError::DecodeFailed)?;
    if !matches!(
        format,
        ImageFormat::Jpeg | ImageFormat::Png | ImageFormat::WebP
    ) {
        return Err(ImageInputError::MediaTypeUnsupported);
    }
    let (width, height) = reader
        .into_dimensions()
        .map_err(|_| ImageInputError::DecodeFailed)?;
    Ok((width, height, format))
}

pub fn load_from_memory_bounded(bytes: &[u8]) -> image::ImageResult<DynamicImage> {
    let mut reader = ImageReader::new(Cursor::new(bytes)).with_guessed_format()?;
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(MAX_IMAGE_SIDE);
    limits.max_image_height = Some(MAX_IMAGE_SIDE);
    limits.max_alloc = Some(MAX_IMAGE_DECODER_ALLOC_BYTES);
    reader.limits(limits);
    reader.decode()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remote_urls_are_rejected_with_stable_code() {
        let error = ImageRequestBudget::default()
            .decode_data_url("https://example.com/image.png")
            .unwrap_err();
        assert_eq!(error.code(), "image_remote_url_forbidden");
    }

    #[test]
    fn encoded_size_is_checked_before_decode() {
        let oversized = "A".repeat(MAX_IMAGE_BASE64_BYTES + 1);
        let error = ImageRequestBudget::default()
            .decode_data_url(&format!("data:image/png;base64,{oversized}"))
            .unwrap_err();
        assert_eq!(error, ImageInputError::EncodedTooLarge);
    }

    #[test]
    fn unsupported_media_type_is_rejected() {
        let error = ImageRequestBudget::default()
            .decode_data_url("data:image/svg+xml;base64,PHN2Zz4=")
            .unwrap_err();
        assert_eq!(error, ImageInputError::MediaTypeUnsupported);
    }

    #[test]
    fn oversized_dimensions_are_rejected_from_metadata() {
        let image = DynamicImage::new_rgba8(MAX_IMAGE_SIDE + 1, 1);
        let mut png = Vec::new();
        image
            .write_to(&mut Cursor::new(&mut png), ImageFormat::Png)
            .unwrap();
        let encoded = base64::engine::general_purpose::STANDARD.encode(png);
        let error = ImageRequestBudget::default()
            .decode_data_url(&format!("data:image/png;base64,{encoded}"))
            .unwrap_err();
        assert_eq!(error, ImageInputError::DimensionsExceeded);
    }

    #[test]
    fn image_count_and_text_budgets_are_enforced() {
        const PNG: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=";
        let mut budget = ImageRequestBudget::default();
        for _ in 0..MAX_IMAGE_COUNT {
            budget
                .decode_data_url(&format!("data:image/png;base64,{PNG}"))
                .unwrap();
        }
        assert_eq!(
            budget
                .decode_data_url(&format!("data:image/png;base64,{PNG}"))
                .unwrap_err(),
            ImageInputError::ImageCountExceeded
        );

        let mut budget = ImageRequestBudget::default();
        assert_eq!(
            budget
                .add_text(&"x".repeat(MAX_TEXT_BYTES + 1))
                .unwrap_err(),
            ImageInputError::TextTooLarge
        );
    }
}
