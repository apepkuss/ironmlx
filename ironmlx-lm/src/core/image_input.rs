use std::fmt;
use std::io::Cursor;

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
pub enum ImageInputError {
    ModelUnsupported,
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
    pub fn code(self) -> &'static str {
        match self {
            Self::ModelUnsupported => "image_input_unsupported",
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
            Self::ModelUnsupported => "The loaded model does not support image input.",
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

fn reader(bytes: &[u8]) -> Result<ImageReader<Cursor<&[u8]>>, ImageInputError> {
    ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|_| ImageInputError::DecodeFailed)
}

pub fn inspect_image(bytes: &[u8]) -> Result<(u32, u32, ImageFormat), ImageInputError> {
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
