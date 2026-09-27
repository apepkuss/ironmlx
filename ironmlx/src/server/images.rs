//! OpenAI-compatible text-to-image generation.

use std::io::Cursor;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::{extract::Multipart, response::IntoResponse, Json};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use image::{DynamicImage, ImageFormat, RgbImage, RgbaImage};
use ironmlx_image::models::QwenImage21GenerationConfig;
use ironmlx_runtime::core::{
    diffusion_execution::DiffusionGemmaLaneError,
    qwen_image_execution::{QwenImageGenerateRequest, QwenImageRuntime},
};
use mlx::{Array, Dtype};
use serde::{Deserialize, Serialize};

use super::api_error::{ApiError, ApiProtocol};
use super::image_input::ImageRequestBudget;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ImagesGenerationRequest {
    pub model: Option<String>,
    pub prompt: String,
    #[serde(default = "default_image_count")]
    pub n: usize,
    pub size: Option<String>,
    pub response_format: Option<String>,
    #[serde(rename = "quality")]
    pub _quality: Option<String>,
    #[serde(rename = "style")]
    pub _style: Option<String>,
    #[serde(rename = "user")]
    pub _user: Option<String>,
    /// IronMLX extension for reproducible local generation.
    pub seed: Option<u64>,
    /// IronMLX extension controlling the flow-matching denoising schedule.
    pub inference_steps: Option<usize>,
}

#[derive(Debug)]
pub(crate) struct ImagesEditRequest {
    pub model: Option<String>,
    pub prompt: String,
    pub image: Vec<u8>,
    pub n: usize,
    pub size: Option<String>,
    pub response_format: Option<String>,
    pub seed: Option<u64>,
    pub inference_steps: Option<usize>,
}

fn default_image_count() -> usize {
    1
}

#[derive(Debug, Serialize)]
struct ImagesGenerationResponse {
    created: u64,
    data: Vec<GeneratedImage>,
}

#[derive(Debug, Serialize)]
struct GeneratedImage {
    b64_json: String,
}

impl ImagesGenerationRequest {
    fn generation_config(&self) -> Result<QwenImage21GenerationConfig, Box<ApiError>> {
        if self.prompt.trim().is_empty() {
            return Err(Box::new(ApiError::invalid_request(
                "invalid_image_prompt",
                "prompt must not be empty",
            )));
        }
        if self.n != 1 {
            return Err(Box::new(ApiError::invalid_request(
                "unsupported_image_count",
                "Qwen Image 2.1 currently supports exactly one image per request (n=1)",
            )));
        }
        if !matches!(self.response_format.as_deref(), None | Some("b64_json")) {
            return Err(Box::new(ApiError::invalid_request(
                "unsupported_image_response_format",
                "response_format must be b64_json; URL-hosted output is not supported",
            )));
        }

        let (width, height) = parse_size(self.size.as_deref().unwrap_or("1024x1024"))?;
        let config = QwenImage21GenerationConfig {
            width,
            height,
            inference_steps: self.inference_steps.unwrap_or(40),
            seed: self.seed.unwrap_or_else(random_seed),
        };
        config.validate().map_err(|error| {
            Box::new(ApiError::invalid_request(
                "invalid_image_parameters",
                error.to_string(),
            ))
        })
    }
}

impl ImagesEditRequest {
    fn generation_config(&self) -> Result<QwenImage21GenerationConfig, Box<ApiError>> {
        ImagesGenerationRequest {
            model: self.model.clone(),
            prompt: self.prompt.clone(),
            n: self.n,
            size: self.size.clone(),
            response_format: self.response_format.clone(),
            _quality: None,
            _style: None,
            _user: None,
            seed: self.seed,
            inference_steps: self.inference_steps,
        }
        .generation_config()
    }
}

pub(crate) async fn parse_edit_request(
    mut multipart: Multipart,
) -> Result<ImagesEditRequest, axum::response::Response> {
    let mut model = None;
    let mut prompt = None;
    let mut image = None;
    let mut n = None;
    let mut size = None;
    let mut response_format = None;
    let mut seed = None;
    let mut inference_steps = None;
    let mut budget = ImageRequestBudget::default();

    while let Some(field) = multipart.next_field().await.map_err(|error| {
        ApiError::invalid_request(
            "invalid_image_edit_multipart",
            format!("invalid multipart image edit request: {error}"),
        )
        .into_response(ApiProtocol::OpenAi)
    })? {
        let Some(name) = field.name().map(str::to_owned) else {
            return Err(ApiError::invalid_request(
                "invalid_image_edit_field",
                "multipart fields must have names",
            )
            .into_response(ApiProtocol::OpenAi));
        };
        match name.as_str() {
            "image" => {
                if image.is_some() {
                    return Err(ApiError::invalid_request(
                        "unsupported_image_count",
                        "Qwen Image 2.1 currently accepts exactly one condition image",
                    )
                    .into_response(ApiProtocol::OpenAi));
                }
                let media_type = field.content_type().map(str::to_owned);
                let bytes = field.bytes().await.map_err(|error| {
                    ApiError::invalid_request(
                        "image_decode_failed",
                        format!("reading condition image failed: {error}"),
                    )
                    .into_response(ApiProtocol::OpenAi)
                })?;
                image = Some(
                    budget
                        .add_upload(media_type.as_deref(), bytes.to_vec())
                        .map_err(|error| {
                            ApiError::from_status(
                                super::image_input::image_error_status(error),
                                error.code(),
                                error.message(),
                            )
                            .into_response(ApiProtocol::OpenAi)
                        })?,
                );
            }
            "mask" => {
                return Err(ApiError::invalid_request(
                    "unsupported_image_mask",
                    "Qwen Image 2.1 condition editing does not accept an inpainting mask",
                )
                .into_response(ApiProtocol::OpenAi));
            }
            "model" | "prompt" | "n" | "size" | "response_format" | "seed" | "inference_steps"
            | "quality" | "user" => {
                let value = field.text().await.map_err(|error| {
                    ApiError::invalid_request(
                        "invalid_image_edit_field",
                        format!("reading multipart field {name} failed: {error}"),
                    )
                    .into_response(ApiProtocol::OpenAi)
                })?;
                budget.add_text(&value).map_err(|error| {
                    ApiError::from_status(
                        super::image_input::image_error_status(error),
                        error.code(),
                        error.message(),
                    )
                    .into_response(ApiProtocol::OpenAi)
                })?;
                match name.as_str() {
                    "model" => model = Some(value),
                    "prompt" => prompt = Some(value),
                    "n" => {
                        n = Some(
                            parse_form_value(&name, &value)
                                .map_err(|error| (*error).into_response(ApiProtocol::OpenAi))?,
                        )
                    }
                    "size" => size = Some(value),
                    "response_format" => response_format = Some(value),
                    "seed" => {
                        seed = Some(
                            parse_form_value(&name, &value)
                                .map_err(|error| (*error).into_response(ApiProtocol::OpenAi))?,
                        )
                    }
                    "inference_steps" => {
                        inference_steps = Some(
                            parse_form_value(&name, &value)
                                .map_err(|error| (*error).into_response(ApiProtocol::OpenAi))?,
                        )
                    }
                    "quality" | "user" => {}
                    _ => unreachable!(),
                }
            }
            _ => {
                return Err(ApiError::invalid_request(
                    "unsupported_image_edit_field",
                    format!("unsupported image edit field: {name}"),
                )
                .into_response(ApiProtocol::OpenAi));
            }
        }
    }

    Ok(ImagesEditRequest {
        model,
        prompt: prompt.ok_or_else(|| {
            ApiError::invalid_request("missing_image_prompt", "prompt is required")
                .into_response(ApiProtocol::OpenAi)
        })?,
        image: image.ok_or_else(|| {
            ApiError::invalid_request("missing_condition_image", "image is required")
                .into_response(ApiProtocol::OpenAi)
        })?,
        n: n.unwrap_or(1),
        size,
        response_format,
        seed,
        inference_steps,
    })
}

fn parse_form_value<T>(name: &str, value: &str) -> Result<T, Box<ApiError>>
where
    T: std::str::FromStr,
{
    value.parse().map_err(|_| {
        Box::new(ApiError::invalid_request(
            "invalid_image_edit_field",
            format!("multipart field {name} has an invalid value"),
        ))
    })
}

fn parse_size(size: &str) -> Result<(u32, u32), Box<ApiError>> {
    let (width, height) = size.split_once('x').ok_or_else(|| {
        Box::new(ApiError::invalid_request(
            "invalid_image_size",
            "size must use the WIDTHxHEIGHT format, for example 1024x1024",
        ))
    })?;
    let width = width.parse::<u32>().map_err(|_| {
        Box::new(ApiError::invalid_request(
            "invalid_image_size",
            "image width must be an integer",
        ))
    })?;
    let height = height.parse::<u32>().map_err(|_| {
        Box::new(ApiError::invalid_request(
            "invalid_image_size",
            "image height must be an integer",
        ))
    })?;
    Ok((width, height))
}

fn random_seed() -> u64 {
    let time = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;
    time ^ (u64::from(std::process::id()) << 32)
}

pub(crate) async fn generate_with_state(
    state: QwenImageRuntime,
    request: ImagesGenerationRequest,
) -> axum::response::Response {
    let config = match request.generation_config() {
        Ok(config) => config,
        Err(error) => return (*error).into_response(ApiProtocol::OpenAi),
    };
    execute_with_state(state, request.prompt, config, None).await
}

pub(crate) async fn edit_with_state(
    state: QwenImageRuntime,
    request: ImagesEditRequest,
) -> axum::response::Response {
    let config = match request.generation_config() {
        Ok(config) => config,
        Err(error) => return (*error).into_response(ApiProtocol::OpenAi),
    };
    execute_with_state(state, request.prompt, config, Some(request.image)).await
}

async fn execute_with_state(
    state: QwenImageRuntime,
    prompt: String,
    config: QwenImage21GenerationConfig,
    condition_image: Option<Vec<u8>>,
) -> axum::response::Response {
    let admitted = match state
        .admit(QwenImageGenerateRequest {
            prompt,
            config,
            condition_image,
        })
        .await
    {
        Ok(admitted) => admitted,
        Err(DiffusionGemmaLaneError::Overloaded) => {
            return ApiError::service_unavailable(
                "image_queue_full",
                "The image generation queue is full; retry later",
            )
            .into_response(ApiProtocol::OpenAi)
        }
        Err(DiffusionGemmaLaneError::Closed) => {
            return ApiError::service_unavailable(
                "image_runtime_unavailable",
                "The image generation runtime is unavailable",
            )
            .into_response(ApiProtocol::OpenAi)
        }
    };

    let generated = admitted
        .spawn(|execution| {
            let image = execution.generate()?;
            let encoded = encode_png_base64(&image)?;
            mlx::transforms::clear_cache();
            Ok::<_, String>(encoded)
        })
        .await;

    let b64_json = match generated {
        Ok(Ok(encoded)) => encoded,
        Ok(Err(error)) => {
            return ApiError::internal("image_generation_failed", error)
                .into_response(ApiProtocol::OpenAi)
        }
        Err(error) => {
            return ApiError::internal(
                "image_generation_task_failed",
                format!("image generation task failed: {error}"),
            )
            .into_response(ApiProtocol::OpenAi)
        }
    };

    Json(ImagesGenerationResponse {
        created: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        data: vec![GeneratedImage { b64_json }],
    })
    .into_response()
}

fn encode_png_base64(image: &Array) -> Result<String, String> {
    let shape = image.shape();
    let shape = shape.as_slice();
    if shape.len() != 4 || shape[0] != 1 || !matches!(shape[1], 3 | 4) {
        return Err(format!(
            "Qwen Image 2.1 returned an invalid image tensor shape: {shape:?}"
        ));
    }
    let height = usize::try_from(shape[2]).map_err(|_| "invalid image height".to_owned())?;
    let width = usize::try_from(shape[3]).map_err(|_| "invalid image width".to_owned())?;
    let values = mlx::ops::astype(image, Dtype::Float32)
        .and_then(|array| array.to_vec::<f32>())
        .map_err(|error| format!("reading generated image pixels failed: {error}"))?;
    let plane = width
        .checked_mul(height)
        .ok_or_else(|| "generated image dimensions overflow".to_owned())?;
    let channels = usize::try_from(shape[1]).map_err(|_| "invalid channel count".to_owned())?;
    if values.len() != plane * channels {
        return Err("generated image tensor has an inconsistent element count".to_owned());
    }

    let mut pixels = Vec::with_capacity(plane * channels);
    for pixel in 0..plane {
        for channel in 0..channels {
            let value = values[channel * plane + pixel];
            pixels.push(((value * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0).round() as u8);
        }
    }
    let image = if channels == 4 {
        let image = RgbaImage::from_raw(width as u32, height as u32, pixels)
            .ok_or_else(|| "constructing RGBA PNG image failed".to_owned())?;
        DynamicImage::ImageRgba8(image)
    } else {
        let image = RgbImage::from_raw(width as u32, height as u32, pixels)
            .ok_or_else(|| "constructing RGB PNG image failed".to_owned())?;
        DynamicImage::ImageRgb8(image)
    };
    let mut bytes = Cursor::new(Vec::new());
    image
        .write_to(&mut bytes, ImageFormat::Png)
        .map_err(|error| format!("encoding generated image as PNG failed: {error}"))?;
    Ok(STANDARD.encode(bytes.into_inner()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request() -> ImagesGenerationRequest {
        ImagesGenerationRequest {
            model: Some("qwen-image".to_owned()),
            prompt: "a red fox".to_owned(),
            n: 1,
            size: None,
            response_format: None,
            _quality: None,
            _style: None,
            _user: None,
            seed: Some(7),
            inference_steps: Some(4),
        }
    }

    #[test]
    fn validates_supported_request_shape() {
        let config = request().generation_config().expect("valid request");
        assert_eq!((config.width, config.height), (1024, 1024));
        assert_eq!(config.seed, 7);
        assert_eq!(config.inference_steps, 4);
    }

    #[test]
    fn rejects_multiple_images_and_url_output() {
        let mut req = request();
        req.n = 2;
        assert!(req.generation_config().is_err());
        req.n = 1;
        req.response_format = Some("url".to_owned());
        assert!(req.generation_config().is_err());
    }

    #[test]
    fn parses_rectangular_size() {
        assert_eq!(parse_size("1536x1024").expect("valid size"), (1536, 1024));
        assert!(parse_size("1024").is_err());
    }

    #[test]
    fn rejects_unknown_request_fields() {
        let error = serde_json::from_value::<ImagesGenerationRequest>(serde_json::json!({
            "model": "qwen-image",
            "prompt": "a red fox",
            "response_formatt": "b64_json"
        }))
        .expect_err("unknown public fields must not be silently ignored");
        assert!(error.to_string().contains("unknown field"));
    }

    #[test]
    fn edit_request_reuses_generation_parameter_contract() {
        let request = ImagesEditRequest {
            model: Some("qwen-image".to_owned()),
            prompt: "make the sky warmer".to_owned(),
            image: vec![1, 2, 3],
            n: 1,
            size: Some("768x1024".to_owned()),
            response_format: Some("b64_json".to_owned()),
            seed: Some(11),
            inference_steps: Some(8),
        };
        let config = request.generation_config().expect("valid edit config");
        assert_eq!((config.width, config.height), (768, 1024));
        assert_eq!(config.seed, 11);
        assert_eq!(config.inference_steps, 8);
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn encodes_nchw_rgba_as_png() {
        let metallib = std::path::PathBuf::from(
            std::env::var("MLX_DIR").expect("MLX_DIR must point to the local MLX install"),
        )
        .join("lib/mlx.metallib");
        mlx::metal::set_metallib_path(
            metallib
                .to_str()
                .expect("MLX_DIR/lib/mlx.metallib must be UTF-8"),
        )
        .expect("load MLX metallib");
        // Two pixels: opaque red, then transparent green.
        let values = [
            1.0_f32, -1.0, // R
            -1.0, 1.0, // G
            -1.0, -1.0, // B
            1.0, -1.0, // A
        ];
        let array: Array = (&values[..], (1_i32, 4_i32, 1_i32, 2_i32))
            .try_into()
            .expect("test image array");
        let encoded = encode_png_base64(&array).expect("encode PNG");
        let bytes = STANDARD.decode(encoded).expect("decode base64");
        let image = image::load_from_memory_with_format(&bytes, ImageFormat::Png)
            .expect("decode PNG")
            .to_rgba8();
        assert_eq!(image.dimensions(), (2, 1));
        assert_eq!(image.get_pixel(0, 0).0, [255, 0, 0, 255]);
        assert_eq!(image.get_pixel(1, 0).0, [0, 255, 0, 0]);
    }
}
