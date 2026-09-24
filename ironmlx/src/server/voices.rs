//! Persistent voice profiles used by OpenAI-compatible speech requests.

use super::api_error::{ApiError, ApiProtocol};
use axum::{
    body::Body,
    extract::{Extension, Path},
    http::{header, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, patch},
    Json, Router,
};
use base64::{engine::general_purpose::STANDARD, Engine};
use ironmlx_audio::{io::NativeAudioIo, AudioIo, DecodeLimits};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::HashSet,
    path::{Component, Path as FilePath, PathBuf},
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};
use tokio::sync::RwLock;

const STORE_VERSION: u32 = 1;

#[derive(Clone)]
pub(crate) struct VoiceStore {
    root: Arc<PathBuf>,
    document: Arc<RwLock<VoiceDocument>>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct VoiceDocument {
    version: u32,
    voices: Vec<StoredVoiceProfile>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct StoredVoiceProfile {
    id: String,
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    language: Option<String>,
    enabled: bool,
    reference_file: String,
    reference_format: String,
    reference_sha256: String,
    sample_rate: u32,
    channels: u16,
    duration_ms: u64,
    created_at: u64,
    updated_at: u64,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct VoiceProfile {
    id: String,
    object: &'static str,
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    language: Option<String>,
    enabled: bool,
    reference_format: String,
    reference_sha256: String,
    sample_rate: u32,
    channels: u16,
    duration_ms: u64,
    created_at: u64,
    updated_at: u64,
}

impl From<&StoredVoiceProfile> for VoiceProfile {
    fn from(value: &StoredVoiceProfile) -> Self {
        Self {
            id: value.id.clone(),
            object: "voice",
            name: value.name.clone(),
            language: value.language.clone(),
            enabled: value.enabled,
            reference_format: value.reference_format.clone(),
            reference_sha256: value.reference_sha256.clone(),
            sample_rate: value.sample_rate,
            channels: value.channels,
            duration_ms: value.duration_ms,
            created_at: value.created_at,
            updated_at: value.updated_at,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
struct PublicVoiceProfile {
    id: String,
    object: &'static str,
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    language: Option<String>,
}

impl From<VoiceProfile> for PublicVoiceProfile {
    fn from(value: VoiceProfile) -> Self {
        Self {
            id: value.id,
            object: value.object,
            name: value.name,
            language: value.language,
        }
    }
}

#[derive(Debug, Serialize)]
struct VoiceList<T> {
    object: &'static str,
    data: Vec<T>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CreateVoiceRequest {
    id: String,
    name: String,
    #[serde(default)]
    language: Option<String>,
    #[serde(default = "enabled_by_default")]
    enabled: bool,
    ref_audio: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct UpdateVoiceRequest {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    language: Option<Option<String>>,
    #[serde(default)]
    enabled: Option<bool>,
    #[serde(default)]
    ref_audio: Option<String>,
}

fn enabled_by_default() -> bool {
    true
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum VoiceStoreError {
    #[error("voice profile '{0}' was not found")]
    NotFound(String),
    #[error("voice profile '{0}' is disabled")]
    Disabled(String),
    #[error("voice profile '{0}' already exists")]
    Conflict(String),
    #[error("invalid voice profile: {0}")]
    Invalid(String),
    #[error("voice profile resource is unavailable: {0}")]
    Resource(String),
    #[error("voice profile store failed: {0}")]
    Internal(String),
}

impl VoiceStoreError {
    pub(crate) fn api_error(&self) -> ApiError {
        match self {
            Self::NotFound(_) | Self::Disabled(_) => {
                ApiError::from_status(StatusCode::NOT_FOUND, "voice_not_found", self.to_string())
            }
            Self::Conflict(_) => ApiError::from_status(
                StatusCode::CONFLICT,
                "voice_already_exists",
                self.to_string(),
            ),
            Self::Invalid(_) => {
                ApiError::invalid_request("invalid_voice_profile", self.to_string())
            }
            Self::Resource(_) => {
                ApiError::service_unavailable("voice_resource_unavailable", self.to_string())
            }
            Self::Internal(_) => ApiError::internal("voice_store_failed", self.to_string()),
        }
    }
}

struct ValidatedReference {
    bytes: Vec<u8>,
    format: &'static str,
    sha256: String,
    sample_rate: u32,
    channels: u16,
    duration_ms: u64,
}

impl VoiceStore {
    pub(crate) fn default_root() -> crate::Result<PathBuf> {
        Ok(dirs::home_dir()
            .ok_or_else(|| anyhow::anyhow!("home directory is unavailable"))?
            .join(".ironmlx/audio/voices"))
    }

    pub(crate) fn open(root: PathBuf) -> crate::Result<Self> {
        std::fs::create_dir_all(root.join("files"))?;
        restrict_permissions(&root, 0o700)?;
        restrict_permissions(&root.join("files"), 0o700)?;
        let path = root.join("profiles.json");
        let document = match std::fs::read(&path) {
            Ok(bytes) => serde_json::from_slice::<VoiceDocument>(&bytes)
                .map_err(|error| anyhow::anyhow!("parsing {}: {error}", path.display()))?,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => VoiceDocument {
                version: STORE_VERSION,
                voices: Vec::new(),
            },
            Err(error) => return Err(error.into()),
        };
        validate_document(&document)?;
        if !path.exists() {
            let bytes = serde_json::to_vec_pretty(&VoiceDocument {
                version: STORE_VERSION,
                voices: Vec::new(),
            })?;
            std::fs::write(path, bytes)?;
        }
        restrict_permissions(&root.join("profiles.json"), 0o600)?;
        Ok(Self {
            root: Arc::new(root),
            document: Arc::new(RwLock::new(document)),
        })
    }

    async fn list(&self, include_disabled: bool) -> Vec<VoiceProfile> {
        let document = self.document.read().await;
        document
            .voices
            .iter()
            .filter(|voice| include_disabled || voice.enabled)
            .map(VoiceProfile::from)
            .collect()
    }

    pub(crate) async fn reference_bytes(&self, id: &str) -> Result<Vec<u8>, VoiceStoreError> {
        let (file, expected_sha256) = {
            let document = self.document.read().await;
            let voice = document
                .voices
                .iter()
                .find(|voice| voice.id == id)
                .ok_or_else(|| VoiceStoreError::NotFound(id.to_owned()))?;
            if !voice.enabled {
                return Err(VoiceStoreError::Disabled(id.to_owned()));
            }
            (voice.reference_file.clone(), voice.reference_sha256.clone())
        };
        let bytes = tokio::fs::read(self.root.join("files").join(file))
            .await
            .map_err(|error| VoiceStoreError::Resource(error.to_string()))?;
        verify_reference_hash(&bytes, &expected_sha256)?;
        Ok(bytes)
    }

    async fn create(&self, request: CreateVoiceRequest) -> Result<VoiceProfile, VoiceStoreError> {
        validate_id(&request.id)?;
        let name = validate_name(request.name)?;
        let language = validate_language(request.language)?;
        let reference = validate_reference(request.ref_audio).await?;
        let now = unix_seconds();
        let file_name = reference_file_name(&request.id, &reference);
        let mut document = self.document.write().await;
        if document.voices.iter().any(|voice| voice.id == request.id) {
            return Err(VoiceStoreError::Conflict(request.id));
        }
        self.write_reference(&file_name, &reference.bytes).await?;
        let stored = StoredVoiceProfile {
            id: request.id,
            name,
            language,
            enabled: request.enabled,
            reference_file: file_name.clone(),
            reference_format: reference.format.to_owned(),
            reference_sha256: reference.sha256,
            sample_rate: reference.sample_rate,
            channels: reference.channels,
            duration_ms: reference.duration_ms,
            created_at: now,
            updated_at: now,
        };
        document.voices.push(stored.clone());
        document
            .voices
            .sort_by(|left, right| left.id.cmp(&right.id));
        if let Err(error) = self.persist(&document).await {
            document.voices.retain(|voice| voice.id != stored.id);
            let _ = tokio::fs::remove_file(self.root.join("files").join(file_name)).await;
            return Err(VoiceStoreError::Internal(error.to_string()));
        }
        Ok(VoiceProfile::from(&stored))
    }

    async fn update(
        &self,
        id: &str,
        request: UpdateVoiceRequest,
    ) -> Result<VoiceProfile, VoiceStoreError> {
        validate_id(id)?;
        if request.name.is_none()
            && request.language.is_none()
            && request.enabled.is_none()
            && request.ref_audio.is_none()
        {
            return Err(VoiceStoreError::Invalid(
                "at least one field must be provided".into(),
            ));
        }
        let name = request.name.map(validate_name).transpose()?;
        let language = request.language.map(validate_language).transpose()?;
        let reference = match request.ref_audio {
            Some(value) => Some(validate_reference(value).await?),
            None => None,
        };
        let mut document = self.document.write().await;
        let index = document
            .voices
            .iter()
            .position(|voice| voice.id == id)
            .ok_or_else(|| VoiceStoreError::NotFound(id.to_owned()))?;
        let previous = document.voices[index].clone();
        let mut replacement_file = None;
        if let Some(reference) = &reference {
            let file_name = reference_file_name(id, reference);
            self.write_reference(&file_name, &reference.bytes).await?;
            replacement_file = Some(file_name);
        }
        let voice = &mut document.voices[index];
        if let Some(name) = name {
            voice.name = name;
        }
        if let Some(language) = language {
            voice.language = language;
        }
        if let Some(enabled) = request.enabled {
            voice.enabled = enabled;
        }
        if let (Some(reference), Some(file_name)) = (&reference, &replacement_file) {
            voice.reference_file = file_name.clone();
            voice.reference_format = reference.format.to_owned();
            voice.reference_sha256 = reference.sha256.clone();
            voice.sample_rate = reference.sample_rate;
            voice.channels = reference.channels;
            voice.duration_ms = reference.duration_ms;
        }
        voice.updated_at = unix_seconds();
        let updated = voice.clone();
        if let Err(error) = self.persist(&document).await {
            document.voices[index] = previous.clone();
            if let Some(file_name) =
                replacement_file.filter(|file_name| file_name != &previous.reference_file)
            {
                let _ = tokio::fs::remove_file(self.root.join("files").join(file_name)).await;
            }
            return Err(VoiceStoreError::Internal(error.to_string()));
        }
        if previous.reference_file != updated.reference_file {
            let _ =
                tokio::fs::remove_file(self.root.join("files").join(previous.reference_file)).await;
        }
        Ok(VoiceProfile::from(&updated))
    }

    async fn delete(&self, id: &str) -> Result<VoiceProfile, VoiceStoreError> {
        validate_id(id)?;
        let mut document = self.document.write().await;
        let index = document
            .voices
            .iter()
            .position(|voice| voice.id == id)
            .ok_or_else(|| VoiceStoreError::NotFound(id.to_owned()))?;
        let removed = document.voices.remove(index);
        if let Err(error) = self.persist(&document).await {
            document.voices.insert(index, removed);
            return Err(VoiceStoreError::Internal(error.to_string()));
        }
        let _ = tokio::fs::remove_file(self.root.join("files").join(&removed.reference_file)).await;
        Ok(VoiceProfile::from(&removed))
    }

    async fn preview(&self, id: &str) -> Result<(StoredVoiceProfile, Vec<u8>), VoiceStoreError> {
        let voice = {
            let document = self.document.read().await;
            document
                .voices
                .iter()
                .find(|voice| voice.id == id)
                .cloned()
                .ok_or_else(|| VoiceStoreError::NotFound(id.to_owned()))?
        };
        let bytes = tokio::fs::read(self.root.join("files").join(&voice.reference_file))
            .await
            .map_err(|error| VoiceStoreError::Resource(error.to_string()))?;
        verify_reference_hash(&bytes, &voice.reference_sha256)?;
        Ok((voice, bytes))
    }

    async fn write_reference(&self, file_name: &str, bytes: &[u8]) -> Result<(), VoiceStoreError> {
        let files = self.root.join("files");
        let temporary = files.join(format!(".{}.{}.tmp", file_name, uuid::Uuid::new_v4()));
        tokio::fs::write(&temporary, bytes)
            .await
            .map_err(|error| VoiceStoreError::Internal(error.to_string()))?;
        restrict_permissions(&temporary, 0o600)
            .map_err(|error| VoiceStoreError::Internal(error.to_string()))?;
        tokio::fs::rename(&temporary, files.join(file_name))
            .await
            .map_err(|error| VoiceStoreError::Internal(error.to_string()))
    }

    async fn persist(&self, document: &VoiceDocument) -> crate::Result<()> {
        let bytes = serde_json::to_vec_pretty(document)?;
        let temporary = self
            .root
            .join(format!(".profiles.{}.tmp", uuid::Uuid::new_v4()));
        tokio::fs::write(&temporary, bytes).await?;
        restrict_permissions(&temporary, 0o600)?;
        tokio::fs::rename(temporary, self.root.join("profiles.json")).await?;
        Ok(())
    }
}

fn verify_reference_hash(bytes: &[u8], expected: &str) -> Result<(), VoiceStoreError> {
    let actual = format!("{:x}", Sha256::digest(bytes));
    if actual == expected {
        Ok(())
    } else {
        Err(VoiceStoreError::Resource(
            "reference audio integrity check failed".into(),
        ))
    }
}

#[cfg(unix)]
fn restrict_permissions(path: &FilePath, mode: u32) -> std::io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode))
}

#[cfg(not(unix))]
fn restrict_permissions(_path: &FilePath, _mode: u32) -> std::io::Result<()> {
    Ok(())
}

fn validate_document(document: &VoiceDocument) -> crate::Result<()> {
    if document.version != STORE_VERSION {
        anyhow::bail!(
            "unsupported voice profile store version {}",
            document.version
        );
    }
    let mut ids = HashSet::new();
    for voice in &document.voices {
        validate_id(&voice.id).map_err(|error| anyhow::anyhow!(error.to_string()))?;
        if !ids.insert(&voice.id) {
            anyhow::bail!("duplicate voice profile id '{}'", voice.id);
        }
        let path = FilePath::new(&voice.reference_file);
        if path.components().count() != 1
            || !matches!(path.components().next(), Some(Component::Normal(_)))
        {
            anyhow::bail!("invalid voice reference file '{}'.", voice.reference_file);
        }
    }
    Ok(())
}

fn validate_id(value: &str) -> Result<(), VoiceStoreError> {
    let valid = !value.is_empty()
        && value.len() <= 64
        && value.bytes().enumerate().all(|(index, byte)| {
            byte.is_ascii_alphanumeric() || (index > 0 && matches!(byte, b'.' | b'_' | b'-'))
        });
    if valid {
        Ok(())
    } else {
        Err(VoiceStoreError::Invalid(
            "id must contain 1-64 ASCII letters, digits, '.', '_' or '-', and start with a letter or digit".into(),
        ))
    }
}

fn validate_name(value: String) -> Result<String, VoiceStoreError> {
    let value = value.trim().to_owned();
    if value.is_empty() || value.len() > 128 {
        return Err(VoiceStoreError::Invalid(
            "name must contain 1-128 UTF-8 bytes".into(),
        ));
    }
    Ok(value)
}

fn validate_language(value: Option<String>) -> Result<Option<String>, VoiceStoreError> {
    value
        .map(|value| {
            let value = value.trim().to_owned();
            if value.is_empty() || value.len() > 32 {
                Err(VoiceStoreError::Invalid(
                    "language must contain 1-32 UTF-8 bytes".into(),
                ))
            } else {
                Ok(value)
            }
        })
        .transpose()
}

async fn validate_reference(value: String) -> Result<ValidatedReference, VoiceStoreError> {
    tokio::task::spawn_blocking(move || {
        let bytes = STANDARD.decode(&value).map_err(|_| {
            VoiceStoreError::Invalid("ref_audio must be canonical standard Base64".into())
        })?;
        if STANDARD.encode(&bytes) != value {
            return Err(VoiceStoreError::Invalid(
                "ref_audio must be canonical standard Base64".into(),
            ));
        }
        let format = reference_format(&bytes).ok_or_else(|| {
            VoiceStoreError::Invalid("ref_audio must contain WAV, FLAC or MP3 audio".into())
        })?;
        let pcm = NativeAudioIo
            .decode(&bytes, &DecodeLimits::default())
            .map_err(|error| VoiceStoreError::Invalid(error.to_string()))?;
        let frames = pcm.samples.len() / usize::from(pcm.format.channels);
        let duration_ms = (frames as u64)
            .saturating_mul(1000)
            .checked_div(u64::from(pcm.format.sample_rate))
            .unwrap_or(0);
        let sha256 = format!("{:x}", Sha256::digest(&bytes));
        Ok(ValidatedReference {
            bytes,
            format,
            sha256,
            sample_rate: pcm.format.sample_rate,
            channels: pcm.format.channels,
            duration_ms,
        })
    })
    .await
    .map_err(|error| VoiceStoreError::Internal(error.to_string()))?
}

fn reference_format(bytes: &[u8]) -> Option<&'static str> {
    if bytes.starts_with(b"RIFF") && bytes.get(8..12) == Some(b"WAVE") {
        Some("wav")
    } else if bytes.starts_with(b"fLaC") {
        Some("flac")
    } else if bytes.starts_with(b"ID3")
        || (bytes.len() > 1 && bytes[0] == 0xff && bytes[1] & 0xe0 == 0xe0)
    {
        Some("mp3")
    } else {
        None
    }
}

fn reference_file_name(id: &str, reference: &ValidatedReference) -> String {
    format!("{}-{}.{}", id, &reference.sha256[..16], reference.format)
}

fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn api_error_response(error: VoiceStoreError) -> Response {
    error.api_error().into_response(ApiProtocol::OpenAi)
}

async fn list_voices(
    Extension(store): Extension<VoiceStore>,
) -> Json<VoiceList<PublicVoiceProfile>> {
    Json(VoiceList {
        object: "list",
        data: store
            .list(false)
            .await
            .into_iter()
            .map(PublicVoiceProfile::from)
            .collect(),
    })
}

async fn list_all_voices(Extension(store): Extension<VoiceStore>) -> Json<VoiceList<VoiceProfile>> {
    Json(VoiceList {
        object: "list",
        data: store.list(true).await,
    })
}

async fn create_voice(
    Extension(store): Extension<VoiceStore>,
    Json(request): Json<CreateVoiceRequest>,
) -> Response {
    match store.create(request).await {
        Ok(profile) => (StatusCode::CREATED, Json(profile)).into_response(),
        Err(error) => api_error_response(error),
    }
}

async fn update_voice(
    Extension(store): Extension<VoiceStore>,
    Path(id): Path<String>,
    Json(request): Json<UpdateVoiceRequest>,
) -> Response {
    match store.update(&id, request).await {
        Ok(profile) => Json(profile).into_response(),
        Err(error) => api_error_response(error),
    }
}

async fn delete_voice(Extension(store): Extension<VoiceStore>, Path(id): Path<String>) -> Response {
    match store.delete(&id).await {
        Ok(profile) => Json(profile).into_response(),
        Err(error) => api_error_response(error),
    }
}

async fn preview_voice(
    Extension(store): Extension<VoiceStore>,
    Path(id): Path<String>,
) -> Response {
    match store.preview(&id).await {
        Ok((profile, bytes)) => {
            let mut response = Response::new(Body::from(bytes));
            response.headers_mut().insert(
                header::CONTENT_TYPE,
                HeaderValue::from_static(match profile.reference_format.as_str() {
                    "wav" => "audio/wav",
                    "flac" => "audio/flac",
                    "mp3" => "audio/mpeg",
                    _ => "application/octet-stream",
                }),
            );
            response
                .headers_mut()
                .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
            response.headers_mut().insert(
                "x-ironmlx-voice-preview",
                HeaderValue::from_static("reference"),
            );
            response
        }
        Err(error) => api_error_response(error),
    }
}

pub(crate) fn router<S>() -> Router<S>
where
    S: Clone + Send + Sync + 'static,
{
    Router::new()
        .route("/v1/audio/voices", get(list_voices).post(create_voice))
        .route("/admin/api/audio/voices", get(list_all_voices))
        .route(
            "/v1/audio/voices/:id",
            patch(update_voice).delete(delete_voice),
        )
        .route("/v1/audio/voices/:id/preview", get(preview_voice))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::{to_bytes, Body},
        http::{Request, StatusCode},
    };
    use ironmlx_audio::{io::write_wav, PcmBuffer, PcmFormat};
    use serde_json::{json, Value};
    use tower::ServiceExt;

    fn reference_wav() -> Vec<u8> {
        let mut wav = Vec::new();
        write_wav(
            &PcmBuffer {
                format: PcmFormat {
                    sample_rate: 16_000,
                    channels: 1,
                },
                samples: vec![0.0; 1_600],
            },
            &mut wav,
        )
        .expect("wav");
        wav
    }

    #[test]
    fn voice_ids_are_stable_path_safe_identifiers() {
        for valid in ["speaker_a", "zh-CN.1", "A"] {
            assert!(validate_id(valid).is_ok(), "{valid}");
        }
        for invalid in ["", "-speaker", "speaker/a", "speaker a", "声音"] {
            assert!(validate_id(invalid).is_err(), "{invalid}");
        }
    }

    #[test]
    fn reference_format_uses_audio_signatures() {
        assert_eq!(reference_format(b"RIFF\0\0\0\0WAVE"), Some("wav"));
        assert_eq!(reference_format(b"fLaC"), Some("flac"));
        assert_eq!(reference_format(b"ID3abc"), Some("mp3"));
        assert_eq!(reference_format(b"not audio"), None);
    }

    #[tokio::test]
    async fn voice_store_persists_create_update_resolve_and_delete() {
        let root = std::env::temp_dir().join(format!("ironmlx-voices-{}", uuid::Uuid::new_v4()));
        let store = VoiceStore::open(root.clone()).expect("voice store");
        let wav = reference_wav();
        let encoded = STANDARD.encode(&wav);
        let created = store
            .create(CreateVoiceRequest {
                id: "speaker_a".into(),
                name: "Speaker A".into(),
                language: Some("zh-CN".into()),
                enabled: true,
                ref_audio: encoded,
            })
            .await
            .expect("create");
        assert_eq!(created.id, "speaker_a");
        assert_eq!(created.duration_ms, 100);
        assert_eq!(store.reference_bytes("speaker_a").await.unwrap(), wav);

        let updated = store
            .update(
                "speaker_a",
                UpdateVoiceRequest {
                    name: Some("Narrator".into()),
                    language: None,
                    enabled: Some(false),
                    ref_audio: None,
                },
            )
            .await
            .expect("update");
        assert_eq!(updated.name, "Narrator");
        assert!(matches!(
            store.reference_bytes("speaker_a").await,
            Err(VoiceStoreError::Disabled(_))
        ));
        assert!(store.list(false).await.is_empty());
        assert_eq!(store.list(true).await.len(), 1);

        let reopened = VoiceStore::open(root.clone()).expect("reopen voice store");
        assert_eq!(reopened.list(true).await[0].name, "Narrator");
        let reference_file = reopened.document.read().await.voices[0]
            .reference_file
            .clone();
        std::fs::write(root.join("files").join(reference_file), b"tampered")
            .expect("tamper reference");
        assert!(matches!(
            reopened.preview("speaker_a").await,
            Err(VoiceStoreError::Resource(_))
        ));
        reopened.delete("speaker_a").await.expect("delete");
        assert!(reopened.list(true).await.is_empty());
        std::fs::remove_dir_all(root).expect("remove temp voice store");
    }

    #[tokio::test]
    async fn voice_routes_expose_discovery_management_and_reference_preview() {
        let root = std::env::temp_dir().join(format!("ironmlx-voice-api-{}", uuid::Uuid::new_v4()));
        let store = VoiceStore::open(root.clone()).expect("voice store");
        let app = router().layer(Extension(store));
        let wav = reference_wav();

        let create = app
            .clone()
            .oneshot(
                Request::post("/v1/audio/voices")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(
                        serde_json::to_vec(&json!({
                            "id": "narrator",
                            "name": "Narrator",
                            "language": "en",
                            "ref_audio": STANDARD.encode(&wav)
                        }))
                        .unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(create.status(), StatusCode::CREATED);

        let list = app
            .clone()
            .oneshot(
                Request::get("/v1/audio/voices")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(list.status(), StatusCode::OK);
        let list: Value =
            serde_json::from_slice(&to_bytes(list.into_body(), 64 * 1024).await.unwrap()).unwrap();
        assert_eq!(list["data"][0]["id"], "narrator");
        assert!(list["data"][0].get("reference_sha256").is_none());

        let preview = app
            .clone()
            .oneshot(
                Request::get("/v1/audio/voices/narrator/preview")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(preview.status(), StatusCode::OK);
        assert_eq!(preview.headers()[header::CONTENT_TYPE], "audio/wav");
        assert_eq!(
            to_bytes(preview.into_body(), 1024 * 1024).await.unwrap(),
            wav
        );

        let disable = app
            .clone()
            .oneshot(
                Request::patch("/v1/audio/voices/narrator")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(r#"{"enabled":false}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(disable.status(), StatusCode::OK);
        let list = app
            .clone()
            .oneshot(
                Request::get("/v1/audio/voices")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let list: Value =
            serde_json::from_slice(&to_bytes(list.into_body(), 64 * 1024).await.unwrap()).unwrap();
        assert!(list["data"].as_array().unwrap().is_empty());

        let delete = app
            .oneshot(
                Request::delete("/v1/audio/voices/narrator")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(delete.status(), StatusCode::OK);
        std::fs::remove_dir_all(root).expect("remove temp voice store");
    }
}
