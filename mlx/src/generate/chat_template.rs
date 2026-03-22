use crate::error::Result;
use serde::Deserialize;

/// A chat message with role and content.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".into(),
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".into(),
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".into(),
            content: content.into(),
        }
    }
}

/// Relevant fields from HuggingFace `tokenizer_config.json`.
#[derive(Debug, Deserialize)]
struct TokenizerConfig {
    chat_template: Option<String>,
    eos_token: Option<EosToken>,
    bos_token: Option<EosToken>,
}

/// EOS token can be a string or an object with `content` field.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum EosToken {
    Simple(String),
    Object { content: String },
}

impl EosToken {
    fn as_str(&self) -> &str {
        match self {
            EosToken::Simple(s) => s,
            EosToken::Object { content } => content,
        }
    }
}

/// Chat template engine backed by Jinja2 (minijinja).
pub struct ChatTemplate {
    template: String,
    eos_token: String,
    bos_token: String,
}

impl ChatTemplate {
    /// Load chat template from a `tokenizer_config.json` file.
    pub fn from_file(path: &str) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| crate::error::Error::Mlx(format!("failed to read {}: {}", path, e)))?;
        Self::from_json(&content)
    }

    /// Parse chat template from JSON string.
    pub fn from_json(json: &str) -> Result<Self> {
        let config: TokenizerConfig = serde_json::from_str(json)
            .map_err(|e| crate::error::Error::Mlx(format!("failed to parse config: {}", e)))?;

        let template = config.chat_template.ok_or_else(|| {
            crate::error::Error::Mlx("no chat_template found in tokenizer_config.json".into())
        })?;

        let eos_token = config
            .eos_token
            .map(|t| t.as_str().to_string())
            .unwrap_or_default();
        let bos_token = config
            .bos_token
            .map(|t| t.as_str().to_string())
            .unwrap_or_default();

        Ok(Self {
            template,
            eos_token,
            bos_token,
        })
    }

    /// Create with an explicit template string.
    pub fn new(template: String, eos_token: String, bos_token: String) -> Self {
        Self {
            template,
            eos_token,
            bos_token,
        }
    }

    /// Apply the chat template to messages, returning the formatted prompt string.
    pub fn apply(&self, messages: &[ChatMessage], add_generation_prompt: bool) -> Result<String> {
        self.apply_with_thinking(messages, add_generation_prompt, None)
    }

    /// Apply the chat template with optional enable_thinking control.
    /// When `enable_thinking` is Some(false), thinking/reasoning mode is disabled.
    pub fn apply_with_thinking(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
        enable_thinking: Option<bool>,
    ) -> Result<String> {
        let mut env = minijinja::Environment::new();

        // Preprocess: replace Python string methods with minijinja equivalents
        let processed = self
            .template
            .replace(".startswith(", " is startingwith(")
            .replace(".endswith(", " is endingwith(")
            .replace(".strip()", " | trim");
        env.add_template("chat", &processed)
            .map_err(|e| crate::error::Error::Mlx(format!("invalid chat template: {}", e)))?;

        // Add raise_exception function (used by some templates)
        env.add_function("raise_exception", raise_exception);

        let tmpl = env.get_template("chat").unwrap();

        let ctx = minijinja::context! {
            messages => messages,
            add_generation_prompt => add_generation_prompt,
            eos_token => &self.eos_token,
            bos_token => &self.bos_token,
            enable_thinking => enable_thinking.unwrap_or(false),
        };

        tmpl.render(ctx)
            .map_err(|e| crate::error::Error::Mlx(format!("chat template render failed: {}", e)))
    }
}

/// Implementation of raise_exception for Jinja2 templates.
fn raise_exception(msg: String) -> std::result::Result<String, minijinja::Error> {
    Err(minijinja::Error::new(
        minijinja::ErrorKind::InvalidOperation,
        msg,
    ))
}
