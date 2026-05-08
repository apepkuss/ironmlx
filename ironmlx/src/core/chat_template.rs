//! Chat template rendering via `minijinja`.
//!
//! HF chat templates use jinja2 syntax with a few HF-specific filters
//! (`tojson`, `raise_exception`). We register the latter as a
//! always-fail function for tolerance — most templates only call it on
//! malformed inputs.
//!
//! ## Python string method compatibility
//!
//! Several Qwen / HF templates call Python string methods (`startswith`,
//! `endswith`, `strip`, `lstrip`, `rstrip`, `split`) directly on template
//! strings. `minijinja` does not expose those methods by default; we install
//! an `unknown_method_callback` that handles the subset used in Qwen templates.

use minijinja::value::{from_args, ValueKind};
use minijinja::{Environment, ErrorKind, Value};
use serde::Serialize;

use crate::Result;

/// One conversation turn: `role` is typically `"user"` / `"assistant"` /
/// `"system"`; `content` is the literal text.
#[derive(Debug, Clone, Serialize)]
pub struct Message {
    /// Speaker role.
    pub role: String,
    /// Message body.
    pub content: String,
}

/// Compiled jinja chat template.
pub struct ChatTemplate {
    env: Environment<'static>,
}

impl ChatTemplate {
    /// Compile a jinja source string. Registers a `raise_exception`
    /// helper that always returns an error (matches HF semantics).
    pub fn new(jinja_source: &str) -> Result<Self> {
        let mut env = Environment::new();
        env.add_function(
            "raise_exception",
            |msg: String| -> std::result::Result<String, minijinja::Error> {
                Err(minijinja::Error::new(
                    minijinja::ErrorKind::InvalidOperation,
                    format!("template raised: {msg}"),
                ))
            },
        );
        // Python string method shim — covers the subset used by Qwen / HF templates.
        env.set_unknown_method_callback(|_state, value, method, args| {
            if value.kind() == ValueKind::String {
                let s = value.to_string();
                match method {
                    "startswith" => {
                        let (prefix,): (&str,) = from_args(args)?;
                        return Ok(Value::from(s.starts_with(prefix)));
                    }
                    "endswith" => {
                        let (suffix,): (&str,) = from_args(args)?;
                        return Ok(Value::from(s.ends_with(suffix)));
                    }
                    "strip" => {
                        let _: () = from_args(args)?;
                        return Ok(Value::from(s.trim().to_owned()));
                    }
                    "lstrip" => {
                        let _: () = from_args(args)?;
                        return Ok(Value::from(s.trim_start().to_owned()));
                    }
                    "rstrip" => {
                        let _: () = from_args(args)?;
                        return Ok(Value::from(s.trim_end().to_owned()));
                    }
                    "upper" => {
                        let _: () = from_args(args)?;
                        return Ok(Value::from(s.to_uppercase()));
                    }
                    "lower" => {
                        let _: () = from_args(args)?;
                        return Ok(Value::from(s.to_lowercase()));
                    }
                    _ => {}
                }
            }
            Err(minijinja::Error::new(
                ErrorKind::UnknownMethod,
                format!("{} has no method named {}", value.kind(), method),
            ))
        });
        env.add_template_owned("chat", jinja_source.to_owned())
            .map_err(|e| anyhow::anyhow!("compile chat template: {e}"))?;
        Ok(Self { env })
    }

    /// Render `messages` through the template. `add_generation_prompt`
    /// is forwarded as a top-level template variable.
    pub fn render(&self, messages: &[Message], add_generation_prompt: bool) -> Result<String> {
        let tmpl = self
            .env
            .get_template("chat")
            .map_err(|e| anyhow::anyhow!("get chat template: {e}"))?;
        let ctx = Value::from_serialize(serde_json::json!({
            "messages": messages,
            "add_generation_prompt": add_generation_prompt,
        }));
        tmpl.render(ctx)
            .map_err(|e| anyhow::anyhow!("render chat template: {e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_simple_chatml_style() {
        let src = r#"{%- for m in messages -%}
<|im_start|>{{ m.role }}
{{ m.content }}<|im_end|>
{% endfor -%}
{%- if add_generation_prompt -%}<|im_start|>assistant
{%- endif -%}"#;
        let t = ChatTemplate::new(src).unwrap();
        let msgs = vec![Message {
            role: "user".into(),
            content: "hi".into(),
        }];
        let out = t.render(&msgs, true).unwrap();
        assert!(out.contains("<|im_start|>user"));
        assert!(out.contains("hi"));
        assert!(out.contains("<|im_end|>"));
        assert!(out.ends_with("<|im_start|>assistant"));
    }

    #[test]
    fn render_without_generation_prompt() {
        let src = r#"{%- for m in messages -%}{{ m.content }}{% endfor -%}{%- if add_generation_prompt -%}<gen>{%- endif -%}"#;
        let t = ChatTemplate::new(src).unwrap();
        let msgs = vec![Message {
            role: "user".into(),
            content: "abc".into(),
        }];
        let out = t.render(&msgs, false).unwrap();
        assert_eq!(out, "abc");
    }

    #[test]
    fn raise_exception_filter_errors() {
        let src = r#"{{ raise_exception('boom') }}"#;
        let t = ChatTemplate::new(src).unwrap();
        let res = t.render(&[], false);
        assert!(res.is_err(), "expected error from raise_exception");
    }
}
