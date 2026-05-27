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
//! strings. Some templates also call dictionary methods such as `get`.
//! `minijinja` does not expose those methods by default; we install an
//! `unknown_method_callback` that handles the subset used by supported models.

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
            match (value.kind(), method) {
                (ValueKind::Map, "get") => {
                    let (key, default): (Value, Option<Value>) = from_args(args)?;
                    let item = value.get_item(&key)?;
                    return Ok(if item.is_undefined() {
                        default.unwrap_or(Value::UNDEFINED)
                    } else {
                        item
                    });
                }
                (ValueKind::String, method) => {
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
                _ => {}
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

    /// Render `messages` through the template. `add_generation_prompt` is
    /// forwarded as a top-level template variable. `extra_kwargs`, when
    /// present, must be a JSON object whose top-level keys are merged into
    /// the template render context (e.g. `{"enable_thinking": false}` from
    /// an OpenAI request's `chat_template_kwargs` field). Reserved keys
    /// `messages` and `add_generation_prompt` in `extra_kwargs` are
    /// ignored — the explicit args take precedence.
    pub fn render(
        &self,
        messages: &[Message],
        add_generation_prompt: bool,
        extra_kwargs: Option<&serde_json::Value>,
    ) -> Result<String> {
        let tmpl = self
            .env
            .get_template("chat")
            .map_err(|e| anyhow::anyhow!("get chat template: {e}"))?;

        let mut ctx = serde_json::json!({
            "messages": messages,
            "add_generation_prompt": add_generation_prompt,
        });
        if let Some(extra) = extra_kwargs {
            let extra_obj = extra.as_object().ok_or_else(|| {
                anyhow::anyhow!(
                    "chat_template_kwargs must be a JSON object, got {}",
                    if extra.is_array() {
                        "array"
                    } else if extra.is_string() {
                        "string"
                    } else {
                        "scalar"
                    }
                )
            })?;
            let dst = ctx.as_object_mut().expect("ctx initialized as object");
            for (k, v) in extra_obj {
                if k == "messages" || k == "add_generation_prompt" {
                    continue;
                }
                dst.insert(k.clone(), v.clone());
            }
        }
        tmpl.render(Value::from_serialize(ctx))
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
        let out = t.render(&msgs, true, None).unwrap();
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
        let out = t.render(&msgs, false, None).unwrap();
        assert_eq!(out, "abc");
    }

    #[test]
    fn raise_exception_filter_errors() {
        let src = r#"{{ raise_exception('boom') }}"#;
        let t = ChatTemplate::new(src).unwrap();
        let res = t.render(&[], false, None);
        assert!(res.is_err(), "expected error from raise_exception");
    }

    #[test]
    fn extra_kwargs_merged_into_context() {
        // Mirrors Qwen3.5's chat_template.jinja:149 idiom:
        //   {%- if enable_thinking is defined and enable_thinking is false %}
        let src = r#"{%- if enable_thinking is defined and enable_thinking is false -%}NOTHINK{%- else -%}THINK{%- endif -%}"#;
        let t = ChatTemplate::new(src).unwrap();
        // No kwargs → enable_thinking undefined → THINK branch.
        assert_eq!(t.render(&[], false, None).unwrap(), "THINK");
        // {"enable_thinking": false} → defined+false → NOTHINK branch.
        let kw = serde_json::json!({"enable_thinking": false});
        assert_eq!(t.render(&[], false, Some(&kw)).unwrap(), "NOTHINK");
        // {"enable_thinking": true} → defined+true → THINK branch (else arm).
        let kw = serde_json::json!({"enable_thinking": true});
        assert_eq!(t.render(&[], false, Some(&kw)).unwrap(), "THINK");
    }

    #[test]
    fn map_get_method_matches_python_defaults() {
        let src = r#"{{ messages[0].get('role') }}|{{ messages[0].get('missing', 'fallback') }}|{{ messages[0].get('missing') is undefined }}"#;
        let t = ChatTemplate::new(src).unwrap();
        let msgs = vec![Message {
            role: "user".into(),
            content: "hi".into(),
        }];
        let out = t.render(&msgs, false, None).unwrap();
        assert_eq!(out, "user|fallback|true");
    }

    #[test]
    fn extra_kwargs_reject_non_object() {
        let src = r#"hello"#;
        let t = ChatTemplate::new(src).unwrap();
        let kw = serde_json::json!([1, 2, 3]);
        let res = t.render(&[], false, Some(&kw));
        assert!(res.is_err(), "array kwargs must error");
        let kw = serde_json::json!("string");
        let res = t.render(&[], false, Some(&kw));
        assert!(res.is_err(), "string kwargs must error");
    }

    #[test]
    fn extra_kwargs_cannot_override_reserved_keys() {
        // Caller tries to overwrite `messages` via kwargs — must be ignored.
        let src = r#"{%- for m in messages -%}{{ m.content }}{%- endfor -%}"#;
        let t = ChatTemplate::new(src).unwrap();
        let msgs = vec![Message {
            role: "user".into(),
            content: "real".into(),
        }];
        let kw = serde_json::json!({"messages": [{"role": "user", "content": "fake"}]});
        let out = t.render(&msgs, false, Some(&kw)).unwrap();
        assert_eq!(
            out, "real",
            "explicit messages must take precedence over kwargs"
        );
    }
}
