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

use minijinja::value::{from_args, Kwargs, ValueKind};
use minijinja::{Environment, ErrorKind, Value};
use serde::Serialize;
use serde_json::ser::PrettyFormatter;

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
    special_tokens: ChatTemplateSpecialTokens,
}

#[derive(Debug, Clone, Default)]
pub struct ChatTemplateSpecialTokens {
    pub bos_token: Option<String>,
    pub eos_token: Option<String>,
    pub pad_token: Option<String>,
}

impl ChatTemplate {
    /// Compile a jinja source string. Registers a `raise_exception`
    /// helper that always returns an error (matches HF semantics).
    pub fn new(jinja_source: &str) -> Result<Self> {
        Self::new_with_special_tokens(jinja_source, ChatTemplateSpecialTokens::default())
    }

    pub fn new_with_special_tokens(
        jinja_source: &str,
        special_tokens: ChatTemplateSpecialTokens,
    ) -> Result<Self> {
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
        env.add_filter(
            "tojson",
            |value: Value, kwargs: Kwargs| -> std::result::Result<String, minijinja::Error> {
                let ensure_ascii: Option<bool> = kwargs.get("ensure_ascii")?;
                let indent: Option<usize> = kwargs.get("indent")?;
                kwargs.assert_all_used()?;
                if ensure_ascii == Some(true) {
                    return Err(minijinja::Error::new(
                        minijinja::ErrorKind::InvalidOperation,
                        "tojson ensure_ascii=true is not supported",
                    ));
                }
                match indent {
                    Some(indent) => {
                        if indent > 16 {
                            return Err(minijinja::Error::new(
                                minijinja::ErrorKind::InvalidOperation,
                                "tojson indent must be at most 16",
                            ));
                        }
                        let indent = vec![b' '; indent];
                        let formatter = PrettyFormatter::with_indent(&indent);
                        let mut output = Vec::new();
                        let mut serializer =
                            serde_json::Serializer::with_formatter(&mut output, formatter);
                        value.serialize(&mut serializer).map_err(|error| {
                            minijinja::Error::new(
                                minijinja::ErrorKind::InvalidOperation,
                                format!("tojson serialization failed: {error}"),
                            )
                        })?;
                        String::from_utf8(output).map_err(|error| {
                            minijinja::Error::new(
                                minijinja::ErrorKind::InvalidOperation,
                                format!("tojson produced invalid UTF-8: {error}"),
                            )
                        })
                    }
                    None => serde_json::to_string(&value).map_err(|error| {
                        minijinja::Error::new(
                            minijinja::ErrorKind::InvalidOperation,
                            format!("tojson serialization failed: {error}"),
                        )
                    }),
                }
            },
        );
        // Python string method shim — covers the subset used by Qwen / HF templates.
        env.set_unknown_method_callback(|state, value, method, args| {
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
                (ValueKind::Map, "items") => {
                    let _: () = from_args(args)?;
                    return state.apply_filter("items", std::slice::from_ref(value));
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
                            let (chars,): (Option<String>,) = from_args(args)?;
                            let stripped = match chars.as_deref() {
                                Some(chars) => s.trim_matches(|c| chars.contains(c)).to_owned(),
                                None => s.trim().to_owned(),
                            };
                            return Ok(Value::from(stripped));
                        }
                        "lstrip" => {
                            let (chars,): (Option<String>,) = from_args(args)?;
                            let stripped = match chars.as_deref() {
                                Some(chars) => {
                                    s.trim_start_matches(|c| chars.contains(c)).to_owned()
                                }
                                None => s.trim_start().to_owned(),
                            };
                            return Ok(Value::from(stripped));
                        }
                        "rstrip" => {
                            let (chars,): (Option<String>,) = from_args(args)?;
                            let stripped = match chars.as_deref() {
                                Some(chars) => s.trim_end_matches(|c| chars.contains(c)).to_owned(),
                                None => s.trim_end().to_owned(),
                            };
                            return Ok(Value::from(stripped));
                        }
                        "upper" => {
                            let _: () = from_args(args)?;
                            return Ok(Value::from(s.to_uppercase()));
                        }
                        "lower" => {
                            let _: () = from_args(args)?;
                            return Ok(Value::from(s.to_lowercase()));
                        }
                        "split" => {
                            let (separator, maxsplit): (Option<String>, Option<i64>) =
                                from_args(args)?;
                            let max_parts =
                                maxsplit
                                    .and_then(|n| if n >= 0 { Some(n as usize + 1) } else { None });
                            let parts: Vec<String> = match (separator.as_deref(), max_parts) {
                                (None, None) => {
                                    s.split_whitespace().map(ToOwned::to_owned).collect()
                                }
                                (None, Some(n)) => s
                                    .split_whitespace()
                                    .take(n)
                                    .map(ToOwned::to_owned)
                                    .collect(),
                                (Some(""), _) => {
                                    return Err(minijinja::Error::new(
                                        minijinja::ErrorKind::InvalidOperation,
                                        "empty separator",
                                    ));
                                }
                                (Some(separator), None) => {
                                    s.split(separator).map(ToOwned::to_owned).collect()
                                }
                                (Some(separator), Some(n)) => {
                                    s.splitn(n, separator).map(ToOwned::to_owned).collect()
                                }
                            };
                            return Ok(Value::from_serialize(parts));
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
        Ok(Self {
            env,
            special_tokens,
        })
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
        self.render_serializable(messages, add_generation_prompt, extra_kwargs)
    }

    /// Render a richer serializable message shape for tool-aware templates.
    pub fn render_serializable<M: Serialize>(
        &self,
        messages: &[M],
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
        let dst = ctx.as_object_mut().expect("ctx initialized as object");
        if let Some(token) = &self.special_tokens.bos_token {
            dst.insert(
                "bos_token".to_owned(),
                serde_json::Value::String(token.clone()),
            );
        }
        if let Some(token) = &self.special_tokens.eos_token {
            dst.insert(
                "eos_token".to_owned(),
                serde_json::Value::String(token.clone()),
            );
        }
        if let Some(token) = &self.special_tokens.pad_token {
            dst.insert(
                "pad_token".to_owned(),
                serde_json::Value::String(token.clone()),
            );
        }
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
            for (k, v) in extra_obj {
                if matches!(
                    k.as_str(),
                    "messages" | "add_generation_prompt" | "bos_token" | "eos_token" | "pad_token"
                ) {
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
    fn special_tokens_are_available_to_templates() {
        let src = r#"{{ bos_token }}{{ messages[0].content }}{{ eos_token }}"#;
        let t = ChatTemplate::new_with_special_tokens(
            src,
            ChatTemplateSpecialTokens {
                bos_token: Some("<bos>".to_owned()),
                eos_token: Some("<eos>".to_owned()),
                pad_token: Some("<pad>".to_owned()),
            },
        )
        .unwrap();
        let msgs = vec![Message {
            role: "user".into(),
            content: "hi".into(),
        }];

        assert_eq!(t.render(&msgs, false, None).unwrap(), "<bos>hi<eos>");
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
    fn glm_python_map_items_and_unicode_tojson_are_supported_exactly() {
        let src = concat!(
            "{%- for key, value in messages[0].items() -%}",
            "{{ key }}={{ value }};",
            "{%- endfor -%}|{{ messages[0] | tojson(ensure_ascii=False) }}"
        );
        let template = ChatTemplate::new(src).unwrap();
        let messages = vec![Message {
            role: "user".into(),
            content: "东京".into(),
        }];
        let output = template.render(&messages, false, None).unwrap();
        assert!(output.contains("role=user;"));
        assert!(output.contains("content=东京;"));
        assert!(output.contains("\"content\":\"东京\""));
        assert!(!output.contains("\\u"));

        let unsupported = ChatTemplate::new("{{ messages[0] | tojson(ensure_ascii=True) }}")
            .unwrap()
            .render(&messages, false, None)
            .unwrap_err();
        assert!(unsupported.to_string().contains("ensure_ascii=true"));
    }

    #[test]
    fn llama_tojson_indent_matches_hugging_face_template_contract() {
        let template = ChatTemplate::new("{{ messages[0] | tojson(indent=4) }}").unwrap();
        let messages = vec![Message {
            role: "user".into(),
            content: "东京".into(),
        }];
        let output = template.render(&messages, false, None).unwrap();
        assert_eq!(
            output,
            concat!(
                "{\n",
                "    \"content\": \"东京\",\n",
                "    \"role\": \"user\"\n",
                "}"
            )
        );

        let unsupported = ChatTemplate::new("{{ messages[0] | tojson(indent=17) }}")
            .unwrap()
            .render(&messages, false, None)
            .unwrap_err();
        assert!(unsupported.to_string().contains("at most 16"));
    }

    #[test]
    fn string_split_method_supports_gemma_strip_thinking_macro() {
        let src = r#"{%- macro strip_thinking(text) -%}
    {%- set ns = namespace(result='') -%}
    {%- for part in text.split('<channel|>') -%}
        {%- if '<|channel>' in part -%}
            {%- set ns.result = ns.result + part.split('<|channel>')[0] -%}
        {%- else -%}
            {%- set ns.result = ns.result + part -%}
        {%- endif -%}
    {%- endfor -%}
    {{- ns.result | trim -}}
{%- endmacro -%}
{{- strip_thinking(messages[0].content) -}}"#;
        let t = ChatTemplate::new(src).unwrap();
        let msgs = vec![Message {
            role: "assistant".into(),
            content: "plain assistant answer".into(),
        }];
        let out = t.render(&msgs, false, None).unwrap();
        assert_eq!(out, "plain assistant answer");
    }

    #[test]
    fn string_strip_methods_accept_python_chars_argument() {
        let src = concat!(
            "{{ messages[0].content.strip('\\nxy') }}|",
            "{{ messages[0].content.lstrip('\\nxy') }}|",
            "{{ messages[0].content.rstrip('\\nxy') }}",
        );
        let t = ChatTemplate::new(src).unwrap();
        let msgs = vec![Message {
            role: "assistant".into(),
            content: "\nyxvaluexy\n".into(),
        }];

        let out = t.render(&msgs, false, None).unwrap();

        assert_eq!(out, "value|valuexy\n|\nyxvalue");
    }

    #[test]
    fn qwen_reasoning_expression_accepts_sdk_round_trip_content() {
        let src = r#"{%- set content = messages[0].content -%}
{%- set reasoning_content = content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n') -%}
{%- set content = content.split('</think>')[-1].lstrip('\n') -%}
{{- reasoning_content + '|' + content -}}"#;
        let t = ChatTemplate::new(src).unwrap();
        let msgs = vec![Message {
            role: "assistant".into(),
            content: "Need weather data.\n</think>\n\nCalling the tool.".into(),
        }];

        let out = t.render(&msgs, false, None).unwrap();

        assert_eq!(out, "Need weather data.|Calling the tool.");
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
