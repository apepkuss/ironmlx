//! Observe protocol fields without retaining payload values. Deriving the shape
//! from the wire DTOs also covers Serde-buffered enums and omitted default fields.
use std::collections::BTreeMap;

use serde::{
    de::{self, DeserializeSeed, IgnoredAny, MapAccess, SeqAccess, Visitor},
    ser, Serialize,
};
use serde_json::{value::RawValue, Error};

#[derive(Debug)]
pub(crate) enum FieldShape {
    Opaque,
    Object(BTreeMap<String, FieldShape>),
    Array(Vec<FieldShape>),
}

impl FieldShape {
    pub(crate) fn set(&mut self, key: &str, shape: FieldShape) {
        if let Self::Object(fields) = self {
            fields.insert(key.to_owned(), shape);
        }
    }

    pub(crate) fn ignored_fields(&self, raw: &[u8]) -> Result<Vec<String>, Error> {
        let mut fields = Vec::new();
        let mut deserializer = serde_json::Deserializer::from_slice(raw);
        FieldVisitor {
            shape: self,
            path: String::new(),
            fields: &mut fields,
        }
        .deserialize(&mut deserializer)?;
        deserializer.end()?;
        Ok(fields)
    }
}

struct FieldVisitor<'a> {
    shape: &'a FieldShape,
    path: String,
    fields: &'a mut Vec<String>,
}

impl<'de> DeserializeSeed<'de> for FieldVisitor<'_> {
    type Value = ();
    fn deserialize<D: de::Deserializer<'de>>(self, deserializer: D) -> Result<(), D::Error> {
        match self.shape {
            FieldShape::Opaque => {
                <IgnoredAny as serde::Deserialize>::deserialize(deserializer)?;
                Ok(())
            }
            FieldShape::Object(_) => deserializer.deserialize_map(self),
            FieldShape::Array(_) => deserializer.deserialize_seq(self),
        }
    }
}

impl<'de> Visitor<'de> for FieldVisitor<'_> {
    type Value = ();
    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("the parsed request's protocol field structure")
    }
    fn visit_map<M: MapAccess<'de>>(self, mut object: M) -> Result<(), M::Error> {
        let FieldShape::Object(known) = self.shape else {
            return Err(de::Error::custom("expected protocol object"));
        };
        // Deferred Value-based protocol objects use the last value of a
        // repeated key. Inspect that same effective node, rather than replaying
        // overwritten values against the final DTO shape. Borrow raw subtrees
        // so unknown/open payloads are not decoded or copied into a Value tree.
        let mut effective = BTreeMap::new();
        while let Some((key, value)) = object.next_entry::<String, &RawValue>()? {
            effective.insert(key, value);
        }
        for (key, value) in effective {
            let escaped = key.replace('~', "~0").replace('/', "~1");
            let path = format!("{}/{escaped}", self.path);
            if let Some(shape) = known.get(&key) {
                let mut deserializer = serde_json::Deserializer::from_str(value.get());
                FieldVisitor {
                    shape,
                    path,
                    fields: self.fields,
                }
                .deserialize(&mut deserializer)
                .map_err(de::Error::custom)?;
                deserializer.end().map_err(de::Error::custom)?;
            } else {
                self.fields.push(path);
            }
        }
        Ok(())
    }
    fn visit_seq<S: SeqAccess<'de>>(self, mut array: S) -> Result<(), S::Error> {
        let FieldShape::Array(known) = self.shape else {
            return Err(de::Error::custom("expected protocol array"));
        };
        for (index, shape) in known.iter().enumerate() {
            if array
                .next_element_seed(FieldVisitor {
                    shape,
                    path: format!("{}/{index}", self.path),
                    fields: self.fields,
                })?
                .is_none()
            {
                return Err(de::Error::custom("protocol array shape changed"));
            }
        }
        if array.next_element::<IgnoredAny>()?.is_some() {
            return Err(de::Error::custom("protocol array shape changed"));
        }
        Ok(())
    }
}

pub(crate) fn shape(value: &(impl Serialize + ?Sized)) -> Result<FieldShape, Error> {
    value.serialize(ShapeSerializer)
}

struct ShapeSerializer;
struct ShapeArray(Vec<FieldShape>);
struct ShapeObject(BTreeMap<String, FieldShape>);
struct ShapeVariant(String, ShapeArray);
struct ShapeStructVariant(String, ShapeObject);
struct OpaqueMap;

macro_rules! opaque_scalar {
    ($($name:ident($ty:ty)),* $(,)?) => {
        $(fn $name(self, _: $ty) -> Result<Self::Ok, Self::Error> { Ok(FieldShape::Opaque) })*
    };
}

impl ser::Serializer for ShapeSerializer {
    type Ok = FieldShape;
    type Error = Error;
    type SerializeSeq = ShapeArray;
    type SerializeTuple = ShapeArray;
    type SerializeTupleStruct = ShapeArray;
    type SerializeTupleVariant = ShapeVariant;
    type SerializeMap = OpaqueMap;
    type SerializeStruct = ShapeObject;
    type SerializeStructVariant = ShapeStructVariant;

    opaque_scalar! {
        serialize_bool(bool), serialize_i8(i8), serialize_i16(i16), serialize_i32(i32), serialize_i64(i64), serialize_i128(i128),
        serialize_u8(u8), serialize_u16(u16), serialize_u32(u32), serialize_u64(u64), serialize_u128(u128),
        serialize_f32(f32), serialize_f64(f64), serialize_char(char), serialize_str(&str), serialize_bytes(&[u8])
    }
    fn serialize_none(self) -> Result<Self::Ok, Error> {
        Ok(FieldShape::Opaque)
    }
    fn serialize_some<T: Serialize + ?Sized>(self, value: &T) -> Result<Self::Ok, Error> {
        shape(value)
    }
    fn serialize_unit(self) -> Result<Self::Ok, Error> {
        Ok(FieldShape::Opaque)
    }
    fn serialize_unit_struct(self, _: &'static str) -> Result<Self::Ok, Error> {
        Ok(FieldShape::Opaque)
    }
    fn serialize_unit_variant(
        self,
        _: &'static str,
        _: u32,
        _: &'static str,
    ) -> Result<Self::Ok, Error> {
        Ok(FieldShape::Opaque)
    }
    fn serialize_newtype_struct<T: Serialize + ?Sized>(
        self,
        _: &'static str,
        value: &T,
    ) -> Result<Self::Ok, Error> {
        shape(value)
    }
    fn serialize_newtype_variant<T: Serialize + ?Sized>(
        self,
        _: &'static str,
        _: u32,
        variant: &'static str,
        value: &T,
    ) -> Result<Self::Ok, Error> {
        Ok(FieldShape::Object(BTreeMap::from([(
            variant.to_owned(),
            shape(value)?,
        )])))
    }
    fn serialize_seq(self, _: Option<usize>) -> Result<ShapeArray, Error> {
        Ok(ShapeArray(Vec::new()))
    }
    fn serialize_tuple(self, _: usize) -> Result<ShapeArray, Error> {
        Ok(ShapeArray(Vec::new()))
    }
    fn serialize_tuple_struct(self, _: &'static str, _: usize) -> Result<ShapeArray, Error> {
        Ok(ShapeArray(Vec::new()))
    }
    fn serialize_tuple_variant(
        self,
        _: &'static str,
        _: u32,
        variant: &'static str,
        _: usize,
    ) -> Result<ShapeVariant, Error> {
        Ok(ShapeVariant(variant.to_owned(), ShapeArray(Vec::new())))
    }
    // Dynamic maps are declared open payloads (tool input, schemas, metadata,
    // template kwargs), rather than protocol structs. Do not inspect their keys.
    fn serialize_map(self, _: Option<usize>) -> Result<OpaqueMap, Error> {
        Ok(OpaqueMap)
    }
    fn serialize_struct(self, _: &'static str, _: usize) -> Result<ShapeObject, Error> {
        Ok(ShapeObject(BTreeMap::new()))
    }
    fn serialize_struct_variant(
        self,
        _: &'static str,
        _: u32,
        variant: &'static str,
        _: usize,
    ) -> Result<ShapeStructVariant, Error> {
        Ok(ShapeStructVariant(
            variant.to_owned(),
            ShapeObject(BTreeMap::new()),
        ))
    }
}

macro_rules! array_serializer {
    ($trait:ident, $method:ident) => {
        impl ser::$trait for ShapeArray {
            type Ok = FieldShape;
            type Error = Error;
            fn $method<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), Error> {
                self.0.push(shape(value)?);
                Ok(())
            }
            fn end(self) -> Result<FieldShape, Error> {
                Ok(FieldShape::Array(self.0))
            }
        }
    };
}
array_serializer!(SerializeSeq, serialize_element);
array_serializer!(SerializeTuple, serialize_element);
array_serializer!(SerializeTupleStruct, serialize_field);

impl ser::SerializeTupleVariant for ShapeVariant {
    type Ok = FieldShape;
    type Error = Error;
    fn serialize_field<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), Error> {
        self.1 .0.push(shape(value)?);
        Ok(())
    }
    fn end(self) -> Result<FieldShape, Error> {
        Ok(FieldShape::Object(BTreeMap::from([(
            self.0,
            FieldShape::Array(self.1 .0),
        )])))
    }
}

impl ser::SerializeStruct for ShapeObject {
    type Ok = FieldShape;
    type Error = Error;
    fn serialize_field<T: Serialize + ?Sized>(
        &mut self,
        key: &'static str,
        value: &T,
    ) -> Result<(), Error> {
        self.0.insert(key.to_owned(), shape(value)?);
        Ok(())
    }
    fn skip_field(&mut self, key: &'static str) -> Result<(), Error> {
        // skip_serializing_if must not turn declared empty/null fields into
        // false unknown-field diagnostics.
        self.0.insert(key.to_owned(), FieldShape::Opaque);
        Ok(())
    }
    fn end(self) -> Result<FieldShape, Error> {
        Ok(FieldShape::Object(self.0))
    }
}

impl ser::SerializeStructVariant for ShapeStructVariant {
    type Ok = FieldShape;
    type Error = Error;
    fn serialize_field<T: Serialize + ?Sized>(
        &mut self,
        key: &'static str,
        value: &T,
    ) -> Result<(), Error> {
        self.1 .0.insert(key.to_owned(), shape(value)?);
        Ok(())
    }
    fn skip_field(&mut self, key: &'static str) -> Result<(), Error> {
        self.1 .0.insert(key.to_owned(), FieldShape::Opaque);
        Ok(())
    }
    fn end(self) -> Result<FieldShape, Error> {
        Ok(FieldShape::Object(BTreeMap::from([(
            self.0,
            FieldShape::Object(self.1 .0),
        )])))
    }
}

impl ser::SerializeMap for OpaqueMap {
    type Ok = FieldShape;
    type Error = Error;
    fn serialize_key<T: Serialize + ?Sized>(&mut self, _: &T) -> Result<(), Error> {
        Ok(())
    }
    fn serialize_value<T: Serialize + ?Sized>(&mut self, _: &T) -> Result<(), Error> {
        Ok(())
    }
    fn serialize_entry<K: Serialize + ?Sized, V: Serialize + ?Sized>(
        &mut self,
        _: &K,
        _: &V,
    ) -> Result<(), Error> {
        Ok(())
    }
    fn end(self) -> Result<FieldShape, Error> {
        Ok(FieldShape::Opaque)
    }
}

#[cfg(test)]
mod tests {
    use crate::server::{
        anthropic::MessagesRequest, api_transport::PublicApiRequest, openai::ChatRequest,
        responses::ResponsesRequest,
    };
    use serde_json::{json, Value};
    use std::collections::BTreeSet;

    fn check<T: PublicApiRequest>(input: Value, expected: &[&str]) {
        let raw = input.to_string();
        let request: T = serde_json::from_str(&raw).unwrap();
        let shape = request.diagnostic_shape().unwrap();
        assert!(!format!("{shape:?}").contains("PRIVATE_PAYLOAD"));
        let paths: BTreeSet<_> = shape
            .ignored_fields(raw.as_bytes())
            .unwrap()
            .into_iter()
            .collect();
        assert_eq!(paths, expected.iter().map(|v| (*v).to_owned()).collect());
    }

    #[test]
    fn chat_covers_buffered_content_formats_tools_and_late_tool_choice() {
        check::<ChatRequest>(
            json!({
                "messages":[
                    {"role":"user","content":[{"type":"text","text":"PRIVATE_PAYLOAD","cache_control":{}},{"type":"image_url","image_url":{"url":"PRIVATE_PAYLOAD","detail":"auto"},"image_extra":true}],"name":"PRIVATE_PAYLOAD","reasoning_content":null,"tool_calls":[],"tool_call_id":null},
                    {"role":"assistant","content":null,"tool_calls":[{"id":"call","type":"function","function":{"name":"f","arguments":"PRIVATE_PAYLOAD","extra":true},"extra":true}]}
                ],
                "tools":[{"type":"function","function":{"name":"f","parameters":{"PRIVATE_PAYLOAD":{"any_key":true}},"extra":true},"extra":true}],
                "tool_choice":{"type":"function","function":{"name":"f","extra":true},"extra":true},
                "response_format":{"type":"json_schema","json_schema":{"name":"s","description":null,"strict":null,"schema":{"PRIVATE_PAYLOAD":true},"extra":true},"extra":true},
                "stream_options":{"include_usage":true,"extra":true},
                "chat_template_kwargs":{"PRIVATE_PAYLOAD":{"arbitrary":true}},
                "root/key~name":{"PRIVATE_PAYLOAD":true}
            }),
            &[
                "/messages/0/content/0/cache_control",
                "/messages/0/content/1/image_url/detail",
                "/messages/0/content/1/image_extra",
                "/messages/0/name",
                "/messages/1/tool_calls/0/function/extra",
                "/messages/1/tool_calls/0/extra",
                "/tools/0/function/extra",
                "/tools/0/extra",
                "/tool_choice/function/extra",
                "/tool_choice/extra",
                "/response_format/json_schema/extra",
                "/response_format/extra",
                "/stream_options/extra",
                "/root~1key~0name",
            ],
        );
    }

    #[test]
    fn responses_covers_reparsed_history_namespaces_text_and_choices() {
        check::<ResponsesRequest>(
            json!({
                "input":[
                    {"role":"user","content":[{"type":"input_text","text":"PRIVATE_PAYLOAD","extra":true},{"type":"input_image","image_url":"PRIVATE_PAYLOAD","extra":true}],"extra":true},
                    {"type":"function_call","call_id":"c","name":"f","arguments":"PRIVATE_PAYLOAD","extra":true},
                    {"type":"function_call_output","call_id":"c","output":[{"type":"output_text","text":"PRIVATE_PAYLOAD","extra":true}],"extra":true},
                    {"type":"reasoning","content":[{"type":"reasoning_text","text":"PRIVATE_PAYLOAD","extra":true}],"summary":[{"type":"summary_text","text":"PRIVATE_PAYLOAD","extra":true}],"extra":true}
                ],
                "tools":[{"type":"namespace","name":"n","description":"PRIVATE_PAYLOAD","extra":true,"tools":[{"type":"function","name":"f","parameters":{"PRIVATE_PAYLOAD":true},"extra":true}]}],
                "tool_choice":{"type":"function","name":"f","extra":true},
                "text":{"format":{"type":"json_schema","name":"s","schema":{"PRIVATE_PAYLOAD":true},"extra":true},"verbosity":"low"},
                "reasoning":{"effort":"none","extra":true},
                "stream_options":{"extra":true},"metadata":{"PRIVATE_PAYLOAD":true},"client_metadata":{"PRIVATE_PAYLOAD":true}
            }),
            &[
                "/input/0/extra",
                "/input/0/content/0/extra",
                "/input/0/content/1/extra",
                "/input/1/extra",
                "/input/2/extra",
                "/input/2/output/0/extra",
                "/input/3/extra",
                "/input/3/content/0/extra",
                "/input/3/summary/0/extra",
                "/tools/0/extra",
                "/tools/0/tools/0/extra",
                "/tool_choice/extra",
                "/text/format/extra",
                "/text/verbosity",
                "/reasoning/extra",
                "/stream_options/extra",
            ],
        );
    }

    #[test]
    fn messages_covers_system_images_tool_results_thinking_and_output() {
        check::<MessagesRequest>(
            json!({
                "system":[{"type":"text","text":"PRIVATE_PAYLOAD","cache_control":{}}],
                "messages":[
                    {"role":"assistant","content":[{"type":"tool_use","id":"c","name":"f","input":{"PRIVATE_PAYLOAD":true},"extra":true}],"extra":true},
                    {"role":"user","content":[{"type":"tool_result","tool_use_id":"c","content":[{"type":"text","text":"PRIVATE_PAYLOAD","extra":true},{"type":"image","source":{"type":"base64","media_type":"image/png","data":"PRIVATE_PAYLOAD","extra":true},"extra":true}],"extra":true}]}
                ],
                "tools":[{"name":"f","input_schema":{"PRIVATE_PAYLOAD":true},"extra":true}],
                "tool_choice":{"type":"auto","extra":true},
                "thinking":{"type":"disabled","extra":true},
                "output_config":{"format":{"type":"json_schema","schema":{"PRIVATE_PAYLOAD":true},"extra":true},"extra":true}
            }),
            &[
                "/system/0/cache_control",
                "/messages/0/extra",
                "/messages/0/content/0/extra",
                "/messages/1/content/0/extra",
                "/messages/1/content/0/content/0/extra",
                "/messages/1/content/0/content/1/extra",
                "/messages/1/content/0/content/1/source/extra",
                "/tools/0/extra",
                "/tool_choice/extra",
                "/thinking/extra",
                "/output_config/extra",
                "/output_config/format/extra",
            ],
        );
    }

    fn check_raw<T: PublicApiRequest>(raw: &str, expected: &[&str]) {
        let request: T = serde_json::from_str(raw).unwrap();
        let paths: BTreeSet<_> = request
            .diagnostic_shape()
            .unwrap()
            .ignored_fields(raw.as_bytes())
            .unwrap()
            .into_iter()
            .collect();
        assert_eq!(paths, expected.iter().map(|v| (*v).to_owned()).collect());
        assert!(!format!("{paths:?}").contains("PRIVATE_PAYLOAD"));
    }

    #[test]
    fn deferred_duplicate_fields_use_the_effective_value_without_losing_siblings() {
        for discarded in [
            r#""PRIVATE_PAYLOAD""#,
            "null",
            "false",
            "42",
            "[]",
            r#"[{"PRIVATE_PAYLOAD":true}]"#,
            r#"{"type":"json_schema","name":"PRIVATE_PAYLOAD","schema":{"PRIVATE_PAYLOAD":true},"overwritten_extra":true}"#,
        ] {
            let raw = format!(
                r#"{{"input":"hi","aaa_extension":true,"text":{{"format":{discarded},"format":{{"type":"text","nested_unknown":true}}}},"zzz_extension":true}}"#
            );
            check_raw::<ResponsesRequest>(
                &raw,
                &[
                    "/aaa_extension",
                    "/text/format/nested_unknown",
                    "/zzz_extension",
                ],
            );
        }
        check_raw::<ChatRequest>(
            r#"{"messages":[],"tool_choice":{"type":"function","function":null,"function":{"name":"f","extra":true}},"client_extension":true}"#,
            &["/tool_choice/function/extra", "/client_extension"],
        );
        check_raw::<ResponsesRequest>(
            r#"{"input":[{"role":"user","content":[],"content":[{"type":"input_text","text":"hi","extra":true}]}],"client_extension":true}"#,
            &["/input/0/content/0/extra", "/client_extension"],
        );
        check_raw::<ResponsesRequest>(
            r#"{"input":[{"role":"user","content":[{"type":"input_text","text":"hi","overwritten_extra":true},{"type":"input_text","text":"bye"}],"content":[{"type":"input_text","text":"hi","extra":true}]}],"client_extension":true}"#,
            &["/input/0/content/0/extra", "/client_extension"],
        );
        check_raw::<ResponsesRequest>(
            r#"{"input":"hi","tool_choice":{"type":{},"type":"function","name":"f","extra":true},"client_extension":true}"#,
            &["/tool_choice/extra", "/client_extension"],
        );
    }

    #[test]
    fn omitted_defaults_and_all_thinking_variants_do_not_create_false_positives() {
        check::<ChatRequest>(
            json!({"messages":[{"role":"user","content":null,"reasoning_content":null,"tool_calls":[],"tool_call_id":null}],"tools":[],"stream_options":null,"response_format":{"type":"text"}}),
            &[],
        );
        for thinking in [
            json!({"type":"disabled"}),
            json!({"type":"enabled","budget_tokens":32,"display":null}),
            json!({"type":"adaptive","display":"summarized"}),
        ] {
            check::<MessagesRequest>(json!({"messages":[],"thinking":thinking}), &[]);
        }
        check::<ResponsesRequest>(
            json!({"input":"hi","text":{"format":{"type":"text"}},"tool_choice":"auto","metadata":{},"stream_options":{}}),
            &[],
        );
    }
}
