//! Complete JSON Schema semantics and optional native-grammar lowering.
//!
//! The standard validator owns meaning. Lowering preserves a faster native
//! representation when possible; its limits never redefine schema validity.

use anyhow::{anyhow, bail, Context};
use serde_json::{Map, Value};

use crate::Result;
use std::collections::{BTreeMap, VecDeque};
use std::sync::{Arc, Mutex, OnceLock};

/// Complete JSON Schema semantics, independent of a model's grammar format.
/// References resolve within the document or explicitly supplied resources.
/// Neither network nor filesystem retrieval is performed by this adapter.
#[derive(Clone, Debug)]
pub struct SchemaValidator(Arc<jsonschema::Validator>);

#[derive(Clone)]
struct ProvidedResources(Arc<BTreeMap<String, Value>>);

impl jsonschema::Retrieve for ProvidedResources {
    fn retrieve(
        &self,
        uri: &jsonschema::Uri<String>,
    ) -> std::result::Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        self.0.get(uri.as_str()).cloned().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("schema resource `{uri}` was not supplied"),
            )
            .into()
        })
    }
}

impl SchemaValidator {
    pub fn compile(schema: &Value) -> Result<Self> {
        Self::with_resources(schema, BTreeMap::new())
    }

    /// Compile with a caller-owned reference catalogue, used by offline
    /// conformance tests and applications that already possess their schemas.
    pub fn with_resources(schema: &Value, resources: BTreeMap<String, Value>) -> Result<Self> {
        Self::build(schema, resources, false)
    }

    /// Explicit format-assertion profile for callers that request it. Ordinary
    /// API schemas retain their dialect's defaults; this is not a Chat field.
    pub fn with_format_assertions(
        schema: &Value,
        resources: BTreeMap<String, Value>,
    ) -> Result<Self> {
        Self::build(schema, resources, true)
    }

    fn build(schema: &Value, resources: BTreeMap<String, Value>, formats: bool) -> Result<Self> {
        if let Some(declaration) = schema.get("$schema") {
            let uri = declaration
                .as_str()
                .ok_or_else(|| anyhow!("$schema must be an absolute URI string"))?;
            jsonschema::Uri::parse(uri.to_owned())
                .map_err(|(error, _)| anyhow!("invalid $schema URI: {error}"))?;
        }
        let mut options =
            jsonschema::options().with_retriever(ProvidedResources(Arc::new(resources)));
        if formats {
            options = options.should_validate_formats(true);
        }
        let compiled = options
            .build(schema)
            .map_err(|error| anyhow!("invalid JSON Schema: {error}"))?;
        Ok(Self(Arc::new(compiled)))
    }

    pub fn is_valid(&self, instance: &Value) -> bool {
        self.0.is_valid(instance)
    }

    pub fn validate(&self, instance: &Value) -> Result<()> {
        self.0
            .validate(instance)
            .map_err(|error| anyhow!("JSON Schema validation failed: {error}"))
    }
}

type ValidatorCache = VecDeque<(Value, SchemaValidator)>;

/// Reuse compiled document-local validators across masks, decoding and replay.
/// Resource catalogues use explicit compiled validators, never this cache.
pub fn validator(schema: &Value) -> Result<SchemaValidator> {
    static CACHE: OnceLock<Mutex<ValidatorCache>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(VecDeque::new()));
    if let Some(validator) = cache
        .lock()
        .map_err(|_| anyhow!("schema validator cache poisoned"))?
        .iter()
        .find(|(key, _)| key == schema)
        .map(|(_, validator)| validator.clone())
    {
        return Ok(validator);
    }
    let compiled = SchemaValidator::compile(schema)?;
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow!("schema validator cache poisoned"))?;
    if cache.len() >= 64 {
        cache.pop_front();
    }
    cache.push_back((schema.clone(), compiled.clone()));
    Ok(compiled)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum ProfilePosition {
    ObjectRoot,
    Value,
    Constraint,
}

// A value needs a declared shape in the strict API profile. Constraint
// fragments (for example allOf: [{minimum: 1}]) need not repeat that shape.
// References are followed separately, so a reference to {} cannot supply it.
fn declares_shape(schema: &Value, object_root: bool, draft: jsonschema::Draft) -> bool {
    if schema == &Value::Bool(false) {
        return true;
    }
    let draft = draft.detect(schema);
    if draft <= jsonschema::Draft::Draft7 && schema.get("$ref").is_some() {
        return true;
    }
    if let Some(kind) = schema.get("type") {
        return !object_root || object_type_only(kind);
    }
    if schema.get("$ref").is_some()
        || draft == jsonschema::Draft::Draft202012 && schema.get("$dynamicRef").is_some()
        || draft == jsonschema::Draft::Draft201909 && schema.get("$recursiveRef").is_some()
    {
        return true;
    }
    if !object_root && (schema.get("const").is_some() || schema.get("enum").is_some()) {
        return true;
    }
    for key in ["anyOf", "oneOf"] {
        if let Some(branches) = schema.get(key).and_then(Value::as_array) {
            return branches
                .iter()
                .all(|branch| declares_shape(branch, object_root, draft));
        }
    }
    schema
        .get("allOf")
        .and_then(Value::as_array)
        .is_some_and(|branches| {
            branches
                .iter()
                .any(|branch| declares_shape(branch, object_root, draft))
        })
}

fn object_type_only(kind: &Value) -> bool {
    kind.as_str() == Some("object")
        || kind
            .as_array()
            .is_some_and(|kinds| kinds.len() == 1 && kinds[0].as_str() == Some("object"))
}

/// Extra API policy, deliberately separate from full JSON Schema semantics.
/// Resolve actual resources with the same library as the validator rather than
/// guessing that every $ref points into the document's root $defs.
pub(super) fn validate_api_profile(schema: &Value, strict: bool, path: &str) -> Result<()> {
    let base = "https://ironmlx.invalid/api-schema";
    let registry = jsonschema::Registry::new()
        .retriever(ProvidedResources(Arc::new(BTreeMap::new())))
        .add(base, jsonschema::Resource::from_contents(schema.clone()))?
        .prepare()?;
    let draft = jsonschema::Draft::default().detect(schema);
    let resolver = registry.resolver(jsonschema::uri::from_str(base)?);
    let resolver = resolver.in_subresource(draft.create_resource_ref(schema))?;
    let mut pending = vec![(
        schema,
        resolver,
        draft,
        path.to_owned(),
        ProfilePosition::ObjectRoot,
    )];
    let mut visited = std::collections::HashSet::new();
    while let Some((node, resolver, draft, path, position)) = pending.pop() {
        if !visited.insert((node as *const Value, resolver.base_uri(), position)) {
            continue;
        }
        let object = node
            .as_object()
            .ok_or_else(|| anyhow!("API schema node at `{path}` must be an object"))?;
        if draft <= jsonschema::Draft::Draft7 {
            if let Some(reference) = object.get("$ref").and_then(Value::as_str) {
                let (target, target_resolver, target_draft) =
                    resolver.lookup(reference)?.into_inner();
                pending.push((
                    target,
                    target_resolver,
                    target_draft,
                    format!("{path}.$ref({reference})"),
                    position,
                ));
                continue;
            }
        }
        if position == ProfilePosition::ObjectRoot {
            anyhow::ensure!(
                declares_shape(node, true, draft),
                "schema root type must be `object` at `{path}` (untyped and nullable roots are not allowed)"
            );
        } else if strict && position == ProfilePosition::Value {
            anyhow::ensure!(declares_shape(node, false, draft),
                "strict value schema at `{path}` must declare a type, const, enum, reference or shaped union");
        }
        if strict {
            let includes = |kind: &str| {
                object.get("type").is_some_and(|value| {
                    value.as_str() == Some(kind)
                        || value.as_array().is_some_and(|types| {
                            types.iter().any(|value| value.as_str() == Some(kind))
                        })
                })
            };
            if includes("object") || object.contains_key("properties") {
                anyhow::ensure!(
                    object.get("additionalProperties") == Some(&Value::Bool(false)),
                    "object at `{path}` must set additionalProperties=false"
                );
                let properties = object
                    .get("properties")
                    .and_then(Value::as_object)
                    .ok_or_else(|| anyhow!("object at `{path}` must declare properties"))?;
                for name in properties.keys() {
                    anyhow::ensure!(
                        object
                            .get("required")
                            .and_then(Value::as_array)
                            .is_some_and(|required| required
                                .iter()
                                .any(|value| value.as_str() == Some(name))),
                        "property `{path}.{name}` must be listed in required"
                    );
                }
            }
            if includes("array") && !object.contains_key("const") && !object.contains_key("enum") {
                anyhow::ensure!(
                    object.contains_key("items"),
                    "strict array at `{path}` must constrain items"
                );
                if object.get("items").is_some_and(Value::is_array) {
                    anyhow::ensure!(
                        object
                            .get("additionalItems")
                            .is_some_and(|tail| tail != &Value::Bool(true)),
                        "strict tuple at `{path}` must constrain additionalItems"
                    );
                }
            }
        }
        let intrinsic_shape = object.contains_key("type")
            || object.contains_key("const")
            || object.contains_key("enum");
        // Before 2019-09 a $ref ignores its assertion siblings.
        let intrinsic_shape =
            intrinsic_shape && !(draft <= jsonschema::Draft::Draft7 && object.contains_key("$ref"));
        for keyword in ["$ref", "$dynamicRef", "$recursiveRef"] {
            if keyword == "$dynamicRef" && draft != jsonschema::Draft::Draft202012
                || keyword == "$recursiveRef" && draft != jsonschema::Draft::Draft201909
            {
                continue;
            }
            if let Some(reference) = object.get(keyword).and_then(Value::as_str) {
                let resolved = if keyword == "$recursiveRef" {
                    resolver.lookup_recursive_ref()?
                } else {
                    resolver.lookup(reference)?
                };
                let (target, target_resolver, target_draft) = resolved.into_inner();
                pending.push((
                    target,
                    target_resolver,
                    target_draft,
                    format!("{path}.{keyword}({reference})"),
                    if intrinsic_shape
                        && (position != ProfilePosition::ObjectRoot
                            || object.get("type").is_some_and(object_type_only))
                    {
                        ProfilePosition::Constraint
                    } else {
                        position
                    },
                ));
            }
        }
        // Root-only validation follows just the branches that establish its
        // type. Full schema validity was already checked by the validator.
        let own_shape = (intrinsic_shape
            && (position != ProfilePosition::ObjectRoot
                || object.get("type").is_some_and(object_type_only)))
            || object.contains_key("$ref")
            || draft == jsonschema::Draft::Draft202012 && object.contains_key("$dynamicRef")
            || draft == jsonschema::Draft::Draft201909 && object.contains_key("$recursiveRef");
        let mut children = Vec::new();
        if strict {
            for keyword in [
                "properties",
                "$defs",
                "definitions",
                "patternProperties",
                "dependentSchemas",
            ] {
                if let Some(entries) = object.get(keyword).and_then(Value::as_object) {
                    for (name, child) in entries {
                        let role = if keyword == "dependentSchemas" {
                            ProfilePosition::Constraint
                        } else {
                            ProfilePosition::Value
                        };
                        children.push((child, format!("{path}.{keyword}.{name}"), role));
                    }
                }
            }
            for keyword in [
                "items",
                "contains",
                "propertyNames",
                "additionalItems",
                "additionalProperties",
                "unevaluatedProperties",
                "unevaluatedItems",
                "if",
                "then",
                "else",
                "not",
            ] {
                if let Some(child) = object.get(keyword) {
                    let role = if matches!(
                        keyword,
                        "items"
                            | "contains"
                            | "propertyNames"
                            | "additionalItems"
                            | "additionalProperties"
                            | "unevaluatedProperties"
                            | "unevaluatedItems"
                    ) {
                        ProfilePosition::Value
                    } else {
                        ProfilePosition::Constraint
                    };
                    if child.is_object()
                        || child == &Value::Bool(true) && role == ProfilePosition::Value
                    {
                        children.push((child, format!("{path}.{keyword}"), role));
                    }
                }
            }
        }
        for keyword in ["allOf", "anyOf", "oneOf", "prefixItems", "items"] {
            if let Some(branches) = object.get(keyword).and_then(Value::as_array) {
                for (index, child) in branches.iter().enumerate() {
                    let role = if matches!(keyword, "prefixItems" | "items") {
                        ProfilePosition::Value
                    } else if !own_shape
                        && position != ProfilePosition::Constraint
                        && (keyword != "allOf"
                            || declares_shape(
                                child,
                                position == ProfilePosition::ObjectRoot,
                                draft,
                            ))
                    {
                        position
                    } else {
                        ProfilePosition::Constraint
                    };
                    if strict || role == ProfilePosition::ObjectRoot {
                        children.push((child, format!("{path}.{keyword}[{index}]"), role));
                    }
                }
            }
        }
        for (child, child_path, role) in children {
            if child == &Value::Bool(false)
                || child == &Value::Bool(true) && role == ProfilePosition::Constraint
            {
                continue;
            }
            let child_draft = draft.detect(child);
            let child_resolver = resolver.in_subresource(child_draft.create_resource_ref(child))?;
            pending.push((child, child_resolver, child_draft, child_path, role));
        }
    }
    Ok(())
}

/// JSON numbers compare by mathematical value, including inside const/enum
/// arrays and objects. Comparing serde_json's storage variants would make
/// `1` different from `1.0` and would also distinguish signed zero from zero.
pub(super) fn values_equal(left: &Value, right: &Value) -> bool {
    match (left, right) {
        (Value::Number(left), Value::Number(right)) => decimal(left) == decimal(right),
        (Value::Array(left), Value::Array(right)) => {
            left.len() == right.len()
                && left
                    .iter()
                    .zip(right)
                    .all(|(left, right)| values_equal(left, right))
        }
        (Value::Object(left), Value::Object(right)) => {
            left.len() == right.len()
                && left.iter().all(|(key, value)| {
                    right
                        .get(key)
                        .is_some_and(|other| values_equal(value, other))
                })
        }
        _ => left == right,
    }
}

// The parser has already checked JSON number syntax. Canonical decimal
// coefficient/exponent pairs avoid f64 rounding when comparing i64/u64 values
// near 2^53 or 2^64 with floating spellings. Trailing coefficient zeros belong
// in the exponent; every spelling of zero has one canonical representation.
fn decimal(number: &serde_json::Number) -> (bool, String, i64) {
    let text = number.to_string();
    let negative = text.starts_with('-');
    let unsigned = text.strip_prefix('-').unwrap_or(&text);
    let (mantissa, exponent) = unsigned
        .split_once(['e', 'E'])
        .map(|(mantissa, exponent)| {
            (
                mantissa,
                exponent.parse::<i64>().expect("JSON number exponent"),
            )
        })
        .unwrap_or((unsigned, 0));
    let fraction = mantissa
        .split_once('.')
        .map_or(0, |(_, fraction)| fraction.len());
    let digits = mantissa.replace('.', "");
    let significant = digits.trim_start_matches('0');
    if significant.is_empty() {
        return (false, "0".into(), 0);
    }
    let coefficient = significant.trim_end_matches('0');
    (
        negative,
        coefficient.to_owned(),
        exponent - fraction as i64 + (significant.len() - coefficient.len()) as i64,
    )
}

pub(super) fn normalize(schema: &Value, max_depth: usize, max_nodes: usize) -> Result<Value> {
    inspect(schema, 0, &mut 0, max_depth, max_nodes)?;
    let mut context = ContextState {
        root: schema,
        active: Vec::new(),
        visits: 0,
        max_depth,
        max_visits: max_nodes.saturating_mul(4),
    };
    context.lower(schema, 0)
}

// Check source structure as well as the expanded result. In particular, an
// unused definition must not establish another resource or change dialect.
fn inspect(
    schema: &Value,
    depth: usize,
    nodes: &mut usize,
    max_depth: usize,
    max_nodes: usize,
) -> Result<()> {
    anyhow::ensure!(
        depth <= max_depth,
        "schema nesting exceeds {max_depth} levels"
    );
    *nodes += 1;
    anyhow::ensure!(*nodes <= max_nodes, "schema exceeds {max_nodes} nodes");
    let object = schema
        .as_object()
        .ok_or_else(|| anyhow!("schema node must be an object"))?;
    for key in [
        "$id",
        "$anchor",
        "$dynamicAnchor",
        "$dynamicRef",
        "$vocabulary",
    ] {
        anyhow::ensure!(
            !object.contains_key(key),
            "unsupported JSON Schema keyword `{key}`"
        );
    }
    anyhow::ensure!(
        depth == 0 || !object.contains_key("$schema"),
        "$schema must appear at the schema root"
    );
    for key in ["$defs", "properties"] {
        if let Some(children) = object.get(key) {
            let children = children
                .as_object()
                .ok_or_else(|| anyhow!("{key} must be an object"))?;
            anyhow::ensure!(
                key != "$defs" || depth == 0,
                "$defs must appear at the schema root"
            );
            for child in children.values() {
                inspect(child, depth + 1, nodes, max_depth, max_nodes)?;
            }
        }
    }
    for key in ["items", "additionalProperties"] {
        if let Some(child) = object.get(key).filter(|value| value.is_object()) {
            inspect(child, depth + 1, nodes, max_depth, max_nodes)?;
        }
    }
    for key in ["allOf", "anyOf", "oneOf"] {
        if let Some(children) = object.get(key) {
            let children = children
                .as_array()
                .ok_or_else(|| anyhow!("{key} must be an array"))?;
            anyhow::ensure!(!children.is_empty(), "{key} must not be empty");
            for child in children {
                inspect(child, depth + 1, nodes, max_depth, max_nodes)?;
            }
        }
    }
    Ok(())
}

struct ContextState<'a> {
    root: &'a Value,
    active: Vec<String>,
    visits: usize,
    max_depth: usize,
    max_visits: usize,
}

impl ContextState<'_> {
    fn lower(&mut self, schema: &Value, depth: usize) -> Result<Value> {
        anyhow::ensure!(
            depth <= self.max_depth,
            "expanded schema nesting exceeds {} levels",
            self.max_depth
        );
        self.visits += 1;
        anyhow::ensure!(
            self.visits <= self.max_visits,
            "schema normalization exceeds {} visits",
            self.max_visits
        );
        let mut object = schema.as_object().expect("source inspected").clone();
        object.remove("$defs");
        for key in ["$comment", "deprecated", "readOnly", "writeOnly"] {
            if let Some(value) = object.get(key) {
                anyhow::ensure!(
                    if key == "$comment" {
                        value.is_string()
                    } else {
                        value.is_boolean()
                    },
                    "invalid annotation `{key}`"
                );
            }
        }
        if let Some(children) = object.get_mut("properties").and_then(Value::as_object_mut) {
            for child in children.values_mut() {
                *child = self.lower(child, depth + 1)?;
            }
        }
        for key in ["items", "additionalProperties"] {
            if let Some(child) = object.get_mut(key).filter(|value| value.is_object()) {
                *child = self.lower(child, depth + 1)?;
            }
        }
        for key in ["anyOf", "oneOf"] {
            if let Some(children) = object.get_mut(key).and_then(Value::as_array_mut) {
                for child in children {
                    *child = self.lower(child, depth + 1)?;
                }
            }
        }
        if let Some(reference) = object.remove("$ref") {
            let reference = reference
                .as_str()
                .ok_or_else(|| anyhow!("$ref must be a string"))?;
            // Draft 7 ignores assertion siblings, unlike 2019-09 and 2020-12.
            // Do not silently interpret it with the modern intersection rules.
            anyhow::ensure!(
                !self
                    .root
                    .get("$schema")
                    .and_then(Value::as_str)
                    .is_some_and(|uri| uri.contains("draft-07")),
                "$ref normalization requires Draft 2019-09 or 2020-12"
            );
            let name = definition_name(reference)?;
            anyhow::ensure!(
                !self.active.contains(&name),
                "recursive $ref `{reference}` is unsupported"
            );
            let target = self
                .root
                .get("$defs")
                .and_then(|defs| defs.get(&name))
                .ok_or_else(|| anyhow!("unresolved local $ref `{reference}`"))?;
            self.active.push(name);
            let target = self
                .lower(target, depth)
                .with_context(|| format!("resolve $ref `{reference}`"))?;
            self.active.pop();
            // A source declaration describes the whole document, not one branch.
            let declaration = object.remove("$schema");
            merge(
                &mut object,
                target.as_object().expect("lowered schema object"),
            )?;
            if let Some(declaration) = declaration {
                object.insert("$schema".into(), declaration);
            }
        }
        if let Some(branches) = object.remove("allOf") {
            for branch in branches.as_array().expect("source inspected") {
                let branch = self.lower(branch, depth + 1)?;
                merge(
                    &mut object,
                    branch.as_object().expect("lowered schema object"),
                )?;
            }
        }
        Ok(Value::Object(object))
    }
}

fn definition_name(reference: &str) -> Result<String> {
    anyhow::ensure!(
        reference.starts_with("#/") && !reference[1..].contains('#'),
        "$ref supports only local root $defs pointers"
    );
    let bytes = reference.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut offset = 0;
    while offset < bytes.len() {
        if bytes[offset] == b'%' {
            let hex = bytes
                .get(offset + 1..offset + 3)
                .ok_or_else(|| anyhow!("invalid URI escape in $ref"))?;
            let digit = |byte: u8| {
                (byte as char)
                    .to_digit(16)
                    .ok_or_else(|| anyhow!("invalid URI escape in $ref"))
            };
            decoded.push((digit(hex[0])? * 16 + digit(hex[1])?) as u8);
            offset += 3;
        } else {
            decoded.push(bytes[offset]);
            offset += 1;
        }
    }
    let decoded = String::from_utf8(decoded).context("$ref URI fragment must be UTF-8")?;
    let reference = decoded.as_str();
    let name = reference.strip_prefix("#/$defs/").ok_or_else(|| anyhow!(
        "$ref supports only local root $defs pointers; external, anchor and resource references are unsupported"
    ))?;
    anyhow::ensure!(!name.contains('/'), "$ref requires one escaped $defs name");
    let mut result = String::new();
    let mut chars = name.chars();
    while let Some(ch) = chars.next() {
        if ch != '~' {
            result.push(ch);
            continue;
        }
        result.push(match chars.next() {
            Some('0') => '~',
            Some('1') => '/',
            _ => bail!("invalid JSON Pointer escape in $ref `{reference}`"),
        });
    }
    Ok(result)
}

fn annotation(key: &str) -> bool {
    matches!(
        key,
        "$schema"
            | "$comment"
            | "description"
            | "title"
            | "default"
            | "examples"
            | "deprecated"
            | "readOnly"
            | "writeOnly"
    )
}

// Joining disjoint keys alone is not sufficient: additionalProperties applies
// to the properties in its own subschema, and native union grammars cannot
// express arbitrary intersections. Keep those cases explicit errors.
fn merge(left: &mut Map<String, Value>, right: &Map<String, Value>) -> Result<()> {
    let assertions = |object: &Map<String, Value>| object.keys().any(|key| !annotation(key));
    let unions =
        |object: &Map<String, Value>| object.contains_key("anyOf") || object.contains_key("oneOf");
    anyhow::ensure!(
        !(unions(left) && assertions(right) || unions(right) && assertions(left)),
        "schema intersection with union assertions is unsupported"
    );
    let object_scope = |object: &Map<String, Value>| {
        ["properties", "required", "additionalProperties"]
            .iter()
            .any(|key| object.contains_key(*key))
    };
    anyhow::ensure!(
        !(object_scope(left) && object_scope(right)),
        "schema intersection across object property scopes is unsupported"
    );
    for (key, value) in right {
        match left.get(key) {
            Some(existing) if values_equal(existing, value) || annotation(key) => {}
            Some(_) => bail!("schema intersection has conflicting `{key}` constraints"),
            None => {
                left.insert(key.clone(), value.clone());
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::constrained::ConstraintTokenizer;
    use sha2::{Digest, Sha256};
    use std::path::Path;

    fn resource_files(root: &Path) -> Vec<std::path::PathBuf> {
        let mut paths = Vec::new();
        for entry in std::fs::read_dir(root).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                paths.extend(resource_files(&path));
            } else if path
                .extension()
                .is_some_and(|extension| extension == "json")
            {
                paths.push(path);
            }
        }
        paths.sort();
        paths
    }

    #[test]
    fn official_draft2020_core_semantics_and_completion_guards() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/json-schema-test-suite/draft2020-12");
        let provenance: Value =
            serde_json::from_slice(&std::fs::read(root.join("provenance.json")).unwrap()).unwrap();
        for (relative, expected) in provenance["sha256"].as_object().unwrap() {
            let actual = format!(
                "{:x}",
                Sha256::digest(std::fs::read(root.join(relative)).unwrap())
            );
            assert_eq!(
                actual,
                expected.as_str().unwrap(),
                "official fixture changed: {relative}"
            );
        }
        let remote_root = root.join("remotes");
        let resources: BTreeMap<_, _> = resource_files(&remote_root)
            .into_iter()
            .map(|path| {
                let uri = format!(
                    "http://localhost:1234/{}",
                    path.strip_prefix(&remote_root).unwrap().display()
                );
                let value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
                (uri, value)
            })
            .collect();
        assert_eq!(resources.len(), 79);
        let tokenizer = ConstraintTokenizer::byte_level().unwrap();
        let json_plan = tokenizer
            .compile_json_output(&serde_json::json!({}))
            .unwrap();
        let mut groups = 0;
        let mut instances = 0;
        for path in resource_files(&root.join("core")) {
            let cases: Vec<Value> = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
            for case in cases {
                groups += 1;
                let validator = SchemaValidator::with_resources(&case["schema"], resources.clone())
                    .unwrap_or_else(|error| {
                        panic!("{} / {}: {error:#}", path.display(), case["description"])
                    });
                let plan = json_plan
                    .clone()
                    .with_json_schema_validator(validator.clone(), None);
                for test in case["tests"].as_array().unwrap() {
                    instances += 1;
                    let expected = test["valid"].as_bool().unwrap();
                    assert_eq!(
                        validator.is_valid(&test["data"]),
                        expected,
                        "validator {} / {} / {}",
                        path.display(),
                        case["description"],
                        test["description"]
                    );
                    let mut session = plan.start_session().unwrap();
                    let text = serde_json::to_string(&test["data"]).unwrap();
                    let mut allowed = true;
                    for byte in text.bytes() {
                        let mask = session.compute_mask().unwrap();
                        if !mask.is_allowed(u32::from(byte)) {
                            allowed = false;
                            break;
                        }
                        session.commit_token(u32::from(byte)).unwrap();
                    }
                    let accepted = allowed && session.is_accepting().unwrap();
                    assert_eq!(
                        accepted,
                        expected,
                        "guard {} / {} / {}: {text}",
                        path.display(),
                        case["description"],
                        test["description"]
                    );
                }
            }
        }
        assert_eq!(groups, 385);
        assert_eq!(instances, 1300);
    }
    #[test]
    fn official_draft2020_optional_profiles_and_completion_guards() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/json-schema-test-suite/draft2020-12");
        let remote_root = root.join("remotes");
        let resources: BTreeMap<_, _> = resource_files(&remote_root)
            .into_iter()
            .map(|path| {
                let uri = format!(
                    "http://localhost:1234/{}",
                    path.strip_prefix(&remote_root).unwrap().display()
                );
                (
                    uri,
                    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap(),
                )
            })
            .collect();
        let tokenizer = ConstraintTokenizer::byte_level().unwrap();
        let json_plan = tokenizer
            .compile_json_output(&serde_json::json!({}))
            .unwrap();
        let mut total = 0;
        let mut default_instances = 0;
        let optional = root.join("optional");
        for path in resource_files(&optional) {
            let format_suite = path.strip_prefix(&optional).unwrap().starts_with("format");
            let groups: Vec<Value> =
                serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
            for group in groups {
                let default =
                    SchemaValidator::with_resources(&group["schema"], resources.clone()).unwrap();
                let assertions =
                    SchemaValidator::with_format_assertions(&group["schema"], resources.clone())
                        .unwrap();
                let plan = json_plan
                    .clone()
                    .with_json_schema_validator(assertions.clone(), None);
                for test in group["tests"].as_array().unwrap() {
                    total += 1;
                    let expected = test["valid"].as_bool().unwrap();
                    assert_eq!(
                        assertions.is_valid(&test["data"]),
                        expected,
                        "format profile {} / {} / {}",
                        path.display(),
                        group["description"],
                        test["description"]
                    );
                    // optional/format requires an explicit format assertion option.
                    // Default Draft 2020-12 schemas in those files only annotate.
                    let default_expected = if format_suite { true } else { expected };
                    assert_eq!(
                        default.is_valid(&test["data"]),
                        default_expected,
                        "default profile {} / {} / {}",
                        path.display(),
                        group["description"],
                        test["description"]
                    );
                    if !format_suite {
                        default_instances += 1;
                    }
                    let mut session = plan.start_session().unwrap();
                    let mut allowed = true;
                    for byte in serde_json::to_vec(&test["data"]).unwrap() {
                        let mask = session.compute_mask().unwrap();
                        if !mask.is_allowed(u32::from(byte)) {
                            allowed = false;
                            break;
                        }
                        session.commit_token(u32::from(byte)).unwrap();
                    }
                    assert_eq!(
                        allowed && session.is_accepting().unwrap(),
                        expected,
                        "optional completion guard {} / {} / {}",
                        path.display(),
                        group["description"],
                        test["description"]
                    );
                }
            }
        }
        assert_eq!(total, 1074);
        assert!(default_instances > 0);
    }
}
