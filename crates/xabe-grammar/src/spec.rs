//! The tool surface a grammar is compiled from, read out of the JSON
//! schemas the caller offered.
//!
//! This is the port of llama.cpp's `foreach_parameter` (`common/chat.cpp`)
//! and `common_schema_info::resolves_to_string`
//! (`common/json-schema-to-grammar.cpp`). Both decide the same two things:
//! which parameters exist and which are required, and whether a parameter's
//! value is written as raw text or as JSON.

use serde_json::Value;

/// How a parameter's value is written between its `<parameter>` tags.
#[derive(Debug, Clone, PartialEq)]
pub enum ValueSpec {
    /// Raw text up to the first `\n</parameter>\n`.
    ///
    /// llama.cpp's `arg_string`: `p.ac(p.until("\n</parameter>\n") + close,
    /// "\n</parameter>\n")`. Any bytes at all, terminated at the first
    /// delimiter — the value cannot contain one.
    Text,
    /// One of a fixed set of JSON literals, from `enum` or `const`.
    Literals(Vec<String>),
    /// `true` or `false`.
    Boolean,
    /// A JSON number, integers only.
    Integer,
    /// A JSON number.
    Number,
    /// The literal `null`.
    Null,
}

/// One parameter of one tool.
#[derive(Debug, Clone, PartialEq)]
pub struct ParamSpec {
    pub name: String,
    pub required: bool,
    pub value: ValueSpec,
}

/// One tool, in the order its parameters appear in `properties` — the order
/// the template rendered them into the prompt.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolSpec {
    pub name: String,
    pub params: Vec<ParamSpec>,
}

impl ToolSpec {
    /// Read a tool out of the `{"type":"function","function":{…}}` wrapper
    /// the prompt renders, which is the shape every dialect is normalized to
    /// before it reaches here.
    pub fn from_wrapper(wrapper: &Value) -> Option<Self> {
        let function = wrapper.get("function").unwrap_or(wrapper);
        let name = function.get("name")?.as_str()?;
        if name.is_empty() {
            return None;
        }
        Some(Self {
            name: name.to_owned(),
            params: parameters(function),
        })
    }
}

/// The parameters of a function schema, mirroring llama.cpp's
/// `foreach_parameter`: `parameters.properties` in declaration order, with
/// `required` read off `parameters.required`. A schema without an object
/// `properties` has no parameters at all — not an error, just a tool that
/// takes none.
fn parameters(function: &Value) -> Vec<ParamSpec> {
    let Some(schema) = function
        .get("parameters")
        .or_else(|| function.get("input_schema"))
    else {
        return Vec::new();
    };
    let Some(properties) = schema.get("properties").and_then(Value::as_object) else {
        return Vec::new();
    };
    let required = schema
        .get("required")
        .and_then(Value::as_array)
        .map(|names| {
            names
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    properties
        .iter()
        // A name carrying the markup's own delimiters could not be written
        // into `<parameter=…>` unambiguously, so it is not offered at all.
        .filter(|(name, _)| !name.is_empty() && !name.contains(['<', '>', '\n']))
        .map(|(name, property)| ParamSpec {
            name: name.clone(),
            required: required.iter().any(|needed| needed == name),
            value: value_spec(schema, property),
        })
        .collect()
}

/// What kind of value a parameter's schema admits.
///
/// The string test is llama.cpp's `resolves_to_string`, which is deliberately
/// generous: *any* branch of the schema that could be a string makes the
/// value raw text, because that is how the model writes it. Everything else
/// is JSON, and this constrains the cases whose grammar is a handful of
/// bytes — a fixed set of literals, a boolean, a number, `null`.
///
/// A composite (`object`, `array`) or an unconstrained schema falls back to
/// [`ValueSpec::Text`]: the structure around the value is still enforced, the
/// bytes inside it are not. This is where the port stops short of llama.cpp,
/// which runs the parameter's full schema through `json_schema_to_grammar`.
fn value_spec(root: &Value, schema: &Value) -> ValueSpec {
    if resolves_to_string(root, schema, &mut Vec::new()) {
        return ValueSpec::Text;
    }
    if let Some(literals) = literal_set(schema) {
        return ValueSpec::Literals(literals);
    }
    match schema_type(schema) {
        Some("boolean") => ValueSpec::Boolean,
        Some("integer") => ValueSpec::Integer,
        Some("number") => ValueSpec::Number,
        Some("null") => ValueSpec::Null,
        _ => ValueSpec::Text,
    }
}

/// The single `type` a schema declares, if it declares exactly one.
fn schema_type(schema: &Value) -> Option<&str> {
    match schema.get("type") {
        Some(Value::String(kind)) => Some(kind.as_str()),
        Some(Value::Array(kinds)) if kinds.len() == 1 => kinds[0].as_str(),
        _ => None,
    }
}

/// The JSON spellings of a `const` or `enum`, when every alternative is a
/// value the grammar can write out verbatim.
fn literal_set(schema: &Value) -> Option<Vec<String>> {
    let values = match (schema.get("const"), schema.get("enum")) {
        (Some(one), _) => vec![one.clone()],
        (None, Some(Value::Array(many))) if !many.is_empty() => many.clone(),
        _ => return None,
    };
    let mut literals = Vec::with_capacity(values.len());
    for value in values {
        // A value spanning a line would collide with the `\n</parameter>\n`
        // that closes it; none of the scalar spellings do, but a nested
        // object's could, so anything non-scalar drops the whole set.
        if value.is_object() || value.is_array() {
            return None;
        }
        literals.push(serde_json::to_string(&value).ok()?);
    }
    literals.sort();
    literals.dedup();
    Some(literals)
}

/// Follow a local `#/…` JSON pointer from the schema it is written in.
///
/// llama.cpp resolves references against the whole schema before asking any
/// question of it (`common_schema_info::resolve_refs`); the same references
/// are resolved here, against the tool's own parameter schema. A reference
/// that points outside it — to a document this server never sees — resolves
/// to nothing, which is what llama.cpp does with one too.
fn resolve<'a>(root: &'a Value, reference: &str) -> Option<&'a Value> {
    let mut node = root;
    for segment in reference.strip_prefix("#/")?.split('/') {
        let segment = segment.replace("~1", "/").replace("~0", "~");
        node = node.get(&segment)?;
    }
    Some(node)
}

/// Whether any branch of the schema can be a string.
///
/// A direct port of `common_schema_info::resolves_to_string`, which is
/// deliberately generous — one string branch anywhere makes the whole value
/// raw text, because that is how the model writes it.
fn resolves_to_string(root: &Value, schema: &Value, seen: &mut Vec<String>) -> bool {
    let Some(object) = schema.as_object() else {
        return false;
    };
    if let Some(reference) = object.get("$ref").and_then(Value::as_str) {
        // A cycle is not a string, and neither is a dangling reference.
        if seen.iter().any(|been| been == reference) {
            return false;
        }
        seen.push(reference.to_owned());
        return resolve(root, reference)
            .is_some_and(|target| resolves_to_string(root, target, seen));
    }
    match object.get("type") {
        Some(Value::String(kind)) if kind == "string" => return true,
        Some(Value::Array(kinds)) if kinds.iter().any(|kind| kind == "string") => return true,
        _ => {}
    }
    for key in ["oneOf", "anyOf"] {
        if let Some(alternatives) = object.get(key).and_then(Value::as_array)
            && alternatives
                .iter()
                .any(|alternative| resolves_to_string(root, alternative, seen))
        {
            return true;
        }
    }
    if let Some(components) = object.get("allOf").and_then(Value::as_array)
        && !components.is_empty()
        && components
            .iter()
            .all(|component| resolves_to_string(root, component, seen))
    {
        return true;
    }
    if object.get("const").is_some_and(Value::is_string) {
        return true;
    }
    if let Some(values) = object.get("enum").and_then(Value::as_array)
        && values.iter().any(Value::is_string)
    {
        return true;
    }
    if ["pattern", "minLength", "maxLength"]
        .iter()
        .any(|key| object.contains_key(*key))
    {
        return true;
    }
    matches!(
        object.get("format").and_then(Value::as_str),
        Some(
            "date" | "time" | "date-time" | "uri" | "email" | "hostname" | "ipv4" | "ipv6" | "uuid"
        )
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn spec(function: Value) -> ToolSpec {
        ToolSpec::from_wrapper(&json!({ "type": "function", "function": function }))
            .expect("a named function parses")
    }

    #[test]
    fn properties_keep_their_declared_order_and_required_flags() {
        let tool = spec(json!({
            "name": "read",
            "parameters": {
                "type": "object",
                "properties": {
                    "filePath": { "type": "string" },
                    "offset": { "type": "number" },
                    "limit": { "type": "number" },
                },
                "required": ["filePath"],
            },
        }));
        assert_eq!(tool.name, "read");
        let names: Vec<&str> = tool.params.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(names, ["filePath", "offset", "limit"]);
        assert_eq!(
            tool.params.iter().map(|p| p.required).collect::<Vec<_>>(),
            [true, false, false]
        );
    }

    #[test]
    fn a_string_by_any_branch_is_raw_text() {
        // Every one of these is a string to llama.cpp's `resolves_to_string`,
        // so the model writes the value verbatim rather than JSON-quoted.
        for schema in [
            json!({ "type": "string" }),
            json!({ "type": ["string", "null"] }),
            json!({ "anyOf": [{ "type": "integer" }, { "type": "string" }] }),
            json!({ "const": "view" }),
            json!({ "enum": ["view", "create"] }),
            json!({ "pattern": "^a+$" }),
            json!({ "format": "uri" }),
        ] {
            assert_eq!(value_spec(&json!({}), &schema), ValueSpec::Text, "{schema}");
        }
    }

    #[test]
    fn scalar_schemas_take_their_own_shapes_and_composites_fall_back_to_text() {
        assert_eq!(
            value_spec(&json!({}), &json!({ "type": "boolean" })),
            ValueSpec::Boolean
        );
        assert_eq!(
            value_spec(&json!({}), &json!({ "type": "integer" })),
            ValueSpec::Integer
        );
        assert_eq!(
            value_spec(&json!({}), &json!({ "type": "number" })),
            ValueSpec::Number
        );
        assert_eq!(
            value_spec(&json!({}), &json!({ "type": "null" })),
            ValueSpec::Null
        );
        assert_eq!(
            value_spec(&json!({}), &json!({ "enum": [1, 2, 3] })),
            ValueSpec::Literals(vec!["1".to_owned(), "2".to_owned(), "3".to_owned()])
        );
        // Composite and unconstrained values keep their structure enforced
        // and their bytes free.
        assert_eq!(
            value_spec(&json!({}), &json!({ "type": "object" })),
            ValueSpec::Text
        );
        assert_eq!(
            value_spec(&json!({}), &json!({ "type": "array" })),
            ValueSpec::Text
        );
        assert_eq!(
            value_spec(&json!({}), &json!({ "description": "?" })),
            ValueSpec::Text
        );
    }

    #[test]
    fn a_tool_without_properties_takes_no_parameters() {
        assert!(spec(json!({ "name": "now" })).params.is_empty());
        assert!(
            spec(json!({ "name": "now", "parameters": { "type": "object" } }))
                .params
                .is_empty()
        );
    }
}
