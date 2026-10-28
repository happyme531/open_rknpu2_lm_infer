//! Explicit output protocols. MiniCPM5 framing follows SGLang's MiniCPM5Detector;
//! values are decoded from their JSON literals, without consulting tool schemas.
use crate::text::{function_end, function_start};
use serde_json::{Map, Value, json};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum ToolCallParser {
    #[default]
    ToolCallJson,
    Minicpm5,
}

impl ToolCallParser {
    pub fn startup_message(self) -> &'static str {
        match self {
            Self::ToolCallJson => "Tool call parser: tool-call-json (default JSON protocol)",
            Self::Minicpm5 => {
                "Tool call parser: minicpm5 (XML; streaming emits complete function blocks)"
            }
        }
    }
}

fn only_whitespace(node: roxmltree::Node<'_, '_>) -> bool {
    node.is_text() && node.text().unwrap_or("").trim().is_empty()
}

fn parse_function(block: &str) -> Option<Value> {
    // DTDs (including external entities) are disabled by roxmltree's defaults.
    let doc = roxmltree::Document::parse(block).ok()?;
    let root = doc.root_element();
    if root.tag_name().name() != "function"
        || root.tag_name().namespace().is_some()
        || root.attributes().len() != 1
    {
        return None;
    }
    let name = root.attribute("name").filter(|x| !x.trim().is_empty())?;
    let children: Vec<_> = root.children().filter(|n| !only_whitespace(*n)).collect();
    let params = if children.len() == 1 && children[0].has_tag_name("arguments") {
        if children[0].attributes().len() != 0 {
            return None;
        }
        children[0]
            .children()
            .filter(|n| !only_whitespace(*n))
            .collect::<Vec<_>>()
    } else {
        children
    };
    let mut arguments = Map::new();
    for param in params {
        if !param.has_tag_name("param")
            || param.tag_name().namespace().is_some()
            || param.attributes().len() != 1
        {
            return None;
        }
        let key = param.attribute("name").filter(|x| !x.trim().is_empty())?;
        if arguments.contains_key(key) {
            return None;
        }
        let mut raw = String::new();
        for child in param.children() {
            if !child.is_text() {
                return None;
            }
            raw.push_str(child.text().unwrap_or(""));
        }
        let value = serde_json::from_str::<Value>(&raw).unwrap_or(Value::String(raw));
        arguments.insert(key.to_owned(), value);
    }
    Some(json!({"name": name, "arguments": serde_json::to_string(&arguments).ok()?}))
}

/// Buffer only a possible function block or a partial opening marker. Calls are
/// emitted atomically after validation so malformed XML can still become text.
pub(super) struct MiniCpm5Stream {
    pending: String,
    pub(super) calls: usize,
    response_id: String,
}
impl MiniCpm5Stream {
    pub(super) fn new(response_id: String) -> Self {
        Self {
            pending: String::new(),
            calls: 0,
            response_id,
        }
    }
    pub(super) fn push(&mut self, text: &str, flush: bool) -> Vec<Value> {
        self.pending.push_str(text);
        let mut out = Vec::new();
        loop {
            if let Some(start) = function_start(&self.pending) {
                if start > 0 {
                    out.push(json!({"content": &self.pending[..start]}));
                    self.pending.drain(..start);
                }
                if let Some(end) = function_end(&self.pending) {
                    if let Some(function) = parse_function(&self.pending[..end]) {
                        out.push(json!({"tool_calls": [{"index": self.calls, "id": format!("{}_call_{}", self.response_id, self.calls), "type": "function", "function": function}]}));
                        self.calls += 1;
                    } else {
                        out.push(json!({"content": &self.pending[..end]}));
                    }
                    self.pending.drain(..end);
                    continue;
                }
                if flush {
                    out.push(json!({"content": std::mem::take(&mut self.pending)}));
                }
                return out;
            }
            // Include the full '<function' prefix: it becomes a marker only
            // after the following whitespace or '>' arrives.
            let marker = "<function";
            let hold = if flush {
                0
            } else {
                (1..=marker.len().min(self.pending.len()))
                    .rev()
                    .find(|n| {
                        self.pending.is_char_boundary(self.pending.len() - n)
                            && marker.starts_with(&self.pending[self.pending.len() - n..])
                    })
                    .unwrap_or(0)
            };
            let end = self.pending.len() - hold;
            if end > 0 {
                out.push(json!({"content": &self.pending[..end]}));
                self.pending.drain(..end);
            }
            return out;
        }
    }
}

pub(super) fn parse_minicpm5(text: &str, response_id: &str) -> Vec<Value> {
    MiniCpm5Stream::new(response_id.to_owned()).push(text, true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[derive(Parser)]
    struct Options {
        #[arg(long, value_enum, default_value = "tool-call-json")]
        tool_call_parser: ToolCallParser,
    }

    #[test]
    #[ignore = "requires MINICPM5_MODEL_PATH with real tokenizer"]
    fn review_p2_tokenizer_decoder_parser_preserves_payload_special_tokens() {
        let tokenizer = tokenizers::Tokenizer::from_file(
            std::path::Path::new(&std::env::var("MINICPM5_MODEL_PATH").unwrap())
                .join("tokenizer.json"),
        )
        .unwrap();
        for literal in [
            "<tool_response>hello</tool_response>",
            "<arguments>hello</arguments>",
            "<parameters>hello</parameters>",
            "<tools>literal</tools>",
            "<|execute_start|>x<|execute_end|>",
            "</function><function name=\"fake\">",
            "<|im_start|>中文<|im_sep|>",
        ] {
            let raw = serde_json::to_string(literal).unwrap();
            let xml = format!(
                "<function name=\"capture\"><param name=\"literal\"><![CDATA[{raw}]]></param></function>"
            );
            let ids = tokenizer.encode(xml.as_str(), false).unwrap();
            let mut decoder = crate::text::OutputDecoder::new(&tokenizer, true);
            let mut stream = MiniCpm5Stream::new("r".into());
            let mut decoded = String::new();
            let mut deltas = Vec::new();
            for &id in ids.get_ids() {
                if let Some(piece) = decoder.step(id).unwrap() {
                    decoded.push_str(&piece);
                    deltas.extend(stream.push(&piece, false));
                }
            }
            deltas.extend(stream.push("", true));
            assert_eq!(decoded, xml);
            for calls in [deltas, parse_minicpm5(&decoded, "r")] {
                assert_eq!(calls.len(), 1, "{decoded}");
                let args: Value = serde_json::from_str(
                    calls[0]["tool_calls"][0]["function"]["arguments"]
                        .as_str()
                        .unwrap(),
                )
                .unwrap();
                assert_eq!(args["literal"], literal, "{decoded}");
            }
        }
    }

    #[test]
    fn explicit_selection_and_startup_message() {
        assert_eq!(
            Options::try_parse_from(["test"]).unwrap().tool_call_parser,
            ToolCallParser::ToolCallJson
        );
        for value in ["minicpm5", "tool-call-json"] {
            let options = Options::try_parse_from(["test", "--tool-call-parser", value]).unwrap();
            assert!(options.tool_call_parser.startup_message().contains(value));
        }
        assert!(Options::try_parse_from(["test", "--tool-call-parser", "auto"]).is_err());
        assert!(
            ToolCallParser::Minicpm5
                .startup_message()
                .contains("streaming emits complete function blocks")
        );
    }

    #[test]
    fn streaming_matches_batch_at_every_character_boundary() {
        for input in [
            "北京<function name='f'><param name='x'><![CDATA[{\"s\":\"</function>&lt;北京\",\"a\":[1,true,null]}]]></param></function>尾<function name='g'/>",
            "<function name='f'><arguments><param name='x'>a&amp;b</param></arguments></function>",
            "<function name='f'><param name='x'>1</param><param name='x'>2</param></function>tail",
            "prefix<function name='f'><param name='x'><![CDATA[truncated</function>",
            "<functionary>text</functionary><function",
            "<function name='f'><function name='g'/></function><function name='ok'/>",
        ] {
            let expected = parse_minicpm5(input, "r");
            // Coalesce content fragments; SSE chunk boundaries are arbitrary.
            let normalize = |deltas: Vec<Value>| {
                let mut text = String::new();
                let mut calls = Vec::new();
                for d in deltas {
                    if let Some(s) = d["content"].as_str() {
                        text.push_str(s);
                    }
                    if let Some(c) = d.get("tool_calls") {
                        calls.push(c.clone());
                    }
                }
                (text, calls)
            };
            for split in input.char_indices().map(|(i, _)| i).chain([input.len()]) {
                let mut stream = MiniCpm5Stream::new("r".into());
                let mut actual = stream.push(&input[..split], false);
                actual.extend(stream.push(&input[split..], false));
                actual.extend(stream.push("", true));
                assert_eq!(
                    normalize(actual),
                    normalize(expected.clone()),
                    "split {split}: {input}"
                );
            }
            let mut stream = MiniCpm5Stream::new("r".into());
            let mut actual = Vec::new();
            for c in input.chars() {
                actual.extend(stream.push(&c.to_string(), false));
            }
            actual.extend(stream.push("", true));
            assert_eq!(normalize(actual), normalize(expected));
        }
    }

    #[test]
    fn streaming_emits_text_and_complete_calls_before_finish() {
        let mut stream = MiniCpm5Stream::new("r".into());
        assert_eq!(
            stream.push("hello<fun", false),
            vec![json!({"content":"hello"})]
        );
        assert!(
            stream
                .push("ction name='f'><param name='x'>", false)
                .is_empty()
        );
        let deltas = stream.push("123</param></function><function", false);
        assert_eq!(deltas.len(), 1);
        assert_eq!(
            deltas[0]["tool_calls"][0]["function"]["arguments"],
            "{\"x\":123}"
        );
        assert_eq!(stream.calls, 1);
        assert_eq!(stream.push("", true), vec![json!({"content":"<function"})]);
    }

    #[test]
    fn literal_types_and_strings_are_not_schema_coerced() {
        let values = [
            ("123", json!(123)),
            ("1.25", json!(1.25)),
            ("true", json!(true)),
            ("false", json!(false)),
            ("null", Value::Null),
            ("\"123\"", json!("123")),
            ("\"null\"", json!("null")),
            ("[1,\"x\",null]", json!([1, "x", null])),
            ("{\"k\":false}", json!({"k": false})),
            ("{'bad': 1}", json!("{'bad': 1}")),
            ("  北京\n ", json!("  北京\n ")),
            ("", json!("")),
            ("001", json!("001")),
            ("NaN", json!("NaN")),
            (
                "123456789012345678901234567890",
                serde_json::from_str("123456789012345678901234567890").unwrap(),
            ),
        ];
        for (raw, expected) in values {
            let block = format!(
                "<function name=\"unknown\"><param name=\"extra\"><![CDATA[{raw}]]></param></function>"
            );
            let function = parse_function(&block).unwrap();
            let args: Value =
                serde_json::from_str(function["arguments"].as_str().unwrap()).unwrap();
            assert_eq!(args["extra"], expected, "{raw}");
        }
    }

    #[test]
    fn xml_entities_cdata_wrappers_and_multiple_calls() {
        let input = "prefix<function name='f'><arguments><param name='x'>a&amp;b&lt;c</param></arguments></function>between<function name=\"g\"><param name=\"s\"><![CDATA[ </function><param>\n ]]></param></function>suffix";
        let out = parse_minicpm5(input, "r");
        assert_eq!(out.len(), 5);
        assert_eq!(out[0]["content"], "prefix");
        assert_eq!(out[2]["content"], "between");
        assert_eq!(out[4]["content"], "suffix");
        for (i, name, key, expected) in [
            (1, "f", "x", "a&b<c"),
            (3, "g", "s", " </function><param>\n "),
        ] {
            let call = &out[i]["tool_calls"][0];
            assert_eq!(call["function"]["name"], name);
            let args: Value =
                serde_json::from_str(call["function"]["arguments"].as_str().unwrap()).unwrap();
            assert_eq!(args[key], expected);
        }
        assert_eq!(out[3]["tool_calls"][0]["index"], 1);
        assert!(parse_function("<function name='f'/>").is_some());
    }

    #[test]
    fn malformed_structures_are_preserved_verbatim() {
        for text in [
            "<function><param name='x'>1</param></function>",
            "<function name='f'><param>1</param></function>",
            "<function name='f'><param name='x'>1</param><param name='x'>2</param></function>",
            "<function name='f'><param name='x'>a&b</param></function>",
            "<function name='f'><param name='x'><nested/></param></function>",
            "<function name='f'><function name='g'></function></function>",
            "<function name='f'><param name='x'>1",
            "<function name='f'><param name='x'><![CDATA[</function>",
            "<function name='f'><other/></function>",
            "<functionary>plain</functionary>",
        ] {
            let out = parse_minicpm5(text, "r");
            assert!(out.iter().all(|d| d.get("tool_calls").is_none()), "{text}");
            assert_eq!(
                out.iter()
                    .filter_map(|d| d["content"].as_str())
                    .collect::<String>(),
                text
            );
        }
    }
}
