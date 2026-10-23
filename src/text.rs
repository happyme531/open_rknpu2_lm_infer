#[path = "text/output_decoder.rs"]
mod output_decoder;
#[path = "text/tool_framing.rs"]
mod tool_framing;
use crate::engine::Prompt;
use anyhow::{Context, Result, anyhow, ensure};
use minijinja::{Environment, context};
pub(crate) use output_decoder::OutputDecoder;
use serde_json::Value;
use std::path::Path;
use tokenizers::Tokenizer;
pub(crate) use tool_framing::{function_end, function_start};

pub struct TextCodec {
    pub tokenizer: Tokenizer,
    template: String,
    eos_token: Value,
    bos_token: Value,
    images: Option<crate::vision::ImageProcessor>,
}

// Compare decimal values without rounding through f64. This accepts alternate
// exponent notation while rejecting numbers a template would silently round.
fn decimal_parts(text: &str) -> Option<(bool, String, i64)> {
    let negative = text.starts_with('-');
    let unsigned = text.trim_start_matches('-');
    let (mantissa, exponent) = unsigned.split_once(['e', 'E']).unwrap_or((unsigned, "0"));
    let fraction = mantissa.split_once('.').map_or(0, |(_, tail)| tail.len());
    let mut exponent = exponent.parse::<i64>().ok()?.checked_sub(fraction as i64)?;
    let digits = mantissa.replace('.', "");
    let mut digits = digits.trim_start_matches('0').to_owned();
    if digits.is_empty() {
        return Some((false, "0".into(), 0));
    }
    while digits.ends_with('0') {
        digits.pop();
        exponent = exponent.checked_add(1)?;
    }
    Some((negative, digits, exponent))
}

/// Do not feed arbitrary_precision Number's private Serde struct to MiniJinja.
/// Native integer values retain up to 128 bits; unsupported numeric precision
/// is an explicit input error rather than a string/object or rounded number.
fn template_value(value: &Value) -> Result<minijinja::Value> {
    use minijinja::Value as J;
    Ok(match value {
        Value::Null => J::from(()),
        Value::Bool(value) => J::from(*value),
        Value::String(value) => J::from(value.clone()),
        Value::Number(number) => {
            let raw = number.to_string();
            if let Ok(integer) = raw.parse::<i128>() {
                J::from(integer)
            } else if let Ok(integer) = raw.parse::<u128>() {
                J::from(integer)
            } else {
                // Integer values outside 128 bits must not pass through f64.
                ensure!(
                    raw.contains(['.', 'e', 'E']),
                    "template integer exceeds 128-bit range: {raw}"
                );
                let float = number.as_f64().filter(|f| f.is_finite()).with_context(|| {
                    format!("template number is outside finite f64 range: {raw}")
                })?;
                let rendered = serde_json::Number::from_f64(float).unwrap().to_string();
                ensure!(
                    decimal_parts(&raw).is_some()
                        && decimal_parts(&raw) == decimal_parts(&rendered),
                    "template number cannot be represented without decimal precision loss: {raw}"
                );
                J::from(float)
            }
        }
        Value::Array(values) => values
            .iter()
            .map(template_value)
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .collect(),
        Value::Object(values) => values
            .iter()
            .map(|(key, value)| Ok((key.clone(), template_value(value)?)))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .collect(),
    })
}

// Transformers templates can pass ensure_ascii to tojson. Delegate all existing
// behavior (including indentation and HTML escaping) to MiniJinja unchanged.
fn template_tojson(
    value: &minijinja::Value,
    indent: Option<minijinja::Value>,
    args: minijinja::value::Kwargs,
) -> std::result::Result<minijinja::Value, minijinja::Error> {
    let ensure_ascii: Option<bool> = args.get("ensure_ascii")?;
    let rendered = minijinja::filters::tojson(value, indent, args)?;
    if ensure_ascii != Some(true) {
        return Ok(rendered);
    }
    let mut ascii = String::new();
    for ch in rendered.as_str().unwrap_or("").chars() {
        if ch.is_ascii() {
            ascii.push(ch);
        } else {
            use std::fmt::Write;
            for unit in ch.encode_utf16(&mut [0; 2]) {
                write!(ascii, "\\u{unit:04x}").unwrap();
            }
        }
    }
    Ok(minijinja::Value::from_safe_string(ascii))
}

impl TextCodec {
    pub fn load(path: &Path) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(path.join("tokenizer.json"))
            .map_err(|e| anyhow!("load tokenizer: {e}"))?;
        let config: Value =
            serde_json::from_slice(&std::fs::read(path.join("tokenizer_config.json"))?)?;
        let template_path = path.join("chat_template.jinja");
        let template = if template_path.exists() {
            std::fs::read_to_string(template_path)?
        } else {
            config["chat_template"]
                .as_str()
                .context("checkpoint needs chat_template.jinja or a tokenizer_config.json chat_template string")?
                .to_owned()
        };
        Ok(Self {
            tokenizer,
            template,
            eos_token: config["eos_token"].clone(),
            bos_token: config["bos_token"].clone(),
            images: crate::vision::ImageProcessor::load(path)?,
        })
    }

    pub fn format(&self, prompt: &Prompt) -> Result<String> {
        match prompt {
            Prompt::Text(text) => Ok(text.clone()),
            Prompt::Tokens(_) => anyhow::bail!("token IDs have no chat template"),
            Prompt::Chat {
                messages,
                tools,
                enable_thinking,
            } => {
                ensure!(!messages.is_empty(), "messages must not be empty");
                let mut env = Environment::new();
                env.add_filter("tojson", template_tojson);
                env.set_trim_blocks(true);
                env.set_lstrip_blocks(true);
                env.set_unknown_method_callback(
                    minijinja_contrib::pycompat::unknown_method_callback,
                );
                env.add_function(
                    "raise_exception",
                    |message: String| -> std::result::Result<String, minijinja::Error> {
                        Err(minijinja::Error::new(
                            minijinja::ErrorKind::InvalidOperation,
                            message,
                        ))
                    },
                );
                env.add_template("chat", &self.template)?;
                let tools = tools
                    .as_ref()
                    .map(template_value)
                    .transpose()?
                    .unwrap_or(minijinja::Value::UNDEFINED);
                Ok(env.get_template("chat")?.render(context! {
                    messages => template_value(&Value::Array(messages.clone()))?, tools => tools, add_generation_prompt => true,
                    enable_thinking => enable_thinking, eos_token => template_value(&self.eos_token)?, bos_token => template_value(&self.bos_token)?,
                })?)
            }
        }
    }

    pub fn encode(&self, prompt: &Prompt) -> Result<Vec<u32>> {
        if let Prompt::Tokens(tokens) = prompt {
            return Ok(tokens.clone());
        }
        let formatted = self.format(prompt)?;
        let encoding = self
            .tokenizer
            .encode(formatted, false)
            .map_err(|e| anyhow!("encode prompt: {e}"))?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn prepare(&self, prompt: &Prompt) -> Result<crate::input::PreparedInput> {
        let images = match prompt {
            Prompt::Chat { messages, .. } => {
                crate::input::message_images(messages, self.images.as_ref())?
            }
            _ => Vec::new(),
        };
        let tokens = self.encode(prompt)?;
        if let Some(processor) = &self.images {
            crate::input::PreparedInput::expand(tokens, images, processor)
        } else {
            Ok(crate::input::PreparedInput::text(tokens))
        }
    }

    pub fn decode(&self, tokens: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.tokenizer
            .decode(tokens, skip_special_tokens)
            .map_err(|e| anyhow!("decode: {e}"))
    }
}

/// Delay only suffixes that could become a stop sequence in the next chunk.
/// Complete stops are removed even when a tokenizer chunk spans the stop.
pub struct StopFilter {
    stops: Vec<String>,
    pending: String,
    stopped: bool,
}

impl StopFilter {
    pub fn new(stops: Vec<String>) -> Result<Self> {
        ensure!(
            stops.iter().all(|s| !s.is_empty()),
            "stop sequences must not be empty"
        );
        Ok(Self {
            stops,
            pending: String::new(),
            stopped: false,
        })
    }
    pub fn push(&mut self, chunk: &str) -> (String, bool) {
        if self.stopped {
            return (String::new(), true);
        }
        self.pending.push_str(chunk);
        if let Some(position) = self
            .stops
            .iter()
            .filter_map(|stop| self.pending.find(stop))
            .min()
        {
            let output = self.pending[..position].to_owned();
            self.pending.clear();
            self.stopped = true;
            return (output, true);
        }
        let mut retained = 0;
        for stop in &self.stops {
            for (length, _) in stop.char_indices().skip(1) {
                if length > retained && self.pending.ends_with(&stop[..length]) {
                    retained = length;
                }
            }
        }
        let split = self.pending.len() - retained;
        let tail = self.pending.split_off(split);
        let output = std::mem::replace(&mut self.pending, tail);
        (output, false)
    }
    pub fn finish(&mut self) -> String {
        std::mem::take(&mut self.pending)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn tojson_supports_ascii_keyword_without_changing_existing_templates() {
        let mut env = Environment::new();
        env.add_filter("tojson", template_tojson);
        let value = serde_json::json!({"text":"北京😀<&'", "items":[1,true]});
        for template in [
            "{{ v|tojson }}",
            "{{ v|tojson(indent=2) }}",
            "{{ v|tojson(2) }}",
        ] {
            let actual = env.render_str(template, context! { v => &value }).unwrap();
            let expected = Environment::new()
                .render_str(template, context! { v => &value })
                .unwrap();
            assert_eq!(actual, expected);
        }
        let unicode = env
            .render_str(
                "{{ v|tojson(ensure_ascii=False) }}",
                context! { v => &value },
            )
            .unwrap();
        assert!(unicode.contains("北京😀"));
        let ascii = env
            .render_str(
                "{{ v|tojson(indent=2, ensure_ascii=True) }}",
                context! { v => &value },
            )
            .unwrap();
        assert!(ascii.is_ascii());
        assert!(ascii.contains("\\ud83d\\ude00"));
        assert_eq!(serde_json::from_str::<Value>(&ascii).unwrap(), value);
        assert!(
            env.render_str("{{ v|tojson(unknown=True) }}", context! { v => &value })
                .is_err()
        );
    }

    #[test]
    #[ignore = "requires MINICPM5_MODEL_PATH with the real template and tokenizer"]
    fn real_minicpm5_template_renders_tools_and_history() {
        let path = std::path::PathBuf::from(
            std::env::var("MINICPM5_MODEL_PATH").expect("set MINICPM5_MODEL_PATH"),
        );
        let codec = TextCodec::load(&path).unwrap();
        let rendered = codec.format(&Prompt::Chat {
            messages: vec![
                serde_json::json!({"role":"user","content":"查询北京天气"}),
                serde_json::json!({"role":"assistant","content":"", "tool_calls":[{"id":"c0","type":"function","function":{"name":"capture", "arguments":{"city":"北京"}}}]}),
                serde_json::json!({"role":"tool", "tool_call_id":"c0", "content":"晴"}),
            ],
            tools: Some(serde_json::json!([{"type":"function", "function":{"name":"capture", "description":"查询天气", "parameters":{"type":"object"}}}])),
            enable_thinking: false,
        }).unwrap();
        assert!(rendered.contains("查询天气"));
        assert!(
            rendered.contains(
                "<function name=\"capture\"><param name=\"city\">北京</param></function>"
            )
        );
        assert!(rendered.contains("<tool_response>\n晴\n</tool_response>"));
    }

    #[test]
    fn review_p1_template_numeric_operations_and_precision_boundaries() {
        let mut codec = TextCodec {
            tokenizer: Tokenizer::new(tokenizers::models::wordlevel::WordLevel::default()),
            template: "{{ messages[0].count + 1 }}|{{ tools[0].minimum == 0 }}|{{ messages[0].fraction is number }}|{{ messages[0].label is string }}|{{ messages[0].nested|tojson }}".into(),
            eos_token: Value::Null, bos_token: Value::Null, images: None,
        };
        let messages: Vec<Value> = serde_json::from_str(r#"[{"count":123,"fraction":1.25,"label":"123","nested":[0,-3,1.25,123456789012345678901234567890]}]"#).unwrap();
        let prompt = Prompt::Chat {
            messages,
            tools: Some(serde_json::from_str(r#"[{"minimum":0}]"#).unwrap()),
            enable_thinking: false,
        };
        assert_eq!(
            codec.format(&prompt).unwrap(),
            "124|True|True|True|[0,-3,1.25,123456789012345678901234567890]"
        );
        codec.template = "{{ messages[0].n|tojson }}".into();
        for (raw, expected) in [
            (
                "340282366920938463463374607431768211455",
                "340282366920938463463374607431768211455",
            ),
            (
                "-170141183460469231731687303715884105728",
                "-170141183460469231731687303715884105728",
            ),
            ("0.000001", "1e-6"),
            ("-0.0", "-0.0"),
            ("1e+20", "1e+20"),
        ] {
            let prompt = Prompt::Chat {
                messages: vec![serde_json::from_str(&format!("{{\"n\":{raw}}}")).unwrap()],
                tools: None,
                enable_thinking: false,
            };
            let text = codec.format(&prompt).unwrap();
            assert_eq!(text, expected);
            assert!(!text.contains("private"));
        }
        for raw in [
            "340282366920938463463374607431768211456",
            "1.25000000000000000000000001",
            "1e400",
            "1e-400",
        ] {
            let prompt = Prompt::Chat {
                messages: vec![serde_json::from_str(&format!("{{\"n\":{raw}}}")).unwrap()],
                tools: None,
                enable_thinking: false,
            };
            assert!(
                codec.format(&prompt).is_err(),
                "must reject rather than round: {raw}"
            );
        }
    }

    #[test]
    #[ignore = "requires MINICPM5_MODEL_PATH with real template/tokenizer"]
    fn review_p1_real_template_keeps_numeric_schema_and_history() {
        let codec = TextCodec::load(std::path::Path::new(
            &std::env::var("MINICPM5_MODEL_PATH").unwrap(),
        ))
        .unwrap();
        let tools: Value = serde_json::from_str(r#"[{"type":"function","function":{"name":"capture","parameters":{"type":"object","properties":{"count":{"type":"integer","minimum":0,"maximum":1000,"multipleOf":2,"default":123,"enum":[123,456]}}}}}]"#).unwrap();
        let messages: Vec<Value> = serde_json::from_str(r#"[{"role":"user","content":"test"},{"role":"assistant","content":"","tool_calls":[{"id":"c0","type":"function","function":{"name":"capture","arguments":{"count":123,"negative":-3,"fraction":1.25,"big":123456789012345678901234567890,"label":"123"}}}]},{"role":"tool","tool_call_id":"c0","content":"ok"}]"#).unwrap();
        let rendered = codec
            .format(&Prompt::Chat {
                messages,
                tools: Some(tools),
                enable_thinking: false,
            })
            .unwrap();
        assert!(
            !rendered.contains("$serde_json::private::Number"),
            "{rendered}"
        );
        for literal in [
            r#""minimum":0"#,
            r#""maximum":1000"#,
            r#""multipleOf":2"#,
            r#""default":123"#,
            r#""enum":[123,456]"#,
        ] {
            assert!(rendered.contains(literal), "{rendered}");
        }
        for (name, value) in [
            ("count", "123"),
            ("negative", "-3"),
            ("fraction", "1.25"),
            ("big", "123456789012345678901234567890"),
        ] {
            assert!(
                rendered.contains(&format!("<param name=\"{name}\">{value}</param>")),
                "{rendered}"
            );
        }
    }

    use super::*;
    #[test]
    fn native_template_tools_and_whitespace_rules() {
        let codec = TextCodec {
            tokenizer: Tokenizer::new(tokenizers::models::wordlevel::WordLevel::default()),
            template: "{% if tools is not defined %}{% set tools = [] %}{% endif %}\n{% if tools is iterable and tools | length > 0 %}tools{% else %}plain{% endif %}".into(),
            eos_token: Value::Null,
            bos_token: Value::Null,
            images: None,
        };
        assert_eq!(
            codec
                .format(&Prompt::Chat {
                    messages: vec![serde_json::json!({"role":"user","content":"你好"})],
                    tools: None,
                    enable_thinking: false
                })
                .unwrap(),
            "plain"
        );
    }
    #[test]
    fn stops_across_chunks_without_leaking_prefixes() {
        let mut filter = StopFilter::new(vec!["结束".into(), "END".into()]).unwrap();
        assert_eq!(filter.push("你好结"), ("你好".into(), false));
        assert_eq!(filter.push("束剩余"), ("".into(), true));
        assert_eq!(filter.finish(), "");
    }
    #[test]
    fn partial_stop_is_flushed_at_length_limit() {
        let mut filter = StopFilter::new(vec!["END".into()]).unwrap();
        assert_eq!(filter.push("hello E"), ("hello ".into(), false));
        assert_eq!(filter.finish(), "E");
    }
}
