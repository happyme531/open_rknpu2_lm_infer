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
    delimiters: Option<ChatDelimiters>,
}

struct ChatDelimiters {
    user: Vec<u32>,
    tool: Vec<u32>,
}
impl ChatDelimiters {
    fn for_template(tokenizer: &Tokenizer, template: &str) -> Result<Option<Self>> {
        if !template.contains("<|im_start|>") || tokenizer.token_to_id("<|im_start|>").is_none() {
            return Ok(None);
        }
        let encode = |text: &str| -> Result<Vec<u32>> {
            Ok(tokenizer
                .encode(text, false)
                .map_err(|e| anyhow!("encode chat delimiter: {e}"))?
                .get_ids()
                .to_vec())
        };
        Ok(Some(Self {
            user: encode("<|im_start|>user")?,
            tool: encode("<|im_start|>user\n<tool_response>")?,
        }))
    }
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

/// Separators, indentation and escaping of transformers' `tojson`, which is
/// `json.dumps(x, ensure_ascii=False, indent=None, separators=None,
/// sort_keys=False)`.
struct JsonStyle {
    indent: Option<String>,
    item: String,
    key: String,
    sort_keys: bool,
    ensure_ascii: bool,
}

fn json_error(message: String) -> minijinja::Error {
    minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, message)
}

/// Python's `repr` of a float: the shortest round-trip digits, positional
/// for decimal exponents -4..16 and otherwise `d.ddde±XX`.
fn python_float(value: f64) -> String {
    if value.is_nan() {
        return "NaN".into();
    }
    if value.is_infinite() {
        return if value > 0.0 { "Infinity" } else { "-Infinity" }.into();
    }
    let sign = if value.is_sign_negative() { "-" } else { "" };
    if value == 0.0 {
        return format!("{sign}0.0");
    }
    // Serde's shortest representation uses ties-to-even, as Python does.
    let shortest = serde_json::Number::from_f64(value.abs())
        .unwrap()
        .to_string();
    let (mantissa, exponent) = shortest.split_once('e').unwrap_or((&shortest, "0"));
    let decimal = mantissa.find('.').unwrap_or(mantissa.len());
    let digits = mantissa.replace('.', "");
    let leading = digits.len() - digits.trim_start_matches('0').len();
    let exponent = exponent.parse::<i32>().unwrap() + decimal as i32 - leading as i32 - 1;
    let digits = digits[leading..].trim_end_matches('0');
    if (-4..16).contains(&exponent) {
        let point = exponent + 1;
        if point <= 0 {
            format!("{sign}0.{}{digits}", "0".repeat((-point) as usize))
        } else if point as usize >= digits.len() {
            format!(
                "{sign}{digits}{}.0",
                "0".repeat(point as usize - digits.len())
            )
        } else {
            let (whole, fraction) = digits.split_at(point as usize);
            format!("{sign}{whole}.{fraction}")
        }
    } else {
        let (first, rest) = digits.split_at(1);
        let mantissa = if rest.is_empty() {
            first.to_owned()
        } else {
            format!("{first}.{rest}")
        };
        let direction = if exponent < 0 { '-' } else { '+' };
        format!("{sign}{mantissa}e{direction}{:02}", exponent.abs())
    }
}

fn python_string(out: &mut String, text: &str, ensure_ascii: bool) {
    use std::fmt::Write;
    out.push('"');
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            ch if ch < ' ' || (ensure_ascii && !(' '..='~').contains(&ch)) => {
                for unit in ch.encode_utf16(&mut [0; 2]) {
                    write!(out, "\\u{unit:04x}").unwrap();
                }
            }
            ch => out.push(ch),
        }
    }
    out.push('"');
}

/// A map key as `json.dumps` writes it: strings unchanged, scalars converted.
fn python_key(key: &minijinja::Value) -> Result<String, minijinja::Error> {
    use minijinja::value::ValueKind;
    Ok(match key.kind() {
        ValueKind::String => key.as_str().unwrap_or_default().to_owned(),
        ValueKind::Bool => if key.is_true() { "true" } else { "false" }.into(),
        ValueKind::None | ValueKind::Undefined => "null".into(),
        ValueKind::Number if key.is_integer() => key.to_string(),
        ValueKind::Number => python_float(f64::try_from(key.clone())?),
        kind => {
            return Err(json_error(format!(
                "keys must be str, int, float, bool or None, not {kind}"
            )));
        }
    })
}

fn python_json(
    out: &mut String,
    value: &minijinja::Value,
    style: &JsonStyle,
    depth: usize,
) -> Result<(), minijinja::Error> {
    use minijinja::value::ValueKind;
    let newline = |out: &mut String, depth: usize| {
        if let Some(indent) = &style.indent {
            out.push('\n');
            out.push_str(&indent.repeat(depth));
        }
    };
    match value.kind() {
        ValueKind::Undefined | ValueKind::None => out.push_str("null"),
        ValueKind::Bool => out.push_str(if value.is_true() { "true" } else { "false" }),
        ValueKind::Number if value.is_integer() => out.push_str(&value.to_string()),
        ValueKind::Number => out.push_str(&python_float(f64::try_from(value.clone())?)),
        ValueKind::String => {
            python_string(out, value.as_str().unwrap_or_default(), style.ensure_ascii)
        }
        ValueKind::Seq | ValueKind::Iterable => {
            let items: Vec<_> = value.try_iter()?.collect();
            if items.is_empty() {
                out.push_str("[]");
                return Ok(());
            }
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push_str(&style.item);
                }
                newline(out, depth + 1);
                python_json(out, item, style, depth + 1)?;
            }
            newline(out, depth);
            out.push(']');
        }
        ValueKind::Map => {
            let mut entries = value
                .try_iter()?
                .map(|key| Ok((python_key(&key)?, value.get_item(&key)?)))
                .collect::<Result<Vec<_>, minijinja::Error>>()?;
            if style.sort_keys {
                entries.sort_by(|a, b| a.0.cmp(&b.0));
            }
            if entries.is_empty() {
                out.push_str("{}");
                return Ok(());
            }
            out.push('{');
            for (index, (key, item)) in entries.iter().enumerate() {
                if index > 0 {
                    out.push_str(&style.item);
                }
                newline(out, depth + 1);
                python_string(out, key, style.ensure_ascii);
                out.push_str(&style.key);
                python_json(out, item, style, depth + 1)?;
            }
            newline(out, depth);
            out.push('}');
        }
        kind => {
            return Err(json_error(format!(
                "Object of type {kind} is not JSON serializable"
            )));
        }
    }
    Ok(())
}

// transformers replaces Jinja's tojson, which escapes HTML characters, with
// json.dumps: ", " and ": " separators, no escaping, keys in their given
// order. Re-rendered tool-call arguments must match what the model learned
// and generated, or the prefix cache cannot be reused.
fn template_tojson(
    value: &minijinja::Value,
    indent: Option<minijinja::Value>,
    args: minijinja::value::Kwargs,
) -> std::result::Result<minijinja::Value, minijinja::Error> {
    let indent = match indent {
        Some(indent) => Some(indent),
        None => args.get("indent")?,
    };
    let indent = match indent {
        None => None,
        Some(value) if value.is_none() || value.is_undefined() => None,
        Some(value) => Some(match value.as_str() {
            Some(text) => text.to_owned(),
            None => " ".repeat(usize::try_from(i64::try_from(value)?.max(0)).unwrap_or(0)),
        }),
    };
    let separators: Option<Vec<String>> = args.get("separators")?;
    let (item, key) = match separators.as_deref() {
        None => (
            if indent.is_some() { "," } else { ", " }.to_owned(),
            ": ".to_owned(),
        ),
        Some([item, key]) => (item.clone(), key.clone()),
        Some(_) => return Err(json_error("separators must be an (item, key) pair".into())),
    };
    let style = JsonStyle {
        indent,
        item,
        key,
        sort_keys: args.get::<Option<bool>>("sort_keys")?.unwrap_or(false),
        ensure_ascii: args.get::<Option<bool>>("ensure_ascii")?.unwrap_or(false),
    };
    args.assert_all_used()?;
    let mut out = String::new();
    python_json(&mut out, value, &style, 0)?;
    Ok(minijinja::Value::from_safe_string(out))
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
        let delimiters = ChatDelimiters::for_template(&tokenizer, &template)?;
        Ok(Self {
            tokenizer,
            template,
            eos_token: config["eos_token"].clone(),
            bos_token: config["bos_token"].clone(),
            images: crate::vision::ImageProcessor::load(path)?,
            delimiters,
        })
    }

    #[cfg(test)]
    pub(crate) fn from_template(template: String) -> Self {
        Self {
            tokenizer: Tokenizer::new(tokenizers::models::wordlevel::WordLevel::default()),
            template,
            eos_token: Value::Null,
            bos_token: Value::Null,
            images: None,
            delimiters: None,
        }
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

    pub fn prepare(
        &self,
        prompt: &Prompt,
        user_boundaries: bool,
    ) -> Result<crate::input::PreparedInput> {
        let images = match prompt {
            Prompt::Chat { messages, .. } => {
                crate::input::message_images(messages, self.images.as_ref())?
            }
            _ => Vec::new(),
        };
        let tokens = self.encode(prompt)?;
        let mut user_starts = Vec::new();
        if user_boundaries
            && matches!(prompt, Prompt::Chat { .. })
            && let Some(delimiters) = &self.delimiters
        {
            // Same ChatML user/tool delimiter precedence as llama.cpp's
            // qwen3-coder parser and common_chat_msg_delimiters::split.
            for i in 0..tokens.len() {
                if tokens[i..].starts_with(&delimiters.user)
                    && !tokens[i..].starts_with(&delimiters.tool)
                {
                    user_starts.push(i);
                }
            }
        }
        let mut input = if let Some(processor) = &self.images {
            crate::input::PreparedInput::expand(tokens, images, processor)
        } else {
            Ok(crate::input::PreparedInput::text(tokens))
        }?;
        input.set_user_starts(&user_starts);
        Ok(input)
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
    fn tojson_matches_transformers_json_dumps() {
        let mut env = Environment::new();
        env.add_filter("tojson", template_tojson);
        let render = |template: &str, value: &Value| {
            env.render_str(template, context! { v => template_value(value).unwrap() })
        };
        // Expected strings are Python 3 json.dumps output for the same values.
        let value =
            serde_json::json!({"text":"北京😀<&'\u{7f}", "items":[1,true,null,1.5e-7,2.0,1e16]});
        let items = "[\n    1,\n    true,\n    null,\n    1.5e-07,\n    2.0,\n    1e+16\n  ]";
        assert_eq!(
            render("{{ v|tojson }}", &value).unwrap(),
            "{\"text\": \"北京😀<&'\u{7f}\", \"items\": [1, true, null, 1.5e-07, 2.0, 1e+16]}"
        );
        for template in ["{{ v|tojson(indent=2) }}", "{{ v|tojson(2) }}"] {
            assert_eq!(
                render(template, &value).unwrap(),
                format!("{{\n  \"text\": \"北京😀<&'\u{7f}\",\n  \"items\": {items}\n}}")
            );
        }
        assert_eq!(
            render("{{ v|tojson(indent=2, ensure_ascii=True) }}", &value).unwrap(),
            format!(
                "{{\n  \"text\": \"\\u5317\\u4eac\\ud83d\\ude00<&'\\u007f\",\n  \"items\": {items}\n}}"
            )
        );
        assert_eq!(
            render(
                "{{ v|tojson(indent='\t', sort_keys=True) }}",
                &serde_json::json!({"b":[], "a":{}})
            )
            .unwrap(),
            "{\n\t\"a\": {},\n\t\"b\": []\n}"
        );
        assert_eq!(
            render(
                "{{ v|tojson(separators=(',', ':')) }}",
                &serde_json::json!({"b":1, "a":[1,2]})
            )
            .unwrap(),
            "{\"b\":1,\"a\":[1,2]}"
        );
        assert!(render("{{ v|tojson(unknown=True) }}", &value).is_err());
        for (value, expected) in [
            (1.25, "1.25"),
            (1e-6, "1e-06"),
            (-0.0, "-0.0"),
            (1e20, "1e+20"),
            (1e-4, "0.0001"),
            (1e15, "1000000000000000.0"),
            (123.0, "123.0"),
            (-2.5e-300, "-2.5e-300"),
            (2.9802322387695312e-8, "2.9802322387695312e-08"),
            (1706789742950125.2, "1706789742950125.2"),
        ] {
            assert_eq!(python_float(value), expected);
        }
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
            eos_token: Value::Null, bos_token: Value::Null, images: None, delimiters: None,
        };
        let messages: Vec<Value> = serde_json::from_str(r#"[{"count":123,"fraction":1.25,"label":"123","nested":[0,-3,1.25,123456789012345678901234567890]}]"#).unwrap();
        let prompt = Prompt::Chat {
            messages,
            tools: Some(serde_json::from_str(r#"[{"minimum":0}]"#).unwrap()),
            enable_thinking: false,
        };
        assert_eq!(
            codec.format(&prompt).unwrap(),
            "124|True|True|True|[0, -3, 1.25, 123456789012345678901234567890]"
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
            ("0.000001", "1e-06"),
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
            r#""minimum": 0"#,
            r#""maximum": 1000"#,
            r#""multipleOf": 2"#,
            r#""default": 123"#,
            r#""enum": [123, 456]"#,
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
    fn cached_delimiters_preserve_tool_precedence_and_boundaries_are_opt_in() {
        let vocab = [
            ("[UNK]", 0),
            ("system", 1),
            ("user", 2),
            ("hello", 3),
            ("<tool_response>", 4),
            ("world", 5),
            ("<|im_start|>", 6),
        ]
        .into_iter()
        .map(|(s, n)| (s.to_owned(), n))
        .collect();
        let mut tokenizer = Tokenizer::new(
            tokenizers::models::wordlevel::WordLevel::builder()
                .vocab(vocab)
                .unk_token("[UNK]".into())
                .build()
                .unwrap(),
        );
        tokenizer.with_pre_tokenizer(Some(
            tokenizers::pre_tokenizers::whitespace::WhitespaceSplit,
        ));
        tokenizer.add_special_tokens(&[tokenizers::AddedToken::from("<|im_start|>", true)]);
        let template =
            "system\n<|im_start|>user\nhello<|im_start|>user\n<tool_response>\nworld".to_owned();
        assert!(
            ChatDelimiters::for_template(&tokenizer, "{{ messages }}")
                .unwrap()
                .is_none()
        );
        let delimiters = ChatDelimiters::for_template(&tokenizer, &template).unwrap();
        assert_eq!(delimiters.as_ref().unwrap().user, [6, 2]);
        assert_eq!(delimiters.as_ref().unwrap().tool, [6, 2, 4]);
        let codec = TextCodec {
            tokenizer,
            template,
            eos_token: Value::Null,
            bos_token: Value::Null,
            images: None,
            delimiters,
        };
        let prompt = Prompt::Chat {
            messages: vec![serde_json::json!({"role":"user","content":"hello"})],
            tools: None,
            enable_thinking: false,
        };
        let input = codec.prepare(&prompt, true).unwrap();
        assert_eq!(input.tokens, [1, 6, 2, 3, 6, 2, 4, 5]);
        assert_eq!(input.user_starts, [1]);
        let without = codec.prepare(&prompt, false).unwrap();
        assert_eq!(without.tokens, input.tokens);
        assert!(without.user_starts.is_empty());
        assert!(
            codec
                .prepare(&Prompt::Text(codec.template.clone()), true)
                .unwrap()
                .user_starts
                .is_empty()
        );
    }

    #[test]
    fn native_template_tools_and_whitespace_rules() {
        let codec = TextCodec {
            tokenizer: Tokenizer::new(tokenizers::models::wordlevel::WordLevel::default()),
            template: "{% if tools is not defined %}{% set tools = [] %}{% endif %}\n{% if tools is iterable and tools | length > 0 %}tools{% else %}plain{% endif %}".into(),
            eos_token: Value::Null,
            bos_token: Value::Null,
            images: None,
            delimiters: None,
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
