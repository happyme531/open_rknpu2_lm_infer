//! OpenAI request validation and content normalization.
use super::*;

pub(super) struct ParsedRequest {
    pub(super) request: GenerationRequest,
    pub(super) stream: bool,
    pub(super) include_usage: bool,
    pub(super) tools_enabled: bool,
}
pub(super) fn function_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 64
        && name
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || c == b'_' || c == b'-')
}
fn parse_tools(v: &Value, chat: bool) -> Result<Option<Value>, ApiError> {
    let tools = match v.get("tools").filter(|x| !x.is_null()) {
        None => None,
        Some(x) => {
            if !chat {
                return Err(ApiError::invalid("tools require chat completions", "tools"));
            }
            let array = x
                .as_array()
                .ok_or_else(|| ApiError::invalid("tools must be an array", "tools"))?;
            for tool in array {
                if tool["type"] != "function" || !tool["function"].is_object() {
                    return Err(ApiError::invalid(
                        "each tool must have type=function and a function object",
                        "tools",
                    ));
                }
                let function = &tool["function"];
                if !function["name"].as_str().is_some_and(function_name) {
                    return Err(ApiError::invalid(
                        "function name must contain 1..64 letters, digits, underscores or hyphens",
                        "tools",
                    ));
                }
                if function.get("parameters").is_some_and(|x| !x.is_object()) {
                    return Err(ApiError::invalid(
                        "function parameters must be a JSON Schema object",
                        "tools",
                    ));
                }
                if function
                    .get("description")
                    .is_some_and(|x| !x.is_null() && !x.is_string())
                {
                    return Err(ApiError::invalid(
                        "function description must be a string",
                        "tools",
                    ));
                }
                if function
                    .get("strict")
                    .is_some_and(|x| !x.is_null() && x.as_bool() != Some(false))
                {
                    return Err(ApiError::invalid(
                        "strict tool schemas are not supported",
                        "tools",
                    ));
                }
            }
            if array.is_empty() {
                None
            } else {
                Some(x.clone())
            }
        }
    };
    match v.get("tool_choice").filter(|x| !x.is_null()) {
        None => Ok(tools),
        Some(x) if x.as_str() == Some("auto") => Ok(tools),
        Some(x) if x.as_str() == Some("none") => Ok(None),
        _ => Err(ApiError::invalid(
            "supported tool_choice values are auto and none",
            "tool_choice",
        )),
    }
}
pub(super) fn normalize_messages(messages: &[Value]) -> Result<Vec<Value>, ApiError> {
    let mut normalized = Vec::with_capacity(messages.len());
    for message in messages {
        let role = message["role"]
            .as_str()
            .ok_or_else(|| ApiError::invalid("every message needs a role", "messages"))?;
        if !matches!(role, "system" | "developer" | "user" | "assistant" | "tool") {
            return Err(ApiError::invalid(
                "supported roles are system, developer, user, assistant and tool",
                "messages",
            ));
        }
        let role = if role == "developer" { "system" } else { role };
        if message.get("function_call").is_some_and(|x| !x.is_null()) {
            return Err(ApiError::invalid(
                "legacy function_call is not supported",
                "messages",
            ));
        }
        let mut calls = vec![];
        if let Some(value) = message.get("tool_calls").filter(|x| !x.is_null()) {
            if role != "assistant" {
                return Err(ApiError::invalid(
                    "tool_calls require an assistant message",
                    "messages",
                ));
            }
            let array = value
                .as_array()
                .ok_or_else(|| ApiError::invalid("tool_calls must be an array", "messages"))?;
            for call in array {
                let id = call["id"]
                    .as_str()
                    .filter(|x| !x.is_empty())
                    .ok_or_else(|| ApiError::invalid("tool call id is required", "messages"))?;
                if call["type"] != "function"
                    || !call["function"]["name"].as_str().is_some_and(function_name)
                {
                    return Err(ApiError::invalid("invalid function tool call", "messages"));
                }
                let arguments = match &call["function"]["arguments"] {
                    Value::String(s) => serde_json::from_str::<Value>(s).map_err(|_| {
                        ApiError::invalid("tool arguments must contain valid JSON", "messages")
                    })?,
                    x => x.clone(),
                };
                if !arguments.is_object() {
                    return Err(ApiError::invalid(
                        "tool arguments must be a JSON object",
                        "messages",
                    ));
                }
                calls.push(json!({"id":id,"type":"function","function":{"name":call["function"]["name"],"arguments":arguments}}));
            }
        }
        let content = match message.get("content") {
            Some(Value::String(s)) => json!(s),
            Some(Value::Array(parts)) => {
                let mut text = String::new();
                let mut has_image = false;
                for part in parts {
                    match part["type"].as_str() {
                        Some("text") => text.push_str(part["text"].as_str().ok_or_else(|| {
                            ApiError::invalid("text parts need text", "messages")
                        })?),
                        Some("image_url") => {
                            if role != "user" {
                                return Err(ApiError::invalid(
                                    "images are supported in user messages only",
                                    "messages",
                                ));
                            }
                            let url = part["image_url"]["url"].as_str().ok_or_else(|| {
                                ApiError::invalid("image_url needs a URL string", "messages")
                            })?;
                            if !url.starts_with("data:image/png;base64,")
                                && !url.starts_with("data:image/jpeg;base64,")
                            {
                                return Err(ApiError::invalid(
                                    "image_url requires an inline PNG/JPEG base64 data URL; remote URLs are unsupported",
                                    "messages",
                                ));
                            }
                            if part["image_url"].get("detail").is_some_and(|v| v != "auto") {
                                return Err(ApiError::invalid(
                                    "only image detail=auto is supported",
                                    "messages",
                                ));
                            }
                            has_image = true;
                        }
                        _ => {
                            return Err(ApiError::invalid(
                                "supported content part types are text and image_url",
                                "messages",
                            ));
                        }
                    }
                }
                if has_image { json!(parts) } else { json!(text) }
            }
            None | Some(Value::Null) if role == "assistant" && !calls.is_empty() => json!(""),
            _ => {
                return Err(ApiError::invalid(
                    "message content must be text or content parts",
                    "messages",
                ));
            }
        };
        let mut result = json!({"role":role,"content":content});
        if role == "assistant"
            && let Some(reasoning) = message.get("reasoning_content").filter(|x| !x.is_null())
        {
            if !reasoning.is_string() {
                return Err(ApiError::invalid(
                    "reasoning_content must be a string",
                    "messages",
                ));
            }
            result["reasoning_content"] = reasoning.clone();
        }
        if !calls.is_empty() {
            result["tool_calls"] = json!(calls);
        }
        if role == "tool" {
            let id = message["tool_call_id"]
                .as_str()
                .filter(|x| !x.is_empty())
                .ok_or_else(|| ApiError::invalid("tool messages need tool_call_id", "messages"))?;
            result["tool_call_id"] = json!(id);
        }
        normalized.push(result);
    }
    Ok(normalized)
}
fn boolean(v: &Value, name: &str, default: bool) -> Result<bool, ApiError> {
    match v.get(name) {
        None | Some(Value::Null) => Ok(default),
        Some(Value::Bool(b)) => Ok(*b),
        _ => Err(ApiError::invalid("expected a boolean", name)),
    }
}
fn number(
    v: &Value,
    name: &str,
    default: f32,
    minimum: f32,
    maximum: f32,
) -> Result<f32, ApiError> {
    let n = match v.get(name) {
        None | Some(Value::Null) => return Ok(default),
        Some(n) => n
            .as_f64()
            .ok_or_else(|| ApiError::invalid("expected a number", name))?,
    };
    if !n.is_finite() || n < minimum as f64 || n > maximum as f64 {
        return Err(ApiError::invalid(
            format!("must be between {minimum} and {maximum}"),
            name,
        ));
    }
    Ok(n as f32)
}
fn integer(v: &Value, name: &str, default: usize, minimum: usize) -> Result<usize, ApiError> {
    match v.get(name) {
        None | Some(Value::Null) => Ok(default),
        Some(n) => n
            .as_u64()
            .and_then(|x| usize::try_from(x).ok())
            .filter(|x| *x >= minimum)
            .ok_or_else(|| ApiError::invalid(format!("expected an integer >= {minimum}"), name)),
    }
}
fn unsupported(
    v: &Value,
    name: &str,
    harmless: impl FnOnce(&Value) -> bool,
) -> Result<(), ApiError> {
    if let Some(value) = v.get(name).filter(|v| !v.is_null())
        && !harmless(value)
    {
        return Err(ApiError::invalid(format!("{name} is not supported"), name));
    }
    Ok(())
}
pub(super) fn parse_request(
    v: Value,
    chat: bool,
    model: &str,
    context: usize,
) -> Result<ParsedRequest, ApiError> {
    if !v.is_object() {
        return Err(ApiError::invalid("expected a JSON object", "body"));
    }
    match v.get("model").and_then(Value::as_str) {
        Some(id) if id == model => (),
        Some(_) => {
            return Err(ApiError::new(
                StatusCode::NOT_FOUND,
                "model not found",
                Some("model"),
            ));
        }
        None => return Err(ApiError::invalid("model is required", "model")),
    }
    unsupported(&v, "n", |x| x.as_u64() == Some(1))?;
    unsupported(&v, "best_of", |x| x.as_u64() == Some(1))?;
    unsupported(&v, "echo", |x| x.as_bool() == Some(false))?;
    unsupported(&v, "logprobs", |x| x.as_bool() == Some(false))?;
    unsupported(&v, "top_logprobs", |x| x.as_u64() == Some(0))?;
    unsupported(&v, "logit_bias", |x| {
        x.as_object().is_some_and(|x| x.is_empty())
    })?;
    unsupported(&v, "response_format", |x| {
        x.as_object()
            .is_some_and(|x| x.len() == 1 && x.get("type").and_then(Value::as_str) == Some("text"))
    })?;
    unsupported(&v, "functions", |x| {
        x.as_array().is_some_and(|x| x.is_empty())
    })?;
    unsupported(&v, "function_call", |x| x.as_str() == Some("none"))?;
    let tools = parse_tools(&v, chat)?;
    let tools_enabled = tools.is_some();
    for name in [
        "suffix",
        "prediction",
        "audio",
        "grammar",
        "guided_json",
        "guided_regex",
        "allowed_token_ids",
        "reasoning_effort",
        "reasoning_format",
        "verbosity",
        "stop_token_ids",
        "prompt_logprobs",
        "add_generation_prompt",
        "continue_final_message",
    ] {
        unsupported(&v, name, |_| false)?;
    }
    unsupported(&v, "modalities", |x| {
        x.as_array().is_some_and(|x| x.len() == 1 && x[0] == "text")
    })?;
    unsupported(&v, "parallel_tool_calls", |x| {
        x.as_bool() == Some(true) || (!tools_enabled && x.as_bool() == Some(false))
    })?;
    unsupported(&v, "ignore_eos", |x| x.as_bool() == Some(false))?;
    unsupported(&v, "min_tokens", |x| x.as_u64() == Some(0))?;
    let defaults = SamplingParams::default();
    let sampling = SamplingParams {
        temperature: number(&v, "temperature", defaults.temperature, 0.0, 2.0)?,
        top_p: number(&v, "top_p", defaults.top_p, 0.0, 1.0)?,
        top_k: integer(&v, "top_k", defaults.top_k, 0)?,
        min_p: number(&v, "min_p", defaults.min_p, 0.0, 1.0)?,
        seed: match v.get("seed") {
            None | Some(Value::Null) => None,
            Some(x) => Some(
                x.as_u64()
                    .or_else(|| x.as_i64().map(|seed| seed as u64))
                    .ok_or_else(|| ApiError::invalid("seed must be an integer", "seed"))?,
            ),
        },
        repeat_penalty: number(
            &v,
            "repeat_penalty",
            defaults.repeat_penalty,
            f32::MIN_POSITIVE,
            f32::MAX,
        )?,
        presence_penalty: number(&v, "presence_penalty", defaults.presence_penalty, -2.0, 2.0)?,
        frequency_penalty: number(
            &v,
            "frequency_penalty",
            defaults.frequency_penalty,
            -2.0,
            2.0,
        )?,
    };
    if v.get("max_tokens").is_some_and(|x| !x.is_null())
        && v.get("max_completion_tokens").is_some_and(|x| !x.is_null())
        && v["max_tokens"] != v["max_completion_tokens"]
    {
        return Err(ApiError::invalid(
            "max_tokens and max_completion_tokens disagree",
            "max_completion_tokens",
        ));
    }
    let max_tokens = if v.get("max_completion_tokens").is_some_and(|x| !x.is_null()) {
        integer(
            &v,
            "max_completion_tokens",
            crate::sampling::DEFAULT_MAX_TOKENS,
            1,
        )?
    } else {
        integer(&v, "max_tokens", crate::sampling::DEFAULT_MAX_TOKENS, 1)?
    };
    if max_tokens > context {
        return Err(ApiError::invalid(
            "maximum output length exceeds model context",
            "max_tokens",
        ));
    }
    let stop = match v.get("stop") {
        None | Some(Value::Null) => vec![],
        Some(Value::String(s)) if !s.is_empty() => vec![s.clone()],
        Some(Value::Array(a))
            if a.len() <= 4 && a.iter().all(|x| x.as_str().is_some_and(|s| !s.is_empty())) =>
        {
            a.iter().map(|x| x.as_str().unwrap().to_owned()).collect()
        }
        _ => {
            return Err(ApiError::invalid(
                "stop must be a nonempty string or up to four nonempty strings",
                "stop",
            ));
        }
    };
    let prompt = if chat {
        let messages = v
            .get("messages")
            .and_then(Value::as_array)
            .filter(|x| !x.is_empty())
            .ok_or_else(|| ApiError::invalid("messages must be a nonempty array", "messages"))?;
        let normalized = normalize_messages(messages)?;
        let mut thinking = boolean(&v, "enable_thinking", true)?;
        if let Some(kwargs) = v.get("chat_template_kwargs").filter(|x| !x.is_null()) {
            let kwargs = kwargs
                .as_object()
                .ok_or_else(|| ApiError::invalid("expected an object", "chat_template_kwargs"))?;
            if kwargs.keys().any(|x| x != "enable_thinking") {
                return Err(ApiError::invalid(
                    "unsupported chat template option",
                    "chat_template_kwargs",
                ));
            }
            if let Some(x) = kwargs.get("enable_thinking") {
                thinking = x.as_bool().ok_or_else(|| {
                    ApiError::invalid("enable_thinking must be a boolean", "chat_template_kwargs")
                })?;
            }
        }
        Prompt::Chat {
            messages: normalized,
            tools,
            enable_thinking: thinking,
        }
    } else {
        match v.get("prompt") {
            Some(Value::String(s)) => Prompt::Text(s.clone()),
            Some(Value::Array(a))
                if !a.is_empty()
                    && a.iter()
                        .all(|x| x.as_u64().is_some_and(|n| n <= u32::MAX as u64)) =>
            {
                Prompt::Tokens(a.iter().map(|x| x.as_u64().unwrap() as u32).collect())
            }
            _ => {
                return Err(ApiError::invalid(
                    "prompt must be a string or a nonempty token ID array; batching is not supported",
                    "prompt",
                ));
            }
        }
    };
    let stream = boolean(&v, "stream", false)?;
    let include_usage = match v.get("stream_options").filter(|x| !x.is_null()) {
        None => false,
        Some(x) => {
            if !stream {
                return Err(ApiError::invalid(
                    "stream_options requires stream=true",
                    "stream_options",
                ));
            }
            let map = x
                .as_object()
                .ok_or_else(|| ApiError::invalid("expected an object", "stream_options"))?;
            if map.keys().any(|x| x != "include_usage") {
                return Err(ApiError::invalid(
                    "unsupported stream option",
                    "stream_options",
                ));
            }
            boolean(x, "include_usage", false)?
        }
    };
    Ok(ParsedRequest {
        request: GenerationRequest {
            prompt,
            sampling,
            max_tokens,
            stop,
        },
        stream,
        include_usage,
        tools_enabled,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn image_parts_preserve_text_order_and_remote_sources_are_rejected() {
        let message = json!({"role":"user","content":[{"type":"text","text":"first"},{"type":"image_url","image_url":{"url":"data:image/png;base64,AA=="}},{"type":"text","text":"last"}]});
        assert_eq!(
            normalize_messages(std::slice::from_ref(&message)).unwrap()[0],
            message
        );
        let mut remote = message.clone();
        remote["content"][1]["image_url"]["url"] = json!("https://example.org/photo.png");
        assert!(normalize_messages(&[remote]).is_err());
        let mut system = message;
        system["role"] = json!("system");
        assert!(normalize_messages(&[system]).is_err());
        let text = json!({"role":"user","content":[{"type":"text","text":"a"},{"type":"text","text":"b"}]});
        assert_eq!(normalize_messages(&[text]).unwrap()[0]["content"], "ab");
    }
}
