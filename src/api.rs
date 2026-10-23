use crate::engine::{
    Engine, GenerationEvent, GenerationHandle, GenerationRequest, GenerationResult, Prompt,
};
use crate::sampling::SamplingParams;
#[path = "api/request.rs"]
mod request;
#[path = "api/tool_parser.rs"]
mod tool_parser;
use axum::{
    Json, Router,
    extract::{State, rejection::JsonRejection},
    http::{HeaderMap, StatusCode},
    response::{
        IntoResponse, Response,
        sse::{Event, KeepAlive, Sse},
    },
    routing::{get, post},
};
use futures_util::stream;
use request::*;
use serde_json::{Value, json};
use std::{
    convert::Infallible,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};
pub use tool_parser::ToolCallParser;

#[derive(Clone)]
struct AppState {
    engine: Engine,
    tool_call_parser: ToolCallParser,
    api_key: Option<Arc<str>>,
}

pub fn router(engine: Engine, api_key: Option<String>) -> Router {
    router_with_parser(engine, api_key, ToolCallParser::default())
}

pub fn router_with_parser(
    engine: Engine,
    api_key: Option<String>,
    tool_call_parser: ToolCallParser,
) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat))
        .route("/v1/completions", post(completions))
        .fallback(|| async { ApiError::new(StatusCode::NOT_FOUND, "unknown endpoint", None) })
        .layer(axum::extract::DefaultBodyLimit::max(24 * 1024 * 1024))
        .with_state(AppState {
            engine,
            tool_call_parser,
            api_key: api_key.map(Arc::from),
        })
}

pub async fn serve(
    engine: Engine,
    host: &str,
    port: u16,
    api_key: Option<String>,
) -> anyhow::Result<()> {
    serve_with_parser(engine, host, port, api_key, ToolCallParser::default()).await
}

pub async fn serve_with_parser(
    engine: Engine,
    host: &str,
    port: u16,
    api_key: Option<String>,
    tool_call_parser: ToolCallParser,
) -> anyhow::Result<()> {
    eprintln!("{}", tool_call_parser.startup_message());
    let listener = tokio::net::TcpListener::bind((host, port)).await?;
    eprintln!("OpenAI API listening on http://{}", listener.local_addr()?);
    let shutdown_engine = engine.clone();
    axum::serve(
        listener,
        router_with_parser(engine, api_key, tool_call_parser),
    )
    .with_graceful_shutdown(async move {
        let _ = tokio::signal::ctrl_c().await;
        shutdown_engine.shutdown();
    })
    .await?;
    Ok(())
}

#[derive(Debug)]
struct ApiError {
    status: StatusCode,
    message: String,
    param: Option<String>,
}
impl ApiError {
    fn new(status: StatusCode, message: impl Into<String>, param: Option<&str>) -> Self {
        Self {
            status,
            message: message.into(),
            param: param.map(str::to_owned),
        }
    }
    fn invalid(message: impl Into<String>, param: &str) -> Self {
        Self::new(StatusCode::BAD_REQUEST, message, Some(param))
    }
}
impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let kind = match self.status {
            StatusCode::UNAUTHORIZED => "authentication_error",
            StatusCode::TOO_MANY_REQUESTS | StatusCode::SERVICE_UNAVAILABLE => "server_error",
            s if s.is_server_error() => "server_error",
            _ => "invalid_request_error",
        };
        (self.status, Json(json!({"error":{"message":self.message,"type":kind,"param":self.param,"code":null}}))).into_response()
    }
}
fn authenticate(state: &AppState, headers: &HeaderMap) -> Result<(), ApiError> {
    if let Some(key) = &state.api_key
        && headers
            .get("authorization")
            .and_then(|h| h.to_str().ok())
            .and_then(|h| h.strip_prefix("Bearer "))
            != Some(key.as_ref())
    {
        return Err(ApiError::new(
            StatusCode::UNAUTHORIZED,
            "invalid API key",
            None,
        ));
    }
    Ok(())
}
async fn health(State(state): State<AppState>) -> Json<Value> {
    Json(json!({"status":"ok","model":state.engine.info().id}))
}
async fn models(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<Value>, ApiError> {
    authenticate(&state, &headers)?;
    Ok(Json(
        json!({"object":"list","data":[{"id":state.engine.info().id,"object":"model","created":0,"owned_by":"local"}]}),
    ))
}
async fn chat(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Result<Json<Value>, JsonRejection>,
) -> Result<Response, ApiError> {
    respond(state, headers, body, true).await
}
async fn completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Result<Json<Value>, JsonRejection>,
) -> Result<Response, ApiError> {
    respond(state, headers, body, false).await
}

fn usage(result: &GenerationResult) -> Value {
    json!({"prompt_tokens":result.prompt_tokens,"completion_tokens":result.completion_tokens,"total_tokens":result.prompt_tokens + result.completion_tokens,"prompt_tokens_details":{"cached_tokens":result.cached_tokens}})
}
fn created() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
fn response_id(chat: bool) -> String {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    format!(
        "{}-{}-{}",
        if chat { "chatcmpl" } else { "cmpl" },
        created(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    )
}
fn generation_error(error: anyhow::Error) -> ApiError {
    let busy = error.downcast_ref::<crate::engine::EngineBusy>().is_some();
    let text = error.to_string();
    let status = if busy {
        StatusCode::SERVICE_UNAVAILABLE
    } else {
        StatusCode::BAD_REQUEST
    };
    ApiError::new(status, text, None)
}
async fn respond(
    state: AppState,
    headers: HeaderMap,
    body: Result<Json<Value>, JsonRejection>,
    chat: bool,
) -> Result<Response, ApiError> {
    authenticate(&state, &headers)?;
    let Json(body) =
        body.map_err(|e| ApiError::new(StatusCode::BAD_REQUEST, e.body_text(), None))?;
    let parsed = parse_request(
        body,
        chat,
        &state.engine.info().id,
        state.engine.info().context,
    )?;
    let preopened_thinking = chat
        && state
            .engine
            .prompt_opens_thinking(&parsed.request.prompt)
            .map_err(generation_error)?;
    let preserve_xml =
        chat && parsed.tools_enabled && state.tool_call_parser == ToolCallParser::Minicpm5;
    let mut handle = state
        .engine
        .generate_with_tool_markers(parsed.request, preserve_xml)
        .map_err(generation_error)?;
    let id = response_id(chat);
    let timestamp = created();
    let model = state.engine.info().id.clone();
    if parsed.stream {
        return Ok(stream_response(
            handle,
            id,
            timestamp,
            model,
            chat,
            parsed.include_usage,
            parsed.tools_enabled,
            preopened_thinking,
            state.tool_call_parser,
        ));
    }
    while let Some(event) = handle.events.recv().await {
        match event {
            GenerationEvent::Token(_) => (),
            GenerationEvent::Error(message) => {
                return Err(ApiError::new(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    message,
                    None,
                ));
            }
            GenerationEvent::Finished(result) => {
                let choice = if chat {
                    chat_choice_with_parser(
                        &result,
                        &id,
                        parsed.tools_enabled,
                        preopened_thinking,
                        state.tool_call_parser,
                    )
                } else {
                    json!({"index":0,"text":result.text,"logprobs":null,"finish_reason":result.finish_reason})
                };
                return Ok(Json(json!({"id":id,"object":if chat {"chat.completion"} else {"text_completion"},"created":timestamp,"model":model,"choices":[choice],"usage":usage(&result)})).into_response());
            }
        }
    }
    Err(ApiError::new(
        StatusCode::INTERNAL_SERVER_ERROR,
        "generation ended without a result",
        None,
    ))
}
fn split_thinking(text: &str, preopened: bool) -> (&str, Option<&str>) {
    if let Some(rest) = text
        .strip_prefix("<think>")
        .or_else(|| preopened.then_some(text))
    {
        if let Some((reasoning, content)) = rest.split_once("</think>") {
            return (content, Some(reasoning));
        }
        return ("", Some(rest));
    }
    (text, None)
}
const TOOL_OPEN: &str = "<tool_call>";
const TOOL_CLOSE: &str = "</tool_call>";
fn parse_tool_block(payload: &str) -> Option<Value> {
    let value: Value = serde_json::from_str(payload).ok()?;
    let name = value["name"].as_str().filter(|name| function_name(name))?;
    let arguments = match &value["arguments"] {
        Value::String(s) => serde_json::from_str::<Value>(s).ok()?,
        x => x.clone(),
    };
    if !arguments.is_object() {
        return None;
    }
    Some(json!({"name":name,"arguments":serde_json::to_string(&arguments).ok()?}))
}
struct ToolStream {
    enabled: bool,
    pending: String,
    calls: usize,
    response_id: String,
    xml: Option<tool_parser::MiniCpm5Stream>,
}
impl ToolStream {
    fn new(enabled: bool, response_id: String) -> Self {
        Self {
            enabled,
            pending: String::new(),
            calls: 0,
            response_id,
            xml: None,
        }
    }
    fn with_parser(enabled: bool, response_id: String, parser: ToolCallParser) -> Self {
        let mut stream = Self::new(enabled, response_id.clone());
        if enabled && parser == ToolCallParser::Minicpm5 {
            stream.xml = Some(tool_parser::MiniCpm5Stream::new(response_id));
        }
        stream
    }
    fn push(&mut self, text: &str, flush: bool) -> Vec<Value> {
        if !self.enabled {
            return if text.is_empty() {
                vec![]
            } else {
                vec![json!({"content":text})]
            };
        }
        if let Some(xml) = &mut self.xml {
            let out = xml.push(text, flush);
            self.calls = xml.calls;
            return out;
        }
        self.pending.push_str(text);
        let mut out = vec![];
        loop {
            if let Some(start) = self.pending.find(TOOL_OPEN) {
                if start > 0 {
                    out.push(json!({"content":self.pending[..start]}));
                    self.pending.drain(..start);
                }
                if let Some(end) = self.pending[TOOL_OPEN.len()..].find(TOOL_CLOSE) {
                    let end = end + TOOL_OPEN.len();
                    let consumed = end + TOOL_CLOSE.len();
                    if let Some(function) = parse_tool_block(&self.pending[TOOL_OPEN.len()..end]) {
                        out.push(json!({"tool_calls":[{"index":self.calls,"id":format!("{}_call_{}",self.response_id,self.calls),"type":"function","function":function}]}));
                        self.calls += 1;
                    } else {
                        out.push(json!({"content":self.pending[..consumed]}));
                    }
                    self.pending.drain(..consumed);
                    continue;
                }
                if flush {
                    out.push(json!({"content":std::mem::take(&mut self.pending)}));
                }
                return out;
            }
            let hold = if flush {
                0
            } else {
                (1..TOOL_OPEN.len().min(self.pending.len() + 1))
                    .rev()
                    .find(|n| {
                        self.pending.is_char_boundary(self.pending.len() - n)
                            && TOOL_OPEN.starts_with(&self.pending[self.pending.len() - n..])
                    })
                    .unwrap_or(0)
            };
            let end = self.pending.len() - hold;
            if end > 0 {
                out.push(json!({"content":self.pending[..end]}));
                self.pending.drain(..end);
            }
            return out;
        }
    }
}
#[cfg(test)]
fn chat_choice(
    result: &GenerationResult,
    id: &str,
    tools_enabled: bool,
    preopened_thinking: bool,
) -> Value {
    chat_choice_with_parser(
        result,
        id,
        tools_enabled,
        preopened_thinking,
        ToolCallParser::ToolCallJson,
    )
}

fn chat_choice_with_parser(
    result: &GenerationResult,
    id: &str,
    tools_enabled: bool,
    preopened_thinking: bool,
    protocol: ToolCallParser,
) -> Value {
    let (text, reasoning) = split_thinking(&result.text, preopened_thinking);
    let mut parser = ToolStream::new(tools_enabled, id.to_owned());
    let mut content = String::new();
    let mut calls = vec![];
    let deltas = if tools_enabled && protocol == ToolCallParser::Minicpm5 {
        tool_parser::parse_minicpm5(text, id)
    } else {
        parser.push(text, true)
    };
    for delta in deltas {
        if let Some(text) = delta["content"].as_str() {
            content.push_str(text);
        }
        if let Some(tools) = delta["tool_calls"].as_array() {
            for call in tools {
                let mut call = call.clone();
                call.as_object_mut().unwrap().remove("index");
                calls.push(call);
            }
        }
    }
    let mut message = json!({"role":"assistant","content":content});
    if let Some(reasoning) = reasoning {
        message["reasoning_content"] = json!(reasoning);
    }
    let finish_reason = if calls.is_empty() {
        result.finish_reason.as_str()
    } else {
        if content.trim().is_empty() {
            message["content"] = Value::Null;
        }
        message["tool_calls"] = json!(calls);
        "tool_calls"
    };
    json!({"index":0,"message":message,"finish_reason":finish_reason})
}

#[derive(Default)]
struct ThinkingStream {
    pending: String,
    mode: u8,
}
impl ThinkingStream {
    fn push(&mut self, text: &str, flush: bool) -> Vec<Value> {
        self.pending.push_str(text);
        if self.mode == 0 {
            if self.pending.starts_with("<think>") {
                self.pending.drain(..7);
                self.mode = 1;
            } else if !flush && "<think>".starts_with(&self.pending) {
                return vec![];
            } else {
                self.mode = 2;
            }
        }
        let mut out = vec![];
        if self.mode == 1 {
            if let Some(end) = self.pending.find("</think>") {
                if end > 0 {
                    out.push(json!({"reasoning_content":self.pending[..end]}));
                }
                self.pending.drain(..end + 8);
                self.mode = 2;
            } else {
                let hold = if flush {
                    0
                } else {
                    (1..=8.min(self.pending.len()))
                        .rev()
                        .find(|n| {
                            self.pending.is_char_boundary(self.pending.len() - n)
                                && "</think>".starts_with(&self.pending[self.pending.len() - n..])
                        })
                        .unwrap_or(0)
                };
                let end = self.pending.len() - hold;
                if end > 0 {
                    out.push(json!({"reasoning_content":self.pending[..end]}));
                    self.pending.drain(..end);
                }
                return out;
            }
        }
        if !self.pending.is_empty() {
            out.push(json!({"content":std::mem::take(&mut self.pending)}));
        }
        out
    }
}
struct StreamState {
    handle: GenerationHandle,
    id: String,
    created: u64,
    model: String,
    chat: bool,
    include_usage: bool,
    started: bool,
    done: bool,
    pending: std::collections::VecDeque<Event>,
    thinking: ThinkingStream,
    tools: ToolStream,
}
impl StreamState {
    fn chat_delta(&mut self, delta: Value) {
        if let Some(content) = delta["content"].as_str() {
            for tool_delta in self.tools.push(content, false) {
                self.pending.push_back(self.chunk(tool_delta, Value::Null));
            }
        } else {
            self.pending.push_back(self.chunk(delta, Value::Null));
        }
    }
    fn chunk(&self, delta: Value, finish: Value) -> Event {
        let choice = if self.chat {
            json!({"index":0,"delta":delta,"finish_reason":finish})
        } else {
            json!({"index":0,"text":delta.as_str().unwrap_or(""),"logprobs":null,"finish_reason":finish})
        };
        let mut chunk = json!({"id":self.id,"object":if self.chat {"chat.completion.chunk"} else {"text_completion"},"created":self.created,"model":self.model,"choices":[choice]});
        if self.include_usage {
            chunk["usage"] = Value::Null;
        }
        Event::default().data(chunk.to_string())
    }
}
#[allow(clippy::too_many_arguments)]
fn stream_response(
    handle: GenerationHandle,
    id: String,
    created: u64,
    model: String,
    chat: bool,
    include_usage: bool,
    tools_enabled: bool,
    preopened_thinking: bool,
    parser: ToolCallParser,
) -> Response {
    let tools = ToolStream::with_parser(tools_enabled, id.clone(), parser);
    let state = StreamState {
        handle,
        id,
        created,
        model,
        chat,
        include_usage,
        started: false,
        done: false,
        pending: Default::default(),
        thinking: ThinkingStream {
            mode: u8::from(preopened_thinking),
            ..Default::default()
        },
        tools,
    };
    let events = stream::unfold(state, |mut state| async move {
        loop {
            if let Some(event) = state.pending.pop_front() {
                return Some((Ok::<_, Infallible>(event), state));
            }
            if state.done {
                return None;
            }
            if !state.started {
                state.started = true;
                if state.chat {
                    let event = state.chunk(json!({"role":"assistant","content":""}), Value::Null);
                    return Some((Ok(event), state));
                }
            }
            match state.handle.events.recv().await {
                Some(GenerationEvent::Token(text)) => {
                    if state.chat {
                        for delta in state.thinking.push(&text, false) {
                            state.chat_delta(delta);
                        }
                    } else {
                        state
                            .pending
                            .push_back(state.chunk(json!(text), Value::Null));
                    }
                }
                Some(GenerationEvent::Finished(result)) => {
                    if state.chat {
                        for delta in state.thinking.push("", true) {
                            state.chat_delta(delta);
                        }
                        for delta in state.tools.push("", true) {
                            state.pending.push_back(state.chunk(delta, Value::Null));
                        }
                    }
                    state.pending.push_back(state.chunk(
                        if state.chat { json!({}) } else { json!("") },
                        json!(if state.tools.calls > 0 {
                            "tool_calls"
                        } else {
                            result.finish_reason.as_str()
                        }),
                    ));
                    if state.include_usage {
                        state.pending.push_back(Event::default().data(json!({"id":state.id,"object":if state.chat {"chat.completion.chunk"} else {"text_completion"},"created":state.created,"model":state.model,"choices":[],"usage":usage(&result)}).to_string()));
                    }
                    state.pending.push_back(Event::default().data("[DONE]"));
                    state.done = true;
                }
                event => {
                    let message = match event {
                        Some(GenerationEvent::Error(message)) => message,
                        _ => "generation ended without a result".to_owned(),
                    };
                    state.pending.push_back(Event::default().data(json!({"error":{"message":message,"type":"server_error","param":null,"code":null}}).to_string()));
                    state.pending.push_back(Event::default().data("[DONE]"));
                    state.done = true;
                }
            }
        }
    });
    Sse::new(events)
        .keep_alive(KeepAlive::new().interval(Duration::from_secs(15)))
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    fn request() -> Value {
        json!({"model":"qwen","messages":[{"role":"user","content":"hi"}]})
    }
    #[test]
    fn rejects_semantic_options() {
        for (key, value) in [
            ("n", json!(2)),
            ("response_format", json!({"type":"json_object"})),
            ("tools", json!([{"type":"function"}])),
            ("logprobs", json!(true)),
            ("seed", json!(-0.5)),
            ("max_tokens", json!(0)),
        ] {
            let mut v = request();
            v[key] = value;
            assert!(
                parse_request(v, true, "qwen", 4096).is_err(),
                "accepted {key}"
            );
        }
    }
    #[test]
    fn developer_role_and_signed_seed_are_accepted() {
        let request = json!({"model":"qwen","messages":[{"role":"developer","content":"be concise"},{"role":"user","content":"hi"}],"seed":-1,"top_p":0.0});
        let parsed = parse_request(request, true, "qwen", 4096).unwrap();
        assert_eq!(parsed.request.sampling.seed, Some(u64::MAX));
        parsed.request.sampling.validate().unwrap();
        match parsed.request.prompt {
            Prompt::Chat { messages, .. } => assert_eq!(messages[0]["role"], "system"),
            _ => panic!("expected chat prompt"),
        }
    }
    #[test]
    fn rejects_invalid_image_urls_and_batched_prompts() {
        let mut v = request();
        v["messages"][0]["content"] = json!([{"type":"image_url","image_url":{"url":"x"}}]);
        assert!(parse_request(v, true, "qwen", 4096).is_err());
        assert!(
            parse_request(
                json!({"model":"qwen","prompt":["one","two"]}),
                false,
                "qwen",
                4096
            )
            .is_err()
        );
    }
    #[test]
    fn carries_sampling_stop_and_text_parts() {
        let mut v = request();
        v["messages"][0]["content"] =
            json!([{"type":"text","text":"a"},{"type":"text","text":"b"}]);
        v["stop"] = json!(["END"]);
        v["seed"] = json!(42);
        v["temperature"] = json!(0);
        let parsed = parse_request(v, true, "qwen", 4096).unwrap();
        assert_eq!(parsed.request.stop, vec!["END"]);
        assert_eq!(parsed.request.sampling.seed, Some(42));
        match parsed.request.prompt {
            Prompt::Chat { messages, .. } => assert_eq!(messages[0]["content"], "ab"),
            _ => panic!("expected chat"),
        }
    }
    #[test]
    fn thinking_markers_can_span_tokens() {
        let mut stream = ThinkingStream::default();
        let mut out = vec![];
        for token in ["<th", "ink>abc</th", "ink>answer"] {
            out.extend(stream.push(token, false));
        }
        out.extend(stream.push("", true));
        assert_eq!(
            out,
            vec![
                json!({"reasoning_content":"abc"}),
                json!({"content":"answer"})
            ]
        );
    }
    #[test]
    fn rejects_usage_without_stream() {
        let mut v = request();
        v["stream_options"] = json!({"include_usage":true});
        assert!(parse_request(v, true, "qwen", 4096).is_err());
    }
    #[test]
    fn tools_and_history_are_normalized_for_template() {
        let mut v = request();
        v["tools"] = json!([{"type":"function","function":{"name":"weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}]);
        v["tool_choice"] = json!("auto");
        v["messages"] = json!([
            {"role":"user","content":"weather?"},
            {"role":"assistant","content":null,"tool_calls":[{"id":"call_0","type":"function","function":{"name":"weather","arguments":"{\"city\":\"Beijing\"}"}}]},
            {"role":"tool","tool_call_id":"call_0","content":"sunny"}
        ]);
        let parsed = parse_request(v.clone(), true, "qwen", 4096).unwrap();
        assert!(parsed.tools_enabled);
        match parsed.request.prompt {
            Prompt::Chat {
                messages, tools, ..
            } => {
                assert!(tools.is_some());
                assert_eq!(
                    messages[1]["tool_calls"][0]["function"]["arguments"]["city"],
                    "Beijing"
                );
                assert_eq!(messages[2]["tool_call_id"], "call_0");
            }
            _ => panic!("expected chat"),
        }
        v["tool_choice"] = json!("none");
        assert!(
            !parse_request(v.clone(), true, "qwen", 4096)
                .unwrap()
                .tools_enabled
        );
        v["tool_choice"] = json!("required");
        assert!(parse_request(v, true, "qwen", 4096).is_err());
    }
    #[test]
    fn complete_tools_parse_but_malformed_and_truncated_stay_text() {
        let valid =
            "<tool_call>{\"name\":\"weather\",\"arguments\":{\"city\":\"北京\"}}</tool_call>";
        let malformed = "<tool_call>{\"name\":\"weather\",\"arguments\":bad}</tool_call>";
        let truncated = "<tool_call>{\"name\":\"weather\"";
        let result = GenerationResult {
            text: format!("<think>reason</think>{valid}{malformed}{truncated}"),
            finish_reason: "length".into(),
            prompt_tokens: 1,
            completion_tokens: 2,
            cached_tokens: 0,
            timings: Value::Null,
            stats: Value::Null,
        };
        let choice = chat_choice(&result, "test", true, false);
        assert_eq!(choice["finish_reason"], "tool_calls");
        assert_eq!(choice["message"]["tool_calls"].as_array().unwrap().len(), 1);
        assert_eq!(
            choice["message"]["content"],
            format!("{malformed}{truncated}")
        );
        assert_eq!(choice["message"]["reasoning_content"], "reason");
        assert_eq!(
            choice["message"]["tool_calls"][0]["function"]["arguments"],
            "{\"city\":\"北京\"}"
        );
        let plain = chat_choice(&result, "test", false, false);
        assert_eq!(plain["finish_reason"], "length");
        assert!(plain["message"].get("tool_calls").is_none());
    }
    #[test]
    fn tool_opening_and_json_can_span_every_token_boundary() {
        let input = "prefix<tool_call>{\"name\":\"f\",\"arguments\":{}}</tool_call>suffix";
        let mut parser = ToolStream::new(true, "test".into());
        let mut deltas = vec![];
        for character in input.chars() {
            deltas.extend(parser.push(&character.to_string(), false));
        }
        deltas.extend(parser.push("", true));
        let content: String = deltas
            .iter()
            .filter_map(|x| x["content"].as_str())
            .collect();
        assert_eq!(content, "prefixsuffix");
        assert_eq!(parser.calls, 1);
        assert!(
            deltas
                .iter()
                .any(|x| x["tool_calls"][0]["function"]["name"] == "f")
        );
        let mut truncated = ToolStream::new(true, "test".into());
        assert!(truncated.push("<tool_call>", false).is_empty());
        assert_eq!(
            truncated.push("", true),
            vec![json!({"content":"<tool_call>"})]
        );
    }
    #[tokio::test]
    async fn sse_emits_complete_tool_call_then_tool_finish_reason() {
        use http_body_util::BodyExt;
        let (sender, events) = tokio::sync::mpsc::channel(8);
        let cancel = Arc::new(std::sync::atomic::AtomicBool::new(false));
        for text in [
            "<think>reason</think><tool_",
            "call>{\"name\":\"f\",\"arguments\":{\"x\":1}}",
            "</tool_call>",
        ] {
            sender
                .send(GenerationEvent::Token(text.into()))
                .await
                .unwrap();
        }
        sender
            .send(GenerationEvent::Finished(GenerationResult {
                text: String::new(),
                finish_reason: "stop".into(),
                prompt_tokens: 1,
                completion_tokens: 3,
                cached_tokens: 0,
                timings: Value::Null,
                stats: Value::Null,
            }))
            .await
            .unwrap();
        drop(sender);
        let response = stream_response(
            GenerationHandle { events, cancel },
            "test".into(),
            1,
            "qwen".into(),
            true,
            false,
            true,
            false,
            ToolCallParser::ToolCallJson,
        );
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        let body = std::str::from_utf8(&bytes).unwrap();
        let chunks: Vec<Value> = body
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .filter(|line| *line != "[DONE]")
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        let calls: Vec<&Value> = chunks
            .iter()
            .filter(|x| x["choices"][0]["delta"].get("tool_calls").is_some())
            .collect();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0]["choices"][0]["delta"]["tool_calls"][0]["index"], 0);
        assert_eq!(
            calls[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"],
            "{\"x\":1}"
        );
        assert_eq!(
            chunks.last().unwrap()["choices"][0]["finish_reason"],
            "tool_calls"
        );
    }
    #[tokio::test]
    async fn sse_finishes_with_usage_and_done() {
        use http_body_util::BodyExt;
        let (sender, events) = tokio::sync::mpsc::channel(8);
        let cancel = Arc::new(std::sync::atomic::AtomicBool::new(false));
        sender
            .send(GenerationEvent::Token("<think>reason</th".into()))
            .await
            .unwrap();
        sender
            .send(GenerationEvent::Token("ink>answer".into()))
            .await
            .unwrap();
        sender
            .send(GenerationEvent::Finished(GenerationResult {
                text: "<think>reason</think>answer".into(),
                finish_reason: "stop".into(),
                prompt_tokens: 12,
                completion_tokens: 3,
                cached_tokens: 8,
                timings: Value::Null,
                stats: Value::Null,
            }))
            .await
            .unwrap();
        drop(sender);
        let response = stream_response(
            GenerationHandle {
                events,
                cancel: cancel.clone(),
            },
            "chatcmpl-test".into(),
            1,
            "qwen".into(),
            true,
            true,
            false,
            false,
            ToolCallParser::ToolCallJson,
        );
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let body = std::str::from_utf8(&body).unwrap();
        let events: Vec<&str> = body
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .collect();
        assert_eq!(events.last(), Some(&"[DONE]"));
        let chunks: Vec<Value> = events[..events.len() - 1]
            .iter()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert!(
            chunks
                .iter()
                .any(|chunk| chunk["choices"][0]["delta"]["reasoning_content"] == "reason")
        );
        assert!(
            chunks
                .iter()
                .any(|chunk| chunk["choices"][0]["delta"]["content"] == "answer")
        );
        assert!(
            chunks
                .iter()
                .any(|chunk| chunk["choices"][0]["finish_reason"] == "stop")
        );
        let usage = chunks.last().unwrap();
        assert_eq!(usage["usage"]["total_tokens"], 15);
        assert_eq!(usage["usage"]["prompt_tokens_details"]["cached_tokens"], 8);
        assert!(cancel.load(Ordering::Relaxed));
    }
    #[tokio::test]
    async fn dropping_sse_body_cancels_generation() {
        let (_sender, events) = tokio::sync::mpsc::channel(1);
        let cancel = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let response = stream_response(
            GenerationHandle {
                events,
                cancel: cancel.clone(),
            },
            "cmpl-test".into(),
            1,
            "qwen".into(),
            false,
            false,
            false,
            false,
            ToolCallParser::ToolCallJson,
        );
        assert!(!cancel.load(Ordering::Relaxed));
        drop(response);
        assert!(cancel.load(Ordering::Relaxed));
    }
    #[tokio::test]
    async fn xml_sse_error_does_not_turn_pending_xml_into_a_call() {
        use http_body_util::BodyExt;
        for error in [false, true] {
            let (sender, events) = tokio::sync::mpsc::channel(2);
            sender
                .send(GenerationEvent::Token(
                    "<function name='f'><param name='x'>1".into(),
                ))
                .await
                .unwrap();
            if error {
                sender
                    .send(GenerationEvent::Error("test failure".into()))
                    .await
                    .unwrap();
            }
            drop(sender);
            let cancel = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let response = stream_response(
                GenerationHandle {
                    events,
                    cancel: cancel.clone(),
                },
                "r".into(),
                1,
                "mock".into(),
                true,
                true,
                true,
                false,
                ToolCallParser::Minicpm5,
            );
            let bytes = response.into_body().collect().await.unwrap().to_bytes();
            let body = std::str::from_utf8(&bytes).unwrap();
            assert!(body.contains("server_error"));
            assert!(!body.contains("\"tool_calls\":["));
            assert_eq!(body.matches("data: [DONE]").count(), 1);
            assert!(cancel.load(Ordering::Relaxed));
        }
    }

    #[tokio::test]
    async fn xml_sse_length_flushes_truncated_xml_as_text() {
        use http_body_util::BodyExt;
        let (sender, events) = tokio::sync::mpsc::channel(2);
        let text = "<function name='f'><param name='x'>123";
        sender
            .send(GenerationEvent::Token(text.into()))
            .await
            .unwrap();
        sender
            .send(GenerationEvent::Finished(GenerationResult {
                text: text.into(),
                finish_reason: "length".into(),
                prompt_tokens: 1,
                completion_tokens: 1,
                cached_tokens: 0,
                timings: Value::Null,
                stats: Value::Null,
            }))
            .await
            .unwrap();
        drop(sender);
        let response = stream_response(
            GenerationHandle {
                events,
                cancel: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            },
            "r".into(),
            1,
            "mock".into(),
            true,
            false,
            true,
            false,
            ToolCallParser::Minicpm5,
        );
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        let body = std::str::from_utf8(&bytes).unwrap();
        let chunks: Vec<Value> = body
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .filter(|line| *line != "[DONE]")
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert!(
            chunks
                .iter()
                .any(|c| c["choices"][0]["delta"]["content"] == text)
        );
        assert!(
            chunks
                .iter()
                .any(|c| c["choices"][0]["finish_reason"] == "length")
        );
        assert!(!body.contains("\"tool_calls\":["));
    }

    #[test]
    fn native_prompt_can_open_thinking_before_generated_text() {
        assert_eq!(
            split_thinking("reason</think>answer", true),
            ("answer", Some("reason"))
        );
        assert_eq!(split_thinking("unfinished", true), ("", Some("unfinished")));
        let mut parser = ThinkingStream {
            mode: 1,
            ..Default::default()
        };
        assert_eq!(
            parser.push("reason</th", false),
            vec![json!({"reasoning_content":"reason"})]
        );
        assert_eq!(
            parser.push("ink>answer", true),
            vec![json!({"content":"answer"})]
        );
    }
}
