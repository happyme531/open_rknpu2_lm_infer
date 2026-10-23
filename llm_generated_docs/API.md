# OpenAI API

HTTP 与 SSE 使用 axum/tokio。标准 OpenAI SDK 可以设置本服务的 `/v1` 为 base URL。

| 路由 | 功能 |
|---|---|
| `GET /health` | 已加载模型的健康状态 |
| `GET /v1/models` | 当前模型列表 |
| `POST /v1/chat/completions` | 文本聊天、thinking 和 function tools |
| `POST /v1/completions` | 原始文本或单组 token IDs 补全 |

支持流式与非流式输出、标准 choices/usage、SSE `[DONE]`、`stream_options.include_usage`、`max_tokens` / `max_completion_tokens`、temperature、top_p、top_k、min_p、seed、repeat_penalty、presence_penalty、frequency_penalty 和最多4个 stop。`messages` 接受 system/developer/user/assistant/tool，developer 转为 Qwen3 的 system 指令；文本内容可用 string 或 text parts。`enable_thinking` 和 `chat_template_kwargs.enable_thinking` 控制 Qwen3 thinking，API 默认开启；reasoning 输出到 `reasoning_content`。

CLI `run`、`chat` 与 HTTP API 的默认温度均为 `1`，默认生成上限为 `2048`（CLI `--n-predict`，HTTP `max_tokens` / `max_completion_tokens`）。`temperature:0` 选择贪心采样。CLI 默认 `top_k=20`、`top_p=0.95`；HTTP 默认 `top_k=0`、`top_p=1`，即完整词表采样。比较速度或输出时应显式使用相同的采样参数。

`usage.prompt_tokens_details.cached_tokens` 记录当前请求实际复用的 KV 前缀。发生 context 超限时在排队前返回400；请求队列满时返回503。关闭流式连接会取消对应推理。

## Function tools

支持 `tools` 中的 `type:function`，以及 `tool_choice:auto` / `none`。请求通过 checkpoint 自带的工具模板格式化。完整合法的 `<tool_call>` 块转换成标准 `tool_calls`；流模式在单个完整工具块结束时发送该调用的 name 和 arguments。多个调用有独立 index/id。

工具响应解析支持 Qwen3 的 JSON 工具块。Granite 与 MiniCPM5 使用不同的原生 XML 函数格式，目前保留为文本；普通聊天、thinking、文本补全和流式输出可共用生成引擎。

```json
{
  "model": "Qwen3-4B-W4A8-Mixed-LMHeadW8",
  "messages": [{"role": "user", "content": "查询杭州的天气"}],
  "enable_thinking": false,
  "tools": [{
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "查询城市的天气",
      "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"]
      }
    }
  }],
  "tool_choice": "auto",
  "max_tokens": 128
}
```

客户端执行函数后，将 assistant 的 `tool_calls` 和 role=tool 的结果追加到消息历史，再请求下一轮。模型生成的参数由客户端按照自己的函数接口校验。

## 限制

当前单请求 `n=1`。不支持批量 prompt、logprobs、logit_bias、JSON/grammar 约束、strict schemas、required/指定函数、启用工具时的 `parallel_tool_calls:false`、旧版 functions/function_call、图像/音频、embeddings 或 Responses API。影响生成语义的这些参数返回400；损坏 JSON 同样返回标准 error envelope。工具块截断或 JSON 无效时保留为普通文本。

HTTP 服务与模型实例顺序调度，多个客户端共享一个模型和一个当前前缀缓存。缓存不是按用户保存的会话存储；客户端需要提交完整消息历史。
