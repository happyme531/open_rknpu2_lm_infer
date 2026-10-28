#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
读取 call.md 作为提示词，调用本地 OpenAI 兼容接口（默认 http://127.0.0.1:8080）。

只依赖 Python 标准库，不需要 pip install openai。

针对 RK3588 板上的 ling3tiny（百灵 MoE）+ --thinking 模式做了适配：
  * 思考过程在 reasoning_content，正文在 content，两者分开处理；
  * 若 content 为空（思考把 max_tokens 吃光了），会明确报错并给出建议；
  * --think / --no-think 通过 chat_template_kwargs.enable_thinking 开关思考。

用法:
    python3 call_llm.py                              # 用 ./call.md，自动挑选模型
    python3 call_llm.py --no-think                   # 关掉思考链，最快
    python3 call_llm.py --show-reasoning             # 同时打印思考过程
    python3 call_llm.py -m qwen2.5-7b-instruct       # 指定模型
    python3 call_llm.py --stream                     # 流式输出
    python3 call_llm.py --list-models                # 只列出本地服务可用模型
    python3 call_llm.py --file other.md -o out.json  # 换提示词文件并保存结果
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

DEFAULT_BASE = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8080")
DEFAULT_KEY = os.environ.get("OPENAI_API_KEY", "EMPTY")  # 本地服务一般不校验


# --------------------------------------------------------------------------- #
# HTTP 辅助
# --------------------------------------------------------------------------- #
def api_root(base_url: str) -> str:
    """把 base_url 归一化成 .../v1，兼容用户传 http://ip:8080 或 http://ip:8080/v1"""
    base = base_url.strip().rstrip("/")
    if base.endswith("/v1"):
        return base
    return base + "/v1"


def _request(url: str, key: str, payload=None, timeout: int = 600):
    data = None
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    req = urllib.request.Request(url, data=data, headers=headers,
                                 method="POST" if data else "GET")
    # 本地服务走 loopback，绕开系统代理
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return opener.open(req, timeout=timeout)


def request_json(url: str, key: str, payload=None, timeout: int = 600) -> dict:
    try:
        with _request(url, key, payload, timeout) as resp:
            body = resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:800]
        raise SystemExit("接口返回 HTTP {}：{}\n{}".format(e.code, e.reason, detail))
    except urllib.error.URLError as e:
        raise SystemExit(
            "连不上 {}（{}）。\n请确认本地 OpenAI 兼容服务已启动，且端口正确。".format(url, e.reason)
        )
    except TimeoutError:
        raise SystemExit("请求超时：{}".format(url))

    try:
        return json.loads(body)
    except json.JSONDecodeError:
        raise SystemExit("返回内容不是合法 JSON：\n" + body[:800])


# --------------------------------------------------------------------------- #
# JSON 提取：模型常把结果包在 ```json 里或前后加说明，这里尽量抠出来
# --------------------------------------------------------------------------- #
def extract_json(text: str):
    """能解析就返回 python 对象，否则返回 None。"""
    if not text:
        return None
    s = text.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    # 去掉 ```json ... ``` 围栏
    if "```" in s:
        for block in s.split("```")[1::2]:
            blk = block.strip()
            if blk.lower().startswith("json"):
                blk = blk[4:].strip()
            try:
                return json.loads(blk)
            except json.JSONDecodeError:
                continue

    # 退而求其次：截取第一个 { 到最后一个 }
    i, j = s.find("{"), s.rfind("}")
    if i != -1 and j > i:
        try:
            return json.loads(s[i:j + 1])
        except json.JSONDecodeError:
            pass
    return None


# --------------------------------------------------------------------------- #
# 针对 call.md 的字段校验（--check）
# --------------------------------------------------------------------------- #
# 字段名 -> 数组元素里的子键（None 表示该字段应为字符串）
REQUIRED_FIELDS = {
    "涉及地址": "地址",
    "涉及单位": "单位",
    "事件缘由": None,
    "事件类别": "类别",
    "当事人的诉求": None,
}
ALLOWED_CATEGORY = {"咨询求助", "投诉举报", "纠纷", "个人扬言与造谣", "事故", "其他事件"}


def validate(obj):
    """返回问题列表，空列表 = 合规。"""
    if not isinstance(obj, dict):
        return ["顶层不是 JSON 对象"]
    problems = []
    for key, sub in REQUIRED_FIELDS.items():
        if key not in obj:
            problems.append("缺少字段「{}」".format(key))
            continue
        val = obj[key]
        if sub is None:
            if not isinstance(val, str):
                problems.append("「{}」应为字符串，实际是 {}".format(key, type(val).__name__))
            continue
        if not isinstance(val, list):
            problems.append("「{}」应为数组，实际是 {}".format(key, type(val).__name__))
            continue
        for i, item in enumerate(val):
            if not isinstance(item, dict):
                problems.append("「{}」第 {} 项不是对象".format(key, i + 1))
            elif sub not in item:
                problems.append("「{}」第 {} 项缺少子键「{}」".format(key, i + 1, sub))

    cats = [it.get("类别") for it in obj.get("事件类别", []) if isinstance(it, dict)]
    for c in cats:
        if c and c not in ALLOWED_CATEGORY:
            problems.append("事件类别「{}」不在允许的 6 个取值内".format(c))
    return problems


# --------------------------------------------------------------------------- #
# 模型列表 / 对话
# --------------------------------------------------------------------------- #
def list_models(base_url: str, key: str) -> list:
    data = request_json(api_root(base_url) + "/models", key, timeout=15)
    return [m.get("id") for m in data.get("data", []) if m.get("id")]


def build_payload(model, messages, temperature, max_tokens, think, stream):
    payload = {"model": model, "messages": messages, "stream": bool(stream)}
    if temperature is not None:
        payload["temperature"] = temperature
    if max_tokens:
        payload["max_tokens"] = max_tokens
    # 仅在用户明确选择时传入，否则沿用服务默认值。
    if think is not None:
        payload["chat_template_kwargs"] = {"enable_thinking": think}
    return payload


def chat(base_url, key, model, messages, temperature, max_tokens, timeout, think):
    """返回 dict: content / reasoning / finish / usage / elapsed"""
    payload = build_payload(model, messages, temperature, max_tokens, think, False)
    t0 = time.time()
    data = request_json(api_root(base_url) + "/chat/completions", key, payload, timeout)
    elapsed = time.time() - t0

    try:
        choice = data["choices"][0]
        msg = choice["message"]
    except (KeyError, IndexError):
        raise SystemExit("返回结构异常：\n" + json.dumps(data, ensure_ascii=False)[:800])

    return {
        "content": (msg.get("content") or "").strip(),
        # 部分是推理模型把思考放在这里（ling3tiny --thinking / Qwen3 / DeepSeek-R1）
        "reasoning": (msg.get("reasoning_content") or msg.get("reasoning") or "").strip(),
        "finish": choice.get("finish_reason"),
        "usage": data.get("usage", {}),
        "elapsed": elapsed,
    }


def chat_stream(base_url, key, model, messages, temperature, max_tokens, timeout,
                think, show_reasoning):
    payload = build_payload(model, messages, temperature, max_tokens, think, True)
    url = api_root(base_url) + "/chat/completions"
    parts, think_parts = [], []
    t0 = time.time()
    first_tok_at = None

    try:
        with _request(url, key, payload, timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                chunk = line[5:].strip()
                if chunk == "[DONE]":
                    break
                try:
                    delta = json.loads(chunk)["choices"][0].get("delta", {})
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
                if first_tok_at is None:
                    first_tok_at = time.time()

                rt = delta.get("reasoning_content") or delta.get("reasoning") or ""
                if rt:
                    think_parts.append(rt)
                    if show_reasoning:
                        sys.stderr.write(rt)
                        sys.stderr.flush()
                piece = delta.get("content") or ""
                if piece:
                    parts.append(piece)
                    sys.stdout.write(piece)
                    sys.stdout.flush()
    except urllib.error.HTTPError as e:
        raise SystemExit("\n接口返回 HTTP {}：{}".format(e.code, e.read().decode("utf-8", "replace")[:500]))
    except urllib.error.URLError as e:
        raise SystemExit("\n连不上 {}（{}）".format(url, e.reason))

    sys.stdout.write("\n")
    return {
        "content": "".join(parts).strip(),
        "reasoning": "".join(think_parts).strip(),
        "finish": None,
        "usage": {},
        "elapsed": time.time() - t0,
        "ttft": (first_tok_at - t0) if first_tok_at else None,
    }


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="用 call.md 作为提示词调用本地 OpenAI 兼容接口")
    ap.add_argument("-f", "--file", default="call.md", help="提示词文件（默认 call.md）")
    ap.add_argument("-b", "--base-url", default=DEFAULT_BASE, help="服务地址（默认 %(default)s）")
    ap.add_argument("-k", "--api-key", default=DEFAULT_KEY, help="API Key（本地服务随便填）")
    ap.add_argument("-m", "--model", default=os.environ.get("OPENAI_MODEL", ""),
                    help="模型名；不填则自动取 /v1/models 返回的第一个")
    ap.add_argument("--system", default=None, help="额外的 system 指令（可选）")
    ap.add_argument("--user", default="请按上述要求输出 JSON。",
                    help="发给模型的 user 消息（默认：请按上述要求输出 JSON。）")
    ap.add_argument("--as-user", action="store_true", help="把整份提示词作为 user 消息发送（默认作为 system 消息）")
    ap.add_argument("--think", dest="think", action="store_true", default=None,
                    help="强制开启思考链")
    ap.add_argument("--no-think", dest="think", action="store_false",
                    help="关闭思考链（chat_template_kwargs.enable_thinking=false）")
    ap.add_argument("--show-reasoning", action="store_true", help="把 reasoning_content 打印到 stderr")
    ap.add_argument("--temperature", type=float, default=0.1, help="默认 0.1")
    ap.add_argument("--max-tokens", type=int, default=2048,
                    help="默认 2048（思考模式会先消耗 token，太小会导致 content 为空）")
    ap.add_argument("--timeout", type=int, default=900, help="单次请求超时秒数，默认 900")
    ap.add_argument("--stream", action="store_true", help="流式输出")
    ap.add_argument("--list-models", action="store_true", help="只列出可用模型后退出")
    ap.add_argument("--check", action="store_true",
                    help="按 call.md 的 5 个字段做结构校验，打印 PASS/FAIL")
    ap.add_argument("-o", "--out", default=None, help="把模型返回的正文写入文件")
    ap.add_argument("--json-out", default=None, help="把解析后的 JSON 写入文件（美化）")
    args = ap.parse_args()

    # 1) 只列模型
    if args.list_models:
        for name in list_models(args.base_url, args.api_key):
            print(name)
        return 0

    # 2) 读提示词
    prompt_path = os.path.abspath(args.file)
    if not os.path.isfile(prompt_path):
        raise SystemExit("找不到提示词文件：{}".format(prompt_path))
    with open(prompt_path, "r", encoding="utf-8") as fp:
        prompt = fp.read().strip()
    if not prompt:
        raise SystemExit("提示词文件是空的：{}".format(prompt_path))

    # 3) 决定模型名
    model = args.model
    if not model:
        try:
            models = list_models(args.base_url, args.api_key)
        except SystemExit:
            models = []
        if not models:
            raise SystemExit(
                "没能从 /v1/models 取到模型列表，请用 -m 手动指定模型名。\n"
                "例如：python3 call_llm.py -m model"
            )
        model = models[0]

    # 4) 组消息：整份提示词默认作为 system，user 只放一句触发语
    if args.as_user:
        messages = [{"role": "user", "content": prompt}]
    else:
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": args.user},
        ]
    if args.system:
        messages.insert(0, {"role": "system", "content": args.system})

    mode = "思考链开" if args.think else ("思考链关" if args.think is False else "思考链跟随服务默认")
    print("[提示词] {} （{} 字）".format(prompt_path, len(prompt)), file=sys.stderr)
    print("[服务]   {}".format(api_root(args.base_url)), file=sys.stderr)
    print("[模型]   {}  |  {}  |  max_tokens={}".format(model, mode, args.max_tokens), file=sys.stderr)
    print("-" * 60, file=sys.stderr)

    # 5) 请求
    if args.stream:
        r = chat_stream(args.base_url, args.api_key, model, messages, args.temperature,
                        args.max_tokens, args.timeout, args.think, args.show_reasoning)
    else:
        r = chat(args.base_url, args.api_key, model, messages, args.temperature,
                 args.max_tokens, args.timeout, args.think)

    answer, reasoning = r["content"], r["reasoning"]

    if args.show_reasoning and reasoning and not args.stream:
        sys.stderr.write("\n===== 思考过程 ({} 字) =====\n{}\n".format(len(reasoning), reasoning))

    # 6) content 为空：多半是思考把 max_tokens 吃光了
    if not answer:
        if reasoning:
            print("模型只输出了思考过程，content 为空（finish_reason={}）。\n"
                  "原因：思考链消耗了全部 max_tokens。建议加大 --max-tokens（如 4096）"
                  "或加 --no-think。".format(r["finish"]), file=sys.stderr)
            if args.show_reasoning:
                sys.stderr.write("\n[思考内容末尾]\n" + reasoning[-600:] + "\n")
        else:
            print("模型返回空内容（finish_reason={}）。".format(r["finish"]), file=sys.stderr)
        return 2

    # 7) 输出：能当 JSON 解析就美化打印，否则原样打印
    obj = extract_json(answer)
    if args.stream:
        try:
            json.loads(answer)
            print("\n[OK] 返回内容是合法 JSON", file=sys.stderr)
        except json.JSONDecodeError:
            print("\n[!] 返回内容不是裸 JSON（可能夹了说明文字或代码围栏）", file=sys.stderr)
    elif obj is not None:
        print(json.dumps(obj, ensure_ascii=False, indent=2))
    else:
        print(answer)
        print("[!] 无法解析为 JSON，已按原文输出", file=sys.stderr)

    # 8) 统计
    u = r.get("usage") or {}
    stats = "耗时 {:.1f}s".format(r["elapsed"])
    if u:
        stats += " | prompt {} tok, completion {} tok".format(
            u.get("prompt_tokens", "?"), u.get("completion_tokens", "?"))
        ct = u.get("completion_tokens") or 0
        if ct and r["elapsed"] > 0:
            stats += " | {:.1f} tok/s".format(ct / r["elapsed"])
    stats += " | finish_reason={}".format(r["finish"])
    print("[" + stats + "]", file=sys.stderr)

    # 8.5) 结构校验
    if args.check:
        if obj is None:
            print("[校验] FAIL：输出无法解析为 JSON", file=sys.stderr)
            return 3
        problems = validate(obj)
        if problems:
            print("[校验] FAIL（{} 处问题）".format(len(problems)), file=sys.stderr)
            for p in problems:
                print("        - " + p, file=sys.stderr)
            return 3
        print("[校验] PASS：5 个字段齐全，结构合法", file=sys.stderr)

    # 9) 落盘
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fp:
            fp.write(answer)
        print("[已保存] {}".format(os.path.abspath(args.out)), file=sys.stderr)
    if args.json_out and obj is not None:
        with open(args.json_out, "w", encoding="utf-8") as fp:
            json.dump(obj, fp, ensure_ascii=False, indent=2)
        print("[已保存] {}".format(os.path.abspath(args.json_out)), file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
