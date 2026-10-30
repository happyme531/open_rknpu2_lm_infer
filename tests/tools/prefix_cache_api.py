"""Compare resident prefix reuse with cold requests through the production API."""
import argparse
import http.client
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import time
import urllib.request
import urllib.error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--mode", choices=("ar", "mtp", "dflash"), default="ar")
    parser.add_argument("--draft-model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with socket.socket() as selection:
        selection.bind(("127.0.0.1", 0))
        port = selection.getsockname()[1]
    command = [str(args.binary), "-m", str(args.model), "--ctx-size", "2048", "--ubatch-size", "128", "--ctx-checkpoints", "4", "--no-tune"]
    if args.mode != "ar":
        command += ["--spec-type", "draft-mtp" if args.mode == "mtp" else "draft-dflash", "--spec-draft-n-max", "3"]
    if args.mode == "dflash":
        assert args.draft_model is not None
        command += ["-md", str(args.draft_model)]
    command += ["serve", "--host", "127.0.0.1", "--port", str(port)]
    log_path = args.output / "server.log"
    log = log_path.open("w")
    process = subprocess.Popen(command, stdout=log, stderr=log, env={**os.environ, "RUST_LOG": "info", "NO_COLOR": "1"})
    report = {"mode": args.mode, "model": str(args.model), "command": command, "checks": [], "requests": [], "status": "RUNNING"}

    def save():
        (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    try:
        deadline = time.monotonic() + 180
        while True:
            assert process.poll() is None, log_path.read_text()[-2000:]
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as response:
                    model_id = json.load(response)["model"]
                break
            except OSError:
                assert time.monotonic() < deadline, "server readiness timeout"
                time.sleep(0.2)

        def request(name, payload, maximum=8):
            offset = log_path.stat().st_size
            body = {"model": model_id, "temperature": 0, "seed": 42, "max_tokens": maximum, **payload}
            endpoint = "chat/completions" if "messages" in payload else "completions"
            start = time.monotonic()
            req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/{endpoint}", json.dumps(body).encode(), {"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=180) as response:
                    value = json.load(response)
            except urllib.error.HTTPError as error:
                raise AssertionError((name, error.code, error.read().decode())) from error
            elapsed = (time.monotonic() - start) * 1000
            with log_path.open() as stream:
                stream.seek(offset)
                lines = re.sub(r"\x1b\[[0-9;]*m", "", stream.read())
            line = next(line for line in lines.splitlines() if "generation finished" in line)
            timings = json.JSONDecoder().raw_decode(line.split("timings=", 1)[1])[0]
            entry = {"name": name, "request": body, "response": value, "wall_ms": elapsed, "timings": timings}
            for field in ("prefix_checkpoint_count", "prefix_checkpoint_positions", "prefix_checkpoint_bytes", "prefix_checkpoint_hits", "prefix_checkpoint_save_ms", "tune_cache_hits", "tune_cache_misses", "tune_calls", "strategy_fallback_calls"):
                match = re.search(r"\b" + field + r"=([^ ]+)", line)
                if match:
                    entry[field] = json.loads(match.group(1))
            report["requests"].append(entry)
            save()
            return entry

        def clear():
            value = request("clear", {"prompt": [1]}, 1)
            assert value["response"]["usage"]["prompt_tokens_details"]["cached_tokens"] == 0

        def signature(entry):
            response = entry["response"]
            return response["choices"], response["usage"]["completion_tokens"]

        def compare(name, source, target, hit=True):
            clear()
            request(name + "-source", source)
            warm = request(name + "-warm", target)
            clear()
            cold = request(name + "-cold", {**target, "cache_prompt": False})
            cached = warm["response"]["usage"]["prompt_tokens_details"]["cached_tokens"]
            if hit is not None:
                assert (cached > 0) == hit, (name, cached)
            assert cold["response"]["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
            output_equal = signature(warm) == signature(cold)
            if name == "repeat-prompt":
                assert output_equal, (name, signature(warm), signature(cold))
            assert warm.get("prefix_checkpoint_count", 0) <= 4
            assert warm.get("prefix_checkpoint_bytes", 0) <= 256 * 1024 * 1024
            result = {"name": name, "prompt_tokens": cold["response"]["usage"]["prompt_tokens"], "cached_tokens": cached,
                      "cold_prefill_ms": cold["timings"]["prefill_ms"], "warm_prefill_ms": warm["timings"]["prefill_ms"],
                      "cold_ttft_ms": cold["timings"]["ttft_ms"], "warm_ttft_ms": warm["timings"]["ttft_ms"],
                      "restore_ms": warm["timings"]["cache_restore_ms"], "output_equal": output_equal}
            if not output_equal:
                result["cached_output"] = warm["response"]["choices"]
                result["cold_output"] = cold["response"]["choices"]
            report["checks"].append(result)
            save()
            status = "PASS" if output_equal else "OUTPUT_VARIATION"
            print(f"{status} {args.mode} {name} cached={cached}/{result['prompt_tokens']} prefill={result['cold_prefill_ms']:.2f}->{result['warm_prefill_ms']:.2f} ms", flush=True)
            return cold["response"]["choices"][0].get("message")

        system = {"role": "system", "content": "你是简洁的中文助手。请根据用户提供的信息回答，尽量只用一句话。"}
        first = {"role": "user", "content": "我叫小林，住在杭州，从事嵌入式软件开发。请记住这些信息。"}
        chat1 = {"messages": [system, first], "enable_thinking": False}
        answer1 = compare("repeat-prompt", chat1, chat1)
        chat2 = {"messages": [system, first, answer1, {"role": "user", "content": "我住在哪个城市？"}], "enable_thinking": False}
        answer2 = compare("chat-turn-2", chat1, chat2)
        chat3 = {"messages": [*chat2["messages"], answer2, {"role": "user", "content": "我的工作是什么？"}], "enable_thinking": False}
        compare("chat-turn-3", chat2, chat3)
        compare("regenerate-earlier-turn", chat3, chat1)
        changed = {"messages": [system, {"role": "user", "content": "我叫小张，住在南京，从事硬件开发。我的工作是什么？"}], "enable_thinking": False}
        compare("edit-history", chat3, changed)
        request("cache-off-chat", {**chat1, "cache_prompt": False})
        enabled = request("cache-on-edited-chat", {**changed, "cache_prompt": True})
        assert enabled["response"]["usage"]["prompt_tokens_details"]["cached_tokens"] > 0
        report["checks"].append({"name": "cache-off-retains-user-boundary", "cached_tokens": enabled["response"]["usage"]["prompt_tokens_details"]["cached_tokens"]})
        other = {"messages": [{"role": "system", "content": "You are an English assistant."}, first], "enable_thinking": False}
        compare("changed-system", chat3, other, hit=None)
        assert report["checks"][-1]["cached_tokens"] <= 4

        prefix = "以下是实验记录，请根据最后的问题回答。\n" + "RK3588 有四个大核和四个小核，NPU 用于矩阵乘法，CPU 保存递归状态。\n" * 14
        a = {"prompt": prefix + "问题：NPU 用于什么？\n答案："}
        b = {"prompt": prefix + "问题：CPU 保存什么？\n答案："}
        c = {"prompt": prefix + "问题：处理器有几个大核？\n答案："}
        compare("shared-prefix-a-b", a, b)
        compare("shared-prefix-b-c", b, c)
        compare("branch-back-to-a", c, a)
        compare("shortened-prompt", a, {"prompt": prefix[:len(prefix) // 2]}, hit=False)

        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=180)
        connection.request("POST", "/v1/completions", json.dumps({"model": model_id, "prompt": prefix + "请详细解释：", "temperature": 0, "max_tokens": 128, "stream": True}), {"Content-Type": "application/json"})
        response = connection.getresponse()
        assert response.status == 200
        while True:
            line = response.readline()
            assert line, "stream ended before cancellation"
            if line.startswith(b"data: ") and b'"text":"' in line and b'"text":""' not in line:
                break
        response.close()
        connection.close()
        time.sleep(0.5)
        recovered = request("cancel-recovery", b)
        assert recovered["response"]["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
        clear()
        cold = request("cancel-recovery-cold", b)
        assert signature(recovered) == signature(cold)
        report["checks"].append({"name": "cancel-recovery", "output_equal": True})
        report["status"] = "PASS" if all(c["output_equal"] for c in report["checks"] if "output_equal" in c) else "OUTPUT_VARIATION"
        save()
    finally:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=10)
        log.close()


if __name__ == "__main__":
    main()
