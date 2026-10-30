"""Check llama.cpp-compatible cache controls through the production server."""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--mode", choices=("ar", "mtp", "dflash"), default="ar")
    parser.add_argument("--draft-model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", action="append", dest="selected_cases", help="Run only named startup configurations")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = [
        ("default", {}, [], True, 4),
        ("disable-flag", {"LLAMA_ARG_CACHE_PROMPT": "1"}, ["--no-cache-prompt"], False, 4),
    ]
    if args.mode == "ar":
        cases += [
            ("positive-env-off", {"LLAMA_ARG_CACHE_PROMPT": "disabled"}, [], False, 4),
            ("legacy-negative-env", {"LLAMA_ARG_NO_CACHE_PROMPT": "0", "LLAMA_ARG_CACHE_PROMPT": "1"}, [], False, 4),
            ("last-flag-on", {"LLAMA_ARG_NO_CACHE_PROMPT": "", "LLAMA_ARG_CACHE_PROMPT": "invalid"},
             ["--no-cache-prompt", "--cache-prompt", "-ctxcp", "4", "-cms", "0"], True, 4),
            ("last-flag-off", {}, ["--cache-prompt", "--no-cache-prompt", "--swa-checkpoints", "4"], False, 4),
            ("checkpoints-env-off", {"LLAMA_ARG_CTX_CHECKPOINTS": "0"}, [], True, 0),
        ]
    if args.selected_cases:
        names = {case[0] for case in cases}
        assert set(args.selected_cases) <= names, set(args.selected_cases) - names
        cases = [case for case in cases if case[0] in args.selected_cases]
    report = {"mode": args.mode, "model": str(args.model), "cases": [], "status": "RUNNING"}

    def save():
        (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    for name, extra_env, flags, enabled, capacity in cases:
        with socket.socket() as selection:
            selection.bind(("127.0.0.1", 0))
            port = selection.getsockname()[1]
        env = {k: v for k, v in os.environ.items() if not k.startswith("LLAMA_ARG_")}
        env.pop("OPEN_RKNPU_API_KEY", None)
        env.update(LLAMA_ARG_CTX_SIZE="512", LLAMA_ARG_UBATCH="64", LLAMA_ARG_CTX_CHECKPOINTS="4",
                   LLAMA_ARG_CHECKPOINT_MIN_SPACING_NT="0", RUST_LOG="info", NO_COLOR="1")
        env.update(extra_env)
        command = [str(args.binary), "-m", str(args.model), "--no-tune"]
        if args.mode != "ar":
            command += ["--spec-type", "draft-" + args.mode, "--spec-draft-n-max", "3"]
        if args.mode == "dflash":
            assert args.draft_model is not None
            command += ["-md", str(args.draft_model)]
        command += ["serve", "--host", "127.0.0.1", "--port", str(port), *flags]
        log_path = args.output / (name + ".log")
        case = {"name": name, "default_enabled": enabled, "checkpoint_capacity": capacity,
                "command": command, "cache_environment": {k: v for k, v in env.items() if k.startswith("LLAMA_ARG_")},
                "requests": []}
        report["cases"].append(case)
        with log_path.open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=log, env=env)
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

                def request(label, endpoint, payload, override="missing", expect_status=200):
                    body = {"model": model_id, "temperature": 0, "seed": 42,
                            "max_tokens": 1 if args.mode == "ar" else 4, **payload}
                    if override != "missing":
                        body["cache_prompt"] = override
                    offset = log_path.stat().st_size
                    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/{endpoint}",
                                                 json.dumps(body).encode(), {"Content-Type": "application/json"})
                    try:
                        with urllib.request.urlopen(req, timeout=180) as response:
                            status, value = response.status, json.load(response)
                    except urllib.error.HTTPError as error:
                        status, value = error.code, json.load(error)
                    assert status == expect_status, (name, label, status, value)
                    entry = {"label": label, "endpoint": endpoint, "request": body, "status": status, "response": value}
                    if status == 200:
                        with log_path.open() as stream:
                            stream.seek(offset)
                            lines = re.sub(r"\x1b\[[0-9;]*m", "", stream.read())
                        line = next(line for line in lines.splitlines() if "generation finished" in line)
                        entry["timings"] = json.JSONDecoder().raw_decode(line.split("timings=", 1)[1])[0]
                        for field in ("prefix_checkpoint_count", "prefix_checkpoint_bytes", "tune_calls"):
                            match = re.search(r"\b" + field + r"=([^ ]+)", line)
                            entry[field] = json.loads(match.group(1))
                        assert entry["prefix_checkpoint_count"] <= capacity
                        assert entry["tune_calls"] == 0
                        entry["cached_tokens"] = value["usage"]["prompt_tokens_details"]["cached_tokens"]
                    case["requests"].append(entry)
                    save()
                    return entry

                for endpoint, payload in [
                    ("completions", {"prompt": "RK3588 is an embedded processor. The NPU computes matrix products. Explain briefly:"}),
                    ("chat/completions", {"messages": [
                        {"role": "system", "content": "你是简洁的中文助手。"},
                        {"role": "user", "content": "我叫小林，住在杭州，从事嵌入式软件开发。请记住这些信息。"}], "enable_thinking": False}),
                ]:
                    source = request("source", endpoint, payload, False)
                    assert source["cached_tokens"] == 0
                    total = source["response"]["usage"]["prompt_tokens"]
                    assert total > 4
                    for override, effective in [("missing", enabled), (False, False), (True, True), (None, enabled)]:
                        result = request("override-" + str(override), endpoint, payload, override)
                        expected = total - 4 if effective and capacity else 0
                        assert result["cached_tokens"] == expected, (name, endpoint, override, result["cached_tokens"], expected)
                        result["output_equal_to_cold"] = result["response"]["choices"] == source["response"]["choices"]
                    invalid = request("invalid-boolean", endpoint, payload, "false", 400)
                    assert invalid["response"]["error"]["param"] == "cache_prompt"

                # The earlier tail checkpoint proves LLAMA_ARG_UBATCH=64
                # reached the model: 200 - (64 + 4) = 132, rather than 68.
                if capacity:
                    request("ubatch-source", "completions", {"prompt": [1] * 200}, False)
                    branch = request("ubatch-branch", "completions", {"prompt": [1] * 184 + [2] * 16}, True)
                    assert branch["cached_tokens"] == 132, branch["cached_tokens"]
                # Prompt alone exceeds LLAMA_ARG_CTX_SIZE=512.
                request("context-env-limit", "completions", {"prompt": [1] * 513}, expect_status=400)
                case["status"] = "PASS"
                print(f"PASS {args.mode} {name} default={enabled} checkpoints={capacity}", flush=True)
            finally:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    process.wait(timeout=10)
                case["server_exit_code"] = process.returncode
                save()
        assert case["server_exit_code"] == 0, case["server_exit_code"]
    report["status"] = "PASS"
    report["output_variations"] = sum(
        entry.get("output_equal_to_cold") is False
        for case in report["cases"] for entry in case["requests"]
    )
    save()


if __name__ == "__main__":
    main()
