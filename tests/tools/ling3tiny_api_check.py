"""Check two real Ling conversations through the standard API in one model load."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.request


def request(url, data=None):
    body = None if data is None else json.dumps(data).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=120) as response:
        return json.load(response)


def check(root):
    models = request(root + "/models")
    results = []
    for prompt, keyword in [
        ("用一句话解释为什么天空是蓝色的。", "散射"),
        ("小明有12个苹果，送出5个，又买了8个，现在有几个？请简短回答。", "15"),
    ]:
        result = request(root + "/chat/completions", {
            "model": models["data"][0]["id"], "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "max_tokens": 48, "stream": False, "enable_thinking": False,
        })
        answer = result["choices"][0]["message"]["content"]
        results.append({"prompt": prompt, "answer": answer, "response": result,
                        "passed": keyword in answer})
    return {"model": models["data"][0]["id"], "cases": results,
            "passed": all(case["passed"] for case in results)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expert-store", type=Path)
    args = parser.parse_args()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = os.environ.copy()
    env.pop("LLM_INFER_LOCK_FD", None)
    command = [str(args.binary), "--model", str(args.model), "--ctx-size", "256",
               "--ubatch-size", "128"]
    if args.expert_store:
        command += ["--expert-store", str(args.expert_store), "--expert-buffered-io"]
    command += ["serve", "--port", str(port)]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.with_suffix(".server.log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        try:
            root = f"http://127.0.0.1:{port}/v1"
            deadline = time.monotonic() + 300
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"server exited: {process.returncode}")
                try:
                    models = request(root + "/models")
                    break
                except (OSError, ValueError):
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Ling API startup")
                    time.sleep(1)
            report = check(root)
            args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            print(json.dumps(report, ensure_ascii=False, indent=2))
            if not report["passed"]:
                raise RuntimeError("Ling generation check failed")
        finally:
            process.terminate()
            process.wait(timeout=30)


if __name__ == "__main__":
    main()
