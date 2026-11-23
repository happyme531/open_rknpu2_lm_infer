#!/usr/bin/env python3
"""Exercise and measure a fixed-slot server on RK3588 with local HTTP/SSE clients."""

import argparse
import concurrent.futures
import contextlib
import http.client
import json
import os
import signal
import subprocess
import time
from pathlib import Path


def request(prompt, tokens, seed=0, **extra):
    return dict(
        prompt=prompt, max_tokens=tokens, seed=seed, temperature=0.7, top_k=20,
        stream=True, stream_options={"include_usage": True}, **extra,
    )


def connection(port):
    return http.client.HTTPConnection("127.0.0.1", port, timeout=90)


def health(port):
    with contextlib.closing(connection(port)) as conn:
        conn.request("GET", "/health")
        response = conn.getresponse()
        assert response.status == 200, response.status
        return json.load(response)


def stream(port, body, disconnect=False, read_delay=0):
    start = time.monotonic()
    with contextlib.closing(connection(port)) as conn:
        conn.request("POST", "/v1/completions", json.dumps(body),
                     {"Content-Type": "application/json"})
        response = conn.getresponse()
        if response.status != 200:
            return {"status": response.status, "body": response.read().decode()}
        text, times = [], []
        reason = usage = None
        for line in response:
            if not line.startswith(b"data: "):
                continue
            data = line[6:].strip()
            if data == b"[DONE]":
                assert reason and usage, (reason, usage)
                return {
                    "status": 200, "text": "".join(text), "finish_reason": reason,
                    "usage": usage, "ttft_ms": (times[0] - start) * 1000 if times else None,
                    "itl_ms": [(b - a) * 1000 for a, b in zip(times, times[1:])],
                    "begin": start, "end": time.monotonic(),
                }
            chunk = json.loads(data)
            assert "error" not in chunk, chunk
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk["choices"]:
                reason = choice["finish_reason"] or reason
                if choice["text"]:
                    times.append(time.monotonic())
                    text.append(choice["text"])
                    if disconnect:
                        return {"status": 200, "disconnected": True}
                    if read_delay:
                        time.sleep(read_delay)
    raise AssertionError("stream closed before [DONE]")


def wait_health(port, predicate, seconds=30):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        try:
            status = health(port)
            if predicate(status):
                return status
        except OSError:
            pass
        time.sleep(0.025)
    raise RuntimeError(f"server did not reach requested state on port {port}")


def percentiles(values):
    values = sorted(values)
    if not values:
        return {}
    return {key: values[round(fraction * (len(values) - 1))]
            for key, fraction in (("p50", .50), ("p95", .95), ("max", 1))}


def measure(results):
    assert all(result["status"] == 200 for result in results), results
    tokens = sum(result["usage"]["completion_tokens"] for result in results)
    return {
        "tokens": tokens,
        "tokens_per_s": tokens / (max(r["end"] for r in results) - min(r["begin"] for r in results)),
        "ttft_ms": percentiles([r["ttft_ms"] for r in results if r["ttft_ms"] is not None]),
        "itl_ms": percentiles([ms for r in results for ms in r["itl_ms"]]),
        "itl_samples": sum(len(r["itl_ms"]) for r in results),
    }


def run_workload(args, port):
    names = ["Venus", "Saturn", "Mercury", "Neptune", "Jupiter", "Mars", "Earth", "Uranus"]
    prompts = [
        (f"Provide a long detailed history of research about {name}. " * (2 + 2 * i))
        + f"An introduction to the study of {name} is"
        for i, name in enumerate(names)
    ]
    model = health(port)["model"]

    def perform(body, disconnect=False, read_delay=0):
        return stream(port, {"model": model, **body}, disconnect, read_delay)

    with concurrent.futures.ThreadPoolExecutor(max_workers=9) as pool:
        warm = [pool.submit(perform, request(prompt, 4, i))
                for i, prompt in enumerate(prompts[:4])]
        assert all(job.result()["status"] == 200 for job in warm)
        initial = wait_health(port, lambda s: not s["active"] and not s["waiting"])

        deterministic = request([1000], 16, 987)
        reference = perform(deterministic)
        assert reference["status"] == 200 and reference["text"], reference
        stop = reference["text"][: min(6, len(reference["text"]))]
        assert stop
        pair = [
            pool.submit(perform, dict(deterministic, stop=[stop])),
            pool.submit(perform, deterministic),
        ]
        stopped, unchanged = (job.result() for job in pair)
        assert stopped["finish_reason"] == "stop" and stopped["text"] == "", stopped
        assert unchanged["text"] == reference["text"], unchanged
        different = perform(dict(deterministic, seed=321))
        assert reference["text"] != different["text"], "seed fixture lacked diversity"

        def burst(tokens, seed):
            jobs = [pool.submit(perform, request(prompts[i], tokens, seed + i))
                    for i in range(4)]
            wait_health(port, lambda s: s["active"] == 4)
            jobs.extend(pool.submit(perform, request(prompts[i], tokens, seed + i))
                        for i in range(4, 8))
            return [job.result() for job in jobs]

        sampled = measure(burst(args.tokens, 0))
        drained = wait_health(port, lambda s: not s["active"] and not s["waiting"])
        assert drained["batch"]["mixed_step_ms"] > initial["batch"]["mixed_step_ms"], drained

        slow = pool.submit(perform, request(prompts[1], 48, 16), False, 0.25)
        gone = pool.submit(perform, request(prompts[0], 96, 17), True)
        fast = pool.submit(perform, request(prompts[2], 16, 18))
        assert gone.result()["disconnected"]
        fast_result, slow_result = fast.result(), slow.result()
        assert fast_result["status"] == 200 and slow_result["status"] == 200
        assert fast_result["end"] < slow_result["end"], (fast_result, slow_result)
        wait_health(port, lambda s: not s["active"] and not s["waiting"])

        # Verify that all five queued requests are accepted while the slots are busy.
        active = [pool.submit(perform, request(prompts[i], 64, 41 + i))
                  for i in range(4)]
        wait_health(port, lambda s: s["active"] == 4)
        waiting = [pool.submit(perform, request(prompts[i], 4, 133 + i))
                   for i in range(5)]
        wait_health(port, lambda s: s["waiting"] == 5)
        queued = [job.result() for job in waiting]
        assert all(result["status"] == 200 for result in queued), queued
        assert all(job.result()["status"] == 200 for job in active)

        rss = []
        for _ in range(args.cycles):
            results = burst(8, 410)
            assert all(result["status"] == 200 for result in results), results
            status = wait_health(port, lambda s: not s["active"] and not s["waiting"])
            assert status["kv_payload_bytes"] == initial["kv_payload_bytes"], status
            rss.append(status["rss_kib"])
        final = wait_health(port, lambda s: not s["active"] and not s["waiting"])

    if len(rss) >= 4:
        assert max(rss[-3:]) - min(rss[-3:]) < 16 * 1024, rss
    return {"load": sampled, "initial_health": initial, "final_health": final,
            "rss_kib": rss, "queued_requests_completed": len(queued), "seed_and_stop": "pass",
            "midstream_disconnect": "pass", "slow_reader": "pass"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--tune-cache", type=Path)
    parser.add_argument("--port", type=int, default=18919)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--cycles", type=int, default=8)
    parser.add_argument("--batches", nargs="+", type=int, default=[32, 64, 128])
    args = parser.parse_args()
    report = {}
    for batch in args.batches:
        log = args.report.with_name(f"{args.report.stem}-m{batch}.log")
        command = [
            str(args.binary), "-m", str(args.model), "--ctx-size", "1024",
            "-np", "4", "--ubatch-size", str(batch),
        ]
        command += ["--tune-cache", str(args.tune_cache)] if args.tune_cache else ["--no-tune"]
        command += ["serve", "--port", str(args.port)]
        with log.open("w") as writer:
            env = dict(os.environ, RUST_LOG="info")
            server = subprocess.Popen(command, stdout=writer, stderr=subprocess.STDOUT, env=env)
            try:
                def ready(status):
                    return status["parallel"] == 4

                wait_health(args.port, ready, 240)
                assert server.poll() is None, log
                report[str(batch)] = run_workload(args, args.port)
                print(json.dumps({"batch": batch, **report[str(batch)]["load"]}), flush=True)
                args.report.write_text(json.dumps(report, indent=2) + "\n")
            finally:
                if server.poll() is None:
                    server.send_signal(signal.SIGINT)
                    try:
                        server.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        server.terminate()
                        server.wait(timeout=10)


if __name__ == "__main__":
    main()
