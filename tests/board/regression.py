#!/usr/bin/env python3
"""Run RK3588 runtime regressions and exact baseline comparisons from one config."""
import argparse
import array
from concurrent.futures import ThreadPoolExecutor
import filecmp
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def check_test_output(text):
    results = re.findall(r"test result: ok\. (\d+) passed;", text)
    if not results or sum(map(int, results)) == 0:
        raise RuntimeError("test filter did not execute any passing tests")


def compare_files(left, right, floats=True):
    size = left.stat().st_size
    if not size or (floats and size % 4):
        raise RuntimeError(f"invalid output length: {left} ({size} bytes)")
    if not filecmp.cmp(left, right, shallow=False):
        raise RuntimeError(f"outputs differ: {left.name}, {right.name}")
    if floats:
        values = array.array("f")
        with left.open("rb") as source:
            values.fromfile(source, size // 4)
        if not all(map(math.isfinite, values)):
            raise RuntimeError(f"non-finite logits: {left}")
    return {"equal": True, "bytes": size}


def scp_host(destination):
    if ":" in destination and "[" not in destination:
        user, sep, host = destination.rpartition("@")
        return (user + sep if sep else "") + "[" + host + "]"
    return destination


class Suite:
    def __init__(self, config, output, resume):
        self.config, self.output, self.resume = config, output, resume
        self.output.mkdir(parents=True, exist_ok=True)
        self.report = {"passed": False, "cases": {}, "comparisons": {}}
        self.previous = {}
        if resume and (output / "summary.json").exists():
            self.previous = json.loads((output / "summary.json").read_text())
        self.binaries = {k: str(Path(v).resolve()) for k, v in config.get("binaries", {}).items()}
        self.save()

    def save(self):
        write_json(self.output / "summary.json", self.report)

    def build(self):
        """Snapshot every executable before building another checkout."""
        sources = {"current": str(ROOT), **self.config.get("sources", {})}
        needed = {"current"} | {m["baseline"] for m in self.config.get("models", {}).values() if m.get("baseline")}
        bindir = self.output / "bin"
        bindir.mkdir(exist_ok=True)
        for label in sorted(needed):
            if label in self.binaries and (label != "current" or "tests" in self.binaries):
                continue
            if label not in sources:
                raise ValueError(f"provide binaries.{label} or sources.{label}")
            source = Path(sources[label]).resolve()
            modes = [(label, ["build", "--bin", "open_rknpu2_lm_infer"])]
            if label == "current":
                modes.append(("tests", ["test", "--lib", "--no-run"]))
            for name, mode in modes:
                if name in self.binaries:
                    continue
                command = ["cargo", *mode, "--release", "--locked", "--manifest-path", str(source / "Cargo.toml"),
                           "--message-format=json-render-diagnostics"]
                print("BUILD", name, flush=True)
                env = os.environ.copy()
                env["RK_NPU_MATMUL_ROOT"] = str(Path(sources["current"]).resolve() / "native/rk_npu_matmul")
                if self.config.get("target_dir"):
                    env["CARGO_TARGET_DIR"] = self.config["target_dir"]
                with (self.output / f"build-{name}.jsonl").open("w") as out, \
                     (self.output / f"build-{name}.log").open("w") as log:
                    subprocess.run(command, cwd=source, env=env, stdout=out, stderr=log, check=True)
                executables = []
                for line in (self.output / f"build-{name}.jsonl").read_text().splitlines():
                    artifact = json.loads(line)
                    if artifact.get("reason") == "compiler-artifact" and artifact.get("executable"):
                        executables.append(artifact["executable"])
                if len(executables) != 1:
                    raise RuntimeError(f"expected one {name} executable, found {executables}")
                target = bindir / name
                shutil.copy2(executables[0], target)
                self.binaries[name] = str(target)
        for name, path in self.binaries.items():
            if not Path(path).is_file():
                raise RuntimeError(f"missing {name} executable: {path}")
        self.report["binaries"] = self.binaries

    def run(self, name, command, env=None, test=False):
        env = {k: str(v) for k, v in (env or {}).items()}
        # Reuse results only for the same invocation, executable and input config.
        binary = Path(command[0]).stat()
        signature = {"command": command, "environment": env, "config": self.config,
                     "binary": {"size": binary.st_size, "mtime_ns": binary.st_mtime_ns},
                     "inputs": {key: {"size": Path(self.config[key]).stat().st_size,
                                      "mtime_ns": Path(self.config[key]).stat().st_mtime_ns}
                                for key in ["tokens", "continuation"] if key in self.config}}
        previous = self.previous.get("cases", {}).get(name, {})
        if self.resume and previous.get("passed") and previous.get("signature") == signature:
            if all((self.output / f"{name}.{suffix}").exists() for suffix in ["log", "stdout", "clocks.json"]):
                self.report["cases"][name] = previous
                self.save()
                print("REUSE", name, flush=True)
                return
        archive = None
        if previous and not previous.get("passed"):
            archive = self.output / "attempts" / name / str(time.time_ns())
            archive.mkdir(parents=True)
            write_json(archive / "result.json", previous)
            for path in self.output.glob(name + ".*"):
                path.rename(archive / path.name)
        row = {"passed": False, "signature": signature}
        if archive:
            row["previous_attempt"] = str(archive.relative_to(self.output))
        self.report["cases"][name] = row
        self.save()
        runner = [sys.executable, str(ROOT / "scripts/run_board.py"), "--wait-lock",
                  "--timeout", str(self.config.get("timeout_seconds", 600)),
                  "--report", str(self.output / f"{name}.clocks.json")]
        for key in ["cpu_khz", "skip_dmc"]:
            value = self.config.get(key)
            if value:
                runner += ["--" + key.replace("_", "-")]
                if key == "cpu_khz":
                    runner.append(str(value))
        print("RUN", name, flush=True)
        start = time.monotonic()
        with (self.output / f"{name}.stdout").open("w") as out, \
             (self.output / f"{name}.log").open("w") as log:
            result = subprocess.run([*runner, "--", *command], env={**os.environ, **env}, stdout=out, stderr=log)
        row.update(exit_code=result.returncode, elapsed_seconds=time.monotonic() - start)
        clock_path = self.output / f"{name}.clocks.json"
        clocks = json.loads(clock_path.read_text()) if clock_path.exists() else {}
        row["clocks_restored"] = clocks.get("restore_errors") == []
        row["timed_out"] = clocks.get("timed_out", False)
        self.save()
        if result.returncode or not row["clocks_restored"] or clocks.get("exit_code") != 0:
            raise RuntimeError(f"{name} failed; see {name}.log and {name}.clocks.json")
        if test:
            check_test_output((self.output / f"{name}.stdout").read_text())
        row["passed"] = True
        self.save()
        print("PASS", name, flush=True)

    def compare(self, name, left, right, floats=True):
        row = {"equal": False}
        self.report["comparisons"][name] = row
        self.save()
        row.update(compare_files(self.output / left, self.output / right, floats))
        self.save()
        print("EXACT", name, flush=True)

    def test(self, name, prefix, env=None):
        self.run(name, [self.binaries["tests"], prefix, "--ignored", "--nocapture", "--test-threads=1"], env, test=True)

    def evaluate(self, binary, family, model, suffix, flags):
        name = f"{binary}-{family}-{suffix}"
        self.run(name, [self.binaries[binary], "--model", model["path"], "--ctx-size", "640",
                       "--ubatch-size", "128", "--no-tune", *flags, "eval",
                       "--tokens-file", self.config["tokens"], "--continuation-file", self.config["continuation"],
                       "--dump-logits", str(self.output / f"{name}.f32"),
                       "--report", str(self.output / f"{name}.json")])
        return f"{name}.f32"

    def paired_eval(self, family, model, suffix, flags):
        actual = self.evaluate("current", family, model, suffix, flags)
        if model.get("baseline"):
            expected = self.evaluate(model["baseline"], family, model, suffix, flags)
            self.compare(f"{family}-{suffix}-baseline", expected, actual)
        return actual

    def dense(self, family, model):
        outputs = []
        for lanes in [1, 2]:
            outputs.append(self.paired_eval(family, model, f"lane{lanes}", ["--prefill-ubatches", str(lanes)]))
        self.compare(f"{family}-lanes", *outputs)
        if family == "qwen3":
            compressed = self.paired_eval(family, model, "compressed", ["--compress-w8a8"])
            self.compare("qwen3-compression-lossless", outputs[0], compressed)
        else:
            self.test("qwen35-dual", "model::qwen35::execution::tests::paired_prefill_preserves_recurrence_features_and_decode",
                      {"RKNPU_MTP_MODEL": model["path"]})
            modes = [("none", 1), ("none", 2), ("mtp", 2)]
            if model.get("draft"):
                modes.append(("dflash", 2))
            for kind, lanes in modes:
                env = {"RKNPU_MTP_MODEL": model["path"], "RKNPU_PREFIX_SPEC": kind, "RKNPU_PREFILL_UBATCHES": str(lanes)}
                if kind == "dflash":
                    env["RKNPU_PREFIX_DRAFT"] = model["draft"]
                self.test(f"qwen35-prefix-{kind}-lane{lanes}",
                          "model::qwen35::cache::tests::board::prefix_restore_matches_all_logits_and_draft_state", env)

    def ling(self, model):
        outputs = []
        for storage in ["resident", "stream"]:
            flags = ["--expert-store", model["store"], "--expert-storage", storage]
            outputs.append(self.paired_eval("ling3tiny", model, storage, flags))
            env = {"LING_SSM_MODEL": model["path"], "LING_SSM_EXPERT_STORE": model["store"]}
            if storage == "stream":
                env["LING_SSM_STREAM"] = "1"
            self.test(f"ling-prefix-{storage}",
                      "model::ling3tiny::cache::tests::ling_prefix_restore_matches_replay_and_preserves_kda_convolution_and_mla", env)
        compressed = self.evaluate("current", "ling3tiny", model, "compressed",
                                   ["--expert-store", model["store"], "--compress-w8a8"])
        self.compare("ling-compression-lossless", outputs[0], compressed)

    def mini(self, model):
        flags = ["--expert-store", model["store"], "--expert-storage", "stream"]
        plain = self.paired_eval("minicpmv47", model, "stream", flags)
        compressed = self.evaluate("current", "minicpmv47", model, "compressed", [*flags, "--compress-w8a8"])
        self.compare("mini-compression-lossless", plain, compressed)
        self.test("mini-prefix", "model::qwen35::cache::tests::board::prefix_restore_matches_all_logits_and_draft_state",
                  {"RKNPU_MTP_MODEL": model["path"], "RKNPU_PREFIX_EXPERT_STORE": model["store"], "RKNPU_PREFIX_STREAM": "1"})
        for name, prefetch, prefix in [("single", "1", "minicpm_prefetch_timeline"),
                                     ("batched", "1", "minicpm_batched_prefetch_timeline"),
                                     ("append-off", "0", "minicpm_prefetch_append_prompt_timeline"),
                                     ("append-on", "1", "minicpm_prefetch_append_prompt_timeline")]:
            self.test(name, "model::qwen35::paging_tests::" + prefix,
                      {"MINICPM_TEST_MODEL": model["path"], "MINICPM_TEST_STORE": model["store"],
                       "MINICPM_TEST_WINDOWS": model["windows"], "MINICPM_TEST_REPORT": str(self.output / f"{name}.json"),
                       "MINICPM_TEST_PREFETCH": prefetch})
        self.compare("mini-append-prefetch", "append-off.logits.f32", "append-on.logits.f32")
        modes = ["ar"] + (["dspark"] if model.get("draft") else [])
        for mode in modes:
            for binary in ["current"] + ([model["baseline"]] if model.get("baseline") else []):
                spec = ["--spec-type", "draft-dspark", "--spec-draft-model", model["draft"], "--spec-draft-n-max", "3"] if mode == "dspark" else []
                self.run(f"{binary}-mini-{mode}", [self.binaries[binary], "--model", model["path"],
                         "--ctx-size", "1024", "--ubatch-size", "128", "--no-tune", *flags,
                         "--expert-cache-percent", "15.625", *spec, "run", "--file", model["prompt"],
                         "--n-predict", "33", "--temp", "0", "--top-k", "0", "--top-p", "1", "--seed", "42", "--stats"])
            if model.get("baseline"):
                self.compare(f"mini-{mode}-baseline", f"{model['baseline']}-mini-{mode}.stdout", f"current-mini-{mode}.stdout", False)
        if "dspark" in modes:
            self.compare("mini-ar-dspark", "current-mini-ar.stdout", "current-mini-dspark.stdout", False)

    def execute(self):
        self.build()
        if self.config.get("operators", True):
            for name, prefix in [
                ("gqa", "npu::tests::native_gqa_append_and_layout"),
                ("gdn", "layers::gated_delta::state::npu_tests::npu_prefill_threshold_state_handoff_and_layer_reuse"),
                ("mla", "npu::mla::tests::"), ("tf32", "npu::tf32::tests::"),
                ("execution-lanes", "npu::lane::tests::independent_execution_lanes_share_weights")]:
                self.test(name, prefix)
        for family, model in self.config.get("models", {}).items():
            if family in ["qwen3", "qwen35"]:
                self.dense(family, model)
            elif family == "ling3tiny":
                self.ling(model)
            elif family == "minicpmv47":
                self.mini(model)
            else:
                raise ValueError(f"unsupported family: {family}")
        self.report["passed"] = True
        self.save()


def remote_boards(config, output, resume):
    """Deploy the two Python runners, then run independent boards concurrently."""
    output.mkdir(parents=True, exist_ok=True)
    aggregate = output / "boards.json"
    write_json(aggregate, {"passed": False, "boards": {}})
    uploads = {}

    def board(name, node, build_only=False):
        row = {"passed": False}
        try:
            remote = node["output"].rstrip("/")
            runner = remote + "/runner"
            ssh = ["ssh", "-o", "BatchMode=yes", node["ssh"]]
            subprocess.run([*ssh, shlex.join(["mkdir", "-p", runner + "/tests/board", runner + "/scripts", remote + "/bin"])], check=True)
            worker = dict(node["config"])
            if node.get("source"):
                worker["sources"] = {"current": node["source"], **worker.get("sources", {})}
            destination = scp_host(node["ssh"])
            if uploads:
                worker["binaries"] = dict(worker.get("binaries", {}))
                for label, path in uploads.items():
                    target = remote + "/bin/" + label
                    subprocess.run(["scp", "-p", "-q", "-o", "BatchMode=yes", str(path), destination + ":" + target], check=True)
                    worker["binaries"][label] = target
            local_config = output / f"{name}.config.json"
            write_json(local_config, worker)
            for source, target in [(Path(__file__), runner + "/tests/board/regression.py"),
                                   (ROOT / "scripts/run_board.py", runner + "/scripts/run_board.py"),
                                   (local_config, runner + "/config.json")]:
                subprocess.run(["scp", "-q", "-o", "BatchMode=yes", str(source), destination + ":" + target], check=True)
            command = ["python3", runner + "/tests/board/regression.py", "--config", runner + "/config.json", "--output", remote]
            if node.get("sudo"):
                command = ["sudo", "-n", *command]
            if resume:
                command.append("--resume")
            if build_only:
                command.append("--build-only")
            print("BOARD", name, node["ssh"], flush=True)
            with (output / f"{name}.log").open("w") as log:
                result = subprocess.run([*ssh, shlex.join(command)], stdout=log, stderr=subprocess.STDOUT)
            report = subprocess.run([*ssh, shlex.join(["cat", remote + "/summary.json"])], capture_output=True, text=True, check=True)
            row.update(json.loads(report.stdout))
            row["passed"] = result.returncode == 0 and row.get("passed", False)
            write_json(output / f"{name}.json", row)
        except (OSError, ValueError, subprocess.CalledProcessError) as error:
            row["error"] = str(error)
        print("PASS" if row["passed"] else "FAIL", name, flush=True)
        return name, row

    if config.get("build_on"):
        name = config["build_on"]
        node = config["boards"][name]
        build_node = {**node, "output": node["output"].rstrip("/") + "/build"}
        _, built = board(name + "-build", build_node, build_only=True)
        if not built["passed"]:
            write_json(aggregate, {"passed": False, "build": built})
            return 1
        bindir = output / "artifacts"
        bindir.mkdir(exist_ok=True)
        for label, path in built["binaries"].items():
            target = bindir / label
            subprocess.run(["scp", "-p", "-q", "-o", "BatchMode=yes", scp_host(node["ssh"]) + ":" + path, str(target)], check=True)
            uploads[label] = target

    results = {}
    with ThreadPoolExecutor(max_workers=len(config["boards"])) as pool:
        for name, row in pool.map(lambda pair: board(*pair), config["boards"].items()):
            results[name] = row
    passed = all(row["passed"] for row in results.values())
    write_json(aggregate, {"passed": passed, "boards": results})
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="reuse passed cases with unchanged invocation and executable")
    parser.add_argument("--build-only", action="store_true", help="prepare and report binaries without running cases")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config.get("boards"):
        return remote_boards(config, args.output.resolve(), args.resume)
    if not config.get("models") and not config.get("operators", True):
        parser.error("config must select models or operator tests")
    suite = Suite(config, args.output.resolve(), args.resume)
    try:
        if args.build_only:
            suite.build()
            suite.report.update(passed=True, build_only=True)
            suite.save()
        else:
            suite.execute()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        suite.report["error"] = str(error)
        suite.save()
        print("FAIL", error, file=sys.stderr)
        return 1
    print("PASS", suite.output / "summary.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
