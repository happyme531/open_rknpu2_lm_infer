#!/usr/bin/env python3
"""Run one board command under the shared submit lock and fixed clocks."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time


TARGETS = {
    Path("/sys/class/devfreq/fdab0000.npu"): ("min_freq", "max_freq", "cur_freq", 1000000000),
    Path("/sys/class/devfreq/dmc"): ("min_freq", "max_freq", "cur_freq", 2112000000),
    Path("/sys/devices/system/cpu/cpufreq/policy4"): ("scaling_min_freq", "scaling_max_freq", "scaling_cur_freq", 2400000),
    Path("/sys/devices/system/cpu/cpufreq/policy6"): ("scaling_min_freq", "scaling_max_freq", "scaling_cur_freq", 2400000),
}


def clock_targets(cpu_khz, skip_dmc):
    result = {}
    for root, (minimum, maximum, current, target) in TARGETS.items():
        if skip_dmc and root.name == "dmc":
            continue
        if minimum.startswith("scaling"):
            target = cpu_khz
        result[root] = (minimum, maximum, current, target)
    return result


def snapshot(targets):
    result = {}
    for root, (minimum, maximum, current, _) in targets.items():
        fields = [minimum, maximum, current]
        fields.append("scaling_governor" if minimum.startswith("scaling") else "governor")
        result[str(root)] = {field: (root / field).read_text().strip() for field in fields}
    result["thermal"] = {
        str(path.parent): {"type": (path.parent / "type").read_text().strip(), "millidegrees_c": int(path.read_text())}
        for path in Path("/sys/class/thermal").glob("thermal_zone*/temp")
    }
    return result


def write(path, value):
    subprocess.run(["sudo", "-n", "tee", str(path)], input=str(value) + "\n",
                   text=True, stdout=subprocess.DEVNULL, check=True)


def run_command(command, env, lock_fd, timeout):
    """Keep descendants bounded too, while leaving clock restoration to main."""
    child = subprocess.Popen(command, env=env, pass_fds=(lock_fd,), start_new_session=True)
    try:
        try:
            return child.wait(timeout=timeout), False
        except subprocess.TimeoutExpired:
            return 124, True
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--wait-lock", action="store_true", help="wait for another board job to release the NPU")
    parser.add_argument("--cpu-khz", type=int, default=2400000, help="fixed big-core frequency in kHz")
    parser.add_argument("--skip-dmc", action="store_true", help="leave DDR uncontrolled on boards without DMC devfreq")
    parser.add_argument("--timeout", type=float, help="command timeout in seconds; terminate its process group and restore clocks")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.cpu_khz <= 0:
        parser.error("--cpu-khz must be positive")
    if args.timeout is not None and args.timeout <= 0:
        parser.error("--timeout must be positive")
    targets = clock_targets(args.cpu_khz, args.skip_dmc)
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("specify the executable and arguments after --")
    report = {"command": command, "started_unix": time.time(), "restore_errors": []}
    report["clock_policy"] = {"cpu_khz": args.cpu_khz, "dmc_controlled": not args.skip_dmc}
    status = 1
    with open("/tmp/rknpu_lowlevel_submit.lock", "a+") as lock:
        try:
            if args.wait_lock:
                print("Waiting for the shared NPU lock", flush=True)
            fcntl.flock(lock, fcntl.LOCK_EX | (0 if args.wait_lock else fcntl.LOCK_NB))
        except BlockingIOError:
            raise SystemExit("NPU is in use: /tmp/rknpu_lowlevel_submit.lock")
        before = snapshot(targets)
        report["before"] = before
        changed = []
        try:
            subprocess.run(["sudo", "-n", "true"], check=True)
            for root, (minimum, maximum, _, target) in targets.items():
                changed.append(root)
                write(root / maximum, target)
                write(root / minimum, target)
            report["fixed"] = snapshot(targets)
            for root, (minimum, maximum, _, target) in targets.items():
                fixed = report["fixed"][str(root)]
                if int(fixed[minimum]) != target or int(fixed[maximum]) != target:
                    raise RuntimeError("fixed clock verification failed: " + str(root))
            env = os.environ.copy()
            env["LLM_INFER_LOCK_FD"] = str(lock.fileno())
            env.setdefault("OMP_WAIT_POLICY", "PASSIVE")
            env.setdefault("GOMP_SPINCOUNT", "0")
            report["environment"] = {k: v for k, v in env.items() if k.startswith(("RK_NPU_", "OMP_", "GOMP_", "OPENBLAS_"))}
            start = time.monotonic()
            status, report["timed_out"] = run_command(command, env, lock.fileno(), args.timeout)
            report["command_elapsed_seconds"] = time.monotonic() - start
            report["after_command"] = snapshot(targets)
        except (OSError, subprocess.CalledProcessError, RuntimeError) as error:
            status = 1
            report["error"] = str(error)
            print("run_board:", error, flush=True)
        finally:
            for root in reversed(changed):
                minimum, maximum, _, _ = targets[root]
                try:
                    write(root / minimum, before[str(root)][minimum])
                    write(root / maximum, before[str(root)][maximum])
                except (OSError, subprocess.CalledProcessError) as error:
                    report["restore_errors"].append(str(error))
            report["restored"] = snapshot(targets)
            if report["restore_errors"]:
                status = 1
            report["exit_code"] = status
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + "\n")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
