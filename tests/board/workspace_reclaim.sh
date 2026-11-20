#!/usr/bin/env bash
# Run on RK3588 from the repository root using the matching C7 weights and tuning cache.
set -euo pipefail
if [[ $# -lt 3 || $# -gt 4 ]]; then
    echo "usage: $0 {prepare|verify|dense} MODEL TUNE_CACHE [CLOCK_REPORT]" >&2
    exit 2
fi
job=$(mktemp -d)
trap 'rm -rf -- "$job"' EXIT
cc -O2 -Wall -Wextra -shared -fPIC tests/board/workspace_fault_ioctl.c -ldl -o "$job/fault.so"
# Export the native attention graph symbols used to select the allocation to fail.
cargo rustc --locked --release --example workspace_reclaim --message-format=json-render-diagnostics -- \
    -C link-arg=-rdynamic >"$job/build.jsonl"
probe=$(python3 - "$job/build.jsonl" <<'PYTHON'
import json
import sys

with open(sys.argv[1]) as stream:
    for line in stream:
        artifact = json.loads(line)
        if artifact.get("executable") and artifact["target"]["name"] == "workspace_reclaim":
            print(artifact["executable"])
            break
    else:
        sys.exit("cargo did not emit the workspace_reclaim executable")
PYTHON
)
RK_NPU_PREFILL_CACHE_DOMAIN_MIB=1792 python3 scripts/run_board.py --wait-lock --timeout 900 --report "${4:-$job/clocks.json}" -- \
    taskset -c 4-7 env LD_PRELOAD="$job/fault.so" "$probe" "$1" "$2" "$3"
