"""Expert I/O trace: prompt read reuse, overlap, and lookahead timing."""
import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
import json
from pathlib import Path

from analyze_expert_timeline import audit, duration, intersect, merged


def percentiles(values):
    if not values:
        return None
    values = sorted(values)
    return {"mean": sum(values) / len(values),
            "p50": values[len(values) // 2],
            "p90": values[min(len(values) - 1, int(len(values) * .9))],
            "max": values[-1], "count": len(values)}


def analyze(path):
    events = json.loads(path.read_text())["traceEvents"]
    routes = sorted((e for e in events if e["name"] == "actual routes"), key=lambda e: e["ts"])
    spans = [e for e in events if e["ph"] == "X"]
    prompt = [e for e in routes if e["args"]["rows"] > 1]
    decode = [e for e in routes if e["args"]["rows"] == 1]
    start = min(e["ts"] for e in prompt)
    end = max(e["ts"] + e["dur"] for e in spans
              if e["name"] == "routed" and e["args"]["rows"] > 1)
    window = [(start, end)]
    reads = [e for e in spans if e["name"] == "pread" and start <= e["ts"] < end]
    keys = Counter((e["args"]["layer"], e["args"]["expert"]) for e in reads)
    required = {(e["args"]["layer"], expert) for e in prompt for expert in e["args"]["experts"]}
    record = reads[0]["args"]["bytes"]
    ranges = lambda name: [(e["ts"], e["ts"] + e["dur"]) for e in spans if e["name"] == name]
    pp = {"timing_scope": "first actual router to last routed FFN; excludes first mixer and final shared FFN",
          "region_seconds": (end - start) / 1e6,
          "logical_chunks": sorted({(e["args"]["position"], e["args"]["rows"]) for e in prompt}),
          "assignments": sum(len(e["args"]["experts"]) for e in prompt),
          "unique_layer_experts": len(required), "read_records": len(reads),
          "distinct_read_keys": len(keys), "repeat_read_records": len(reads) - len(keys),
          "repeat_read_bytes": (len(reads) - len(keys)) * record,
          "read_bytes": len(reads) * record, "once_per_layer_expert_bytes": len(required) * record,
          "read_repeat_histogram": dict(Counter(keys.values()))}
    for name in ["pread", "wait record", "NPU experts", "import"]:
        pp[name + "_seconds"] = duration(intersect(ranges(name), window)) / 1000
    pp["io_expert_overlap_seconds"] = duration(intersect(
        intersect(ranges("pread"), ranges("NPU experts")), window)) / 1000
    pp["io_nonwaiting_overlap_seconds"] = pp["pread_seconds"] - duration(intersect(
        intersect(ranges("pread"), ranges("wait record")), window)) / 1000

    # A group's first ready record can wait for the rest of its group before import.
    ready = {}
    delays = []
    points = []
    for e in events:
        if e["name"] == "read ready":
            points.append((e["ts"], 0, e))
        elif e["name"] == "import" and e["args"]["bank"] == "cold" and start <= e["ts"] < end:
            points.append((e["ts"], 1, e))
    for timestamp, kind, e in sorted(points, key=lambda x: (x[0], x[1])):
        key = (e["args"]["layer"], e["args"]["expert"])
        if kind == 0:
            ready[key] = timestamp
        elif key in ready:
            delays.append((timestamp - ready.pop(key)) / 1000)
    pp["ready_to_import_ms"] = percentiles(delays)

    last_prediction = {}
    lead = []
    for e in sorted(events, key=lambda e: e["ts"]):
        if e["name"] == "LES prediction":
            last_prediction[e["args"]["layer"]] = e["ts"]
        elif e["name"] == "actual routes" and e["args"]["rows"] == 1:
            if e["args"]["layer"] in last_prediction:
                lead.append((e["ts"] - last_prediction[e["args"]["layer"]]) / 1000)
    by_layer = defaultdict(list)
    for e in decode:
        by_layer[e["args"]["layer"]].append(e)
    starts = {layer: [e["ts"] for e in group] for layer, group in by_layer.items()}
    io_lead = []
    durations = []
    started_correct = 0
    issued = 0
    for e in spans:
        if e["name"] != "pread" or not e["args"]["prefetch"]:
            continue
        group = by_layer.get(e["args"]["layer"], [])
        if not group:
            continue
        i = bisect_left(starts[e["args"]["layer"]], e["ts"])
        candidates = group[max(0, i - 1):min(len(group), i + 1)]
        actual = min(candidates, key=lambda a: abs(a["ts"] - e["ts"]))
        issued += 1
        io_lead.append((actual["ts"] - e["ts"]) / 1000)
        durations.append(e["dur"] / 1000)
        started_correct += int(e["args"]["expert"] in actual["args"]["experts"])
    result = {"trace": str(path), "events": len(events), "prefill": pp,
              "decode": {"prediction_lead_ms": percentiles(lead),
                         "prefetch_read_lead_ms": percentiles(io_lead),
                         "prefetch_read_duration_ms": percentiles(durations),
                         "prefetch_records_started": issued,
                         "prefetch_started_correct": started_correct},
              "audit": audit(events)}
    return result, events, (start, end)


def plot(events, bounds, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    spans = [e for e in events if e["ph"] == "X"]
    first = min((e for e in spans if e["name"] == "routed" and e["args"]["rows"] > 1),
                key=lambda e: e["ts"])
    regions = [bounds, (first["ts"], first["ts"] + first["dur"])]
    colors = {"pread": "#2878b5", "wait record": "#e68632",
              "NPU experts": "#3a923a", "import": "#9566b0"}
    labels = {"pread": "Disk read", "wait record": "Main-thread wait",
              "NPU experts": "Expert complete call", "import": "Import + sync"}
    fig, axes = plt.subplots(2, 1, figsize=(14, 6.2))
    for ax, (begin, end) in zip(axes, regions):
        scale = 1e6 if ax is axes[0] else 1e3
        for row, name in enumerate(colors):
            intervals = merged((max(begin, e["ts"]), min(end, e["ts"] + e["dur"]))
                               for e in spans if e["name"] == name
                               and e["ts"] < end and e["ts"] + e["dur"] > begin)
            ax.broken_barh([((a - begin) / scale, (b - a) / scale) for a, b in intervals],
                           (row - .3, .6), facecolors=colors[name])
        ax.set_yticks(range(4), [labels[name] for name in colors])
        ax.set_xlim(0, (end - begin) / scale)
        ax.set_xlabel("Seconds" if ax is axes[0] else "Milliseconds")
        ax.grid(axis="x", alpha=.25)
    axes[0].set_title("Complete prompt expert region")
    axes[1].set_title("First layer / first logical prompt chunk")
    fig.tight_layout()
    fig.subplots_adjust(top=.94, bottom=.09)
    fig.savefig(path, dpi=180, bbox_inches="tight", pad_inches=.15)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("trace", type=Path)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--plot", type=Path)
    args = p.parse_args()
    result, events, bounds = analyze(args.trace)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    if args.plot:
        plot(events, bounds, args.plot)
    print(json.dumps({"prefill": result["prefill"], "decode": result["decode"],
                      "audit_violations": len(result["audit"]["violations"])}, indent=2))


if __name__ == "__main__":
    main()
