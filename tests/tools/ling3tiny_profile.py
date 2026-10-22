#!/usr/bin/env python3
"""Summarize Ling Perfetto traces without adding nested native/model spans."""
import argparse
import csv
import io
import json
from pathlib import Path
import statistics
import subprocess

TABLES = ("phases", "model", "moe", "native_cpu", "layers", "native",
          "native_by_model", "geometry", "health")


def value(text):
    if text in ("[NULL]", "NULL"):
        return None
    try:
        return int(text)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return text


def query(processor, sql, trace):
    result = subprocess.run([str(processor.resolve()), "query", "-f", str(sql), str(trace)],
                            text=True, capture_output=True, check=True)
    groups, rows = [], []
    for row in csv.reader(io.StringIO(result.stdout)):
        if not row:
            if rows:
                groups.append(rows)
                rows = []
        else:
            rows.append(row)
    if rows:
        groups.append(rows)
    if len(groups) != len(TABLES):
        raise ValueError(f"{trace}: expected {len(TABLES)} SQL results, got {len(groups)}")
    data = {name: [dict(zip(group[0], map(value, row))) for row in group[1:]]
            for name, group in zip(TABLES, groups)}
    if data["health"]:
        raise ValueError(f"{trace}: non-info trace health counters: {data['health']}")
    for phase, factor in [("llm.prefill", 4), ("llm.decode", 128)]:
        calls = {r["stage"]: r["calls"] for r in data["model"] if r["phase"] == phase}
        expected = {"ling.kda_input_projection": 18 * factor,
                    "ling.kda_recurrence": 18 * factor,
                    "ling.kda_convolution": 54 * factor,
                    "ling.mla_core": 6 * factor,
                    "ling.moe_routed_experts": 23 * factor,
                    "ling.output_head": 0 if factor == 4 else 128}
        for stage, count in expected.items():
            if calls.get(stage, 0) != count:
                raise ValueError(f"{trace}: {phase}/{stage} calls {calls.get(stage)} != {count}")
        moe = next(r for r in data["moe"] if r["phase"] == phase)
        if moe["calls"] != 23 * factor:
            raise ValueError(f"{trace}: incomplete MoE timings")
    return data


def summarize(primary, detail, control):
    result = {}
    groups = {
        "routed_experts": ["ling.moe_routed_experts"],
        "shared_expert": ["ling.shared_expert_ffn"],
        "kda_recurrence": ["ling.kda_recurrence"],
        "kda_projection": ["ling.kda_input_projection", "ling.kda_output_projection"],
        "kda_norm_gate_beta": ["ling.kda_qk_norm_gates", "ling.kda_output_norm_gate", "ling.kda_beta_projection"],
        "kda_conv": ["ling.kda_convolution"],
        "mla_core": ["ling.mla_core"],
        "mla_projection": ["ling.mla_q_a_projection", "ling.mla_q_b_projection", "ling.mla_kv_a_projection", "ling.mla_gate_projection", "ling.mla_output_projection"],
        "mla_norm_rope": ["ling.mla_norm_rope"],
        "router": ["ling.moe_router_projection", "ling.moe_router"],
        "lm_head": ["ling.output_head"],
    }
    for phase, key in [("llm.prefill", "prefill"), ("llm.decode", "decode")]:
        wall = [next(r["wall_ms"] for r in d["phases"] if r["name"] == phase) for d in primary]
        total = statistics.mean(wall)
        models = [{r["stage"]: r for r in d["model"] if r["phase"] == phase} for d in primary]
        stages = {name: {"ms": statistics.mean(m[name]["ms"] for m in models),
                         "calls": models[0][name]["calls"]} for name in models[0]}
        for stage in stages.values():
            stage["percent"] = 100 * stage["ms"] / total
        grouped = {name: sum(stages.get(s, {"ms": 0})["ms"] for s in names) for name, names in groups.items()}
        grouped["other"] = total - sum(grouped.values())
        remainder = total - sum(s["ms"] for s in stages.values())
        if remainder < 0:
            raise ValueError("Model spans overlap or exceed the phase duration")
        moes = [next(r for r in d["moe"] if r["phase"] == phase) for d in primary]
        moe = {name: statistics.mean(r[name] for r in moes) for name in moes[0] if name != "phase"}
        control_ms = statistics.mean(control[key]["samples_ms"])
        detail_ms = next(r["wall_ms"] for r in detail["phases"] if r["name"] == phase)
        result[phase] = {
            "wall_samples_ms": wall, "wall_ms": total, "control_ms": control_ms,
            "primary_overhead_pct": 100 * (total / control_ms - 1),
            "detail_wall_ms": detail_ms, "detail_overhead_pct": 100 * (detail_ms / control_ms - 1),
            "model": stages, "uncovered_model_ms": remainder,
            "groups": {k: {"ms": v, "percent": 100 * v / total} for k, v in grouped.items()},
            "moe": moe,
            "detail_native": [r for r in detail["native"] if r["phase"] == phase],
            "detail_native_by_model": [r for r in detail["native_by_model"] if r["phase"] == phase],
            "geometry": [r for r in detail["geometry"] if r["phase"] == phase],
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--processor", type=Path, default=Path(".cache/perfetto-v58.2/trace_processor_shell"))
    parser.add_argument("--sql", type=Path, default=Path("llm_generated_docs/LING3TINY_TRACE_20261023.sql"))
    args = parser.parse_args()
    paths = sorted(args.directory.glob("primary.case0.rep*.pftrace"))
    if not paths:
        raise ValueError("No primary trace files")
    details = list(args.directory.glob("detail.case0.rep*.pftrace"))
    if len(details) != 1:
        raise ValueError("Expected one detail trace")
    raw = {str(p): query(args.processor, args.sql, p) for p in paths + details}
    control = json.loads((args.directory / "control.json").read_text())
    after = args.directory / "control-after.json"
    if after.exists():
        later = json.loads(after.read_text())
        for phase in ("prefill", "decode"):
            control[phase]["samples_ms"].extend(later[phase]["samples_ms"])
    summary = summarize([raw[str(p)] for p in paths], raw[str(details[0])], control)
    (args.directory / "trace-raw.json").write_text(json.dumps(raw, indent=2) + "\n")
    (args.directory / "trace-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.directory / "layers.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["phase", "layer", "stage", "mean_ms", "calls"])
        keys = {(r["phase"], r["layer"], r["stage"]) for r in raw[str(paths[0])]["layers"]}
        for phase, layer, stage in sorted(keys):
            samples = [next(r for r in raw[str(p)]["layers"] if (r["phase"], r["layer"], r["stage"]) == (phase, layer, stage)) for p in paths]
            writer.writerow([phase, layer, stage, statistics.mean(r["ms"] for r in samples), samples[0]["calls"]])
    for phase, data in summary.items():
        print(f"{phase}: {data['wall_ms']:.3f} ms; tracing {data['primary_overhead_pct']:+.2f}%")
        for name, row in sorted(data["groups"].items(), key=lambda item: -item[1]["ms"]):
            print(f"  {name}: {row['ms']:.3f} ms ({row['percent']:.2f}%)")


if __name__ == "__main__":
    main()
