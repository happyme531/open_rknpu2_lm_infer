#!/usr/bin/env python3
"""Build an optimized delivery executable with a chosen name and model families."""
import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", help="Project and executable name (default: RKNPU_PROJECT_NAME or Cargo package name)")
    parser.add_argument("--models", help="Comma-separated model families; omit to use Cargo defaults")
    parser.add_argument("--output", type=Path, help="Output directory (default: dist/NAME)")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    metadata = json.loads(subprocess.check_output(
        ["cargo", "metadata", "--format-version", "1", "--no-deps", "--locked"],
        cwd=root, text=True,
    ))
    project = next(package for package in metadata["packages"]
                   if Path(package["manifest_path"]) == root / "Cargo.toml")
    name = args.name if args.name is not None else os.environ.get("RKNPU_PROJECT_NAME", project["name"])
    target = next(target["name"] for target in project["targets"] if "bin" in target["kind"])
    command = ["cargo", "build", "--release", "--locked", "--bin", target,
               "--message-format=json-render-diagnostics"]
    if args.models is not None:
        families = args.models.split(",")
        unknown = [family for family in families if f"model-{family}" not in project["features"]]
        if unknown:
            parser.error(f"unknown model families: {', '.join(unknown)}")
        command += ["--no-default-features", "--features", ",".join(f"model-{family}" for family in families)]
    env = dict(os.environ, RKNPU_PROJECT_NAME=name)
    result = subprocess.run(command, cwd=root, env=env, text=True, stdout=subprocess.PIPE, check=True)
    records = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
    artifact = next(record["executable"] for record in records
                    if record.get("reason") == "compiler-artifact"
                    and record["target"]["name"] == target and record.get("executable"))
    build = json.loads(subprocess.check_output([artifact, "models"], text=True))
    if build["project_name"] != name:
        parser.error("built executable has an unexpected project name")
    output = (args.output or root / "dist" / name).resolve()
    output.mkdir(parents=True, exist_ok=True)
    binary = output / name
    shutil.copy2(artifact, binary)
    subprocess.run(["strip", "--strip-unneeded", str(binary)], check=True)
    (output / "build-info.json").write_text(json.dumps(build, indent=2) + "\n")
    print(binary)


if __name__ == "__main__":
    main()
