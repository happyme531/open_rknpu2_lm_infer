#!/usr/bin/env python3
"""Package the ARM64 executable, notices and pinned source dependency."""
import argparse
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    metadata = json.loads(subprocess.check_output(
        ["cargo", "metadata", "--format-version", "1", "--locked"], cwd=root, text=True,
    ))
    project = next(package for package in metadata["packages"]
                   if Path(package["manifest_path"]) == root / "Cargo.toml")
    version = project["version"]
    if os.environ.get("GITHUB_REF_TYPE") == "tag":
        if os.environ["GITHUB_REF_NAME"] != f"v{version}":
            parser.error("release tag must match the Cargo package version")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    prefix = f"open_rknpu2_lm_infer-v{version}"
    with tempfile.TemporaryDirectory() as temporary:
        staging = Path(temporary)
        binary = staging / f"{prefix}-linux-aarch64"
        (binary / "bin").mkdir(parents=True)
        shutil.copy2(root / "target/release/open_rknpu2_lm_infer", binary / "bin")
        for name in ["LICENSE", "THIRD_PARTY_NOTICES.md", "README.md", "README_zh.md", "Cargo.lock",
                     "src/model/experts/Strata-LICENSE", "src/model/experts/Omni-LICENSE",
                     "src/model/dflash/DeepSpec-LICENSE"]:
            destination = binary / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / name, destination)
        docs = subprocess.check_output(
            ["git", "ls-files", "-z", "--", "llm_generated_docs"], cwd=root,
        ).decode().split("\0")
        for name in filter(None, docs):
            destination = binary / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / name, destination)
        (binary / "scripts").mkdir()
        shutil.copy2(root / "scripts/run_board.py", binary / "scripts/run_board.py")
        operator = binary / "native/rk_npu_matmul"
        operator.mkdir(parents=True)
        for name in ["LICENSE", "THIRD_PARTY_NOTICES.md"]:
            shutil.copy2(root / "native/rk_npu_matmul" / name, operator / name)
        rust_notices = binary / "third_party/rust"
        rust_notices.mkdir(parents=True)
        entries = ["# Rust dependencies", "", "| Package | License | Source |", "|---|---|---|"]
        for package in sorted(metadata["packages"], key=lambda item: (item["name"], item["version"])):
            if not package["source"]:
                continue
            directory = Path(package["manifest_path"]).parent
            files = {path for path in directory.iterdir()
                     if path.name.upper().startswith(("LICENSE", "LICENCE", "COPYING", "NOTICE"))}
            if package["license_file"]:
                files.add(directory / package["license_file"])
            destination = rust_notices / f'{package["name"]}-{package["version"]}'
            for path in sorted(files):
                destination.mkdir(parents=True, exist_ok=True)
                if path.is_dir():
                    shutil.copytree(path, destination / path.name)
                else:
                    shutil.copy2(path, destination / path.name)
            entries.append(f'| {package["name"]} {package["version"]} | '
                           f'{package["license"] or ""} | {package["repository"] or package["source"]} |')
        (binary / "third_party/RUST_DEPENDENCIES.md").write_text("\n".join(entries) + "\n")
        source = staging / f"{prefix}-source"
        subprocess.run(["python3", str(root / "scripts/package_source.py"),
                        "--output", str(source)], check=True)
        for directory in [binary, source]:
            archive = output / f"{directory.name}.tar.gz"
            with tarfile.open(archive, "w:gz") as handle:
                handle.add(directory, arcname=directory.name)
            print(archive)


if __name__ == "__main__":
    main()
