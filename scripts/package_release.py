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
    parser.add_argument("--binary", type=Path, help="Executable to package (defaults to Cargo target/release)")
    parser.add_argument("--models", help="Require this exact comma-separated family list in the built executable")
    parser.add_argument("--name", help="Require this project name in the built executable")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    cargo_metadata = ["cargo", "metadata", "--format-version", "1", "--locked"]
    basic = json.loads(subprocess.check_output(cargo_metadata + ["--no-deps"], cwd=root, text=True))
    project = next(package for package in basic["packages"]
                   if Path(package["manifest_path"]) == root / "Cargo.toml")
    version = project["version"]
    target = next(target["name"] for target in project["targets"] if "bin" in target["kind"])
    executable = (args.binary or Path(basic["target_directory"]) / "release" / target).resolve()
    build = json.loads(subprocess.check_output([str(executable), "models"], text=True))
    project_name = build["project_name"]
    if args.name is not None and args.name != project_name:
        parser.error(f"binary project name {project_name!r} does not match requested {args.name!r}; rebuild with --name")
    families = sorted(family["name"] for family in build["model_families"])
    if not families or build["version"] != version:
        parser.error("binary must enable a model family and match the Cargo package version")
    if args.models is not None:
        requested = sorted(set(args.models.split(",")))
        if requested != families:
            parser.error(f"binary families {families} do not match requested {requested}; rebuild with the intended features")
    features = [f"model-{name}" for name in families]
    metadata = json.loads(subprocess.check_output(
        cargo_metadata + ["--no-default-features", "--features", ",".join(features)], cwd=root, text=True,
    ))
    if os.environ.get("GITHUB_REF_TYPE") == "tag":
        if os.environ["GITHUB_REF_NAME"] != f"v{version}":
            parser.error("release tag must match the Cargo package version")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    default_families = sorted(name.removeprefix("model-") for name in project["features"]["all-models"])
    variant = "" if families == default_families else "-" + "-".join(families)
    prefix = f"{project_name}-v{version}{variant}"
    with tempfile.TemporaryDirectory() as temporary:
        staging = Path(temporary)
        binary = staging / f"{prefix}-linux-aarch64"
        (binary / "bin").mkdir(parents=True)
        shutil.copy2(executable, binary / "bin" / project_name)
        (binary / "build-info.json").write_text(json.dumps(build, indent=2) + "\n")
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
