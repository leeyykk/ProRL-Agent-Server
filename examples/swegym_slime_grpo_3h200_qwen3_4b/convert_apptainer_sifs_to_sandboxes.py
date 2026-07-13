#!/usr/bin/env python3
"""Convert Apptainer/Singularity SIF-backed SWE-Gym data to sandbox dirs.

This is for containerized HPC allocations where loop devices and /dev/fuse are
not exposed. In that environment, executing a SIF repeatedly forces Singularity
to extract the image on every exec. A persistent sandbox moves that extraction
cost into setup and keeps the profiled run representative.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-data", type=Path, required=True)
    parser.add_argument("--output-data", type=Path, required=True)
    parser.add_argument("--sandbox-dir", type=Path, required=True)
    parser.add_argument("--tmp-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args()


def singularity_binary() -> str:
    override = os.environ.get("POLAR_APPTAINER_BIN")
    if override:
        return override
    for candidate in ("singularity", "apptainer"):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return "singularity"


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def sandbox_path_for(sif: Path, sandbox_root: Path) -> Path:
    return sandbox_root / sif.with_suffix("").name


def build_sandbox(
    binary: str,
    sif: Path,
    sandbox: Path,
    env: dict[str, str],
    *,
    force: bool,
) -> None:
    if sandbox.exists() and not force:
        print(f"skipped: {sif} -> {sandbox}", flush=True)
        return
    if sandbox.exists():
        raise SystemExit(
            f"Refusing to overwrite existing sandbox without manual cleanup: {sandbox}"
        )
    sandbox.parent.mkdir(parents=True, exist_ok=True)
    command = [binary, "build", "--sandbox", str(sandbox), str(sif)]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def verify_sandbox(binary: str, sandbox: Path, env: dict[str, str]) -> None:
    command = [binary, "exec", "--userns", str(sandbox), "true"]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def main() -> int:
    args = parse_args()
    binary = singularity_binary()
    args.tmp_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.sandbox_dir.mkdir(parents=True, exist_ok=True)

    env = {
        **os.environ,
        "APPTAINER_TMPDIR": str(args.tmp_dir.resolve()),
        "APPTAINER_CACHEDIR": str(args.cache_dir.resolve()),
        "SINGULARITY_TMPDIR": str(args.tmp_dir.resolve()),
        "SINGULARITY_CACHEDIR": str(args.cache_dir.resolve()),
    }

    rows = load_rows(args.input_data)
    seen: dict[str, Path] = {}
    for row in rows:
        metadata = row.setdefault("metadata", {})
        image = Path(str(metadata["runtime_image"]))
        if image.is_dir():
            sandbox = image
        else:
            sandbox = sandbox_path_for(image, args.sandbox_dir)
            if str(image) not in seen:
                build_sandbox(binary, image, sandbox, env, force=args.force)
                if args.verify:
                    verify_sandbox(binary, sandbox, env)
                seen[str(image)] = sandbox
        metadata["runtime_image"] = str(sandbox.resolve())
        metadata["runtime_image_source_sif"] = str(image.resolve())

    args.output_data.parent.mkdir(parents=True, exist_ok=True)
    args.output_data.write_text(
        "\n".join(json.dumps(row, ensure_ascii=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote sandbox-backed data: {args.output_data}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
