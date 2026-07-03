#!/usr/bin/env python3
"""Prepare SWE-Gym JSONL data backed by local Apptainer SIF images.

This is the Docker-free counterpart to ``build_docker_images.py`` for servers
where the Docker daemon is unavailable. It pulls the public SWE-Gym/SWE-bench
registry images with Apptainer and writes rows whose runtime image is the local
``.sif`` path consumed by Polar's Apptainer runtime.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
SWEGYM_EXAMPLE_DIR = PROJECT_ROOT / "examples" / "swegym_slime_grpo"
sys.path.insert(0, str(SWEGYM_EXAMPLE_DIR))

from sample_tasks import fetch_dataset_instances, registry_image_for_instance_id  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-data", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--tmp-dir", type=Path, required=True)
    parser.add_argument("--instance-id", action="append", default=[])
    parser.add_argument("--max-tasks", type=int, default=24)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--refresh-dataset-cache", action="store_true")
    return parser.parse_args()


def sanitize_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-").lower()


def sif_path_for_instance(instance_id: str, image_dir: Path) -> Path:
    return image_dir / f"{sanitize_name(instance_id)}.sif"


def apptainer_binary() -> str:
    override = os.environ.get("POLAR_APPTAINER_BIN")
    if override:
        return override
    for candidate in ("/usr/bin/apptainer", "/bin/apptainer"):
        if Path(candidate).is_file():
            return candidate
    resolved = shutil.which("apptainer")
    if resolved:
        return resolved
    return "apptainer"


def apptainer_uri(image_ref: str) -> str:
    if image_ref.startswith(("docker://", "oras://", "library://", "shub://")):
        return image_ref
    return f"docker://{image_ref}"


def select_instances(args: argparse.Namespace) -> list[dict[str, Any]]:
    instances = fetch_dataset_instances("train", refresh=args.refresh_dataset_cache)
    if args.instance_id:
        wanted = set(args.instance_id)
        selected = [i for i in instances if str(i.get("instance_id")) in wanted]
        missing = sorted(wanted - {str(i.get("instance_id")) for i in selected})
        if missing:
            raise SystemExit(f"Unknown instance_id(s): {', '.join(missing)}")
        return selected
    if args.max_tasks > 0:
        return instances[: args.max_tasks]
    return instances


def image_ready(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def pull_sif(
    instance: dict[str, Any],
    *,
    image_dir: Path,
    force: bool,
    env: dict[str, str],
) -> tuple[str, str, str]:
    instance_id = str(instance["instance_id"])
    registry_image = registry_image_for_instance_id(instance_id)
    target = sif_path_for_instance(instance_id, image_dir)
    if image_ready(target) and not force:
        return ("skipped", instance_id, str(target))

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(
        f".{target.name}.tmp-{os.getpid()}-{threading.get_ident()}"
    )
    if tmp_path.exists():
        tmp_path.unlink()

    command = [
        apptainer_binary(),
        "pull",
        "--force",
        str(tmp_path),
        apptainer_uri(registry_image),
    ]
    print("+", " ".join(command), flush=True)
    try:
        subprocess.run(command, check=True, env=env)
        tmp_path.replace(target)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return ("pulled", instance_id, str(target))


def ensure_sifs(
    instances: list[dict[str, Any]],
    *,
    image_dir: Path,
    cache_dir: Path,
    tmp_dir: Path,
    force: bool,
    jobs: int,
) -> None:
    env = {
        **os.environ,
        "APPTAINER_CACHEDIR": str(cache_dir.resolve()),
        "APPTAINER_TMPDIR": str(tmp_dir.resolve()),
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    if jobs <= 1:
        for instance in instances:
            status, instance_id, path = pull_sif(
                instance, image_dir=image_dir, force=force, env=env
            )
            print(f"{status}: {instance_id} -> {path}", flush=True)
        return

    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = [
            executor.submit(
                pull_sif, instance, image_dir=image_dir, force=force, env=env
            )
            for instance in instances
        ]
        for future in as_completed(futures):
            status, instance_id, path = future.result()
            print(f"{status}: {instance_id} -> {path}", flush=True)


def row_for_instance(instance: dict[str, Any], sif_path: Path) -> dict[str, Any]:
    instance_id = str(instance["instance_id"])
    registry_image = registry_image_for_instance_id(instance_id)
    return {
        "prompt": [
            {
                "role": "user",
                "content": str(instance["problem_statement"]).strip(),
            }
        ],
        "label": "",
        "metadata": {
            "instance_id": instance_id,
            "instance": instance,
            "runtime_image": str(sif_path.resolve()),
            "registry_runtime_image": registry_image,
            "split": "train",
        },
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    binary = apptainer_binary()
    if shutil.which(binary) is None and not Path(binary).is_file():
        raise SystemExit(f"apptainer not found: {binary}")

    instances = select_instances(args)
    if not instances:
        raise SystemExit("No SWE-Gym instances selected.")

    print(f"Preparing {len(instances)} SWE-Gym Apptainer SIF image(s).")
    print(f"Image dir: {args.image_dir.resolve()}")
    print(f"Output data: {args.output_data.resolve()}")
    ensure_sifs(
        instances,
        image_dir=args.image_dir,
        cache_dir=args.cache_dir,
        tmp_dir=args.tmp_dir,
        force=args.force,
        jobs=max(args.jobs, 1),
    )

    rows = [
        row_for_instance(instance, sif_path_for_instance(str(instance["instance_id"]), args.image_dir))
        for instance in instances
    ]
    write_jsonl(args.output_data, rows)
    print(f"Prepared {len(rows)} SWE-Gym Apptainer row(s): {args.output_data}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
