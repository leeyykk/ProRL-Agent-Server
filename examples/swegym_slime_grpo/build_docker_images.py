#!/usr/bin/env python3
"""Build Docker runtime images and JSONL data for SWE-Gym Slime runs.

Each output row includes ``metadata.runtime_image`` so Polar can start the
matching per-instance Docker runtime.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

EXAMPLE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXAMPLE_DIR.parents[1]
RUNTIME_DOCKERFILE_DIR = PROJECT_ROOT / "examples" / "swebench_verified" / "runtime"
IMAGE_LAYOUT_VERSION = "1"
IMAGE_VERSION_LABEL = "io.polar.swegym-image-version"

sys.path.insert(0, str(EXAMPLE_DIR))
from sample_tasks import fetch_dataset_instances, registry_image_for_instance_id  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-data", type=Path, required=True)
    parser.add_argument("--image-prefix", default="polar-swegym-runtime")
    parser.add_argument("--instance-id", action="append", default=[])
    parser.add_argument("--max-tasks", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--refresh-dataset-cache", action="store_true")
    return parser.parse_args()


def sanitize_instance_id(instance_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", instance_id).lower()


def runtime_image_for_instance(instance_id: str, image_prefix: str) -> str:
    return f"{image_prefix}:{sanitize_instance_id(instance_id)}"


def run_command(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def image_exists(image_ref: str) -> bool:
    return subprocess.run(
        ["docker", "image", "inspect", image_ref],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def image_layout_version(image_ref: str) -> str | None:
    result = subprocess.run(
        [
            "docker",
            "image",
            "inspect",
            "--format",
            '{{ index .Config.Labels "' + IMAGE_VERSION_LABEL + '" }}',
            image_ref,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


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


def row_for_instance(instance: dict[str, Any], runtime_image: str) -> dict[str, Any]:
    instance_id = str(instance["instance_id"])
    return {
        "prompt": [{"role": "user", "content": str(instance["problem_statement"]).strip()}],
        "label": "",
        "metadata": {
            "instance_id": instance_id,
            "instance": instance,
            "runtime_image": runtime_image,
            "split": "train",
        },
    }


def build_image(instance: dict[str, Any], image_prefix: str, *, force: bool) -> str:
    instance_id = str(instance["instance_id"])
    base_image = registry_image_for_instance_id(instance_id)
    runtime_image = runtime_image_for_instance(instance_id, image_prefix)

    if image_exists(runtime_image) and not force:
        version = image_layout_version(runtime_image)
        if version == IMAGE_LAYOUT_VERSION:
            print(f"skip: {runtime_image}", flush=True)
            return runtime_image
        print(f"rebuild: {runtime_image} (v{version} -> v{IMAGE_LAYOUT_VERSION})", flush=True)

    run_command(["docker", "pull", base_image])
    run_command(
        [
            "docker",
            "build",
            "--build-arg",
            f"BASE_IMAGE={base_image}",
            "--label",
            f"{IMAGE_VERSION_LABEL}={IMAGE_LAYOUT_VERSION}",
            "--tag",
            runtime_image,
            str(RUNTIME_DOCKERFILE_DIR),
        ]
    )
    return runtime_image


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=True) for row in rows) + "\n")


def main() -> int:
    args = parse_args()
    if not (RUNTIME_DOCKERFILE_DIR / "Dockerfile").is_file():
        raise SystemExit(f"No Dockerfile found at {RUNTIME_DOCKERFILE_DIR / 'Dockerfile'}")

    instances = select_instances(args)
    if not instances:
        raise SystemExit("No SWE-Gym instances selected.")

    rows = []
    for idx, instance in enumerate(instances, 1):
        instance_id = str(instance["instance_id"])
        print(f"[{idx}/{len(instances)}] preparing {instance_id}", flush=True)
        runtime_image = build_image(instance, args.image_prefix, force=args.force)
        rows.append(row_for_instance(instance, runtime_image))

    write_jsonl(args.output_data, rows)
    print(f"Prepared {len(rows)} SWE-Gym Docker row(s): {args.output_data}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
