#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

WORK_ROOT="${WORK_ROOT:-/work1/yokyung/prorl_agent_server_env}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
SWEGYM_DATA="${SWEGYM_DATA:-${WORK_ROOT}/data/swegym_train_docker_3h200.jsonl}"
SWEGYM_MAX_TASKS="${SWEGYM_MAX_TASKS:-8}"
SWEGYM_RUNTIME_IMAGE_PREFIX="${SWEGYM_RUNTIME_IMAGE_PREFIX:-yokyung-polar-swegym-runtime}"

export TMPDIR="${WORK_ROOT}/tmp"
export TEMP="${TMPDIR}"
export TMP="${TMPDIR}"
export HF_HOME="${WORK_ROOT}/hf-home"
export HF_HUB_CACHE="${WORK_ROOT}/hf-hub-cache"
export TRANSFORMERS_CACHE="${WORK_ROOT}/transformers-cache"
export PYTHONPYCACHEPREFIX="${WORK_ROOT}/pycache"
export DOCKER_BUILDKIT="${DOCKER_BUILDKIT:-1}"
mkdir -p "${TMPDIR}" "$(dirname "${SWEGYM_DATA}")" "${HF_HOME}" \
    "${HF_HUB_CACHE}" "${TRANSFORMERS_CACHE}" "${PYTHONPYCACHEPREFIX}"

args=(
    --output-data "${SWEGYM_DATA}"
    --image-prefix "${SWEGYM_RUNTIME_IMAGE_PREFIX}"
)

if [ -n "${SWEGYM_INSTANCE_ID:-}" ]; then
    # Pass one or more exact instance ids by repeating SWEGYM_INSTANCE_ID in the
    # environment is not shell-portable, so use comma separation for this script.
    IFS=',' read -r -a ids <<< "${SWEGYM_INSTANCE_ID}"
    for id in "${ids[@]}"; do
        args+=(--instance-id "${id}")
    done
else
    args+=(--max-tasks "${SWEGYM_MAX_TASKS}")
fi

if [ "${SWEGYM_FORCE_REBUILD:-0}" = "1" ]; then
    args+=(--force)
fi
if [ "${SWEGYM_REFRESH_DATASET_CACHE:-0}" = "1" ]; then
    args+=(--refresh-dataset-cache)
fi

"${PYTHON_BIN}" "${PROJECT_ROOT}/examples/swegym_slime_grpo/build_docker_images.py" "${args[@]}"

echo "Prepared SWE-Gym Docker data: ${SWEGYM_DATA}"

