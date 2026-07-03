#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

WORK_ROOT="${WORK_ROOT:-/home/korea_bupj/vialab/yokyung/prorl_agent_server_env}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
SWEGYM_DATA="${SWEGYM_DATA:-${WORK_ROOT}/data/swegym_train_apptainer_b200.jsonl}"
SWEGYM_MAX_TASKS="${SWEGYM_MAX_TASKS:-24}"
SWEGYM_INSTANCE_ID="${SWEGYM_INSTANCE_ID:-}"
SWEGYM_REFRESH_DATASET_CACHE="${SWEGYM_REFRESH_DATASET_CACHE:-0}"
SWEGYM_FORCE_SIF="${SWEGYM_FORCE_SIF:-0}"
APPTAINER_IMAGE_DIR="${APPTAINER_IMAGE_DIR:-${WORK_ROOT}/apptainer_sifs/swegym_b200}"
APPTAINER_CACHE_DIR="${APPTAINER_CACHE_DIR:-${WORK_ROOT}/apptainer_cache}"
APPTAINER_TMP_DIR="${APPTAINER_TMP_DIR:-${WORK_ROOT}/apptainer_tmp}"
APPTAINER_JOBS="${APPTAINER_JOBS:-2}"
POLAR_APPTAINER_BIN="${POLAR_APPTAINER_BIN:-apptainer}"

export TMPDIR="${WORK_ROOT}/tmp"
export TEMP="${TMPDIR}"
export TMP="${TMPDIR}"
export HF_HOME="${WORK_ROOT}/hf-home"
export HF_HUB_CACHE="${WORK_ROOT}/hf-hub-cache"
export TRANSFORMERS_CACHE="${WORK_ROOT}/transformers-cache"
export PYTHONPYCACHEPREFIX="${WORK_ROOT}/pycache"
export APPTAINER_CACHEDIR="${APPTAINER_CACHE_DIR}"
export APPTAINER_TMPDIR="${APPTAINER_TMP_DIR}"
export POLAR_APPTAINER_BIN

mkdir -p "${TMPDIR}" "$(dirname "${SWEGYM_DATA}")" "${APPTAINER_IMAGE_DIR}" \
    "${APPTAINER_CACHE_DIR}" "${APPTAINER_TMP_DIR}" "${HF_HOME}" \
    "${HF_HUB_CACHE}" "${TRANSFORMERS_CACHE}" "${PYTHONPYCACHEPREFIX}"

if [ ! -x "${PYTHON_BIN}" ]; then
    echo "ERROR: Python env not found: ${PYTHON_BIN}" >&2
    exit 1
fi

if ! command -v "${POLAR_APPTAINER_BIN}" >/dev/null 2>&1; then
    echo "ERROR: Apptainer binary not found: ${POLAR_APPTAINER_BIN}" >&2
    echo "Set POLAR_APPTAINER_BIN=/path/to/apptainer if needed." >&2
    exit 1
fi

args=(
    --output-data "${SWEGYM_DATA}"
    --image-dir "${APPTAINER_IMAGE_DIR}"
    --cache-dir "${APPTAINER_CACHE_DIR}"
    --tmp-dir "${APPTAINER_TMP_DIR}"
    --jobs "${APPTAINER_JOBS}"
)

if [ -n "${SWEGYM_INSTANCE_ID}" ]; then
    IFS=',' read -r -a ids <<< "${SWEGYM_INSTANCE_ID}"
    for id in "${ids[@]}"; do
        args+=(--instance-id "${id}")
    done
else
    args+=(--max-tasks "${SWEGYM_MAX_TASKS}")
fi

if [ "${SWEGYM_FORCE_SIF}" = "1" ]; then
    args+=(--force)
fi
if [ "${SWEGYM_REFRESH_DATASET_CACHE}" = "1" ]; then
    args+=(--refresh-dataset-cache)
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_swegym_apptainer_data.py" "${args[@]}"

echo "Prepared SWE-Gym Apptainer data: ${SWEGYM_DATA}"
echo "Prepared SWE-Gym Apptainer SIF dir: ${APPTAINER_IMAGE_DIR}"
