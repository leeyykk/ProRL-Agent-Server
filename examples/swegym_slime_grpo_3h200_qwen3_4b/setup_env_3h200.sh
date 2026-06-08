#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

WORK_ROOT="${WORK_ROOT:-/work1/yokyung/prorl_agent_server_env}"
RUN_ROOT="${RUN_ROOT:-/home/yokyung/prorl_agent_server_runs/swegym_slime_grpo_3h200}"
VENV_DIR="${VENV_DIR:-${PROJECT_ROOT}/.venv}"
SLIME_DIR="${SLIME_DIR:-${PROJECT_ROOT}/slime}"
MEGATRON_DIR="${MEGATRON_DIR:-${PROJECT_ROOT}/Megatron-LM}"
SLIME_REPO="${SLIME_REPO:-https://github.com/THUDM/slime.git}"
SLIME_REF="${SLIME_REF:-v0.2.4}"
MEGATRON_REPO="${MEGATRON_REPO:-https://github.com/NVIDIA/Megatron-LM.git}"
MEGATRON_REF="${MEGATRON_REF:-main}"

export TMPDIR="${WORK_ROOT}/tmp"
export TEMP="${TMPDIR}"
export TMP="${TMPDIR}"
export UV_CACHE_DIR="${WORK_ROOT}/uv-cache"
export HF_HOME="${WORK_ROOT}/hf-home"
export HF_HUB_CACHE="${WORK_ROOT}/hf-hub-cache"
export TRANSFORMERS_CACHE="${WORK_ROOT}/transformers-cache"
export PYTHONPYCACHEPREFIX="${WORK_ROOT}/pycache"
export RAY_TMPDIR="${WORK_ROOT}/ray"
export FLASHINFER_WORKSPACE_BASE="${WORK_ROOT}/flashinfer"
export MPLCONFIGDIR="${WORK_ROOT}/mplconfig"
mkdir -p "${TMPDIR}" "${UV_CACHE_DIR}" "${HF_HOME}" "${HF_HUB_CACHE}" \
    "${TRANSFORMERS_CACHE}" "${PYTHONPYCACHEPREFIX}" "${RAY_TMPDIR}" \
    "${FLASHINFER_WORKSPACE_BASE}" "${MPLCONFIGDIR}" "${RUN_ROOT}"

require_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "ERROR: required command not found: $1" >&2
        exit 1
    fi
}

clone_if_missing() {
    local name="$1" repo="$2" ref="$3" dest="$4"
    if [ -d "${dest}/.git" ]; then
        echo "${name} checkout exists: ${dest}"
        return
    fi
    if [ -e "${dest}" ]; then
        echo "ERROR: ${name} path exists but is not a git checkout: ${dest}" >&2
        exit 1
    fi
    git clone --branch "${ref}" --depth 1 "${repo}" "${dest}"
}

require_cmd git
require_cmd python3

if ! command -v uv >/dev/null 2>&1; then
    python3 -m pip install --user uv
fi
require_cmd uv

if [ ! -x "${VENV_DIR}/bin/python" ]; then
    uv venv "${VENV_DIR}" --python python3
fi

clone_if_missing "Slime" "${SLIME_REPO}" "${SLIME_REF}" "${SLIME_DIR}"
clone_if_missing "Megatron-LM" "${MEGATRON_REPO}" "${MEGATRON_REF}" "${MEGATRON_DIR}"

uv pip install --python "${VENV_DIR}/bin/python" -e .
uv pip install --python "${VENV_DIR}/bin/python" -e "${SLIME_DIR}"
uv pip install --python "${VENV_DIR}/bin/python" -e "${MEGATRON_DIR}"

bash "${PROJECT_ROOT}/scripts/patch/patch_slime.sh" "${SLIME_DIR}"
bash "${PROJECT_ROOT}/scripts/patch/patch_sglang.sh"

echo "Environment ready:"
echo "  venv:       ${VENV_DIR}"
echo "  work root:  ${WORK_ROOT}"
echo "  run root:   ${RUN_ROOT}"
echo "  slime:      ${SLIME_DIR}"
echo "  megatron:   ${MEGATRON_DIR}"

