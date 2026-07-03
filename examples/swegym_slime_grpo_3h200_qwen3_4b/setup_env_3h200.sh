#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

WORK_ROOT="${WORK_ROOT:-/work1/yokyung/prorl_agent_server_env}"
RUN_ROOT="${RUN_ROOT:-/home/yokyung/prorl_agent_server_runs/swegym_slime_grpo_3h200}"
VENV_DIR="${VENV_DIR:-${PROJECT_ROOT}/.venv}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
PYTHON_BOOTSTRAP="${PYTHON_BOOTSTRAP:-python3}"
SLIME_DIR="${SLIME_DIR:-${PROJECT_ROOT}/slime}"
MEGATRON_DIR="${MEGATRON_DIR:-${PROJECT_ROOT}/Megatron-LM}"
SLIME_REPO="${SLIME_REPO:-https://github.com/THUDM/slime.git}"
SLIME_REF="${SLIME_REF:-v0.2.4}"
MEGATRON_REPO="${MEGATRON_REPO:-https://github.com/NVIDIA/Megatron-LM.git}"
MEGATRON_REF="${MEGATRON_REF:-3714d81d418c9f1bca4594fc35f9e8289f652862}"
MEGATRON_PATCH_VERSION="${MEGATRON_PATCH_VERSION:-v0.5.9}"

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

ensure_git_commit() {
    local name="$1" repo="$2" ref="$3" dest="$4"
    if [ ! -d "${dest}/.git" ]; then
        git clone --depth 1 "${repo}" "${dest}"
    fi
    local current
    current="$(git -C "${dest}" rev-parse HEAD)"
    if [ "${current}" = "${ref}" ]; then
        echo "${name} checkout is pinned: ${dest} @ ${ref}"
        return
    fi
    if ! git -C "${dest}" diff --quiet || ! git -C "${dest}" diff --cached --quiet; then
        echo "ERROR: ${name} checkout is dirty and not at required ref ${ref}: ${dest}" >&2
        echo "       Please move or clean that dependency checkout, then rerun setup." >&2
        exit 1
    fi
    echo "Updating ${name} checkout to required ref ${ref}: ${dest}"
    git -C "${dest}" fetch --depth 1 origin "${ref}"
    git -C "${dest}" checkout --detach FETCH_HEAD
}

apply_git_patch() {
    local name="$1" dest="$2" patch_file="$3"
    if git -C "${dest}" apply --check "${patch_file}" 2>/dev/null; then
        echo "Applying ${name} patch: ${patch_file}"
        git -C "${dest}" apply "${patch_file}"
    elif git -C "${dest}" apply --reverse --check "${patch_file}" 2>/dev/null; then
        echo "${name} patch already applied: ${patch_file}"
    else
        echo "Applying ${name} patch with 3-way merge: ${patch_file}"
        git -C "${dest}" apply --3way "${patch_file}"
        if grep -R -n '^<<<<<<< ' "${dest}"; then
            echo "ERROR: ${name} patch left merge conflicts in ${dest}" >&2
            exit 1
        fi
    fi
}

require_cmd git
require_cmd "${PYTHON_BOOTSTRAP}"

if ! command -v uv >/dev/null 2>&1; then
    "${PYTHON_BOOTSTRAP}" -m pip install --user uv
fi
require_cmd uv

if [ ! -x "${VENV_DIR}/bin/python" ]; then
    uv venv "${VENV_DIR}" --python "${PYTHON_VERSION}"
fi

actual_python_version="$("${VENV_DIR}/bin/python" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
if [ "${actual_python_version}" != "${PYTHON_VERSION}" ]; then
    cat >&2 <<EOF
ERROR: ${VENV_DIR} uses Python ${actual_python_version}, but this setup expects Python ${PYTHON_VERSION}.

Remove the stale venv and rerun setup, for example:
  rm -rf ${VENV_DIR}
  PYTHON_VERSION=${PYTHON_VERSION} bash ${BASH_SOURCE[0]}

This matters because several ML wheels used by SGLang/Megatron are not reliably
available for newer Python versions.
EOF
    exit 1
fi

clone_if_missing "Slime" "${SLIME_REPO}" "${SLIME_REF}" "${SLIME_DIR}"
ensure_git_commit "Megatron-LM" "${MEGATRON_REPO}" "${MEGATRON_REF}" "${MEGATRON_DIR}"

uv pip install --python "${VENV_DIR}/bin/python" -e ".[swebench]"
uv pip install --python "${VENV_DIR}/bin/python" -e "${SLIME_DIR}"
uv pip install --python "${VENV_DIR}/bin/python" -e "${MEGATRON_DIR}"
uv pip install --python "${VENV_DIR}/bin/python" --prerelease=allow "sglang[all]==0.5.10"
uv pip install --python "${VENV_DIR}/bin/python" \
    "mbridge @ git+https://github.com/ISEEKYAN/mbridge.git@89eb10887887bc74853f89a4de258c0702932a1c" \
    --no-deps
uv pip install --python "${VENV_DIR}/bin/python" "numpy<2"
uv pip install --python "${VENV_DIR}/bin/python" "scipy<1.17"
uv pip install --python "${VENV_DIR}/bin/python" "flash-linear-attention~=0.3.2"
uv pip install --python "${VENV_DIR}/bin/python" "nvidia-cuda-nvrtc-cu12"

bash "${PROJECT_ROOT}/scripts/patch/patch_slime.sh" "${SLIME_DIR}"
apply_git_patch \
    "Slime Megatron compatibility" \
    "${MEGATRON_DIR}" \
    "${SLIME_DIR}/docker/patch/${MEGATRON_PATCH_VERSION}/megatron.patch"
bash "${PROJECT_ROOT}/scripts/patch/patch_sglang.sh"

echo "Environment ready:"
echo "  venv:       ${VENV_DIR}"
echo "  work root:  ${WORK_ROOT}"
echo "  run root:   ${RUN_ROOT}"
echo "  slime:      ${SLIME_DIR}"
echo "  megatron:   ${MEGATRON_DIR}"
