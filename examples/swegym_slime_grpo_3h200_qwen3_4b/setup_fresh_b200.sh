#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

WORK_ROOT="${WORK_ROOT:-}"
RUN_ROOT="${RUN_ROOT:-}"
HF_CHECKPOINT="${HF_CHECKPOINT:-}"
HF_MODEL_ID="${HF_MODEL_ID:-Qwen/Qwen3.5-4B}"
SWEGYM_DATA="${SWEGYM_DATA:-}"
SWEGYM_MAX_TASKS="${SWEGYM_MAX_TASKS:-24}"
SWEGYM_INSTANCE_ID="${SWEGYM_INSTANCE_ID:-}"
SWEGYM_RUNTIME_IMAGE_PREFIX="${SWEGYM_RUNTIME_IMAGE_PREFIX:-polar-swegym-runtime}"
TORCH_DIST_DIR="${TORCH_DIST_DIR:-}"
TP_SIZE="${TP_SIZE:-2}"
PP_SIZE="${PP_SIZE:-1}"
CP_SIZE="${CP_SIZE:-1}"
TRANSFORMER_IMPL="${TRANSFORMER_IMPL:-transformer_engine}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
PREINSTALLED_AGENT_CLI="${PREINSTALLED_AGENT_CLI:-}"
AGENT_NPM_PACKAGE="${AGENT_NPM_PACKAGE:-@openai/codex@0.121.0}"
SKIP_ENV="${SKIP_ENV:-0}"
SKIP_MODEL_DOWNLOAD="${SKIP_MODEL_DOWNLOAD:-0}"
SKIP_DATA="${SKIP_DATA:-0}"
SKIP_CONVERT="${SKIP_CONVERT:-0}"
SKIP_CODEX_INSTALL="${SKIP_CODEX_INSTALL:-0}"

usage() {
    cat <<'EOF'
Usage:
  setup_fresh_b200.sh --work-root PATH --run-root PATH --hf-checkpoint PATH [options]

Required:
  --work-root PATH        Large writable env/artifact/cache directory.
  --run-root PATH         Directory for run outputs.
  --hf-checkpoint PATH    Local Hugging Face checkpoint directory to use/create.

Common options:
  --hf-model-id ID        HF repo to download if checkpoint is missing.
                          Default: Qwen/Qwen3.5-4B
  --swegym-data PATH      Output/input SWE-Gym JSONL path.
                          Default: WORK_ROOT/data/swegym_train_docker_b200.jsonl
  --swegym-max-tasks N    Number of SWE-Gym tasks to prepare when not using exact IDs.
                          Default: 24
  --swegym-instance-id A,B
                          Exact comma-separated SWE-Gym instance IDs.
  --torch-dist-dir PATH   Converted Megatron checkpoint path.
                          Default: WORK_ROOT/checkpoints/Qwen3.5-4B_torch_dist_tp2
  --tp-size N             Tensor parallel size for conversion/training. Default: 2
  --python-version X.Y    Python minor version for the venv. Default: 3.12
  --preinstalled-agent-cli PATH
                          Prefix dir where Codex CLI is installed/read from.
                          Default: WORK_ROOT/agent_cli/codex_0.121.0

Skip switches:
  --skip-env
  --skip-model-download
  --skip-data
  --skip-convert
  --skip-codex-install
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --work-root) WORK_ROOT="$2"; shift 2 ;;
        --run-root) RUN_ROOT="$2"; shift 2 ;;
        --hf-checkpoint) HF_CHECKPOINT="$2"; shift 2 ;;
        --hf-model-id) HF_MODEL_ID="$2"; shift 2 ;;
        --swegym-data) SWEGYM_DATA="$2"; shift 2 ;;
        --swegym-max-tasks) SWEGYM_MAX_TASKS="$2"; shift 2 ;;
        --swegym-instance-id) SWEGYM_INSTANCE_ID="$2"; shift 2 ;;
        --swegym-runtime-image-prefix) SWEGYM_RUNTIME_IMAGE_PREFIX="$2"; shift 2 ;;
        --torch-dist-dir) TORCH_DIST_DIR="$2"; shift 2 ;;
        --tp-size) TP_SIZE="$2"; shift 2 ;;
        --pp-size) PP_SIZE="$2"; shift 2 ;;
        --cp-size) CP_SIZE="$2"; shift 2 ;;
        --transformer-impl) TRANSFORMER_IMPL="$2"; shift 2 ;;
        --python-version) PYTHON_VERSION="$2"; shift 2 ;;
        --preinstalled-agent-cli) PREINSTALLED_AGENT_CLI="$2"; shift 2 ;;
        --agent-npm-package) AGENT_NPM_PACKAGE="$2"; shift 2 ;;
        --skip-env) SKIP_ENV=1; shift ;;
        --skip-model-download) SKIP_MODEL_DOWNLOAD=1; shift ;;
        --skip-data) SKIP_DATA=1; shift ;;
        --skip-convert) SKIP_CONVERT=1; shift ;;
        --skip-codex-install) SKIP_CODEX_INSTALL=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [ -z "${WORK_ROOT}" ] || [ -z "${RUN_ROOT}" ] || [ -z "${HF_CHECKPOINT}" ]; then
    echo "ERROR: --work-root, --run-root, and --hf-checkpoint are required." >&2
    usage >&2
    exit 2
fi

SWEGYM_DATA="${SWEGYM_DATA:-${WORK_ROOT}/data/swegym_train_docker_b200.jsonl}"
TORCH_DIST_DIR="${TORCH_DIST_DIR:-${WORK_ROOT}/checkpoints/Qwen3.5-4B_torch_dist_tp2}"
PREINSTALLED_AGENT_CLI="${PREINSTALLED_AGENT_CLI:-${WORK_ROOT}/agent_cli/codex_0.121.0}"

export WORK_ROOT RUN_ROOT PYTHON_VERSION
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
mkdir -p "${WORK_ROOT}" "${RUN_ROOT}" "${TMPDIR}" "${UV_CACHE_DIR}" "${HF_HOME}" \
    "${HF_HUB_CACHE}" "${TRANSFORMERS_CACHE}" "${PYTHONPYCACHEPREFIX}" \
    "${RAY_TMPDIR}" "${FLASHINFER_WORKSPACE_BASE}" "${MPLCONFIGDIR}" \
    "$(dirname "${HF_CHECKPOINT}")" "$(dirname "${SWEGYM_DATA}")" \
    "$(dirname "${TORCH_DIST_DIR}")" "${PREINSTALLED_AGENT_CLI}"

if [ "${SKIP_ENV}" != "1" ]; then
    WORK_ROOT="${WORK_ROOT}" RUN_ROOT="${RUN_ROOT}" PYTHON_VERSION="${PYTHON_VERSION}" \
        bash "${SCRIPT_DIR}/setup_env_3h200.sh"
fi

PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
if [ ! -x "${PYTHON_BIN}" ]; then
    echo "ERROR: Python env not found after setup: ${PYTHON_BIN}" >&2
    exit 1
fi

if [ "${SKIP_MODEL_DOWNLOAD}" != "1" ] && [ ! -f "${HF_CHECKPOINT}/config.json" ]; then
    uv pip install --python "${PYTHON_BIN}" -U "huggingface_hub[cli]"
    HF_CLI="${PROJECT_ROOT}/.venv/bin/huggingface-cli"
    if [ ! -x "${HF_CLI}" ]; then
        HF_CLI="$(command -v huggingface-cli || true)"
    fi
    if [ -z "${HF_CLI}" ]; then
        echo "ERROR: huggingface-cli was not found after installing huggingface_hub[cli]." >&2
        exit 1
    fi
    "${HF_CLI}" download \
        "${HF_MODEL_ID}" \
        --local-dir "${HF_CHECKPOINT}" \
        --local-dir-use-symlinks False
fi

if [ "${SKIP_CODEX_INSTALL}" != "1" ]; then
    if ! command -v npm >/dev/null 2>&1; then
        echo "ERROR: npm is required to preinstall Codex CLI." >&2
        exit 1
    fi
    npm install -g --prefix "${PREINSTALLED_AGENT_CLI}" --no-audit --no-fund "${AGENT_NPM_PACKAGE}"
fi

if [ "${SKIP_DATA}" != "1" ]; then
    env \
        WORK_ROOT="${WORK_ROOT}" \
        PYTHON_BIN="${PYTHON_BIN}" \
        SWEGYM_DATA="${SWEGYM_DATA}" \
        SWEGYM_MAX_TASKS="${SWEGYM_MAX_TASKS}" \
        SWEGYM_INSTANCE_ID="${SWEGYM_INSTANCE_ID}" \
        SWEGYM_RUNTIME_IMAGE_PREFIX="${SWEGYM_RUNTIME_IMAGE_PREFIX}" \
        bash "${SCRIPT_DIR}/prepare_swegym_docker_data.sh"
fi

if [ "${SKIP_CONVERT}" != "1" ]; then
    env \
        WORK_ROOT="${WORK_ROOT}" \
        HF_CHECKPOINT="${HF_CHECKPOINT}" \
        TORCH_DIST_DIR="${TORCH_DIST_DIR}" \
        TP_SIZE="${TP_SIZE}" \
        PP_SIZE="${PP_SIZE}" \
        CP_SIZE="${CP_SIZE}" \
        NPROC_PER_NODE="${TP_SIZE}" \
        TRANSFORMER_IMPL="${TRANSFORMER_IMPL}" \
        bash "${SCRIPT_DIR}/convert_qwen35_4b.sh"
fi

cat <<EOF
Fresh-server setup complete.

Use these paths for the run script:
  WORK_ROOT=${WORK_ROOT}
  RUN_ROOT=${RUN_ROOT}
  HF_CHECKPOINT=${HF_CHECKPOINT}
  SWEGYM_DATA=${SWEGYM_DATA}
  TORCH_DIST_DIR=${TORCH_DIST_DIR}
  PREINSTALLED_AGENT_CLI=${PREINSTALLED_AGENT_CLI}
EOF
