#!/usr/bin/env bash
# B200 4-GPU characterization launcher for SWE-Gym + Polar + Slime GRPO.
#
# Default topology:
#   visible GPU 0,1 -> Megatron training, tensor parallel 2
#   visible GPU 2   -> SGLang rollout
#   visible GPU 3   -> left unused as headroom unless CUDA_VISIBLE_DEVICES is changed
#
# Override the worker sweep knobs directly:
#   MAX_INIT_WORKERS=4 MAX_RUN_WORKERS=4 MAX_EVAL_WORKERS=2 \
#     bash examples/swegym_slime_grpo/run_b200_characterization.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

SHARED_ROOT="${SHARED_ROOT:-/home/korea_bupj/vialab/yokyung}"
case "${PROJECT_ROOT}" in
    "${SHARED_ROOT}"/*) ;;
    *)
        if [ "${STRICT_SHARED_ROOT:-1}" = "1" ]; then
            echo "ERROR: project root is outside SHARED_ROOT." >&2
            echo "  PROJECT_ROOT=${PROJECT_ROOT}" >&2
            echo "  SHARED_ROOT=${SHARED_ROOT}" >&2
            echo "Set SHARED_ROOT=/path/to/shared/root or STRICT_SHARED_ROOT=0 to override." >&2
            exit 1
        fi
        ;;
esac

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_ID="${RUN_ID:-b200_gs4_20groups_${timestamp}}"
RUN_DIR="${RUN_DIR:-${PROJECT_ROOT}/tmp/swegym_slime_grpo_b200/${RUN_ID}}"
SCRATCH_ROOT="${SCRATCH_ROOT:-${SHARED_ROOT}/prorl_agent_server_env/runtime/polar_b200}"
mkdir -p "${RUN_DIR}" "${SCRATCH_ROOT}"

export TMPDIR="${TMPDIR:-${SCRATCH_ROOT}/tmp}"
export TMP="${TMP:-${TMPDIR}}"
export TEMP="${TEMP:-${TMPDIR}}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${SCRATCH_ROOT}/uv_cache}"
export HF_HOME="${HF_HOME:-${SCRATCH_ROOT}/hf_home}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${SCRATCH_ROOT}/xdg_cache}"
export RAY_TEMP_DIR="${RAY_TEMP_DIR:-${RUN_DIR}/ray_tmp}"
export PRORL_STOP_EXISTING_RAY="${PRORL_STOP_EXISTING_RAY:-0}"
export PRORL_RAY_STOP_ON_EXIT="${PRORL_RAY_STOP_ON_EXIT:-0}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${SCRATCH_ROOT}/apptainer_cache}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-${SCRATCH_ROOT}/apptainer_tmp}"
mkdir -p \
    "${TMPDIR}" \
    "${UV_CACHE_DIR}" \
    "${HF_HOME}" \
    "${TRANSFORMERS_CACHE}" \
    "${XDG_CACHE_HOME}" \
    "${RAY_TEMP_DIR}" \
    "${APPTAINER_CACHEDIR}" \
    "${APPTAINER_TMPDIR}"

SOURCE_PROMPT_DATA="${SOURCE_PROMPT_DATA:-/home/korea_bupj/vialab/yokyung/prorl_agent_server_env/data/swegym_train_apptainer_b200.jsonl}"
HF_CHECKPOINT="${HF_CHECKPOINT:-/home/korea_bupj/vialab/yokyung/huggingface_models/Qwen3.5-4B}"
APPTAINER_IMAGE_DIR="${APPTAINER_IMAGE_DIR:-/home/korea_bupj/vialab/yokyung/prorl_agent_server_env/apptainer_sifs/swegym_b200}"
TORCH_DIST_DIR="${TORCH_DIST_DIR:-/home/korea_bupj/vialab/yokyung/prorl_agent_server_env/checkpoints/Qwen3.5-4B_torch_dist}"
SAVE_DIR="${SAVE_DIR:-${PROJECT_ROOT}/tmp/ckpt/swegym_slime_grpo_qwen35_4b/${RUN_ID}}"
AGENT_CLI_DIR="${AGENT_CLI_DIR:-/home/korea_bupj/vialab/yokyung/prorl_agent_server_env/swegym_agent_cli/opt_node}"
AGENT_HARNESS="${AGENT_HARNESS:-codex}"
PROMPT_DATA="${PROMPT_DATA:-${RUN_DIR}/swegym_train_20groups.jsonl}"
ROLLOUT_SAVE_DIR="${ROLLOUT_SAVE_DIR:-${RUN_DIR}/rollout_results}"

TASK_GROUPS="${TASK_GROUPS:-20}"
if [ ! -f "${SOURCE_PROMPT_DATA}" ]; then
    echo "ERROR: source dataset JSONL not found: ${SOURCE_PROMPT_DATA}" >&2
    exit 1
fi
if [ ! -d "${HF_CHECKPOINT}" ]; then
    echo "ERROR: HF checkpoint directory not found: ${HF_CHECKPOINT}" >&2
    exit 1
fi
if [ ! -d "${APPTAINER_IMAGE_DIR}" ]; then
    echo "ERROR: Apptainer SIF directory not found: ${APPTAINER_IMAGE_DIR}" >&2
    exit 1
fi

python3 - "${SOURCE_PROMPT_DATA}" "${PROMPT_DATA}" "${TASK_GROUPS}" <<'PY'
from pathlib import Path
import sys

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
limit = int(sys.argv[3])
rows = []
with src.open(encoding="utf-8") as fh:
    for line in fh:
        if line.strip():
            rows.append(line.rstrip("\n"))
        if len(rows) >= limit:
            break
if len(rows) < limit:
    raise SystemExit(f"Requested {limit} task groups but only found {len(rows)} rows in {src}")
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text("\n".join(rows) + "\n", encoding="utf-8")
print(f"Wrote {len(rows)} task groups to {dst}")
PY

if [ "${PREPARE_AGENT_CLI:-1}" = "1" ]; then
    missing_cli=0
    required_bins="node npm npx ${AGENT_HARNESS}"
    if [ "${AGENT_HARNESS}" = "claude_code" ]; then
        required_bins="node npm npx claude"
    elif [ "${AGENT_HARNESS}" = "qwen_code" ]; then
        required_bins="node npm npx qwen"
    fi
    for bin_name in ${required_bins}; do
        if [ ! -x "${AGENT_CLI_DIR}/bin/${bin_name}" ]; then
            missing_cli=1
            break
        fi
    done
    if [ "${missing_cli}" = "1" ]; then
        PYTHON_FOR_CLI="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python3}"
        if [ ! -x "${PYTHON_FOR_CLI}" ]; then
            PYTHON_FOR_CLI="$(command -v python3 || command -v python)"
        fi
        "${PYTHON_FOR_CLI}" "${SCRIPT_DIR}/prepare_apptainer_images.py" \
            --cli-only \
            --agent-cli-dir "${AGENT_CLI_DIR}" \
            --image-dir "${APPTAINER_IMAGE_DIR}" \
            --cache-dir "${APPTAINER_CACHEDIR}" \
            --tmp-dir "${APPTAINER_TMPDIR}"
    else
        echo "Shared agent CLI directory already exists: ${AGENT_CLI_DIR}"
    fi
fi

export RUN_ID RUN_DIR PROMPT_DATA ROLLOUT_SAVE_DIR
export HF_CHECKPOINT TORCH_DIST_DIR REF_LOAD="${TORCH_DIST_DIR}" SAVE_DIR
export APPTAINER_IMAGE_DIR AGENT_CLI_DIR AGENT_HARNESS
export PREPARE_DATA=0 PREPARE_IMAGES=0
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
export RAY_NUM_GPUS="${RAY_NUM_GPUS:-3}"
export TRAIN_GPUS="${TRAIN_GPUS:-0,1}"
export ROLLOUT_GPUS="${ROLLOUT_GPUS:-2}"

export ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-2}"
export ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-1}"
export ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}"
export TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-2}"
export CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-1}"

export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-1}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-4}"
export NUM_EPOCH="${NUM_EPOCH:-1}"
export NUM_STEPS_PER_ROLLOUT="${NUM_STEPS_PER_ROLLOUT:-1}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-20}"

export POLAR_MAX_ASYNC_LEVEL="${POLAR_MAX_ASYNC_LEVEL:-1}"
export MAX_INIT_WORKERS="${MAX_INIT_WORKERS:-4}"
export MAX_RUN_WORKERS="${MAX_RUN_WORKERS:-4}"
export MAX_POSTRUN_WORKERS="${MAX_POSTRUN_WORKERS:-${MAX_EVAL_WORKERS:-2}}"
export POLAR_MIN_COMPLETE_ACCEPT_FRACTION="${POLAR_MIN_COMPLETE_ACCEPT_FRACTION:-0.5}"
export POLAR_REQUEST_TIMEOUT="${POLAR_REQUEST_TIMEOUT:-2400}"

export SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-32768}"
export ROLLOUT_MAX_PROMPT_LEN="${ROLLOUT_MAX_PROMPT_LEN:-28672}"
export ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-4096}"
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-32768}"
export SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.70}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:2048,expandable_segments:True}"
export USE_DYNAMIC_BATCH_SIZE="${USE_DYNAMIC_BATCH_SIZE:-1}"
export USE_WANDB="${USE_WANDB:-0}"
export WANDB_GROUP="${WANDB_GROUP:-b200-gs4-characterization}"

echo "=== B200 characterization config ==="
echo "RUN_ID=${RUN_ID}"
echo "agent harness=${AGENT_HARNESS}"
echo "agent CLI dir=${AGENT_CLI_DIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (Ray GPUs: ${RAY_NUM_GPUS})"
echo "train GPUs=${TRAIN_GPUS}, rollout GPUs=${ROLLOUT_GPUS}"
echo "task groups=${TASK_GROUPS}, group size=${N_SAMPLES_PER_PROMPT}, rollout batch=${ROLLOUT_BATCH_SIZE}"
echo "workers: init=${MAX_INIT_WORKERS}, run=${MAX_RUN_WORKERS}, postrun/eval=${MAX_POSTRUN_WORKERS}, async=${POLAR_MAX_ASYNC_LEVEL}"
echo "tokens: context=${SGLANG_CONTEXT_LENGTH}, prompt=${ROLLOUT_MAX_PROMPT_LEN}, response=${ROLLOUT_MAX_RESPONSE_LEN}, train max/gpu=${MAX_TOKENS_PER_GPU}"
echo "run dir=${RUN_DIR}"
echo "scratch root=${SCRATCH_ROOT}"
echo "ray temp=${RAY_TEMP_DIR}"
echo "ray cleanup: stop_existing=${PRORL_STOP_EXISTING_RAY}, stop_on_exit=${PRORL_RAY_STOP_ON_EXIT}"

bash "${SCRIPT_DIR}/launch_e2e.sh"

python3 - "${ROLLOUT_SAVE_DIR}" "${RUN_DIR}/timing_summary.csv" <<'PY'
from __future__ import annotations

from pathlib import Path
import csv
import json
import statistics
import sys

root = Path(sys.argv[1])
csv_path = Path(sys.argv[2])
records = []

def collect(obj):
    if not isinstance(obj, dict):
        return
    timing = obj.get("timing")
    if isinstance(timing, dict) and timing:
        records.append(timing)
    result = obj.get("result")
    if isinstance(result, dict):
        collect(result)

for path in root.rglob("*.json"):
    try:
        collect(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        continue

if not records:
    print(f"No timing JSON records found under {root}")
    raise SystemExit(0)

keys = ["register_to_init_queue_ms", "init_ms", "run_ms", "postrun_ms"]
csv_path.parent.mkdir(parents=True, exist_ok=True)
with csv_path.open("w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=keys)
    writer.writeheader()
    for rec in records:
        writer.writerow({key: rec.get(key, 0.0) for key in keys})

print(f"Timing records: {len(records)}")
for key in keys:
    values = [float(rec.get(key, 0.0)) / 1000.0 for rec in records]
    print(
        f"{key.removesuffix('_ms')}: "
        f"mean={statistics.mean(values):.2f}s "
        f"p50={statistics.median(values):.2f}s "
        f"max={max(values):.2f}s"
    )
print(f"Wrote timing CSV: {csv_path}")
PY
