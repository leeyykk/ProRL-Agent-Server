#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

WORK_ROOT="${WORK_ROOT:-}"
RUN_ROOT="${RUN_ROOT:-}"
MAIN_FIGURES_DIR="${MAIN_FIGURES_DIR:-}"
HF_CHECKPOINT="${HF_CHECKPOINT:-}"
SWEGYM_DATA="${SWEGYM_DATA:-}"
TORCH_DIST_DIR="${TORCH_DIST_DIR:-}"
PREINSTALLED_AGENT_CLI="${PREINSTALLED_AGENT_CLI:-}"
RUNTIME_BACKEND="${RUNTIME_BACKEND:-docker}"
TRAIN_GPUS="${TRAIN_GPUS:-}"
ROLLOUT_GPU="${ROLLOUT_GPU:-}"
SWEEP_ID="${SWEEP_ID:-}"
FIGURE_SUFFIX="${FIGURE_SUFFIX:-accept10_sp16k_te_24rollouts}"
STEPS_LABEL="${STEPS_LABEL:-24rollouts}"
NUM_ROLLOUT="${NUM_ROLLOUT:-24}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-4}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-1}"
NUM_STEPS_PER_ROLLOUT="${NUM_STEPS_PER_ROLLOUT:-1}"
POLAR_MIN_COMPLETE_ACCEPT_FRACTION="${POLAR_MIN_COMPLETE_ACCEPT_FRACTION:-1.0}"
POLAR_MAX_ASYNC_LEVEL="${POLAR_MAX_ASYNC_LEVEL:-4}"
POLAR_MAX_POSTRUN_WORKERS="${POLAR_MAX_POSTRUN_WORKERS:-1}"
SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-16384}"
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-1024}"
ROLLOUT_MAX_PROMPT_LEN="${ROLLOUT_MAX_PROMPT_LEN:-12288}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-16384}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.70}"
TP_SIZE="${TP_SIZE:-2}"
CP_SIZE="${CP_SIZE:-1}"
SEQUENCE_PARALLEL="${SEQUENCE_PARALLEL:-1}"
TRANSFORMER_IMPL="${TRANSFORMER_IMPL:-transformer_engine}"
HOST_MIN_AVAILABLE_GIB="${HOST_MIN_AVAILABLE_GIB:-120}"
MAX_PRELAUNCH_GPU_MEM_MIB="${MAX_PRELAUNCH_GPU_MEM_MIB:-1024}"
POLL_SECONDS="${POLL_SECONDS:-60}"
RUN_NOW="${RUN_NOW:-0}"

usage() {
    cat <<'EOF'
Usage:
  run_qwen35_codex_worker_sweep_b200.sh \
    --work-root PATH --run-root PATH --hf-checkpoint PATH \
    --swegym-data PATH --torch-dist-dir PATH \
    --train-gpus 4,6 --rollout-gpu 1

This waits for the selected GPUs and then runs the 4 worker configs:
  init=1 run=4
  init=4 run=1
  init=4 run=4
  init=4 run=8

Defaults mirror the current Qwen3.5/Codex/SP/TE 16K run:
  accept fraction: 1.0
  NUM_ROLLOUT: 24
  N_SAMPLES_PER_PROMPT: 4
  TP_SIZE: 2
  SEQUENCE_PARALLEL: 1
  TRANSFORMER_IMPL: transformer_engine

Options:
  --work-root PATH
  --run-root PATH
  --main-figures-dir PATH
  --hf-checkpoint PATH
  --swegym-data PATH
  --torch-dist-dir PATH
  --preinstalled-agent-cli PATH
  --runtime-backend docker|apptainer
  --train-gpus CSV       Example: 4,6
  --rollout-gpu GPU      Example: 1
  --num-rollout N
  --accept-fraction X    Default: 1.0
  --poll-seconds N
  --host-min-available-gib N
  --max-prelaunch-gpu-mem-mib N
  --run-now              Do one preflight check and fail if resources are busy.
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --work-root) WORK_ROOT="$2"; shift 2 ;;
        --run-root) RUN_ROOT="$2"; shift 2 ;;
        --main-figures-dir) MAIN_FIGURES_DIR="$2"; shift 2 ;;
        --hf-checkpoint) HF_CHECKPOINT="$2"; shift 2 ;;
        --swegym-data) SWEGYM_DATA="$2"; shift 2 ;;
        --torch-dist-dir) TORCH_DIST_DIR="$2"; shift 2 ;;
        --preinstalled-agent-cli) PREINSTALLED_AGENT_CLI="$2"; shift 2 ;;
        --runtime-backend) RUNTIME_BACKEND="$2"; shift 2 ;;
        --train-gpus) TRAIN_GPUS="$2"; shift 2 ;;
        --rollout-gpu) ROLLOUT_GPU="$2"; shift 2 ;;
        --sweep-id) SWEEP_ID="$2"; shift 2 ;;
        --figure-suffix) FIGURE_SUFFIX="$2"; shift 2 ;;
        --num-rollout) NUM_ROLLOUT="$2"; shift 2 ;;
        --n-samples-per-prompt) N_SAMPLES_PER_PROMPT="$2"; shift 2 ;;
        --accept-fraction) POLAR_MIN_COMPLETE_ACCEPT_FRACTION="$2"; shift 2 ;;
        --poll-seconds) POLL_SECONDS="$2"; shift 2 ;;
        --host-min-available-gib) HOST_MIN_AVAILABLE_GIB="$2"; shift 2 ;;
        --max-prelaunch-gpu-mem-mib) MAX_PRELAUNCH_GPU_MEM_MIB="$2"; shift 2 ;;
        --run-now) RUN_NOW=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [ -z "${WORK_ROOT}" ] || [ -z "${RUN_ROOT}" ] || [ -z "${HF_CHECKPOINT}" ] \
    || [ -z "${SWEGYM_DATA}" ] || [ -z "${TORCH_DIST_DIR}" ] \
    || [ -z "${TRAIN_GPUS}" ] || [ -z "${ROLLOUT_GPU}" ]; then
    echo "ERROR: --work-root, --run-root, --hf-checkpoint, --swegym-data, --torch-dist-dir, --train-gpus, and --rollout-gpu are required." >&2
    usage >&2
    exit 2
fi
case "${RUNTIME_BACKEND}" in
    docker|apptainer) ;;
    *) echo "ERROR: --runtime-backend must be docker or apptainer, got: ${RUNTIME_BACKEND}" >&2; exit 2 ;;
esac

if [ ! -f "${HF_CHECKPOINT}/config.json" ]; then
    echo "ERROR: Hugging Face checkpoint is missing or incomplete: ${HF_CHECKPOINT}" >&2
    echo "       Expected file: ${HF_CHECKPOINT}/config.json" >&2
    echo "       Either pass the actual model directory with --hf-checkpoint, or rerun setup without --skip-model-download." >&2
    exit 1
fi
if [ ! -f "${SWEGYM_DATA}" ]; then
    echo "ERROR: SWE-Gym data JSONL not found: ${SWEGYM_DATA}" >&2
    echo "       Rerun setup without --skip-data, or pass the correct --swegym-data path." >&2
    exit 1
fi
if [ ! -d "${TORCH_DIST_DIR}" ]; then
    echo "ERROR: converted Megatron torch_dist checkpoint not found: ${TORCH_DIST_DIR}" >&2
    echo "       Rerun setup without --skip-convert, or pass the correct --torch-dist-dir path." >&2
    exit 1
fi

MAIN_FIGURES_DIR="${MAIN_FIGURES_DIR:-${RUN_ROOT}/main_figures}"
SWEEP_ID="${SWEEP_ID:-qwen35_codex_worker_sweep_accept10_sp16k_te_24rollouts_$(date -u +%Y%m%dT%H%M%SZ)}"
SWEEP_DIR="${SWEEP_DIR:-${RUN_ROOT}/${SWEEP_ID}}"
PREINSTALLED_AGENT_CLI="${PREINSTALLED_AGENT_CLI:-${WORK_ROOT}/agent_cli/codex_0.121.0}"

mkdir -p "${SWEEP_DIR}" "${MAIN_FIGURES_DIR}"

CONFIGS=(
    "init1_run4 1 4"
    "init4_run1 4 1"
    "init4_run4 4 4"
    "init4_run8 4 8"
)

read_gpu_field() {
    local gpu_csv="$1"
    local gpu_idx="$2"
    local field_idx="$3"
    awk -F, -v gpu="${gpu_idx}" -v field="${field_idx}" \
        '$1 ~ "^ *" gpu " *$" {gsub(/ /,"",$field); print $field}' <<<"${gpu_csv}"
}

selected_gpus() {
    printf "%s,%s" "${TRAIN_GPUS}" "${ROLLOUT_GPU}" | tr ',' '\n' | awk 'NF {gsub(/ /,""); if (!seen[$0]++) print $0}'
}

wait_for_resources() {
    echo "Waiting for TRAIN_GPUS=${TRAIN_GPUS}, ROLLOUT_GPU=${ROLLOUT_GPU}, MemAvailable>=${HOST_MIN_AVAILABLE_GIB} GiB" >&2
    while true; do
        local gpu_csv mem_avail_gib stamp ready
        gpu_csv="$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)"
        mem_avail_gib="$(awk '/MemAvailable:/ {printf "%.0f", $2 / 1024 / 1024}' /proc/meminfo)"
        stamp="$(date -Is)"
        echo "${stamp} $(tr '\n' ';' <<<"${gpu_csv}") mem_avail=${mem_avail_gib}GiB" >&2

        ready=1
        while read -r gpu; do
            local used
            used="$(read_gpu_field "${gpu_csv}" "${gpu}" 2)"
            if [ -z "${used}" ] || [ "${used}" -gt "${MAX_PRELAUNCH_GPU_MEM_MIB}" ]; then
                ready=0
            fi
        done < <(selected_gpus)
        if [ "${mem_avail_gib}" -lt "${HOST_MIN_AVAILABLE_GIB}" ]; then
            ready=0
        fi
        if [ "${ready}" = "1" ]; then
            return 0
        fi
        if [ "${RUN_NOW}" = "1" ]; then
            echo "ERROR: resources are not free and --run-now was set." >&2
            return 1
        fi
        sleep "${POLL_SECONDS}"
    done
}

MANIFEST="${SWEEP_DIR}/worker_sweep_manifest.tsv"
printf "label\tinit_workers\trun_workers\trun_dir\ttrain_gpus\trollout_gpu\tstatus\texit_code\tfigure\n" > "${MANIFEST}"

for config in "${CONFIGS[@]}"; do
    read -r label init_workers run_workers <<<"${config}"
    wait_for_resources 2> >(tee -a "${SWEEP_DIR}/${label}_wait.log" >&2)
    run_id="codex_qwen35_4b_exact_workers_${label}_${FIGURE_SUFFIX}_gs4_async4_${STEPS_LABEL}_$(date -u +%Y%m%dT%H%M%SZ)"
    run_dir="${RUN_ROOT}/${run_id}"
    figure="${MAIN_FIGURES_DIR}/codex_qwen35_4b_exact_workers_${label}_${FIGURE_SUFFIX}.svg"

    echo "Launching ${label}: init=${init_workers}, run=${run_workers}, train_gpus=${TRAIN_GPUS}, rollout_gpu=${ROLLOUT_GPU}"
    printf "%s\t%s\t%s\t%s\t%s\t%s\trunning\t\t%s\n" "${label}" "${init_workers}" "${run_workers}" "${run_dir}" "${TRAIN_GPUS}" "${ROLLOUT_GPU}" "${figure}" >> "${MANIFEST}"

    env \
        WORK_ROOT="${WORK_ROOT}" \
        RUN_ROOT="${RUN_ROOT}" \
        HF_CHECKPOINT="${HF_CHECKPOINT}" \
        TORCH_DIST_DIR="${TORCH_DIST_DIR}" \
        REF_LOAD="${TORCH_DIST_DIR}" \
        SWEGYM_DATA="${SWEGYM_DATA}" \
        MODEL_FLAVOR=qwen35_4b \
        NUM_ROLLOUT="${NUM_ROLLOUT}" \
        ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE}" \
        N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT}" \
        NUM_STEPS_PER_ROLLOUT="${NUM_STEPS_PER_ROLLOUT}" \
        POLAR_MAX_POSTRUN_WORKERS="${POLAR_MAX_POSTRUN_WORKERS}" \
        POLAR_MAX_ASYNC_LEVEL="${POLAR_MAX_ASYNC_LEVEL}" \
        POLAR_MIN_COMPLETE_ACCEPT_FRACTION="${POLAR_MIN_COMPLETE_ACCEPT_FRACTION}" \
        SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH}" \
        ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN}" \
        ROLLOUT_MAX_PROMPT_LEN="${ROLLOUT_MAX_PROMPT_LEN}" \
        MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU}" \
        USE_DYNAMIC_BATCH_SIZE=0 \
        TP_SIZE="${TP_SIZE}" \
        CP_SIZE="${CP_SIZE}" \
        SEQUENCE_PARALLEL="${SEQUENCE_PARALLEL}" \
        TRANSFORMER_IMPL="${TRANSFORMER_IMPL}" \
        AGENT_HARNESS=codex \
        PREINSTALLED_AGENT_CLI="${PREINSTALLED_AGENT_CLI}" \
        RUNTIME_BACKEND="${RUNTIME_BACKEND}" \
        AGENT_NPM_PACKAGE="@openai/codex@0.121.0" \
        SAVE_INTERVAL=1000000 \
        KEEP_CHECKPOINTS=0 \
        KEEP_RUNTIME_ARTIFACTS=0 \
        MAX_PRELAUNCH_GPU_MEM_MIB="${MAX_PRELAUNCH_GPU_MEM_MIB}" \
        HOST_MIN_AVAILABLE_GIB="${HOST_MIN_AVAILABLE_GIB}" \
        RUN_ID="${run_id}" \
        RUN_DIR="${run_dir}" \
        SELECTED_GPUS="${TRAIN_GPUS},${ROLLOUT_GPU}" \
        TRAIN_GPUS="${TRAIN_GPUS}" \
        ROLLOUT_GPU="${ROLLOUT_GPU}" \
        POLAR_MAX_INIT_WORKERS="${init_workers}" \
        POLAR_MAX_RUN_WORKERS="${run_workers}" \
        SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC}" \
        bash examples/swegym_slime_grpo_3h200_qwen3_4b/launch_qwen3_4b_3h200.sh
    status=$?

    if [ -d "${run_dir}" ]; then
        .venv/bin/python examples/swegym_slime_grpo_3h200_qwen3_4b/plot_exact_characterization.py \
            "${run_dir}" \
            --output "${run_dir}/exact_characterization.svg" || true
        .venv/bin/python examples/swegym_slime_grpo_3h200_qwen3_4b/plot_exact_characterization.py \
            "${run_dir}" \
            --output "${figure}" || true
    fi

    final_status="complete"
    if [ "${status}" -ne 0 ]; then
        final_status="failed"
    fi
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "${label}" "${init_workers}" "${run_workers}" "${run_dir}" "${TRAIN_GPUS}" "${ROLLOUT_GPU}" "${final_status}" "${status}" "${figure}" >> "${MANIFEST}"
done

echo "Worker sweep complete: ${SWEEP_DIR}"
echo "Manifest: ${MANIFEST}"
