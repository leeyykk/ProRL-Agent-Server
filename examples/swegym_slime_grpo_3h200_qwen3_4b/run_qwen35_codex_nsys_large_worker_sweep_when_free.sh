#!/usr/bin/env bash
set -euo pipefail

GPU_CANDIDATES="${GPU_CANDIDATES:-0,1,2,3}"
MAX_USED_MIB="${MAX_USED_MIB:-1024}"
MAX_UTIL_PCT="${MAX_UTIL_PCT:-10}"
POLL_SECONDS="${POLL_SECONDS:-60}"
WORK_ROOT="${WORK_ROOT:-${HOME}/prorl_agent_server_env}"
RUN_ROOT="${RUN_ROOT:-${HOME}/prorl_agent_server_runs/swegym_slime_grpo_3h200}"
HF_CHECKPOINT="${HF_CHECKPOINT:-${WORK_ROOT}/models/Qwen3.5-4B}"
SWEGYM_DATA="${SWEGYM_DATA:-${WORK_ROOT}/data/swegym_train_docker_3h200.jsonl}"
TORCH_DIST_DIR="${TORCH_DIST_DIR:-${WORK_ROOT}/checkpoints/Qwen3.5-4B_torch_dist_tp2}"
PREINSTALLED_AGENT_CLI="${PREINSTALLED_AGENT_CLI:-${WORK_ROOT}/agent_cli/codex_0.121.0}"
PROFILE_ROOT="${PROFILE_ROOT:-${HOME}/profiled_runs}"
SWEEP_STEM="${SWEEP_STEM:-qwen35_large_worker_nsys_$(date -u +%Y%m%dT%H%M%SZ)}"
PROFILE_DIR="${PROFILE_DIR:-${PROFILE_ROOT}/${SWEEP_STEM}}"
LOG="${LOG:-${PROFILE_DIR}/sweep.log}"
NSYS_TRACE="${NSYS_TRACE:-cuda,nvtx}"
NSYS_SAMPLE="${NSYS_SAMPLE:-none}"
NSYS_EXTRA_ARGS="${NSYS_EXTRA_ARGS:-}"
NSYS_BIN="${NSYS_BIN:-nsys}"
RAY_NUM_CPUS_FOR_NSYS="${RAY_NUM_CPUS_FOR_NSYS:-64}"
KEEP_RUNTIME_ARTIFACTS="${KEEP_RUNTIME_ARTIFACTS:-0}"

mkdir -p "${PROFILE_DIR}"
if ! [ -x "${NSYS_BIN}" ] && ! command -v "${NSYS_BIN}" >/dev/null 2>&1; then
    echo "ERROR: Nsight Systems CLI not found or executable: ${NSYS_BIN}" >&2
    exit 1
fi
echo "$(date -Is) large-worker Nsight sweep watcher started; candidates=${GPU_CANDIDATES}" | tee -a "${LOG}"
echo "$(date -Is) profile_dir=${PROFILE_DIR}" | tee -a "${LOG}"
echo "$(date -Is) nsys_bin=${NSYS_BIN}" | tee -a "${LOG}"
"${NSYS_BIN}" --version 2>&1 | sed 's/^/nsys_version=/' | tee -a "${LOG}"
echo "$(date -Is) applying repository SGLang compatibility/profiling patch" | tee -a "${LOG}"
bash scripts/patch/patch_sglang.sh >>"${LOG}" 2>&1

free_gpu_list() {
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits |
        awk -F, -v candidates="${GPU_CANDIDATES}" -v max_mem="${MAX_USED_MIB}" -v max_util="${MAX_UTIL_PCT}" '
            BEGIN {
                split(candidates, ids, ",")
                for (i in ids) {
                    gsub(/ /, "", ids[i])
                    keep[ids[i]] = 1
                }
            }
            {
                idx = $1
                mem = $2
                util = $3
                gsub(/ /, "", idx)
                gsub(/ /, "", mem)
                gsub(/ /, "", util)
                if ((idx in keep) && (mem + 0) <= (max_mem + 0) && (util + 0) <= (max_util + 0)) {
                    print idx
                }
            }
        '
}

wait_for_three_gpus() {
    while true; do
        mapfile -t free_gpus < <(free_gpu_list)
        echo "$(date -Is) free_gpus=${free_gpus[*]:-none}" | tee -a "${LOG}"
        if [ "${#free_gpus[@]}" -ge 3 ]; then
            SELECTED_GPUS_CHOSEN="${free_gpus[0]},${free_gpus[1]},${free_gpus[2]}"
            TRAIN_GPUS_CHOSEN="${free_gpus[0]},${free_gpus[1]}"
            ROLLOUT_GPU_CHOSEN="${free_gpus[2]}"
            return
        fi
        sleep "${POLL_SECONDS}"
    done
}

run_one_config() {
    local init_workers="$1"
    local run_workers="$2"
    local label="init${init_workers}_run${run_workers}"
    local run_stem="${SWEEP_STEM}_${label}"
    local nsys_out="${PROFILE_DIR}/${run_stem}"

    wait_for_three_gpus
    echo "$(date -Is) launching ${label}: selected=${SELECTED_GPUS_CHOSEN} train=${TRAIN_GPUS_CHOSEN} rollout=${ROLLOUT_GPU_CHOSEN} nsys_out=${nsys_out}" | tee -a "${LOG}"

    unset RAY_ADDRESS || true
    .venv/bin/ray stop --force >>"${LOG}" 2>&1 || true

    local launch_env=(
        PRORL_ENABLE_NVTX=1
        SWEEP_ID="${run_stem}"
        RUN_ID="${run_stem}"
        FIGURE_SUFFIX="nsys_large_workers_${label}"
        WORK_ROOT="${WORK_ROOT}"
        RUN_ROOT="${RUN_ROOT}"
        MODEL_FLAVOR=qwen35_4b
        HF_CHECKPOINT="${HF_CHECKPOINT}"
        SWEGYM_DATA="${SWEGYM_DATA}"
        TORCH_DIST_DIR="${TORCH_DIST_DIR}"
        REF_LOAD="${TORCH_DIST_DIR}"
        AGENT_HARNESS=codex
        PREINSTALLED_AGENT_CLI="${PREINSTALLED_AGENT_CLI}"
        SELECTED_GPUS="${SELECTED_GPUS_CHOSEN}"
        TRAIN_GPUS="${TRAIN_GPUS_CHOSEN}"
        ROLLOUT_GPU="${ROLLOUT_GPU_CHOSEN}"
        MAX_PRELAUNCH_GPU_MEM_MIB="${MAX_USED_MIB}"
        HOST_MIN_AVAILABLE_GIB=80
        RAY_NUM_CPUS="${RAY_NUM_CPUS_FOR_NSYS}"
        NUM_ROLLOUT=4
        ROLLOUT_BATCH_SIZE=8
        N_SAMPLES_PER_PROMPT=4
        NUM_STEPS_PER_ROLLOUT=4
        POLAR_MAX_ASYNC_LEVEL=32
        POLAR_MAX_INIT_WORKERS="${init_workers}"
        POLAR_MAX_RUN_WORKERS="${run_workers}"
        POLAR_MAX_POSTRUN_WORKERS=4
        POLAR_MIN_COMPLETE_ACCEPT_FRACTION=0.5
        TP_SIZE=2
        CP_SIZE=1
        SEQUENCE_PARALLEL=0
        TRANSFORMER_IMPL=local
        SGLANG_CONTEXT_LENGTH=16384
        ROLLOUT_MAX_RESPONSE_LEN=2048
        ROLLOUT_MAX_PROMPT_LEN=14336
        MAX_TOKENS_PER_GPU=16384
        KEEP_RUNTIME_ARTIFACTS="${KEEP_RUNTIME_ARTIFACTS}"
    )

    printf '%s launch_env %s\n' \
        "$(date -Is)" \
        "$(printf '%s ' "${launch_env[@]}")" >>"${LOG}"

    set +e
    env "${launch_env[@]}" "${NSYS_BIN}" profile \
        --trace="${NSYS_TRACE}" \
        --sample="${NSYS_SAMPLE}" \
        --trace-fork-before-exec=true \
        --wait=primary \
        --force-overwrite=true \
        -o "${nsys_out}" \
        ${NSYS_EXTRA_ARGS} \
        bash examples/swegym_slime_grpo_3h200_qwen3_4b/launch_qwen3_4b_3h200.sh >>"${LOG}" 2>&1
    local status=$?
    set -e
    echo "$(date -Is) finished ${label} status=${status} nsys_out=${nsys_out}" | tee -a "${LOG}"
    return "${status}"
}

CONFIGS="${CONFIGS:-8:8 8:16 16:16 16:32}"
for pair in ${CONFIGS}; do
    IFS=: read -r init_workers run_workers <<<"${pair}"
    if ! run_one_config "${init_workers}" "${run_workers}"; then
        echo "$(date -Is) ${pair} failed; continuing to next config" | tee -a "${LOG}"
    fi
done

echo "$(date -Is) large-worker Nsight sweep complete profile_dir=${PROFILE_DIR}" | tee -a "${LOG}"
