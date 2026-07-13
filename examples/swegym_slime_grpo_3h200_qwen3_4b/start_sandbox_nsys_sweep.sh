#!/usr/bin/env bash
set -euo pipefail

SWEEP_STEM="${SWEEP_STEM:-qwen35_large_worker_nsys_sandbox_nogpumetrics_$(date -u +%Y%m%dT%H%M%SZ)}"
PROFILE_DIR="${PROFILE_DIR:-/NHNHOME/home/profiled_runs/${SWEEP_STEM}}"

mkdir -p "${PROFILE_DIR}"

export SWEGYM_DATA="${SWEGYM_DATA:-/NHNHOME/home/prorl_agent_server_env/data/swegym_train_apptainer_3h200_sandbox.jsonl}"
export POLAR_APPTAINER_BIN="${POLAR_APPTAINER_BIN:-/usr/bin/singularity}"
export POLAR_APPTAINER_EXEC_ARGS="${POLAR_APPTAINER_EXEC_ARGS:---userns}"
export NSYS_GPU_METRICS_DEVICES="${NSYS_GPU_METRICS_DEVICES:-none}"
export PROFILE_DIR
export SWEEP_STEM

nohup bash examples/swegym_slime_grpo_3h200_qwen3_4b/run_qwen35_codex_nsys_large_worker_sweep_when_free.sh \
    >"${PROFILE_DIR}/launcher.nohup.log" 2>&1 &

echo "$!" >"${PROFILE_DIR}/sweep.pid"
echo "PROFILE_DIR=${PROFILE_DIR}"
echo "PID=$(cat "${PROFILE_DIR}/sweep.pid")"
echo "LOG=${PROFILE_DIR}/sweep.log"
echo "NOHUP_LOG=${PROFILE_DIR}/launcher.nohup.log"
