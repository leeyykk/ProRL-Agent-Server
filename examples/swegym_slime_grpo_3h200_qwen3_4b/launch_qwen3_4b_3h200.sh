#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

WORK_ROOT="${WORK_ROOT:-/work1/yokyung/prorl_agent_server_env}"
RUN_ROOT="${RUN_ROOT:-/home/yokyung/prorl_agent_server_runs/swegym_slime_grpo_3h200}"
RUN_ID="${RUN_ID:-qwen3_4b_3h200_$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="${RUN_DIR:-${RUN_ROOT}/${RUN_ID}}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
SLIME_DIR="${SLIME_DIR:-${PROJECT_ROOT}/slime}"
MEGATRON_DIR="${MEGATRON_DIR:-${PROJECT_ROOT}/Megatron-LM}"
HF_CHECKPOINT="${HF_CHECKPOINT:-/work1/huggingface_models/Qwen3-4B}"
TORCH_DIST_DIR="${TORCH_DIST_DIR:-${WORK_ROOT}/checkpoints/Qwen3-4B_torch_dist}"
REF_LOAD="${REF_LOAD:-${TORCH_DIST_DIR}}"
SAVE_DIR="${SAVE_DIR:-${RUN_DIR}/checkpoints}"
SWEGYM_DATA="${SWEGYM_DATA:-${WORK_ROOT}/data/swegym_train_docker_3h200.jsonl}"
TOPOLOGY_TEMPLATE="${TOPOLOGY_TEMPLATE:-${SCRIPT_DIR}/topology.docker_3h200.yaml}"
POLAR_CONFIG_TEMPLATE="${POLAR_CONFIG_TEMPLATE:-${SCRIPT_DIR}/polar_config.docker_3h200.yaml}"
TOPOLOGY_PATH="${RUN_DIR}/topology.yaml"
POLAR_CONFIG_PATH="${RUN_DIR}/polar_config.yaml"

export TMPDIR="${RUN_DIR}/runtime/tmp"
export TEMP="${TMPDIR}"
export TMP="${TMPDIR}"
export RAY_TMPDIR="${RUN_DIR}/runtime/ray"
export HF_HOME="${WORK_ROOT}/hf-home"
export HF_HUB_CACHE="${WORK_ROOT}/hf-hub-cache"
export TRANSFORMERS_CACHE="${WORK_ROOT}/transformers-cache"
export PYTHONPYCACHEPREFIX="${RUN_DIR}/runtime/pycache"
export FLASHINFER_WORKSPACE_BASE="${RUN_DIR}/runtime/flashinfer"
export MPLCONFIGDIR="${RUN_DIR}/runtime/mplconfig"
export WANDB_DIR="${RUN_DIR}/wandb"
export POLAR_SESSION_BASE_DIR="${RUN_DIR}/polar_session_dirs"
export POLAR_PRESERVE_SESSION_DIRS="${POLAR_PRESERVE_SESSION_DIRS:-1}"
export POLAR_MAX_COMPLETION_TOKENS="${POLAR_MAX_COMPLETION_TOKENS:-1024}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONPATH="${MEGATRON_DIR}:${SLIME_DIR}:${PROJECT_ROOT}/src"

mkdir -p "${RUN_DIR}/logs" "${TMPDIR}" "${RAY_TMPDIR}" "${HF_HOME}" \
    "${HF_HUB_CACHE}" "${TRANSFORMERS_CACHE}" "${PYTHONPYCACHEPREFIX}" \
    "${FLASHINFER_WORKSPACE_BASE}" "${MPLCONFIGDIR}" "${WANDB_DIR}" \
    "${POLAR_SESSION_BASE_DIR}"

if [ ! -x "${PYTHON_BIN}" ]; then
    echo "ERROR: Python env not found: ${PYTHON_BIN}" >&2
    echo "Run setup_env_3h200.sh first." >&2
    exit 1
fi
if [ ! -d "${HF_CHECKPOINT}" ]; then
    echo "ERROR: HF checkpoint not found: ${HF_CHECKPOINT}" >&2
    exit 1
fi
if [ ! -f "${REF_LOAD}/latest_checkpointed_iteration.txt" ]; then
    echo "ERROR: converted torch_dist checkpoint not found: ${REF_LOAD}" >&2
    echo "Run convert_qwen3_4b.sh first." >&2
    exit 1
fi
if [ ! -f "${SWEGYM_DATA}" ]; then
    echo "ERROR: SWE-Gym JSONL not found: ${SWEGYM_DATA}" >&2
    echo "Run prepare_swegym_docker_data.sh first." >&2
    exit 1
fi

MAX_PRELAUNCH_GPU_MEM_MIB="${MAX_PRELAUNCH_GPU_MEM_MIB:-1024}"
HOST_MIN_AVAILABLE_GIB="${HOST_MIN_AVAILABLE_GIB:-120}"
"${PYTHON_BIN}" - <<PY
import subprocess
out = subprocess.check_output([
    "nvidia-smi",
    "--query-gpu=index,memory.used,memory.total",
    "--format=csv,noheader,nounits",
], text=True)
rows = []
for line in out.strip().splitlines():
    idx, used, total = [int(x.strip()) for x in line.split(",")]
    rows.append((idx, used, total))
if len(rows) < 3:
    raise SystemExit(f"Need at least 3 GPUs, found {len(rows)}")
bad = [(idx, used) for idx, used, _ in rows[:3] if used > ${MAX_PRELAUNCH_GPU_MEM_MIB}]
if bad:
    raise SystemExit(f"Refusing launch; selected GPUs not free enough: {bad}")
print("GPU preflight OK:", rows[:3])
PY

"${PYTHON_BIN}" - <<PY
from pathlib import Path
available_kib = None
for line in Path("/proc/meminfo").read_text().splitlines():
    if line.startswith("MemAvailable:"):
        available_kib = int(line.split()[1])
        break
available_gib = available_kib / (1024 ** 2) if available_kib else 0
if available_gib < ${HOST_MIN_AVAILABLE_GIB}:
    raise SystemExit(
        f"Refusing launch; host MemAvailable={available_gib:.1f} GiB "
        f"< HOST_MIN_AVAILABLE_GIB=${HOST_MIN_AVAILABLE_GIB}"
    )
print(f"Host RAM preflight OK: MemAvailable={available_gib:.1f} GiB")
PY

cp "${TOPOLOGY_TEMPLATE}" "${TOPOLOGY_PATH}"
cp "${POLAR_CONFIG_TEMPLATE}" "${POLAR_CONFIG_PATH}"
"${PYTHON_BIN}" - <<PY
from pathlib import Path
import yaml

topology_path = Path("${TOPOLOGY_PATH}")
topology = yaml.safe_load(topology_path.read_text()) or {}
topology.setdefault("rollout", {})["save_dir"] = "${RUN_DIR}/rollout_results"
for node in topology.get("gateway", {}).get("nodes", []):
    node["model_served"] = "${HF_CHECKPOINT}"
    node.setdefault("sglang", {})["base_url"] = "http://127.0.0.1:19000"
topology_path.write_text(yaml.safe_dump(topology, sort_keys=False))

config_path = Path("${POLAR_CONFIG_PATH}")
config = yaml.safe_load(config_path.read_text()) or {}
task = config.setdefault("polar_task_template", {})
task.setdefault("agent", {})["model_name"] = "${HF_CHECKPOINT}"
config_path.write_text(yaml.safe_dump(config, sort_keys=False))
PY

MODEL_ARGS=(
    --swiglu
    --num-layers 36
    --hidden-size 2560
    --ffn-hidden-size 9728
    --num-attention-heads 32
    --group-query-attention
    --num-query-groups 8
    --use-rotary-position-embeddings
    --disable-bias-linear
    --normalization RMSNorm
    --norm-epsilon 1e-6
    --rotary-base 1000000
    --vocab-size 151936
    --kv-channels 128
    --qk-layernorm
)

PIDS=()
cleanup() {
    touch "${RUN_DIR}/monitor.stop" 2>/dev/null || true
    for pid in "${PIDS[@]}"; do kill "${pid}" 2>/dev/null || true; done
    ray stop --force >/dev/null 2>&1 || true
    if [ "${KEEP_CHECKPOINTS:-0}" != "1" ]; then
        rm -rf "${SAVE_DIR}"
    fi
    wait 2>/dev/null || true
}
trap cleanup EXIT

cat > "${RUN_DIR}/run_settings.txt" <<EOF
RUN_DIR=${RUN_DIR}
WORK_ROOT=${WORK_ROOT}
HF_CHECKPOINT=${HF_CHECKPOINT}
REF_LOAD=${REF_LOAD}
SAVE_DIR=${SAVE_DIR}
SWEGYM_DATA=${SWEGYM_DATA}
TRAIN_GPUS=0,1
ROLLOUT_GPU=2
EOF

cat > "${RUN_DIR}/monitor_host.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
OUT="$1"
STOP="$2"
while [ ! -f "$STOP" ]; do
  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  mem="$(free -g | awk '/Mem:/ {print $3\",\"$2\",\"$7}')"
  gpu="$(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | tr '\n' ';')"
  docker="$(docker stats --no-stream --format '{{.Name}},{{.MemUsage}},{{.CPUPerc}}' 2>/dev/null | tr '\n' ';' || true)"
  printf '%s host_mem_used_total_avail_gib=%s gpu="%s" docker="%s"\n' "$ts" "$mem" "$gpu" "$docker" >> "$OUT"
  sleep 5
done
EOF
chmod +x "${RUN_DIR}/monitor_host.sh"
rm -f "${RUN_DIR}/monitor.stop"
"${RUN_DIR}/monitor_host.sh" "${RUN_DIR}/monitor.log" "${RUN_DIR}/monitor.stop" &
PIDS+=("$!")

"${PYTHON_BIN}" -m polar.cli serve_rollout -c "${TOPOLOGY_PATH}" > "${RUN_DIR}/logs/polar_rollout.log" 2>&1 &
PIDS+=("$!")
sleep 2
"${PYTHON_BIN}" -m polar.cli serve_gateway -c "${TOPOLOGY_PATH}" --node-id localhost-node-01 > "${RUN_DIR}/logs/polar_gateway.log" 2>&1 &
PIDS+=("$!")

"${PYTHON_BIN}" - <<'PY'
import time, urllib.request
for url in ("http://127.0.0.1:48080/health", "http://127.0.0.1:48100/health"):
    last = None
    for _ in range(120):
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    break
        except Exception as exc:
            last = exc
            time.sleep(1)
    else:
        raise SystemExit(f"service not healthy: {url}: {last}")
PY

ray stop --force >/dev/null 2>&1 || true
ray start --head --node-ip-address 127.0.0.1 --num-gpus 3 --disable-usage-stats --include-dashboard=false

NUM_ROLLOUT="${NUM_ROLLOUT:-4}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-1}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-1}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-32768}"
SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-32768}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.70}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000000}"
OPTIMIZER_OFFLOAD_ARGS=()
if [ "${ENABLE_OPTIMIZER_CPU_OFFLOAD:-0}" = "1" ]; then
    OPTIMIZER_OFFLOAD_ARGS=(
        --optimizer-cpu-offload
        --use-precision-aware-optimizer
        --use-distributed-optimizer
    )
fi

LOAD_DIR="${REF_LOAD}"
if [ -f "${SAVE_DIR}/latest_checkpointed_iteration.txt" ]; then
    LOAD_DIR="${SAVE_DIR}"
fi
rm -rf "${SAVE_DIR}"

set +e
RAY_ADDRESS=auto "${PYTHON_BIN}" "${SLIME_DIR}/train_async.py" \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node 2 \
    --rollout-num-gpus 1 \
    --rollout-num-gpus-per-engine 1 \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "${HF_CHECKPOINT}" \
    --ref-load "${REF_LOAD}" \
    --load "${LOAD_DIR}" \
    --save "${SAVE_DIR}" \
    --save-interval "${SAVE_INTERVAL}" \
    --update-weights-interval 1 \
    --rollout-function-path slime_bridge.rollout.generate_rollout_polar_async \
    --custom-rm-path slime_bridge.reward.reward_func \
    --custom-reward-post-process-path slime_bridge.reward_post_process.post_process_rewards \
    --custom-config-path "${POLAR_CONFIG_PATH}" \
    --data-source-path slime_bridge.data_source.CeilEpochRolloutDataSourceWithBuffer \
    --prompt-data "${SWEGYM_DATA}" \
    --input-key prompt \
    --label-key label \
    --metadata-key metadata \
    --reward-key score \
    --num-epoch 1 \
    --num-rollout "${NUM_ROLLOUT}" \
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE}" \
    --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}" \
    --rollout-max-response-len 1024 \
    --rollout-max-prompt-len 31744 \
    --dynamic-history \
    --num-steps-per-rollout 1 \
    --tensor-model-parallel-size 2 \
    --sequence-parallel \
    --pipeline-model-parallel-size 1 \
    --context-parallel-size 1 \
    --expert-model-parallel-size 1 \
    --expert-tensor-parallel-size 1 \
    --recompute-granularity full \
    --recompute-method uniform \
    --recompute-num-layers 1 \
    --use-dynamic-batch-size \
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}" \
    --log-probs-chunk-size 128 \
    --advantage-estimator grpo \
    --normalize-advantages \
    --use-tis \
    --use-kl-loss \
    --kl-loss-coef 0.001 \
    --kl-loss-type low_var_kl \
    --entropy-coef 0.0 \
    --eps-clip 0.2 \
    --eps-clip-high 0.28 \
    "${OPTIMIZER_OFFLOAD_ARGS[@]}" \
    --optimizer adam \
    --lr 1e-6 \
    --lr-decay-style constant \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.98 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --accumulate-allreduce-grads-in-fp32 \
    --attention-softmax-in-fp32 \
    --attention-backend auto \
    --no-gradient-accumulation-fusion \
    --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}" \
    --sglang-context-length "${SGLANG_CONTEXT_LENGTH}" \
    --sglang-tool-call-parser qwen3_coder \
    --router-policy round_robin \
    --sglang-router-port 19000 \
    2>&1 | tee "${RUN_DIR}/train.log"
status="${PIPESTATUS[0]}"
set -e

echo "Run finished with status ${status}"
echo "Run dir: ${RUN_DIR}"
exit "${status}"
