#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

WORK_ROOT="${WORK_ROOT:-/work1/yokyung/prorl_agent_server_env}"
RUN_ROOT="${RUN_ROOT:-/home/yokyung/prorl_agent_server_runs/swegym_slime_grpo_3h200}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
RAY_BIN="${RAY_BIN:-${PROJECT_ROOT}/.venv/bin/ray}"
SLIME_DIR="${SLIME_DIR:-${PROJECT_ROOT}/slime}"
MEGATRON_DIR="${MEGATRON_DIR:-${PROJECT_ROOT}/Megatron-LM}"
HF_CHECKPOINT="${HF_CHECKPOINT:-/work1/huggingface_models/Qwen3-4B}"
MODEL_FLAVOR="${MODEL_FLAVOR:-}"
if [ -z "${MODEL_FLAVOR}" ]; then
    case "${HF_CHECKPOINT}" in
        *Qwen3.5-4B*|*Qwen/Qwen3.5-4B*) MODEL_FLAVOR="qwen35_4b" ;;
        *) MODEL_FLAVOR="qwen3_4b" ;;
    esac
fi
if [ "${MODEL_FLAVOR}" = "qwen35_4b" ]; then
    DEFAULT_TORCH_DIST_DIR="${WORK_ROOT}/checkpoints/Qwen3.5-4B_torch_dist"
    DEFAULT_RUN_PREFIX="qwen35_4b_3h200"
else
    DEFAULT_TORCH_DIST_DIR="${WORK_ROOT}/checkpoints/Qwen3-4B_torch_dist"
    DEFAULT_RUN_PREFIX="qwen3_4b_3h200"
fi
RUN_ID="${RUN_ID:-${DEFAULT_RUN_PREFIX}_$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="${RUN_DIR:-${RUN_ROOT}/${RUN_ID}}"
RUN_STEM="${RUN_STEM:-$(printf '%s' "${RUN_ID}" | sha1sum | awk '{print $1}' | cut -c1-12)}"
TORCH_DIST_DIR="${TORCH_DIST_DIR:-${DEFAULT_TORCH_DIST_DIR}}"
REF_LOAD="${REF_LOAD:-${TORCH_DIST_DIR}}"
SAVE_DIR="${SAVE_DIR:-${RUN_DIR}/checkpoints}"
SWEGYM_DATA="${SWEGYM_DATA:-${WORK_ROOT}/data/swegym_train_docker_3h200.jsonl}"
TOPOLOGY_TEMPLATE="${TOPOLOGY_TEMPLATE:-${SCRIPT_DIR}/topology.docker_3h200.yaml}"
POLAR_CONFIG_TEMPLATE="${POLAR_CONFIG_TEMPLATE:-${SCRIPT_DIR}/polar_config.docker_3h200.yaml}"
RUNTIME_BACKEND="${RUNTIME_BACKEND:-}"
TOPOLOGY_PATH="${RUN_DIR}/topology.yaml"
POLAR_CONFIG_PATH="${RUN_DIR}/polar_config.yaml"
SGLANG_ROUTER_HOST="${SGLANG_ROUTER_HOST:-$(hostname -I | awk '{print $1}')}"
SELECTED_GPUS="${SELECTED_GPUS:-0,1,2}"
TRAIN_GPUS="${TRAIN_GPUS:-$(echo "${SELECTED_GPUS}" | cut -d, -f1,2)}"
ROLLOUT_GPU="${ROLLOUT_GPU:-$(echo "${SELECTED_GPUS}" | cut -d, -f3)}"
DEBUG_ROLLOUT_ONLY="${DEBUG_ROLLOUT_ONLY:-0}"
LOAD_DEBUG_ROLLOUT_DATA="${LOAD_DEBUG_ROLLOUT_DATA:-}"
if [ "${DEBUG_ROLLOUT_ONLY}" = "1" ]; then
    TRAIN_GPUS="${TRAIN_GPUS:-}"
    ROLLOUT_GPU="${ROLLOUT_GPU:-$(echo "${SELECTED_GPUS}" | cut -d, -f1)}"
fi

export CUDA_VISIBLE_DEVICES="${SELECTED_GPUS}"
export TMPDIR="${TMPDIR:-/tmp/prorl_tmp_${RUN_STEM}}"
export TEMP="${TMPDIR}"
export TMP="${TMPDIR}"
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/prorl_ray_${RUN_STEM}}"
export HF_HOME="${WORK_ROOT}/hf-home"
export HF_HUB_CACHE="${WORK_ROOT}/hf-hub-cache"
export TRANSFORMERS_CACHE="${WORK_ROOT}/transformers-cache"
export PYTHONPYCACHEPREFIX="${RUN_DIR}/runtime/pycache"
export FLASHINFER_WORKSPACE_BASE="${RUN_DIR}/runtime/flashinfer"
export MPLCONFIGDIR="${RUN_DIR}/runtime/mplconfig"
export WANDB_DIR="${RUN_DIR}/wandb"
export POLAR_SESSION_BASE_DIR="${POLAR_SESSION_BASE_DIR:-${RUN_DIR}/runtime/polar_session_dirs}"
export POLAR_PRESERVE_SESSION_DIRS="${POLAR_PRESERVE_SESSION_DIRS:-0}"
export POLAR_MAX_COMPLETION_TOKENS="${POLAR_MAX_COMPLETION_TOKENS:-1024}"
export SGLANG_ENABLE_JIT_DEEPGEMM="${SGLANG_ENABLE_JIT_DEEPGEMM:-0}"
export SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM="${SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM:-0}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
NVIDIA_SITE_PACKAGES="${PROJECT_ROOT}/.venv/lib/python3.12/site-packages/nvidia"
if [ -d "${NVIDIA_SITE_PACKAGES}/cu13/lib" ]; then
    export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
    export LD_LIBRARY_PATH="${NVIDIA_SITE_PACKAGES}/cu13/lib:${NVIDIA_SITE_PACKAGES}/cudnn/lib:${NVIDIA_SITE_PACKAGES}/nccl/lib:${NVIDIA_SITE_PACKAGES}/nvshmem/lib:${LD_LIBRARY_PATH:-}"
fi
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
    if [ "${MODEL_FLAVOR}" = "qwen35_4b" ]; then
        echo "Run convert_qwen35_4b.sh first." >&2
    else
        echo "Run convert_qwen3_4b.sh first." >&2
    fi
    exit 1
fi
if [ ! -f "${SWEGYM_DATA}" ]; then
    echo "ERROR: SWE-Gym JSONL not found: ${SWEGYM_DATA}" >&2
    echo "Run prepare_swegym_docker_data.sh or prepare_swegym_apptainer_data_b200.sh first." >&2
    exit 1
fi

MAX_PRELAUNCH_GPU_MEM_MIB="${MAX_PRELAUNCH_GPU_MEM_MIB:-1024}"
HOST_MIN_AVAILABLE_GIB="${HOST_MIN_AVAILABLE_GIB:-120}"
"${PYTHON_BIN}" - <<PY
import subprocess
selected = [int(x) for x in "${SELECTED_GPUS}".split(",") if x.strip()]
debug_rollout_only = "${DEBUG_ROLLOUT_ONLY}" == "1"
debug_train_only = bool("${LOAD_DEBUG_ROLLOUT_DATA}")
out = subprocess.check_output([
    "nvidia-smi",
    "--query-gpu=index,memory.used,memory.total",
    "--format=csv,noheader,nounits",
], text=True)
rows = []
for line in out.strip().splitlines():
    idx, used, total = [int(x.strip()) for x in line.split(",")]
    rows.append((idx, used, total))
by_idx = {idx: (idx, used, total) for idx, used, total in rows}
expected = 1 if debug_rollout_only else (2 if debug_train_only else 3)
if len(selected) != expected:
    raise SystemExit(f"Need exactly {expected} selected GPUs, got {selected}")
missing = [idx for idx in selected if idx not in by_idx]
if missing:
    raise SystemExit(f"Selected GPUs not found: {missing}")
selected_rows = [by_idx[idx] for idx in selected]
bad = [(idx, used) for idx, used, _ in selected_rows if used > ${MAX_PRELAUNCH_GPU_MEM_MIB}]
if bad:
    raise SystemExit(f"Refusing launch; selected GPUs not free enough: {bad}")
print("GPU preflight OK:", selected_rows)
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
POLAR_MAX_INIT_WORKERS="${POLAR_MAX_INIT_WORKERS:-1}"
POLAR_MAX_RUN_WORKERS="${POLAR_MAX_RUN_WORKERS:-1}"
POLAR_MAX_POSTRUN_WORKERS="${POLAR_MAX_POSTRUN_WORKERS:-1}"
POLAR_MAX_ASYNC_LEVEL="${POLAR_MAX_ASYNC_LEVEL:-1}"
POLAR_MIN_COMPLETE_ACCEPT_FRACTION="${POLAR_MIN_COMPLETE_ACCEPT_FRACTION:-}"
AGENT_HARNESS="${AGENT_HARNESS:-qwen_code}"
case "${AGENT_HARNESS}" in
    qwen_code) AGENT_NPM_PACKAGE="${AGENT_NPM_PACKAGE:-@qwen-code/qwen-code@0.14.5}" ;;
    codex) AGENT_NPM_PACKAGE="${AGENT_NPM_PACKAGE:-@openai/codex@0.121.0}" ;;
    *) echo "ERROR: unsupported AGENT_HARNESS=${AGENT_HARNESS}" >&2; exit 1 ;;
esac
PREINSTALLED_AGENT_CLI="${PREINSTALLED_AGENT_CLI:-}"
"${PYTHON_BIN}" - <<PY
from pathlib import Path
import yaml

topology_path = Path("${TOPOLOGY_PATH}")
topology = yaml.safe_load(topology_path.read_text()) or {}
topology.setdefault("rollout", {})["save_dir"] = "${RUN_DIR}/rollout_results"
for node in topology.get("gateway", {}).get("nodes", []):
    node["model_served"] = "${HF_CHECKPOINT}"
    node["max_init_workers"] = int("${POLAR_MAX_INIT_WORKERS}")
    node["max_run_workers"] = int("${POLAR_MAX_RUN_WORKERS}")
    node["max_postrun_workers"] = int("${POLAR_MAX_POSTRUN_WORKERS}")
    node.setdefault("sglang", {})["base_url"] = "http://${SGLANG_ROUTER_HOST}:19000"
topology_path.write_text(yaml.safe_dump(topology, sort_keys=False))

config_path = Path("${POLAR_CONFIG_PATH}")
config = yaml.safe_load(config_path.read_text()) or {}
config["polar_max_async_level"] = int("${POLAR_MAX_ASYNC_LEVEL}")
if "${POLAR_MIN_COMPLETE_ACCEPT_FRACTION}":
    config["polar_min_complete_accept_fraction"] = float("${POLAR_MIN_COMPLETE_ACCEPT_FRACTION}")
task = config.setdefault("polar_task_template", {})
runtime = task.setdefault("runtime", {})
if "${RUNTIME_BACKEND}":
    runtime["backend"] = "${RUNTIME_BACKEND}"
runtime_env = runtime.setdefault("env", {})
prepare_steps = runtime.get("prepare") or []
for step in prepare_steps:
    if isinstance(step, dict) and "command" in step:
        command = str(step["command"])
        if "${PREINSTALLED_AGENT_CLI}" and "${AGENT_HARNESS}" == "codex":
            command = command.replace(
                "npm install -g --no-audit --no-fund @qwen-code/qwen-code@0.14.5",
                "command -v codex && codex --version",
            )
        else:
            command = command.replace(
                "npm install -g --no-audit --no-fund @qwen-code/qwen-code@0.14.5",
                "npm install -g --no-audit --no-fund ${AGENT_NPM_PACKAGE}",
            )
        step["command"] = command
if "${PREINSTALLED_AGENT_CLI}" and "${AGENT_HARNESS}" == "codex":
    runtime_env["PATH"] = "/agent-cli/bin:" + str(runtime_env.get("PATH", ""))
    kwargs = runtime.setdefault("kwargs", {})
    volumes = kwargs.setdefault("volumes", [])
    volume = "${PREINSTALLED_AGENT_CLI}:/agent-cli:ro"
    if volume not in volumes:
        volumes.append(volume)
agent = task.setdefault("agent", {})
agent["harness"] = "${AGENT_HARNESS}"
agent["model_name"] = "${HF_CHECKPOINT}"
exclude_patterns = task.setdefault("evaluator", {}).setdefault("config", {}).setdefault("exclude_patterns", [])
for pattern in [".codex/**", "**/.codex/**", ".qwen/**", "**/.qwen/**"]:
    if pattern not in exclude_patterns:
        exclude_patterns.append(pattern)
config_path.write_text(yaml.safe_dump(config, sort_keys=False))
PY

if [ "${MODEL_FLAVOR}" = "qwen35_4b" ]; then
    MODEL_ARGS=(
        --spec "slime_plugins.models.qwen3_5" "get_qwen3_5_spec"
        --disable-bias-linear
        --qk-layernorm
        --group-query-attention
        --num-attention-heads 16
        --num-query-groups 4
        --kv-channels 256
        --num-layers 32
        --hidden-size 2560
        --ffn-hidden-size 9216
        --use-gated-attention
        --normalization RMSNorm
        --position-embedding-type rope
        --norm-epsilon 1e-6
        --rotary-percent 0.25
        --swiglu
        --vocab-size 248320
        --rotary-base 10000000
        --attention-output-gate
    )
    MODEL_COMPAT_ARGS=(
        --loss-mask-type qwen3_5
    )
else
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
    MODEL_COMPAT_ARGS=(
        --loss-mask-type qwen3
    )
fi

PIDS=()
cleanup() {
    touch "${RUN_DIR}/monitor.stop" 2>/dev/null || true
    for pid in "${PIDS[@]}"; do kill "${pid}" 2>/dev/null || true; done
    "${RAY_BIN}" stop --force >/dev/null 2>&1 || true
    if [ "${KEEP_CHECKPOINTS:-0}" != "1" ]; then
        rm -rf "${SAVE_DIR}"
    fi
    if [ "${KEEP_RUNTIME_ARTIFACTS:-0}" != "1" ]; then
        rm -rf "${TMPDIR}" "${RAY_TMPDIR}" "${POLAR_SESSION_BASE_DIR}" \
            "${RUN_DIR}/runtime/pycache" "${RUN_DIR}/runtime/flashinfer" \
            "${RUN_DIR}/runtime/mplconfig" 2>/dev/null || true
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
SELECTED_GPUS=${SELECTED_GPUS}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
TRAIN_GPUS=${TRAIN_GPUS}
ROLLOUT_GPU=${ROLLOUT_GPU}
NUM_ROLLOUT=${NUM_ROLLOUT:-4}
START_ROLLOUT_ID=${START_ROLLOUT_ID:-}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-1}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-1}
NUM_STEPS_PER_ROLLOUT=${NUM_STEPS_PER_ROLLOUT:-1}
MODEL_FLAVOR=${MODEL_FLAVOR}
POLAR_MAX_INIT_WORKERS=${POLAR_MAX_INIT_WORKERS}
POLAR_MAX_RUN_WORKERS=${POLAR_MAX_RUN_WORKERS}
POLAR_MAX_POSTRUN_WORKERS=${POLAR_MAX_POSTRUN_WORKERS}
POLAR_MAX_ASYNC_LEVEL=${POLAR_MAX_ASYNC_LEVEL}
POLAR_MIN_COMPLETE_ACCEPT_FRACTION=${POLAR_MIN_COMPLETE_ACCEPT_FRACTION}
TP_SIZE=${TP_SIZE:-1}
CP_SIZE=${CP_SIZE:-1}
CP_COMM_TYPE=${CP_COMM_TYPE:-p2p}
PP_SIZE=${PP_SIZE:-1}
QKV_FORMAT=${QKV_FORMAT:-bshd}
TRANSFORMER_IMPL=${TRANSFORMER_IMPL:-local}
ATTENTION_BACKEND=${ATTENTION_BACKEND:-auto}
SEQUENCE_PARALLEL=${SEQUENCE_PARALLEL:-0}
SGLANG_CONTEXT_LENGTH=${SGLANG_CONTEXT_LENGTH:-16384}
ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-1024}
ROLLOUT_MAX_PROMPT_LEN=${ROLLOUT_MAX_PROMPT_LEN:-15360}
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-16384}
USE_DYNAMIC_BATCH_SIZE=${USE_DYNAMIC_BATCH_SIZE:-0}
RAY_NUM_CPUS=${RAY_NUM_CPUS:-}
LOAD_DEBUG_ROLLOUT_DATA=${LOAD_DEBUG_ROLLOUT_DATA}
AGENT_HARNESS=${AGENT_HARNESS}
AGENT_NPM_PACKAGE=${AGENT_NPM_PACKAGE}
EOF

cat > "${RUN_DIR}/monitor_host.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
OUT="$1"
STOP="$2"
while [ ! -f "$STOP" ]; do
  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  mem="$(free -g | awk '/Mem:/ {print $3","$2","$7}')"
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

"${RAY_BIN}" stop --force >/dev/null 2>&1 || true
RAY_START_ARGS=(
    start
    --head
    --node-ip-address 127.0.0.1
    --num-gpus "$(echo "${SELECTED_GPUS}" | awk -F, '{print NF}')"
    --disable-usage-stats
    --include-dashboard=false
)
if [ -n "${RAY_NUM_CPUS:-}" ]; then
    RAY_START_ARGS+=(--num-cpus "${RAY_NUM_CPUS}")
fi
"${RAY_BIN}" "${RAY_START_ARGS[@]}"

NUM_ROLLOUT="${NUM_ROLLOUT:-4}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-1}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-1}"
NUM_STEPS_PER_ROLLOUT="${NUM_STEPS_PER_ROLLOUT:-1}"
SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-16384}"
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-1024}"
ROLLOUT_MAX_PROMPT_LEN="${ROLLOUT_MAX_PROMPT_LEN:-$((SGLANG_CONTEXT_LENGTH - ROLLOUT_MAX_RESPONSE_LEN))}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-${SGLANG_CONTEXT_LENGTH}}"
USE_DYNAMIC_BATCH_SIZE="${USE_DYNAMIC_BATCH_SIZE:-0}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.70}"
SGLANG_CUDA_GRAPH_MAX_BS="${SGLANG_CUDA_GRAPH_MAX_BS:-16}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000000}"
START_ROLLOUT_ARGS=()
if [ -n "${START_ROLLOUT_ID:-}" ]; then
    START_ROLLOUT_ARGS=(--start-rollout-id "${START_ROLLOUT_ID}")
fi
DEBUG_ROLLOUT_ARGS=()
if [ "${DEBUG_ROLLOUT_ONLY}" = "1" ]; then
    DEBUG_ROLLOUT_ARGS=(
        --debug-rollout-only
        --save-debug-rollout-data "${RUN_DIR}/debug_rollout_{rollout_id}.pt"
    )
fi
if [ -n "${LOAD_DEBUG_ROLLOUT_DATA}" ]; then
    DEBUG_ROLLOUT_ARGS+=(--load-debug-rollout-data "${LOAD_DEBUG_ROLLOUT_DATA}")
fi
OPTIMIZER_OFFLOAD_ARGS=()
if [ "${ENABLE_OPTIMIZER_CPU_OFFLOAD:-0}" = "1" ]; then
    OPTIMIZER_OFFLOAD_ARGS=(
        --optimizer-cpu-offload
        --use-precision-aware-optimizer
        --use-distributed-optimizer
    )
fi
TP_SIZE="${TP_SIZE:-1}"
CP_SIZE="${CP_SIZE:-1}"
CP_COMM_TYPE="${CP_COMM_TYPE:-p2p}"
PP_SIZE="${PP_SIZE:-1}"
QKV_FORMAT="${QKV_FORMAT:-bshd}"
TRANSFORMER_IMPL="${TRANSFORMER_IMPL:-local}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-auto}"
USE_TIS="${USE_TIS:-1}"
SEQUENCE_PARALLEL_ARGS=()
if [ "${SEQUENCE_PARALLEL:-0}" = "1" ]; then
    SEQUENCE_PARALLEL_ARGS=(--sequence-parallel)
fi
DYNAMIC_BATCH_ARGS=()
if [ "${USE_DYNAMIC_BATCH_SIZE}" = "1" ]; then
    DYNAMIC_BATCH_ARGS=(--use-dynamic-batch-size)
fi
CP_COMM_ARGS=(--cp-comm-type "${CP_COMM_TYPE}")
TIS_ARGS=()
if [ "${USE_TIS}" = "1" ]; then
    TIS_ARGS=(--use-tis)
fi
TRAIN_ENV_VARS_JSON="$("${PYTHON_BIN}" - <<'PY'
import json
import os

keys = [
    "CUDA_HOME",
    "LD_LIBRARY_PATH",
    "PATH",
]
print(json.dumps({key: os.environ[key] for key in keys if os.environ.get(key)}))
PY
)"

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
    "${DEBUG_ROLLOUT_ARGS[@]}" \
    "${MODEL_ARGS[@]}" \
    "${MODEL_COMPAT_ARGS[@]}" \
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
    --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}" \
    --rollout-max-prompt-len "${ROLLOUT_MAX_PROMPT_LEN}" \
    --dynamic-history \
    --num-steps-per-rollout "${NUM_STEPS_PER_ROLLOUT}" \
    "${START_ROLLOUT_ARGS[@]}" \
    --tensor-model-parallel-size "${TP_SIZE}" \
    --pipeline-model-parallel-size "${PP_SIZE}" \
    --context-parallel-size "${CP_SIZE}" \
    "${CP_COMM_ARGS[@]}" \
    "${SEQUENCE_PARALLEL_ARGS[@]}" \
    --train-env-vars "${TRAIN_ENV_VARS_JSON}" \
    --expert-model-parallel-size 1 \
    --expert-tensor-parallel-size 1 \
    --qkv-format "${QKV_FORMAT}" \
    --transformer-impl "${TRANSFORMER_IMPL}" \
    --no-rope-fusion \
    --no-persist-layer-norm \
    --recompute-granularity full \
    --recompute-method uniform \
    --recompute-num-layers 1 \
    "${DYNAMIC_BATCH_ARGS[@]}" \
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}" \
    --log-probs-chunk-size 128 \
    --advantage-estimator grpo \
    --normalize-advantages \
    "${TIS_ARGS[@]}" \
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
    --attention-backend "${ATTENTION_BACKEND}" \
    --no-gradient-accumulation-fusion \
    --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}" \
    --sglang-context-length "${SGLANG_CONTEXT_LENGTH}" \
    --sglang-disable-cuda-graph \
    --sglang-disable-piecewise-cuda-graph \
    --sglang-disable-overlap-schedule \
    --sglang-cuda-graph-max-bs "${SGLANG_CUDA_GRAPH_MAX_BS}" \
    --sglang-rl-on-policy-target fsdp \
    --sglang-tool-call-parser qwen3_coder \
    --router-policy round_robin \
    --sglang-router-port 19000 \
    2>&1 | tee "${RUN_DIR}/train.log"
status="${PIPESTATUS[0]}"
set -e

echo "Run finished with status ${status}"
echo "Run dir: ${RUN_DIR}"
exit "${status}"
