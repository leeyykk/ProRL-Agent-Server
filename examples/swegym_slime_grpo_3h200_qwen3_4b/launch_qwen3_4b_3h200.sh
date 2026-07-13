#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

WORK_ROOT="${WORK_ROOT:-/NHNHOME/home/prorl_agent_server_env}"
RUN_ROOT="${RUN_ROOT:-/NHNHOME/home/prorl_agent_server_runs/swegym_slime_grpo_3h200}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
RAY_BIN="${RAY_BIN:-${PROJECT_ROOT}/.venv/bin/ray}"
SLIME_DIR="${SLIME_DIR:-${PROJECT_ROOT}/slime}"
MEGATRON_DIR="${MEGATRON_DIR:-${PROJECT_ROOT}/Megatron-LM}"
HF_CHECKPOINT="${HF_CHECKPOINT:-/NHNHOME/home/huggingface_models/Qwen3.5-4B}"
MODEL_FLAVOR="${MODEL_FLAVOR:-}"
if [ -z "${MODEL_FLAVOR}" ]; then
    case "${HF_CHECKPOINT}" in
        *Qwen3.5-4B*|*Qwen/Qwen3.5-4B*) MODEL_FLAVOR="qwen35_4b" ;;
        *) MODEL_FLAVOR="qwen3_4b" ;;
    esac
fi
if [ "${MODEL_FLAVOR}" = "qwen35_4b" ]; then
    DEFAULT_TORCH_DIST_DIR="${WORK_ROOT}/checkpoints/Qwen3.5-4B_torch_dist_tp2"
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
SWEGYM_DATA="${SWEGYM_DATA:-${WORK_ROOT}/data/swegym_train_apptainer_3h200.jsonl}"
TOPOLOGY_TEMPLATE="${TOPOLOGY_TEMPLATE:-${SCRIPT_DIR}/topology.docker_3h200.yaml}"
POLAR_CONFIG_TEMPLATE="${POLAR_CONFIG_TEMPLATE:-${SCRIPT_DIR}/polar_config.docker_3h200.yaml}"
RUNTIME_BACKEND="${RUNTIME_BACKEND:-apptainer}"
TOPOLOGY_PATH="${RUN_DIR}/topology.yaml"
POLAR_CONFIG_PATH="${RUN_DIR}/polar_config.yaml"
SGLANG_ROUTER_HOST="${SGLANG_ROUTER_HOST:-$(hostname -I | awk '{print $NF}')}"
SELECTED_GPUS="${SELECTED_GPUS:-0,1,2}"
TRAIN_GPUS="${TRAIN_GPUS:-$(echo "${SELECTED_GPUS}" | cut -d, -f1,2)}"
ROLLOUT_GPU="${ROLLOUT_GPU:-$(echo "${SELECTED_GPUS}" | cut -d, -f3)}"
GPU_MONITOR_INTERVAL_SECONDS="${GPU_MONITOR_INTERVAL_SECONDS:-1}"
GPU_IDLE_UTIL_THRESHOLD="${GPU_IDLE_UTIL_THRESHOLD:-10}"
DEBUG_ROLLOUT_ONLY="${DEBUG_ROLLOUT_ONLY:-0}"
LOAD_DEBUG_ROLLOUT_DATA="${LOAD_DEBUG_ROLLOUT_DATA:-}"
if [ "${DEBUG_ROLLOUT_ONLY}" = "1" ]; then
    TRAIN_GPUS="${TRAIN_GPUS:-}"
    ROLLOUT_GPU="${ROLLOUT_GPU:-$(echo "${SELECTED_GPUS}" | cut -d, -f1)}"
fi

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${SELECTED_GPUS}"
SHORT_TMP_ROOT="${SHORT_TMP_ROOT:-${WORK_ROOT%/*}}"
export TMPDIR="${TMPDIR:-${SHORT_TMP_ROOT}/tmp_${RUN_STEM}}"
export TEMP="${TMPDIR}"
export TMP="${TMPDIR}"
export RAY_TMPDIR="${RAY_TMPDIR:-${SHORT_TMP_ROOT}/ray_${RUN_STEM}}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${TMPDIR}/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${TMPDIR}/triton}"
RAY_PORT="${RAY_PORT:-}"
export HF_HOME="${WORK_ROOT}/hf-home"
export HF_HUB_CACHE="${WORK_ROOT}/hf-hub-cache"
export TRANSFORMERS_CACHE="${WORK_ROOT}/transformers-cache"
export PYTHONPYCACHEPREFIX="${RUN_DIR}/runtime/pycache"
export FLASHINFER_WORKSPACE_BASE="${RUN_DIR}/runtime/flashinfer"
export MPLCONFIGDIR="${RUN_DIR}/runtime/mplconfig"
export WANDB_DIR="${RUN_DIR}/wandb"
export POLAR_SESSION_BASE_DIR="${POLAR_SESSION_BASE_DIR:-${RUN_DIR}/runtime/polar_session_dirs}"
export POLAR_PRESERVE_SESSION_DIRS="${POLAR_PRESERVE_SESSION_DIRS:-0}"
export PRORL_KILL_STALE_RAY_PROCESSES="${PRORL_KILL_STALE_RAY_PROCESSES:-0}"
export PRORL_GLOBAL_RAY_STOP="${PRORL_GLOBAL_RAY_STOP:-0}"
export POLAR_MAX_COMPLETION_TOKENS="${POLAR_MAX_COMPLETION_TOKENS:-1024}"
export SLIME_RESPECT_CUDA_VISIBLE_DEVICES_ORDER="${SLIME_RESPECT_CUDA_VISIBLE_DEVICES_ORDER:-1}"
export SLIME_ROLLOUT_USE_RAY_CUDA_VISIBLE_DEVICES="${SLIME_ROLLOUT_USE_RAY_CUDA_VISIBLE_DEVICES:-1}"
export SGLANG_ENABLE_JIT_DEEPGEMM="${SGLANG_ENABLE_JIT_DEEPGEMM:-0}"
export SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM="${SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM:-0}"
export SGLANG_NUMA_BIND_V2="${SGLANG_NUMA_BIND_V2:-0}"
export PRORL_DISABLE_NUMACTL_BIND="${PRORL_DISABLE_NUMACTL_BIND:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
NVIDIA_SITE_PACKAGES="${PROJECT_ROOT}/.venv/lib/python3.12/site-packages/nvidia"
TORCH_LIBRARY_DIR="${PROJECT_ROOT}/.venv/lib/python3.12/site-packages/torch/lib"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
LIBRARY_PATHS=()
if [ -d "${TORCH_LIBRARY_DIR}" ]; then
    LIBRARY_PATHS+=("${TORCH_LIBRARY_DIR}")
fi
if [ -d "${NVIDIA_SITE_PACKAGES}" ]; then
    while IFS= read -r lib_dir; do
        LIBRARY_PATHS+=("${lib_dir}")
    done < <(find "${NVIDIA_SITE_PACKAGES}" -type d -name lib -print 2>/dev/null | sort)
fi
if [ "${#LIBRARY_PATHS[@]}" -gt 0 ]; then
    IFS=:
    export LD_LIBRARY_PATH="${LIBRARY_PATHS[*]}:${LD_LIBRARY_PATH:-}"
    unset IFS
fi
export CUDA_DEVICE_MAX_CONNECTIONS=1
PRORL_PROCESS_NAME="${PRORL_PROCESS_NAME:-}"
PROCESS_TITLE_PYTHONPATH=""
if [ -n "${PRORL_PROCESS_NAME}" ]; then
    PROCESS_TITLE_PYTHONPATH="${RUN_DIR}/runtime/process_title"
fi
export PYTHONPATH="${PROCESS_TITLE_PYTHONPATH:+${PROCESS_TITLE_PYTHONPATH}:}${MEGATRON_DIR}:${SLIME_DIR}:${PROJECT_ROOT}/src"
SELECTED_GPU_COUNT="$(echo "${SELECTED_GPUS}" | awk -F, '{print NF}')"

mkdir -p "${RUN_DIR}/logs" "${TMPDIR}" "${RAY_TMPDIR}" "${HF_HOME}" \
    "${HF_HUB_CACHE}" "${TRANSFORMERS_CACHE}" "${PYTHONPYCACHEPREFIX}" \
    "${FLASHINFER_WORKSPACE_BASE}" "${MPLCONFIGDIR}" "${WANDB_DIR}" \
    "${POLAR_SESSION_BASE_DIR}"
if [ "${PRORL_DISABLE_NUMACTL_BIND}" = "1" ]; then
    mkdir -p "${RUN_DIR}/runtime/bin"
    cat > "${RUN_DIR}/runtime/bin/numactl" <<'EOF'
#!/usr/bin/env bash
args=()
skip_next=0
for arg in "$@"; do
  if [ "${skip_next}" = "1" ]; then
    skip_next=0
    continue
  fi
  case "${arg}" in
    --cpunodebind=*|--membind=*|--physcpubind=*|--preferred=*|--interleave=*)
      ;;
    --cpunodebind|--membind|--physcpubind|--preferred|--interleave)
      skip_next=1
      ;;
    *)
      args+=("${arg}")
      ;;
  esac
done
exec "${args[@]}"
EOF
    chmod +x "${RUN_DIR}/runtime/bin/numactl"
    export PATH="${RUN_DIR}/runtime/bin:${PATH}"
fi
if [ -n "${PRORL_PROCESS_NAME}" ]; then
    mkdir -p "${PROCESS_TITLE_PYTHONPATH}"
    cat > "${PROCESS_TITLE_PYTHONPATH}/sitecustomize.py" <<'PY'
import os

name = os.environ.get("PRORL_PROCESS_NAME")
if name:
    try:
        from setproctitle import setproctitle

        setproctitle(name)
    except Exception:
        pass
PY
fi

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
if [ "${RUNTIME_BACKEND}" = "apptainer" ]; then
    if [ -z "${POLAR_APPTAINER_BIN:-}" ]; then
        if command -v apptainer >/dev/null 2>&1; then
            POLAR_APPTAINER_BIN="$(command -v apptainer)"
        elif command -v singularity >/dev/null 2>&1; then
            POLAR_APPTAINER_BIN="$(command -v singularity)"
        else
            POLAR_APPTAINER_BIN="apptainer"
        fi
    fi
    if ! command -v "${POLAR_APPTAINER_BIN}" >/dev/null 2>&1; then
        echo "ERROR: RUNTIME_BACKEND=apptainer but Apptainer/Singularity is not available: ${POLAR_APPTAINER_BIN}" >&2
        echo "Set POLAR_APPTAINER_BIN=/absolute/path/to/apptainer-or-singularity." >&2
        exit 1
    fi
    if [ -z "${POLAR_APPTAINER_EXEC_ARGS:-}" ] && [ "$(basename "${POLAR_APPTAINER_BIN}")" = "singularity" ]; then
        POLAR_APPTAINER_EXEC_ARGS="--userns"
    fi
    export POLAR_APPTAINER_BIN POLAR_APPTAINER_EXEC_ARGS
    APPTAINER_PREFLIGHT_IMAGE="$(${PYTHON_BIN} - "${SWEGYM_DATA}" <<PY
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
with path.open() as f:
    row = json.loads(f.readline())
image = (row.get("metadata") or {}).get("runtime_image")
if image:
    print(image)
PY
)"
    if [ -n "${APPTAINER_PREFLIGHT_IMAGE}" ]; then
        read -r -a APPTAINER_PREFLIGHT_ARGS <<<"${POLAR_APPTAINER_EXEC_ARGS:-}"
        if ! "${POLAR_APPTAINER_BIN}" exec "${APPTAINER_PREFLIGHT_ARGS[@]}" "${APPTAINER_PREFLIGHT_IMAGE}" true >/dev/null 2>"${RUN_DIR}/logs/apptainer_preflight.err"; then
            echo "ERROR: Apptainer is present but cannot execute SWE-Gym SIF: ${APPTAINER_PREFLIGHT_IMAGE}" >&2
            echo "See ${RUN_DIR}/logs/apptainer_preflight.err" >&2
            exit 1
        fi
    fi
fi

MAX_PRELAUNCH_GPU_MEM_MIB="${MAX_PRELAUNCH_GPU_MEM_MIB:-1024}"
HOST_MIN_AVAILABLE_GIB="${HOST_MIN_AVAILABLE_GIB:-120}"
PRORL_STRICT_GPU_ISOLATION="${PRORL_STRICT_GPU_ISOLATION:-0}"
strict_gpu_preflight() {
    local stage="$1"
    "${PYTHON_BIN}" - "$stage" <<PY
import os
import sys
import subprocess
selected = [int(x) for x in "${SELECTED_GPUS}".split(",") if x.strip()]
debug_rollout_only = "${DEBUG_ROLLOUT_ONLY}" == "1"
debug_train_only = bool("${LOAD_DEBUG_ROLLOUT_DATA}")
stage = sys.argv[1]
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
    raise SystemExit(f"Refusing launch at {stage}; selected GPUs not free enough: {bad}")
if "${PRORL_STRICT_GPU_ISOLATION}" == "1":
    uuid_out = subprocess.check_output([
        "nvidia-smi",
        "--query-gpu=index,uuid",
        "--format=csv,noheader",
    ], text=True)
    uuid_to_idx = {}
    for line in uuid_out.strip().splitlines():
        idx, uuid = [x.strip() for x in line.split(",", 1)]
        uuid_to_idx[uuid] = int(idx)
    try:
        apps_out = subprocess.check_output([
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader",
        ], text=True)
    except subprocess.CalledProcessError:
        apps_out = ""
    offenders = []
    for line in apps_out.strip().splitlines():
        parts = [x.strip() for x in line.split(",", 3)]
        if len(parts) != 4:
            continue
        uuid, pid, proc_name, used = parts
        idx = uuid_to_idx.get(uuid)
        if idx not in selected:
            continue
        cwd = ""
        try:
            cwd = os.readlink(f"/proc/{pid}/cwd")
        except OSError:
            pass
        offenders.append((idx, pid, used, proc_name, cwd))
    if offenders:
        lines = [
            f"gpu={idx} pid={pid} mem={used} proc={proc_name} cwd={cwd}"
            for idx, pid, used, proc_name, cwd in offenders
        ]
        raise SystemExit(
            f"Refusing launch at {stage}; strict GPU isolation found existing compute "
            "processes on selected GPUs:\n" + "\n".join(lines)
        )
print(f"GPU preflight OK at {stage}:", selected_rows)
PY
}

strict_gpu_preflight "startup"

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
AGENT_HARNESS="${AGENT_HARNESS:-codex}"
case "${AGENT_HARNESS}" in
    qwen_code) AGENT_NPM_PACKAGE="${AGENT_NPM_PACKAGE:-@qwen-code/qwen-code@0.14.5}" ;;
    codex) AGENT_NPM_PACKAGE="${AGENT_NPM_PACKAGE:-@openai/codex@0.121.0}" ;;
    *) echo "ERROR: unsupported AGENT_HARNESS=${AGENT_HARNESS}" >&2; exit 1 ;;
esac
PREINSTALLED_AGENT_CLI="${PREINSTALLED_AGENT_CLI:-${WORK_ROOT}/agent_cli/codex_0.121.0}"
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
if runtime.get("backend") == "apptainer":
    # The Polar Apptainer runtime does not implement cgroup-style resource
    # limits. Leaving Docker limits in the template makes session init fail
    # before Codex or SGLang can run.
    for key in ("cpus", "memory_mb", "storage_mb"):
        runtime.pop(key, None)
runtime_env = runtime.setdefault("env", {})
prepare_steps = runtime.get("prepare") or []
for step in prepare_steps:
    if isinstance(step, dict) and "command" in step:
        command = str(step["command"])
        if runtime.get("backend") == "apptainer":
            command = command.replace(
                "cp -a /testbed/. /polar/session/workspace/",
                "cp -R /testbed/. /polar/session/workspace/",
            )
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
stop_private_ray() {
    local pids=()
    while IFS= read -r pid; do
        [ -n "${pid}" ] && pids+=("${pid}")
    done < <(
        pgrep -u "$(id -u)" -f "${RAY_TMPDIR}/ray/session_|--temp-dir=${RAY_TMPDIR}/ray|--raylet_socket_name=${RAY_TMPDIR}/ray" \
            2>/dev/null || true
    )
    if [ "${#pids[@]}" -gt 0 ]; then
        kill "${pids[@]}" 2>/dev/null || true
        sleep 2
        kill -KILL "${pids[@]}" 2>/dev/null || true
    fi
}
cleanup() {
    touch "${RUN_DIR}/monitor.stop" 2>/dev/null || true
    for pid in "${PIDS[@]}"; do kill "${pid}" 2>/dev/null || true; done
    if [ "${PRORL_GLOBAL_RAY_STOP}" = "1" ]; then
        "${RAY_BIN}" stop --force >/dev/null 2>&1 || true
    else
        stop_private_ray
    fi
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

PHASE_FILE="${RUN_DIR}/monitor.phase"
phase_mark() {
    local phase="$1"
    local detail="${2:-}"
    printf '%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${phase}" "${detail}" >> "${RUN_DIR}/phase_events.tsv"
    printf '%s\t%s\n' "${phase}" "${detail}" > "${PHASE_FILE}"
}

phase_mark "configure" "writing run settings"

cat > "${RUN_DIR}/run_settings.txt" <<EOF
RUN_DIR=${RUN_DIR}
WORK_ROOT=${WORK_ROOT}
HF_CHECKPOINT=${HF_CHECKPOINT}
REF_LOAD=${REF_LOAD}
SAVE_DIR=${SAVE_DIR}
SWEGYM_DATA=${SWEGYM_DATA}
SELECTED_GPUS=${SELECTED_GPUS}
CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
RAY_PORT=${RAY_PORT}
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
PRORL_DISABLE_NUMACTL_BIND=${PRORL_DISABLE_NUMACTL_BIND}
GPU_MONITOR_INTERVAL_SECONDS=${GPU_MONITOR_INTERVAL_SECONDS}
GPU_IDLE_UTIL_THRESHOLD=${GPU_IDLE_UTIL_THRESHOLD}
PATH=${PATH}
EOF

cat > "${RUN_DIR}/monitor_host.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
OUT="$1"
STOP="$2"
PHASE_FILE="$3"
SELECTED="$4"
INTERVAL="$5"
PYTHON_BIN="$6"
IDLE_THRESHOLD="$7"
if [ ! -f "$OUT" ]; then
  printf 'timestamp,phase,detail,gpu,memory_used_mib,memory_total_mib,util_gpu_pct,is_idle,compute_apps\n' > "$OUT"
fi
while [ ! -f "$STOP" ]; do
  "$PYTHON_BIN" - "$OUT" "$PHASE_FILE" "$SELECTED" "$IDLE_THRESHOLD" <<'PY_SAMPLE'
import csv
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

out = Path(sys.argv[1])
phase_file = Path(sys.argv[2])
selected = {x.strip() for x in sys.argv[3].split(",") if x.strip()}
idle_threshold = float(sys.argv[4])
phase = "unknown"
detail = ""
try:
    text = phase_file.read_text().strip()
    if text:
        parts = text.split("\t", 1)
        phase = parts[0]
        detail = parts[1] if len(parts) > 1 else ""
except FileNotFoundError:
    pass

def run_query(args):
    proc = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    return proc.stdout if proc.returncode == 0 else ""

gpu_text = run_query([
    "nvidia-smi",
    "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu",
    "--format=csv,noheader,nounits",
])
apps_text = run_query([
    "nvidia-smi",
    "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
    "--format=csv,noheader,nounits",
])

apps_by_uuid = {}
for line in apps_text.splitlines():
    parts = [p.strip() for p in line.split(",", 3)]
    if len(parts) != 4:
        continue
    uuid, pid, proc_name, used = parts
    apps_by_uuid.setdefault(uuid, []).append(f"{pid}:{used}MiB:{proc_name}")

timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
rows = []
for line in gpu_text.splitlines():
    parts = [p.strip() for p in line.split(",", 4)]
    if len(parts) != 5:
        continue
    idx, uuid, mem_used, mem_total, util = parts
    if selected and idx not in selected:
        continue
    try:
        util_value = float(util)
    except ValueError:
        util_value = 0.0
    rows.append({
        "timestamp": timestamp,
        "phase": phase,
        "detail": detail,
        "gpu": idx,
        "memory_used_mib": mem_used,
        "memory_total_mib": mem_total,
        "util_gpu_pct": util,
        "is_idle": "1" if util_value < idle_threshold else "0",
        "compute_apps": "|".join(apps_by_uuid.get(uuid, [])),
    })

with out.open("a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "timestamp",
        "phase",
        "detail",
        "gpu",
        "memory_used_mib",
        "memory_total_mib",
        "util_gpu_pct",
        "is_idle",
        "compute_apps",
    ])
    writer.writerows(rows)
PY_SAMPLE
  sleep "$INTERVAL"
done
EOF
chmod +x "${RUN_DIR}/monitor_host.sh"
rm -f "${RUN_DIR}/monitor.stop"
"${RUN_DIR}/monitor_host.sh" \
    "${RUN_DIR}/gpu_samples.csv" \
    "${RUN_DIR}/monitor.stop" \
    "${PHASE_FILE}" \
    "${SELECTED_GPUS}" \
    "${GPU_MONITOR_INTERVAL_SECONDS}" \
    "${PYTHON_BIN}" \
    "${GPU_IDLE_UTIL_THRESHOLD}" &
PIDS+=("$!")

phase_mark "polar_services_start" "rollout_gateway"
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
phase_mark "polar_services_ready" "rollout_gateway_health_ok"

if [ "${PRORL_GLOBAL_RAY_STOP}" = "1" ]; then
    phase_mark "ray_cleanup" "global_ray_stop"
    "${RAY_BIN}" stop --force >/dev/null 2>&1 || true
else
    phase_mark "ray_cleanup" "private_ray_stop"
    stop_private_ray
fi
if [ "${PRORL_KILL_STALE_RAY_PROCESSES}" = "1" ]; then
    phase_mark "ray_cleanup" "kill_stale_processes"
    pkill -u "$(id -un)" -f "ray::WorkerDict" >/dev/null 2>&1 || true
    pkill -u "$(id -un)" -f "ray::SGLangEngine" >/dev/null 2>&1 || true
    pkill -u "$(id -un)" -f "ray::RolloutManager" >/dev/null 2>&1 || true
    pkill -u "$(id -un)" -f "sglang.launch_server|sglang.srt|train_async.py" >/dev/null 2>&1 || true
    sleep 3
fi
strict_gpu_preflight "before_ray_start"
phase_mark "ray_start" "start_head"
if [ -z "${RAY_PORT}" ]; then
    RAY_PORT="$("${PYTHON_BIN}" - <<'PY'
import socket

sock = socket.socket()
sock.bind(("127.0.0.1", 0))
print(sock.getsockname()[1])
sock.close()
PY
)"
fi
RAY_ADDRESS_URI="127.0.0.1:${RAY_PORT}"
echo "Starting Ray at ${RAY_ADDRESS_URI} with CUDA_VISIBLE_DEVICES=${SELECTED_GPUS}"
RAY_START_ARGS=(
    start
    --head
    --node-ip-address 127.0.0.1
    --port "${RAY_PORT}"
    --num-gpus "$(echo "${SELECTED_GPUS}" | awk -F, '{print NF}')"
    --disable-usage-stats
    --include-dashboard=false
)
if [ -n "${RAY_NUM_CPUS:-}" ]; then
    RAY_START_ARGS+=(--num-cpus "${RAY_NUM_CPUS}")
fi
CUDA_VISIBLE_DEVICES="${SELECTED_GPUS}" "${RAY_BIN}" "${RAY_START_ARGS[@]}"

CUDA_VISIBLE_DEVICES="${SELECTED_GPUS}" RAY_ADDRESS="${RAY_ADDRESS_URI}" "${PYTHON_BIN}" - <<PY
import ray

expected = len([x for x in "${SELECTED_GPUS}".split(",") if x.strip()])
ray.init(address="${RAY_ADDRESS_URI}", ignore_reinit_error=True)
resources = ray.cluster_resources()
actual = int(resources.get("GPU", 0))
ray.shutdown()
if actual != expected:
    raise SystemExit(
        f"Ray GPU resource mismatch: expected {expected} from "
        f"SELECTED_GPUS=${SELECTED_GPUS}, got {actual}. "
        "Refusing to launch because this may attach to the wrong Ray cluster."
    )
print(f"Ray GPU preflight OK: {actual} GPU(s) at ${RAY_ADDRESS_URI}")
PY
strict_gpu_preflight "before_train_async"
phase_mark "train_async" "slime_megatron_sglang"

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
SGLANG_SKIP_SERVER_WARMUP="${SGLANG_SKIP_SERVER_WARMUP:-1}"
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
    "CUDA_DEVICE_ORDER",
    "CUDA_VISIBLE_DEVICES",
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
CUDA_VISIBLE_DEVICES="${SELECTED_GPUS}" RAY_ADDRESS="${RAY_ADDRESS_URI}" "${PYTHON_BIN}" "${SLIME_DIR}/train_async.py" \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node 2 \
    --num-gpus-per-node "${SELECTED_GPU_COUNT}" \
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
    --no-masked-softmax-fusion \
    --no-gradient-accumulation-fusion \
    --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}" \
    --sglang-context-length "${SGLANG_CONTEXT_LENGTH}" \
    --sglang-disable-cuda-graph \
    --sglang-disable-piecewise-cuda-graph \
    --sglang-disable-overlap-schedule \
    $(if [ "${SGLANG_SKIP_SERVER_WARMUP}" = "1" ]; then printf '%s\n' "--sglang-skip-server-warmup"; fi) \
    --sglang-cuda-graph-max-bs "${SGLANG_CUDA_GRAPH_MAX_BS}" \
    --sglang-rl-on-policy-target fsdp \
    --sglang-tool-call-parser qwen3_coder \
    --router-policy round_robin \
    --sglang-router-port 19000 \
    2>&1 | tee "${RUN_DIR}/train.log"
status="${PIPESTATUS[0]}"
set -e

phase_mark "summarize" "gpu_idle_summary"
touch "${RUN_DIR}/monitor.stop"
sleep 2
"${PYTHON_BIN}" - "${RUN_DIR}/gpu_samples.csv" "${RUN_DIR}/gpu_idle_summary.tsv" "${GPU_IDLE_UTIL_THRESHOLD}" "${GPU_MONITOR_INTERVAL_SECONDS}" <<'PY_SUMMARY'
import csv
import sys
from collections import defaultdict
from pathlib import Path

samples_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
idle_threshold = float(sys.argv[3])
interval = float(sys.argv[4])

rows = []
if samples_path.exists():
    with samples_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

by_gpu = defaultdict(list)
for row in rows:
    by_gpu[row["gpu"]].append(row)

with summary_path.open("w") as out:
    out.write(
        "gpu\tphase\tsamples\tseconds_est\tidle_samples\tidle_seconds_est\t"
        "idle_fraction\tactive_process_samples\tidle_without_process_samples\t"
        "idle_with_process_samples\tidle_util_threshold_pct\n"
    )
    for gpu in sorted(by_gpu, key=lambda x: int(x)):
        by_phase = defaultdict(list)
        for row in by_gpu[gpu]:
            by_phase[row["phase"]].append(row)
        for phase in sorted(by_phase):
            phase_rows = by_phase[phase]
            samples = len(phase_rows)
            idle_rows = [
                r for r in phase_rows
                if float(r["util_gpu_pct"] or 0.0) < idle_threshold
            ]
            active_proc = sum(1 for r in phase_rows if r["compute_apps"])
            idle_no_proc = sum(1 for r in idle_rows if not r["compute_apps"])
            idle_with_proc = sum(1 for r in idle_rows if r["compute_apps"])
            seconds = samples * interval
            idle_seconds = len(idle_rows) * interval
            out.write(
                f"{gpu}\t{phase}\t{samples}\t{seconds:.1f}\t"
                f"{len(idle_rows)}\t{idle_seconds:.1f}\t"
                f"{(len(idle_rows) / samples if samples else 0):.4f}\t"
                f"{active_proc}\t{idle_no_proc}\t{idle_with_proc}\t"
                f"{idle_threshold:g}\n"
            )
PY_SUMMARY

echo "Run finished with status ${status}"
echo "Run dir: ${RUN_DIR}"
echo "GPU samples: ${RUN_DIR}/gpu_samples.csv"
echo "GPU idle summary: ${RUN_DIR}/gpu_idle_summary.tsv"
exit "${status}"
