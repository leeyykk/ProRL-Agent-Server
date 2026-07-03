#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

WORK_ROOT="${WORK_ROOT:-/work1/yokyung/prorl_agent_server_env}"
HF_CHECKPOINT="${HF_CHECKPOINT:-/work1/huggingface_models/Qwen3.5-4B}"
TORCH_DIST_DIR="${TORCH_DIST_DIR:-${WORK_ROOT}/checkpoints/Qwen3.5-4B_torch_dist}"
SLIME_DIR="${SLIME_DIR:-${PROJECT_ROOT}/slime}"
MEGATRON_DIR="${MEGATRON_DIR:-${PROJECT_ROOT}/Megatron-LM}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
TP_SIZE="${TP_SIZE:-1}"
PP_SIZE="${PP_SIZE:-1}"
CP_SIZE="${CP_SIZE:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${TP_SIZE}}"
TRANSFORMER_IMPL="${TRANSFORMER_IMPL:-local}"

export TMPDIR="${WORK_ROOT}/tmp"
export TEMP="${TMPDIR}"
export TMP="${TMPDIR}"
export HF_HOME="${WORK_ROOT}/hf-home"
export HF_HUB_CACHE="${WORK_ROOT}/hf-hub-cache"
export TRANSFORMERS_CACHE="${WORK_ROOT}/transformers-cache"
export PYTHONPYCACHEPREFIX="${WORK_ROOT}/pycache"
export FLASHINFER_WORKSPACE_BASE="${WORK_ROOT}/flashinfer"
mkdir -p "${TMPDIR}" "${HF_HOME}" "${HF_HUB_CACHE}" "${TRANSFORMERS_CACHE}" \
    "${PYTHONPYCACHEPREFIX}" "${FLASHINFER_WORKSPACE_BASE}" "${TORCH_DIST_DIR}"

if [ ! -d "${HF_CHECKPOINT}" ]; then
    echo "ERROR: HF checkpoint directory not found: ${HF_CHECKPOINT}" >&2
    exit 1
fi
if [ ! -f "${SLIME_DIR}/tools/convert_hf_to_torch_dist.py" ]; then
    echo "ERROR: Slime checkout missing: ${SLIME_DIR}" >&2
    exit 1
fi

# Mirrors slime/slime/scripts/models/qwen3.5-4B.sh. Qwen3.5-4B is a
# Qwen3_5ForConditionalGeneration checkpoint with nested text weights and a
# hybrid GatedDeltaNet/full-attention layout, so it must use the Qwen3.5 spec.
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

echo "Converting ${HF_CHECKPOINT} -> ${TORCH_DIST_DIR}"
CUDA_DEVICE_MAX_CONNECTIONS=1 \
PYTHONPATH="${MEGATRON_DIR}:${SLIME_DIR}:${PROJECT_ROOT}/src" \
"${PYTHON_BIN}" -m torch.distributed.run --nproc_per_node "${NPROC_PER_NODE}" \
    "${SLIME_DIR}/tools/convert_hf_to_torch_dist.py" \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "${HF_CHECKPOINT}" \
    --save "${TORCH_DIST_DIR}" \
    --tensor-model-parallel-size "${TP_SIZE}" \
    --pipeline-model-parallel-size "${PP_SIZE}" \
    --context-parallel-size "${CP_SIZE}" \
    --expert-model-parallel-size 1 \
    --expert-tensor-parallel-size 1 \
    --transformer-impl "${TRANSFORMER_IMPL}" \
    --no-rope-fusion \
    --no-persist-layer-norm \
    --no-gradient-accumulation-fusion

echo "Done: ${TORCH_DIST_DIR}"
