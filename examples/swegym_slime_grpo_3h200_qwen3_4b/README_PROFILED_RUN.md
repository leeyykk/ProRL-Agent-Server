# Fresh H200 Profiled Worker Sweep

This workflow prepares and runs the four Qwen3.5-4B SWE-Gym configurations:

- init=8, run=8
- init=8, run=16
- init=16, run=16
- init=16, run=32

It uses two GPUs for TP2 Megatron training and one GPU for SGLang rollout.
The host must provide NVIDIA drivers, three H200-class GPUs, Docker, Git, npm,
Python 3.12, network access during setup, and a downloaded NVIDIA Nsight
Systems Linux x86_64 `.run` installer.

## 1. Clone And Select The Branch

```bash
git clone git@github.com:leeyykk/ProRL-Agent-Server.git
cd ProRL-Agent-Server
git switch profiled_run
```

## 2. Install Nsight Systems Privately

```bash
bash examples/swegym_slime_grpo_3h200_qwen3_4b/install_nsight_systems_private.sh \
  --installer "$HOME/Downloads/NsightSystems-linux-public-2026.3.1.157-3804839.run" \
  --install-dir "$HOME/tools/nsight-systems-2026.3.1"
```

Copy the printed `NSYS_BIN` value for the run command.

## 3. Prepare The Environment, Model, Dataset, And Harness

Choose writable locations with enough space:

```bash
export WORK_ROOT=/work1/$USER/prorl_agent_server_env
export RUN_ROOT=/work1/$USER/prorl_agent_server_runs/swegym_slime_grpo_3h200
export HF_CHECKPOINT=/work1/$USER/huggingface_models/Qwen3.5-4B
export SWEGYM_DATA=$WORK_ROOT/data/swegym_train_docker_3h200.jsonl
export TORCH_DIST_DIR=$WORK_ROOT/checkpoints/Qwen3.5-4B_torch_dist_tp2
export PREINSTALLED_AGENT_CLI=$WORK_ROOT/agent_cli/codex_0.121.0

bash examples/swegym_slime_grpo_3h200_qwen3_4b/setup_fresh_profiled_h200.sh \
  --work-root "$WORK_ROOT" \
  --run-root "$RUN_ROOT" \
  --hf-checkpoint "$HF_CHECKPOINT" \
  --hf-model-id Qwen/Qwen3.5-4B \
  --swegym-data "$SWEGYM_DATA" \
  --swegym-max-tasks 24 \
  --torch-dist-dir "$TORCH_DIST_DIR" \
  --tp-size 2 \
  --preinstalled-agent-cli "$PREINSTALLED_AGENT_CLI"
```

The setup is restartable. Existing model and dependency checkouts are reused;
the skip switches documented by `setup_fresh_profiled_h200.sh --help` can bypass
completed stages explicitly.

## 4. Run The Profiled Sweep

```bash
export NSYS_BIN=$HOME/tools/nsight-systems-2026.3.1/pkg/target-linux-x64/nsys
export PROFILE_ROOT=$HOME/profiled_runs

NSYS_BIN="$NSYS_BIN" \
PROFILE_ROOT="$PROFILE_ROOT" \
WORK_ROOT="$WORK_ROOT" \
RUN_ROOT="$RUN_ROOT" \
HF_CHECKPOINT="$HF_CHECKPOINT" \
SWEGYM_DATA="$SWEGYM_DATA" \
TORCH_DIST_DIR="$TORCH_DIST_DIR" \
PREINSTALLED_AGENT_CLI="$PREINSTALLED_AGENT_CLI" \
GPU_CANDIDATES=0,1,2,3 \
bash examples/swegym_slime_grpo_3h200_qwen3_4b/run_qwen35_codex_nsys_large_worker_sweep_when_free.sh
```

The script waits until three candidate GPUs are free before each configuration.
Reports are written under `PROFILE_ROOT`; ProRL logs and rollout results are
written under `RUN_ROOT`. Full copied session workspaces are disabled by
default. Set `KEEP_RUNTIME_ARTIFACTS=1` only for runtime-workspace debugging.

The report contains:

- `polar:*` lifecycle and per-session ranges
- `slime:*` training and synchronization ranges
- `sglang:prefill:batch_size=N`
- `sglang:decode:batch_size=N`
- CUDA kernels and memory operations
