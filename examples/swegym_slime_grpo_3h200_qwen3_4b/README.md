# SWE-Gym Slime GRPO on 3xH200 with Qwen3-4B

This example is a pushable launcher for a single-node server with 3 H200 GPUs:

- GPU 0-1: Megatron actor/ref GRPO training
- GPU 2: SGLang rollout inference
- Model weights: `/work1/huggingface_models/Qwen3-4B`
- Big reusable artifacts: `/work1/yokyung/prorl_agent_server_env`
- Runtime logs/session/temp/Ray state: `/home/yokyung/prorl_agent_server_runs`
- No intentional use of host `/tmp` or host `/opt`

The launcher uses Docker SWE-Gym task containers. It prepares a small SWE-Gym
subset if no SWE-Gym data/images are present yet, converts Qwen3-4B HF weights
to Slime/Megatron `torch_dist`, starts Polar rollout/gateway services, and then
starts Slime GRPO.

## Files

| File | Purpose |
|---|---|
| `setup_env_3h200.sh` | Creates the Python env, installs deps, clones Slime/Megatron, applies patches |
| `convert_qwen3_4b.sh` | Converts `/work1/huggingface_models/Qwen3-4B` to Megatron `torch_dist` |
| `prepare_swegym_docker_data.sh` | Builds Docker runtime images and a JSONL SWE-Gym subset |
| `launch_qwen3_4b_3h200.sh` | Main 3-H200 GRPO launcher |
| `polar_config.docker_3h200.yaml` | Polar/Qwen-Code/SWE-bench harness config |
| `topology.docker_3h200.yaml` | Local Polar rollout/gateway topology |

## Quick Start On The 3-H200 Server

From a fresh clone:

```bash
cd ~/ProRL-Agent-Server
bash examples/swegym_slime_grpo_3h200_qwen3_4b/setup_env_3h200.sh
bash examples/swegym_slime_grpo_3h200_qwen3_4b/prepare_swegym_docker_data.sh
bash examples/swegym_slime_grpo_3h200_qwen3_4b/convert_qwen3_4b.sh
bash examples/swegym_slime_grpo_3h200_qwen3_4b/launch_qwen3_4b_3h200.sh
```

If the converted checkpoint and Docker images already exist, the last command is
the only command needed.

## Safe Defaults

The launcher is intentionally conservative:

- `rollout_batch_size=1`
- `n_samples_per_prompt=1`
- `32k` context
- `1024` max completion tokens
- `num_rollout=4` by default, meaning one warmup-like first rollout plus three more rollout/update opportunities
- checkpoint save interval is very large by default
- the launch script deletes its run checkpoint directory on exit unless `KEEP_CHECKPOINTS=1`

For a longer real run, increase these only after the smoke run completes:

```bash
NUM_ROLLOUT=20 KEEP_CHECKPOINTS=1 bash examples/swegym_slime_grpo_3h200_qwen3_4b/launch_qwen3_4b_3h200.sh
```

## Important Paths

Defaults can be overridden with environment variables:

```bash
export HF_CHECKPOINT=/work1/huggingface_models/Qwen3-4B
export WORK_ROOT=/work1/yokyung/prorl_agent_server_env
export RUN_ROOT=/home/yokyung/prorl_agent_server_runs/swegym_slime_grpo_3h200
export TORCH_DIST_DIR=$WORK_ROOT/checkpoints/Qwen3-4B_torch_dist
export SWEGYM_DATA=$WORK_ROOT/data/swegym_train_docker_3h200.jsonl
```

The setup/data scripts set build/cache paths away from `/tmp`, mostly under
`/work1/yokyung/prorl_agent_server_env`. The launch script puts runtime temp,
Ray state, Python bytecode cache, Matplotlib cache, FlashInfer workspace, logs,
and Polar session directories under `/home/yokyung/prorl_agent_server_runs`.
Docker itself still uses the host Docker daemon storage location. The SWE-bench
task containers may reference `/opt/miniconda3` internally because that is where
their testbed Python environment lives; the launcher does not write to host
`/opt`.

## SWE-Gym Data

`prepare_swegym_docker_data.sh` uses the existing
`examples/swegym_slime_grpo/build_docker_images.py` helper. It fetches SWE-Gym
metadata, builds Docker images for selected rows, and writes a JSONL dataset
whose rows include `metadata.runtime_image`.

By default it prepares 8 tasks:

```bash
SWEGYM_MAX_TASKS=8 bash examples/swegym_slime_grpo_3h200_qwen3_4b/prepare_swegym_docker_data.sh
```

You can pin exact instances:

```bash
SWEGYM_INSTANCE_ID=getmoto__moto-7365,pandas-dev__pandas-58335 \
bash examples/swegym_slime_grpo_3h200_qwen3_4b/prepare_swegym_docker_data.sh
```

## Before Launch

Check that all 3 GPUs are free:

```bash
nvidia-smi
```

The launch script refuses to run unless it sees at least 3 GPUs and each chosen
GPU is below `MAX_PRELAUNCH_GPU_MEM_MIB` memory use. It also checks host RAM;
override `HOST_MIN_AVAILABLE_GIB` if the target server has a different safe
threshold.

Optional optimizer CPU offload can be enabled with:

```bash
ENABLE_OPTIMIZER_CPU_OFFLOAD=1 bash examples/swegym_slime_grpo_3h200_qwen3_4b/launch_qwen3_4b_3h200.sh
```

The default leaves optimizer offload disabled because the intended hardware has
2 H200s for training. Enable it only if actor training still approaches OOM.

## Expected First Result

The first run is a systems characterization run, not a useful trained model. It
should answer:

- whether 32k rollouts produce multi-call SWE-agent traces,
- whether 2 training H200s avoid the previous actor-backward OOM,
- how much host RAM Docker/Slime use,
- and whether Qwen3-4B can produce non-empty patches on this SWE-Gym subset.
