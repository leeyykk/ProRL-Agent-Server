# Fresh B200 Reproduction: Qwen3.5-4B Codex SWE-Gym GRPO

This is the portable path for reproducing the worker-sweep experiment on a new
single-node B200 server.

It targets the experiment currently being rerun with:

- Qwen3.5-4B
- Codex harness
- SWE-Gym Docker-backed tasks
- GRPO group size 4
- 16K rollout/training token cap
- TP=2 + sequence parallelism on 2 training GPUs
- 1 rollout GPU
- `polar_min_complete_accept_fraction=1.0`
- Four worker configs:
  - `init=1, run=4`
  - `init=4, run=1`
  - `init=4, run=4`
  - `init=4, run=8`

## 1. Fresh Setup

Choose the paths for the new server. These are examples; change them for the
B200 machine.

```bash
cd /path/to/ProRL-Agent-Server

bash examples/swegym_slime_grpo_3h200_qwen3_4b/setup_fresh_b200.sh \
  --work-root /work1/$USER/prorl_agent_server_env \
  --run-root /work1/$USER/prorl_agent_server_runs/swegym_slime_grpo_b200 \
  --hf-checkpoint /work1/$USER/huggingface_models/Qwen3.5-4B \
  --hf-model-id Qwen/Qwen3.5-4B \
  --swegym-data /work1/$USER/prorl_agent_server_env/data/swegym_train_docker_b200.jsonl \
  --swegym-max-tasks 24 \
  --torch-dist-dir /work1/$USER/prorl_agent_server_env/checkpoints/Qwen3.5-4B_torch_dist_tp2 \
  --tp-size 2
```

This does the following:

- creates/repairs `.venv`
- clones/patches Slime and Megatron-LM
- installs SGLang with pre-release dependencies enabled
- pins NumPy to `<2`
- downloads Qwen3.5-4B if the local checkpoint is missing
- builds the SWE-Gym Docker runtime images and JSONL
- preinstalls Codex CLI to a host prefix that is mounted into containers
- converts Qwen3.5-4B to Megatron `torch_dist` with TP=2

If some artifacts already exist, use skip flags:

```bash
--skip-env
--skip-model-download
--skip-data
--skip-convert
--skip-codex-install
```

## 2. Run The 4-Config Sweep

Example: rollout on GPU 1, training on GPUs 4 and 6.

```bash
bash examples/swegym_slime_grpo_3h200_qwen3_4b/run_qwen35_codex_worker_sweep_b200.sh \
  --work-root /work1/$USER/prorl_agent_server_env \
  --run-root /work1/$USER/prorl_agent_server_runs/swegym_slime_grpo_b200 \
  --hf-checkpoint /work1/$USER/huggingface_models/Qwen3.5-4B \
  --swegym-data /work1/$USER/prorl_agent_server_env/data/swegym_train_docker_b200.jsonl \
  --torch-dist-dir /work1/$USER/prorl_agent_server_env/checkpoints/Qwen3.5-4B_torch_dist_tp2 \
  --preinstalled-agent-cli /work1/$USER/prorl_agent_server_env/agent_cli/codex_0.121.0 \
  --train-gpus 4,6 \
  --rollout-gpu 1
```

The script waits until the selected GPUs are below
`--max-prelaunch-gpu-mem-mib` and host memory is above
`--host-min-available-gib`.

For a fail-fast launch attempt instead of waiting:

```bash
... --run-now
```

## 3. Outputs

The sweep manifest is written to:

```text
$RUN_ROOT/qwen35_codex_worker_sweep_accept10_sp16k_te_24rollouts_*/worker_sweep_manifest.tsv
```

The four comparison figures are written to:

```text
$RUN_ROOT/main_figures/codex_qwen35_4b_exact_workers_init1_run4_accept10_sp16k_te_24rollouts.svg
$RUN_ROOT/main_figures/codex_qwen35_4b_exact_workers_init4_run1_accept10_sp16k_te_24rollouts.svg
$RUN_ROOT/main_figures/codex_qwen35_4b_exact_workers_init4_run4_accept10_sp16k_te_24rollouts.svg
$RUN_ROOT/main_figures/codex_qwen35_4b_exact_workers_init4_run8_accept10_sp16k_te_24rollouts.svg
```

Each individual run directory also contains:

- `train.log`
- `monitor.log`
- `polar_config.yaml`
- `topology.yaml`
- `exact_characterization.svg`

## Notes

- The setup script intentionally preinstalls Codex CLI once and mounts it into
  task containers. This avoids per-container `npm install` overhead.
- The sweep uses `TP_SIZE=2`, `SEQUENCE_PARALLEL=1`, and
  `TRANSFORMER_IMPL=transformer_engine`; this is the SP configuration that
  avoided the long-context OOM path in the prior 16K runs.
- `polar_min_complete_accept_fraction=1.0` means a task group must complete all
  four samples before being accepted for training.
- Checkpoint saving is effectively disabled for characterization runs
  (`SAVE_INTERVAL=1000000`, `KEEP_CHECKPOINTS=0`).
