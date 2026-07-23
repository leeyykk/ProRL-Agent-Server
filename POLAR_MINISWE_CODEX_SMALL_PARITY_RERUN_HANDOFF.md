# POLAR Mini-SWE/Codex small-parity exact rerun handoff

Last verified: 2026-07-23

This is the canonical zero-context handoff for reproducing the repaired
five-experiment comparison. It supersedes the historical debugging diary and
pre-repair runbook.

## Goal and exact matrix

Run sequentially on the same three B200 GPUs:

1. SkyRL + Mini-SWE, four generation workers.
2. POLAR + Mini-SWE, `init4/run2/eval4`.
3. POLAR + Mini-SWE, `init4/run4/eval4`.
4. POLAR + Codex CLI, `init4/run2/eval4`.
5. POLAR + Codex CLI, `init4/run4/eval4`.

Common protocol: Qwen3.5-4B; four rollout iterations; rollout batch four; two
samples per prompt; microbatch one; async/staleness two; one training and two
rollout GPUs; TP1. Use a 32,768-token total sequence/training budget, split as
30,720 retained input plus at most 2,048 tokens for the next completion.
Dynamic batching is disabled. Mini-SWE has 40 turns. Codex uses four empty-diff
retries and two review retries.

SkyRL assigns GPUs 0/1 to vLLM and GPU 2 to FSDP. POLAR assigns GPU 0 to
Megatron and GPUs 1/2 to SGLang.

## Locations

```text
worktree: /NHNHOME/home/ProRL-Agent-Server/.worktrees/prorl-miniswe-harness
branch: prorl-miniswe-harness
SkyRL: /NHNHOME/home/SkyRL
model: /NHNHOME/home/huggingface_models/Qwen3.5-4B
dataset: /NHNHOME/home/prorl_agent_server_env/data/swegym_train_apptainer_3h200_sandbox.jsonl
TP1 checkpoint: /NHNHOME/home/prorl_agent_server_env/checkpoints/Qwen3.5-4B_torch_dist_tp1
Codex CLI: /NHNHOME/home/prorl_agent_server_env/agent_cli/codex_0.121.0
Singularity: /usr/bin/singularity
```

Required ignored operational assets:

```text
tmp/prorl_miniswe_full_matrix_20260720T085614Z/
tmp/parity32k_instrumented_20260723/run_polar_miniswe.sh
tmp/parity32k_instrumented_20260723/run_polar_codex.sh
```

The runtime snapshot under the first directory is executable experiment input.
Do not delete or replace it with the installed package.

## Repairs that must remain

Mini-SWE edits `/polar/session/workspace`. The evaluator rewrites conventional
`/testbed` commands to that writable checkout and records
`evaluated_repo_dir`. Direct sandbox execution replaces unsupported Apptainer
instance mode. The runtime fallback grader supports SWE-Gym repositories that
the installed SWE-bench package does not recognize.

Codex must not force `unified_exec`. The harness enables
`apply_patch_freeform`, tells the model to edit and test the existing workspace,
continues after empty diffs, and performs bounded same-session reviews. The
matrix controller forwards `CODEX_EMPTY_DIFF_RETRIES`,
`CODEX_REVIEW_RETRIES`, and `CODEX_EVALUATOR_IN_PLACE`. Evaluator preparation
copies read-only `/testbed` to `/polar/session/workspace`. The fallback test
command uses `/polar/session/home/.venv/bin/python`.

The repaired Codex reference results were 6/40 resolved for run2 and 8/40 for
run4, with zero patch-application or evaluator failures. Older 0/40 results
used a broken patch-tool path and are invalid quality measurements.

## Runtime constraints

Use extracted sandbox directories only. Never use SIF images: this host has no
usable loop devices. Do not use Docker or Nsight. Use:

```text
POLAR_APPTAINER_BIN=/usr/bin/singularity
POLAR_APPTAINER_EXEC_ARGS=--no-mount bind-paths
RUNTIME_BACKEND=apptainer
NSYS_CAPTURE_RANGE=none
NSYS_EXPORT_FORMATS=none
NSYS_GPU_METRICS_DEVICES=none
NSYS_BIN=/bin/false
```

`/tmp` is noexec; use configured session directories or `/var/tmp`. A command
failure containing `bwrap: loopback: Failed RTM_NEWADDR` is the outer tool
sandbox, not the experiment.

## Preflight

```bash
cd /NHNHOME/home/ProRL-Agent-Server/.worktrees/prorl-miniswe-harness
PYTHONPATH=src /NHNHOME/home/ProRL-Agent-Server/.venv/bin/pytest -q \
  tests/agent/test_codex_harness.py \
  tests/agent/test_mini_swe_agent_harness.py \
  tests/trajectory/test_swebench_harness.py
git diff --check
bash -n tmp/parity32k_instrumented_20260723/run_polar_miniswe.sh
bash -n tmp/parity32k_instrumented_20260723/run_polar_codex.sh
python3 -m py_compile /NHNHOME/home/SkyRL/skyrl/train/fully_async_trainer.py
```

Sandbox Codex preflight:

```bash
/usr/bin/singularity exec --no-mount bind-paths \
  --bind /NHNHOME/home/prorl_agent_server_env/agent_cli/codex_0.121.0:/agent-cli:ro \
  /NHNHOME/home/prorl_agent_server_env/apptainer_sandboxes/swegym_b200/getmoto__moto-4918 \
  /bin/sh -c "test -d /testbed && test -x /agent-cli/bin/codex && PATH=/agent-cli/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin codex --version"
# codex-cli 0.121.0
```

Confirm no competing processes and all GPUs are free. Never kill another job to
obtain GPUs and never use broad `pkill`.

## Required instrumentation

SkyRL `fully_async_trainer.py` contains `group_start_wall_s`,
`group_end_wall_s`, `SKYRL_BATCH_TIMING_JSONL`, and `collection_span_s`.
Preserve these while retaining unrelated dirty SkyRL changes. Each accepted
batch record contains global step, group/sample counts, earliest start, latest
finish, duration, and accepted IDs.

SkyRL stores raw one-second GPU samples in `gpu_samples.jsonl`; POLAR stores
`gpu_samples.csv`. For every GPU report peak used MiB, timestamp, utilization
at the peak, nearest phase/update/batch boundary, sampling window, and idle
percentage. Keep raw samples.

SkyRL phase timers include `step`, `wait_for_generation_buffer`,
`run_training`, `fwd_logprobs_values_reward`, `train_critic_and_policy`, and
`sync_weights`. Report update one separately. Later updates define steady
state. POLAR session JSON timestamps define batch span: group by rollout step,
then use earliest accepted-session start through latest accepted-session end.
Do not substitute queue-wait time.

## Active instrumented suite

```text
SkyRL tmux: parity32k_20260723T040000Z
continuation tmux: parity32k_cont_20260723T040000Z
SkyRL stem: miniswe_small_parity_cap32k_instrumented_20260723T040000Z
POLAR Mini stem: miniswe_small_parity_cap32k_instrumented_polar_miniswe_20260723T040000Z
POLAR Codex stem: miniswe_small_parity_cap32k_instrumented_polar_codex_20260723T040000Z
```

The continuation waits for SkyRL, then runs both Mini-SWE rows and both repaired
Codex rows. Expected total is roughly four to five hours.

```bash
tmux has-session -t parity32k_20260723T040000Z 2>/dev/null || true
tmux has-session -t parity32k_cont_20260723T040000Z 2>/dev/null || true
tail -80 /NHNHOME/home/profiled_runs/miniswe_small_parity_cap32k_instrumented_20260723T040000Z/skyrl_console.log
tail -80 /NHNHOME/home/profiled_runs/miniswe_small_parity_cap32k_instrumented_polar_miniswe_20260723T040000Z/suite.log 2>/dev/null || true
tail -80 /NHNHOME/home/profiled_runs/miniswe_small_parity_cap32k_instrumented_polar_codex_20260723T040000Z/suite.log 2>/dev/null || true
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
```

## Fresh exact rerun

Generate a unique UTC tag and launch SkyRL exactly:

```bash
cd /NHNHOME/home/ProRL-Agent-Server/.worktrees/prorl-miniswe-harness
tag=$(date -u +%Y%m%dT%H%M%SZ)
base="miniswe_small_parity_cap32k_instrumented_${tag}"
sky_tmux="parity32k_${tag}"
cont_tmux="parity32k_cont_${tag}"
sky_dir="/NHNHOME/home/profiled_runs/${base}"
mkdir -p "${sky_dir}"
batch_file="/NHNHOME/home/skyrl_prorl_runs/${base}_skyrl_workers4/init4_run4/batch_collection_timing.jsonl"

tmux new-session -d -s "${sky_tmux}" \
  "cd /NHNHOME/home/SkyRL && env DATA_DIR=/NHNHOME/home/skyrl_prorl_data_small16 RUN_ROOT=/NHNHOME/home/skyrl_prorl_runs MINISWE_SESSION_ROOT=/var/tmp TMPDIR=/var/tmp/skyrl_small_parity_tmp TILELANG_CACHE_DIR=/var/tmp/skyrl_small_parity_tilelang PYTHONPATH=/NHNHOME/home/SkyRL/examples/train/mini_swe_agent/fla_shim CUDA_VISIBLE_DEVICES=0,1,2 LOGGER=console CONFIGS=init4_run4 STOP_ON_FAILURE=1 FORCE_PREPARE_DATA=1 SWEEP_STEM=${base}_skyrl_workers4 REPORT_PATH=${sky_dir}/SKYRL_WORKERS4_REPORT.md NUM_ROLLOUT=4 ROLLOUT_BATCH_SIZE=4 N_SAMPLES_PER_PROMPT=2 NUM_STEPS_PER_ROLLOUT=2 MICRO_BATCH_SIZE=1 NUM_POLICY_GPUS=1 NUM_INFERENCE_GPUS=2 TP_SIZE=1 MAX_INPUT_LENGTH=30720 MAX_GENERATE_LENGTH=2048 SKYRL_BATCH_TIMING_JSONL=${batch_file} bash examples/train/mini_swe_agent/run_skyrl_prorl_worker_sweep.sh > ${sky_dir}/skyrl_console.log 2>&1"
```

Immediately queue a second tmux that waits for SkyRL to disappear, then runs:

```bash
tmux new-session -d -s "${cont_tmux}" \
  "while tmux has-session -t ${sky_tmux} 2>/dev/null; do sleep 15; done; cd /NHNHOME/home/ProRL-Agent-Server/.worktrees/prorl-miniswe-harness; SUITE_STEM=${base}_polar_miniswe bash tmp/parity32k_instrumented_20260723/run_polar_miniswe.sh; mini_status=\$?; if [ \$mini_status -eq 0 ]; then SUITE_STEM=${base}_polar_codex bash tmp/parity32k_instrumented_20260723/run_polar_codex.sh; fi; exit \$mini_status"
```

Both wrappers use `CONFIGS="4:2 4:4"`; do not launch run4 separately. Verify:

```text
MAX_TOKENS_PER_GPU=32768
SGLANG_CONTEXT_LENGTH_FOR_SWEEP=32768
ROLLOUT_MAX_PROMPT_LEN_FOR_SWEEP=30720
ROLLOUT_MAX_RESPONSE_LEN_FOR_SWEEP=2048
CODEX_EMPTY_DIFF_RETRIES=4
CODEX_REVIEW_RETRIES=2
CODEX_EVALUATOR_IN_PLACE=0
```

## Validity gates

A valid row requires exit status 0, expected terminal artifacts, the uniform
32K settings, raw GPU samples, batch-timing data, and no SIF/Docker/Nsight.
Resolved POLAR evaluations must record
`evaluated_repo_dir=/polar/session/workspace`. Codex must advertise a working
patch tool, evaluate nonempty patches, report apply/evaluator failures, accept
real training traces, and never loop at `accepted=0/4`.

A vanished tmux or `suite.complete` alone is not proof. Inspect manifests,
sweep logs, result JSON, and GPUs. After completion tmux must be gone and GPUs
must return to zero allocation.

## Results and report update

For every row record start/finish/wall time, terminal statuses, resolved and
unique tasks, calls/tokens/throughput, every accepted-batch span, first and
steady update phases, GPU idle, peak memory with timestamp and phase, cap drops,
and complete artifact paths. For Codex also record empty/nonempty diffs and
patch/evaluator failures.

Update:

```text
examples/swegym_slime_grpo_3h200_qwen3_4b/POLAR_MINISWE_SMALL_PARITY_SKYRL4_CODEX_COMPARISON.md
```

Every comparison table must contain all five experiments. Use `not collected`
instead of `pending extraction`. Run `git diff --check`.

## Safe stop

Stopping destroys the active row and requires explicit authorization. Resolve
the exact tmux and process group first. Never kill all Ray, Python, or
Singularity processes and never recursively delete a worktree or run root.
Preserve partial artifacts and restart with a fresh stem.

## Historical sanity values

| Experiment | Wall min | Terminal | Resolved | Unique groups |
| --- | ---: | ---: | ---: | ---: |
| SkyRL Mini-SWE workers4 | 32.7 | 40 | 12 | not reported |
| POLAR Mini-SWE run2 | 47.1 | 29 | 14 | 8 |
| POLAR Mini-SWE run4 | 38.2 | 39 | 14 | 8 |
| POLAR Codex run2, tool-fixed | 54.9 | 40 | 6 | 5 |
| POLAR Codex run4, tool-fixed | 54.0 | 40 | 8 | 6 |

These are sanity references, not substitutes for the uniform-cap instrumented
rerun. Older Codex 0/40 rows used a broken tool path and are invalid.
