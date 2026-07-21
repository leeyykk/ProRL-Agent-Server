# POLAR Mini-SWE small-parity experiment handoff

Last operational audit: 2026-07-22

This is the canonical zero-context runbook for the reduced SkyRL/POLAR/Codex comparison on this server. Read it before launching, stopping, or interpreting the suite.

Related files:

```text
# Long debugging and implementation history
/NHNHOME/home/ProRL-Agent-Server/.worktrees/prorl-miniswe-harness/POLAR_MINISWE_HARNESS_CHANGE_HANDOFF.md

# Living results table
/NHNHOME/home/ProRL-Agent-Server/.worktrees/prorl-miniswe-harness/examples/swegym_slime_grpo_3h200_qwen3_4b/POLAR_MINISWE_SMALL_PARITY_SKYRL4_CODEX_COMPARISON.md
```

## 1. Goal and matrix

The sequential suite compares:

1. SkyRL + Mini-SWE-Agent, 4 parallel workers.
2. POLAR + Mini-SWE-Agent, init/run/eval concurrency 4/2/4.
3. POLAR + Mini-SWE-Agent, init/run/eval concurrency 4/4/4.
4. POLAR + Codex CLI, init/run/eval concurrency 4/2/4.
5. POLAR + Codex CLI, init/run/eval concurrency 4/4/4.

All rows use Qwen3.5-4B and three B200 GPUs. POLAR assigns GPU 0 to training and GPUs 1/2 to rollout. SkyRL assigns GPUs 0/1 to inference and GPU 2 to training. POLAR uses Megatron/SGLang; SkyRL uses FSDP/vLLM.

Common reduced workload:

| Setting | Value |
|---|---:|
| Training GPUs | 1 |
| Rollout/inference GPUs | 2 |
| Tensor parallel size | 1 |
| Rollout iterations | 4 |
| Rollout batch size | 4 |
| Samples per prompt | 2 |
| Steps per rollout | 2 |
| Training microbatch | 1 |
| Dynamic batching | disabled |
| Max training tokens per microbatch | 8,192 |
| Agent/inference context | 40,960 |
| Per-call completion cap | 2,048 |
| Rollout prompt/response allocation | 36,864 / 4,096 |
| POLAR async/staleness level | 2 |
| Completed acceptance fraction | 1.0 |
| Mini-SWE max turns | 40 |

Async admission and tail cancellation can change terminal trajectory counts, so this is not a fixed-denominator accuracy benchmark. Compare wall time, terminal count, calls/tokens, resolved count, and GPU utilization together.

## 2. Current suite state

The active controller at the last audit was:

```text
tmux session: miniswe_small_parity_20260721
suite stem:   miniswe_small_parity_1train2rollout_20260721T083500Z
suite dir:    /NHNHOME/home/profiled_runs/miniswe_small_parity_1train2rollout_20260721T083500Z
```

Refresh it rather than trusting this snapshot:

```bash
SUITE_DIR=/NHNHOME/home/profiled_runs/miniswe_small_parity_1train2rollout_20260721T083500Z

tmux has-session -t miniswe_small_parity_20260721
cat "${SUITE_DIR}/manifest.tsv"
tail -n 80 "${SUITE_DIR}/suite.log"
tail -n 80 "${SUITE_DIR}/polar_miniswe/sweep.log" 2>/dev/null || true
tail -n 80 "${SUITE_DIR}/polar_codex/sweep.log" 2>/dev/null || true
ps -eo pid,ppid,pgid,etime,args | \
  rg 'run_small_parity_suite|run_miniswe_full_matrix|launch_qwen3_4b_3h200' | \
  rg -v 'rg '
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
```

As of 2026-07-22:

- SkyRL workers4 completed with exit 0.
- POLAR Mini-SWE 4/2/4 completed with exit 0.
- POLAR Mini-SWE 4/4/4 completed with exit 0.
- POLAR Codex 4/2/4 was still active after about nine hours.
- POLAR Codex 4/4/4 had not started.

### Critical current Codex non-progress condition

The Codex row is alive and serving requests, but it is not converging on an accepted rollout batch. Its log repeatedly shows:

```text
Dropping trace ... total_len=~12000 > max_tokens=8192
no usable trace ... emitting dummy placeholder
Dropping Polar group ... zero trainable tokens
No progress for 60s. Queue=0, accepted=0/4
```

This is a configuration liveness bug, not useful nine-hour benchmark work. Codex traces are longer than the 8,192 training-token cap, and `POLAR_MIN_COMPLETE_ACCEPT_FRACTION=1.0` forces indefinite replenishment when all groups are rejected.

Do **not** relaunch the Codex rows unchanged. Before a new Codex characterization, make one validated change that permits nonempty trainable traces. The most direct candidate is a bounded `MAX_TOKENS_PER_GPU=16384` with `MICRO_BATCH_SIZE=1`; retain the cap and watch GPU 0 for OOM. An alternative is to shorten Codex traces enough to fit 8,192. Merely lowering worker concurrency does not solve this rejection loop. Do not publish a Codex result until the logs show accepted groups and training steps.

Stopping or modifying the current live run requires an explicit decision; this runbook does not do it automatically.

## 3. Completion semantics

The master runs all stages sequentially. It deliberately continues when a stage command exits nonzero.

- `suite.complete` only means the controller reached the end.
- Every top-level `manifest.tsv` row must have `status=complete` and `exit_code=0`.
- Each POLAR `sweep.log` must show `finished init4_run2 status=0` and `finished init4_run4 status=0`.
- A POLAR stage appears in the top-level manifest only after both of its rows return.

Never infer success from a tmux session disappearing or from `suite.complete` alone.

## 4. Repository and required assets

```text
main checkout: /NHNHOME/home/ProRL-Agent-Server
worktree:      /NHNHOME/home/ProRL-Agent-Server/.worktrees/prorl-miniswe-harness
branch:        prorl-miniswe-harness
remote branch: origin/prorl-miniswe-harness
```

Important implementation commits:

```text
b15af91a Add Mini-SWE-Agent harness
39f706c9 Complete Mini-SWE Apptainer integration
8ad3729e Fix POLAR Mini-SWE evaluation target
3f0fb92a Add small parity comparison report
ef3d5e19 Add Codex run8 comparison metrics
```

Inspect before editing:

```bash
cd /NHNHOME/home/ProRL-Agent-Server/.worktrees/prorl-miniswe-harness
git status --short --branch
git log -6 --oneline --decorate
```

Tracked implementation files include:

```text
src/polar/trajectory/agent/mini_swe_agent.py
src/polar/trajectory/evaluator/swebench_harness.py
src/polar/runtime/apptainer.py
examples/swebench_verified/mini_swe_agent_swebench.yaml
examples/swebench_verified/miniswe_reminder_agent.py
tests/agent/test_mini_swe_agent_harness.py
tests/trajectory/test_swebench_harness.py
```

`miniswe_reminder_agent.py` supplies the turn reminders used for closer SkyRL wrapper parity.

### Required ignored local assets

The exact active run also imports and executes ignored files:

```text
/NHNHOME/home/ProRL-Agent-Server/.worktrees/prorl-miniswe-harness/tmp/prorl_miniswe_small_parity_20260721/run_small_parity_suite.sh

/NHNHOME/home/ProRL-Agent-Server/.worktrees/prorl-miniswe-harness/tmp/prorl_miniswe_full_matrix_20260720T085614Z/run_miniswe_full_matrix.sh
/NHNHOME/home/ProRL-Agent-Server/.worktrees/prorl-miniswe-harness/tmp/prorl_miniswe_full_matrix_20260720T085614Z/launch_qwen3_4b_3h200.sh
/NHNHOME/home/ProRL-Agent-Server/.worktrees/prorl-miniswe-harness/tmp/prorl_miniswe_full_matrix_20260720T085614Z/polar_config.miniswe.yaml
/NHNHOME/home/ProRL-Agent-Server/.worktrees/prorl-miniswe-harness/tmp/prorl_miniswe_full_matrix_20260720T085614Z/runtime_src
```

Preflight:

```bash
WORKTREE=/NHNHOME/home/ProRL-Agent-Server/.worktrees/prorl-miniswe-harness
ASSET_DIR=${WORKTREE}/tmp/prorl_miniswe_full_matrix_20260720T085614Z
MASTER=${WORKTREE}/tmp/prorl_miniswe_small_parity_20260721/run_small_parity_suite.sh

test -x "${MASTER}"
test -x "${ASSET_DIR}/run_miniswe_full_matrix.sh"
test -x "${ASSET_DIR}/launch_qwen3_4b_3h200.sh"
test -f "${ASSET_DIR}/polar_config.miniswe.yaml"
test -f "${ASSET_DIR}/runtime_src/polar/trajectory/evaluator/swebench_harness.py"
```

Do not delete these `tmp/prorl_miniswe_*` directories during cleanup. The active master passes `POLAR_SOURCE_ROOT=<asset-dir>/runtime_src`. That snapshot adds the fallback SWE-Gym test spec/grader needed for repositories unsupported by the installed PyPI SWE-bench package. Omitting it changes evaluator behavior.

## 5. External inputs

```text
model:       /NHNHOME/home/huggingface_models/Qwen3.5-4B
dataset:     /NHNHOME/home/prorl_agent_server_env/data/swegym_train_apptainer_3h200_sandbox.jsonl
TP1 ckpt:    /NHNHOME/home/prorl_agent_server_env/checkpoints/Qwen3.5-4B_torch_dist_tp1
Codex CLI:   /NHNHOME/home/prorl_agent_server_env/agent_cli/codex_0.121.0
venv:        /NHNHOME/home/ProRL-Agent-Server/.venv
Slime:       /NHNHOME/home/ProRL-Agent-Server/slime
Megatron:    /NHNHOME/home/ProRL-Agent-Server/Megatron-LM
SkyRL:       /NHNHOME/home/SkyRL
Apptainer:   /NHNHOME/home/tools/apptainer-1.5.2/usr/bin/apptainer
wheelhouse:  /tmp/prorl-miniswe-wheelhouse
```

The dataset supplies converted Apptainer sandbox-directory paths. Do not use Docker or launch a SIF directly. Mini-SWE-Agent 2.4.2 installs inside the runtime from the offline wheelhouse mounted as `/polar/wheelhouse`, using `--no-index`.

## 6. Evaluator correctness: why the old POLAR result was 0%

The old 0%-resolved Mini-SWE characterization was invalid for accuracy comparison.

Mini-SWE edited `/polar/session/workspace`, while the generated SWE-bench/SWE-Gym script ran `cd /testbed`. Because `refresh_runtime: false`, it tested the untouched base checkout.

The committed evaluator rewrites `/testbed` in the evaluation script to the configured repository and records:

```json
"evaluated_repo_dir": "/polar/session/workspace"
```

Verify that field in every completed Mini-SWE row:

```bash
RUN_DIR=/NHNHOME/home/prorl_agent_server_runs/swegym_slime_grpo_3h200/miniswe_small_parity_1train2rollout_20260721T083500Z_polar_miniswe_init4_run2
rg -n 'evaluated_repo_dir|"resolved"' "${RUN_DIR}/rollout_results"
```

The runtime snapshot falls back to applying `test_patch` and running the listed `FAIL_TO_PASS` and `PASS_TO_PASS` pytest targets if the installed harness does not recognize the SWE-Gym repository. A zero test exit is graded resolved. Preserve this when reproducing and interpreting results.

## 7. Validate and launch

```bash
cd /NHNHOME/home/ProRL-Agent-Server/.worktrees/prorl-miniswe-harness
PYTHONPATH=src /NHNHOME/home/ProRL-Agent-Server/.venv/bin/python \
  -m pytest -q \
  tests/trajectory/test_swebench_harness.py \
  tests/agent/test_mini_swe_agent_harness.py
```

Expected at the last audit: `5 passed`.

Before launch, inspect tmux, the exact controller processes, and GPU ownership. Do not duplicate a live suite or kill another job to obtain GPUs.

```bash
tmux list-sessions 2>/dev/null || true
ps -eo pid,ppid,pgid,etime,args | \
  rg 'run_small_parity_suite|run_miniswe_full_matrix|launch_qwen3_4b_3h200' | \
  rg -v 'rg ' || true
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
```

The master includes Mini-SWE and Codex rows. Because the unchanged Codex cap is non-terminating, fix or disable Codex before a fresh full-matrix launch. Mini-SWE-only reproduction can keep the 8,192 cap.

After validating the intended master, launch with unique identifiers:

```bash
WORKTREE=/NHNHOME/home/ProRL-Agent-Server/.worktrees/prorl-miniswe-harness
MASTER=${WORKTREE}/tmp/prorl_miniswe_small_parity_20260721/run_small_parity_suite.sh
run_tag="$(date -u +%Y%m%dT%H%M%SZ)"
session_name="miniswe_small_parity_${run_tag}"
suite_stem="miniswe_small_parity_1train2rollout_${run_tag}"

tmux new-session -d -s "${session_name}" \
  "/engrid/ensh/gpubin/ctn_gcsudo env SUITE_STEM=${suite_stem} bash ${MASTER}"

printf 'session=%s\nsuite_stem=%s\nsuite_dir=/NHNHOME/home/profiled_runs/%s\n' \
  "${session_name}" "${suite_stem}" "${suite_stem}"
```

Keep `ctn_gcsudo` *inside* tmux. This form survives detachment. The driver waits for GPUs 0-2 to meet its free thresholds, so apparent idleness may be a wait. Never reuse a suite stem because the manifest is recreated.

### No Nsight

The user explicitly does not want Nsight. The master sets:

```text
NSYS_CAPTURE_RANGE=none
NSYS_EXPORT_FORMATS=none
NSYS_GPU_METRICS_DEVICES=none
NSYS_BIN=/bin/false
```

The stale log label `large-worker Nsight sweep watcher` and `nsys_out` names are harmless. Confirm `running launcher directly without Nsight`; no `.nsys-rep` is expected.

## 8. Known broken paths

Do not repeat:

1. Plain `nohup`, `setsid`, or a detached shell without tmux: controllers received SIGHUP or disappeared.
2. Plain tmux without `ctn_gcsudo`: Apptainer reports `Could not write info to setgroups: Permission denied`.
3. Direct SIF execution: unprivileged mode hits setgroups; privileged mode cannot allocate `/dev/loop0`.
4. `apptainer instance start`: unsupported here.
5. Docker: unavailable. Use direct `apptainer exec` on dataset sandbox directories.
6. Executable sessions under `/tmp`: it is `noexec`. POLAR uses the run directory; SkyRL uses `/var/tmp`.
7. Omitting venv NVIDIA libraries: SGLang cannot load `libcublas.so.12`.
8. Omitting `POLAR_SOURCE_ROOT`: the exact fallback evaluator changes.
9. Trusting template defaults: the launcher rewrites the model and the suite sets async level 2.

The launcher already encodes Apptainer flags, `NCCL_P2P_DISABLE=1`, disabled NUMA binding, B200/SGLang fixes, selected-GPU Ray visibility, shortened socket names, and temporary cleanup.

An agent-shell error `bwrap: loopback: Failed RTM_NEWADDR` is the tool sandbox, not training. Retry inspection through the approved elevated path.

## 9. Outputs and measurements

POLAR rows are under:

```text
/NHNHOME/home/prorl_agent_server_runs/swegym_slime_grpo_3h200/<suite-stem>_polar_miniswe_init4_run2
/NHNHOME/home/prorl_agent_server_runs/swegym_slime_grpo_3h200/<suite-stem>_polar_miniswe_init4_run4
/NHNHOME/home/prorl_agent_server_runs/swegym_slime_grpo_3h200/<suite-stem>_polar_codex_init4_run2
/NHNHOME/home/prorl_agent_server_runs/swegym_slime_grpo_3h200/<suite-stem>_polar_codex_init4_run4
```

Inspect `train.log`, `run_settings.txt`, `phase_events.tsv`, `rollout_results/`, and `lightweight_trace_summary/*.tsv`.

```bash
rg 'polar/reward_mean|polar/eval/resolved_rate|train_metric_utils.py:45 - perf|rollout [0-9]+:|No progress|Dropping trace' \
  "${RUN_DIR}/train.log"
```

SkyRL uses `/NHNHOME/home/skyrl_prorl_runs/<suite-stem>_skyrl_workers4/init4_run4`; its generated report is `/NHNHOME/home/profiled_runs/<suite-stem>/SKYRL_WORKERS4_REPORT.md`.

The SkyRL report prose may claim batch 8/samples 4. That generator text is stale; the captured master uses batch 4/samples 2. Its measured timing, trajectory, token, reward, and GPU fields remain usable.

| Row | Wall time | Terminal | Resolved | Status |
|---|---:|---:|---:|---|
| SkyRL Mini-SWE workers4 | 32.7 min | 40 | 12 (30.0%) | exit 0 |
| POLAR Mini-SWE 4/2/4 | 47.1 min | 29 | 14 (48.3%) | exit 0; correct repo |
| POLAR Mini-SWE 4/4/4 | 38.2 min | 39 | 14 (35.9%) | exit 0; 8 unique solved tasks; correct repo |
| POLAR Codex 4/2/4 | non-terminating | no valid final count | no valid final count | accepted 0/4 loop |

Mini-SWE 4/2/4 also had 1,217 calls, 211,349 completion tokens, 0.616 terminal trajectories/minute, and eight unique solved tasks. Use the living report for later metrics. This was not a one-hour suite: SkyRL plus Mini-SWE took about two hours, and Codex then ran indefinitely.

## 10. Safe stop

Stopping destroys the current row; do it only with explicit authorization. Resolve the current process group first:

```bash
ps -eo pid,ppid,pgid,etime,args | \
  rg 'run_small_parity_suite|run_miniswe_full_matrix|launch_qwen3_4b_3h200' | \
  rg -v 'rg '
```

Never copy a stale PGID. After verification:

```bash
kill -TERM -- -<verified-pgid>
# Only if root-owned members of that exact group survive:
/engrid/ensh/gpubin/ctn_gcsudo kill -KILL -- -<verified-pgid>
```

Verify GPUs 0-2 are free before relaunch. Never use broad `pkill`, kill all Python/Ray/Apptainer processes, or recursively clean `/NHNHOME/home` or the worktree.

## 11. Results and next-agent checklist

Update `POLAR_MINISWE_SMALL_PARITY_SKYRL4_CODEX_COMPARISON.md` only after a row exits and artifacts agree. Record timestamps, wall/exit status, terminal/nonempty traces, resolved/unique solved, calls/tokens and throughput, reward, training timings, GPU idle, evaluated repo, paths, and caveats. A Codex row stuck at `accepted=0/4` is a failed configuration, not a fair slow result.

```bash
cd /NHNHOME/home/ProRL-Agent-Server/.worktrees/prorl-miniswe-harness
git diff --check
git status --short
```

- Refresh tmux, processes, manifests, sweep logs, and GPUs.
- Do not duplicate the live suite.
- Treat Codex with an 8,192 cap as non-terminating.
- Preserve both ignored `tmp/prorl_miniswe_*` asset directories.
- Use one training GPU and two rollout GPUs.
- Use tmux with `ctn_gcsudo` inside it; keep Nsight disabled.
- Verify Mini-SWE evaluated `/polar/session/workspace`.
- Exclude the old POLAR 0%-resolved result.
- Validate every row, update the living report, and push relevant branch changes.
