# POLAR_MINISWE_HARNESS_CHANGE_HANDOFF

This file is a handoff for a new agent with no prior context. The current user goal is to run **ProRL with the Mini-SWE-Agent harness**, starting with a smoke test, then eventually use it for the `init8_run8` comparison against the existing SkyRL Mini-SWE baseline.

## What The User Expected

The user wanted a comparative run between SkyRL and ProRL using the same Mini-SWE-style harness. A previous run accidentally repeated **SkyRL Mini-SWE** instead of running **ProRL Mini-SWE**. The current task is to implement and verify the ProRL side.

Important: do not rerun SkyRL as the answer to this task. The target is **ProRL + Mini-SWE-Agent harness**.

## Background Files To Read

Read these first to understand the experiment and why this branch exists:

- `/NHNHOME/home/SkyRL/examples/train/mini_swe_agent/RUNBOOK_SKYRL_PRORL_COMPARISON.md`
- `/NHNHOME/home/SkyRL/examples/train/mini_swe_agent/SKYRL_WORKER_MATRIX_METRICS_REPORT_ASYNC2_1TRAIN2ROLLOUT_MB1.md`
- `/NHNHOME/home/ProRL-Agent-Server/examples/swegym_slime_grpo_3h200_qwen3_4b/RUNBOOK_SKYRL_MINISWE_PRORL_COMPARISON_HANDOFF.md`
- `/NHNHOME/home/ProRL-Agent-Server/examples/swegym_slime_grpo_3h200_qwen3_4b/SKYRL_MINISWE_PRORL_COMPARE_INIT8_RUN8_REPORT_ASYNC2_1TRAIN2ROLLOUT_MB1.md`

Useful previous diagnosis:

- The old ProRL Codex CLI baseline was not a quality baseline.
- Direct artifact inspection showed `0/328` resolved, `317` empty patches, and `11` patches that failed to apply.
- So Codex CLI had short trajectories because it was not producing usable patches, not because it was more efficient.
- The current branch is meant to test Mini-SWE-Agent inside ProRL instead.

## Branch And Worktree

The branch was created from `stable`:

- Branch: `prorl-miniswe-harness`
- Remote: `origin git@github.com:leeyykk/ProRL-Agent-Server.git`
- Worktree: `/NHNHOME/home/ProRL-Agent-Server/.worktrees/prorl-miniswe-harness`

The initial harness commit was pushed:

```bash
git log --oneline -1
# b15af91a Add Mini-SWE-Agent harness
```

Work from this branch/worktree unless the user says otherwise:

```bash
cd /NHNHOME/home/ProRL-Agent-Server/.worktrees/prorl-miniswe-harness
```

## Committed Changes

The pushed commit added a new ProRL harness:

- `src/polar/agent/harnesses/mini_swe_agent.py`
- Registered in `src/polar/agent/factory.py`
- Tests added in `tests/agent/test_mini_swe_agent_harness.py`

The harness runs Mini-SWE-Agent inside the benchmark runtime workdir. ProRL's existing evaluator then collects `git diff`, so there is no Mini-SWE-specific patch extraction path.

The harness command shape is:

```bash
mini --yolo \
  --model <litellm_model> \
  --task <instruction> \
  --cost-limit 0 \
  --output /polar/session/mini-swe-agent.json \
  --exit-immediately \
  --config default.yaml \
  --config /polar/session/mini_swe_agent_swebench.yaml \
  --model-class litellm \
  --agent-class default \
  --environment-class local
```

It prefixes local model paths with `openai/` for LiteLLM. For example:

```text
/NHNHOME/home/huggingface_models/Qwen3.5-4B
```

becomes:

```text
openai//NHNHOME/home/huggingface_models/Qwen3.5-4B
```

This mirrors the SkyRL Mini-SWE behavior.

Unit tests passed:

```bash
PYTHONPATH=src pytest -q tests/agent/test_mini_swe_agent_harness.py
# 3 passed
```

Compile check passed:

```bash
python3 -m py_compile src/polar/agent/harnesses/mini_swe_agent.py src/polar/agent/factory.py
```

## Uncommitted Experimental Changes

There are uncommitted edits in the worktree:

- `examples/swebench_verified/dataset.py`
- `examples/swebench_verified/submit_swebench_tasks.py`
- `examples/swebench_verified/mini_swe_agent_swebench.yaml`

These were started to support `mini_swe_agent` in the SWE-bench Verified submitter. However, this server does not have Docker available, so the current smoke path pivoted to SWE-Gym with Apptainer.

Do not assume these uncommitted files are final. Inspect them before committing.

Current status command to inspect:

```bash
git status --short --branch
```

## Server Constraints

The command sandbox often fails on this host with:

```text
bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted
```

When that happens, rerun necessary commands with escalation.

Docker is not installed:

```text
docker: command not found
```

Use the Apptainer-based SWE-Gym runtime instead.

Known Apptainer binary:

```text
/NHNHOME/home/tools/apptainer-1.5.2/usr/bin/apptainer
```

Known SWE-Gym dataset row source:

```text
/NHNHOME/home/prorl_agent_server_env/data/swegym_train_apptainer_3h200_sandbox.jsonl
```

Known model path:

```text
/NHNHOME/home/huggingface_models/Qwen3.5-4B
```

## Smoke Test Attempt

A smoke directory was created:

```text
/NHNHOME/home/ProRL-Agent-Server/.worktrees/prorl-miniswe-harness/tmp/miniswe_smoke_20260720T045719Z
```

Important files:

- `topology.yaml`
- `request.json`
- `README.txt`

The smoke target was the first SWE-Gym row:

- Instance: `getmoto__moto-7365`
- Runtime SIF:
  `/NHNHOME/home/prorl_agent_server_env/apptainer_sifs/swegym_b200/getmoto__moto-7365.sif`
- Problem: Moto DynamoDB `update_item` float arithmetic bug

The smoke request used:

- `num_samples: 1`
- `timeout_seconds: 2400`
- runtime backend: `apptainer`
- runtime workdir: `/polar/session/workspace`
- agent harness: `mini_swe_agent`
- evaluator: `swebench_harness`
- builder: `prefix_merging`

## SGLang / Polar Services

The base Python was missing dependencies, but the main ProRL virtualenv has what is needed:

```text
/NHNHOME/home/ProRL-Agent-Server/.venv
```

Use that venv, but force branch code via `PYTHONPATH=src`.

SGLang initially failed because it could not find `libcublas.so.12`:

```text
ImportError: libcublas.so.12: cannot open shared object file
```

The fix was to set `LD_LIBRARY_PATH` using NVIDIA wheel libraries from the ProRL venv.

Relevant library dirs:

```text
/NHNHOME/home/ProRL-Agent-Server/.venv/lib/python3.12/site-packages/torch/lib
/NHNHOME/home/ProRL-Agent-Server/.venv/lib/python3.12/site-packages/nvidia/nccl/lib
/NHNHOME/home/ProRL-Agent-Server/.venv/lib/python3.12/site-packages/nvidia/cublas/lib
/NHNHOME/home/ProRL-Agent-Server/.venv/lib/python3.12/site-packages/nvidia/nvshmem/lib
/NHNHOME/home/ProRL-Agent-Server/.venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib
/NHNHOME/home/ProRL-Agent-Server/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib
/NHNHOME/home/ProRL-Agent-Server/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib
/NHNHOME/home/ProRL-Agent-Server/.venv/lib/python3.12/site-packages/nvidia/cusparselt/lib
/NHNHOME/home/ProRL-Agent-Server/.venv/lib/python3.12/site-packages/nvidia/cu13/lib
```

The working SGLang command used GPU 0 and served:

```text
http://127.0.0.1:58000
```

SGLang successfully reached:

```text
The server is fired up and ready to roll!
```

Polar services were then started:

- Rollout: `http://127.0.0.1:58080`
- Gateway: `http://127.0.0.1:58100`

The gateway initially logged one registration failure because it started before rollout accepted connections, but the rollout later saw it as healthy:

```json
{
  "health": {
    "nodes": 1,
    "status": "ok"
  }
}
```

## Exact Failure Point

The smoke task was submitted and failed before reaching Mini-SWE-Agent.

Task:

```text
prorl-miniswe-smoke-getmoto__moto-7365-1784523439
```

Session:

```text
sk-polar-ce3ad153-c377-4bab-a5eb-459420b56736
```

Result path:

```text
/NHNHOME/home/ProRL-Agent-Server/.worktrees/prorl-miniswe-harness/tmp/miniswe_smoke_20260720T045719Z/rollout_results/task_prorl-miniswe-smoke-getmoto__moto-7365-1784523439/ses_sk-polar-ce3ad153-c377-4bab-a5eb-459420b56736.json
```

Failure:

```text
runtime initialization failed:
/NHNHOME/home/tools/apptainer-1.5.2/usr/bin/apptainer instance start failed with exit code 255
```

Gateway log showed:

```text
Initialization failed for session sk-polar-ce3ad153-c377-4bab-a5eb-459420b56736
RuntimeError: /NHNHOME/home/tools/apptainer-1.5.2/usr/bin/apptainer instance start failed with exit code 255
```

This means:

- SGLang was working.
- Polar rollout was working.
- Polar gateway was working.
- The Mini-SWE harness was not reached yet.
- The next blocker is Apptainer runtime startup.

## What To Do Next

First, reproduce the Apptainer startup failure directly.

Use the same SIF:

```bash
/NHNHOME/home/tools/apptainer-1.5.2/usr/bin/apptainer exec \
  /NHNHOME/home/prorl_agent_server_env/apptainer_sifs/swegym_b200/getmoto__moto-7365.sif \
  true
```

Then reproduce closer to Polar's `ApptainerRuntime.start()` logic:

```bash
mkdir -p /tmp/prorl-miniswe-apptainer-smoke/session/overlay

/NHNHOME/home/tools/apptainer-1.5.2/usr/bin/apptainer instance start \
  --overlay /tmp/prorl-miniswe-apptainer-smoke/session/overlay \
  --bind /tmp/prorl-miniswe-apptainer-smoke/session:/polar/session \
  /NHNHOME/home/prorl_agent_server_env/apptainer_sifs/swegym_b200/getmoto__moto-7365.sif \
  polar-manual-miniswe-smoke
```

If it starts, stop it:

```bash
/NHNHOME/home/tools/apptainer-1.5.2/usr/bin/apptainer instance stop polar-manual-miniswe-smoke
```

If the manual command fails, inspect the actual stderr. The current Polar runtime only reports the exit code, not the stderr, so manual reproduction is the fastest way to see the real Apptainer issue.

Likely things to check:

- Whether the SIF works at all on this server.
- Whether Apptainer needs a different writable overlay form.
- Whether the host disallows instance mode but allows plain `apptainer exec`.
- Whether `--overlay <directory>` is invalid for this build/configuration.
- Whether the prebuilt sandbox path should be used instead of the SIF path.

If Apptainer needs different startup flags, patch:

```text
src/polar/runtime/apptainer.py
```

or adjust the smoke request/runtime configuration cleanly.

## Success Criteria

Continue until the smoke test succeeds.

For this smoke, success should mean:

1. Apptainer runtime starts.
2. Runtime prepare completes.
3. Mini-SWE-Agent runs inside `/polar/session/workspace`.
4. ProRL builder/evaluator run.
5. Session reaches `COMPLETED`.
6. Ideally the result contains a non-empty patch and evaluator output.

If the model fails to solve the task but the harness produces a valid trajectory, patch, and evaluator result, that is still useful harness smoke progress. But if the patch is empty, inspect Mini-SWE logs before calling it done.

Mini-SWE log should be under the session logs path, typically:

```text
/polar/session/logs/agent/mini-swe-agent.txt
```

The host-side saved result should be under the smoke dir's `rollout_results`.

## After Smoke Succeeds

After a successful smoke:

1. Commit the finalized harness and runtime/config changes.
2. Push branch `prorl-miniswe-harness`.
3. Prepare the actual ProRL Mini-SWE `init8_run8` comparison run.
4. Generate a report comparable to:
   `/NHNHOME/home/ProRL-Agent-Server/examples/swegym_slime_grpo_3h200_qwen3_4b/WORKER_MATRIX_METRICS_REPORT_ASYNC2_1TRAIN2ROLLOUT_MB1.md`

Do not proceed to a large worker sweep until the one-task smoke is clean.

## Update From Apptainer Debugging

After the first failed smoke, we debugged the Apptainer failure directly.

Direct unprivileged SIF exec failed with the exact cluster namespace error:

```bash
/NHNHOME/home/tools/apptainer-1.5.2/usr/bin/apptainer exec \
  /NHNHOME/home/prorl_agent_server_env/apptainer_sifs/swegym_b200/getmoto__moto-7365.sif \
  true
```

stderr:

```text
ERROR  : Could not write info to setgroups: Permission denied
ERROR  : Error while waiting event for user namespace mappings: no event received
```

Then `ctn_gcsudo` with the SIF got past that but failed because SIF mounting needs loop devices:

```text
failed to attach loop device: could not open /dev/loop0: operation not permitted
```

Conclusion: on this server, for this smoke path, use the converted sandbox directory, not the `.sif`:

```text
/NHNHOME/home/prorl_agent_server_env/apptainer_sandboxes/swegym_b200/getmoto__moto-7365
```

Manual sandbox exec under `ctn_gcsudo` works when using the documented runbook flags:

```bash
/engrid/ensh/gpubin/ctn_gcsudo \
  /NHNHOME/home/tools/apptainer-1.5.2/usr/bin/apptainer exec \
  --no-privs --no-mount bind-paths \
  /NHNHOME/home/prorl_agent_server_env/apptainer_sandboxes/swegym_b200/getmoto__moto-7365 \
  /bin/true
```

This returned exit code 0.

Running the same sandbox without those flags fails with capability/localtime errors:

```text
WARNING: could not mount /etc/localtime: not a directory
ERROR  : Requesting capability set ... while permitted capability set is ...
```

So the correct recipe is:

```bash
/engrid/ensh/gpubin/ctn_gcsudo bash -lc '
  export POLAR_APPTAINER_BIN=/NHNHOME/home/tools/apptainer-1.5.2/usr/bin/apptainer
  export POLAR_APPTAINER_EXEC_ARGS="--no-privs --no-mount bind-paths"
  ... run Polar gateway/rollout/submit here ...
'
```

## Branch Runtime Fix Applied

The Mini-SWE branch originally had an older `src/polar/runtime/apptainer.py` that used:

```text
apptainer instance start ...
apptainer exec instance://...
```

That does not work on this cluster.

The newer direct-exec implementation from the main checkout was copied into the branch worktree:

```text
/NHNHOME/home/ProRL-Agent-Server/.worktrees/prorl-miniswe-harness/src/polar/runtime/apptainer.py
```

It now:

- does not call `apptainer instance start`
- creates the host session/artifacts dirs in `start()`
- runs every command with direct `apptainer exec`
- honors `POLAR_APPTAINER_EXEC_ARGS`
- binds the host session dir to `/polar/session`
- supports extra `kwargs.volumes`

Verification after this change:

```bash
python3 -m py_compile src/polar/runtime/apptainer.py src/polar/agent/harnesses/mini_swe_agent.py src/polar/agent/factory.py
# passed

PYTHONPATH=src /NHNHOME/home/ProRL-Agent-Server/.venv/bin/pytest -q tests/agent/test_mini_swe_agent_harness.py
# 3 passed in 0.15s
```

Important: this runtime change is currently uncommitted in the branch worktree unless a later agent commits it.

## Second Smoke Retry Created

A second isolated smoke directory was created after the Apptainer findings:

```text
/NHNHOME/home/ProRL-Agent-Server/.worktrees/prorl-miniswe-harness/tmp/miniswe_smoke2_20260720T052649Z
```

Files:

```text
/NHNHOME/home/ProRL-Agent-Server/.worktrees/prorl-miniswe-harness/tmp/miniswe_smoke2_20260720T052649Z/topology.yaml
/NHNHOME/home/ProRL-Agent-Server/.worktrees/prorl-miniswe-harness/tmp/miniswe_smoke2_20260720T052649Z/request.json
```

Differences from the first smoke:

- Uses the sandbox runtime image:
  `/NHNHOME/home/prorl_agent_server_env/apptainer_sandboxes/swegym_b200/getmoto__moto-7365`
- Uses rollout port `58081`
- Uses gateway port `58101`
- Reuses SGLang at `http://127.0.0.1:58000`
- Should be run under `ctn_gcsudo` with `POLAR_APPTAINER_EXEC_ARGS="--no-privs --no-mount bind-paths"`

Fresh services were started under `ctn_gcsudo`:

- rollout session id in Codex tools: `95944`
- gateway session id in Codex tools: `55650`

Commands used:

```bash
/engrid/ensh/gpubin/ctn_gcsudo bash -lc 'cd /NHNHOME/home/ProRL-Agent-Server/.worktrees/prorl-miniswe-harness; export PYTHONPATH=src; export POLAR_APPTAINER_BIN=/NHNHOME/home/tools/apptainer-1.5.2/usr/bin/apptainer; export POLAR_APPTAINER_EXEC_ARGS="--no-privs --no-mount bind-paths"; /NHNHOME/home/ProRL-Agent-Server/.venv/bin/python -m polar.cli serve_rollout -c /NHNHOME/home/ProRL-Agent-Server/.worktrees/prorl-miniswe-harness/tmp/miniswe_smoke2_20260720T052649Z/topology.yaml'
```

```bash
/engrid/ensh/gpubin/ctn_gcsudo bash -lc 'cd /NHNHOME/home/ProRL-Agent-Server/.worktrees/prorl-miniswe-harness; export PYTHONPATH=src; export POLAR_APPTAINER_BIN=/NHNHOME/home/tools/apptainer-1.5.2/usr/bin/apptainer; export POLAR_APPTAINER_EXEC_ARGS="--no-privs --no-mount bind-paths"; /NHNHOME/home/ProRL-Agent-Server/.venv/bin/python -m polar.cli serve_gateway -c /NHNHOME/home/ProRL-Agent-Server/.worktrees/prorl-miniswe-harness/tmp/miniswe_smoke2_20260720T052649Z/topology.yaml --node-id local-miniswe-smoke2'
```

The gateway initially logged the same harmless registration race because it started before rollout was fully accepting connections. Check status before submit with:

```bash
/engrid/ensh/gpubin/ctn_gcsudo bash -lc 'cd /NHNHOME/home/ProRL-Agent-Server/.worktrees/prorl-miniswe-harness; export PYTHONPATH=src; /NHNHOME/home/ProRL-Agent-Server/.venv/bin/python -m polar.cli status -c /NHNHOME/home/ProRL-Agent-Server/.worktrees/prorl-miniswe-harness/tmp/miniswe_smoke2_20260720T052649Z/topology.yaml --json'
```

A status command was launched and may have still been running when this update was written:

- status tool session id: `51068`

If status is healthy, submit the second smoke with:

```bash
/engrid/ensh/gpubin/ctn_gcsudo bash -lc 'cd /NHNHOME/home/ProRL-Agent-Server/.worktrees/prorl-miniswe-harness; export PYTHONPATH=src; export POLAR_APPTAINER_BIN=/NHNHOME/home/tools/apptainer-1.5.2/usr/bin/apptainer; export POLAR_APPTAINER_EXEC_ARGS="--no-privs --no-mount bind-paths"; /NHNHOME/home/ProRL-Agent-Server/.venv/bin/python -m polar.cli submit /NHNHOME/home/ProRL-Agent-Server/.worktrees/prorl-miniswe-harness/tmp/miniswe_smoke2_20260720T052649Z/request.json -c /NHNHOME/home/ProRL-Agent-Server/.worktrees/prorl-miniswe-harness/tmp/miniswe_smoke2_20260720T052649Z/topology.yaml --json --poll-interval 5'
```

## Updated Next Steps

1. First check whether the second retry services are still alive and healthy.
2. If healthy, submit `miniswe_smoke2_20260720T052649Z/request.json` using the command above.
3. If init now passes, watch for the next likely blocker: runtime prepare may fail if `pip install --target /polar/session/miniswe-py mini-swe-agent==2.4.2` cannot access packages from inside the container.
4. If prepare passes, inspect whether Mini-SWE-Agent reaches SGLang through `OPENAI_BASE_URL` and produces a patch.
5. If Mini-SWE runs but patch is empty, inspect the agent log under the saved rollout session artifacts and the in-runtime path `/polar/session/logs/agent/mini-swe-agent.txt`.
6. Once smoke completes cleanly, commit at least:
   - `src/polar/runtime/apptainer.py`
   - `src/polar/agent/harnesses/mini_swe_agent.py`
   - `src/polar/agent/factory.py`
   - `tests/agent/test_mini_swe_agent_harness.py`
   - any finalized Mini-SWE config/submitter files that are actually needed.



## Completion Update — 2026-07-20

The integration described above was completed, smoke-tested, committed, and pushed.

Branch and commits:

- Branch: `prorl-miniswe-harness`
- Initial harness commit: `b15af91a Add Mini-SWE-Agent harness`
- Final integration commit: `39f706c9 Complete Mini-SWE Apptainer integration`
- Remote branch is synchronized at `39f706c9`.

Final implementation includes:

- Mini-SWE-Agent 2.4.2 installation and task construction in
  `examples/swebench_verified/submit_swebench_tasks.py`.
- The checked-in Mini-SWE SWE-bench prompt/config at
  `examples/swebench_verified/mini_swe_agent_swebench.yaml`.
- Mini-SWE factory registration and a `pipefail`-safe harness command so CLI
  failures are not hidden by `tee`.
- Direct `apptainer exec` execution instead of unsupported instance mode.
- `POLAR_APPTAINER_EXEC_ARGS` support for the required cluster flags.
- Writable Apptainer workspace evaluation with `refresh_runtime: false`, which
  avoids trying to apply generated patches to the read-only `/testbed` mount.
- Gateway session roots configurable with `gateway.nodes[].session_base_dir` or
  `POLAR_SESSION_BASE_DIR`, required because this host mounts `/tmp` as `noexec`.
- Mini-SWE launch and `noexec` guidance in the SWE-bench example README.

Smoke evidence:

- SGLang Qwen3.5-4B ran successfully on GPU 2 with a 32768-token context after
  applying the B200 compatibility environment from the profiling runbook.
- `prorl-miniswe-smoke13-workspace-getmoto__moto-7365` completed 34 records and
  generated a real patch to `moto/dynamodb/models/dynamo_type.py`. This run
  exposed the old read-only evaluator target.
- After changing Apptainer evaluation to the writable session workspace,
  `prorl-miniswe-smoke14-eval-getmoto__moto-7365` completed 40 records with no
  session, trajectory, patch-application, or evaluator error. That stochastic
  trajectory produced no patch, so its report correctly recorded
  `empty_generation: true`.
- Together these runs prove Mini-SWE trajectory generation, patch production,
  corrected workspace evaluation, and POLAR result persistence end to end.

Validation:

- Focused topology and Mini-SWE tests: `8 passed`.
- Full repository suite: `66 passed`; four unrelated failures remained (three
  async tests lacked `pytest-asyncio`, and one dashboard route expectation did
  not match the current API).
- Python compilation, YAML loading, generated request validation, and
  `git diff --check` passed.
- Temporary SGLang, rollout, and gateway processes were stopped after smoke.

## Full Mini-SWE Worker Sweep Requested

The next run requested is the complete existing worker matrix:

```text
4:2 4:4 8:4 8:8 8:16 16:4 16:8 16:16 32:16 32:32
```

`POLAR_MAX_POSTRUN_WORKERS` matches init workers for every row, so the matrix
contains the explicitly requested `init8/run8/eval8` configuration. The sweep
must use the Mini-SWE harness, Apptainer sandbox flags, executable session root,
B200 SGLang compatibility environment, lightweight tracing, and no Nsight
capture. Record the exact sweep stem, PID, log, manifest, and run directories
below once launched.


### Active full-matrix launch — 2026-07-20

The clean active sweep is:

- Sweep stem: prorl_miniswe_full_matrix_async2_mb2_20260720T093119Z
- Codex managed execution session: 94080
- Wrapper PID / process group: 57690
- Sweep log:
  /NHNHOME/home/profiled_runs/prorl_miniswe_full_matrix_async2_mb2_20260720T093119Z/sweep.log
- Run directory pattern:
  /NHNHOME/home/prorl_agent_server_runs/swegym_slime_grpo_3h200/prorl_miniswe_full_matrix_async2_mb2_20260720T093119Z_init<INIT>_run<RUN>
- Runtime launch assets:
  tmp/prorl_miniswe_full_matrix_20260720T085614Z/

Matrix and fixed settings:

    CONFIGS=4:2 4:4 8:4 8:8 8:16 16:4 16:8 16:16 32:16 32:32
    POLAR_MAX_POSTRUN_WORKERS=<init workers>
    POLAR_MAX_ASYNC_LEVEL=2
    NUM_ROLLOUT=8
    ROLLOUT_BATCH_SIZE=8
    N_SAMPLES_PER_PROMPT=4
    NUM_STEPS_PER_ROLLOUT=2
    MICRO_BATCH_SIZE=2
    SGLANG_CONTEXT_LENGTH=32768
    ROLLOUT_MAX_RESPONSE_LEN=4096
    ROLLOUT_MAX_PROMPT_LEN=28672

GPU allocation is training on GPUs 0 and 1 with TP2, and SGLang rollout on
GPU 2. The sweep uses the validated Apptainer sandbox arguments, executable
per-run POLAR_SESSION_BASE_DIR, offline Mini-SWE wheelhouse, and a runtime
source snapshot combining the newer main-checkout lightweight instrumentation
with the branch Mini-SWE harness.

There is deliberately no Nsight capture or report:

    NSYS_CAPTURE_RANGE=none
    NSYS_EXPORT_FORMATS=none
    NSYS_GPU_METRICS_DEVICES=none
    NSYS_BIN=/bin/false

The inherited orchestrator still prints an old "Nsight sweep watcher" label,
but every row follows its direct-launch branch and cannot execute Nsight.
Outputs are lightweight JSONL traces, GPU samples, summaries, POLAR results,
and train/service logs only.

First-row health proof for init4/run2/eval4:

- SGLang loaded Qwen3.5-4B at 32K context and reports healthy.
- GPU 2 holds about 130 GiB for rollout; GPUs 0/1 loaded the train actors.
- POLAR reported 64 pending sessions with 2 RUNNING, 2 READY, 2 INITIALIZING,
  and 6 queued in the first admitted set.
- Repeated Mini-SWE OpenAI-compatible model calls returned HTTP 200.
- SGLang decode was active at about 111 aggregate tokens/s for two running
  Mini-SWE trajectories when this update was written.

Two startup attempts preceded the clean stem and completed no experiment row:

1. The first stopped before launch because the inherited patch step selected a
   newly created minimal worktree venv without SGLang.
2. prorl_miniswe_full_matrix_async2_mb2_20260720T085614Z reached Ray but its
   first row failed before model load because CUDA libraries were derived from
   the worktree venv path; a second row was stopped immediately. The copied
   launcher now derives Torch/NVIDIA library paths from the established main
   VIRTUAL_ENV, and the clean stem passed that failure point.

Useful live checks:

    tail -f /NHNHOME/home/profiled_runs/prorl_miniswe_full_matrix_async2_mb2_20260720T093119Z/sweep.log
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
    ps -eo pid,ppid,pgid,stat,etimes,args | grep -E "prorl_miniswe_full_matrix|train_async.py|SGLangEngine|MegatronTrainRayActor"


## Token-capped MB2 restart - 2026-07-20

The original full-matrix sweep `prorl_miniswe_full_matrix_async2_mb2_20260720T093119Z` was stopped at the user's request after about 3.5 hours. It had not completed its first `init4/run2/eval4` row. The stop targeted only process group 57690; a graceful TERM released the services and GPUs, then the few remaining launcher/Mini-SWE children in the same group were killed. All three GPUs were verified at 0 MiB before restart.

The run was already using `MICRO_BATCH_SIZE=2`. The memory pressure came from Mini-SWE traces approaching the 32768-token context, allowing a fixed two-sample microbatch to approach 65536 packed tokens. Slime exposes `--use-dynamic-batch-size` with `--max-tokens-per-gpu`, but dynamic batching explicitly ignores `--micro-batch-size`. To preserve the requested fixed MB2 setting, the replacement uses the POLAR bridge's existing per-trace limit: `MAX_TOKENS_PER_GPU=8192` with `USE_DYNAMIC_BATCH_SIZE=0`. Traces above 8192 total prompt-plus-response tokens are excluded from training rather than truncated. Therefore a fixed two-sample microbatch is bounded at 16384 tokens, four times below the prior 65536-token worst case. Rollout generation and evaluation are unchanged.

The copied matrix driver was changed from a hard-coded `MAX_TOKENS_PER_GPU=32768` launch assignment to `MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-32768}"`, allowing the wrapper's cap to reach `train_async`. Both shell scripts passed `bash -n`, and the generated run settings plus live process arguments confirm `MICRO_BATCH_SIZE=2`, `MAX_TOKENS_PER_GPU=8192`, `USE_DYNAMIC_BATCH_SIZE=0`, and `QKV_FORMAT=thd`.

Replacement sweep details:

- Sweep stem: `prorl_miniswe_full_matrix_async2_mb2_tokcap16k_20260720T131500Z`
- Managed execution session: `60427`
- Wrapper PID / process group: `376245`
- First row: `init4/run2/eval4`
- Matrix: `4:2 4:4 8:4 8:8 8:16 16:4 16:8 16:16 32:16 32:32`
- Sweep log: `/NHNHOME/home/profiled_runs/prorl_miniswe_full_matrix_async2_mb2_tokcap16k_20260720T131500Z/sweep.log`
- Run directory pattern: `/NHNHOME/home/prorl_agent_server_runs/swegym_slime_grpo_3h200/prorl_miniswe_full_matrix_async2_mb2_tokcap16k_20260720T131500Z_init<INIT>_run<RUN>`
- Launch wrapper: `tmp/prorl_miniswe_full_matrix_20260720T085614Z/launch_full_matrix_mb2_tokcap16k.sh`

Nsight remains disabled: `NSYS_CAPTURE_RANGE=none`, `NSYS_EXPORT_FORMATS=none`, `NSYS_GPU_METRICS_DEVICES=none`, and `NSYS_BIN=/bin/false`. The first row passed GPU/Ray placement checks, loaded SGLang and both Megatron actors, started the async POLAR worker, and produced repeated Mini-SWE chat-completion HTTP 200 responses.

## Overnight result and persistent continuation - 2026-07-21

The token-capped sweep completed `init4/run2/eval4` successfully at
2026-07-21T04:09:18+09:00 after about 5 hours 53 minutes. It started
`init4/run4/eval4` at 04:20:19 and reached training step 6. At 07:20:32 the
managed execution terminal disappeared and delivered SIGHUP to the process
group. All services stopped together with `Hangup` messages. There was no CUDA
OOM or Python traceback. Valid matrix progress was one of ten configurations.

Plain nohup/setsid and plain-tmux restart probes did no GPU work: outside the
long-lived cluster wrapper, Apptainer failed preflight with `Could not write
info to setgroups: Permission denied`. Their preflight-only directories must
not be counted as experiments.

The active continuation keeps `ctn_gcsudo` alive inside tmux:

- Tmux session: `prorl_miniswe_mb2_ctn_20260721`
- Tmux server PID: `1103715`
- `ctn_gcsudo`/wrapper PID: `1103716`
- Stem: `prorl_miniswe_full_matrix_async2_mb2_tokcap16k_resume_20260721T001113Z`
- Log: `/NHNHOME/home/profiled_runs/prorl_miniswe_full_matrix_async2_mb2_tokcap16k_resume_20260721T001113Z/sweep.log`
- Remaining matrix: `4:4 8:4 8:8 8:16 16:4 16:8 16:16 32:16 32:32`

The corrected continuation began at 2026-07-21T09:27:06+09:00. Its first row
passed Apptainer, GPU isolation, host RAM, and Ray startup preflights. It keeps
fixed `MICRO_BATCH_SIZE=2`, `MAX_TOKENS_PER_GPU=8192`,
`USE_DYNAMIC_BATCH_SIZE=0`, all prior Mini-SWE settings, and no Nsight capture.
Check it with `tmux has-session -t prorl_miniswe_mb2_ctn_20260721` and the log
above.
