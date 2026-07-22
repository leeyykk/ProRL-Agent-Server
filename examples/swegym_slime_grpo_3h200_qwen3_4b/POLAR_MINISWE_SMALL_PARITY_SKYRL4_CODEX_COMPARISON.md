# Small-Workload SkyRL, POLAR Mini-SWE, and POLAR Codex Comparison

Living report updated from measured artifacts on 2026-07-22. All five rows in
the table completed, but both Codex rows are diagnostic runs rather than valid
quality comparisons. No Nsight data is used.

## Current result

SkyRL Mini-SWE workers4 completed in 32.7 minutes and resolved 12 of 40
trajectories (30.0%). Corrected POLAR Mini-SWE `init4/run2/eval4` completed in
47.1 minutes and resolved 14 of 29 terminal trajectories (48.3%), covering 8
unique solved task groups. Matched POLAR Mini-SWE `init4/run4/eval4`
completed in 38.2 minutes and resolved 14 of 39 terminal trajectories (35.9%),
also covering 8 unique solved task groups. All 68 persisted POLAR Mini-SWE
terminal results have status `COMPLETED`.

These are the first valid POLAR Mini-SWE quality measurements in this
experiment lineage. Every resolved artifact records `evaluated_repo_dir` as
`/polar/session/workspace`, confirming that the evaluator tested the repository
the agent edited. The old zero-resolved POLAR runs evaluated untouched
`/testbed` and must not be used as accuracy measurements.

The corrected-cap Codex suite completed both worker points in 23.7 minutes
total and trained for three iterations per point without OOM. The run2 point
persisted 58 terminal sessions and the run4 point persisted 60; neither
resolved a task. Those zeroes are not usable Codex accuracy measurements:
57/58 run2 sessions and 57/60 run4 sessions exited without any tracked patch.
The four sessions that did produce a patch were then evaluated through a
refreshed, read-only `/testbed`, and every patch failed to apply. A corrected
retry is required before comparing Codex quality with Mini-SWE.

## Main comparison

| System | Worker point | Status | Wall min | Terminal trajectories | Resolved | Trajectory resolved rate | Nonzero traced sessions | Calls/nonzero trace | Completion tok/nonzero trace |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SkyRL + Mini-SWE | generation workers 4 | complete | 32.7 | 40 | 12 | 30.0% | 40 | 36.10 | 8864.6 |
| POLAR + Mini-SWE | init 4, run 2, eval 4 | complete | 47.1 | 29 | 14 | 48.3% | 32 | 38.03 | 6604.7 |
| POLAR + Mini-SWE | init 4, run 4, eval 4 | complete | 38.2 | 39 | 14 | 35.9% | 40 | 39.50 | 7477.6 |
| POLAR + Codex CLI | init 4, run 2, eval 4 | complete; invalid for quality | 11.8 | 58 | 0 | 0.0% | 58 | 5.97 | 393.6 |
| POLAR + Codex CLI | init 4, run 4, eval 4 | complete; invalid for quality | 11.8 | 60 | 0 | 0.0% | 60 | 6.25 | 448.8 |

POLAR run2 produced 211,349 completion tokens and 1,217 model calls across 32
nonzero traces. That is 0.616 terminal trajectories/minute and approximately
4,486 completion tokens/minute. Eight additional trace rows were empty
asynchronous tail entries and are excluded from per-trace averages.

POLAR run4 produced 299,105 completion tokens and 1,580 model calls across 40
nonzero traced sessions. That is 1.02 terminal trajectories/minute and about
7,839 completion tokens/minute. The direct worker-matched wall comparison is
SkyRL workers4 at 32.7 minutes versus POLAR run4 at 38.2 minutes: POLAR was 5.5
minutes, or 16.8%, slower on this small sample while processing 39 versus 40
terminal trajectories.

## POLAR Codex cap-16k diagnostic

| Worker point | Wall | Terminal sessions | LLM calls | Completion tokens | Empty tracked diff | Patch apply failures | Dropped over cap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| init4/run2/eval4 | 11m50s | 58 | 346 | 22,831 | 57 | 1 | 12 |
| init4/run4/eval4 | 11m49s | 60 | 375 | 26,927 | 57 | 3 | 12 |

Both points exited 0, checkpointed, and completed three training iterations.
Raising the trace cap from 8,192 to 16,384 therefore fixed training liveness,
but not rollout correctness. The run2 rollout steps recorded 24/16/18 sessions,
141/99/106 calls, and 9,542/6,159/7,130 completion tokens. Run4 recorded
28/16/16 sessions, 182/83/110 calls, and 13,033/4,587/9,307 completion tokens.

The dominant failure was premature Codex termination. In a matched task where
Mini-SWE ran 40 model turns and edited `moto/organizations/models.py`, Codex
stopped after 8-10 calls with an ordinary assistant sentence describing the
next inspection step but no tool call. Codex CLI treated that sentence as its
final answer, so the checkout stayed unchanged. The maintained harness now
detects an empty tracked diff and resumes the same Codex conversation; that
fix has unit coverage but has not yet produced a completed experiment.

The remaining four nonempty attempts exposed a separate evaluator mismatch.
Codex used `refresh_runtime: true` and edited `/testbed`, whereas the successful
Mini-SWE rows used `refresh_runtime: false` and evaluated
`/polar/session/workspace`. All four Codex patches failed application with a
read-only-filesystem error, so none reached a meaningful correctness test.

Codex trace construction is also different from Mini-SWE. Codex tool
observations remain `tool` messages and prefix grouping merges the session into
one training trace. Mini-SWE shell observations are `user` messages, so its
turns generally remain separate traces. For the original failed run, however,
the decisive problem was simpler: Codex's initial prompt alone was about
11,003 tokens, already above the old 8,192-token cap.

## POLAR Mini-SWE run2 detail

| Rollout batch | Completed-session resolved rate | Rollout success rate | Mean run time |
| --- | ---: | ---: | ---: |
| 1 | 12.5% | 100% | 183.3 s |
| 2 | 75.0% | 100% | 168.7 s |
| 3 | 75.0% | 100% | 168.1 s |

Final persisted artifacts, rather than a simple mean of batch rates, define the
reported 14/29 outcome. The result directory contains 29 terminal files, all
`COMPLETED`; 14 are resolved. Resolved trajectories represent 8 distinct task
groups because two samples were generated per prompt.

## POLAR Mini-SWE run4 detail

| Rollout batch | Completed-session resolved rate | Rollout success rate | Mean run time |
| --- | ---: | ---: | ---: |
| 1 | 25.0% | 100% | 257.0 s |
| 2 | 50.0% | 100% | 178.6 s |
| 3 | 37.5% | 100% | 171.1 s |

Final artifacts contain 39 terminal files, all `COMPLETED`; 14 are resolved
across 8 unique task groups. All resolved evaluations targeted
`/polar/session/workspace`.

## Training and GPU behavior

| Metric | SkyRL workers4 | POLAR run2 | POLAR run4 |
| --- | ---: | ---: | ---: |
| Steady train time/step | pending extraction | 407.2 s | 434.4 s |
| Actor train time/step | pending extraction | 292.1 s | 315.8 s |
| Reference logprob time/step | pending extraction | 70.1 s | 71.6 s |
| Policy logprob time/step | pending extraction | 44.7 s | 46.6 s |
| Actor train tokens/s | pending extraction | 1607.3 | 1688.4 |
| Train-GPU idle | 43.6% | 60.8% | 46.8% |
| Rollout-GPU idle, mean | 75.9% | 34.8% | 21.2% |

POLAR steady training values average steps 2 and 3, excluding the first step's
startup wait. GPU roles differ by framework: SkyRL used GPUs 0-1 for inference
and GPU2 for training; POLAR used GPU0 for training and GPUs 1-2 for rollout.

## Matched setup

| Axis | Value |
| --- | --- |
| Model | Qwen3.5-4B |
| Data | SWE-Gym 3h200 lineage; 16-row small training workload |
| Rollout batches | 4 |
| Prompts per batch | 4 |
| Samples per prompt | 2 |
| Training microbatch | 1 |
| Async/staleness level | 2 |
| GPU layout | 1 training GPU, 2 rollout GPUs |
| Tensor parallel size | 1 |
| Agent context | 40,960 tokens |
| Maximum completion per model call | 2,048 tokens |
| Maximum Mini-SWE turns | 40 |
| POLAR train trace cap | 8,192 for Mini-SWE; completed Codex diagnostic 16,384; next corrected Codex retry 32,768 |
| Profiling | Lightweight summaries only; Nsight disabled |

Mini-SWE uses the same remaining-turn reminder behavior in both frameworks.
The training/inference implementations still differ: SkyRL uses FSDP/vLLM,
while POLAR uses Megatron/SGLang. SkyRL exposes one generation-worker pool;
POLAR separately controls init, run, and evaluator workers.

## Codex run and retry map

- Invalid 8,192-cap Codex run2 attempt:
  `/NHNHOME/home/prorl_agent_server_runs/swegym_slime_grpo_3h200/miniswe_small_parity_1train2rollout_20260721T083500Z_polar_codex_init4_run2`
- Completed 16,384-cap diagnostic run2:
  `/NHNHOME/home/prorl_agent_server_runs/swegym_slime_grpo_3h200/miniswe_small_parity_codex_cap16k_1train2rollout_20260722T002427Z_polar_codex_cap16k_init4_run2`
- Completed 16,384-cap diagnostic run4:
  `/NHNHOME/home/prorl_agent_server_runs/swegym_slime_grpo_3h200/miniswe_small_parity_codex_cap16k_1train2rollout_20260722T002427Z_polar_codex_cap16k_init4_run4`
- Corrected retry suite (empty-diff resume, in-place evaluation, cap 32,768):
  `/NHNHOME/home/profiled_runs/miniswe_small_parity_codex_fixed_retry4_cap32k_inplace_1train2rollout_20260722T020800Z`

The corrected retry has not run successfully yet. Its first launch attempt
failed during Apptainer preflight, before GPU allocation, with
`Could not write info to setgroups: Permission denied`. No corrected-retry
quality result exists as of this update.

## Source artifacts

SkyRL workers4:

- Report: `/NHNHOME/home/profiled_runs/miniswe_small_parity_1train2rollout_20260721T083500Z/SKYRL_WORKERS4_REPORT.md`
- Manifest: `/NHNHOME/home/skyrl_prorl_runs/miniswe_small_parity_1train2rollout_20260721T083500Z_skyrl_workers4/manifest.tsv`
- Run directory: `/NHNHOME/home/skyrl_prorl_runs/miniswe_small_parity_1train2rollout_20260721T083500Z_skyrl_workers4/init4_run4`

POLAR Mini-SWE:

- Run2: `/NHNHOME/home/prorl_agent_server_runs/swegym_slime_grpo_3h200/miniswe_small_parity_1train2rollout_20260721T083500Z_polar_miniswe_init4_run2`
- Run4: `/NHNHOME/home/prorl_agent_server_runs/swegym_slime_grpo_3h200/miniswe_small_parity_1train2rollout_20260721T083500Z_polar_miniswe_init4_run4`
- Sweep log: `/NHNHOME/home/profiled_runs/miniswe_small_parity_1train2rollout_20260721T083500Z/polar_miniswe/sweep.log`
- Trace summaries: `lightweight_trace_summary/`
- GPU samples: `gpu_samples.csv`

Invalid Codex 8,192-cap attempt:

- Run: `/NHNHOME/home/prorl_agent_server_runs/swegym_slime_grpo_3h200/miniswe_small_parity_1train2rollout_20260721T083500Z_polar_codex_init4_run2`
- Sweep log: `/NHNHOME/home/profiled_runs/miniswe_small_parity_1train2rollout_20260721T083500Z/polar_codex/sweep.log`
- Signature: `total_len > max_tokens=8192`, `traces=1`, `accepted=0/4`.

Corrected Codex 16,384-cap suite:

- Suite: `/NHNHOME/home/profiled_runs/miniswe_small_parity_codex_cap16k_1train2rollout_20260722T002427Z`
- Sweep log: `polar_codex_cap16k/sweep.log`
- Run2 lightweight summaries: `/NHNHOME/home/prorl_agent_server_runs/swegym_slime_grpo_3h200/miniswe_small_parity_codex_cap16k_1train2rollout_20260722T002427Z_polar_codex_cap16k_init4_run2/lightweight_trace_summary/`
- Run4 lightweight summaries: `/NHNHOME/home/prorl_agent_server_runs/swegym_slime_grpo_3h200/miniswe_small_parity_codex_cap16k_1train2rollout_20260722T002427Z_polar_codex_cap16k_init4_run4/lightweight_trace_summary/`

Original suite controller:

- Manifest: `/NHNHOME/home/profiled_runs/miniswe_small_parity_1train2rollout_20260721T083500Z/manifest.tsv`
- Log: `/NHNHOME/home/profiled_runs/miniswe_small_parity_1train2rollout_20260721T083500Z/suite.log`

## Interpretation limits

This is a deliberately small workload, so accuracy rates have high sampling
variance. Trajectory-level resolution is not the same as unique-task pass rate
when each prompt has two samples. The table reports both raw resolved
trajectories and the unique solved-task count in the detail text. The matched
SkyRL/POLAR speed observation is one small run. The failed 8,192-cap Codex
attempt is excluded from all speed and quality conclusions. The completed
16,384-cap Codex rows measure runtime and training liveness only; their 0%
resolved rates are excluded from framework-quality conclusions because patch
generation and evaluation were both defective.
