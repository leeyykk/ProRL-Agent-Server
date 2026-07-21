# Small-Workload SkyRL, POLAR Mini-SWE, and POLAR Codex Comparison

Living report generated from measured artifacts on 2026-07-21. The suite is
still running. Completed rows contain measured values; future rows are marked
`pending` and should be filled from their final artifacts rather than live
partial metrics. No Nsight data is used.

## Current result

SkyRL Mini-SWE workers4 completed in 32.7 minutes and resolved 12 of 40
trajectories (30.0%). Corrected POLAR Mini-SWE `init4/run2/eval4` completed in
47.1 minutes and resolved 14 of 29 terminal trajectories (48.3%), covering 8
unique solved task groups. All 29 POLAR terminal results have status
`COMPLETED`.

This is the first valid POLAR Mini-SWE quality measurement in this experiment
lineage. Every resolved artifact records `evaluated_repo_dir` as
`/polar/session/workspace`, confirming that the evaluator tested the repository
the agent edited. The old zero-resolved POLAR runs evaluated untouched
`/testbed` and must not be used as accuracy measurements.

## Main comparison

| System | Worker point | Status | Wall min | Terminal trajectories | Resolved | Trajectory resolved rate | Nonzero traced sessions | Calls/nonzero trace | Completion tok/nonzero trace |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SkyRL + Mini-SWE | generation workers 4 | complete | 32.7 | 40 | 12 | 30.0% | 40 | 36.10 | 8864.6 |
| POLAR + Mini-SWE | init 4, run 2, eval 4 | complete | 47.1 | 29 | 14 | 48.3% | 32 | 38.03 | 6604.7 |
| POLAR + Mini-SWE | init 4, run 4, eval 4 | running | pending | pending | pending | pending | pending | pending | pending |
| POLAR + Codex CLI | init 4, run 2, eval 4 | pending | pending | pending | pending | pending | pending | pending | pending |
| POLAR + Codex CLI | init 4, run 4, eval 4 | pending | pending | pending | pending | pending | pending | pending | pending |

POLAR run2 produced 211,349 completion tokens and 1,217 model calls across 32
nonzero traces. That is 0.616 terminal trajectories/minute and approximately
4,486 completion tokens/minute. Eight additional trace rows were empty
asynchronous tail entries and are excluded from per-trace averages.

The direct framework-speed comparison is SkyRL workers4 versus POLAR run4,
which is still running. POLAR run2 intentionally has half as many run workers,
so its 47.1-minute wall time should not be treated as the matched speed result.

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

## Training and GPU behavior

| Metric | SkyRL Mini-SWE workers4 | POLAR Mini-SWE init4/run2 |
| --- | ---: | ---: |
| Steady train time/step | pending extraction | 407.2 s |
| Actor train time/step | pending extraction | 292.1 s |
| Reference logprob time/step | pending extraction | 70.1 s |
| Policy logprob time/step | pending extraction | 44.7 s |
| Actor train tokens/s | pending extraction | 1607.3 |
| Train-GPU idle | 43.6% | 60.8% |
| Rollout-GPU idle, mean | 75.9% | 34.8% |

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
| POLAR train trace cap | 8,192 tokens |
| Profiling | Lightweight summaries only; Nsight disabled |

Mini-SWE uses the same remaining-turn reminder behavior in both frameworks.
The training/inference implementations still differ: SkyRL uses FSDP/vLLM,
while POLAR uses Megatron/SGLang. SkyRL exposes one generation-worker pool;
POLAR separately controls init, run, and evaluator workers.

## Pending update map

Fill the remaining rows from these paths after each row reaches exit status 0:

- POLAR Mini-SWE run4:
  `/NHNHOME/home/prorl_agent_server_runs/swegym_slime_grpo_3h200/miniswe_small_parity_1train2rollout_20260721T083500Z_polar_miniswe_init4_run4`
- POLAR Codex run2:
  `/NHNHOME/home/prorl_agent_server_runs/swegym_slime_grpo_3h200/miniswe_small_parity_1train2rollout_20260721T083500Z_polar_codex_init4_run2`
- POLAR Codex run4:
  `/NHNHOME/home/prorl_agent_server_runs/swegym_slime_grpo_3h200/miniswe_small_parity_1train2rollout_20260721T083500Z_polar_codex_init4_run4`

For every completed row, record wall time, terminal status counts, resolved
trajectory count, unique solved task groups, nonzero trace count, calls,
completion tokens, steady training timing, and role-aware GPU idle fractions.

## Source artifacts

SkyRL workers4:

- Report: `/NHNHOME/home/profiled_runs/miniswe_small_parity_1train2rollout_20260721T083500Z/SKYRL_WORKERS4_REPORT.md`
- Manifest: `/NHNHOME/home/skyrl_prorl_runs/miniswe_small_parity_1train2rollout_20260721T083500Z_skyrl_workers4/manifest.tsv`
- Run directory: `/NHNHOME/home/skyrl_prorl_runs/miniswe_small_parity_1train2rollout_20260721T083500Z_skyrl_workers4/init4_run4`

POLAR Mini-SWE run2:

- Run directory: `/NHNHOME/home/prorl_agent_server_runs/swegym_slime_grpo_3h200/miniswe_small_parity_1train2rollout_20260721T083500Z_polar_miniswe_init4_run2`
- Sweep log: `/NHNHOME/home/profiled_runs/miniswe_small_parity_1train2rollout_20260721T083500Z/polar_miniswe/sweep.log`
- Trace summaries: `lightweight_trace_summary/`
- GPU idle: `gpu_idle_summary.tsv`

Suite controller:

- Manifest: `/NHNHOME/home/profiled_runs/miniswe_small_parity_1train2rollout_20260721T083500Z/manifest.tsv`
- Log: `/NHNHOME/home/profiled_runs/miniswe_small_parity_1train2rollout_20260721T083500Z/suite.log`

## Interpretation limits

This is a deliberately small workload, so accuracy rates have high sampling
variance. Trajectory-level resolution is not the same as unique-task pass rate
when each prompt has two samples. The table reports both raw resolved
trajectories and the unique solved-task count in the detail text. Framework
speed conclusions should wait for the matched SkyRL workers4 versus POLAR
run4 row.
