# Small-Workload SkyRL, POLAR Mini-SWE, and POLAR Codex Comparison

Living report updated from measured artifacts on 2026-07-22. Three rows are
complete. The original Codex 4/2/4 row is an invalid non-terminating
configuration; it is not a quality or speed result. No Nsight data is used.

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

The first small-parity Codex 4/2/4 attempt used an 8,192-token per-trace
training cap. Codex serialized each session as one 11k-24k prompt/tool/response
trace, so every trace was dropped and the rollout manager remained at
`accepted=0/4`. That row is invalid and must be rerun with a higher cap.

## Main comparison

| System | Worker point | Status | Wall min | Terminal trajectories | Resolved | Trajectory resolved rate | Nonzero traced sessions | Calls/nonzero trace | Completion tok/nonzero trace |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SkyRL + Mini-SWE | generation workers 4 | complete | 32.7 | 40 | 12 | 30.0% | 40 | 36.10 | 8864.6 |
| POLAR + Mini-SWE | init 4, run 2, eval 4 | complete | 47.1 | 29 | 14 | 48.3% | 32 | 38.03 | 6604.7 |
| POLAR + Mini-SWE | init 4, run 4, eval 4 | complete | 38.2 | 39 | 14 | 35.9% | 40 | 39.50 | 7477.6 |
| POLAR + Codex CLI | init 4, run 2, eval 4 | rerunning: 16k cap | pending | pending | pending | pending | pending | pending | pending |
| POLAR + Codex CLI | init 4, run 4, eval 4 | queued after corrected run2 | pending | pending | pending | pending | pending | pending | pending |

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
| POLAR train trace cap | 8,192 for Mini-SWE; corrected Codex rerun 16,384; failed Codex attempt 8,192 |
| Profiling | Lightweight summaries only; Nsight disabled |

Mini-SWE uses the same remaining-turn reminder behavior in both frameworks.
The training/inference implementations still differ: SkyRL uses FSDP/vLLM,
while POLAR uses Megatron/SGLang. SkyRL exposes one generation-worker pool;
POLAR separately controls init, run, and evaluator workers.

## Pending update map

Fill Codex rows only after a corrected-cap row reaches exit status 0:

- Invalid 8,192-cap Codex run2 attempt:
  `/NHNHOME/home/prorl_agent_server_runs/swegym_slime_grpo_3h200/miniswe_small_parity_1train2rollout_20260721T083500Z_polar_codex_init4_run2`
- Corrected 16,384-cap run2:
  `/NHNHOME/home/prorl_agent_server_runs/swegym_slime_grpo_3h200/miniswe_small_parity_codex_cap16k_1train2rollout_20260722T002427Z_polar_codex_cap16k_init4_run2`
- Corrected 16,384-cap run4 (queued):
  `/NHNHOME/home/prorl_agent_server_runs/swegym_slime_grpo_3h200/miniswe_small_parity_codex_cap16k_1train2rollout_20260722T002427Z_polar_codex_cap16k_init4_run4`

For every completed row, record wall time, terminal status counts, resolved
trajectory count, unique solved task groups, nonzero trace count, calls,
completion tokens, steady training timing, and role-aware GPU idle fractions.

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
- Tmux: `polar_codex_cap16k_20260722T002427Z`

Original suite controller:

- Manifest: `/NHNHOME/home/profiled_runs/miniswe_small_parity_1train2rollout_20260721T083500Z/manifest.tsv`
- Log: `/NHNHOME/home/profiled_runs/miniswe_small_parity_1train2rollout_20260721T083500Z/suite.log`

## Interpretation limits

This is a deliberately small workload, so accuracy rates have high sampling
variance. Trajectory-level resolution is not the same as unique-task pass rate
when each prompt has two samples. The table reports both raw resolved
trajectories and the unique solved-task count in the detail text. The matched
SkyRL/POLAR speed observation is one small run. The failed 8,192-cap Codex
attempt is excluded from all speed and quality conclusions.
