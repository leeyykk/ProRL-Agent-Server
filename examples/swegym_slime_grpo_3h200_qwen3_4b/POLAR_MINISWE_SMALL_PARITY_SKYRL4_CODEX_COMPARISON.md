# Small-Workload SkyRL, POLAR Mini-SWE, and POLAR Codex Comparison

Living report updated from measured artifacts on 2026-07-23. All five matched
rows are complete. The original Codex rows used a broken model/tool path and
are superseded by the final tool-fixed rows below. No Nsight data is used.

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

The final Codex rerun additionally repaired the advertised patch tool, added
two bounded review passes, refreshed the runtime, and evaluated a writable
copy at `/polar/session/workspace`. Both worker points completed all 40
terminal sessions with no patch-application or evaluator errors.
`init4/run2/eval4` resolved 6 trajectories across 5 unique task groups;
`init4/run4/eval4` resolved 8 across 6 unique groups. These replace the
earlier zero-resolved Codex rows as the fair quality comparison.

## Main comparison

| System | Worker point | Status | Wall min | Terminal trajectories | Resolved | Trajectory resolved rate | Nonzero traced sessions | Calls/nonzero trace | Completion tok/nonzero trace |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SkyRL + Mini-SWE | generation workers 4 | complete | 32.7 | 40 | 12 | 30.0% | 40 | 36.10 | 8864.6 |
| POLAR + Mini-SWE | init 4, run 2, eval 4 | complete | 47.1 | 29 | 14 | 48.3% | 32 | 38.03 | 6604.7 |
| POLAR + Mini-SWE | init 4, run 4, eval 4 | complete | 38.2 | 39 | 14 | 35.9% | 40 | 39.50 | 7477.6 |
| POLAR + Codex CLI | init 4, run 2, eval 4 | complete | 54.9 | 40 | 6 | 15.0% | 40 | 52.00 | 9797.5 |
| POLAR + Codex CLI | init 4, run 4, eval 4 | complete | 54.0 | 40 | 8 | 20.0% | 40 | 52.90 | 10498.2 |

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

Rollout batches 1-3 had completed-session resolved rates of 12.5%, 75.0%,
and 75.0%, respectively. Rollout success was 100% in every batch, with mean
run times of 183.3 s, 168.7 s, and 168.1 s.

Final persisted artifacts, rather than a simple mean of batch rates, define the
reported 14/29 outcome. The result directory contains 29 terminal files, all
`COMPLETED`; 14 are resolved. Resolved trajectories represent 8 distinct task
groups because two samples were generated per prompt.

## POLAR Mini-SWE run4 detail

Rollout batches 1-3 had completed-session resolved rates of 25.0%, 50.0%,
and 37.5%, respectively. Rollout success was 100% in every batch, with mean
run times of 257.0 s, 178.6 s, and 171.1 s.

Final artifacts contain 39 terminal files, all `COMPLETED`; 14 are resolved
across 8 unique task groups. All resolved evaluations targeted
`/polar/session/workspace`.

## Training and GPU behavior

| Metric | SkyRL Mini-SWE workers4 | POLAR Mini-SWE run2 | POLAR Mini-SWE run4 | POLAR Codex run2 | POLAR Codex run4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Steady train-phase time/update | not collected | 407.2 s | 434.4 s | 533.0 s | 501.8 s |
| Actor optimization time/update | not collected | 292.1 s | 315.8 s | 410.2 s | 386.1 s |
| Reference logprob time/update | not collected | 70.1 s | 71.6 s | not retained | not retained |
| Policy logprob time/update | not collected | 44.7 s | 46.6 s | not retained | not retained |
| Actor train tokens/s | not collected | 1607.3 | 1688.4 | 1280.6 | 1296.1 |
| Train-GPU idle | 43.6% | 60.8% | 46.8% | 26.8% | 24.6% |
| Rollout-GPU idle, mean | 75.9% | 34.8% | 21.2% | 74.8% | 76.7% |

POLAR train-phase values average outer updates 2 and 3, excluding the first
update. The `train` timer begins after rollout data is available and comprises
reference log probabilities, policy log probabilities, actor optimization, and
small bookkeeping overhead. Asynchronous rollout generation is not included in
this timer and can overlap training; only wall time is end-to-end. SkyRL did not
retain equivalent phase timers, so those cells say `not collected` rather than
`pending extraction`. GPU roles differ by framework: SkyRL used GPUs 0-1 for inference
and GPU2 for training; POLAR used GPU0 for training and GPUs 1-2 for rollout.

## Execution and patch detail

| Metric | SkyRL Mini-SWE workers4 | POLAR Mini-SWE run2 | POLAR Mini-SWE run4 | POLAR Codex run2 | POLAR Codex run4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Wall time | 32.7 min | 47.1 min | 38.2 min | 54.9 min | 54.0 min |
| Terminal status | 40 terminal | 29 COMPLETED | 39 COMPLETED | 40 COMPLETED | 40 COMPLETED |
| Resolved / unique solved groups | 12 / not reported | 14 / 8 | 14 / 8 | 6 / 5 | 8 / 6 |
| Empty / nonempty tracked diff | not reported | not reported | not reported | 21 / 19 | 20 / 20 |
| Nonempty patches evaluated in workspace | not reported | resolved artifacts verified | resolved artifacts verified | 19 | 20 |
| Patch-application / evaluator failures | not reported | 0 evaluator-path errors in resolved artifacts | 0 evaluator-path errors in resolved artifacts | 0 / 0 | 0 / 0 |
| Model calls | 1,444 | 1,217 | 1,580 | 2,080 | 2,116 |
| Completion tokens | 354,584 | 211,349 | 299,105 | 391,900 | 419,928 |
| Persisted traces over configured cap | not reported | not reported | not reported | 0 / 258 | 1 / 250 |

The final Codex runs used the repaired advertised patch tool, four empty-diff
retries, and two same-session review retries. Unlike the superseded zero-result
suite, neither final row recorded a patch-application or evaluator failure.
Mini-SWE still uses its shell loop and remaining-turn reminder, whereas Codex
uses bounded continuation behavior, so call and token counts reflect different
agent-control policies. The SWE-Gym fallback grader is also all-or-nothing: a
nonzero combined pytest exit marks every listed FAIL_TO_PASS and PASS_TO_PASS
target as failed; those lists do not prove every individual test failed.

## Matched setup

| Axis | SkyRL Mini-SWE workers4 | POLAR Mini-SWE run2 | POLAR Mini-SWE run4 | POLAR Codex run2 | POLAR Codex run4 |
| --- | --- | --- | --- | --- | --- |
| Model | Qwen3.5-4B | Qwen3.5-4B | Qwen3.5-4B | Qwen3.5-4B | Qwen3.5-4B |
| Data | SWE-Gym 3h200 lineage | same 16-row workload | same 16-row workload | same 16-row workload | same 16-row workload |
| Worker point | generation workers 4 | init4/run2/eval4 | init4/run4/eval4 | init4/run2/eval4 | init4/run4/eval4 |
| Rollout batches | 4 | 4 | 4 | 4 | 4 |
| Prompts per batch | 4 | 4 | 4 | 4 | 4 |
| Samples per prompt | 2 | 2 | 2 | 2 | 2 |
| Training microbatch | 1 | 1 | 1 | 1 | 1 |
| Async/staleness level | 2 | 2 | 2 | 2 | 2 |
| GPU layout | 1 train, 2 rollout | 1 train, 2 rollout | 1 train, 2 rollout | 1 train, 2 rollout | 1 train, 2 rollout |
| Agent harness | Mini-SWE | Mini-SWE | Mini-SWE | Codex CLI | Codex CLI |
| Agent context | 40,960 | 40,960 | 40,960 | 40,960 | 40,960 |
| Maximum completion/call | 2,048 | 2,048 | 2,048 | 2,048 | 2,048 |
| Maximum agent turns | 40 | 40 | 40 | Codex-controlled plus retries | Codex-controlled plus retries |
| POLAR train trace cap | n/a | 8,192 | 8,192 | 32,768 | 32,768 |
| Profiling | lightweight; no Nsight | lightweight; no Nsight | lightweight; no Nsight | lightweight; no Nsight | lightweight; no Nsight |

Mini-SWE uses the same remaining-turn reminder behavior in both frameworks.
The training/inference implementations still differ: SkyRL uses FSDP/vLLM,
while POLAR uses Megatron/SGLang. SkyRL exposes one generation-worker pool;
POLAR separately controls init, run, and evaluator workers.

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

Final tool-fixed POLAR Codex suite:

- Suite log: `/NHNHOME/home/profiled_runs/miniswe_small_parity_codex_toolfix_review2_writableeval_1train2rollout_20260723T013500Z/suite.log`
- Sweep log: `/NHNHOME/home/profiled_runs/miniswe_small_parity_codex_toolfix_review2_writableeval_1train2rollout_20260723T013500Z/polar_codex_fixed/sweep.log`
- Run2: `/NHNHOME/home/prorl_agent_server_runs/swegym_slime_grpo_3h200/miniswe_small_parity_codex_toolfix_review2_writableeval_1train2rollout_20260723T013500Z_polar_codex_fixed_init4_run2`
- Run4: `/NHNHOME/home/prorl_agent_server_runs/swegym_slime_grpo_3h200/miniswe_small_parity_codex_toolfix_review2_writableeval_1train2rollout_20260723T013500Z_polar_codex_fixed_init4_run4`

## Interpretation limits

This is a deliberately small workload, so accuracy rates have high sampling
variance. Trajectory-level resolution is not the same as unique-task pass rate
when each prompt has two samples. The table reports both raw resolved
trajectories and the unique solved-task count in the detail text. Framework
speed conclusions should wait for the matched SkyRL workers4 versus POLAR
run4 row.

## Codex patch-path validation — 2026-07-23

Follow-up diagnosis showed that the zero-resolved Codex rows above did not
exercise a healthy Codex patch path. `codex exec` had been forced onto
`unified_exec`, while the model repeatedly selected `apply_patch`; the latter
was not advertised, producing 17 run2 and 24 run4 undefined-tool errors. The
harness now enables `apply_patch_freeform`, does not force `unified_exec`, and
supports a bounded same-session review continuation after a nonempty diff.

The corrected path was validated on `getmoto__moto-7023`, a task Mini-SWE
resolved repeatedly in both parity configurations. Codex independently
produced the reference source fix: check for an unknown Lake Formation
resource and raise `EntityNotFound` before deletion. A refreshed Singularity
evaluator initially exposed two more infrastructure defects: `/testbed` was
read-only, and the fallback test command used an ambiguous/corrupted Python
path. The evaluator now copies `/testbed` to
`/polar/session/workspace`, applies the patch there, and runs tests through
`/polar/session/home/.venv/bin/python`.

Final validation suite:

- Stem: `prorl_codex_toolfix_lakeformation7023_finaleval2_20260723T012000Z`
- Run: `/NHNHOME/home/prorl_agent_server_runs/swegym_slime_grpo_3h200/prorl_codex_toolfix_lakeformation7023_finaleval2_20260723T012000Z_polar_codex_fixed_init2_run2`
- Result: 6 `COMPLETED`, 3 resolved and 3 unresolved; no patch-application or
  evaluator errors. The three resolved evaluations exited 0 in
  `/polar/session/workspace`.
- Constraints: existing sandbox image only, no SIF, no Nsight, and no SkyRL or
  Mini-SWE rerun.

This proves that corrected POLAR Codex can return patches that apply, execute
the benchmark tests, and resolve a task. The 3/6 smoke is a patch-path
validation, not a replacement accuracy estimate for the original 16-task
matrix. The earlier 0/40 rows remain historical measurements of the broken
tool configuration and must not be treated as a fair Codex-versus-Mini-SWE
quality comparison.
