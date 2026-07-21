# POLAR Mini-SWE `init8/run4` vs SkyRL Workers 8 and Codex CLI

Generated from completed measured artifacts on 2026-07-21. No Nsight data is
used in this report.

## Bottom line

POLAR Mini-SWE completed `init8/run4/eval8` in 218.9 minutes. That is 9.8
minutes, or 4.3%, faster than the requested SkyRL Mini-SWE parallel-workers-8
baseline. POLAR processed 272 accepted trajectories versus 200 for SkyRL and
had 1.42 times the accepted-trajectory throughput. Completion-token throughput
was nearly tied: approximately 8.35k tokens/minute for POLAR and 8.58k for
SkyRL.

This is not a quality win. SkyRL resolved 40 of 200 tasks with mean reward
0.200. POLAR reported mean reward 0 and resolved rate 0 in every rollout batch.
The matched Codex CLI row was much faster at 52.9 minutes, but all 324 saved
trajectory rewards were 0. Its trajectories were also far shorter: 6.09 calls
and 409.5 completion tokens per trajectory, compared with 36.06 calls and
6718.7 tokens for POLAR Mini-SWE. Codex timing is therefore a systems/load
reference, not evidence of useful agent throughput.

## Main comparison

| System | Requested/matched worker point | Wall min | Accepted sessions | Sessions/min | LLM calls/session | Completion tok/session | Completion tok/min | Reward avg | Resolved |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| POLAR + Mini-SWE | init 8, run 4, eval 8 | 218.9 | 272 | 1.243 | 36.06 | 6718.7 | 8350 | 0.000 | 0 |
| SkyRL + Mini-SWE | parallel generation workers 8 | 228.7 | 200 | 0.875 | 37.70 | 9811.4 | 8580 | 0.200 | 40/200 |
| POLAR + Codex CLI | init 8, run 4, eval 8 | 52.9 | 324 | 6.125 | 6.09 | 409.5 | 2508 | 0.000 | 0/324 reward-positive |

Derived values use:

- `sessions/min = accepted sessions / wall minutes`
- `completion tok/min = accepted sessions * average completion tokens / wall minutes`

The POLAR Mini-SWE trace summaries contain seven rollout/training summary rows.
Its 272 sessions are accepted, traced trajectories, not every attempted
Mini-SWE session. The result directory has 372 terminal attempts: 243
`COMPLETED`, 124 `TIMEOUT`, and 5 `ERROR`.

## Relative performance

| Comparison | Measured result |
| --- | --- |
| POLAR Mini-SWE vs SkyRL wall time | POLAR was 9.8 min faster, a 4.3% reduction |
| POLAR Mini-SWE vs SkyRL accepted sessions/min | POLAR was 1.42x higher |
| POLAR Mini-SWE vs SkyRL completion tok/min | POLAR was about 2.7% lower |
| POLAR Mini-SWE vs matched Codex wall time | POLAR was 4.14x slower |
| POLAR Mini-SWE vs matched Codex calls/session | POLAR used 5.92x more calls |
| POLAR Mini-SWE vs matched Codex completion tok/session | POLAR generated 16.4x more tokens |

## Configuration parity

The Codex CLI row is the topology-matched comparator for the completed POLAR
row. SkyRL parallel workers 8 is the separately requested baseline and is not
an exact topology match.

| Axis | POLAR Mini-SWE | SkyRL Mini-SWE workers 8 | POLAR Codex CLI |
| --- | --- | --- | --- |
| Model | Qwen3.5-4B | Qwen3.5-4B | Qwen3.5-4B |
| Data lineage | SWE-Gym 3h200 | Same SWE-Gym/SWE-bench source lineage | SWE-Gym 3h200 |
| Harness | Mini-SWE-Agent | Mini-SWE-Agent | Codex CLI 0.121.0 |
| Worker controls | init 8, run 4, postrun/eval 8 | one generation-worker pool of 8 | init 8, run 4, postrun/eval 8 |
| Async/staleness | async level 2 | max staleness 2 | async level 2 |
| Train batch / samples | 8 / 4 per prompt | 8 / 4 per prompt | 8 / 4 per prompt |
| Microbatch | 2 | 1 | 2 |
| GPU layout | 2 train, 1 rollout | 1 train, 2 rollout | 2 train, 1 rollout |
| Train/inference backend | Megatron / SGLang | FSDP / vLLM | Megatron / SGLang |
| Context | 32768 | not reported in source summary | 16384 |
| Train trace cap | 8192 tokens per trace; over-limit prefixes dropped | not reported in source summary | 16384 max tokens/GPU |
| Max agent turns | 40 | 40 | Codex-controlled short trajectories |

SkyRL has no separate init and postrun worker pools. Its parallel generation
workers correspond most closely to the POLAR run-worker pool. Therefore the
SkyRL workers-8 row is closer to POLAR `run8` than to the completed POLAR
`run4` row. GPU topology and microbatch also differ. Wall-time comparisons
against SkyRL should be treated as an observed cross-system result, not a pure
framework speed ratio.

## Training and GPU behavior

| Metric | POLAR Mini-SWE init8/run4 | POLAR Codex init8/run4 |
| --- | ---: | ---: |
| Steady train time/step | 1226.9 s | 377.5 s |
| Actor train time/step | 901.2 s | 287.2 s |
| Reference logprob time/step | 167.3 s | 54.7 s |
| Policy logprob time/step | 158.2 s | 35.4 s |
| Actor train tokens/s | 1784.5 | 1403.0 |
| GPU0 idle | 32.1% | 17.1% |
| GPU1 idle | 32.2% | 17.4% |
| GPU2 rollout idle | 2.1% | 62.0% |

POLAR Mini-SWE kept its single rollout GPU nearly continuously active, whereas
the short Codex workload left that GPU idle for 62% of the `train_async`
window. Mini-SWE training took 3.25 times longer per steady step because the
accepted sequences were much longer, even though measured actor token
throughput was higher.

SkyRL GPU labels have the opposite role assignment: GPU0/GPU1 were rollout
GPUs and GPU2 was the train GPU. SkyRL reported 91.2%, 91.3%, and 13.7% idle
respectively. These percentages should not be compared column-for-column with
the POLAR table.

## Quality interpretation

- SkyRL is the only quality-positive point: 40 resolved tasks and mean reward
  0.200.
- POLAR Mini-SWE successfully executed long inspect/edit/test trajectories, but
  every rollout summary reported reward mean 0 and resolved rate 0. Rollout
  execution success ranged from 87.5% to 100%; execution success is not task
  resolution.
- The exact matched Codex row has 324 `COMPLETED` result files and every one has
  `reward: 0.0`. A separate Codex audit also found predominantly empty patches
  in this local-Qwen Responses/tool-call setup. Its 52.9-minute wall time must
  not be interpreted as successful SWE throughput.
- The immediate issue exposed by this comparison is quality, not GPU OOM or
  harness availability. POLAR Mini-SWE completed without OOM under the 8192
  token trace cap, but did not produce evaluator reward.

## Source artifacts

POLAR Mini-SWE completed row:

- Run directory: `/NHNHOME/home/prorl_agent_server_runs/swegym_slime_grpo_3h200/prorl_miniswe_mb2_tokcap16k_workers8_16_20260721T011600Z_init8_run4`
- Sweep log: `/NHNHOME/home/profiled_runs/prorl_miniswe_mb2_tokcap16k_workers8_16_20260721T011600Z/sweep.log`
- Session summary: `lightweight_trace_summary/session_summary.tsv`
- Training summary: `lightweight_trace_summary/train_summary.tsv`
- GPU idle: `gpu_idle_summary.tsv`

SkyRL workers-8 source:

- `SKYRL_MINISWE_PRORL_COMPARE_INIT8_RUN8_REPORT_ASYNC2_1TRAIN2ROLLOUT_MB1.md`
- Manifest: `/NHNHOME/home/skyrl_prorl_runs/skyrl_miniswe_prorl_compare_init8_run8_20260720T002127Z/manifest.tsv`

Topology-matched Codex source:

- `WORKER_MATRIX_METRICS_REPORT_ASYNC2.md`
- Run directory: `/NHNHOME/home/prorl_agent_server_runs/swegym_slime_grpo_3h200/qwen35_lighttrace_mb2_steps2_worker_matrix_async2_ctn_20260717T051225Z_init8_run4`

## Scope

This report compares one completed POLAR Mini-SWE row with the explicitly
requested SkyRL workers-8 baseline and one topology-matched Codex CLI row. It
does not claim statistical significance from a single run and does not use the
still-running later POLAR worker configurations.
