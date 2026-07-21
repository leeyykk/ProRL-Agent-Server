# POLAR Mini-SWE `init8/run4` and `init8/run8` vs SkyRL Workers 8 and Codex CLI

Generated from completed measured artifacts on 2026-07-21. No Nsight data is
used in this report.

## Bottom line

POLAR Mini-SWE completed `init8/run4/eval8` in 218.9 minutes. It also completed
`init8/run8/eval8` in 175.2 minutes before the later sweep was stopped. The
run8 row processed 323 accepted trajectories at 1.844 sessions/minute, 48.3%
more accepted-session throughput than run4 and 2.11x the SkyRL workers-8 rate.

The run4 result is 9.8 minutes, or 4.3%, faster than the requested SkyRL Mini-SWE parallel-workers-8
baseline. POLAR processed 272 accepted trajectories versus 200 for SkyRL and
had 1.42 times the accepted-trajectory throughput. Completion-token throughput
was nearly tied: approximately 8.35k tokens/minute for POLAR and 8.58k for
SkyRL.

The legacy POLAR quality result is invalid, rather than a quality loss. SkyRL
resolved 40 of 200 tasks with mean reward 0.200. POLAR reported mean reward 0
and resolved rate 0 in every rollout batch, but a later audit found that its
evaluator ran the generated SWE-bench script in `/testbed` while the agent had
edited `/polar/session/workspace`. Consequently, the evaluator tested an
untouched checkout and could not observe POLAR's patch.
The Codex CLI rows were much faster at 52.9 minutes for run4 and 52.5 minutes
for run8, but all 324 and 320 respective saved trajectory rewards were 0. The
run8 trajectories were also far shorter: 5.91 calls and 373.7 completion tokens
per trajectory, compared with 38.84 calls and 6845.1 tokens for POLAR Mini-SWE
run8. Codex timing is therefore a systems/load reference, not evidence of
useful agent throughput.

## Main comparison

| System | Requested/matched worker point | Wall min | Accepted sessions | Sessions/min | LLM calls/session | Completion tok/session | Completion tok/min | Reward avg | Resolved |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| POLAR + Mini-SWE | init 8, run 4, eval 8 | 218.9 | 272 | 1.243 | 36.06 | 6718.7 | 8350 | 0.000 | 0 |
| POLAR + Mini-SWE | init 8, run 8, eval 8 | 175.2 | 323 | 1.844 | 38.84 | 6845.1 | 12622 | 0.000* | 0* |
| SkyRL + Mini-SWE | parallel generation workers 8 | 228.7 | 200 | 0.875 | 37.70 | 9811.4 | 8580 | 0.200 | 40/200 |
| POLAR + Codex CLI | init 8, run 4, eval 8 | 52.9 | 324 | 6.125 | 6.09 | 409.5 | 2508 | 0.000 | 0/324 reward-positive |
| POLAR + Codex CLI | init 8, run 8, eval 8 | 52.5 | 320 | 6.095 | 5.91 | 373.7 | 2278 | 0.000 | 0/320 reward-positive |

Derived values use:

- `sessions/min = accepted sessions / wall minutes`
- `completion tok/min = accepted sessions * average completion tokens / wall minutes`

`*` The run8 reward and resolved values were produced by the evaluator path bug
described below and must not be used for an accuracy comparison.

Each POLAR Mini-SWE trace summary contains seven rollout/training summary rows.
The sessions are accepted, traced trajectories, not every attempted Mini-SWE
session. The result directory has 372 terminal attempts: 243 `COMPLETED`, 124
`TIMEOUT`, and 5 `ERROR` for run4. Run8 has 324 terminal attempts: 317
`COMPLETED`, 6 `ERROR`, and 1 `TIMEOUT`.

## Relative performance

| Comparison | Measured result |
| --- | --- |
| POLAR Mini-SWE vs SkyRL wall time | POLAR was 9.8 min faster, a 4.3% reduction |
| POLAR Mini-SWE vs SkyRL accepted sessions/min | POLAR was 1.42x higher |
| POLAR Mini-SWE vs SkyRL completion tok/min | POLAR was about 2.7% lower |
| POLAR Mini-SWE run8 vs SkyRL accepted sessions/min | POLAR was 2.11x higher |
| POLAR Mini-SWE run8 vs SkyRL completion tok/min | POLAR was 1.47x higher |
| POLAR Mini-SWE vs matched Codex wall time | POLAR was 4.14x slower |
| POLAR Mini-SWE vs matched Codex calls/session | POLAR used 5.92x more calls |
| POLAR Mini-SWE vs matched Codex completion tok/session | POLAR generated 16.4x more tokens |
| POLAR Mini-SWE run8 vs matched Codex run8 wall time | POLAR was 3.34x slower |
| POLAR Mini-SWE run8 vs matched Codex run8 calls/session | POLAR used 6.57x more calls |
| POLAR Mini-SWE run8 vs matched Codex run8 completion tok/session | POLAR generated 18.3x more tokens |

## Configuration parity

The Codex CLI rows are topology-matched comparators for the corresponding
completed POLAR rows. SkyRL parallel workers 8 is the separately requested
baseline and is not an exact topology match.

| Axis | POLAR Mini-SWE | SkyRL Mini-SWE workers 8 | POLAR Codex CLI |
| --- | --- | --- | --- |
| Model | Qwen3.5-4B | Qwen3.5-4B | Qwen3.5-4B |
| Data lineage | SWE-Gym 3h200 | Same SWE-Gym/SWE-bench source lineage | SWE-Gym 3h200 |
| Harness | Mini-SWE-Agent | Mini-SWE-Agent | Codex CLI 0.121.0 |
| Worker controls | init 8, run 4 or 8, postrun/eval 8 | one generation-worker pool of 8 | init 8, run 4 or 8, postrun/eval 8 |
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

| Metric | POLAR Mini-SWE init8/run4 | POLAR Mini-SWE init8/run8 | POLAR Codex init8/run4 | POLAR Codex init8/run8 |
| --- | ---: | ---: | ---: | ---: |
| Steady train time/step | 1226.9 s | 1249.4 s | 377.5 s | 383.6 s |
| Actor train time/step | 901.2 s | 921.0 s | 287.2 s | 293.1 s |
| Reference logprob time/step | 167.3 s | 166.8 s | 54.7 s | 53.1 s |
| Policy logprob time/step | 158.2 s | 161.3 s | 35.4 s | 37.1 s |
| Actor train tokens/s | 1784.5 | 1844.3 | 1403.0 | 1413.2 |
| GPU0 idle | 32.1% | 14.6% | 17.1% | 14.9% |
| GPU1 idle | 32.2% | 14.4% | 17.4% | 14.0% |
| GPU2 rollout idle | 2.1% | 5.2% | 62.0% | 78.8% |

POLAR Mini-SWE kept its single rollout GPU nearly continuously active, whereas
the short Codex workloads left that GPU idle for 62.0% and 78.8% of their
respective `train_async` windows. Mini-SWE training took 3.25 times longer per steady step because the
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
  the historical zero-reward figures do not characterize its accuracy. The
  evaluator script hard-coded `cd /testbed`, whereas the agent's patch was in
  `/polar/session/workspace`. The rerun fixes the evaluator to test the latter.
- The matched Codex run4 and run8 rows have 324 and 320 accepted trajectories,
  respectively, and every saved trajectory has `reward: 0.0`. A separate Codex
  audit also found predominantly empty patches
  in this local-Qwen Responses/tool-call setup. Their 52.9- and 52.5-minute wall times must
  not be interpreted as successful SWE throughput.
- POLAR Mini-SWE completed without OOM under the 8192-token trace cap. Its
  historical accuracy is unknown because of the evaluator path mismatch, so a
  corrected rerun is required before comparing accuracy with SkyRL.

## Source artifacts

POLAR Mini-SWE completed row:

- Run directory: `/NHNHOME/home/prorl_agent_server_runs/swegym_slime_grpo_3h200/prorl_miniswe_mb2_tokcap16k_workers8_16_20260721T011600Z_init8_run4`
- Sweep log: `/NHNHOME/home/profiled_runs/prorl_miniswe_mb2_tokcap16k_workers8_16_20260721T011600Z/sweep.log`
- Session summary: `lightweight_trace_summary/session_summary.tsv`
- Training summary: `lightweight_trace_summary/train_summary.tsv`
- GPU idle: `gpu_idle_summary.tsv`

Additional completed POLAR Mini-SWE run8 row:

- Run directory: `/NHNHOME/home/prorl_agent_server_runs/swegym_slime_grpo_3h200/prorl_miniswe_mb2_tokcap16k_workers8_16_20260721T011600Z_init8_run8`
- Sweep log: `/NHNHOME/home/profiled_runs/prorl_miniswe_mb2_tokcap16k_workers8_16_20260721T011600Z/sweep.log`
- Session/training summaries: `lightweight_trace_summary/`
- GPU idle: `gpu_idle_summary.tsv`

SkyRL workers-8 source:

- `SKYRL_MINISWE_PRORL_COMPARE_INIT8_RUN8_REPORT_ASYNC2_1TRAIN2ROLLOUT_MB1.md`
- Manifest: `/NHNHOME/home/skyrl_prorl_runs/skyrl_miniswe_prorl_compare_init8_run8_20260720T002127Z/manifest.tsv`

Topology-matched Codex source:

- `WORKER_MATRIX_METRICS_REPORT_ASYNC2.md`
- Run4 directory: `/NHNHOME/home/prorl_agent_server_runs/swegym_slime_grpo_3h200/qwen35_lighttrace_mb2_steps2_worker_matrix_async2_ctn_20260717T051225Z_init8_run4`
- Run8 directory: `/NHNHOME/home/prorl_agent_server_runs/swegym_slime_grpo_3h200/qwen35_lighttrace_mb2_steps2_worker_matrix_async2_ctn_20260717T051225Z_init8_run8`

## Scope

This report compares two completed POLAR Mini-SWE rows with the explicitly
requested SkyRL workers-8 baseline and two topology-matched Codex CLI rows. It
does not claim statistical significance from a single run and does not use the
later POLAR worker configurations that were stopped before completion. The
historical POLAR resolved rates are retained for artifact fidelity but are not
valid accuracy measurements because of the evaluator path bug.
