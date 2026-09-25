# Experiment catalog

| Front | Question | Ordered stages |
|---|---|---|
| `experiment.slurm` | Retrieval benefit, context restriction, stronger backbone, and validation mixing | prepare, vanilla, extract, predict, mix, evaluate, report |
| `experiment_selena.slurm` | Same scientific experiment on the overflow execution surface | same stages |
| `seasonal_naive.slurm` | Produce the common finite-support grid and matched scaling baseline | Seasonal forecast/evaluation |
| `seasonal_naive_selena.slurm` | Same Seasonal producer on the overflow execution surface | same stage |

All Selena fronts request one GPU, partition `an`, QoS `an_preemptable`,
exclusive allocation and WCKey `P12CU:DATASCIENCE`, retaining cluster requeue.
Their defaults are one node, one task, eight CPUs, 80 GB memory and 23 hours.

Use `bash scripts/submit_experiment.sh dgx` from the project root. The default `full`
mode selects all three terms present for each configured dataset except the
four documented Adaptime exclusions, giving 90 tasks. `EXPERIMENT_MODE=test`
selects `SG_Weather/D`, short when `datasets=[all]`; explicit datasets/terms
remain authoritative. Scale is selection only and does not change run identity.

The main experiment's six method labels distinguish four vanilla contexts, TS-RAG, and the mixture.
All backbones are frozen and forecasts are univariate medians. T5 is sampled
with the single run seed and `t5_samples=20`; the retrieval encoder does not
sample forecasts. Bolt-512 maintains its cap throughout the official quantile
rollout, while native TS-RAG rolls out median 64-point chunks.

`validation_length=null` uses each dataset's TIME `val_length`; `0` disables
validation. A nonzero interval shorter than the requested horizon, or leaving
no observed prefix, is unavailable rather than changing the official test set.
Without usable validation, the mixture is explicitly pure Chronos-2.

`datastore_stride=1` and `max_datastore_windows=null` admit every eligible date.
The same causal admission rule applies to validation and testing. Candidate
history and future are 512 and 64 points regardless of the official horizon.
After validation selects the mixture weight, test retrieval extends through all
available observations, including validation dates, subject to stride, cap and
the complete-neighbor boundary. The selected weight remains frozen.
Changing stride, cap, validation, T5 sample count, or seed produces a different
plain scientific computation and cannot silently reuse incompatible results.

`STAGES` recovers selected completed-producer stages without changing science.
Conflict/repeat controls retain the shared TIME behavior. Reports select the
requested exact configuration and its selected repeat, record every evaluation
and source manifest, and reject additional metric-coverage loss relative to
the selected Seasonal baseline.

## One-axis retrieval ablation

The former 16-cell factorial grid is retired. The current `retrieval_axes`
configuration compares four independent changes with released TS-RAG:

| Method label | Changed setting |
|---|---|
| `tsrag_same_series` | `scope=same_series` |
| `tsrag_aligned` | `aligned=true` |
| `tsrag_inst_l2` | `representation=instance_l2` |
| `tsrag_query_scale` | `query_scale=true` |

Selena job 3506764 never completed the first new factorial cell. That schedule
is too long and must not be resumed. `scripts/submit_ablation.sh` is the normal
front for the four one-axis variants. Each keeps every unspecified setting at
the released default, so no interaction between axes is estimated.

All items and variates are pooled within the selected dataset. Same-series
retrieval means the query's own item and variate. Alignment uses Adaptime's
observation-count date periods, accounting for multiplied frequencies and
different item starts. `alignment_period` explicitly overrides those periods.

Query-scale alignment uses lookback-only neighbor statistics before retrieval;
T5 candidates are encoded per query. Fusion uses query normalization rather
than full-neighbor normalization. L2 cells use cached IN lookbacks of length
512, with finite-overlap distances and `minimum_overlap_fraction=0.8`, and do
not invoke T5 selection. The normalization axis still changes their fusion.

Every variant falls back to Bolt-max (2,048 points). The ablation contains five
TS-RAG methods per task: the released default plus four variants. Main and ablation executions
reuse exact matching prepared data, caches and default predictions. Reports
identify every axis, alignment period and fallback method.

Scheduler runtimes enforce project-owned artifact roots while retaining shared
dataset, checkpoint and Seasonal configuration. The current scientific
manifests prevent reuse of previous incompatible TS-RAG computations.
