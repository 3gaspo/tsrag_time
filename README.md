# TS-RAG TIME

**Running TS-RAG on the TIME benchmark.** This independent project evaluates
the retrieval-augmented method of [Ning et al. (2025)](https://arxiv.org/abs/2503.07649)
using the forecasting tasks and evaluation protocol of
[Qiao et al. (2026)](https://arxiv.org/abs/2602.12147).
It produces reusable TS-RAG forecasts and compares them with frozen Chronos models.
Every method uses the same official test windows and Seasonal Naive metric grid.

Benchmark infrastructure derives from [Improved TIME at revision 541a280](https://github.com/3gaspo/improved_TIME/tree/541a2802cd2a35d39156aef4c37de6964d112786).
The repository retains this parent's Git history for future synchronization.

| Method | Context |
|---|---|
| Vanilla `chronos_t5` (base) | 512 |
| Vanilla `chronos_bolt` (base) | 512 |
| Vanilla `chronos_bolt` (base) | Maximum: 2,048 |
| Vanilla `chronos2` | Maximum: 8,192 |
| Native TS-RAG, released MoE ARM | 512; Bolt-max fallback |
| Validation-weighted Chronos-2 / TS-RAG mixture | Frozen Beta-smoothed win-frequency weight |

Forecasts are univariate medians. TS-RAG uses Chronos-T5-base EOS embeddings,
cross-user, cross-variate exact FAISS retrieval within each dataset, ten neighbors, and 64-step native forecasts.
The independent T5 control uses 20 forecast samples by default. No Ridge fitting,
training split, learned wrapper training, or hyperparameter selection is performed.

## Documentation map

- [Method overview](latex/method_overview.pdf): model and mixture formulation.
- [Experiment guideline](latex/experiment_guideline.pdf): complete protocol.
- [Architecture](docs/architecture.md): source ownership and stage contracts.
- [Experiment catalog](docs/experiment_catalog.md): questions and launch controls.
- [Results recap](docs/results_recap.md) and [executive summary](latex/executive_summary.pdf): evidence status.

## Setup

The user prepares the Python 3.12 execution environment on the cluster:

```bash
uv sync
```

Use an already prepared TIME saved-Arrow dataset. Configure `.env` from
`.env.example`; set `TIME_DATASET`, `TIME_WEIGHTS`, and the selected Seasonal
store. Required checkpoint directories below `TIME_WEIGHTS` are
`chronos-t5-base/`, `chronos-bolt-base/`, `chronos2/`, and `ts-rag/`.
The last must contain exactly one released `best.pth`, loaded strictly.
Inference loads local checkpoints only and never downloads weights.

The main path variables retain TIME's defaults: `TIME_DATASET=datasets/hf_dataset`,
`TIME_WEIGHTS=weights`, `TIME_OUTPUTS=outputs`, and `TIME_LOGS=logs`.
`TIME_SEASONAL_SCOPE=shared` selects the existing common Seasonal store;
`TIME_SEASONAL_ROOT` or `TIME_SEASONAL_TASKS_ROOT` may override its location.
Prepared Arrow targets are consumed directly; no CSV exclusions or missing-value
policy is applied a second time.

## Main executions

Run from the project root. If the selected Seasonal grid does not exist, first
submit its producer and wait for completion:

```bash
bash scripts/submit_seasonal_naive.sh dgx shared
```

Run the narrow remote smoke task (`SG_Weather/D`, short), then the full comparison:

```bash
EXPERIMENT_MODE=test bash scripts/submit_experiment.sh dgx
bash scripts/submit_experiment.sh dgx
```

The full configuration preserves Adaptime's 90-task scope, excluding
`Coastal_T_S/5T`, `current_velocity/20T`, `azure2019_D/5T`, and `azure2019_I/5T`.
Overrides apply to every stage and report:

```bash
bash scripts/submit_experiment.sh dgx 'datasets=[SG_Weather/D]' 'terms=[short]'
bash scripts/submit_experiment.sh dgx validation_length=0
```

Default validation uses TIME's configured interval immediately before testing.
The mixture estimates a Beta(1,1)-smoothed probability that TS-RAG has lower
per-date mean-variate MSSE than Chronos-2, counting ties as half wins. With
no usable validation dates it uses pure Chronos-2. Validation length never
changes the official test interval.

Once the validation mixture weight is selected, it stays frozen for testing.
Test retrieval includes all observations available at each query, including the
validation period. There is no cutoff at the start of validation; stride,
the optional window cap and complete-neighbor boundaries still apply.

The datastore includes every historical date by default, with no fitting-derived
boundary. At each real query, a neighbor is eligible only if its entire
512-point history and 64-point future are observed before the real query date. Previously observed test
dates may enter later queries' datastores. During long-horizon rollout the
cutoff stays at the real query date. `datastore_stride` and
`max_datastore_windows` explicitly restrict this pool when supplied.

## Retrieval ablation

The default is cross-user and cross-variate retrieval within the selected dataset,
without date alignment or query-scale normalization, using T5 EOS representations.
The ablation computes all **16 combinations** of:

| Axis | Values |
|---|---|
| Retrieval scope | All items/variates; same item and variate |
| Date alignment | Disabled; same calendar phase as query |
| Neighbor normalization | Released full-trajectory IN; align to query scale before retrieval and fusion |
| Retrieval representation | T5 EOS; instance-normalized L2 over 512 lookback points |

Date alignment uses the observation-count periods copied from Adaptime's dataset
protocol, including multiplied sampling frequencies. Calendar eligibility requires
the neighbor's 64-point continuation to end by the real query date, even for items
with different starts. Validation observations remain available to test queries.

Query-scale alignment uses each neighbor's **512-point lookback** mean/std and
the query's mean/std. T5 cells encode aligned candidates per query because these
representations depend on query scale. During fusion, aligned neighbors use the
query's Bolt normalization rather than their own 576-point normalization.

L2 cells select neighbors without T5. They cache IN lookbacks and compute squared
L2 on finite overlap, rescaled to 512 coordinates, with Adaptime's default minimum
overlap fraction of 0.8. Affine alignment leaves IN L2 rankings unchanged while
changing fusion; these combinations remain in the full grid.

```bash
bash scripts/submit_ablation.sh dgx
```

The ablation includes the four vanilla controls and the default TS-RAG mixture,
giving 21 method labels per task. Main and ablation executions share exact matching
prepared data, extraction caches and default predictions. Comparison CSV/JSON
reports identify every cell's retrieval settings, alignment period and fallback.

## Outputs and cluster operations

One allocation executes `prepare,vanilla,extract,predict,mix,evaluate,report` in
order. Launchers under `scripts/` submit the experiment, ablation and Seasonal root Slurm fronts and source
`src/slurm/` implementations. Each stage allocates
schema-1 `run_n` manifests before work; completion occurs after successful
`srun`. Restarting the launcher skips exact completed tasks and recomputes
interrupted tasks from their beginning. `STAGES` is a comma-separated recovery
override. `TIME_RUN_CONFLICT_POLICY=overwrite_exact|overwrite_path|new`,
`TIME_SKIP_COMPLETED`, and `TIME_FORCE_RERUN` retain the shared lifecycle controls.

Scheduler launchers enforce project-owned artifact roots, ignoring inherited or
copied artifact-path settings while preserving shared data/weight settings.
Scientific stage artifacts live under
`outputs/tsrag/{data,extractions,predictions,evaluations}`. Launch-exact reports
live under `outputs/reports/tsrag/<launch-id>/`; their `performance/` bundle
contains task/domain/timing tables and matched PNG/PDF figures.
Canonical `predictions/<method>/<dataset>/<term>/run_n/test.npy` files retain
float32 medians. Evaluation runs contain standard TIME predictions, metric
arrays, and compact summaries. TS-RAG fallback masks/reasons and mixture weights
are persisted. Reports include mean, population variance, standard deviation,
finite-value counts, scaled MASE, and the matched Seasonal MASE variance ratio.
An unavailable or zero Seasonal variance leaves that ratio undefined.

Fallback counts, rates, and aggregate reason counts in comparison artifacts are
test-only because test is the evaluated split. Validation fallback remains
available only in its split-specific mask and per-row reason artifact for mixture
diagnosis; it is not included in reported fallback totals.

Every TS-RAG variant falls back to Bolt-max (2,048 points) for insufficient history/retrieval, extraction or
inference errors, and non-finite native predictions. A fallback
replaces the complete window/variate forecast. Foundation controls must be finite
on the selected grid. The mixture also uses Bolt-max if its combined output is invalid.

Recorded timings separate datastore preprocessing and query inference.
TS-RAG reuses the already computed Bolt-max fallback; mixture inference totals
include both components and mixing. These are pipeline timings, not independent
fresh-process latency measurements.

`sync_code_to_selena.sh`, `sync_results_to_dgx.sh`, `clear_selena_artifacts.sh`, and
`publish_job.sh` retain project-scoped cluster operations. Lightweight transfer
includes reports and compact lifecycle/metric metadata, including aggregate
test fallback causes and mixture support in `prediction.json`/`weight.json`;
`--size detailed` adds metric arrays and per-row fallback reasons; `--size full`
includes canonical predictions and embedding binaries. Consumers should select
scientifically equivalent completed manifests and read canonical results in place.

Every scheduled allocation logs a compute-node resource snapshot before its
scientific stages: Slurm/job identity, visible and inventoried GPUs, free/total
GPU memory, host RAM and exposed cgroup limits. Model loaders separately record
the device actually selected for each backbone.

## Documentation maintenance

All implementation and tests are under `src/`: `data/` owns windows;
`model_loading/` owns checkpoints and foundation adapters; `external_models/tsrag/`
owns borrowed ARM/retrieval and native inference; `proposal/` owns the mixture;
`pipeline/` owns orchestration/manifests; `results/` owns reports; `conf/` and
`scripts/` own configuration and entry points. `src/slurm/` owns scheduler shells.
Root-level `scripts/` contains the concise experiment, ablation and Seasonal launchers.

`src/scripts/build_docs.py --render all` builds the three PDFs with pdfLaTeX;
`--render protocol` updates only the method and experiment PDFs.
its default mode validates the required public views. Update scientific protocol
documents when behavior changes and evidence documents only after result analysis.

## References

- **TS-RAG:** Kanghui Ning, Zijie Pan, Yu Liu, Yushan Jiang, James Yiming Zhang,
  Kashif Rasul, Anderson Schneider, Lintao Ma, Yuriy Nevmyvaka, and Dongjin Song.
  (2025). [*TS-RAG: Retrieval-Augmented Generation based Time Series Foundation Models are Stronger Zero-Shot Forecaster*](https://arxiv.org/abs/2503.07649).
  arXiv:2503.07649. [Official implementation](https://github.com/UConn-DSIS/TS-RAG).
- **TIME benchmark:** Zhongzheng Qiao, Sheng Pan, Anni Wang, Viktoriya Zhukova,
  Yong Liu, Xudong Jiang, Qingsong Wen, Mingsheng Long, Ming Jin, and Chenghao Liu.
  (2026). [*It's TIME: Towards the Next Generation of Time Series Forecasting Benchmarks*](https://arxiv.org/abs/2602.12147).
  arXiv:2602.12147. [Official benchmark implementation](https://github.com/zqiao11/TIME).
