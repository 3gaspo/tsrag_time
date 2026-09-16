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
| Native TS-RAG, released MoE ARM | 512; Bolt-512 fallback |
| Validation-weighted Chronos-2 / TS-RAG mixture | Frozen Beta-smoothed win-frequency weight |

Forecasts are univariate medians. TS-RAG uses Chronos-T5-base EOS embeddings,
same-series exact FAISS retrieval, ten neighbors, and 64-step native forecasts.
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
bash submit_seasonal_naive.sh dgx shared
```

Run the narrow remote smoke task (`SG_Weather/D`, short), then the full comparison:

```bash
EXPERIMENT_MODE=test bash submit_experiment.sh dgx
bash submit_experiment.sh dgx
```

The full configuration preserves Adaptime's 90-task scope, excluding
`Coastal_T_S/5T`, `current_velocity/20T`, `azure2019_D/5T`, and `azure2019_I/5T`.
Overrides apply to every stage and report:

```bash
bash submit_experiment.sh dgx 'datasets=[SG_Weather/D]' 'terms=[short]'
bash submit_experiment.sh dgx validation_length=0
```

Default validation uses TIME's configured interval immediately before testing.
The mixture estimates a Beta(1,1)-smoothed probability that TS-RAG has lower
per-date mean-variate MSSE than Chronos-2, counting ties as half wins. With
no usable validation dates it uses pure Chronos-2. Validation length never
changes the official test interval.

The datastore includes every historical date by default, with no fitting-derived
boundary. At each real query, a neighbor is eligible only if its entire
512-point history and 64-point future are observed. Previously observed test
dates may enter later queries' datastores. During long-horizon rollout the
cutoff stays at the real query date. `datastore_stride` and
`max_datastore_windows` explicitly restrict this pool when supplied.

## Outputs and cluster operations

One allocation executes `prepare,vanilla,extract,predict,mix,evaluate,report` in
order. Root fronts source `src/slurm/` implementations. Each stage allocates
schema-1 `run_n` manifests before work; completion occurs after successful
`srun`. Restarting the launcher skips exact completed tasks and recomputes
interrupted tasks from their beginning. `STAGES` is a comma-separated recovery
override. `TIME_RUN_CONFLICT_POLICY=overwrite_exact|overwrite_path|new`,
`TIME_SKIP_COMPLETED`, and `TIME_FORCE_RERUN` retain the shared lifecycle controls.

Artifacts live under `outputs/tsrag/{data,extractions,predictions,evaluations,reports}`.
Canonical `predictions/<method>/<dataset>/<term>/run_n/test.npy` files retain
float32 medians. Evaluation runs contain standard TIME predictions, metric
arrays, and compact summaries. TS-RAG fallback masks/reasons and mixture weights
are persisted. Reports include mean, population variance, standard deviation,
finite-value counts, scaled MASE, and the matched Seasonal MASE variance ratio.
An unavailable or zero Seasonal variance leaves that ratio undefined.

TS-RAG falls back to Bolt-512 for insufficient history/retrieval, extraction or
inference errors, and non-finite predictions on required target steps. A fallback
replaces the complete window/variate forecast. Foundation controls must be finite
on the selected grid. The mixture uses Chronos-2 if its combined output is invalid.

Recorded timings separate datastore preprocessing and query inference.
TS-RAG reuses the already computed Bolt-512 fallback; mixture inference totals
include both components and mixing. These are pipeline timings, not independent
fresh-process latency measurements.

`sync_code_to_selena.sh`, `sync_results_to_dgx.sh`, `clear_selena_artifacts.sh`, and
`publish_job.sh` retain project-scoped cluster operations. Lightweight transfer
includes reports and compact lifecycle/metric metadata; `--size detailed` adds
metric arrays and fallback reasons; `--size full` includes canonical predictions
and embedding binaries. Consumers should select scientifically equivalent
completed manifests and read canonical results in place.

## Documentation maintenance

All implementation and tests are under `src/`: `data/` owns windows;
`model_loading/` owns checkpoints and foundation adapters; `external_models/tsrag/`
owns borrowed ARM/retrieval and native inference; `proposal/` owns the mixture;
`pipeline/` owns orchestration/manifests; `results/` owns reports; `conf/` and
`scripts/` own configuration and entry points. `src/slurm/` owns scheduler shells.

`src/scripts/build_docs.py --render all` builds the three PDFs with pdfLaTeX;
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
