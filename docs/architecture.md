# Architecture

All Python implementation, entry points, tests, and workflow shells are below `src/`.
Concise shell submission fronts live in root-level `scripts/`; they resolve the
project root before delegating to `src/slurm/` and submitting the root Slurm files.

| Owner | Responsibility |
|---|---|
| `timebench/data/windows.py` | Arrow-backed target access, official row order, validation rows, causal datastore candidates |
| `timebench/model_loading/` | Official Chronos controls and strict released ARM loading |
| `timebench/external_models/tsrag/` | Pinned borrowed Bolt core, ARM, T5 encoder, FAISS rule, causal native rollout |
| `timebench/proposal/mixture.py` | Per-date MSSE comparison, Beta smoothing, frozen convex mixture |
| `timebench/evaluation/` | Current Improved TIME loader, grid, metric computation, dispersion, saving and synchronized timing |
| `timebench/pipeline/` | Plain scientific manifests, recovery, producer resolution and stage orchestration |
| `timebench/results/` | Exact matched evaluation reports and dispersion comparisons |
| `timebench/conf/`, `timebench/scripts/` | Hydra configuration and experiment/Seasonal/finalization entry points |
| `src/slurm/` | Submission/runtime implementation shared by concise root fronts |
| `src/tests/` | Locally runnable synthetic scientific and lifecycle contracts |

The visible workflow is preparation, four vanilla controls, retrieval extraction,
one or sixteen native TS-RAG predictions, mixture weighting/prediction, TIME evaluation, and report.
Prepared references use `(Arrow item, variate, origin)` and follow TIME's
item-variate-window order. The selected shared Seasonal grid is flattened into
that same order before finite-output checks. The inherited saver reverses this
flattening for standard series-window-variate metric arrays.

Extraction stores the union of potentially usable historical T5 EOS and IN-512 representations
across all dataset items and variates once. Its presence never grants query eligibility: the native
retriever filters each candidate by the neighbor's complete 64-point continuation in calendar coordinates.
An unbounded index adds newly observed dates incrementally; a capped index
retains the most recent eligible dates. Recursive forecast chunks retain the unchanged real cutoff. Date-aligned cells
add a query-calendar-phase restriction. Same-variate cells restrict item/variate;
IN-L2 cells use blockwise finite-overlap distances without T5. Query-scaled T5
cells encode eligible aligned lookbacks at each query and chunk; this cost is
included in query timings. Aligned neighbors share the query's Bolt normalization
during fusion, so full-trajectory IN cannot cancel the alignment.

The prepared union extends through the final test query. After validation selects
the frozen mixture weight, testing admits validation-period observations under
the same stride, most-recent-window cap and complete 64-point future boundary.
Neither validation length nor weight selection freezes the test datastore at
the start of validation. Validation queries still see only their own observed past.

Every phase owns an independent `run_n/manifest.json`. Producer dependencies
embed schema, identity, model/pipeline/experiment configuration, and seed as
plain scientific values. Paths and inherited revisions remain provenance.
Source/data/weight files are never hashed. Exact producer resolution requires
one completed scientific match; selected repeat handling uses the shared
`SELECTED_RUNS.json` contract.

The scheduler launches each stage with one `srun --ntasks=1`. Python writes a
ready-artifact declaration while leaving new task manifests running; only the
post-srun finalizer marks them completed. Failure/exit interrupts the remaining
running manifests owned by that launch. A restarted task recomputes from its
beginning, and separate completed controls remain reusable.

The native ARM and retriever are adapted from
[UConn-DSIS/TS-RAG, revision 73ac807](https://github.com/UConn-DSIS/TS-RAG/tree/73ac807789d2e61b8a3dfc8514e3fc947fe185cc).
The local Bolt core is adapted from
[Chronos forecasting, revision 7dc4435](https://github.com/amazon-science/chronos-forecasting/tree/7dc4435706a4454feb79df44ca9f33631f3027bf).
Default MoE forward computation and full-neighbor normalization are retained;
the explicit query-scale ablation changes neighbor normalization only. Training and
alternate ARM paths are narrowed away. Causal datastore admission, rollout,
fallback, lifecycle, one-axis ablation and the Bayesian-style mixture are project-specific.
Date-period settings, lookback IN, finite-overlap L2 and affine query-scale
alignment are narrowed from Adaptime; no fitting/training boundary was imported.
Scheduler runtimes derive artifact roots from the owning project, so another
checkout's environment settings cannot redirect predictions or logs.
