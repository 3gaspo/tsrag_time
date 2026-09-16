# Architecture

All Python implementation, entry points, tests, and workflow shells are below `src/`.

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
native TS-RAG prediction, mixture weighting/prediction, TIME evaluation, and report.
Prepared references use `(Arrow item, variate, origin)` and follow TIME's
item-variate-window order. The selected shared Seasonal grid is flattened into
that same order before finite-output checks. The inherited saver reverses this
flattening for standard series-window-variate metric arrays.

Extraction stores the union of potentially usable same-series historical
representations once. Its presence never grants query eligibility: the native
retriever filters each candidate by `candidate_origin + 64 <= real_query_origin`.
An unbounded index adds newly observed dates incrementally; a capped index
retains the most recent eligible dates. Recursive forecast chunks re-embed and
retrieve using the unchanged real cutoff.

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
Released MoE forward computation and normalization are retained; training and
alternate ARM paths are narrowed away. Causal datastore admission, rollout,
fallback, lifecycle, and the Bayesian-style mixture are project-specific.
