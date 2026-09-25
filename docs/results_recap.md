# Results recap

The first completed comparison covers all 90 tasks with seed 0. All six
methods have finite MASE on the same 67,502 declared evaluation cells.
The completed report selects 540 evaluations; their compact metric means,
population variance and standard deviation agree with its comparison table.

| Method | Mean task MASE | Recorded inference seconds |
|---|---:|---:|
| Chronos-T5, 512 | 1.497300 | 4,875.84 |
| Chronos-Bolt, 512 | 1.213402 | 84.77 |
| Chronos-Bolt, maximum context | 1.167905 | 251.32 |
| Chronos-2, maximum context | 1.069230 | 236.76 |
| TS-RAG | 1.200871 | 18,115.12 |
| Frozen Chronos-2 / TS-RAG mixture | 1.071780 | 18,351.90 |

These are arithmetic means over matched tasks. TS-RAG has 12.31% higher mean
MASE than Chronos-2, winning 8 tasks and losing 82. The frozen mixture has
0.24% higher mean MASE, winning 36 tasks and losing 54. TS-RAG also has 2.82%
higher mean MASE than its Bolt-max fallback (19 wins, 52 losses and 19 ties).
This run therefore provides no aggregate accuracy benefit for retrieval or
the validation mixture.

TS-RAG uses Bolt-max on 28,860 scored rows (42.75%). Fallback occurs in 49
tasks, including 19 with complete fallback; 41 have none. No task-level errors
are reported. The published aggregate reason counts combine validation and test,
so they are excluded from evaluation findings. The corrected contract reports
test reasons alone; its reason distribution requires a future current-contract
prediction/report run, not resumption of the retired 16-cell grid.

Frozen TS-RAG mixture weights range from 0.0455 to 0.7143, with mean 0.3454.
All 90 are estimated: usable validation dates total 27,914 and range from 1 to
14,868 per task. Recorded TS-RAG datastore preprocessing adds 55,578.80 seconds.
Inference totals reuse pipeline components and are not independent fresh-process
latency measurements.

The retrieval setting is all items/variates, unaligned, without query-scale
normalization, using T5 representations. The unfinished 16-cell attempt produced
no result and is retired; a smaller replacement grid remains to be designed.
One seed provides no across-run uncertainty; population dispersion describes
within-task metric cells. All 810 compact preparation, extraction, prediction
and weight records are now published. Raw predictions, metric arrays, shared
Seasonal payloads and per-row fallback-reason arrays were not independently
audited. Test-only cause totals and native-only accuracy on a common support are
not yet evidenced. The
executive-summary PDF presents this completed default comparison and its
fallback, timing and evidence limitations.
