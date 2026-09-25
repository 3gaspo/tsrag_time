"""Shared task-level performance tables and artifact-only report bundles."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


TASK_KEYS = ["dataset", "frequency", "term"]


def prepare_tasks(rows, domains=None):
    """Accept already selected/reduced task means, never pool metric cells."""
    frame = pd.DataFrame(rows).copy()
    required = ["model", *TASK_KEYS, "horizon_steps", "MASE", "scaled_MASE", "inference_seconds"]
    missing = set(required) - set(frame.columns)
    if missing or frame.empty:
        raise ValueError(f"Empty/incomplete performance task table: {sorted(missing)}")
    if frame.duplicated(["model", *TASK_KEYS]).any():
        raise ValueError("Reduce repeats/configurations or give distinct model labels before reporting")
    for column in ("MASE", "scaled_MASE"):
        values = frame[column].to_numpy(dtype=float)
        if np.isinf(values).any() or (values[np.isfinite(values)] < 0).any():
            raise ValueError(f"Task {column} must be non-negative or NaN")
    timing_columns = [column for column in frame if column.endswith("_seconds")]
    for column in timing_columns:
        values = frame[column].to_numpy(dtype=float)
        if np.isinf(values).any() or (values[np.isfinite(values)] < 0).any():
            raise ValueError(f"Task {column} must be non-negative or missing")
    horizons = frame["horizon_steps"].to_numpy(dtype=float)
    if not np.isfinite(horizons).all() or (horizons < 1).any() or (horizons != np.floor(horizons)).any():
        raise ValueError("Forecast horizons must be positive observation counts")
    frame["horizon_steps"] = horizons.astype(int)
    if (frame.groupby(TASK_KEYS)["horizon_steps"].nunique() != 1).any():
        raise ValueError("Models disagree on a task's forecast horizon")
    if domains is None:
        domain_path = Path(__file__).resolve().parents[1] / "config/dataset_domains.json"
        domains = json.loads(domain_path.read_text(encoding="utf-8"))
    if "domain" not in frame:
        frame["domain"] = frame["dataset"].map(domains).fillna("Unclassified")
    return frame


def relative_task_values(tasks, reference, loss="MASE"):
    """Keep all tasks; zero reference loss makes percentage change undefined."""
    selected = tasks[tasks["model"] == reference]
    baseline = selected[TASK_KEYS].copy()
    if baseline.empty:
        raise ValueError(f"Reference model is absent: {reference}")
    baseline["reference_loss"] = selected[loss].to_numpy()
    baseline["reference_scaled_MASE"] = selected["scaled_MASE"].to_numpy()
    baseline["reference_inference_seconds"] = selected["inference_seconds"].to_numpy()
    paired = tasks.merge(baseline, on=TASK_KEYS, how="left", validate="many_to_one", indicator=True)
    if (paired["_merge"] != "both").any():
        raise ValueError(f"Reference {reference} does not cover every reported task")
    paired = paired.drop(columns="_merge")
    positive = paired["reference_loss"] > 0
    ratios = np.full(len(paired), np.nan)
    np.divide(paired[loss].to_numpy(dtype=float), paired["reference_loss"].to_numpy(dtype=float),
              out=ratios, where=positive.to_numpy())
    paired["loss_ratio_to_reference"] = ratios
    paired["paired_improvement_percent"] = 100 * (1 - paired["loss_ratio_to_reference"])
    return paired


def _scaled_mean(values, aggregation):
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return None
    if aggregation == "arithmetic":
        return float(np.nanmean(array))
    if aggregation != "geometric":
        raise ValueError("scaled_aggregation must be arithmetic or geometric")
    return 0.0 if np.any(array == 0) else float(np.exp(np.nanmean(np.log(array))))


def _finite_mean(values):
    array = np.asarray(values, dtype=float)
    return float(np.nanmean(array)) if np.isfinite(array).any() else None


def _complete_sum(values):
    array = np.asarray(values, dtype=float)
    return float(array.sum()) if np.isfinite(array).all() else None


def _summary(group, loss, scaled_aggregation):
    result = {
        "tasks": len(group),
        f"finite_{loss}_tasks": int(np.isfinite(group[loss].to_numpy(dtype=float)).sum()),
        f"mean_task_{loss}": _finite_mean(group[loss]),
        "scaled_MASE": _scaled_mean(group["scaled_MASE"], scaled_aggregation),
        "total_inference_seconds": _complete_sum(group["inference_seconds"]),
        "timed_tasks": int(np.isfinite(group["inference_seconds"].to_numpy(dtype=float)).sum()),
    }
    if {"prediction_nan_values", "prediction_values"} <= set(group):
        result["prediction_nan_values"] = int(group["prediction_nan_values"].sum())
        result["prediction_values"] = int(group["prediction_values"].sum())
        result["prediction_nan_rate"] = (result["prediction_nan_values"] / result["prediction_values"]
                                         if result["prediction_values"] else None)
    for column in group:
        if column.endswith("_seconds") and not column.startswith("reference_") and column != "inference_seconds":
            result[f"total_{column}"] = _complete_sum(group[column])
    if "reference_loss" in group:
        paired = np.isfinite(group[[loss, "reference_loss"]].to_numpy(dtype=float)).all(axis=1)
        baseline = _finite_mean(group.loc[paired, "reference_loss"])
        candidate = _finite_mean(group.loc[paired, loss])
        scaled_pair = np.isfinite(group[["scaled_MASE", "reference_scaled_MASE"]]
                                  .to_numpy(dtype=float)).all(axis=1)
        scaled = _scaled_mean(group.loc[scaled_pair, "reference_scaled_MASE"], scaled_aggregation)
        candidate_scaled = _scaled_mean(group.loc[scaled_pair, "scaled_MASE"], scaled_aggregation)
        result.update(
            relative_improvement_percent=100 * (1 - candidate / baseline)
                if baseline is not None and baseline > 0 and candidate is not None else None,
            relative_scaled_improvement_percent=100 * (1 - candidate_scaled / scaled)
                if scaled is not None and scaled > 0 and candidate_scaled is not None else None,
            relative_tasks=int(paired.sum()),
            mean_paired_improvement_percent=float(group["paired_improvement_percent"].mean())
                if group["paired_improvement_percent"].notna().any() else None,
            paired_percentage_tasks=int(group["paired_improvement_percent"].notna().sum()),
            reference_mean_loss=baseline,
            reference_total_inference_seconds=_complete_sum(group["reference_inference_seconds"]),
        )
    return result


def build_performance_tables(tasks, *, reference=None, loss="MASE", scaled_aggregation="geometric"):
    """Equal task weights; reference aggregates use each model's identical tasks."""
    values = relative_task_values(tasks, reference, loss) if reference is not None else tasks.copy()
    summary = pd.DataFrame([
        {"model": model, **_summary(group, loss, scaled_aggregation)}
        for model, group in values.groupby("model", sort=False)])
    domain = pd.DataFrame([
        {"domain": name, "model": model, **_summary(group, loss, scaled_aggregation)}
        for (name, model), group in values.groupby(["domain", "model"], sort=False)])
    return values, summary, domain


def build_horizon_frequency_tables(tasks, *, loss="MASE"):
    """Cell averages are arithmetic means of task losses, not pooled cells."""
    rows, winners = [], []
    models = list(tasks["model"].drop_duplicates())
    for (frequency, horizon), cell in tasks.groupby(["frequency", "horizon_steps"], sort=False):
        groups = {model: group for model, group in cell.groupby("model", sort=False)}
        task_sets = [set(map(tuple, group[TASK_KEYS].to_numpy())) for group in groups.values()]
        comparable = set(groups) == set(models) and all(keys == task_sets[0] for keys in task_sets)
        comparable = comparable and np.isfinite(cell[loss].to_numpy(dtype=float)).all()
        scores = {}
        for model, group in groups.items():
            score = _finite_mean(group[loss])
            scores[model] = score
            row = {"frequency": frequency, "horizon_steps": int(horizon), "model": model,
                   "mean_loss": score, "tasks": len(group)}
            if "loss_ratio_to_reference" in group:
                finite = group["loss_ratio_to_reference"].dropna()
                row.update(mean_loss_ratio=float(finite.mean()) if len(finite) else None,
                           mean_paired_improvement_percent=float(group["paired_improvement_percent"].mean())
                               if len(finite) else None, relative_tasks=len(finite))
            rows.append(row)
        best = min(scores.values()) if comparable else None
        tied = [model for model, score in scores.items() if np.isclose(score, best, rtol=1e-10, atol=1e-12)] if comparable else []
        winners.append({"frequency": frequency, "horizon_steps": int(horizon),
                        "best_mean_loss": best, "winners": json.dumps(tied),
                        "comparable": comparable, "tasks_per_model": len(task_sets[0]) if comparable else None})
    return pd.DataFrame(rows), pd.DataFrame(winners)


def _text(value):
    if pd.isna(value):
        return "NA"
    return f"{value:.6g}" if isinstance(value, (float, np.floating)) else str(value)


def _tex(value):
    replacements = {"\\": r"\textbackslash{}", "_": r"\_", "%": r"\%", "&": r"\&",
                    "#": r"\#", "$": r"\$", "{": r"\{", "}": r"\}"}
    return "".join(replacements.get(char, char) for char in value)


def write_table(frame, path, *, columns=None):
    """Export a numeric CSV and readable Markdown/booktabs LaTeX table."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path.with_suffix(".csv"), index=False)
    display = frame[columns] if columns is not None else frame
    headers = [str(column).replace("_", " ") for column in display.columns]
    rows = [[_text(value) for value in row] for row in display.itertuples(index=False, name=None)]
    markdown = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    markdown.extend("| " + " | ".join(value.replace("|", r"\|") for value in row) + " |" for row in rows)
    path.with_suffix(".md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    latex = [r"\begin{tabular}{" + "l" * len(headers) + "}", r"\toprule",
             " & ".join(map(_tex, headers)) + r" \\", r"\midrule"]
    latex.extend(" & ".join(map(_tex, row)) + r" \\" for row in rows)
    latex.extend([r"\bottomrule", r"\end{tabular}"])
    path.with_suffix(".tex").write_text("\n".join(latex) + "\n", encoding="utf-8")
    return [path.with_suffix(suffix) for suffix in (".csv", ".md", ".tex")]


def write_performance_report(
    rows, destination, *, reference=None, loss="MASE", scaled_aggregation="geometric",
    domains=None, inputs=None, plots=True,
):
    """Write generic artifacts from the owning reporter's selected task means.

    Callers supply task means and scaled statistics reduced under their own
    repeat/configuration contract; this writer does not recompute those ratios.
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    tasks = prepare_tasks(rows, domains)
    values, summary, domain = build_performance_tables(
        tasks, reference=reference, loss=loss, scaled_aggregation=scaled_aggregation)
    cells, best = build_horizon_frequency_tables(values, loss=loss)
    artifacts = []
    values.to_csv(destination / "performance_tasks.csv", index=False)
    artifacts.append(destination / "performance_tasks.csv")
    columns = ["model", "tasks", f"mean_task_{loss}", "scaled_MASE", "total_inference_seconds"]
    if "prediction_nan_values" in summary:
        columns.extend(["prediction_nan_values", "prediction_values", "prediction_nan_rate"])
    if reference is not None:
        columns.extend(["relative_improvement_percent", "relative_scaled_improvement_percent",
                        "mean_paired_improvement_percent"])
    artifacts.extend(write_table(summary, destination / "performance_summary", columns=columns))
    if reference is not None:
        improvement_columns = ["model", "tasks", "relative_improvement_percent",
                               "relative_scaled_improvement_percent", "mean_paired_improvement_percent",
                               "paired_percentage_tasks"]
        artifacts.extend(write_table(summary[improvement_columns], destination / "relative_improvement"))
    artifacts.extend(write_table(domain, destination / "domain_performance",
                                 columns=["domain", *columns]))
    average = domain.pivot(index="domain", columns="model", values=f"mean_task_{loss}").reset_index()
    artifacts.extend(write_table(average, destination / "domain_average_loss"))
    scaled_average = domain.pivot(index="domain", columns="model", values="scaled_MASE").reset_index()
    artifacts.extend(write_table(scaled_average, destination / "domain_average_scaled_MASE"))
    time_columns = ["model", "tasks", "timed_tasks", *[
        column for column in summary if column.startswith("total_") and column.endswith("_seconds")]]
    artifacts.extend(write_table(summary[time_columns], destination / "time_totals"))
    if reference is not None:
        relative = domain.pivot(index="domain", columns="model", values="mean_paired_improvement_percent").reset_index()
        artifacts.extend(write_table(relative, destination / "domain_relative_improvement"))
    cells.to_csv(destination / "horizon_frequency.csv", index=False)
    best.to_csv(destination / "best_model_cells.csv", index=False)
    artifacts.extend([destination / "horizon_frequency.csv", destination / "best_model_cells.csv"])
    variants = [("scaled_MASE", "scaled_MASE", "Mean task scaled MASE")]
    if reference is not None:
        variants.append(("relative_loss", "loss_ratio_to_reference", f"Mean task {loss} / {reference}"))
    extra_grids = []
    for name, metric, label in variants:
        extra_cells, extra_best = build_horizon_frequency_tables(values, loss=metric)
        for frame, filename in [(extra_cells, f"horizon_frequency_{name}.csv"),
                                (extra_best, f"best_model_cells_{name}.csv")]:
            frame.to_csv(destination / filename, index=False)
            artifacts.append(destination / filename)
        extra_grids.append((name, label, extra_cells, extra_best))
    if plots:
        from timebench.visualization.performance import (
            _styles, plot_accuracy_time, plot_loss_grid, plot_best_model_grid, plot_domain_loss,
            plot_task_dispersion)
        styles = _styles(tasks, "model", None)
        # Undefined selected-method latency stays in tables; do not invent it.
        timed = summary[np.isfinite(summary[["total_inference_seconds", f"mean_task_{loss}"]]
                                    .to_numpy(dtype=float)).all(axis=1)]
        if len(timed):
            timed_styles = {model: styles[model] for model in timed["model"]}
            for suffix in (".png", ".pdf"):
                path = destination / ("accuracy_time" + suffix)
                plot_accuracy_time(timed, path, styles=timed_styles, time_column="total_inference_seconds",
                                   time_label="Total recorded test inference time (seconds)")
                artifacts.append(path)
        for suffix in (".png", ".pdf"):
            path = destination / ("task_mean_std" + suffix)
            if plot_task_dispersion(tasks, path, styles=styles):
                artifacts.append(path)
            path = destination / ("task_dispersion" + suffix)
            relative_tasks = tasks[tasks["model"] != "seasonal_naive"]
            if plot_task_dispersion(relative_tasks, path, relative=True, styles=styles):
                artifacts.append(path)
            path = destination / ("loss_horizon_frequency" + suffix)
            if plot_loss_grid(cells, path, value="mean_loss", label=f"Mean task {loss}"):
                artifacts.append(path)
            path = destination / ("best_model_horizon_frequency" + suffix)
            plot_best_model_grid(best, path, models=list(tasks["model"].drop_duplicates()), styles=styles,
                                 label=f"mean task {loss}")
            artifacts.append(path)
            for name, label, extra_cells, extra_best in extra_grids:
                path = destination / (f"{name}_horizon_frequency" + suffix)
                available = plot_loss_grid(extra_cells, path, value="mean_loss", label=label, parity=1)
                if available:
                    artifacts.append(path)
                path = destination / (f"best_model_{name}_horizon_frequency" + suffix)
                plot_best_model_grid(extra_best, path, models=list(tasks["model"].drop_duplicates()),
                                     styles=styles, label=label)
                artifacts.append(path)
            path = destination / ("domain_loss" + suffix)
            plot_domain_loss(domain, path, value=f"mean_task_{loss}", label=f"Mean task {loss}", styles=styles)
            artifacts.append(path)
            path = destination / ("domain_scaled_MASE" + suffix)
            plot_domain_loss(domain, path, value="scaled_MASE",
                             label=f"Task scaled MASE ({scaled_aggregation} mean)", parity=1, styles=styles)
            artifacts.append(path)
            if reference is not None:
                relative_domain = domain.copy()
                relative_domain["mean_loss_ratio"] = 1 - relative_domain["mean_paired_improvement_percent"] / 100
                path = destination / ("domain_relative_loss" + suffix)
                plot_domain_loss(relative_domain, path, value="mean_loss_ratio",
                                 label=f"Mean task {loss} / {reference}", parity=1, styles=styles)
                artifacts.append(path)
    metadata = {
        "schema_version": 1, "reference": reference, "loss": loss,
        "scaled_MASE_aggregation": scaled_aggregation,
        "task_aggregation": "producer-selected repeat/config statistics; equal task weights; no metric-cell pooling",
        "task_dispersion": "within-task population std/variance; selected repeat/config statistics are averaged, not pooled or treated as seed uncertainty",
        "dispersion_available_tasks": {
            model: int(np.isfinite(group["MASE_std"].to_numpy(dtype=float)).sum())
            if "MASE_std" in group else 0
            for model, group in tasks.groupby("model", sort=False)
        },
        "relative_variance_available_tasks": {
            model: int((np.isfinite(group[["MASE_variance", "seasonal_MASE_variance"]]
                                   .to_numpy(dtype=float)).all(axis=1)
                        & (group["seasonal_MASE_variance"].to_numpy(dtype=float) > 0)).sum())
            if {"MASE_variance", "seasonal_MASE_variance"}.issubset(group.columns) else 0
            for model, group in tasks.groupby("model", sort=False)
            if model != "seasonal_naive"
        },
        "undefined_dispersion": "missing population statistics or non-positive Seasonal variance stay unavailable; no zero substitutions or pseudocounts",
        "relative_improvement_percent": "100*(1-mean_model_loss/mean_reference_loss) on identical tasks",
        "mean_paired_improvement_percent": "mean of 100*(1-model_task_loss/reference_task_loss); zero reference undefined",
        "heatmap_axes": {"x": "forecast horizon in observations", "y": "literal task sampling frequency"},
        "heatmap_values": "arithmetic task means; relative heatmaps average paired task ratios",
        "best_model": "minimum cell mean on identical task sets; ties retained at rtol=1e-10, atol=1e-12; otherwise cell unavailable",
        "domain_grouping": "explicit dataset labels; unmapped datasets Unclassified",
        "domain_aggregation": f"equal task weights: arithmetic raw-loss means and {scaled_aggregation} scaled-MASE means",
        "unclassified_datasets": sorted(tasks.loc[tasks["domain"] == "Unclassified", "dataset"].unique()),
        "timing": "sum of producer-reduced recorded task timings; missing stays undefined; preprocessing/retrieval columns are not added to inference (may overlap); not end-to-end job walltime",
        "inputs": inputs,
        "artifacts": [path.name for path in artifacts],
    }
    manifest = destination / "performance_report_manifest.json"
    manifest.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return [*artifacts, manifest]
