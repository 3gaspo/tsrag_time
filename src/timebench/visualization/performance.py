"""Artifact-only performance plots shared by TIME projects."""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")


def _styles(frame, model_column, styles):
    if styles is not None:
        return styles
    colors = ("#005bbb", "#ffa02f", "#669966", "#8f4b99", "#b44a40", "#555555")
    markers = ("o", "s", "^", "D", "v", "P")
    models = list(frame[model_column].drop_duplicates())
    if len(models) > len(colors):
        import matplotlib.pyplot as plt
        palette = plt.get_cmap("tab20" if len(models) <= 20 else "hsv")
        colors = [palette(index if len(models) <= 20 else index / len(models))
                  for index in range(len(models))]
    return {
        model: (str(model), colors[index % len(colors)], markers[index % len(markers)])
        for index, model in enumerate(models)
    }


def plot_accuracy_time(
    summary: pd.DataFrame,
    path: Path,
    *,
    styles=None,
    model_column="model",
    score_column="scaled_MASE",
    time_column="inference_seconds",
    score_label="Scaled MASE",
    time_label="Total recorded forecast-loop time (seconds)",
):
    """Plot one precomputed aggregate score/time pair per model or method.

    No aggregation or normalization is performed here. Supply a unique display
    label column for distinct configurations. A styles mapping optionally
    selects models and assigns (label, color, marker); absent selections fail.
    Missing timing must remain missing, never be replaced by zero.
    """
    import matplotlib.pyplot as plt

    series = _styles(summary, model_column, styles)
    if not series:
        raise ValueError("No accuracy/timing summaries to plot")
    many_labels = len(series) > 8
    figure, axis = plt.subplots(figsize=(10.5 if many_labels else 8.5, 4.2 if many_labels else 3.6),
                                constrained_layout=True)
    try:
        for model, (label, color, marker) in series.items():
            selected = summary[summary[model_column] == model]
            if len(selected) != 1:
                raise ValueError(f"Expected one aggregate summary for {model}")
            row = selected.iloc[0]
            if "state" in summary and row["state"] != "completed":
                raise ValueError(f"Incomplete aggregate summary for {model}")
            score, seconds = float(row[score_column]), float(row[time_column])
            if not np.isfinite(score) or not np.isfinite(seconds) or seconds < 0:
                raise ValueError(f"Invalid accuracy/timing summary for {model}")
            axis.scatter(seconds, score, s=110, marker=marker, color=color,
                         label=label if many_labels else None)
            if not many_labels:
                axis.annotate(label, (seconds, score), xytext=(8, 8),
                              textcoords="offset points", color=color)
        axis.set_xlabel(time_label)
        axis.set_ylabel(score_label)
        axis.margins(x=0.18, y=0.25)
        axis.grid(alpha=0.25)
        if many_labels:
            axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1), ncol=1,
                        fontsize=8, frameon=True)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=220, bbox_inches="tight")
        if Path(path).suffix.lower() == ".png":
            figure.savefig(Path(path).with_suffix(".pdf"), bbox_inches="tight")
    finally:
        plt.close(figure)


def _frequency_order(values):
    """Order literal sampling frequencies without equating calendar durations."""
    import re

    units = {"s": 0, "sec": 0, "t": 1, "min": 1, "h": 2, "d": 3,
             "b": 3, "w": 4, "m": 5, "ms": 5, "me": 5, "q": 6,
             "qs": 6, "y": 7, "a": 7, "ys": 7}
    def key(value):
        match = re.fullmatch(r"(\d*)([a-z]+)(?:-.*)?", str(value).lower())
        if not match:
            return (99, 0, str(value))
        return (units.get(match[2], 99), int(match[1] or 1), str(value))
    return sorted(values, key=key)


def plot_task_dispersion(tasks, path, *, relative=False, styles=None):
    """One point per selected model/task; dispersion is within-task, not seed error."""
    import matplotlib.pyplot as plt

    columns = ["MASE", "MASE_std"]
    if relative:
        columns = ["scaled_MASE", "MASE_variance", "seasonal_MASE_variance"]
    if any(column not in tasks for column in columns):
        return False
    selected = tasks.copy()
    if relative:
        denominator = selected["seasonal_MASE_variance"].to_numpy(dtype=float)
        ratios = np.full(len(selected), np.nan)
        np.divide(selected["MASE_variance"].to_numpy(dtype=float), denominator,
                  out=ratios, where=np.isfinite(denominator) & (denominator > 0))
        selected["dispersion"] = ratios
        selected["mean"] = selected["scaled_MASE"]
    else:
        selected["mean"] = selected["MASE"]
        selected["dispersion"] = selected["MASE_std"]
    finite = np.isfinite(selected[["mean", "dispersion"]].to_numpy(dtype=float)).all(axis=1)
    selected = selected[finite]
    if selected.empty:
        return False
    figure, axis = plt.subplots(figsize=(8.5, 4.5), constrained_layout=True)
    try:
        for model, (label, color, marker) in _styles(selected, "model", styles).items():
            group = selected[selected["model"] == model]
            if group.empty:
                continue
            axis.scatter(group["mean"], group["dispersion"], color=color, marker=marker,
                         s=32, alpha=0.7, label=f"{label} ({len(group)} tasks)")
        axis.set_xlabel("Mean MASE / Seasonal mean MASE" if relative else "Mean task MASE")
        axis.set_ylabel("Variance / Seasonal variance" if relative else "Within-task MASE standard deviation")
        if relative:
            if (selected[["mean", "dispersion"]].to_numpy(dtype=float) > 0).all():
                axis.set_xscale("log")
                axis.set_yscale("log")
            else:
                axis.set_xscale("symlog", linthresh=0.01)
                axis.set_yscale("symlog", linthresh=0.01)
            axis.axvline(1, color="#555555", linestyle="--", linewidth=1)
            axis.axhline(1, color="#555555", linestyle="--", linewidth=1)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=9)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=220, bbox_inches="tight")
        if Path(path).suffix.lower() == ".png":
            figure.savefig(Path(path).with_suffix(".pdf"), bbox_inches="tight")
    finally:
        plt.close(figure)
    return True


def _grid_labels(axis, frequencies, horizons):
    axis.set_xticks(range(len(horizons)), labels=horizons, rotation=45, ha="right")
    axis.set_yticks(range(len(frequencies)), labels=frequencies)
    axis.set_xlabel("Forecast horizon (observations)")
    axis.set_ylabel("Task sampling frequency")


def plot_loss_grid(cells, path, *, value="mean_loss", label="Mean task MASE", parity=None):
    """One panel/model; common scale; missing cells are gray, not zero."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, TwoSlopeNorm

    models = list(cells["model"].drop_duplicates())
    frequencies = _frequency_order(cells["frequency"].unique())
    horizons = sorted(cells["horizon_steps"].unique())
    finite = cells[value].to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return False
    low, high = float(finite.min()), float(finite.max())
    if parity is None:
        norm = Normalize(vmin=min(0, low), vmax=max(high, 1e-12))
        cmap = plt.get_cmap("YlOrRd").copy()
    else:
        span = max(abs(low - parity), abs(high - parity), 1e-3)
        norm = TwoSlopeNorm(vmin=parity - span, vcenter=parity, vmax=parity + span)
        cmap = plt.get_cmap("RdYlGn_r").copy()
    cmap.set_bad("#dddddd")
    columns = min(3, len(models))
    rows = int(np.ceil(len(models) / columns))
    figure, axes = plt.subplots(rows, columns, squeeze=False,
                                figsize=(max(6, columns * 4.5), max(3.5, rows * 3.5)),
                                constrained_layout=True)
    try:
        for axis, model in zip(axes.flat, models):
            selected = cells[cells["model"] == model]
            grid = selected.pivot(index="frequency", columns="horizon_steps", values=value)
            grid = grid.reindex(index=frequencies, columns=horizons).to_numpy(dtype=float)
            artist = axis.imshow(np.ma.masked_invalid(grid), aspect="auto", cmap=cmap, norm=norm)
            axis.set_title(str(model), fontsize=10)
            _grid_labels(axis, frequencies, horizons)
            if grid.size <= 80:
                for (i, j), number in np.ndenumerate(grid):
                    if np.isfinite(number):
                        axis.text(j, i, f"{number:.2g}", ha="center", va="center", fontsize=7)
        for axis in list(axes.flat)[len(models):]:
            axis.set_visible(False)
        figure.colorbar(artist, ax=list(axes.flat)[:len(models)], label=label, shrink=0.85)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=220)
        if Path(path).suffix.lower() == ".png":
            figure.savefig(Path(path).with_suffix(".pdf"), bbox_inches="tight")
    finally:
        plt.close(figure)
    return True


def plot_best_model_grid(cells, path, *, models, styles=None, label="Mean task MASE"):
    """Color matched cells by winner; split tied cells; unmatched cells gray."""
    import json
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle

    series = _styles(pd.DataFrame({"model": models}), "model", styles)
    frequencies = _frequency_order(cells["frequency"].unique())
    horizons = sorted(cells["horizon_steps"].unique())
    figure, axis = plt.subplots(figsize=(max(7, len(horizons) * 0.55),
                                         max(3.5, len(frequencies) * 0.5)),
                                constrained_layout=True)
    try:
        for row in cells.itertuples(index=False):
            x, y = horizons.index(row.horizon_steps), frequencies.index(row.frequency)
            winners = json.loads(row.winners)
            if not winners:
                axis.add_patch(Rectangle((x - .5, y - .5), 1, 1,
                                         facecolor="#dddddd", edgecolor="white"))
            for index, winner in enumerate(winners):
                axis.add_patch(Rectangle((x - .5 + index / len(winners), y - .5),
                                         1 / len(winners), 1, facecolor=series[winner][1],
                                         edgecolor="white"))
            if len(winners) > 1:
                axis.text(x, y, "=", ha="center", va="center")
        axis.set_facecolor("#dddddd")
        axis.set_xlim(-.5, len(horizons) - .5)
        axis.set_ylim(len(frequencies) - .5, -.5)
        _grid_labels(axis, frequencies, horizons)
        axis.set_title(f"Best {label} on identical task sets; split colors = ties")
        handles = [Patch(color=color, label=label) for label, color, _ in series.values()]
        handles.append(Patch(color="#dddddd", label="Unavailable / unmatched task coverage"))
        axis.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=220)
        if Path(path).suffix.lower() == ".png":
            figure.savefig(Path(path).with_suffix(".pdf"), bbox_inches="tight")
    finally:
        plt.close(figure)


def plot_domain_loss(domain, path, *, value="mean_task_MASE", label="Mean task MASE",
                     parity=None, styles=None):
    """Grouped per-domain mean-loss bars, not a distribution of individual losses."""
    import matplotlib.pyplot as plt

    models = list(domain["model"].drop_duplicates())
    domains = list(domain["domain"].drop_duplicates())
    series = _styles(domain, "model", styles)
    figure, axis = plt.subplots(figsize=(max(8, len(domains) * 1.1), 4.8),
                                constrained_layout=True)
    try:
        width = .8 / len(models)
        for index, model in enumerate(models):
            selected = domain[domain["model"] == model].set_index("domain")[value].reindex(domains)
            display, color, _ = series[model]
            axis.bar(np.arange(len(domains)) - .4 + width * (index + .5),
                     selected.to_numpy(dtype=float), width=width, label=display, color=color)
        if parity is not None:
            axis.axhline(parity, linestyle=":", color="#555555", linewidth=1)
        axis.set_xticks(range(len(domains)), labels=domains, rotation=35, ha="right")
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=.25)
        axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=220)
        if Path(path).suffix.lower() == ".png":
            figure.savefig(Path(path).with_suffix(".pdf"), bbox_inches="tight")
    finally:
        plt.close(figure)



def plot_feature_scatter(
    data: pd.DataFrame,
    path: Path,
    *,
    feature="temporal_heterogeneity",
    styles=None,
    model_column="model",
    score_column="scaled_MASE",
    feature_label=None,
    score_label="Scaled MASE (geometric mean over configuration horizons)",
    parity=1,
):
    """Plot precomputed configuration errors against one selected feature.

    The caller owns task selection, feature joins, horizon aggregation and
    baseline scaling. No tasks are silently dropped. Set parity=None for an
    unscaled metric, and describe the supplied score through score_label.
    """
    import matplotlib.pyplot as plt

    series = _styles(data, model_column, styles)
    if not series:
        raise ValueError("No feature/performance data to plot")
    figure, axis = plt.subplots(figsize=(8.5, 3.7), constrained_layout=True)
    try:
        for model, (label, color, marker) in series.items():
            frame = data[data[model_column] == model]
            values = frame[[feature, score_column]].to_numpy(dtype=float)
            duplicates = "dataset_id" in frame and frame["dataset_id"].duplicated().any()
            if frame.empty or duplicates or not np.isfinite(values).all():
                raise ValueError(f"Invalid configuration feature data for {model}")
            axis.scatter(values[:, 0], values[:, 1],
                         label=f"{label} ({len(frame)} configurations)",
                         color=color, marker=marker, s=38, alpha=0.75)
        if parity is not None:
            axis.axhline(parity, linestyle=":", color="#555555", linewidth=1)
        axis.set_xlabel(feature_label or feature.replace("_", " ").capitalize())
        axis.set_ylabel(score_label)
        axis.legend(fontsize=10)
        axis.grid(alpha=0.25)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=220)
        if Path(path).suffix.lower() == ".png":
            figure.savefig(Path(path).with_suffix(".pdf"), bbox_inches="tight")
    finally:
        plt.close(figure)
