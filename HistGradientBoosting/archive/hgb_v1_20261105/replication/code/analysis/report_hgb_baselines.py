"""Build an immutable, self-contained HGB evaluation report without fitting models.

The public interface is ``build_report(version_root: Path) -> Path``. The runner
supplies frozen_config.json and development/backtest/horizon/final summaries.
All comparison rows are independently checked against the original RF/Bagging
prediction files before any comparative report is written.
"""
import argparse
import base64
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
import sys
from urllib.parse import quote
import uuid

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from analysis.run_tree_baselines import grouped_metrics, point_metrics, sha256, write_json

TRACKS = ("off_market", "list_aware")
PHASES = ("development", "backtest", "horizon")
METRICS = ("n", "mdape", "mape", "median_signed_error")
IDENTITY = ("ml_num", "month", "training_cutoff", "sold_price")
PARAMETERS = ("encoding", "target_mode", "max_iter", "max_leaf_nodes", "min_samples_leaf")
PHASE_TITLES = {"development": "Development selection", "backtest": "Reused rolling backtest",
                "horizon": "Reused horizon backtest"}


def _json(path):
    return json.loads(Path(path).read_text())


def _configuration(result):
    """Normalize report metadata, without changing the model configuration."""
    config = result.get("model_config", {})
    values = dict(config) if isinstance(config, dict) else {}
    for name in ("options", "estimator_params"):
        if isinstance(config.get(name), dict):
            values.update(config[name])
    for key in PARAMETERS:
        if key in result:
            values[key] = result[key]
    if "iterations_evaluated" in result:
        values["max_iter"] = result["iterations_evaluated"]
    values["target_mode"] = values.get("target_mode", values.get("target"))
    values["target_mode"] = {"natural_log_sold_price": "log_price", "natural_log_sold_to_list_ratio": "log_ratio"}.get(
        values["target_mode"], values["target_mode"])
    return {key: values.get(key) for key in PARAMETERS}


def _metrics(result):
    metrics = result.get("metrics", result)
    return metrics.get("pooled", metrics)


def _matches(result, selected):
    actual, expected = _configuration(result), _configuration(selected)
    return all(expected[key] is None or actual[key] == expected[key] for key in PARAMETERS)


def _selected_result(summary, frozen, track, phase):
    selected = frozen["tracks"][track]
    candidates = [row for row in summary["results"] if row["track"] == track]
    if phase == "development":
        candidates = [row for row in candidates if row["experiment_id"] == selected["selected_experiment_id"]]
    candidates = [row for row in candidates if _matches(row, selected)]
    if not candidates:
        raise ValueError(f"No {phase} result matches the frozen {track} configuration")
    signatures = {json.dumps({"id": row["experiment_id"], "parameters": _configuration(row),
                              "metrics": _metrics(row)}, sort_keys=True) for row in candidates}
    if len(signatures) != 1:
        raise ValueError(f"Ambiguous selected {phase} result for {track}")
    return candidates[0]


def _prediction_path(version_root, result):
    path = version_root / "experiments" / result["experiment_id"] / "predictions.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _read_predictions(path, result, phase):
    frame = pd.read_csv(path, dtype={"ml_num": str}, float_precision="round_trip")
    parameters = _configuration(result)
    if "track" in frame:
        frame = frame.loc[frame["track"] == result["track"]]
    for key in ("encoding", "target_mode"):
        if key in frame and parameters[key] is not None:
            frame = frame.loc[frame[key] == parameters[key]]
    if "iterations_evaluated" in frame and parameters["max_iter"] is not None:
        frame = frame.loc[frame["iterations_evaluated"] == parameters["max_iter"]]
    required = [*IDENTITY, "predicted_price_raw", "house_type_name", "community"]
    missing = set(required).difference(frame.columns)
    if missing or frame.empty:
        raise ValueError(f"Missing fields {sorted(missing)} or no matching rows in {path}")
    if frame[list(IDENTITY)].isna().any().any() or frame.duplicated(["ml_num", "month"]).any():
        raise ValueError(f"Invalid or duplicated prediction identities in {path}")
    frame = frame.copy()
    frame["month"] = frame["month"].astype(str)
    frame["training_cutoff"] = pd.to_datetime(frame["training_cutoff"], utc=True, format="mixed")
    month_start = pd.to_datetime(frame["month"] + "-01", utc=True)
    if phase == "horizon":
        if not (frame["training_cutoff"] == pd.Timestamp("2026-02-01", tz="UTC")).all():
            raise ValueError(f"Horizon predictions do not use the fixed February cutoff: {path}")
        if "horizon_months" not in frame:
            start = pd.Period("2026-02", freq="M").ordinal
            frame["horizon_months"] = [pd.Period(month, freq="M").ordinal - start + 1 for month in frame["month"]]
    elif not (frame["training_cutoff"] == month_start).all():
        raise ValueError(f"Monthly predictions do not use sale-month cutoffs: {path}")
    measured = point_metrics(frame["sold_price"], frame["predicted_price_raw"])
    supplied = _metrics(result)
    for key in METRICS:
        if key in supplied and not np.isclose(float(supplied[key]), measured[key], rtol=1e-9, atol=1e-9):
            raise ValueError(f"Summary {key} disagrees with predictions: {path}")
    return frame


def _check_identity(actual, reference, actual_path, reference_path):
    left = actual[list(IDENTITY)].sort_values(["ml_num", "month"]).reset_index(drop=True)
    right = reference[list(IDENTITY)].sort_values(["ml_num", "month"]).reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(left, right, check_dtype=False, check_exact=True)
    except AssertionError as error:
        raise ValueError(f"Incomparable rows, truth or cutoffs: {actual_path} versus {reference_path}") from error
    return {"n": len(left), "matched_fields": list(IDENTITY), "status": "passed",
            "hgb_predictions": str(actual_path), "hgb_sha256": sha256(actual_path),
            "reference_predictions": str(reference_path), "reference_sha256": sha256(reference_path)}


def _environment_evidence(directory, phase, track, model):
    path = directory / "environment.json"
    value = _json(path)
    return {"phase": phase, "track": track, "model": model, "environment_path": str(path),
            "environment_sha256": sha256(path), "packages": value.get("packages"),
            "thread_settings": value.get("thread_settings"), "data_identity": value.get("data_identity"),
            "source_sha256": value.get("source_sha256")}


def _old_results(tree_root, phase, frozen):
    if phase == "development":
        rows = [row for track in TRACKS for row in frozen["tracks"][track]["comparison"]]
    else:
        old_phase = "holdout" if phase == "backtest" else "horizon"
        frozen_hash = sha256(tree_root / "frozen_config.json")
        paths = [path for path in sorted(tree_root.glob(f"{old_phase}_summary_*.json"))
                 if _json(path).get("frozen_config_sha256") == frozen_hash]
        if not paths:
            raise FileNotFoundError(f"No original {old_phase} manifest matches the original frozen configuration")
        rows = _json(paths[-1])["results"]
    selected = [row for row in rows if row.get("family") in ("rf", "bagging")]
    for track in TRACKS:
        if {row["family"] for row in selected if row["track"] == track} != {"rf", "bagging"}:
            raise ValueError(f"Both original RF and Bagging references are required for {phase}, {track}")
    return selected


def _tag(result, phase, *, selected=False, model="hgb"):
    return {"phase": phase, "track": result["track"], "model": model,
            "experiment_id": result["experiment_id"], "stage": result.get("stage"),
            **_configuration(result), "selected": selected}


def _cohort_diagnostics(diags, tag):
    rows, flat_rows, runtime = [], [], []

    def flatten(value, prefix, base):
        if isinstance(value, dict):
            for key, item in value.items():
                flatten(item, f"{prefix}.{key}" if prefix else str(key), base)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                flatten(item, f"{prefix}[{index}]", base)
        else:
            flat_rows.append({**base, "field": prefix, "value": value})

    if isinstance(diags, dict):
        diags = diags.get("folds", diags.get("diagnostics", [diags]))
    for index, diagnostic in enumerate(diags):
        validation = diagnostic.get("validation", [])
        if isinstance(validation, dict):
            validation = [validation]
        month = diagnostic.get("month", validation[0].get("month") if validation else None)
        cutoff = diagnostic.get("training_cutoff")
        physical_id = diagnostic.get("physical_fit_id", f"{tag['experiment_id']}::fold{index}")
        common = {**tag, "fold": index, "physical_fit_id": physical_id, "month": month,
                  "training_cutoff": cutoff,
                  "iterations_trained": diagnostic.get("iterations_trained", diagnostic.get("n_iter_")),
                  "iterations_evaluated": diagnostic.get("iterations_evaluated", tag.get("max_iter"))}
        flatten(diagnostic, "", common)
        timing = {key: value for key, value in diagnostic.items()
                  if ("seconds" in key or key in ("n_train", "n_recent_train")) and np.isscalar(value)}
        runtime.append({**common, **timing})
        for cohort, metric_key, mse_key in (
            ("full_history_training", "training_price_error", "training_log_mse"),
            ("recent_12m_training", "recent_training_price_error", "recent_training_log_mse"),
        ):
            metric = diagnostic.get(metric_key)
            if isinstance(metric, dict):
                rows.append({**common, "cohort": cohort,
                             **{key: metric.get(key) for key in METRICS},
                             "log_price_mse": diagnostic.get(mse_key)})
        for value in validation:
            metric = value.get("metrics", value)
            rows.append({**common, "cohort": "future_month", "month": value.get("month", month),
                         **{key: metric.get(key) for key in METRICS},
                         "log_price_mse": value.get("log_mse", value.get("log_price_mse"))})
    return rows, flat_rows, runtime


def _unique_runtime(rows):
    """Checkpoint aliases repeat physical-fit times; count the work only once."""
    unique = {}
    for row in rows:
        key = row["physical_fit_id"]
        if key in unique:
            for field in ("fit_seconds", "preprocessing_seconds", "estimator_fit_seconds"):
                left, right = unique[key].get(field), row.get(field)
                if left is not None and right is not None and not np.isclose(left, right, rtol=1e-9, atol=1e-9):
                    raise ValueError(f"Inconsistent {field} for shared physical fit {key}")
            unique[key]["_experiments"].add(row["experiment_id"])
            unique[key]["_checkpoints"].add(row.get("iterations_evaluated"))
            unique[key]["selected"] = unique[key]["selected"] or row["selected"]
            continue
        unique[key] = {**row, "_experiments": {row["experiment_id"]},
                       "_checkpoints": {row.get("iterations_evaluated")}}
    values = []
    for row in unique.values():
        experiments = row.pop("_experiments")
        checkpoints = sorted(value for value in row.pop("_checkpoints") if value is not None)
        row["experiment_id"] = " | ".join(sorted(experiments))
        row["evaluated_checkpoints"] = ", ".join(str(value) for value in checkpoints)
        if len(checkpoints) > 1:
            row["max_iter"] = row["iterations_evaluated"] = None
        values.append(row)
    return pd.DataFrame(values)


def _table(frame, percent_columns=("mdape", "mape", "median_signed_error")):
    if frame.empty:
        return "<p class='muted'>No applicable records.</p>"
    value = frame.copy()
    for column in value:
        if column in percent_columns:
            value[column] = value[column].map(lambda item: "N/A" if pd.isna(item) else f"{float(item):.3f}%")
        elif pd.api.types.is_float_dtype(value[column]):
            value[column] = value[column].map(lambda item: "N/A" if pd.isna(item) else f"{float(item):.6g}")
    return "<div class='scroll'>" + value.to_html(index=False, border=0, escape=True, classes="data", na_rep="N/A") + "</div>"


def _plot_style(ax):
    ax.grid(axis="y", color="#e2e7ed", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelrotation=25)
    ax.set_ylabel("MdAPE (%)")


def _finish_y_limits(axes):
    """Give markers headroom after all shared-axis panels have been populated."""
    for ax in np.asarray(axes).flat:
        maximum = 0.0
        for sibling in ax.get_shared_y_axes().get_siblings(ax):
            for line in sibling.lines:
                if line.get_label() == "Frozen iteration":
                    continue
                values = np.asarray(line.get_ydata(), dtype=float)
                finite = values[np.isfinite(values)]
                if finite.size:
                    maximum = max(maximum, float(finite.max()))
        ax.set_ylim(0, max(1.0, 1.08 * maximum))


def _monthly(frame):
    return [{"month": str(month), **point_metrics(group["sold_price"], group["predicted_price_raw"])}
            for month, group in frame.groupby("month", sort=True)]


def _charts(destination, frames, cohort_metrics, candidates, frozen):
    images = []
    colors = {"hgb": "#117e86", "rf": "#8b97a8", "bagging": "#d08b2e"}
    names = {"hgb": "Selected HGB", "rf": "RF reference", "bagging": "Bagging reference"}
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey="row", sharex="col")
    for row, track in enumerate(TRACKS):
        for column, phase in enumerate(PHASES):
            ax = axes[row, column]
            for model in ("rf", "bagging", "hgb"):
                values = pd.DataFrame(_monthly(frames[(phase, track, model)]))
                ax.plot(values["month"], values["mdape"], "o-", color=colors[model], label=names[model], linewidth=2)
            ax.set_title(f"{track}\n{PHASE_TITLES[phase]}", loc="left", fontsize=11)
            _plot_style(ax)
            ax.legend(frameon=False, fontsize=8)
    _finish_y_limits(axes)
    fig.suptitle("HGB vs. original references on verified identical homes and cutoffs", fontsize=15)
    fig.tight_layout()
    path = destination / "temporal_errors.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    images.append(("Temporal evaluation on the same rows", path))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, track in zip(axes, TRACKS):
        for model in ("rf", "bagging", "hgb"):
            frame = frames[("backtest", track, model)].copy()
            frame["decile"] = pd.qcut(frame["sold_price"], 10, labels=False, duplicates="drop") + 1
            values = pd.DataFrame([{"decile": decile, **point_metrics(group["sold_price"], group["predicted_price_raw"])}
                                   for decile, group in frame.groupby("decile", observed=True)])
            ax.plot(values["decile"], values["mdape"], "o-", color=colors[model], label=names[model], linewidth=2)
        ax.set_title(track, loc="left")
        ax.set_xlabel("Observed sold-price decile (1 = cheapest)")
        _plot_style(ax)
        ax.tick_params(axis="x", labelrotation=0)
        ax.legend(frameon=False, fontsize=8)
    _finish_y_limits(axes)
    fig.suptitle("Reused rolling backtest: error across observed price groups")
    fig.tight_layout()
    path = destination / "price_decile_errors.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    images.append(("Error by observed sold-price decile", path))

    selected = cohort_metrics.loc[(cohort_metrics["selected"] == True) & (cohort_metrics["model"] == "hgb")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharey="row", sharex="col")
    cohort_labels = {"full_history_training": "Full-history training", "recent_12m_training": "Recent-12m training",
                     "future_month": "Future month"}
    cohort_colors = {"full_history_training": "#76909c", "recent_12m_training": "#a582b7", "future_month": "#117e86"}
    for row, track in enumerate(TRACKS):
        for column, phase in enumerate(("development", "backtest")):
            ax = axes[row, column]
            subset = selected.loc[(selected["track"] == track) & (selected["phase"] == phase)]
            for cohort in cohort_labels:
                points = subset.loc[subset["cohort"] == cohort].sort_values("month")
                if not points.empty and points["mdape"].notna().any():
                    ax.plot(points["month"], points["mdape"], "o-", color=cohort_colors[cohort], label=cohort_labels[cohort])
            ax.set_title(f"{track} | {PHASE_TITLES[phase]}", loc="left", fontsize=11)
            _plot_style(ax)
            handles, _ = ax.get_legend_handles_labels()
            if handles:
                ax.legend(frameon=False, fontsize=8)
    _finish_y_limits(axes)
    fig.suptitle("Selected HGB: training cohorts and future errors")
    fig.tight_layout()
    path = destination / "training_cohort_errors.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    images.append(("Training cohort diagnostics", path))

    if not candidates.empty:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
        any_curve = False
        for ax, track in zip(axes, TRACKS):
            values = candidates.loc[candidates["track"] == track].copy()
            keys = ["encoding", "target_mode", "max_leaf_nodes", "min_samples_leaf"]
            for group, points in values.groupby(keys, dropna=False):
                points = points.drop_duplicates("max_iter").sort_values("max_iter")
                if len(points) < 2:
                    continue
                any_curve = True
                encoding, target, leaves, leaf_size = group
                label = f"{encoding}, leaves={leaves}, min leaf={leaf_size}, {target}"
                ax.plot(points["max_iter"], points["mdape"], "o-", label=label)
            chosen = _configuration(frozen["tracks"][track])
            if chosen["max_iter"] is not None:
                ax.axvline(chosen["max_iter"], color="#5c6d7a", linestyle="--", linewidth=1, label="Frozen iteration")
            ax.set_title(track, loc="left")
            ax.set_xlabel("Evaluated iterations")
            ax.set_xscale("log", base=2)
            ax.set_xticks([100, 200, 400, 800], ["100", "200", "400", "800"])
            _plot_style(ax)
            ax.tick_params(axis="x", labelrotation=0)
            ax.legend(frameon=False, fontsize=7)
        if any_curve:
            _finish_y_limits(axes)
            fig.suptitle("Development checkpoint scores: hyperparameter selection evidence")
            fig.tight_layout()
            path = destination / "iteration_selection.png"
            fig.savefig(path, dpi=170)
            images.append(("Iteration selection on development data", path))
        plt.close(fig)
    return images


def _image_html(title, path):
    encoded = base64.b64encode(path.read_bytes()).decode()
    return f"<figure><img alt='{html.escape(title, quote=True)}' src='data:image/png;base64,{encoded}'><figcaption>{html.escape(title)}</figcaption></figure>"


def _relative_link(path, destination, label):
    path = Path(path)
    if not path.exists():
        return f"{html.escape(label)}: <span class='muted'>not found at report generation</span>"
    href = quote(os.path.relpath(path, destination), safe="/._-")
    return f"<a href='{href}'>{html.escape(label)}</a>"


def build_report(version_root: Path) -> Path:
    """Validate frozen experiments and write a new HTML/CSV/PNG report bundle."""
    version_root = Path(version_root).resolve()
    frozen_path = version_root / "frozen_config.json"
    frozen = _json(frozen_path)
    frozen_hash = sha256(frozen_path)
    summaries = {phase: _json(version_root / f"{phase}_summary.json") for phase in (*PHASES, "final")}
    for phase, summary in summaries.items():
        if summary.get("frozen_config_sha256") not in (None, frozen_hash):
            raise ValueError(f"{phase} summary belongs to a different frozen configuration")
        if phase in ("backtest", "horizon", "final") and summary.get("frozen_config_sha256") != frozen_hash:
            raise ValueError(f"{phase} summary must identify the frozen configuration")
    for track in TRACKS:
        if track not in frozen["tracks"]:
            raise ValueError(f"Missing frozen track: {track}")

    tree_root = ROOT / "outputs" / "tree_baselines"
    tree_frozen = _json(tree_root / "frozen_config.json")
    frames, selected_results, comparisons, checks, all_groups = {}, {}, [], [], []
    reference_cohorts = []
    environments, new_environments = [], {}
    for phase in PHASES:
        for track in TRACKS:
            result = _selected_result(summaries[phase], frozen, track, phase)
            path = _prediction_path(version_root, result)
            frame = _read_predictions(path, result, phase)
            evidence = _environment_evidence(path.parent, phase, track, "hgb")
            environments.append(evidence)
            new_environments[(phase, track)] = evidence
            frames[(phase, track, "hgb")] = frame
            selected_results[(phase, track)] = result
            metric = point_metrics(frame["sold_price"], frame["predicted_price_raw"])
            metric["log_price_mse"] = float(np.mean((np.log(frame["predicted_price_raw"]) - np.log(frame["sold_price"])) ** 2))
            comparisons.append({**_tag(result, phase, selected=True), **metric, "old_frozen_reference": False})
        for old in _old_results(tree_root, phase, tree_frozen):
            old = {**old, "encoding": "onehot", "target_mode": "log_price"}
            track, family = old["track"], old["family"]
            path = tree_root / old["experiment_id"] / "predictions.csv"
            frame = _read_predictions(path, old, phase)
            new = selected_results[(phase, track)]
            check = _check_identity(frames[(phase, track, "hgb")], frame,
                                    _prediction_path(version_root, new), path)
            evidence = _environment_evidence(path.parent, phase, track, family)
            environments.append(evidence)
            new_identity = new_environments[(phase, track)].get("data_identity") or {}
            old_identity = evidence.get("data_identity") or {}
            new_hash = new_identity.get("listings", {}).get("sha256")
            old_hash = old_identity.get("listings", {}).get("sha256")
            if new_hash and old_hash and new_hash != old_hash:
                raise ValueError(f"Published training listings differ for {phase}, {track}, {family}")
            check["published_training_data_sha256_match"] = bool(new_hash == old_hash) if new_hash and old_hash else None
            checks.append({"phase": phase, "track": track, "reference_family": family, **check})
            frames[(phase, track, family)] = frame
            metric = point_metrics(frame["sold_price"], frame["predicted_price_raw"])
            metric["log_price_mse"] = float(np.mean((np.log(frame["predicted_price_raw"]) - np.log(frame["sold_price"])) ** 2))
            comparisons.append({**_tag(old, phase, model=family), **metric,
                                "old_frozen_reference": family == tree_frozen["tracks"][track]["family"]})
            diagnostic_path = tree_root / old["experiment_id"] / "model_diagnostics.json"
            rows, _, _ = _cohort_diagnostics(_json(diagnostic_path), _tag(old, phase, model=family))
            for row in rows:
                if row["cohort"] == "future_month" and row.get("log_price_mse") is None:
                    month = frame.loc[frame["month"] == row["month"]]
                    if not month.empty:
                        row["log_price_mse"] = float(np.mean((np.log(month["predicted_price_raw"]) - np.log(month["sold_price"])) ** 2))
            reference_cohorts.extend(rows)
        for track in TRACKS:
            for model in ("hgb", "rf", "bagging"):
                groups = grouped_metrics(frames[(phase, track, model)])
                all_groups.append(groups.assign(phase=phase, track=track, model=model))

    candidates = []
    cohort_rows, flat_rows, timing_rows, checkpoint_rows = reference_cohorts, [], [], []
    seen = set()
    for phase in PHASES:
        selected_ids = {selected_results[(phase, track)]["experiment_id"] for track in TRACKS}
        for result in summaries[phase]["results"]:
            tag = _tag(result, phase, selected=result["experiment_id"] in selected_ids)
            if phase == "development":
                candidates.append({**tag, **{key: _metrics(result).get(key) for key in METRICS}})
            key = (phase, result["experiment_id"])
            if key in seen:
                continue
            seen.add(key)
            directory = version_root / "experiments" / result["experiment_id"]
            rows, flat, timing = _cohort_diagnostics(_json(directory / "model_diagnostics.json"), tag)
            cohort_rows.extend(rows)
            flat_rows.extend(flat)
            timing_rows.extend(timing)
            checkpoint_path = directory / "iteration_metrics.csv"
            if checkpoint_path.exists() and checkpoint_path.stat().st_size:
                try:
                    checkpoint = pd.read_csv(checkpoint_path)
                except pd.errors.EmptyDataError:
                    checkpoint = pd.DataFrame()
                if not checkpoint.empty:
                    for name, value in tag.items():
                        if name not in checkpoint:
                            checkpoint[name] = value
                    checkpoint_rows.append(checkpoint)

    comparison_frame = pd.DataFrame(comparisons)
    candidate_frame = pd.DataFrame(candidates)
    cohort_frame = pd.DataFrame(cohort_rows)
    if cohort_frame.empty:
        raise ValueError("No per-fold training/future diagnostics were available")
    runtime = _unique_runtime(timing_rows)
    groups = pd.concat(all_groups, ignore_index=True)
    report_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid.uuid4().hex[:8]
    destination = version_root / "reports" / report_id
    destination.mkdir(parents=True, exist_ok=False)
    comparison_frame.to_csv(destination / "phase_comparison.csv", index=False)
    candidate_frame.to_csv(destination / "development_candidates.csv", index=False)
    cohort_frame.to_csv(destination / "diagnostic_metrics.csv", index=False)
    pd.DataFrame(flat_rows).to_csv(destination / "diagnostics_long.csv", index=False)
    runtime.to_csv(destination / "physical_fit_runtime.csv", index=False)
    groups.to_csv(destination / "grouped_metrics.csv", index=False)
    if checkpoint_rows:
        pd.concat(checkpoint_rows, ignore_index=True).to_csv(destination / "iteration_metrics.csv", index=False)
    else:
        cohort_frame.to_csv(destination / "iteration_metrics.csv", index=False)
    write_json(destination / "identity_checks.json", checks)
    write_json(destination / "environment_evidence.json", environments)

    images = _charts(destination, frames, cohort_frame, candidate_frame, frozen)
    descriptions = []
    for track in TRACKS:
        parameters = _configuration(frozen["tracks"][track])
        descriptions.append({"track": track, **parameters})
    selected_table = _table(pd.DataFrame(descriptions), ())
    phase_tables = ""
    columns = ["track", "model", "encoding", "target_mode", "max_iter", "n", "mdape", "mape",
               "median_signed_error", "old_frozen_reference"]
    for phase in PHASES:
        phase_tables += f"<h2>{PHASE_TITLES[phase]}</h2>"
        phase_tables += _table(comparison_frame.loc[comparison_frame["phase"] == phase, columns])

    selected_cohorts = cohort_frame.loc[cohort_frame["selected"] == True]
    training_comparison = cohort_frame.loc[((cohort_frame["model"] != "hgb") | cohort_frame["selected"]) &
                                           cohort_frame["cohort"].isin(["full_history_training", "recent_12m_training"])]
    training_comparison = training_comparison.groupby(["phase", "track", "model", "cohort"], sort=False).agg(
        fits=("mdape", "size"), n_min=("n", "min"), n_max=("n", "max"),
        mean_fold_mdape=("mdape", "mean"), mean_fold_mape=("mape", "mean"),
        mean_fold_log_price_mse=("log_price_mse", "mean")).reset_index()
    training_comparison.to_csv(destination / "training_diagnostic_comparison.csv", index=False)
    diagnostic_columns = [column for column in ["phase", "track", "month", "cohort", "iterations_trained",
                         "iterations_evaluated", "n", "mdape", "mape", "median_signed_error", "log_price_mse"]
                          if column in selected_cohorts]
    group_view = groups.loc[(groups["model"] == "hgb") &
                           groups["group_type"].isin(["month", "house_type_name", "sold_price_decile", "horizon_months"])]
    community = groups.loc[(groups["model"] == "hgb") & (groups["group_type"] == "community")]
    if not community.empty:
        community = community.sort_values("mdape", ascending=False).groupby(["phase", "track"], sort=False).head(10)
    candidate_columns = [column for column in ["track", "stage", *PARAMETERS, "n", "mdape", "mape",
                         "median_signed_error", "selected", "experiment_id"] if column in candidate_frame]
    image_html = "".join(_image_html(title, path) for title, path in images)
    downloads = " · ".join(_relative_link(destination / filename, destination, filename) for filename in
                           ["phase_comparison.csv", "development_candidates.csv", "diagnostic_metrics.csv", "training_diagnostic_comparison.csv",
                            "iteration_metrics.csv", "physical_fit_runtime.csv", "grouped_metrics.csv", "identity_checks.json", "environment_evidence.json"])
    final_links = []
    for track in TRACKS:
        model_path = summaries["final"].get("model_paths", {}).get(track, version_root / "final_models" / f"{track}.joblib")
        submission_path = summaries["final"].get("submission_paths", {}).get(
            track, ROOT / "submissions" / version_root.name / f"submission_{track}.csv")
        final_links.append(_relative_link(model_path, destination,
                                          f"{track} fitted HGB"))
        final_links.append(_relative_link(submission_path,
                                          destination, f"{track} candidate submission"))
    accounting_path = version_root / "experiment_manifest.json"
    accounting = _json(accounting_path) if accounting_path.exists() else None
    provenance = {"created_utc": datetime.now(timezone.utc).isoformat(), "version_root": str(version_root),
                  "report_source_sha256": sha256(Path(__file__)),
                  "helper_source_sha256": sha256(ROOT / "analysis" / "run_tree_baselines.py"),
                  "frozen_config_sha256": frozen_hash,
                  "summary_sha256": {phase: sha256(version_root / f"{phase}_summary.json") for phase in summaries},
                  "old_frozen_config_sha256": sha256(tree_root / "frozen_config.json"),
                  "identity_comparisons_passed": len(checks), "unique_physical_fit_ids_in_diagnostics": len(runtime),
                  "evidence_status": "adaptive development and reused temporal/horizon backtests",
                  "final_summary": summaries["final"], "experiment_accounting": accounting}
    write_json(destination / "report_provenance.json", provenance)
    report = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>HGB evaluation — {html.escape(version_root.name)}</title>
<style>body{{font:15px/1.55 -apple-system,BlinkMacSystemFont,Arial,sans-serif;margin:0;color:#213446;background:#f3f6f8}}
main{{max-width:1240px;padding:34px;margin:28px auto;background:white;border-radius:12px}}h1,h2{{line-height:1.25}}h2{{margin-top:34px}}
.muted,figcaption{{color:#637483}}.callout{{background:#e8f4f3;border-left:4px solid #117e86;padding:16px 20px}}
.notice{{background:#fff3dd;border-left:4px solid #d08b2e;padding:16px 20px}}.scroll{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:12px;margin:15px 0}}
th,td{{padding:7px 9px;border-bottom:1px solid #dfe5ea;text-align:right}}th{{background:#edf2f6}}td:first-child,th:first-child{{text-align:left}}
figure{{margin:22px 0}}img{{max-width:100%;height:auto}}code,pre{{background:#edf2f6}}code{{padding:2px 5px;overflow-wrap:anywhere}}
pre{{padding:14px;overflow:auto;font-size:12px}}details{{margin:18px 0}}a{{color:#0c647a}}</style></head><body><main>
<p class="muted">SSU Data Dash · {html.escape(version_root.name)} · measured experiment artifacts</p>
<h1>HistGradientBoosting evaluation</h1><div class="callout"><b>Frozen HGB selection</b>{selected_table}</div>
<p>Errors are measured on original price scale. MdAPE and MAPE are percentages; lower is better.
Pooled MdAPE uses individual homes from all evaluated months, not the mean of monthly medians.
Reference rows below are recomputed from unrounded saved predictions.</p>
<div class="notice"><b>Evidence boundary.</b> June–November 2025 is development data used to choose hyperparameters;
November had already been examined in an HGB probe. December 2025–May 2026 and the fixed-February horizon were observed
in earlier model development. These are <b>reused/adaptive backtests</b>, not a new untouched holdout or an unbiased
estimate after model selection. June–September 2026 hidden targets are unavailable.</div>
<p>Every HGB/reference comparison below verified identical <code>ml_num, month, training_cutoff, sold_price</code>.
There were {len(checks)} successful identity comparisons. RF and Bagging are both shown; the original selected family
is identified by <code>old_frozen_reference=True</code>. Each month uses sales strictly before its start;
horizon models fit once before 2026-02-01 without adding February–May labels.</p>
{phase_tables}{image_html}
<h2>Training, recent-training and future diagnostics</h2>
<p>Full-history and recent-12-month training errors are in-sample diagnostics. They represent different time/cohort
distributions from future-month evaluation. Six overlapping training sets cannot be pooled or averaged and interpreted
as a standard i.i.d. generalization gap. Compare the same cohort across model configurations and inspect monthly trends.
The evaluated checkpoint and number of iterations physically trained are recorded separately.</p>
<details open><summary>Mean fold training diagnostics: selected HGB and original references</summary>
{_table(training_comparison, ('mean_fold_mdape', 'mean_fold_mape'))}</details>
<p>These are means of per-fit training metrics, not pooled training scores. The original RF/Bagging artifacts did not
record recent-12m training metrics; those values are not fabricated or inferred.</p>
<details open><summary>Selected HGB: each fold and cohort</summary>{_table(selected_cohorts[diagnostic_columns])}</details>
<p>All candidate/checkpoint cohort metrics are in <code>diagnostic_metrics.csv</code> and <code>iteration_metrics.csv</code>;
the complete recorded diagnostics, including actual category counts and timing fields, are in <code>diagnostics_long.csv</code>.
Missing diagnostic values are displayed as N/A, never inferred from future errors.</p>
<h2>Development selection and checkpoint search</h2>
<p>Capacity is screened at 200 iterations before the retained capacity paths are compared at predeclared checkpoints.
This is a budget-limited sequential search; capacity and iteration interactions can cause the best joint configuration
to be missed. The globally selected iteration count for each track is reused in subsequent folds and final fitting;
an evaluation month does not receive its own retrospectively selected iteration count. List-aware target ablation
changes only log-price versus log-ratio target after choosing encoding and capacity.</p>
<details><summary>All development candidates</summary>{_table(candidate_frame[candidate_columns])}</details>
<h2>Grouped error analysis</h2><p>All groups include sample counts. Sold-price deciles use observed targets for retrospective
evaluation only; they cannot support prediction-time corrections or establish underfitting on their own.</p>
<details><summary>Selected HGB: monthly, property-type, price-decile and horizon groups</summary>{_table(group_view)}</details>
<details><summary>Communities with highest observed MdAPE: at most 10 per phase/track</summary>{_table(community)}</details>
<p>Community rankings are descriptive, especially for small groups. Complete community groups and both original reference
models are retained in <code>grouped_metrics.csv</code>. Home-resampling confidence intervals, if calculated elsewhere,
do not fully describe shared market shocks and do not correct adaptive selection.</p>
<h2>Runtime and accounting</h2><p>Multiple checkpoint artifacts can refer to one physical fit. The runtime CSV deduplicates
<code>physical_fit_id</code>; duplicated checkpoint times are not summed. It contains {len(runtime)} unique recorded HGB fit IDs.
These are evaluation fits; final full-data fitting is recorded in the final summary separately.
Candidate metrics and checkpoint evaluations are separate accounting units.</p>
<details><summary>Physical fits and timing</summary>{_table(runtime, ())}</details>
<details><summary>Runner experiment accounting</summary><pre>{html.escape(json.dumps(accounting, indent=2))}</pre></details>
<details><summary>Selected HGB environment, dependency and data provenance</summary>
<pre>{html.escape(json.dumps([row for row in environments if row['model'] == 'hgb'], indent=2))}</pre></details>
<h2>Candidate models, submissions and verification evidence</h2><p>{' · '.join(final_links)}</p>
<p>These artifacts are versioned candidates. The original frozen configuration, fitted models and root submission files
remain the reference. Submission validity verifies IDs and numeric prices; it does not measure hidden-test accuracy.</p>
<details><summary>Recorded final summary and check evidence</summary><pre>{html.escape(json.dumps(summaries['final'], indent=2))}</pre></details>
<p><b>Downloads:</b> {downloads}</p>
<p>This protocol uses sale-month cutoffs. Homes listed before a cutoff do not establish strict listing-day valuation.
Trees do not reliably extrapolate a previously unseen market trend. Lower training error alone does not establish
better future performance.</p><p>Frozen configuration SHA-256: <code>{frozen_hash}</code>.</p>
<p class="muted">Generated {html.escape(provenance['created_utc'])}. Images and result tables are embedded for offline reading.</p>
</main></body></html>"""
    path = destination / "evaluation_report.html"
    path.write_text(report)
    provenance["html_sha256"] = sha256(path)
    write_json(destination / "report_provenance.json", provenance)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version_root", type=Path)
    print(build_report(parser.parse_args().version_root))
