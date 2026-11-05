"""Reproducible development -> frozen holdout -> frozen-cutoff horizon evaluation.

From the repository root, with DATA_DIR set, run each phase in this order:
    python analysis/run_tree_baselines.py --phase development
    python analysis/run_tree_baselines.py --phase holdout
    python analysis/run_tree_baselines.py --phase horizon
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
import uuid

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import data
import models

TRACKS = ("off_market", "list_aware")
FAMILIES = ("dt", "bagging", "rf")
DEVELOPMENT_MONTHS = pd.period_range("2025-06", "2025-11", freq="M")
HOLDOUT_MONTHS = pd.period_range("2025-12", "2026-05", freq="M")
HORIZON_MONTHS = pd.period_range("2026-02", "2026-05", freq="M")
OUTPUT = ROOT / "outputs" / "tree_baselines"


def write_json(path, value, exclusive=False):
    def numeric_scalar(item):
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(f"Cannot serialize {type(item).__name__}")
    with path.open("x" if exclusive else "w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False, default=numeric_scalar)
        handle.write("\n")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment():
    def git(*args):
        result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    return {
        "python": sys.version, "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in
                     ["numpy", "pandas", "scikit-learn", "pyarrow", "matplotlib", "joblib"]},
        "git_revision": git("rev-parse", "HEAD"), "git_status": git("status", "--short"),
        "source_sha256": {name: sha256(ROOT / name) for name in
                          ["models.py", "features.py", "data.py", "run.py", "requirements.txt",
                           "analysis/run_tree_baselines.py"]},
        "data_identity": {name: {"path": str(data.find_file(name).resolve()),
                                  "sha256": sha256(data.find_file(name)),
                                  "bytes": data.find_file(name).stat().st_size}
                          for name in ["listings", *TRACKS]},
    }


def point_metrics(actual, predicted):
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if len(actual) == 0 or not (np.isfinite(actual) & (actual > 0)).all():
        raise ValueError("Metrics require nonempty, positive finite targets")
    if predicted.shape != actual.shape or not (np.isfinite(predicted) & (predicted > 0)).all():
        raise ValueError("Metrics require matching positive finite predictions")
    signed = 100 * (predicted - actual) / actual
    return {"n": len(actual), "mdape": float(np.median(np.abs(signed))),
            "mape": float(np.mean(np.abs(signed))),
            "median_signed_error": float(np.median(signed))}


def temporal_fold(listings, month, cutoff=None):
    """Sale-month protocol; a strict cutoff excludes all validation-month labels."""
    cutoff = month.start_time if cutoff is None else pd.Timestamp(cutoff)
    train = listings.loc[listings["date_sold"] < cutoff]
    valid = listings.loc[listings["date_sold"].dt.to_period("M") == month]
    if train.empty or valid.empty or cutoff > month.start_time:
        raise ValueError(f"Invalid / empty temporal fold: {month}, cutoff={cutoff}")
    assert (train["date_sold"] < cutoff).all()
    assert set(train["ml_num"]).isdisjoint(valid["ml_num"])
    return train, valid, cutoff


def fit_and_diagnose(model, train, track, cutoff):
    started = perf_counter()
    model.fit(train)
    fit_seconds = perf_counter() - started
    if isinstance(model, models.NearestSalesBaseline):
        return {"training_cutoff": cutoff.isoformat(), "n_train": len(train),
                "max_training_date_sold": train["date_sold"].max().isoformat(),
                "fit_seconds": fit_seconds, "training_predict_seconds": None,
                "training_log_mse": None, "training_price_error": None,
                "training_error_note": "Not defined for earliest listing dates with no earlier sales; this reference baseline has no fitted regression objective."}
    started = perf_counter()
    homes = data.hide_answers(train, track)
    if isinstance(model, models.TreePriceModel):
        log_prediction = model.predict_log(homes)
        prediction = np.exp(log_prediction)
    else:
        prediction = model.predict(homes)
        log_prediction = np.log(prediction)
    diagnostic = {
        "training_cutoff": cutoff.isoformat(), "n_train": len(train),
        "max_training_date_sold": train["date_sold"].max().isoformat(),
        "fit_seconds": fit_seconds, "training_predict_seconds": perf_counter() - started,
        "training_log_mse": float(np.mean((log_prediction - np.log(train["sold_price"])) ** 2)),
        "training_price_error": point_metrics(train["sold_price"], prediction),
    }
    if isinstance(model, models.TreePriceModel):
        diagnostic["feature_order"] = model.feature_order_
        diagnostic["n_encoded_features"] = len(model.feature_order_)
        if model.family == "dt":
            estimator = model.estimator
            leaf_counts = estimator.tree_.n_node_samples[estimator.tree_.children_left == -1]
            diagnostic.update(actual_depth=estimator.get_depth(), leaf_count=estimator.get_n_leaves(),
                              leaf_sample_counts=leaf_counts.tolist())
        else:
            trees = model.estimator.estimators_
            diagnostic["n_trees"] = len(trees)
            diagnostic["tree_depths"] = [tree.get_depth() for tree in trees]
            diagnostic["tree_leaf_counts"] = [tree.get_n_leaves() for tree in trees]
    return diagnostic


def grouped_metrics(predictions):
    rows = []
    frame = predictions.copy()
    frame["sold_price_decile"] = pd.qcut(frame["sold_price"], 10, labels=False, duplicates="drop") + 1
    group_columns = ["month", "house_type_name", "sold_price_decile", "community"]
    if "horizon_months" in frame:
        group_columns.append("horizon_months")
    for column in group_columns:
        for value, group in frame.groupby(column, observed=True):
            rows.append({"group_type": column, "group": str(value),
                         **point_metrics(group["sold_price"], group["predicted_price_raw"])})
    return pd.DataFrame(rows)


def evaluate(listings, track, family, depth, leaf, phase, run_id, env):
    experiment_id = f"{phase}_{family}_{track}_d{depth}_l{leaf}_{run_id}"
    directory = OUTPUT / experiment_id
    directory.mkdir(parents=True, exist_ok=False)
    start_utc = datetime.now(timezone.utc).isoformat()
    started = perf_counter()
    months = {"development": DEVELOPMENT_MONTHS, "holdout": HOLDOUT_MONTHS, "horizon": HORIZON_MONTHS}[phase]
    frames, diagnostics = [], []
    model = None
    for month in months:
        cutoff = pd.Timestamp("2026-02-01") if phase == "horizon" else month.start_time
        train, valid, cutoff = temporal_fold(listings, month, cutoff)
        if phase != "horizon" or model is None:
            model = (models.ListPriceBaseline() if track == "list_aware" else models.NearestSalesBaseline(k=10)) \
                if family == "starter" else models.TreePriceModel(track, family, depth, leaf)
            diagnostics.append(fit_and_diagnose(model, train, track, cutoff))
        tick = perf_counter()
        predicted = model.predict(data.hide_answers(valid, track))
        scores = point_metrics(valid["sold_price"], predicted)
        diagnostics[-1].setdefault("validation", []).append({
            "month": str(month), "predict_seconds": perf_counter() - tick, **scores})
        result = valid[["ml_num", "house_type_name", "community", "sold_price"]].copy()
        result = result.assign(track=track, model_family=family, phase=phase, month=str(month),
                               training_cutoff=cutoff.isoformat(), predicted_price=np.round(predicted),
                               predicted_price_raw=predicted)
        if phase == "horizon":
            result["horizon_months"] = month.ordinal - HORIZON_MONTHS[0].ordinal + 1
        frames.append(result)
        print(f"{phase} {track} {family} d={depth} leaf={leaf} {month}: "
              f"n_train={len(train):,} n={len(valid):,} MdAPE={scores['mdape']:.4f}%", flush=True)
    predictions = pd.concat(frames, ignore_index=True)
    scores = point_metrics(predictions["sold_price"], predictions["predicted_price_raw"])
    config = {"phase": phase, "track": track, "family": family,
              "model": model.configuration() if family != "starter" else
                       {"class": type(model).__name__, "k": 10 if track == "off_market" else None},
              "prediction_months": [str(month) for month in months], "sample_weight": "equal",
              "training_window": "all sales strictly before cutoff", "metric_prediction_column": "predicted_price_raw",
              "evaluation_protocol": "sale-month cutoff; not strict listing-day valuation"}
    if phase != "development":
        config["frozen_config_sha256"] = sha256(OUTPUT / "frozen_config.json")
    write_json(directory / "config.json", config)
    write_json(directory / "environment.json", {**env, "start_utc": start_utc,
               "end_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": perf_counter() - started})
    predictions.to_csv(directory / "predictions.csv", index=False)
    write_json(directory / "metrics.json", {"pooled": scores, "precision": "unrounded predictions",
               "total_fit_seconds": sum(row["fit_seconds"] for row in diagnostics),
               "total_validation_predict_seconds": sum(v["predict_seconds"] for row in diagnostics for v in row["validation"])})
    grouped_metrics(predictions).to_csv(directory / "grouped_metrics.csv", index=False)
    write_json(directory / "model_diagnostics.json", diagnostics)
    return {"experiment_id": experiment_id, "track": track, "family": family,
            "max_depth": depth, "min_samples_leaf": leaf, **scores}


def check_same_rows(results):
    """All candidates on one track must score exactly the same IDs and truth."""
    for track in TRACKS:
        reference = None
        for result in results:
            if result["track"] != track:
                continue
            frame = pd.read_csv(OUTPUT / result["experiment_id"] / "predictions.csv")
            identity = frame[["ml_num", "month", "training_cutoff", "sold_price"]]
            if reference is None:
                reference = identity
            else:
                pd.testing.assert_frame_equal(reference, identity)


def development(listings, run_id, env):
    frozen_path = OUTPUT / "frozen_config.json"
    if frozen_path.exists():
        raise FileExistsError("A configuration is already frozen. Preserve it; use a separate versioned project for a new selection.")
    results, selected = [], {}
    for track in TRACKS:
        grid = [evaluate(listings, track, "dt", depth, leaf, "development", run_id, env)
                for depth in (4, 8, 12) for leaf in (20, 50, 100)]
        results.extend(grid)
        best = min(grid, key=lambda row: (row["mdape"], row["max_depth"], -row["min_samples_leaf"]))
        depth, leaf = best["max_depth"], best["min_samples_leaf"]
        comparison = [best] + [evaluate(listings, track, family, depth, leaf, "development", run_id, env)
                               for family in ("bagging", "rf")]
        results.extend(comparison[1:])
        winner = min(comparison, key=lambda row: (row["mdape"], FAMILIES.index(row["family"])))
        selected[track] = {"family": winner["family"], "max_depth": depth, "min_samples_leaf": leaf,
                           "selection_metric": "development pooled MdAPE (percent, unrounded predictions)",
                           "development_mdape": winner["mdape"],
                           "selected_experiment_id": winner["experiment_id"],
                           "complexity_experiment_id": best["experiment_id"],
                           "comparison": comparison,
                           "model_config": models.TreePriceModel(track, winner["family"], depth, leaf).configuration()}
    check_same_rows(results)
    frozen = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
              "development_months": [str(month) for month in DEVELOPMENT_MONTHS],
              "holdout_months": [str(month) for month in HOLDOUT_MONTHS],
              "seed": 42, "selection_run_id": run_id, "tracks": selected,
              "selection_environment": env,
              "selection_policy": "DT grid by pooled MdAPE; ties: depth ascending, leaf descending; family ties: DT, Bagging, RF",
              "holdout_used_for_selection": False}
    # Exclusive creation prevents an evaluation run from silently changing selection.
    write_json(frozen_path, frozen, exclusive=True)
    return results


def frozen_evaluation(listings, phase, run_id, env):
    frozen_path = OUTPUT / "frozen_config.json"
    frozen = json.loads(frozen_path.read_text())
    if env["data_identity"] != frozen["selection_environment"]["data_identity"]:
        raise ValueError("Data identity differs from the frozen development data")
    results = []
    for track in TRACKS:
        models.make_model(track)  # Check schema / exact frozen parameters before evaluation.
        selected = frozen["tracks"][track]
        families = (*FAMILIES, "starter") if phase == "holdout" else FAMILIES
        for family in families:
            results.append(evaluate(listings, track, family, selected["max_depth"],
                                    selected["min_samples_leaf"], phase, run_id, env))
    check_same_rows(results)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=["development", "holdout", "horizon"])
    args = parser.parse_args()
    if importlib.metadata.version("scikit-learn") != "1.7.2":
        raise RuntimeError("Use the pinned requirements: scikit-learn==1.7.2")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.phase == "development" and (OUTPUT / "frozen_config.json").exists():
        parser.error("Configuration already frozen; no overwrite or holdout-driven reselection is allowed")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid.uuid4().hex[:8]
    env = environment()
    listings = data.load("listings")
    if listings["ml_num"].duplicated().any() or listings["date_sold"].isna().any():
        raise ValueError("Listings require unique IDs and valid sale dates")
    if not (listings["date_sold"] < pd.Timestamp("2026-06-01")).all():
        raise ValueError("Published training sales must precede 2026-06-01")
    results = development(listings, run_id, env) if args.phase == "development" else \
        frozen_evaluation(listings, args.phase, run_id, env)
    manifest = OUTPUT / f"{args.phase}_summary_{run_id}.json"
    write_json(manifest, {"phase": args.phase, "run_id": run_id, "results": results,
                         "frozen_config_sha256": sha256(OUTPUT / "frozen_config.json")})
    print(f"Saved {manifest}", flush=True)


if __name__ == "__main__":
    main()
