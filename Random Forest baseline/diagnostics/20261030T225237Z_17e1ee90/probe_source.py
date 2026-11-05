"""Single-development-fold diagnostic probes; these do not select production models.

All variants use the frozen baseline feature whitelist, historical training rows,
train-only preprocessing, and seed. Holdout has already been observed; new model
scores below use only November 2025 development labels.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from time import perf_counter
import uuid

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import data
import features
import models
from analysis.run_tree_baselines import environment, point_metrics, sha256, temporal_fold, write_json


def main():
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    output = ROOT.parent / "Random Forest baseline" / "diagnostics" / run_id
    output.mkdir(parents=True, exist_ok=False)
    frozen = ROOT / "outputs/tree_baselines/frozen_config.json"
    frozen_hash = sha256(frozen)
    month = pd.Period("2025-11", "M")
    listings = data.load("listings")
    train, valid, cutoff = temporal_fold(listings, month)
    recent = (train["date_sold"] >= cutoff - pd.DateOffset(months=12)).to_numpy()
    write_json(output / "environment.json", environment())
    write_json(output / "config.json", {
        "purpose": "diagnosis only, one development fold; no model selection or submission replacement",
        "development_month": str(month), "cutoff": cutoff.isoformat(),
        "n_train": len(train), "n_recent_train": int(recent.sum()), "n_validation": len(valid),
        "holdout_previously_observed": True, "new_probe_uses_holdout_labels": False,
        "frozen_config_sha256_before": frozen_hash,
        "rf_diagnostic_n_estimators": 100, "seed": 42,
        "rf_comparison_note": "100 vs 100 trees isolates the capacity bundle; not directly identical to the 300-tree frozen score",
        "capacity_bundle": "max_depth 12 -> 20 and min_samples_leaf 20/50 -> 5",
    })
    scores, prediction_frames, configurations = [], [], []
    for track in ("off_market", "list_aware"):
        leaf = 20 if track == "off_market" else 50
        template = models.TreePriceModel(track)
        tick = perf_counter()
        x_train = template.preprocessor.fit_transform(features.tree_features(train, track))
        x_valid = template.preprocessor.transform(features.tree_features(data.hide_answers(valid, track), track))
        preprocessing_seconds = perf_counter() - tick
        base_log = np.log(train["sold_price"].to_numpy(dtype=float))
        variants = [
            ("dt_d12", models.TreePriceModel(track, "dt", 12, leaf).estimator, "log_price"),
            ("dt_d20", models.TreePriceModel(track, "dt", 20, leaf).estimator, "log_price"),
            ("dt_unrestricted_sanity", models.TreePriceModel(track, "dt", None, 1).estimator, "log_price"),
            ("rf100_d12", models.TreePriceModel(track, "rf", 12, leaf, n_estimators=100).estimator, "log_price"),
            ("rf100_d20_leaf5", models.TreePriceModel(track, "rf", 20, 5, n_estimators=100).estimator, "log_price"),
            ("hgb_log_price", HistGradientBoostingRegressor(max_iter=200, max_leaf_nodes=31,
                 learning_rate=0.1, min_samples_leaf=20, early_stopping=False, random_state=42), "log_price"),
        ]
        if track == "list_aware":
            variants.append(("hgb_log_ratio", HistGradientBoostingRegressor(max_iter=200, max_leaf_nodes=31,
                learning_rate=0.1, min_samples_leaf=20, early_stopping=False, random_state=42), "log_sold_to_list"))
        for name, estimator, target in variants:
            z_train = base_log if target == "log_price" else base_log - np.log(train["list_price"].to_numpy(dtype=float))
            tick = perf_counter()
            estimator.fit(x_train, z_train)
            fit_seconds = perf_counter() - tick
            train_log, valid_log = estimator.predict(x_train), estimator.predict(x_valid)
            if target == "log_sold_to_list":
                train_log += np.log(train["list_price"].to_numpy(dtype=float))
                valid_log += np.log(valid["list_price"].to_numpy(dtype=float))
            train_prediction, valid_prediction = np.exp(train_log), np.exp(valid_log)
            row = {"track": track, "variant": name, "target": target, "fit_seconds": fit_seconds,
                   "preprocessing_seconds_shared": preprocessing_seconds,
                   "training_log_mse": float(np.mean((train_log - base_log) ** 2)),
                   "validation_log_mse": float(np.mean((valid_log - np.log(valid["sold_price"])) ** 2))}
            for prefix, actual, predicted in [
                ("training", train["sold_price"], train_prediction),
                ("recent_training", train.loc[recent, "sold_price"], train_prediction[recent]),
                ("validation", valid["sold_price"], valid_prediction),
            ]:
                row.update({f"{prefix}_{key}": value for key, value in point_metrics(actual, predicted).items()})
            if name.startswith("dt_"):
                counts = estimator.tree_.n_node_samples[estimator.tree_.children_left == -1]
                row.update(actual_depth=int(estimator.get_depth()), leaf_count=int(estimator.get_n_leaves()),
                           fraction_training_in_leaves_over_1000=float(counts[counts > 1000].sum() / len(train)),
                           largest_leaf=int(counts.max()))
            scores.append(row)
            configurations.append({"track": track, "variant": name, "target": target,
                                   "estimator_class": type(estimator).__name__, "params": estimator.get_params()})
            prediction_frames.append(pd.DataFrame({
                "ml_num": valid["ml_num"], "track": track, "variant": name, "month": str(month),
                "sold_price": valid["sold_price"], "predicted_price_raw": valid_prediction}))
            pd.DataFrame(scores).to_csv(output / "metrics.csv", index=False)
            print(f"{track} {name}: train MdAPE={row['training_mdape']:.4f}%, "
                  f"recent train={row['recent_training_mdape']:.4f}%, "
                  f"development={row['validation_mdape']:.4f}%, fit={fit_seconds:.2f}s", flush=True)
    write_json(output / "metrics.json", scores)
    write_json(output / "variants.json", configurations)
    pd.concat(prediction_frames, ignore_index=True).to_csv(output / "development_predictions.csv", index=False)
    assert sha256(frozen) == frozen_hash
    write_json(output / "verification.json", {"frozen_config_unchanged": True, "new_probe_uses_holdout_labels": False})
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
