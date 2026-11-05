"""Execute the versioned HGB protocol in HistGradientBoosting/INSTRUCTIONS.md.

Development is adaptive selection. Backtest and horizon use previously observed
evaluation periods, and never update the frozen configuration or default models.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import sys
from time import perf_counter
import uuid

import joblib
import numpy as np
import pandas as pd
from filelock import FileLock, Timeout

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import check_submission
import data
import features
import models
from analysis.run_tree_baselines import (
    DEVELOPMENT_MONTHS, HOLDOUT_MONTHS, HORIZON_MONTHS, TRACKS,
    environment, grouped_metrics, point_metrics, sha256, temporal_fold, write_json,
)

ENCODINGS = ("onehot", "native")
CAPACITY_LEAVES = (15, 31, 63)
CAPACITY_MIN_SAMPLES = (20, 50)
ITERATION_CHECKPOINTS = (100, 200, 400, 800)
BASELINE_NAMES = {"make_model", "TreePriceModel", "ListPriceBaseline", "NearestSalesBaseline", "FeatureModel"}


def now():
    return datetime.now(timezone.utc).isoformat()


def unique_id():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid.uuid4().hex[:8]


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    write_json(temporary, value, exclusive=True)
    temporary.replace(path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_predictions(path):
    return pd.read_csv(path, dtype={"ml_num": str}, float_precision="round_trip")


def identity(frame):
    columns = ["ml_num", "month", "training_cutoff", "sold_price"]
    return frame[columns].sort_values(["month", "ml_num"]).reset_index(drop=True)


def assert_same_rows(first, second):
    if first["ml_num"].duplicated().any() or second["ml_num"].duplicated().any():
        raise ValueError("Evaluation predictions must contain each home once")
    pd.testing.assert_frame_equal(identity(first), identity(second), check_dtype=False)


def selection_key(result):
    option = result["model_config"]["options"]
    metric = result["metrics"]
    return (metric["mdape"], metric["mape"], option["max_leaf_nodes"], option["max_iter"],
            -option["min_samples_leaf"], ENCODINGS.index(option["encoding"]),
            0 if option["target_mode"] == "log_price" else 1)


def collect_log_checkpoints(model, homes, checkpoints):
    """Copy snapshots: sklearn staged generators may reuse their working array."""
    requested = set(checkpoints)
    if requested == {model.estimator.n_iter_}:
        return {model.estimator.n_iter_: np.array(model.predict_log(homes), copy=True)}
    result = {}
    for number, prediction in enumerate(model.staged_predict_log(homes), start=1):
        if number in requested:
            result[number] = np.array(prediction, dtype=float, copy=True)
        if number == max(requested):
            break
    if set(result) != requested:
        raise ValueError("Estimator did not produce all declared checkpoints")
    return result


def baseline_guard(version_root, create=False):
    """Protect artifacts by bytes and baseline implementations by AST, not file hash."""
    path = version_root / "baseline_guard.json"
    if not path.exists():
        if not create:
            raise FileNotFoundError(path)
        protected = []
        for folder in ("outputs/tree_baselines", "outputs/fitted_models", "reports", "figures", "submissions"):
            protected.extend(p for p in (ROOT / folder).rglob("*") if p.is_file())
        nodes = ast.parse((ROOT / "models.py").read_text()).body
        guard = {
            "protected_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in protected},
            "baseline_ast": {node.name: ast.dump(node, include_attributes=False) for node in nodes
                             if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in BASELINE_NAMES},
            "features_sha256": sha256(ROOT / "features.py"),
            "requirements_sha256": sha256(ROOT / "requirements.txt"),
        }
        write_json(path, guard, exclusive=True)
        before = version_root / "source_before"
        before.mkdir(exist_ok=True)
        for name in ("models.py", "features.py", "data.py", "requirements.txt"):
            shutil.copy2(ROOT / name, before / name)
    guard = read_json(path)
    for name, expected in guard["protected_sha256"].items():
        if not (ROOT / name).is_file() or sha256(ROOT / name) != expected:
            raise ValueError(f"Protected baseline artifact changed: {name}")
    nodes = ast.parse((ROOT / "models.py").read_text()).body
    current = {node.name: ast.dump(node, include_attributes=False) for node in nodes
               if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in BASELINE_NAMES}
    if current != guard["baseline_ast"]:
        raise ValueError("Baseline-relevant model implementations changed")
    for name in ("features", "requirements"):
        filename = "features.py" if name == "features" else "requirements.txt"
        if sha256(ROOT / filename) != guard[name + "_sha256"]:
            raise ValueError(f"Baseline dependency changed: {filename}")
    return {"passed": True, "protected_files": len(guard["protected_sha256"]),
            "baseline_ast_unchanged": True, "guard_sha256": sha256(path)}


class HGBRunner:
    def __init__(self, version_root, listings=None, guard=True):
        self.root = Path(version_root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.version = self.root.name
        self.guard_enabled = guard
        if guard:
            baseline_guard(self.root, create=True)
        self.listings = data.load("listings") if listings is None else listings
        if (self.listings["ml_num"].duplicated().any() or self.listings["date_sold"].isna().any()
                or not (self.listings["date_sold"] < pd.Timestamp("2026-06-01")).all()):
            raise ValueError("Listings require unique IDs and valid sales strictly before 2026-06-01")
        prices = self.listings["sold_price"].to_numpy(dtype=float)
        if not (np.isfinite(prices) & (prices > 0)).all():
            raise ValueError("Published sold_price must be positive and finite")
        self.env = environment() if listings is None else {"test_fixture": True}
        self.env["source_sha256"] = {**self.env.get("source_sha256", {}),
            "analysis/run_hgb_baselines.py": sha256(Path(__file__)), "models.py": sha256(ROOT / "models.py")}
        self.env["thread_settings"] = {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
        self.env["packages_all"] = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
                                    if d.metadata["Name"]}
        contract = {
            "version": self.version,
            "model_sources": {name: sha256(ROOT / name) for name in
                              ("models.py", "features.py", "data.py", "analysis/run_tree_baselines.py",
                               "analysis/run_hgb_baselines.py")},
            "data_identity": self.env.get("data_identity", {"fixture_fingerprint": fingerprint(
                self.listings[["ml_num", "sold_price"]].to_dict("list"))}),
            "packages": {name: importlib.metadata.version(name) for name in
                         ("numpy", "pandas", "scikit-learn", "pyarrow")},
            "protocol": {"development_months": list(map(str, DEVELOPMENT_MONTHS)),
                         "holdout_months": list(map(str, HOLDOUT_MONTHS)),
                         "horizon_months": list(map(str, HORIZON_MONTHS)),
                         "capacity_leaves": list(CAPACITY_LEAVES), "capacity_min_samples": list(CAPACITY_MIN_SAMPLES),
                         "iteration_checkpoints": list(ITERATION_CHECKPOINTS)},
        }
        contract_path = self.root / "execution_contract.json"
        if contract_path.exists() and read_json(contract_path) != contract:
            raise ValueError("Model/data/protocol contract changed; use a new version")
        if not contract_path.exists():
            write_json(contract_path, contract, exclusive=True)
        self.contract_hash = sha256(contract_path)
        self.manifest_path = self.root / "experiment_manifest.json"
        self.manifest = read_json(self.manifest_path) if self.manifest_path.exists() else {
            "version": self.version, "created_utc": now(), "physical_runs": {}, "experiments": {}, "phases": {}}

    def save_manifest(self):
        self.manifest["updated_utc"] = now()
        atomic_json(self.manifest_path, self.manifest)

    def guard(self):
        return baseline_guard(self.root) if self.guard_enabled else {"test_fixture": True}

    def fit_count(self, phase=None):
        return sum(len(run["folds"]) for run in self.manifest["physical_runs"].values()
                   if phase is None or run["phase"] == phase)

    def frame(self, result):
        return load_predictions(self.root / "experiments" / result["experiment_id"] / "predictions.csv")

    def read_frozen(self, require_summary=True):
        path = self.root / "frozen_config.json"
        frozen = read_json(path)
        if frozen["execution_contract_sha256"] != self.contract_hash or frozen["version"] != self.version:
            raise ValueError("Frozen selection belongs to a different execution contract")
        for track in TRACKS:
            selected = frozen["tracks"][track]
            recorded = self.manifest["experiments"][selected["selected_experiment_id"]]
            if selected["model_config"] != recorded["model_config"] or selected["development_metrics"] != recorded["metrics"]:
                raise ValueError("Frozen selection differs from recorded development evidence")
        summary = self.root / "development_summary.json"
        if require_summary:
            if not summary.exists() or read_json(summary)["frozen_config_sha256"] != sha256(path):
                raise ValueError("Frozen configuration changed after development")
        return frozen

    def _folds(self, phase):
        if phase == "horizon":
            return [("2026-02_fixed", list(HORIZON_MONTHS), pd.Timestamp("2026-02-01"))]
        months = DEVELOPMENT_MONTHS if phase == "development" else HOLDOUT_MONTHS
        return [(str(month), [month], month.start_time) for month in months]

    def _fit_fold(self, physical, label, months, cutoff, checkpoints, reference):
        if label in physical["folds"]:
            cached = self.root / physical["folds"][label]
            if read_json(cached / "completed.json")["execution_contract_sha256"] != self.contract_hash:
                raise ValueError("Cached fold contract differs")
            return cached
        attempt = self.root / "physical_runs" / physical["physical_key"] / f"{label}_{unique_id()}"
        attempt.mkdir(parents=True)
        started_utc, tick = now(), perf_counter()
        try:
            train = self.listings.loc[self.listings["date_sold"] < cutoff]
            if train.empty or not (train["date_sold"] < cutoff).all():
                raise ValueError("Invalid training cutoff")
            validation = []
            for month in months:
                _, rows, _ = temporal_fold(self.listings, month, cutoff)
                validation.append((month, rows))
            model = models.HistPriceModel(**physical["options"])
            fit_start = perf_counter()
            model.fit(train)
            fit_seconds = perf_counter() - fit_start
            if model.estimator.n_iter_ != physical["options"]["max_iter"]:
                raise ValueError("Unexpected automatic early stopping")
            train_start = perf_counter()
            train_logs = collect_log_checkpoints(model, data.hide_answers(train, model.track), checkpoints)
            training_predict_seconds = perf_counter() - train_start
            recent = (train["date_sold"] >= cutoff - pd.DateOffset(months=12)).to_numpy()
            if not recent.any():
                raise ValueError("Recent training cohort is empty")
            frames = {number: [] for number in checkpoints}
            future = {number: [] for number in checkpoints}
            predict_seconds = 0.0
            for month, rows in validation:
                predict_start = perf_counter()
                logs = collect_log_checkpoints(model, data.hide_answers(rows, model.track), checkpoints)
                predict_seconds += perf_counter() - predict_start
                for number in checkpoints:
                    prices = np.exp(logs[number])
                    score = point_metrics(rows["sold_price"], prices)
                    future[number].append({"month": str(month), **score,
                        "log_mse": float(np.mean((logs[number] - np.log(rows["sold_price"].to_numpy(dtype=float))) ** 2))})
                    frame = rows[["ml_num", "house_type_name", "community", "sold_price"]].copy()
                    frame = frame.assign(track=model.track, model_family="hgb", phase=physical["phase"],
                        month=str(month), training_cutoff=cutoff.isoformat(), encoding=model.encoding,
                        target_mode=model.target_mode, iterations_evaluated=number,
                        predicted_price_raw=prices, predicted_price=np.round(prices))
                    if physical["phase"] == "horizon":
                        frame["horizon_months"] = month.ordinal - HORIZON_MONTHS[0].ordinal + 1
                    if number == 200 and reference is not None:
                        previous = reference.loc[reference["month"] == str(month)]
                        assert_same_rows(previous, frame)
                        aligned = previous.set_index("ml_num").loc[frame["ml_num"]]
                        np.testing.assert_allclose(prices, aligned["predicted_price_raw"], rtol=1e-10, atol=1e-6)
                    frames[number].append(frame)
            diagnostics = {}
            target_log = np.log(train["sold_price"].to_numpy(dtype=float))
            for number in checkpoints:
                price = np.exp(train_logs[number])
                score = point_metrics(train["sold_price"], price)
                recent_score = point_metrics(train.loc[recent, "sold_price"], price[recent])
                diagnostics[number] = {
                    "physical_fit_id": str(attempt.relative_to(self.root)), "training_cutoff": cutoff.isoformat(),
                    "max_training_date_sold": train["date_sold"].max().isoformat(),
                    "iterations_trained": int(model.estimator.n_iter_), "iterations_evaluated": number,
                    "n_train": len(train), "fit_seconds": fit_seconds,
                    "preprocessing_seconds": getattr(model, "fit_preprocessing_seconds_", None),
                    "estimator_fit_seconds": getattr(model, "fit_estimator_seconds_", None),
                    "training_predict_seconds_total": training_predict_seconds,
                    "validation_predict_seconds_total": predict_seconds,
                    "training_log_mse": float(np.mean((train_logs[number] - target_log) ** 2)),
                    "training_price_error": score,
                    "recent_training_log_mse": float(np.mean((train_logs[number][recent] - target_log[recent]) ** 2)),
                    "recent_training_price_error": recent_score, "validation": future[number],
                    "n_encoded_features": len(model.feature_order_), "feature_order": model.feature_order_,
                    "category_counts": {name: len(values) for name, values in model.category_levels_.items()},
                    "category_levels": model.category_levels_, "staged_200_matches_reference": reference is not None,
                }
                pd.concat(frames[number], ignore_index=True).to_csv(attempt / f"predictions_{number}.csv", index=False)
            write_json(attempt / "diagnostics.json", diagnostics)
            write_json(attempt / "environment.json", {**self.env, "start_utc": started_utc, "end_utc": now(),
                       "elapsed_seconds": perf_counter() - tick})
            write_json(attempt / "config.json", {"model_config": model.configuration(), "checkpoints": checkpoints,
                       "months": list(map(str, months)), "training_cutoff": cutoff.isoformat()})
            write_json(attempt / "completed.json", {"execution_contract_sha256": self.contract_hash,
                       "completed_utc": now(), "fit_seconds": fit_seconds}, exclusive=True)
            physical["folds"][label] = str(attempt.relative_to(self.root))
            self.save_manifest()
            metrics_text = "; ".join(f"i{k} train={diagnostics[k]['training_price_error']['mdape']:.3f}% "
                f"future={np.mean([r['mdape'] for r in future[k]]):.3f}%" for k in checkpoints)
            print(f"{physical['phase']} {model.track} {model.encoding}/{model.target_mode} "
                  f"leaves={model.estimator.max_leaf_nodes} minleaf={model.estimator.min_samples_leaf} "
                  f"{label}: {metrics_text}; fit={fit_seconds:.2f}s; physical_fits={self.fit_count()}", flush=True)
            atomic_json(self.root / "progress.json", {"phase": physical["phase"], "track": model.track,
                        "encoding": model.encoding, "fold": label, "completed_physical_fits": self.fit_count(),
                        "updated_utc": now()})
            return attempt
        except BaseException as error:
            write_json(attempt / "failure.json", {"type": type(error).__name__, "message": str(error), "time_utc": now()})
            raise

    def evaluate(self, options, phase="development", stage="capacity", checkpoints=None, reference_200=None):
        physical_model = models.HistPriceModel(**options)
        options = physical_model.configuration()["options"]
        checkpoints = sorted(set(checkpoints or [options["max_iter"]]))
        if not checkpoints or checkpoints[0] < 1 or checkpoints[-1] > options["max_iter"]:
            raise ValueError("Invalid checkpoints")
        key = fingerprint({"options": options, "phase": phase, "checkpoints": checkpoints,
                           "execution_contract_sha256": self.contract_hash})
        physical = self.manifest["physical_runs"].setdefault(key, {
            "physical_key": key, "phase": phase, "stage": stage, "options": options,
            "checkpoints": checkpoints, "folds": {}, "experiments": {}})
        if all(str(number) in physical["experiments"] for number in checkpoints):
            return [self.manifest["experiments"][physical["experiments"][str(number)]] for number in checkpoints]
        reference = self.frame(reference_200) if reference_200 is not None else None
        attempts = [self._fit_fold(physical, label, months, cutoff, checkpoints, reference)
                    for label, months, cutoff in self._folds(phase)]
        results = []
        for number in checkpoints:
            if str(number) in physical["experiments"]:
                results.append(self.manifest["experiments"][physical["experiments"][str(number)]])
                continue
            evaluated = models.HistPriceModel(**{**options, "max_iter": number}).configuration()
            experiment_id = f"{phase}_{stage}_{options['track']}_{options['encoding']}_{options['target_mode']}_i{number}_{unique_id()}"
            directory = self.root / "experiments" / experiment_id
            directory.mkdir(parents=True)
            frames = [load_predictions(attempt / f"predictions_{number}.csv") for attempt in attempts]
            predictions = pd.concat(frames, ignore_index=True)
            metric = point_metrics(predictions["sold_price"], predictions["predicted_price_raw"])
            diagnostics = [read_json(attempt / "diagnostics.json")[str(number)] for attempt in attempts]
            config = {"phase": phase, "stage": stage, "model_config": evaluated,
                      "physical_model_config": physical_model.configuration(), "physical_key": key,
                      "execution_contract_sha256": self.contract_hash,
                      "holdout_previously_observed": True, "evaluation_protocol": "sale-month cutoff",
                      "metric_prediction_column": "predicted_price_raw"}
            if phase != "development":
                config["frozen_config_sha256"] = sha256(self.root / "frozen_config.json")
            write_json(directory / "config.json", config)
            write_json(directory / "environment.json", {**self.env, "created_utc": now(),
                "physical_fit_ids": [row["physical_fit_id"] for row in diagnostics],
                "fit_seconds_physical": sum(row["fit_seconds"] for row in diagnostics),
                "time_accounting": "shared across checkpoints; deduplicate by physical_fit_id"})
            predictions.to_csv(directory / "predictions.csv", index=False)
            write_json(directory / "metrics.json", {"pooled": metric})
            grouped_metrics(predictions).to_csv(directory / "grouped_metrics.csv", index=False)
            write_json(directory / "model_diagnostics.json", diagnostics)
            pd.DataFrame([{"month": v["month"], "iterations_evaluated": number,
                "training_mdape": row["training_price_error"]["mdape"],
                "recent_training_mdape": row["recent_training_price_error"]["mdape"],
                "training_log_mse": row["training_log_mse"], "validation_log_mse": v["log_mse"],
                **v} for row in diagnostics for v in row["validation"]]).to_csv(directory / "iteration_metrics.csv", index=False)
            result = {"experiment_id": experiment_id, "phase": phase, "stage": stage, "track": options["track"],
                      "encoding": options["encoding"], "target_mode": options["target_mode"],
                      "iterations_evaluated": number, "model_config": evaluated, "metrics": metric,
                      "physical_key": key}
            self.manifest["experiments"][experiment_id] = result
            physical["experiments"][str(number)] = experiment_id
            self.save_manifest()
            results.append(result)
        return results

    def save_summary(self, phase, results, **extra):
        payload = {"phase": phase, "version": self.version, "created_utc": now(), "results": results,
                   "physical_fits": self.fit_count(phase), "execution_contract_sha256": self.contract_hash, **extra}
        if (self.root / "frozen_config.json").exists():
            payload["frozen_config_sha256"] = sha256(self.root / "frozen_config.json")
        path = self.root / f"{phase}_summary.json"
        if path.exists():
            existing = read_json(path)
            if existing["results"] != results:
                raise ValueError(f"Cannot overwrite completed {phase} results")
        else:
            write_json(path, payload, exclusive=True)
        self.manifest["phases"][phase] = {"summary": path.name, "completed_utc": now()}
        self.save_manifest()
        return payload

    def development(self):
        frozen_path = self.root / "frozen_config.json"
        if frozen_path.exists():
            if (self.root / "development_summary.json").exists():
                self.read_frozen()
                print("Development already frozen; reusing completed selection without new fits.", flush=True)
                return read_json(self.root / "development_summary.json")
            frozen = self.read_frozen(require_summary=False)
            recorded = [r for r in self.manifest["experiments"].values() if r["phase"] == "development"]
            return self.save_summary("development", recorded, frozen_tracks=frozen["tracks"])
        all_results = {}
        capacity = {}
        # Stage A precedes the rest of the grid; the cache reuses these in Stage B.
        for track in TRACKS:
            for encoding in ENCODINGS:
                option = dict(track=track, encoding=encoding, max_iter=200, max_leaf_nodes=31, min_samples_leaf=20)
                result = self.evaluate(option, stage="reference")[0]
                capacity[(track, encoding, 31, 20)] = result
                all_results[result["experiment_id"]] = result
        for track in TRACKS:
            for encoding in ENCODINGS:
                for leaves in CAPACITY_LEAVES:
                    for leaf in CAPACITY_MIN_SAMPLES:
                        result = self.evaluate(dict(track=track, encoding=encoding, max_iter=200,
                            max_leaf_nodes=leaves, min_samples_leaf=leaf), stage="capacity")[0]
                        capacity[(track, encoding, leaves, leaf)] = result
                        all_results[result["experiment_id"]] = result
        retained = {}
        iteration_results = []
        for track in TRACKS:
            for encoding in ENCODINGS:
                choices = [r for (t, e, _, _), r in capacity.items() if (t, e) == (track, encoding)]
                best = min(choices, key=selection_key)
                retained[(track, encoding)] = best
                options = {**best["model_config"]["options"], "max_iter": 800}
                results = self.evaluate(options, stage="iterations", checkpoints=ITERATION_CHECKPOINTS, reference_200=best)
                iteration_results.extend(results)
                all_results.update({r["experiment_id"]: r for r in results})
        chosen = {track: min([r for r in iteration_results if r["track"] == track], key=selection_key) for track in TRACKS}
        list_price = chosen["list_aware"]
        ratio = self.evaluate({**list_price["model_config"]["options"], "target_mode": "log_ratio"}, stage="target")[0]
        all_results[ratio["experiment_id"]] = ratio
        chosen["list_aware"] = min([list_price, ratio], key=selection_key)
        for track in TRACKS:
            reference = None
            for result in all_results.values():
                if result["track"] != track:
                    continue
                frame = self.frame(result)
                if reference is not None:
                    assert_same_rows(reference, frame)
                reference = frame
        if len(DEVELOPMENT_MONTHS) == 6 and self.fit_count("development") != 174:
            raise ValueError(f"Unexpected development fit count: {self.fit_count('development')}")
        self.guard()
        frozen = {"schema_version": 1, "version": self.version, "created_utc": now(),
            "holdout_previously_observed": True, "selection_used_new_backtest_scores": False,
            "development_months": list(map(str, DEVELOPMENT_MONTHS)),
            "selection_policy": "pooled raw-price MdAPE, MAPE, fewer leaves/iterations, larger leaf, onehot then native; target ties log_price",
            "execution_contract_sha256": self.contract_hash, "seed": 42,
            "tracks": {track: {"model_config": result["model_config"], "development_metrics": result["metrics"],
                         "selected_experiment_id": result["experiment_id"], "encoding": result["encoding"],
                         "target_mode": result["target_mode"]} for track, result in chosen.items()},
            "capacity_retained": {f"{track}/{encoding}": r["experiment_id"] for (track, encoding), r in retained.items()},
            "list_target_comparison": [list_price["experiment_id"], ratio["experiment_id"]],
        }
        write_json(frozen_path, frozen, exclusive=True)
        summary = self.save_summary("development", list(all_results.values()), frozen_tracks=frozen["tracks"])
        for track, result in chosen.items():
            print(f"FROZEN {track}: {result['model_config']['options']}; pooled MdAPE={result['metrics']['mdape']:.6f}%", flush=True)
        return summary

    def frozen_evaluation(self, phase):
        if phase not in ("backtest", "horizon"):
            raise ValueError(phase)
        frozen_path = self.root / "frozen_config.json"
        frozen_hash, frozen = sha256(frozen_path), self.read_frozen()
        results = []
        for track in TRACKS:
            model = models.make_hgb_model(track, frozen_path)
            results.extend(self.evaluate(model.configuration()["options"], phase=phase, stage="frozen_evaluation"))
        if sha256(frozen_path) != frozen_hash:
            raise ValueError("Frozen configuration changed during evaluation")
        self.guard()
        return self.save_summary(phase, results, holdout_previously_observed=True)

    def predict(self):
        self.read_frozen()
        final_summary = self.root / "final_summary.json"
        if final_summary.exists():
            existing = read_json(final_summary)
            self.check()
            print("Final models already verified; reusing without retraining.", flush=True)
            return existing
        frozen_path = self.root / "frozen_config.json"
        frozen_hash = sha256(frozen_path)
        exports = [self.root / "final_models" / f"{track}.joblib" for track in TRACKS]
        candidate_folder = ROOT / "submissions" / self.version
        exports += [candidate_folder / f"submission_{track}.csv" for track in TRACKS]
        if any(path.exists() for path in exports):
            raise FileExistsError("Incomplete or existing stable exports; retain them and use a new version")
        destination = self.root / "final_runs" / unique_id()
        destination.mkdir(parents=True)
        model_folder, submission_folder = destination / "models", destination / "submissions"
        model_folder.mkdir()
        submission_folder.mkdir()
        diagnostics = {}
        started = perf_counter()
        for track in TRACKS:
            model = models.make_hgb_model(track, frozen_path)
            tick = perf_counter()
            model.fit(self.listings)
            fit_seconds = perf_counter() - tick
            homes = data.load(track)
            prices = model.predict(homes)
            model_path = model_folder / f"{track}.joblib"
            joblib.dump(model, model_path, compress=3)
            restored = joblib.load(model_path)
            np.testing.assert_allclose(restored.predict(homes), prices, rtol=1e-12, atol=1e-6)
            pd.DataFrame({"ml_num": homes["ml_num"], "predicted_price": np.round(prices)}).to_csv(
                submission_folder / f"submission_{track}.csv", index=False)
            diagnostics[track] = {"fit_seconds": fit_seconds, "n_train": len(self.listings),
                "n_test": len(homes), "iterations": int(model.estimator.n_iter_), "model_config": model.configuration(),
                "model_reload": "passed", "model_sha256": sha256(model_path)}
            print(f"FINAL {track}: fitted {len(self.listings):,} sales, predicted {len(homes):,} homes; fit={fit_seconds:.2f}s", flush=True)
        if not check_submission.check_all(submission_folder):
            raise ValueError("Candidate submission check failed")
        self.guard()
        if sha256(frozen_path) != frozen_hash:
            raise ValueError("Frozen configuration changed during final fit")
        (self.root / "final_models").mkdir(exist_ok=True)
        candidate_folder.mkdir(parents=True, exist_ok=True)
        for track in TRACKS:
            for source, target in [(model_folder / f"{track}.joblib", self.root / "final_models" / f"{track}.joblib"),
                                   (submission_folder / f"submission_{track}.csv", candidate_folder / f"submission_{track}.csv")]:
                with source.open("rb") as reader, target.open("xb") as writer:
                    shutil.copyfileobj(reader, writer)
        write_json(destination / "environment.json", {**self.env, "created_utc": now(), "elapsed_seconds": perf_counter() - started})
        write_json(destination / "config.json", read_json(frozen_path))
        write_json(destination / "model_diagnostics.json", diagnostics)
        summary = {"version": self.version, "phase": "final", "created_utc": now(),
            "frozen_config_sha256": frozen_hash, "final_run": str(destination.relative_to(self.root)),
            "model_paths": {track: str((self.root / "final_models" / f"{track}.joblib").resolve()) for track in TRACKS},
            "submission_paths": {track: str((candidate_folder / f"submission_{track}.csv").resolve()) for track in TRACKS},
            "verification": {"submission_checker": "passed", "model_reload_predictions": "passed",
                             "baseline_guard": self.guard(), "final_training_cutoff": "2026-06-01",
                             "n_train": len(self.listings), "submission_rows": {t: d["n_test"] for t, d in diagnostics.items()}},
            "diagnostics": diagnostics, "physical_fits": 2,
            "total_successful_physical_fits": self.fit_count() + 2}
        write_json(final_summary, summary, exclusive=True)
        return summary

    def check(self):
        self.read_frozen()
        summary = read_json(self.root / "final_summary.json")
        if summary["frozen_config_sha256"] != sha256(self.root / "frozen_config.json"):
            raise ValueError("Candidate frozen hash differs")
        for track in TRACKS:
            model_path = Path(summary["model_paths"][track])
            if sha256(model_path) != summary["diagnostics"][track]["model_sha256"]:
                raise ValueError("Exported fitted model changed")
            submission = load_predictions(summary["submission_paths"][track])
            homes = data.load(track)
            if check_submission.check_file(summary["submission_paths"][track], homes["ml_num"]):
                raise ValueError("Submission check failed")
            restored = joblib.load(model_path)
            np.testing.assert_array_equal(submission["ml_num"], homes["ml_num"])
            np.testing.assert_array_equal(submission["predicted_price"], np.round(restored.predict(homes)))
        self.guard()
        print("HGB candidate checks, saved-model reloads, and original baseline fingerprints passed.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="hgb_v1")
    parser.add_argument("--phase", required=True, choices=("development", "backtest", "horizon", "predict", "check", "report"))
    args = parser.parse_args()
    if not re.fullmatch(r"hgb_v[1-9][0-9]*", args.version):
        parser.error("Use a version such as hgb_v1")
    if importlib.metadata.version("scikit-learn") != "1.7.2":
        raise RuntimeError("Use the pinned scikit-learn==1.7.2 environment")
    version_root = ROOT / "outputs/hgb_baselines" / args.version
    version_root.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(str(version_root / ".runner.lock"), timeout=0):
            runner = HGBRunner(version_root)
            if args.phase == "development":
                runner.development()
            elif args.phase in ("backtest", "horizon"):
                runner.frozen_evaluation(args.phase)
            elif args.phase == "predict":
                runner.predict()
            elif args.phase == "check":
                runner.check()
            else:
                from analysis.report_hgb_baselines import build_report
                runner.guard()
                path = build_report(runner.root)
                print(f"REPORT {path}", flush=True)
    except Timeout:
        parser.error("Another phase is already running for this version; run phases sequentially")


if __name__ == "__main__":
    main()
