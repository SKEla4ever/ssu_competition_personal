#!/usr/bin/env python3
"""Verify the archive, reproduce saved inference, or refit its frozen HGB models.

Run this file from the extracted bundle. Data is loaded only from its data/
directory. Inference and refitting write exclusively to a new output directory;
they never overwrite archived models, records, or expected submissions.

Exit codes: 0 success; 1 integrity/configuration/input failure; 2 completed
inference or refit differs from the archived reference. See comparison.json.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter


ROOT = Path(__file__).resolve().parent
TRACKS = ("off_market", "list_aware")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def verify_archive():
    """Check every file recorded in manifest.json before importing saved code."""
    manifest = read_json(ROOT / "manifest.json")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("manifest.json must contain a nonempty files mapping")
    problems = []
    for name, expected in files.items():
        relative = Path(name)
        path = (ROOT / relative).resolve()
        if relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(ROOT):
            problems.append(f"Unsafe manifest path: {name}")
            continue
        if not isinstance(expected, dict) or not {"sha256", "bytes"} <= set(expected):
            problems.append(f"Invalid manifest entry: {name}")
            continue
        if not path.is_file():
            problems.append(f"Missing file: {name}")
        elif path.stat().st_size != expected["bytes"]:
            problems.append(f"Size differs: {name}")
        elif sha256(path) != expected["sha256"]:
            problems.append(f"SHA256 differs: {name}")
    if problems:
        raise ValueError("Archive verification failed:\n" + "\n".join(problems))
    print(f"Archive verification passed: {len(files)} files.", flush=True)
    return {"passed": True, "files_checked": len(files),
            "manifest_sha256": sha256(ROOT / "manifest.json")}


def load_runtime():
    # Apply these settings before importing NumPy, scikit-learn, or archived code.
    os.environ["DATA_DIR"] = str(ROOT / "data")
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MPLBACKEND"] = "Agg"
    sys.path.insert(0, str(ROOT / "code"))
    if importlib.metadata.version("scikit-learn") != "1.7.2":
        raise RuntimeError("Use scikit-learn==1.7.2 from the archived environment")
    import joblib
    import numpy as np
    import pandas as pd
    import check_submission
    import data
    import models
    return joblib, np, pd, check_submission, data, models


def compare_submission(path, expected_path, homes, np, pd, checker):
    problems = checker.check_file(path, homes["ml_num"])
    if problems:
        raise ValueError(f"Submission check failed for {path.name}: {problems}")
    actual = pd.read_csv(path, dtype={"ml_num": str}, float_precision="round_trip")
    expected = pd.read_csv(expected_path, dtype={"ml_num": str}, float_precision="round_trip")
    if list(expected.columns) != ["ml_num", "predicted_price"]:
        raise ValueError(f"Unexpected archived columns: {expected_path}")
    if not np.array_equal(actual["ml_num"], homes["ml_num"].astype(str)):
        raise ValueError("Generated submission order differs from the test input")
    ids_match = np.array_equal(actual["ml_num"], expected["ml_num"])
    prices_match = ids_match and np.array_equal(actual["predicted_price"], expected["predicted_price"])
    differences = None
    if ids_match:
        differences = int(np.count_nonzero(actual["predicted_price"].to_numpy() !=
                                          expected["predicted_price"].to_numpy()))
    return {"submission_checker": "passed", "rows": len(actual),
            "ids_and_order_match": bool(ids_match),
            "rounded_prices_match_exactly": bool(prices_match),
            "rounded_price_differences": differences,
            "generated_csv_sha256": sha256(path), "archived_csv_sha256": sha256(expected_path)}


def run_models(mode, output, verification):
    joblib, np, pd, checker, data, models = load_runtime()
    frozen_path = ROOT / "records" / "frozen_config.json"
    frozen = read_json(frozen_path)
    training = None
    if mode == "refit":
        training = data.load("listings")
        if len(training) != 150000 or training["ml_num"].duplicated().any():
            raise ValueError("Expected exactly 150000 distinct training sales")
        if training["date_sold"].isna().any() or not (
                training["date_sold"] < pd.Timestamp("2026-06-01")).all():
            raise ValueError("Training sales must all be strictly before 2026-06-01")
        target = training["sold_price"].to_numpy(dtype=float)
        if not (np.isfinite(target) & (target > 0)).all():
            raise ValueError("Training sold_price must be positive and finite")
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    results = {}
    started = perf_counter()
    for track in TRACKS:
        homes = data.load(track)
        archived = joblib.load(ROOT / "models" / f"{track}.joblib")
        expected_config = frozen["tracks"][track]["model_config"]
        if archived.configuration() != expected_config:
            raise ValueError(f"Archived {track} model differs from frozen configuration")
        fresh = models.make_hgb_model(track, frozen_path)
        if fresh.configuration() != expected_config:
            raise ValueError(f"Current {track} implementation differs from frozen configuration")
        model = archived
        fit_seconds = None
        if mode == "refit":
            tick = perf_counter()
            model = fresh.fit(training)
            fit_seconds = perf_counter() - tick
            model_path = output / f"{track}.joblib"
            joblib.dump(model, model_path, compress=3)
        prices = np.asarray(model.predict(homes), dtype=float)
        if prices.shape != (len(homes),) or not (np.isfinite(prices) & (prices > 0)).all():
            raise ValueError(f"Invalid predictions for {track}")
        path = output / f"submission_{track}.csv"
        pd.DataFrame({"ml_num": homes["ml_num"], "predicted_price": np.round(prices)}).to_csv(path, index=False)
        result = compare_submission(path, ROOT / "expected_submissions" / path.name, homes, np, pd, checker)
        if mode == "refit":
            reference = np.asarray(archived.predict(homes), dtype=float)
            delta = np.abs(prices - reference)
            restored = joblib.load(output / f"{track}.joblib")
            reload_prices = restored.predict(homes)
            result.update({"n_train": len(training), "training_cutoff_exclusive": "2026-06-01",
                "fit_seconds": fit_seconds, "iterations": int(model.estimator.n_iter_),
                "raw_predictions_match_bitwise": bool(np.array_equal(prices, reference)),
                "raw_predictions_allclose": bool(np.allclose(prices, reference, rtol=1e-10, atol=1e-6)),
                "raw_comparison_rtol": 1e-10, "raw_comparison_atol": 1e-6,
                "max_absolute_price_difference": float(delta.max()),
                "mean_absolute_price_difference": float(delta.mean()),
                "max_relative_price_difference": float((delta / np.abs(reference)).max()),
                "refit_model_reload_allclose": bool(np.allclose(reload_prices, prices, rtol=1e-12, atol=1e-6)),
                "refit_model_sha256": sha256(output / f"{track}.joblib")})
        results[track] = result
        print(f"{track}: {len(homes)} rows; exact rounded reference match="
              f"{result['rounded_prices_match_exactly']}", flush=True)
    matched = all(row["rounded_prices_match_exactly"] for row in results.values())
    if mode == "refit":
        matched = matched and all(row["raw_predictions_allclose"] and row["refit_model_reload_allclose"]
                                  for row in results.values())
    report = {"mode": mode, "status": "matched_reference" if matched else "reference_differences",
        "created_utc": datetime.now(timezone.utc).isoformat(), "archive_verification": verification,
        "frozen_config_sha256": sha256(frozen_path), "python": sys.version,
        "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in
                     ("numpy", "pandas", "scikit-learn", "pyarrow", "joblib")},
        "thread_settings": {name: os.environ[name] for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "elapsed_seconds": perf_counter() - started, "tracks": results,
        "interpretation": "Refitting on another platform may change floating-point predictions or model bytes. "
                          "The reported comparisons, not matching filenames, establish replication."}
    write_json(output / "comparison.json", report)
    print(f"Comparison report: {output / 'comparison.json'}", flush=True)
    return 0 if matched else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("verify", "infer", "refit"),
                        help="verify hashes; infer with saved models; or refit frozen models without search")
    parser.add_argument("--output", type=Path,
                        help="new output folder for infer/refit; must not already exist")
    args = parser.parse_args()
    if args.mode in ("infer", "refit") and args.output is None:
        parser.error("--output is required for infer and refit")
    if args.mode == "verify" and args.output is not None:
        parser.error("verify does not write an output folder")
    if args.output is not None and args.output.expanduser().exists():
        parser.error("Output folder already exists; choose a new folder")
    try:
        verification = verify_archive()
        return 0 if args.mode == "verify" else run_models(args.mode, args.output, verification)
    except (Exception, KeyboardInterrupt) as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
