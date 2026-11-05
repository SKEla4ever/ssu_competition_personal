HGB v1 archival replication package
Report date: 2026-11-05 America/Toronto

Contents
  code/                  Exact final source snapshot plus hash-verified run.py
                         and 12 prior RF/Bagging reference experiments.
  data/                  Original three parquet inputs, including 150000 sales.
  models/                The two fitted HGB v1 joblib models.
  expected_submissions/  Verified original HGB v1 rounded predictions.
  records/               Original frozen config, summaries, experiment evidence,
                         environments, logs, and self-contained HTML evaluation.
  manifest.json          SHA256 and byte length of every archived payload file.
  verify_and_replicate.py Portable integrity, inference, and frozen-refit helper.

Start here (macOS/Linux, from this replication directory)
  python3.12 -m venv .venv
  .venv/bin/python -m pip install -r code/requirements-lock.txt
  .venv/bin/python verify_and_replicate.py verify
  .venv/bin/python verify_and_replicate.py infer --output ../inference_new
  .venv/bin/python verify_and_replicate.py refit --output ../refit_new

Output directories must be new. Inference writes two submission CSVs and a
comparison.json. Refit trains only the frozen configurations (2 fits), saves
new joblibs, checks saved-model reload, and compares raw predictions and rounded
CSVs to the archive. Exit0 means matching reference; exit1 integrity/config/input
failure; exit2 completed with prediction differences. See comparison.json.
Original environment: Python3.12.5, scikit-learn1.7.2, numpy2.2.6, pandas2.3.3,
pyarrow22.0.0, joblib1.6.0; OMP_NUM_THREADS=4, OPENBLAS_NUM_THREADS=1.
The helper sets DATA_DIR to bundled data before importing archived code.
Use only the trusted joblib files in this locally generated archive.

Full search reproduction (190 fits, optional, from this replication directory)
  export DATA_DIR="$PWD/data"
  export OMP_NUM_THREADS=4
  export OPENBLAS_NUM_THREADS=1
  export MPLBACKEND=Agg
  export MPLCONFIGDIR="$(mktemp -d)"
  export XDG_CACHE_HOME="$(mktemp -d)"
  cd code
  ../.venv/bin/python -m pytest -q
  ../.venv/bin/python analysis/run_hgb_baselines.py --version hgb_v2 --phase development
  ../.venv/bin/python analysis/run_hgb_baselines.py --version hgb_v2 --phase backtest
  ../.venv/bin/python analysis/run_hgb_baselines.py --version hgb_v2 --phase horizon
  ../.venv/bin/python analysis/run_hgb_baselines.py --version hgb_v2 --phase predict
  ../.venv/bin/python analysis/run_hgb_baselines.py --version hgb_v2 --phase check
  ../.venv/bin/python analysis/run_hgb_baselines.py --version hgb_v2 --phase report

Run phases sequentially. If hgb_v2 already exists from another experiment, use a
new hgb_vN. Continue only an unchanged version: source, data paths/content,
protocol, and key package versions are checked by execution_contract.json.
Original records/final_summary.json contains historic absolute paths, and the
original contract includes historic DATA_DIR paths. Keep these records intact;
they are evidence, not a portable resume directory. Physical-run caches are not
included; all45 virtual experiment outputs and original raw inputs are included.
Do not copy records into code/outputs/hgb_baselines/hgb_v2.

Current-model interface
  make_hgb_model(track, config_path) constructs an unfitted frozen-config HGB.
  joblib.load(models/<track>.joblib) restores the fitted model.
  predict(homes) returns unrounded positive prices in input order.
  predict_log(homes) always returns final log(price), including List-aware ratio.
  The old run.py / make_model still use RF/Bagging and do not select HGB.

Evaluation interpretation
Development Jun-Nov2025 selected the configurations. Dec2025-May2026 rolling
backtest and Feb-May2026 fixed-cutoff horizon are reused evaluation periods,
not untouched tests. HGB v1 config was frozen before its new backtest scores.
Hidden test targets are unavailable. Metrics use unrounded prices; exported
submission CSVs use numpy.round(price). 100-MdAPE is not accuracy.
