# SSU Data Dash: predict Toronto home prices

Start with the case handout, [`handout.md`](handout.md), and the judging rubric, [`rubric.md`](rubric.md).

Two tracks:
- **Off-market:** value a home with no list price.
- **List-aware:** predict a listed home's sale price, knowing its list price.

Train on 150,000 sales (January 2021 to May 2026). Predict homes that sold June to September 2026.

**Data:** [huggingface.co/datasets/Baliyang/SSU_data_dash](https://huggingface.co/datasets/Baliyang/SSU_data_dash).
All of it is synthetic.

## Setup

Python 3.10 or newer (3.11+ on macOS 27):

```
pip install -r requirements.txt
```

The data downloads itself on first use. No Python? Use the
[Colab notebook](https://colab.research.google.com/github/erwinzhang7/SSU-Data-Dash/blob/main/notebooks/quickstart.ipynb).
Prefer R? See `r/baseline.R`.

## Commands

```
python run.py validate   # test on Dec 2025 to May 2026
python run.py report     # results table and a figure
python run.py predict    # write both submission files
python run.py check      # check them
```

On Windows, use `py` if `python` is not found.

## What to edit

- **`features.py`**: the features.
- **`models.py`**: the models. Two simple baselines to beat: list price (list-aware) and the
  10 nearest earlier sales (off-market). `FeatureModel` (gradient-boosted trees) is ready to switch on.

## Tree baseline experiments

The DT / Bagging / RF implementation follows `../Random Forest baseline/INSTRUCTIONS.md`.
Use the pinned requirements in a project virtual environment and point `DATA_DIR` at the supplied data folder.
Run these phases in order:

```
python analysis/run_tree_baselines.py --phase development
python analysis/run_tree_baselines.py --phase holdout
python analysis/run_tree_baselines.py --phase horizon
python -m pytest -q
python run.py validate
python run.py report
python run.py predict
python run.py check
python analysis/summarize_tree_baselines.py
```

Development selects each track's tree complexity and family, then creates
`outputs/tree_baselines/frozen_config.json`. `models.make_model(track)` reads that frozen
configuration; it never tunes parameters at prediction time. The original baseline classes
are preserved for comparison. Development refuses to overwrite an existing frozen selection.

Every phase creates independent experiment folders with predictions, unrounded metrics,
grouped errors, timing, tree diagnostics and provenance. `predicted_price` is rounded for
compatibility with the official pipeline; `predicted_price_raw` is used for model selection
and experiment metrics. The official report uses rounded prices, so its last decimal digits
may differ. The HTML evaluation report and summary CSVs are written to
`../Random Forest baseline/results/`.

The four-month horizon backtest fits once before February 2026 and predicts February–May
without updating labels. Both temporal protocols use sale-month cutoffs and do not establish
strict listing-day valuation. Hidden-test accuracy is unavailable because test prices are withheld.
`run.py predict` also saves each fitted pipeline to `outputs/fitted_models/<track>.joblib`.
From the repository root, reload it with `joblib.load(path)` and call `model.predict(data.load(track))`.
The frozen JSON and experiment outputs are ignored by Git; preserve them with the deliverables
when sharing an already selected model.

After the holdout run, `python analysis/finalize_tree_baselines.py` can replace the four
official commands above. It verifies that the selected runner predictions use the same
frozen factory, features, data, cutoffs and rounding as the official validation pipeline,
reuses them for `run.report()`, then calls the official prediction and checker functions.
It also verifies saved-model reloads and archives all official outputs in a unique final
experiment folder. This avoids retraining identical monthly models.

## HistGradientBoosting experiments

The versioned HGB workflow follows
[`../HistGradientBoosting/INSTRUCTIONS.md`](../HistGradientBoosting/INSTRUCTIONS.md).
It keeps the original factory, fitted models and submissions as references. With
`DATA_DIR` set, run the HGB phases sequentially:

```
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
python analysis/run_hgb_baselines.py --version hgb_v1 --phase development
python analysis/run_hgb_baselines.py --version hgb_v1 --phase backtest
python analysis/run_hgb_baselines.py --version hgb_v1 --phase horizon
python analysis/run_hgb_baselines.py --version hgb_v1 --phase predict
python analysis/run_hgb_baselines.py --version hgb_v1 --phase check
python analysis/run_hgb_baselines.py --version hgb_v1 --phase report
```

Development selects one configuration per track using pooled temporal MdAPE, then
freezes it. Completed folds are cached by source/data/config identity; an interrupted
run resumes without repeating completed fits. Only one phase may write a version at
a time. Later phases reuse the frozen selection. These backtest periods were already
observed during earlier work, so the HGB reports label them as reused evaluation.

Candidate models are saved under `outputs/hgb_baselines/hgb_v1/final_models/`,
candidate CSVs under `submissions/hgb_v1/`, and self-contained HTML reports under
`outputs/hgb_baselines/hgb_v1/reports/`. Create a selected model with
`models.make_hgb_model(track, config_path)` or reload a fitted pipeline with joblib.

## Validation

`validate` predicts each month from the sales before it, as in the real test. A random split lets
the model see the future and makes your score look better than it is. Judges compare your
validation error with your hidden-test error.

## Submission

One PDF report (template: `report/report.tex`, 4 pages max) that links to:
1. your video on YouTube, 5 minutes max (unlisted is fine, not private);
2. your GitHub repository (public, or shared with the organisers), holding your code and two files:

```
submissions/submission_off_market.csv    one row per home in test_off_market
submissions/submission_list_aware.csv    one row per home in test_list_aware

ml_num,predicted_price
SYN-0123456,815000
```

`predict` writes both; run `check` before you hand in. Deadline: see the case email.

## Rules

- Outside data only if it covers May 2026 or earlier. List every source with its URL.
- Neighbourhoods, coordinates and streets are synthetic, so map data will not line up.
- No real listing or sales records.
- Do not try to recover the hidden test prices.
