# HGB v1 Implementation Explanation

HGB v1 predicts house sale prices using two separate **Histogram-based Gradient Boosting (HGB)** models. The project handles feature preparation, target transformations, temporal model selection, and restoration to the original price scale. Scikit-learn 1.7.2 implements the underlying algorithm through `HistGradientBoostingRegressor`. [1] [2] [4]

The central idea is to **add regression trees sequentially, with each new tree correcting the current predictions**. Histograms make the search for tree splits more efficient. [6]

## 1. What the two models learn

Let $y$ denote `sold_price`, $p$ denote `list_price`, and $x$ denote the processed features. Each track has its own prediction function $F_M(x)$ after $M$ boosting iterations. [2] [4]

| Setting | Off-market | List-aware |
|---|---|---|
| Listing price available | No | Yes |
| Training target $z$ | $\ln y$ | $\ln y-\ln p=\ln(y/p)$ |
| Raw HGB output | Log price | Log sale-to-list ratio |
| Restored price | $\hat y=\exp(F_M(x))$ | $\hat y=p\exp(F_M(x))$ |
| Categorical encoding | Native categorical | Dense OneHot |
| Final input columns | 18 | 187 |
| `max_iter` | 800 | 400 |
| `min_samples_leaf` | 20 | 50 |

List-aware learns a **multiplicative correction to the listing price**. For example, if $p=1{,}000{,}000$ and the model outputs $\ln(1.05)$, the predicted sale price is $1{,}050{,}000$. In this track, `list_price` is both a numeric feature and an offset used to restore the prediction. [2]

Both models use `loss="squared_error"`, `learning_rate=0.1`, `max_leaf_nodes=63`, `max_bins=255`, `l2_regularization=0`, `early_stopping=False`, `warm_start=False`, and `random_state=42`. Thus, 800/400 specifies the number of trees, while 63 is the maximum number of leaves **per tree**. [4]

## 2. Feature preparation and categorical encoding

`features.tree_features()` constructs the inputs from an explicit whitelist. [3]

| Information | Implementation |
|---|---|
| Floor area | Convert a `house_area` range to its midpoint, e.g. `900-1099` becomes `999.5`; add `area_is_range`. |
| Property characteristics | Retain `bedrooms`, `washrooms`, `parking_total`, and `garage`; add `garage_gt_parking`. |
| Lot dimensions | Retain `lot_front` and `lot_depth`; add `lot_area = lot_front * lot_depth` and `lot_missing`. |
| Location | Retain `latitude` and `longitude`. |
| Listing date | Compute `months_since_2021 = (date_listed - 2021-01-01).days / 30.44`. |
| Categorical information | Use `house_type_name`, `house_style`, `municipality`, and `community`. |

**Off-market** keeps 14 numeric/binary columns and four categorical columns. The categorical columns use pandas `category` dtype, which HGB recognizes through `categorical_features="from_dtype"`. Category levels are learned from the current training data; unseen categories become missing values at prediction time. [2] [4]

**List-aware** adds `list_price` and applies `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`. The final fit produces 15 numeric/binary columns plus 172 indicator columns, for 187 columns in total. Each fold learns its own categories, so its encoded dimension can differ. An unseen category produces an all-zero block for that feature. [1] [2]

Native categorical splits can put a group of categories on one side, such as `community in {A, C}`. OneHot expands categories into separate 0/1 columns, so expressing the same grouping may require additional splits. **Dense** describes storage: all matrix entries, including zeros, are stored in an ordinary array. It is not a separately evaluated predictive advantage. [2] [6] [12]

Numeric missing values remain `NaN` for HGB's native missing-value handling. There is no mean/median imputation or scaling. `ml_num` identifies records; `sold_price`, `date_sold`, and `days_on_market` do not enter the predictor matrix. The time feature uses the **listing date**, while validation cutoffs use the **sale date**. [2] [3] [5]

## 3. What happens during each boosting iteration

Write $z_i$ for the transformed target defined in Section 1. No `sample_weight` is supplied, so observations receive equal weights. [2]

### Step 1. Initialize a constant prediction

For `squared_error`, scikit-learn uses the per-observation half-squared loss:

$$
\ell_i(F)=\frac12\bigl(z_i-F(x_i)\bigr)^2.
$$

Before adding any trees, the prediction is the mean transformed target:

$$
F_0(x)=\bar z=\frac1n\sum_{i=1}^n z_i.
$$

For Off-market, this means the initial price prediction is $\exp(\operatorname{mean}(\ln y))$, the geometric mean of training prices. [7]

### Step 2. Calculate the current residuals

At iteration $m$, the existing model is $F_{m-1}$. Its residuals are:

$$
r_i^{(m)}=z_i-F_{m-1}(x_i).
$$

For this loss, the negative gradient equals the residual:

$$
-\frac{\partial\ell_i}{\partial F}=r_i^{(m)}.
$$

A positive residual calls for an upward correction; a negative residual calls for a downward correction.

### Step 3. Build a regression tree for the correction

Using the same input features, HGB builds a new tree $T_m$ to fit the current residuals. Feature splits place observations into leaves, with one correction shared by all observations in a leaf. Here, $T_m$ denotes the correction **before learning-rate scaling**. [6] [8]

More precisely, HGB uses gradients and Hessians to compute leaf values. For the current iteration, $g_i=-r_i$ and $h_i=1$. A leaf containing observations $R$ receives:

$$
w_R=-\frac{\sum_{i\in R}g_i}{\sum_{i\in R}h_i+\lambda}
=\frac{\sum_{i\in R}r_i}{|R|+\lambda}.
$$

Because this project sets $\lambda=\texttt{l2\_regularization}=0$, **the leaf value is its mean residual**. This formula omits the tiny denominator safeguard used by the implementation for numerical stability. [4] [8]

### Step 4. Add the scaled correction

The model updates its predictions using `learning_rate=0.1`:

$$
F_m(x)=F_{m-1}(x)+0.1\,T_m(x).
$$

Residuals are recomputed for the next iteration. Earlier trees remain in the ensemble, giving:

$$
F_M(x)=F_0+0.1\sum_{m=1}^{M}T_m(x),
\qquad M\in\{800,400\}.
$$

**Illustrative example.** Suppose an Off-market leaf contains 20 homes with a mean log residual of $0.10$. This iteration adds $0.1\times0.10=0.01$ to their log predictions. On the price scale, predictions are multiplied by $e^{0.01}\approx1.01005$, turning $1{,}000{,}000$ into approximately $1{,}010{,}050$.

This is the connection to gradient descent: each tree approximates a correction in prediction space. Tree split thresholds are chosen through candidate search; they are not updated through backpropagation. [6]

## 4. How histograms accelerate split search

For a continuous feature such as `square_feet`, HGB establishes bin thresholds from the training data. When there are many distinct values, thresholds are based on quantiles. Each feature has at most **255 nonmissing bins**, plus a separate missing-value bin. [9] [10]

At a node during tree construction, HGB:

1. Aggregates observation counts, gradient sums, and Hessian sums by feature bin.
2. Uses those summaries to evaluate splits at candidate bin boundaries.
3. Chooses the highest-gain valid split, subject to restrictions such as `min_samples_leaf`, and continues growing the tree within its leaf limit. [8] [9]

For this project's squared error and zero L2 penalty, split gain measures how much better two child leaves, each predicting its own mean residual, fit the residuals than a single parent leaf.

Binning reduces the number of thresholds to examine. It approximates exact split search, so it can change the resulting trees as well as training speed. **The features are binned; targets and leaf values are not binned and remain real-valued.** Native categorical features use category-group splits rather than thresholds on arbitrary category IDs. [6] [9]

## 5. Training loss versus model selection metric

Training uses squared error on the transformed target. Model selection uses **pooled median absolute percentage error (MdAPE)** on restored, unrounded prices: [2] [5]

$$
\operatorname{MdAPE}
=\operatorname{median}_i\left(100\frac{|\hat y_i-y_i|}{y_i}\right).
$$

For both tracks, $F(x_i)-z_i=\ln(\hat y_i/y_i)$. When the relative error is small:

$$
\ln(\hat y_i/y_i)\approx\frac{\hat y_i-y_i}{y_i}.
$$

This connects the log-scale objective to relative prediction error, but **minimizing squared log error is not equivalent to minimizing MdAPE**. Temporal validation determines which trained configuration performs best on the selection metric.

## 6. How the final configurations were selected

Development uses six temporal folds covering **June-November 2025**, with 12,169 validation observations in total. For example, the September fold trains only on sales with `date_sold < 2025-09-01` and evaluates September sales. Each fit starts from scratch using all available earlier sales. [4] [5] [11]

The search proceeds sequentially:

1. **Compare capacity and encoding.** With 200 iterations and `log_price` targets, compare both encodings across `max_leaf_nodes` in `{15, 31, 63}` and `min_samples_leaf` in `{20, 50}`.
2. **Compare iteration counts.** Retain one capacity setting per track and encoding, fit to 800 iterations, and evaluate the 100/200/400/800 checkpoints through staged predictions.
3. **Select each track's configuration.** Minimize pooled MdAPE across all six months. Use one iteration count per track, rather than selecting a different count for each fold.
4. **Check the List-aware target.** For the already selected List-aware configuration, compare `log_price` with `log_ratio`. [5]

The best retained candidates before the target comparison were: [11]

| Track | Encoding | Iterations | Minimum samples per leaf | Pooled MdAPE |
|---|---|---:|---:|---:|
| Off-market | **Native** | 800 | 20 | **8.864%** |
| Off-market | OneHot | 800 | 50 | 9.053% |
| List-aware | Native | 800 | 50 | 4.370% |
| List-aware | **OneHot** | 400 | 50 | **4.317%** |

All four candidates use `log_price` and a maximum of 63 leaves. These are comparisons between selected configurations; other hyperparameters also differ, so the entire score difference cannot be attributed solely to encoding.

Changing the selected List-aware OneHot model to `log_ratio` then reduced MdAPE from **4.317% to 4.264%**. **Native categorical with `log_ratio` was not evaluated in this search.** The final choices therefore describe the best candidates found by this sequential procedure, not an exhaustive optimum. [5] [11]

Pooled MdAPE is calculated over all validation observations together, not by averaging monthly MdAPEs. Ties are resolved using MAPE, fewer leaves, fewer iterations, larger minimum leaf size, OneHot, and finally `log_price`, in that order. [5]

Because `early_stopping=False`, there is no random internal validation split or automatic early stopping. The recorded `validation_fraction=0.1` is inactive; the iteration counts come from the temporal selection procedure above. [2] [4]

## 7. Final fitting and prediction

After freezing the configuration, each track is fitted to all **150,000 historical sales through May 31, 2026**. The fitted models are saved and used to predict the hidden test properties. [1] [5]

The following illustrates the interface when working from the archive's `replication/code/` directory:

```python
from models import make_hgb_model

# train: historical DataFrame containing sold_price
# homes: DataFrame containing the properties to predict
model = make_hgb_model("off_market", "../records/frozen_config.json")
model.fit(train)
prices = model.predict(homes)

# For List-aware, use track="list_aware".
# Its input DataFrames must also contain positive, finite list_price values.
```

`make_hgb_model()` creates an unfitted model from the frozen configuration. `fit()` prepares features, transforms the target, and calls `estimator.fit(matrix, target)`. `predict()` reuses the learned preprocessing and restores predictions to the price scale. [2]

For List-aware, the underlying `estimator.predict()` returns a **log ratio**, while the wrapper's `predict_log()` adds `log(list_price)` and returns **log price**. The public `predict()` method returns prices. The older `make_model(track)` factory still selects the RF/Bagging baselines. [2]

Metrics use unrounded prices; submission generation applies `np.round(prices)`. There is no additional smearing correction or posthoc calibration. [1] [5]

The reported reused rolling-backtest MdAPEs are **8.607% for Off-market** and **4.292% for List-aware**. Those periods had already been inspected in earlier experiments. They support comparison with prior baselines but are not untouched-test results; hidden test labels were unavailable in this archive. [1]

## Sources

Project references are pinned to commit `4a01288c4b2e8f55267cbba6dff9e29fd14cb6c1`, containing the `hgb_v1_20261105` archive.

- [1 Original implementation and replication report][1].
- [2 models.py][2]: model wrapper, targets, encoding, and price restoration.
- [3 features.py][3]: feature construction and whitelist.
- [4 frozen_config.json][4]: final settings and schema.
- [5 run_hgb_baselines.py][5]: temporal selection and final prediction workflow.
- [11 development_summary.json][11]: candidate configurations and measured scores.
- [6 HGB User Guide][6] and [10 HGB API][10]: scikit-learn 1.7.2 documentation.
- [7 Loss initialization][7], [8 leaf values and split search][8], and [9 binning][9]: version-matched scikit-learn source.
- [12 OneHotEncoder API][12]: dense output and unknown-category handling.

[1]: https://github.com/SKEla4ever/ssu_competition_personal/blob/4a01288c4b2e8f55267cbba6dff9e29fd14cb6c1/HistGradientBoosting/archive/hgb_v1_20261105/HGB_v1_%E6%96%B9%E6%A1%88%E4%B8%8E%E5%A4%8D%E7%8E%B0%E6%8A%A5%E5%91%8A.docx
[2]: https://github.com/SKEla4ever/ssu_competition_personal/blob/4a01288c4b2e8f55267cbba6dff9e29fd14cb6c1/HistGradientBoosting/archive/hgb_v1_20261105/replication/code/models.py
[3]: https://github.com/SKEla4ever/ssu_competition_personal/blob/4a01288c4b2e8f55267cbba6dff9e29fd14cb6c1/HistGradientBoosting/archive/hgb_v1_20261105/replication/code/features.py
[4]: https://github.com/SKEla4ever/ssu_competition_personal/blob/4a01288c4b2e8f55267cbba6dff9e29fd14cb6c1/HistGradientBoosting/archive/hgb_v1_20261105/replication/records/frozen_config.json
[5]: https://github.com/SKEla4ever/ssu_competition_personal/blob/4a01288c4b2e8f55267cbba6dff9e29fd14cb6c1/HistGradientBoosting/archive/hgb_v1_20261105/replication/code/analysis/run_hgb_baselines.py
[6]: https://scikit-learn.org/1.7/modules/ensemble.html#histogram-based-gradient-boosting
[7]: https://github.com/scikit-learn/scikit-learn/blob/1.7.2/sklearn/_loss/loss.py
[8]: https://github.com/scikit-learn/scikit-learn/blob/1.7.2/sklearn/ensemble/_hist_gradient_boosting/splitting.pyx
[9]: https://github.com/scikit-learn/scikit-learn/blob/1.7.2/sklearn/ensemble/_hist_gradient_boosting/binning.py
[10]: https://scikit-learn.org/1.7/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html
[11]: https://github.com/SKEla4ever/ssu_competition_personal/blob/4a01288c4b2e8f55267cbba6dff9e29fd14cb6c1/HistGradientBoosting/archive/hgb_v1_20261105/replication/records/development_summary.json
[12]: https://scikit-learn.org/1.7/modules/generated/sklearn.preprocessing.OneHotEncoder.html
