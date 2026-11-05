# Decision Tree → Random Forest baseline：执行 instruction

本文件定义 SSU Data Dash 房价预测的实施顺序、模型接口、实验配置、validation 协议与验收条件。以 CSC311 的 Decision Tree、Bias–Variance、Bagging 和 Random Forest 为理论依据。

**当前交付是 instruction；本文件中的新模型和 experiment runner 尚需实现，不能把建议参数或参考成绩写成已经验证的结果。**

## 1. 目标与实验边界

为 `off_market` 和 `list_aware` 分别建立以下模型：

1. **DT**：一棵经过 complexity control 的 CART Regression Tree。
2. **Bagged Trees**：对 bootstrap samples 训练多棵 CART trees，平均预测。
3. **RF**：在 Bagged Trees 上增加 node-level feature subsampling。

第一轮固定相同的 features、target、training window、validation folds 和 base-tree depth / leaf size，比较 single tree → Bagging → RF 的变化。

- 两个 track 分别 fit 模型，可分别选择 depth / leaf size。
- 两个 track 均学习 `log(sold_price)`；List-aware 仅比 Off-market 多输入 `list_price`。
- 第一轮使用各 fold cutoff 之前的全部 training sales，所有 examples 等权。
- Historical price aggregates、Target Encoding、`remarks`、recent sample weighting、recent training window、`log(sold_price / list_price)` 放在后续独立 ablations。
- 保留 starter 的原有 baseline classes，作为 reference；不改写其定义。
- 不使用真实地图、census 或真实房产记录补充 synthetic location。

完成标准：代码可以复现三个模型在两个 track 上的 temporal evaluation；生成清楚区分 development、holdout 和 horizon backtest 的结果；选定模型可以生成并通过检查的 submission files。

## 2. 工作目录与现有文件

```text
SSU_CaseCompetition/
├── Random Forest baseline/
│   └── INSTRUCTIONS.md             # 本文件
├── SSU-Data-Dash/
│   ├── features.py                 # 复用并扩展 deterministic features
│   ├── models.py                   # 新增模型，保留原有 baselines
│   ├── data.py                     # 复用 load() 与 hide_answers()
│   ├── run.py                      # 保留官方命令及标准 validation 协议
│   ├── check_submission.py         # 复用提交检查
│   └── tests/test_starter.py       # 保留现有 tests
└── SSU_data_dash/
    ├── listings.parquet
    ├── test_off_market.parquet
    └── test_list_aware.parquet
```

先阅读 `handout.md`、`rubric.md`、`DATA_QUALITY.md`、`features.py`、`models.py` 和 `run.py`。检查已有修改，保留已有 data audit 和用户工作。

### 环境准备：现有可执行命令

以下命令从当前机器的实际路径执行。已有可用 virtual environment 时直接复用；否则创建项目 `.venv`。

```bash
cd "/Users/lxb_happypuppy/Downloads/Study/SSU_CaseCompetition/SSU-Data-Dash"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export DATA_DIR="/Users/lxb_happypuppy/Downloads/Study/SSU_CaseCompetition/SSU_data_dash"
python -m pytest -q
```

后续新 terminal session 需要重新 activate environment 和设置 `DATA_DIR`。使用项目固定的 package versions，包括 `scikit-learn==1.7.2`。

## 3. Target 与 prediction contract

定义 `y = sold_price > 0`、`z = log(y)`，使用 natural logarithm。

```python
z_train = np.log(train["sold_price"].to_numpy(dtype=float))
z_pred = estimator.predict(X_homes)
predicted_price = np.exp(z_pred)
```

- `fit()` 前检查 training prices 全部 positive、finite；不通过时定位数据问题。
- 不用任意 clipping 掩盖非法 target 或 prediction。
- Leaf 输出其中 training log targets 的 mean；还原后的 leaf price 是 geometric mean。
- Log-scale squared error 只是 training objective，不是比赛 MdAPE 的直接优化。
- `predict()` 返回原始 price scale 的一维 numeric array；其长度和输入行数一致，行顺序保持不变，所有价格 positive、finite。
- 沿用 starter，在保存 validation / submission predictions 时统一 round price；model selection 的指标数值保留完整 precision，展示时再 round。

对于 Bagged Trees 与 RF，平均发生在 log scale：`exp(mean(tree_log_predictions))`；不要改成 `mean(exp(tree_log_predictions))`。

## 4. 固定 feature schema 与 preprocessing

采用明确的 feature whitelist。`fit(train)` 即使收到完整 training table，也只能读取 whitelist 中的列来生成 predictors；`sold_price` 单独用于 target。

### 4.1 Numeric / binary features

| Feature | 定义 |
|---|---|
| `square_feet` | 复用 `features.square_feet()`：exact 转 numeric，range 取 midpoint |
| `area_is_range` | 原始 `house_area` 是 range 时为 1，否则为 0 |
| `bedrooms` | 保留原值，包括 0 |
| `washrooms` | 保留原值 |
| `parking_total` | 保留原值，包括 0 |
| `garage` | 保留原值 |
| `garage_gt_parking` | `garage > parking_total` 时为 1 |
| `lot_front` | 保留 numeric / NaN |
| `lot_depth` | 保留 numeric / NaN |
| `lot_area` | 两个维度均有效时计算乘积，否则 NaN |
| `lot_missing` | `lot_front` 或 `lot_depth` missing 时为 1；只保留这一份 lot missing indicator |
| `latitude`、`longitude` | 保留 synthetic coordinates |
| `months_since_2021` | `(date_listed - 2021-01-01).days / 30.44`，复用 starter 的计算方式 |
| `list_price` | 仅 List-aware 使用，保留 numeric |

`date_listed` 的解析复用 `data.load()`；不将原始 datetime 直接传给 estimator。

### 4.2 Categorical features

固定为 `house_type_name`、`house_style`、`municipality`、`community`。

```python
OneHotEncoder(handle_unknown="ignore", sparse_output=False)
```

- 每个 fold 在 training set 上重新 fit encoder；validation / test 仅 transform。
- 不把任意 category IDs 当成有序 numeric features。
- 使用 dense numeric output，保留 lot features 中的 NaN；必要时设置 `ColumnTransformer(sparse_threshold=0.0)`。
- 不做 feature scaling。
- 对 RF，feature subsampling 的单位是编码后的 columns，并不是原始 categorical feature groups。

### 4.3 Missingness 与排除字段

Condo / Co-Op 的 lot NaN 是 structural missingness。固定使用支持 NaN 的 `criterion="squared_error"`、`splitter="best"` Decision Tree；不统一填 0 或做全局 median imputation。RF 同样保留这套 dense / NaN preprocessing。

以下字段不得进入 predictor matrix：

| 字段 | 用途 / 原因 |
|---|---|
| `ml_num` | 仅 ID、join key 与提交标识 |
| `sold_price` | 仅 training target |
| `date_sold` | 仅 temporal split 与 evaluation |
| `days_on_market` | Test 中不可用 |
| `external_avm_estimate` | Test 中不可用 |
| `external_avm_estimate_date` | Test 中不可用 |
| `remarks` | 本轮暂不建模，后续单独做 text ablation |
| `list_price` | Off-market 必须排除 |

## 5. Decision Tree 的学习规则

对 node 内的 training examples 集合 `S`：

```text
z_i = log(sold_price_i)
mean_z(S) = mean(z_i for i in S)
I(S) = mean((z_i - mean_z(S))**2 for i in S)

gain = I(S) - len(S_left)/len(S) * I(S_left)
            - len(S_right)/len(S) * I(S_right)
```

每个 node 在全部可用 features 上搜索合法 binary splits，选择最大 gain，再递归处理 children。Numeric split 形如 `x_j <= threshold`；missing routing 交给 estimator 学习。

达到 maximum depth、无法让两个 children 都满足 minimum leaf size，或没有 positive impurity reduction 时停止。Leaf 输出其 training log targets 的 mean。

这是 greedy recursive partitioning，不代表找到了 globally optimal tree；不手工编造房价 threshold 或强制指定 root feature。

### 初始 DT 配置

```python
DecisionTreeRegressor(
    criterion="squared_error",
    splitter="best",
    max_depth=8,
    min_samples_leaf=50,
    min_samples_split=2,
    max_features=None,
    ccp_alpha=0.0,
    random_state=42,
)
```

Depth 8 与 leaf size 50 是待验证的 conservative starting point。`min_samples_leaf=50` 已要求每个合法 split 的两个 children 分别至少有 50 条 examples；即使 `min_samples_split=2`，小于 100 条的 node 也不能 split。

本轮采用 pre-pruning；固定 `ccp_alpha=0.0`，不同时搜索大量近似重复的 complexity controls。Post-pruning 可以另开 ablation。

## 6. Bagging 与 RF 的 controlled comparison

先选择每个 track 的 DT depth / leaf size，再将同样约束用于该 track 的 Bagging 与 RF。

复用 `RandomForestRegressor` 实现 Bagged Trees 和 RF，避免新增两套重复训练逻辑：

```python
RandomForestRegressor(
    n_estimators=300,
    criterion="squared_error",
    max_depth=selected_depth,
    min_samples_leaf=selected_leaf,
    min_samples_split=2,
    max_features=selected_feature_fraction,
    bootstrap=True,
    max_samples=1.0,
    ccp_alpha=0.0,
    random_state=42,
    n_jobs=-1,
)
```

| Experiment | `selected_feature_fraction` | 解释 |
|---|---|---|
| `dt` | 不适用；DT 使用 `max_features=None` | 一棵 tree，全部 training examples |
| `bagging` | `1.0` | Bootstrap trees，每个 node 考虑全部 features |
| `rf` | `0.7` | Bootstrap trees，每个 node 考虑随机 feature subset |

`0.7` 和 300 trees 是预先指定的比较配置，不是已验证的最优值。第一轮不继续搜索 RF 的 feature fraction 或 tree count。

RF 的理论动机是降低 individual trees 的 correlation，以获得更有效的 variance reduction；是否改善这个 case 的 future error 需要实测。所有模型都不能仅靠一个时间 feature 可靠地 extrapolate 未见过的 market trend。

## 7. 模型代码与 runner 接口

保持现有约定：

```python
model = make_model(track)
model.fit(train)                  # train 包含 target；内部按 whitelist 构造 X
prices = model.predict(homes)     # homes 只有 prediction-time fields
```

建议步骤：

1. 在 `features.py` 复用现有面积解析，新增一个共用的 deterministic feature builder。
2. 在 `models.py` 新增一个共用 wrapper，内部保存 track、encoder、feature order 和 estimator。
3. 通过 estimator factory 选择 DT / Bagging / RF，复用 preprocessing、log target 与 price restoration。
4. 保留原有 `ListPriceBaseline`、`NearestSalesBaseline` 和其他已有模型。
5. 新增一个薄 experiment runner：`analysis/run_tree_baselines.py`；核心模型逻辑仍放在 `models.py`。
6. Runner 直接复用 `data.load()`、`data.hide_answers()` 和同一个 model factory，不复制一套不同的 prediction pipeline。
7. `models.make_model(track)` 最终读取冻结配置，供官方 `run.py` 使用；不要在 runtime 自动重新选参数。

下面第 8–10 节的新 runner 命令是**待实现的接口要求**。实现 runner 前，这些命令不能直接执行；官方现有命令仍只有 `validate`、`report`、`predict`、`check`。

## 8. Phase A：development 与配置冻结

### 时间划分

| Phase | Prediction months | 用途 |
|---|---|---|
| Development | 2025-06 至 2025-11 | 选择 DT complexity 与最终模型 family |
| Temporal holdout | 2025-12 至 2026-05 | 冻结配置后的 evaluation |
| Hidden test | 2026-06 至 2026-09 | 最终 submission，无 target 可用 |

每个 prediction month `m`：

```python
train = listings[listings["date_sold"] < m.start_time]
validation = listings[listings["date_sold"].dt.to_period("M") == m]
homes = data.hide_answers(validation, track)
```

每次重新 fit preprocessing 和 model；禁止 random train/test split、random K-fold、默认 `GridSearchCV` folds，以及用 OOB score 替代 temporal evaluation。

### 固定搜索顺序

1. 每个 track 在 development 的六个月上比较 DT 的九组配置：`max_depth in {4, 8, 12}` × `min_samples_leaf in {20, 50, 100}`。
2. 合并同一 track 六个月 predictions，用 pooled MdAPE 选配置；精确并列时先选 smaller depth，再选 larger leaf size。
3. 在选定 depth / leaf size 下，运行同一 development folds 的 Bagging 与 RF。
4. 按 development pooled MdAPE 为每个 track 选定最终 family；精确并列时依次优先 DT、Bagging、RF。
5. 保存并冻结 `frozen_config.json`，记录每个 track 的 family、全部 estimator / preprocessing 参数、seed、features、development windows 与选择指标。
6. 后续 holdout 和 horizon backtest 不再用于修改这份配置。若修改，作为新版本，并承认已经观察过该 holdout。

Development 仅计算 point metrics，无需为九个候选逐一运行包含 bootstrap CI 的完整 `report()`。

实现 runner 后执行：

```bash
python analysis/run_tree_baselines.py --phase development
```

此命令的验收条件：生成九组 DT 的两 track results、选定 complexity 下的三模型比较，以及 `outputs/tree_baselines/frozen_config.json`。

## 9. Phase B：冻结后的 temporal holdout

使用冻结的每个 track depth / leaf size，分别评估 DT、Bagging、RF 及该 track 的原有 starter baseline；所有模型使用相同 holdout rows。

主指标与辅助指标：

```text
APE_i = abs(predicted_price_i - sold_price_i) / sold_price_i
MdAPE = 100 * median(APE_i)
MAPE = 100 * mean(APE_i)
MedianSignedError = 100 * median((predicted_price_i - sold_price_i) / sold_price_i)
```

- **Pooled MdAPE 是合并所有月份后取 median，不是 monthly MdAPE 的 mean。**
- 对每个 track 单独报告 pooled、monthly、property-type、sold-price-decile errors，并同时报告各组 sample count。
- 可用 `ml_num` 从 listings join 回 `community` 做额外分析；原有 validation CSV 没有该列。
- 记录 training log-MSE、training price-scale error、fit time、predict time。DT 额外记录 actual depth、leaf count 和 leaf sample counts。
- Training error 很低而 holdout error 很高是 overfitting 的线索；market drift 也可能造成 gap，不能直接把全部 gap 归为 variance。
- 需要 CI 时复用现有 bootstrap helper；注明其按 homes resample，不能完整表达同月 market shock 的 uncertainty。

实现 runner 后执行：

```bash
python analysis/run_tree_baselines.py --phase holdout
```

此命令读取 frozen config，只做 evaluation，不搜索参数。即使其他模型在 holdout 上表现更好，也不能静默切换最终 family 并继续把该 holdout 描述为 untouched model-selection evidence。

## 10. Phase C：frozen-cutoff horizon backtest

每个 track 只使用 `date_sold < 2026-02-01` 的 sales fit 一次，预测 February–May 2026 的 homes；期间不加入新 labels，也不每月 refit。

比较 DT、Bagging、RF 在 horizon 1、2、3、4 months 及四个月 pooled 的 errors。使用已冻结的 depth / leaf size 和 preprocessing 规则。

```bash
python analysis/run_tree_baselines.py --phase horizon
```

目的：检验最终用 May 底数据预测 June–September 的 frozen-data 场景。不能只依赖 monthly refitting 的成绩描述未来四个月表现。

### 预测时点必须准确表述

本轮沿用 starter 的 sale-month cutoff 协议。对于挂牌早于 cutoff 的房屋，模型可能使用了挂牌之后、cutoff 之前的 sales，因此结果不能自动称为严格的 listing-day valuation。

如未来要评价严格的 listing-day valuation，需要单独建立 listing-time model vintages。若再加入 historical sales features，其 labels 必须满足 `date_sold < min(date_listed, fold_cutoff)`，training rows 也需要逐条遵守各自的 as-of date。

## 11. 结果保存与可复现要求

新 runner 每次创建独立 experiment directory，不覆盖之前结果：

```text
outputs/tree_baselines/
├── frozen_config.json
└── <experiment_id>/
    ├── config.json
    ├── environment.json
    ├── predictions.csv
    ├── metrics.json
    ├── grouped_metrics.csv
    └── model_diagnostics.json
```

`experiment_id` 至少包含 phase、family、track 和唯一 run identifier。`environment.json` 记录 Python / package versions、code revision（可用时）、data identity 与运行时间。

`predictions.csv` 至少保存：`ml_num`、`track`、`model_family`、`phase`、`month`、`training_cutoff`、`house_type_name`、`sold_price`、`predicted_price`。

官方 `run.py` 使用固定输出路径，会覆盖 `outputs/validation_predictions.csv`、reports、figures 和 submissions。每次官方运行结束后，将相关文件保存到该次 experiment directory，再切换模型或运行下一次实验。

## 12. 必要验证与最终 submission

模型实现完成后，保留并运行现有 tests：

```bash
python -m pytest -q
```

为新增模型补充少量有实际意义的 contract checks：

- [ ] Off-market 在 training / prediction 输入中增加或改变 `list_price`，结果不变。
- [ ] Prediction 输入只含 test 可用字段时，可以正常预测。
- [ ] Unseen categorical value 可以 transform / predict，不触发异常。
- [ ] Structural lot NaN 可以经过整个 preprocessing / model pipeline。
- [ ] 输出长度和行顺序正确，全部价格 positive、finite。
- [ ] Fold 的 training rows 全部满足 `date_sold < training_cutoff`。
- [ ] 同一 seed、配置和输入产生可复现结果。

将 development 选定的每 track family 与参数接入 `make_model(track)` 后，运行官方完整流程：

```bash
python run.py validate
python run.py report
python run.py predict
python run.py check
```

前两个命令用于确认最终接入与标准 monthly holdout evaluation 一致；若同样结果已经由 runner 验证且没有新修改，不需要反复重训。

最终 fit 使用全部已发布 training sales，确认 `date_sold < 2026-06-01`，不使用 June–September hidden labels。

两个提交文件为：

```text
submissions/submission_off_market.csv
submissions/submission_list_aware.csv
```

文件必须严格只有 `ml_num,predicted_price` 两列，各自完整覆盖其 test IDs，无重复、缺失或额外 ID。当前数据规模分别为 3,974 / 4,064；实际验收以 checker 读取的 test ID 集合为准。

## 13. 完成 checklist

- [ ] 明确实现 feature whitelist，复用已有面积解析与数据加载。
- [ ] 两个 track 都学习 `log(sold_price)`；仅 List-aware 使用 `list_price`。
- [ ] 完成 DT、Bagging、RF 的共用 wrapper / factory，保留 starter baselines。
- [ ] 完成 development grid，在查看本轮 holdout 前冻结配置。
- [ ] 完成 temporal holdout 与四个月 horizon backtest，分别保存结果。
- [ ] 所有比较使用相同 rows、features、target 和 base-tree constraints。
- [ ] 报告实际 MdAPE / MAPE、各组 sample count、tree diagnostics 与运行时间。
- [ ] 不把参考成绩、建议参数或 theoretical motivation 写成实测提升。
- [ ] 正确解释 sale-month cutoff 与 listing-day valuation 的区别。
- [ ] 运行必要 tests，最终两个 CSV 通过 `python run.py check`。

## 14. 参考资料

- [CSC311：Introduction to Decision Trees](https://www.teach.cs.toronto.edu/~csc311notes/decision_trees/dt_intro.html)
- [CSC311：Information Theory for Decision Trees](https://www.teach.cs.toronto.edu/~csc311notes/decision_trees/dt_infogain.html)
- [CSC311：Decision Tree Learning](https://www.teach.cs.toronto.edu/~csc311notes/decision_trees/dt_learn.html)
- [CSC311：Ensemble Methods](https://www.teach.cs.toronto.edu/~csc311notes/bias_variance_decomposition/ensemble.html)
- [scikit-learn 1.7：DecisionTreeRegressor](https://scikit-learn.org/1.7/modules/generated/sklearn.tree.DecisionTreeRegressor.html)
- [scikit-learn 1.7：Decision Trees / Missing Values Support](https://scikit-learn.org/1.7/modules/tree.html#missing-values-support)
- [scikit-learn 1.7：RandomForestRegressor](https://scikit-learn.org/1.7/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
- 本地比赛资料：`SSU-Data-Dash/handout.md`、`rubric.md`、`DATA_QUALITY.md`、`README.md`。
