# HistGradientBoosting 房价模型实施方案

为 Off-market 和 List-aware 分别建立可复现的 HistGradientBoostingRegressor，验证更灵活的 model family 能否降低当前 error。先在相同 features 上比较 estimator，再分别比较 categorical representation、capacity、iterations 和 List-aware target；每一步保留可追溯的 predictions。

HGB wrapper 和 experiment runner 已实现，下文为 hgb_v1 的运行协议；各 phase 的实际完成状态以该版本的 experiment_manifest.json 为准。现有 RF 和 Bagging 的 frozen configuration、fitted models、reports 和 submissions 继续作为 reference。

## 1 已有证据与本轮目标

2025-11 单月 probe 使用 138,064 条 earlier training sales 和 1,867 条 validation homes，固定原 feature whitelist、training rows 与 seed。

| Track | Probe model | Training MdAPE | November MdAPE |
|---|---|---:|---:|
| Off-market | RF 100 trees depth 12 leaf 20 | 14.292% | 18.116% |
| Off-market | RF 100 trees depth 20 leaf 5 | 11.140% | 15.012% |
| Off-market | HGB OneHot log price | 10.095% | 11.563% |
| List-aware | RF 100 trees depth 12 leaf 50 | 5.285% | 4.996% |
| List-aware | HGB OneHot log price | 4.961% | 4.337% |
| List-aware | HGB OneHot log ratio | 4.899% | 4.299% |

HGB probe 使用 200 iterations、learning_rate=0.1、max_leaf_nodes=31、min_samples_leaf=20、l2_regularization=0 和 early_stopping=False。这些结果支持开展完整 temporal evaluation，不能替代六个月 pooled 成绩或 hidden-test 成绩。

证据：[probe metrics](<../Random Forest baseline/diagnostics/20261030T225237Z_17e1ee90/metrics.csv>)、[probe configuration](<../Random Forest baseline/diagnostics/20261030T225237Z_17e1ee90/variants.json>)、[原 baseline report](<../Random Forest baseline/results/evaluation_report.html>)。

本轮主要回答：HGB 的收益能否跨月份保持；native categorical 是否改善 community representation；更高 capacity 是否同时改善 training 和 future error；List-aware residual target 是否在完整 development pool 上改善价格预测。

## 2 版本与 evaluation 边界

本轮正式 HGB 版本记为 `hgb_v1`，属于项目的下一轮 adaptive model development。

| 数据时间 | 本轮用途 | 结果表述 |
|---|---|---|
| 2025-06 至 2025-11 | 比较配置、选择一个统一 max_iter 和最终 target | Development selection；November 已参与过 probe |
| 2025-12 至 2026-05 | 配置冻结后的 rolling monthly evaluation | Reused temporal backtest；旧 holdout 已观察 |
| 2026-02 至 2026-05 | 固定 February cutoff 的四个月预测 | Reused horizon backtest |
| 2026-06 至 2026-09 | 生成 hidden-test predictions | Target 未公开，不计算 accuracy |

继续使用原六个 development months，保证与 RF 和 Bagging 的 rows/cutoffs 可比较。Development metrics 用于 selection，不作为 unbiased generalization estimate；不存在因为重新命名或重排这些月份而获得的新 untouched holdout。

在读取本版本 backtest 成绩前，冻结每个 track 的 encoding、target 和全部 estimator 参数。后续若依据 backtest 再改配置，建立新版本并保留旧版本。

## 3 Features 与 preprocessing

复用 `SSU-Data-Dash/features.py` 的 `tree_features(homes, track)`，保持第一轮 HGB 与原 baseline 的 raw information 相同。

- Numeric/binary：square_feet、area_is_range、bedrooms、washrooms、parking_total、garage、garage_gt_parking、lot_front、lot_depth、lot_area、lot_missing、latitude、longitude、months_since_2021。
- Categorical：house_type_name、house_style、municipality、community。
- List-aware 额外输入 list_price；Off-market 完全忽略该字段，即使 caller 提供它。
- 复用 square_feet range midpoint 和现有 date parsing。保留 structural lot NaN；不做 scaling 或统一 imputation。
- ml_num、sold_price、date_sold、days_on_market、external AVM 字段和 remarks 不进入 predictor matrix。sold_price 仅作 target，date_sold 仅用于 split/evaluation。

### OneHot path

沿用原 train-only OneHotEncoder，`handle_unknown="ignore"`、`sparse_output=False`，numeric columns 保留 NaN。输入 HGB 的 matrix 为 dense numeric；`categorical_features="from_dtype"` 在这个 numeric matrix 上不会启用 categorical splits，与 probe 一致。

### Native categorical path

仍使用同一个 deterministic feature builder，仅改变四个 categorical columns 的 representation。

1. 每个 fold 只从其 training set 学习 category levels，并保存在 model state。
2. 将四列转换为使用固定 training levels 的 pandas Categorical；predict 时沿用同一组 levels 和 feature order。
3. Unseen value 映射为 missing，不将 validation/test categories 与 training categories 做 union。
4. 保持 DataFrame 及 categorical dtype 进入 estimator，设置 `categorical_features="from_dtype"`。不能在转换成纯 numeric array 后把 category codes 当作连续数值使用。
5. 每个 categorical column 检查 training cardinality 不超过 max_bins；本数据 community 的 145 类可由 max_bins=255 容纳。超限时显式报错，由独立方案决定处理方式。

HGB 支持 native categorical 和 NaN；类别映射、binning 和 missing routing 都只在 training fold 中学习。[Categorical support](https://scikit-learn.org/1.7/modules/ensemble.html#categorical-features-support)

## 4 Target 与 prediction contract

### Log price

两个 track 的 reference target 均为 natural log：

```python
z_train = np.log(train["sold_price"].to_numpy(dtype=float))
log_price = estimator.predict(X_homes)
price = np.exp(log_price)
```

### List aware log ratio

仅 List-aware 可选，保持其他 features 和参数一致：

```python
z_train = (np.log(train["sold_price"].to_numpy(dtype=float))
           - np.log(train["list_price"].to_numpy(dtype=float)))
log_ratio = estimator.predict(X_homes)
log_price = np.log(homes["list_price"].to_numpy(dtype=float)) + log_ratio
price = np.exp(log_price)
```

fit 前检查 sold_price positive/finite；ratio mode 同时检查 training 和 prediction list_price positive/finite。Off-market 拒绝 ratio mode。非法值应定位并报错，不用 clipping 掩盖。

`predict(homes)` 返回原 price scale 的一维 positive/finite array，长度和顺序与输入一致。若提供 `predict_log()`，两个 target modes 都返回最终 **log(price)**，方便共用 diagnostics。Raw estimator output 在 ratio mode 中才是 log ratio。

Model selection 使用 unrounded prices。保存 predictions 时同时保留 `predicted_price_raw` 和 rounded `predicted_price`；submission 使用 rounded prices。Log-scale squared error 是 training objective，主 selection metric 仍为 price-scale pooled MdAPE。

## 5 Reference estimator

从已运行 probe 的配置开始：

```python
HistGradientBoostingRegressor(
    loss="squared_error",
    learning_rate=0.1,
    max_iter=200,
    max_leaf_nodes=31,
    max_depth=None,
    min_samples_leaf=20,
    l2_regularization=0.0,
    max_features=1.0,
    max_bins=255,
    categorical_features="from_dtype",
    early_stopping=False,
    warm_start=False,
    random_state=42,
)
```

使用项目 pinned scikit-learn==1.7.2。HGB 没有 `n_jobs` 参数；建议首轮采用与 probe 相同的 `OMP_NUM_THREADS=4`、`OPENBLAS_NUM_THREADS=1`，串行执行 folds 并记录 thread settings。不要把 RF 的参数字典整体传入 HGB。

显式设置 early_stopping=False：默认 auto 在大样本上会启用内部 validation，默认 validation_fraction 为 0.1。本轮由 temporal development folds 选择固定 iterations。[Estimator API](https://scikit-learn.org/1.7/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html)

## 6 Development 执行顺序

每个 prediction month m 都重新 fit preprocessing 和 estimator：training rows 满足 `date_sold < m.start_time`，validation rows 的 sale month 等于 m。复用 `data.hide_answers()`；每个 candidate 使用相同 IDs、targets 和 cutoffs。

### A Reference comparison

对两个 tracks 分别执行 OneHot/log_price 和 native/log_price，固定第 5 节的 200-iteration 配置，共 24 fits。

OneHot HGB 与原 RF/Bagging reference 保持相同 features/target/rows，各自 estimator 参数清楚列出；native 与 OneHot HGB 只改变 categorical representation。既有 RF/Bagging predictions 在 data/code/config identity 相符时直接复用，不重复训练。List-aware 同时展示原已选 Bagging，避免只选择较弱 reference 来比较。

Reference 的 code identity 指 baseline 相关 classes/functions、features、参数和依赖版本的实现一致性。向 models.py 新增 HGB class 会改变 whole-file SHA-256：应保留新旧 hashes，并对照旧 source snapshot 检查相关代码，必要时做小数据 behavior-equivalence checks；不能把不同 hashes 声称相等，也不应仅因新增无关 class 就重训 reference。若确实改变 baseline 行为，额外 comparison/refits 另记预算。

### B Capacity comparison

保留两条 encoding paths，在每个 track 的六个 development folds 上比较：

```text
max_leaf_nodes   ∈ {15, 31, 63}
min_samples_leaf ∈ {20, 50}
max_iter        = 200
learning_rate   = 0.1
l2_regularization = 0.0
```

每个 track/encoding 有 6 个 capacity configs，合计 144 fits，包含 Stage A 的 24 fits。相同输入、config 和 preprocessing 的结果按 fingerprint 复用，新增 120 fits。

每个 track/encoding 按 development pooled MdAPE 保留一个 capacity config；此时保留两条 encoding candidates，避免只凭默认 capacity 就淘汰其中一条。

### C Iteration comparison

对 Stage B 的四条 retained paths，即两个 tracks 各两种 encoding，使用完全相同的 training fold，fit 至 max_iter=800，并用 `staged_predict()` 取得预先声明的 checkpoints：

```text
max_iter checkpoints ∈ {100, 200, 400, 800}
learning_rate        = 0.1
```

合计新增 24 个 physical fits。每个 checkpoint 单独保存 predictions 和 metrics，并记录 `iterations_trained=800` 与 `iterations_evaluated`，避免将 800-tree model 的普通 predict 当作较早 checkpoint 的结果。先校验 staged 200 与独立 fit 200 的 predictions 一致。

同一个 checkpoint 的 training、recent-training 和 validation diagnostics 必须使用相同 iterations_evaluated。不能将 800-iteration training error 与较早 checkpoint 的 future error 拼成一组 gap。

**按合并六个月 predictions 的 MdAPE 选择每个 track 的统一 encoding、capacity 和 max_iter。** 不给每个 evaluation month 各挑一个事后最优 iteration。上述 development scores 是 hyperparameter selection scores。

每个 fold 都 fresh fit，禁止跨 folds warm-start。冻结后的 backtest/final fit 直接训练所选 max_iter，不再训练至 800 或搜索 checkpoints。

这是 budget-limited sequential search：Stage B 在 200 iterations 下的 capacity ranking 可能在其他 iteration counts 下改变。最终配置是本轮 retained candidates 中的最优结果，不等同于 capacity × iterations 的 joint global optimum。

### D List aware target comparison

固定 Stage C 选中的 List-aware encoding、capacity 和 max_iter，只把 target 从 log_price 改为 log_ratio，新增 6 fits。按相同 raw-price pooled MdAPE 比较并选择 target；保留两种 target 的结果。不能因为 November probe 的 4.299% 就预设 ratio 必然胜出。

### Selection 规则与预算

主指标为 full-precision development pooled MdAPE；精确并列时依次比较较低 MAPE、较少 max_leaf_nodes、较少 max_iter、较大 min_samples_leaf，最后按 onehot、native 的顺序选择 encoding。List target 精确并列时保留 log_price。

核心 development 上限为 **174 physical fits**：144 capacity/reference fits、24 iteration fits、6 target fits。Checkpoint predictions 不重复算 physical fits。随后 backtest/horizon 共 14 fits，最终 fit 2 次，核心流程共 190 fits。这个预算不含小数据 contract checks、失败重试或额外扩展实验。

首轮固定 learning_rate=0.1、l2_regularization=0、全历史 training window、equal sample weights 和原 features。Learning rate、L2、recent weighting、historical price features、remarks 和其他 losses 放在下一轮独立 ablations。若最佳 capacity/iteration 持续位于 grid 边界，报告该限制；在 development 中扩展的实验需单独记录，不能由后续 backtest 成绩触发后仍声称配置未变。

## 7 配置冻结与 backtest

将每个 track 的 target、encoding、features、category handling、全部 estimator parameters、seed、selected max_iter、development windows、selection metric、data/code hashes 和已观察 holdout 的事实写入：

```text
SSU-Data-Dash/outputs/hgb_baselines/hgb_v1/frozen_config.json
```

冻结采用 exclusive creation；已存在时不能覆盖。原 `outputs/tree_baselines/frozen_config.json` 保留原值。

Rolling backtest 使用 2025-12 至 2026-05，每月重新 fit 全部 earlier sales；每个 track 只评估冻结的最终 HGB，复用原 baseline 相同 rows 的结果作 comparison。

Horizon backtest 每个 track 只用 `date_sold < 2026-02-01` 的 sales fit 一次，依次预测 February–May，不加入新 labels。报告 horizon 1/2/3/4 及四个月 pooled errors。

两类结果均为 reused/adaptive backtest。固定 February 的表现不能自动证明 June–September accuracy。整个流程沿用 sale-month cutoff；挂牌早于 cutoff 的 homes 不具备严格 listing-day valuation 的解释。

## 8 Metrics 与 error analysis

每个 track 单独报告 pooled MdAPE、MAPE、MedianSignedError，并保存 monthly、property-type、community、sold-price-decile 的 metrics 和 sample counts。Pooled MdAPE 是合并 rows 后取 median，不是 monthly MdAPE 的 mean。

每个 fold 还需记录：

- Full-history training log-MSE、price-scale MdAPE/MAPE/signed error。
- 该 fold cutoff 前最近 12 months training cohort 的相同 metrics。
- Future-month log-price MSE、price-scale errors 和 sample count。
- 实际 n_iter_、每阶段 validation metrics、fit/predict/preprocessing time、encoded feature count 和类别数量。

Training metrics 是 in-sample；六个重叠 full-history training sets 的 median 均值与 future pooled median 不是同分布的 standard generalization gap。优先比较同一个 cohort 在不同 models 下的变化，并展示每月时间轨迹。

按真实 sold_price 分 decile 的 extreme-price error pattern 不能单独证明 underfitting，也不能用于 prediction-time correction。重点结合 capacity curves、community bias、recent training error 和 temporal drift 分析。

需要 CI 时复用现有 home-level bootstrap helper；比较 models 时使用相同 resampled IDs。注明 CI 不完整表达共同 market shock，也不修正 adaptive model selection。

## 9 工程接口与结果结构

在 `models.py` 新增 `HistPriceModel` 和独立 HGB factory，保持现有契约：

```python
model = make_hgb_model(track, config_path)
model.fit(train)
prices = model.predict(homes)
```

Wrapper 的配置使用 `encoding in {"onehot", "native"}` 和 `target_mode in {"log_price", "log_ratio"}`；其他取值应明确报错。

新增 wrapper 复用 `features.tree_features()`、`data.load()` 和 `data.hide_answers()`。若抽取共用 OneHot preprocessing helper，只移动构造逻辑，并用旧 pipeline predictions 验证行为一致。原 `make_model(track)` 继续读取旧冻结配置。

新增薄 runner `analysis/run_hgb_baselines.py`。可复用纯 helpers，例如 temporal_fold、point_metrics、grouped_metrics、write_json 和 sha256；原 tree runner 的 evaluate、family/output globals 含有专用逻辑，不能靠修改一个 family 字符串直接接入。

每次创建独立目录，禁止覆盖实验：

```text
outputs/hgb_baselines/hgb_v1/
├── frozen_config.json
├── experiment_manifest.json
└── <phase_track_encoding_target_unique_id>/
    ├── config.json
    ├── environment.json
    ├── predictions.csv
    ├── metrics.json
    ├── grouped_metrics.csv
    ├── model_diagnostics.json
    └── iteration_metrics.csv
```

Environment 记录 Python/packages、OpenMP settings、run time、Git revision/dirty state、所有实际模型与 runner source hashes、data identities。对 physical fit 与 checkpoint evaluation 分别记账，避免重复累计 fit time。

Predictions 至少包含 ml_num、track、model_family、phase、month、training_cutoff、encoding、target_mode、iterations_evaluated、house_type_name、community、sold_price、predicted_price_raw 和 predicted_price。Test predictions 不含 sold_price。

## 10 执行命令

从 `SSU-Data-Dash` 运行，复用已有 `.venv` 和 pinned requirements，按顺序执行：

```bash
export DATA_DIR="/Users/lxb_happypuppy/Downloads/Study/SSU_CaseCompetition/SSU_data_dash"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1

.venv/bin/python analysis/run_hgb_baselines.py --version hgb_v1 --phase development
.venv/bin/python analysis/run_hgb_baselines.py --version hgb_v1 --phase backtest
.venv/bin/python analysis/run_hgb_baselines.py --version hgb_v1 --phase horizon
.venv/bin/python analysis/run_hgb_baselines.py --version hgb_v1 --phase predict
.venv/bin/python analysis/run_hgb_baselines.py --version hgb_v1 --phase check
.venv/bin/python analysis/run_hgb_baselines.py --version hgb_v1 --phase report
```

Development 完成 A–D 并冻结配置。其余 phases 只读取冻结配置；predict 使用全部已发布 sales，并断言 `date_sold < 2026-06-01`。

## 11 验证与交付

保留现有 tests，并为 HGB 补充有实际意义的 contracts：Off-market list_price 隔离和 ratio rejection；test-only fields；native/OHE unseen categories；category map 在 predict 后不变化；structural lot NaN；price restoration 与 predict_log 语义；行顺序、positive/finite 输出；staged 与 standalone checkpoint 一致；严格 temporal cutoff；horizon 只 fit 一次；same seed/config reproducibility；joblib reload 一致。

保存旧 frozen configuration、fitted model files 和根目录 submissions 的 SHA-256，完成后确认未变。新的 fitted HGB models 和 candidate submissions 独立保存：

```text
outputs/hgb_baselines/hgb_v1/final_models/off_market.joblib
outputs/hgb_baselines/hgb_v1/final_models/list_aware.joblib
submissions/hgb_v1/submission_off_market.csv
submissions/hgb_v1/submission_list_aware.csv
```

调用现有 `check_submission.check_all(new_folder)`：严格只有 `ml_num,predicted_price` 两列，完整覆盖各自 test IDs，无重复、缺失、额外 IDs 或非法价格。当前数据对应 3,974 / 4,064 rows，验收以实际 checker ID 集合为准。

Final fit 先写入唯一 final-run directory，完成 model reload 和 CSV checks 后，再以 exclusive creation 导出上述固定候选路径。固定路径已有文件时拒绝覆盖；有完整且 fingerprints 一致的 verification 时可只读复用，否则建立新版本或保留独立 run。Report 写入 `outputs/hgb_baselines/hgb_v1/reports/<unique_id>/`，不调用会覆盖原 reports 的 `run.report()`。

最终 report 同时列出原冻结 baseline、新 HGB development selection、reused monthly backtest 和 reused horizon backtest，清楚标记 target/encoding；模型优劣以 future errors 评价，training error 用于 diagnosis。将 HGB 接入默认 factory 或替换根目录 submissions 应作为明确的版本变更，并记录采用的 evidence，不能静默覆盖旧 reference。

## 12 后续实施验收条件

- 两个 tracks 都完成 OneHot 和 native categorical 的完整 temporal comparison。
- Capacity 和 iteration selection 使用相同 development rows，并为每个 track 冻结一个统一配置。
- List-aware residual target 的效果有完整六个月 comparison；Off-market 无 list-price leakage。
- Training、recent-training 和 future errors 分开保存，报告清楚区分 selection 与 reused evaluation。
- 新旧 source/config/data identities 和 fitted-model reload 可以核验。
- Candidate CSVs 通过原 checker；旧 reference artifacts 保持完整。
- 单月 probe 和建议参数始终与完整实测结果分开表述。
