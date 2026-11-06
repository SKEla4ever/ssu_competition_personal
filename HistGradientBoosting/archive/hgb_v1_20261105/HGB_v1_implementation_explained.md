# HGB v1 项目中的 HistGradientBoosting 实施说明

本项目用房屋信息预测 `sold_price`。核心流程是：**构造 features → 转换 target → 在 log space 中逐轮训练 regression trees → 还原为房价 → 用 temporal validation 选择配置。** 本文依据你提供的报告、同一 archive 的 source code 和 frozen configuration。[1] [2] [4]

准确名称是 **Histogram-based Gradient Boosting（HGB）**。它与 gradient descent 的联系是沿 negative gradient 改进预测；在这里，每轮增加一棵 regression tree。项目调用 scikit-learn 1.7.2 的 `HistGradientBoostingRegressor`，histogram、gradient 和 tree construction 由 sklearn 内部完成。[2] [6]

## 1 两条 tracks 分别学习什么

令 $y$ 为 `sold_price`，$p$ 为 `list_price`，$x$ 为处理后的 features，$F_M(x)$ 为完成 $M$ 轮后的 HGB output。两条 tracks 各训练一个独立 model。[2] [4]

| 项目 | Off-market | List-aware |
|---|---|---|
| 可使用挂牌价 | 否 | 是 |
| Training target $z$ | $\ln y$ | $\ln y-\ln p=\ln(y/p)$ |
| HGB output 的含义 | log price | log sale-to-list ratio |
| 还原为房价 | $\hat y=\exp(F_M(x))$ | $\hat y=p\exp(F_M(x))$ |
| Categorical encoding | native categorical | dense OneHot |
| Final input columns | 18 | 187 |
| `max_iter` | 800 | 400 |
| `min_samples_leaf` | 20 | 50 |

List-aware 相当于在挂牌价基础上学习一个 **multiplicative correction**。例如挂牌价为 1,000,000，若 HGB output 为 $\ln(1.05)$，则预测成交价为 1,050,000。这里 `list_price` **同时作为 numeric feature 和 restoration offset**。[2]

两条 tracks 共同使用 `loss="squared_error"`、`learning_rate=0.1`、`max_leaf_nodes=63`、`max_bins=255`、`l2_regularization=0`、`early_stopping=False`、`warm_start=False`、`random_state=42`。因此 800/400 是 boosting iterations；63 是每棵 tree 的最大 leaf 数，实际可以更少。[4]

## 2 原始房屋信息怎样变成 features

`features.tree_features()` 使用固定的 feature whitelist，主要处理如下。[3]

| 信息 | 实际处理 |
|---|---|
| 房屋面积 | `house_area` 为 range 时取 midpoint，例如 `900-1099 → 999.5`；另加 `area_is_range` |
| 房屋配置 | 保留 `bedrooms`、`washrooms`、`parking_total`、`garage`；增加 `garage_gt_parking` |
| 土地 | 保留 `lot_front`、`lot_depth`；增加 `lot_area = lot_front × lot_depth` 和 `lot_missing` |
| 地理位置 | 保留 `latitude`、`longitude` |
| 挂牌时间 | `months_since_2021 = (date_listed - 2021-01-01).days / 30.44` |
| Categorical features | `house_type_name`、`house_style`、`municipality`、`community` |

Off-market 最终使用 **14 个 numeric/binary columns + 4 个 categorical columns**。List-aware 再加入 `list_price`，四个 categorical columns 经 OneHot 展开，final fit 得到 **15 + 172 = 187 columns**；每个 validation fold 的 category levels 都只从自己的 training data 学习，column 数未必相同。[1] [2]

**Numeric missing values 保留为 `NaN`，没有 mean/median imputation，也没有 scaling。** HGB 根据 split gain 学习 missing samples 应走哪一侧。Off-market 的 unseen category 被转换为 missing；List-aware 使用 `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`，unseen category 对应的 OneHot block 全为 0。[2] [6]

`ml_num` 只用于识别房屋；`sold_price`、`date_sold`、`days_on_market` 等不进入 predictor matrix。时间 feature 来自 **listing date**，而 training/validation 的时间边界按 **sale date** 划分。[3] [5]

## 3 每轮 Gradient Boosting 实际做什么

以下统一用 $z_i$ 表示第 1 节的 transformed target。项目没有传入 `sample_weight`，所以 training samples 使用 equal weights。[2]

### Step 1 从一个 constant prediction 开始

项目使用 squared error；sklearn 内部对应的 per-sample loss 为：

$$
\ell_i(F)=\frac12\bigl(z_i-F(x_i)\bigr)^2.
$$

最初还没有任何 tree，令所有 samples 的预测相同：

$$
F_0(x)=\bar z=\frac1n\sum_{i=1}^n z_i.
$$

因此 Off-market 的初始 **log prediction** 是 $\operatorname{mean}(\ln y)$，还原后的初始 price 是 geometric mean，不是原始 price 的 arithmetic mean。[7]

### Step 2 计算当前 residual 和 negative gradient

第 $m$ 轮开始时，已有 prediction function $F_{m-1}$：

$$
r_i^{(m)}=z_i-F_{m-1}(x_i),
\qquad
-\frac{\partial\ell_i}{\partial F}=r_i^{(m)}.
$$

对本项目的 squared error，**negative gradient 恰好就是 residual**。$r_i>0$ 表示当前预测偏低，需要向上修正；$r_i<0$ 表示偏高，需要向下修正。

### Step 3 训练一棵 tree 来预测 correction

用同一组 features $x_i$ 建立新的 regression tree $T_m$，让它拟合当前 residual。Tree 通过 feature splits 把房屋分到不同 leaves；每个 leaf 内共享一个 correction。[6] [8]

更精确地，HGB 使用 gradient 和 Hessian 来计算 leaf value。这里 $g_i=-r_i$，$h_i=1$；对一个 leaf $R$：

$$
w_R=-\frac{\sum_{i\in R}g_i}{\sum_{i\in R}h_i+\lambda}
=\frac{\sum_{i\in R}r_i}{|R|+\lambda}.
$$

项目设置 $\lambda=\texttt{l2\_regularization}=0$，所以 **leaf value 就是该 leaf 的 mean residual**。公式省略了源码用于 numerical stability 的极小 denominator safeguard。[4] [8]

### Step 4 按 learning rate 加上新 tree

$$
F_m(x)=F_{m-1}(x)+0.1\,T_m(x).
$$

接着重新计算 residual，再训练下一棵 tree。此前的 trees 保留，最终得到：

$$
F_M(x)=F_0+0.1\sum_{m=1}^{M}T_m(x),
\qquad M=800\text{ 或 }400.
$$

**一个数值例子：** 假设 Off-market 某个 leaf 有 20 套房，mean log residual 为 0.10，则这轮对该 leaf 的 log prediction 加上 $0.1\times0.10=0.01$。还原到 price 后，原预测乘以 $e^{0.01}\approx1.01005$；1,000,000 会变成约 1,010,050。这是说明机制的例子，不是项目中的实际记录。

普通 parameter gradient descent 更新一组 parameters；这里的更新对象是 prediction function，每轮通过新增 tree 表达 correction。Tree 的 split 由候选搜索决定，不对 split threshold 做 backpropagation。[6]

## 4 Hist 怎样帮助这棵 tree 找到 split

对 `square_feet` 这类 continuous feature，HGB 先在 training data 上建立 bin thresholds：distinct values 较多时依据 quantiles，最多划分为 **255 个 non-missing bins**；missing values 另有一个 bin。Binning 不是简单的 equal-width intervals。[6] [9]

在每轮 tree construction 的某个 node 内，算法：

1. 按 feature 的 bin 汇总 sample count、gradient sum 和 Hessian sum。
2. 扫描候选 bin boundaries，用这些汇总计算 split gain。
3. 选择满足 `min_samples_leaf` 等限制且 gain 最大的 split，继续生长，直到无法继续或达到最多 63 个 leaves。[6] [8] [9]

在本项目的 squared error、$\lambda=0$ 下，可以把 split gain 理解为：**分成两个 children、分别预测 mean residual，比 parent 只预测一个 mean residual，能减少多少 squared error。**

Histogram 使 split search 无需反复检查所有原始 distinct values。它是对 candidate thresholds 的 approximation，主要改善计算效率；不保证与 exact split search 得到相同的 trees 或更低的 prediction error。**Binning 作用于 features，target 和 leaf output 仍然是 continuous values。**[6] [9]

Categorical splits 另作处理：Off-market 的 native categorical 可以把一组 communities 分到一侧；List-aware 先把 categories 变成 0/1 columns，再对这些 columns 做 numerical splits。因此，List-aware 虽然保留 `categorical_features="from_dtype"`，实际的 dense numeric input 不会触发 native categorical splits。[2] [6]

## 5 为什么训练用 squared error 却用 MdAPE 选 model

两者负责不同环节：**每轮 learning 使用 transformed target 上的 squared error；比较完整候选 models 时使用原始房价上的 pooled MdAPE。**[2] [5]

$$
\operatorname{MdAPE}
=\operatorname{median}_i\left(100\frac{|\hat y_i-y_i|}{y_i}\right).
$$

两条 tracks 都满足 $F(x_i)-z_i=\ln(\hat y_i/y_i)$。当 relative error 较小时：

$$
\ln(\hat y_i/y_i)\approx\frac{\hat y_i-y_i}{y_i}.
$$

这解释了 log target 与 relative error 的联系，但 **minimize squared log error 不等于 minimize MdAPE**。项目通过 validation 检查哪套训练配置最终更符合 MdAPE 这一评价标准。

## 6 800 和 400 iterations 怎样选出来

Development 使用 **2025 年 6–11 月的六个 temporal folds**。例如验证 2025 年 9 月时，只用 `date_sold < 2025-09-01` 的历史成交记录训练，再预测 9 月成交的房屋；各 fold 都从头 fit。[4] [5]

Selection 按以下顺序进行：

1. 比较 `onehot` 与 `native`；先固定 200 iterations，搜索 `max_leaf_nodes ∈ {15, 31, 63}`、`min_samples_leaf ∈ {20, 50}`。
2. 每条 track、每种 encoding 保留一套 capacity；训练到 800 iterations，通过 `staged_predict` 读取 100、200、400、800 轮的 predictions。
3. 每条 track 根据六个月全部 validation rows 的 **pooled MdAPE** 选择统一配置和 iteration 数；不是为每个 fold 各选一个最佳 iteration。
4. 最后仅对选中的 List-aware 配置比较 `log_price` 与 `log_ratio`；development MdAPE 从约 **4.317%** 降至 **4.264%**，因而保留 `log_ratio`。[4] [5]

Pooled MdAPE 是把所有 validation rows 合在一起取 median，不是 monthly MdAPE 的平均。若出现 tie，依次比较 MAPE、更少 leaves、更少 iterations、更大的 `min_samples_leaf`、OneHot、`log_price`。[5]

`early_stopping=False`，所以这里没有随机抽取 10% data 做 internal early stopping；记录中的 `validation_fraction=0.1` 不生效。800/400 由上述 temporal validation 选定。这是 sequential search 的结果，不能视为所有 hyperparameter combinations 的 global optimum。[2] [4] [5]

## 7 最后怎样生成预测

配置冻结后，每条 track 使用全部 **150,000 条、截至 2026-05-31 的历史 sales** 重新 fit。Off-market 训练 800 棵 trees；List-aware 训练 400 棵。项目保存 fitted model，并对 hidden prediction rows 输出房价。[1] [5]

从 archive 的 `replication/code/` 目录理解调用关系，可以看这几行：

```python
from models import make_hgb_model

# train: 包含 sold_price 的历史 DataFrame
# homes: 待预测房屋 DataFrame
model = make_hgb_model("off_market", "../records/frozen_config.json")
model.fit(train)
prices = model.predict(homes)
# List-aware 将 track 改为 "list_aware"，homes 必须包含 list_price。
```

`make_hgb_model()` 创建一个尚未 fitted 的 model；`fit()` 内部完成 feature preprocessing、target transformation，并调用 `estimator.fit(matrix, target)`。`predict()` 内部完成同样的 feature transformation，再把 output 还原为 price。List-aware 的 `predict_log()` 已加回 `log(list_price)`；底层 `estimator.predict()` 才是 log ratio。[2]

项目对未 rounding 的 prices 计算 metrics，生成 submission 时才使用 `np.round(prices)`。本版本没有额外的 smearing correction 或 posthoc calibration。旧入口 `make_model(track)` 仍返回 RF/Bagging；使用 HGB 时应调用上面的专用入口，或加载已保存的 HGB model。[1] [2] [5]

报告中的 reused rolling backtest MdAPE 为 Off-market **8.607%**、List-aware **4.292%**。这些 periods 在早期实验中已被观察，适合用于方案比较；hidden test labels 尚未公开，不能把这两个数当作 hidden test performance。[1]

## Sources

Project sources 固定到 GitHub snapshot `4a01288c4b2e8f55267cbba6dff9e29fd14cb6c1`，对应 `hgb_v1_20261105` archive。

- [1 原始方案与复现报告][1]
- [2 models.py][2]：`HistPriceModel` 与 `make_hgb_model()`。
- [3 features.py][3]：feature whitelist 与 feature construction。
- [4 frozen_config.json][4]：最终 hyperparameters、encoding 和 target。
- [5 run_hgb_baselines.py][5]：temporal validation、selection 与 final predictions。
- [6 sklearn 1.7.2 HGB User Guide][6]；[API documentation][10]。
- [7 sklearn loss initialization][7]、[8 leaf values 与 split search][8]、[9 feature binning][9]：与项目版本一致的官方 source code。

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
