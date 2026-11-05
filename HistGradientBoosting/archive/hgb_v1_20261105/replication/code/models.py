"""EDIT ME: the models. Every model has the same two steps:

    model.fit(train)        train = past sales, including sold_price and date_sold
    model.predict(homes)    homes = homes to value, with only the columns of a test file;
                            returns one predicted sold price per home

make_model() picks the model for each track. The two starting models are baselines:
deliberately simple, so you can measure how much your own ideas add.
"""
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

import features


def make_model(track):
    """Return a fresh, untrained model for "off_market" or "list_aware"."""
    path = Path(__file__).parent / "outputs" / "tree_baselines" / "frozen_config.json"
    if not path.exists():
        raise FileNotFoundError("Run python analysis/run_tree_baselines.py --phase development first")
    config = json.loads(path.read_text())
    if track not in config["tracks"]:
        raise ValueError(f"Unknown track: {track}")
    selected = config["tracks"][track]["model_config"]
    model = TreePriceModel(**selected["options"])
    if model.configuration() != selected:
        raise ValueError("Frozen configuration differs from the current feature / estimator schema")
    return model


class TreePriceModel:
    """Shared DT / Bagging / RF pipeline, trained and averaged on natural log(price)."""

    def __init__(self, track, family="rf", max_depth=8, min_samples_leaf=50,
                 n_estimators=300, random_state=42, n_jobs=-1):
        if track not in ("off_market", "list_aware"):
            raise ValueError(f"Unknown track: {track}")
        if family not in ("dt", "bagging", "rf"):
            raise ValueError(f"Unknown model family: {family}")
        self.track = track
        self.family = family
        self.options = dict(track=track, family=family, max_depth=max_depth,
                            min_samples_leaf=min_samples_leaf, n_estimators=n_estimators,
                            random_state=random_state, n_jobs=n_jobs)
        self.numeric_features = features.TREE_NUMERIC_FEATURES + (["list_price"] if track == "list_aware" else [])
        self.preprocessor = ColumnTransformer([
            ("numeric", "passthrough", self.numeric_features),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             features.TREE_CATEGORICAL_FEATURES),
        ], sparse_threshold=0.0, remainder="drop")
        common = dict(criterion="squared_error", max_depth=max_depth,
                      min_samples_leaf=min_samples_leaf, min_samples_split=2,
                      ccp_alpha=0.0, random_state=random_state)
        if family == "dt":
            self.estimator = DecisionTreeRegressor(**common, splitter="best", max_features=None)
        else:
            self.estimator = RandomForestRegressor(
                **common, n_estimators=n_estimators, max_features=1.0 if family == "bagging" else 0.7,
                bootstrap=True, max_samples=1.0, n_jobs=n_jobs)

    def configuration(self):
        encoder_params = self.preprocessor.transformers[1][1].get_params()
        encoder_params["dtype"] = np.dtype(encoder_params["dtype"]).name
        return {
            "options": self.options,
            "estimator_class": type(self.estimator).__name__,
            "estimator_params": self.estimator.get_params(),
            "raw_features": features.TREE_RAW_FEATURES + (["list_price"] if self.track == "list_aware" else []),
            "preprocessing": {
                "numeric_features": self.numeric_features,
                "categorical_features": features.TREE_CATEGORICAL_FEATURES,
                "one_hot_encoder_params": encoder_params,
                "sparse_threshold": 0.0, "remainder": "drop", "imputation": None, "scaling": None,
                "square_feet": "features.square_feet: exact or range midpoint",
                "months_since_2021": "(date_listed - 2021-01-01).days / 30.44",
            },
            "target": "natural_log_sold_price", "restore": "exp(mean_tree_log_predictions)",
        }

    def fit(self, train):
        target = train["sold_price"].to_numpy(dtype=float)
        valid = np.isfinite(target) & (target > 0)
        if not valid.all():
            raise ValueError(f"sold_price must be positive and finite; invalid row indices: {train.index[~valid].tolist()[:10]}")
        matrix = self.preprocessor.fit_transform(features.tree_features(train, self.track))
        self.feature_order_ = self.preprocessor.get_feature_names_out().tolist()
        self.estimator.fit(matrix, np.log(target))
        return self

    def predict_log(self, homes):
        matrix = self.preprocessor.transform(features.tree_features(homes, self.track))
        return self.estimator.predict(matrix)

    def predict(self, homes):
        prices = np.exp(self.predict_log(homes))
        if prices.shape != (len(homes),) or not (np.isfinite(prices) & (prices > 0)).all():
            raise ValueError("Predictions must be a positive, finite, one-dimensional price array")
        return prices


class ListPriceBaseline:
    """List-aware baseline: every home sells for exactly its list price."""

    def fit(self, train):
        return self   # nothing to learn

    def predict(self, homes):
        return homes["list_price"].to_numpy(dtype=float)


class NearestSalesBaseline:
    """Off-market baseline: the median sold price of the k nearest earlier sales of
    the same property type.

    "Nearest" means closest by latitude and longitude. "Earlier" means sold before the
    home was listed, so the model never looks into the future, even when the
    training table holds later sales. It makes no adjustment for size, condition or
    for prices moving since those sales: improving on that is your job.
    """

    def __init__(self, k=10):
        self.k = k

    def fit(self, train):
        columns = ["ml_num", "house_type_name", "date_sold", "latitude", "longitude", "sold_price"]
        # Newest sales first, so when several sales are equally near (a condo
        # building shares one location), the most recent ones are picked.
        self.sales = train[columns].sort_values(["date_sold", "ml_num"], ascending=[False, True])
        self.sales_by_type = dict(list(self.sales.groupby("house_type_name")))
        return self

    def predict(self, homes):
        return np.array([self.predict_one(home) for _, home in homes.iterrows()])

    def predict_one(self, home):
        sales = self.sales_by_type.get(home["house_type_name"], self.sales)
        earlier = sales[sales["date_sold"] < home["date_listed"]]
        if len(earlier) == 0:   # no earlier sale of this type: use every type
            earlier = self.sales[self.sales["date_sold"] < home["date_listed"]]

        # Distance in degrees. In Toronto a degree of longitude is only 0.72 times as
        # long as a degree of latitude (0.72 = cosine of 43.7 degrees north).
        dx = (earlier["longitude"].to_numpy() - home["longitude"]) * 0.72
        dy = earlier["latitude"].to_numpy() - home["latitude"]
        distance = np.sqrt(dx ** 2 + dy ** 2)

        nearest = np.argsort(distance, kind="stable")[: self.k]
        return float(np.median(earlier["sold_price"].to_numpy()[nearest]))


class FeatureModel:
    """A machine-learning model (gradient-boosted trees) on the features in features.py.

    It learns log(price) rather than price, so a 10% miss on a cheap condo counts
    as much as a 10% miss on an expensive house. Not used until you switch it on
    in make_model().
    """

    def __init__(self, make_features):
        self.make_features = make_features
        self.model = HistGradientBoostingRegressor(random_state=0)

    def fit(self, train):
        self.model.fit(self.make_features(train), np.log(train["sold_price"]))
        return self

    def predict(self, homes):
        return np.exp(self.model.predict(self.make_features(homes)))


class HistPriceModel:
    """HGB on the shared whitelist, with train-only encoding and log-price output.

    ``log_ratio`` learns a List-aware residual; public log predictions always
    include the log(list_price) offset. Estimator parameters are kept separately
    from learned category levels so a frozen configuration creates a fresh model.
    """

    def __init__(self, track, encoding="onehot", target_mode="log_price", **hgb_params):
        if track not in ("off_market", "list_aware"):
            raise ValueError(f"Unknown track: {track}")
        if encoding not in ("onehot", "native"):
            raise ValueError(f"Unknown HGB encoding: {encoding}")
        if target_mode not in ("log_price", "log_ratio"):
            raise ValueError(f"Unknown HGB target_mode: {target_mode}")
        if track == "off_market" and target_mode == "log_ratio":
            raise ValueError("Off-market cannot use log_ratio or list_price")
        params = dict(
            loss="squared_error", learning_rate=0.1, max_iter=200,
            max_leaf_nodes=31, max_depth=None, min_samples_leaf=20,
            l2_regularization=0.0, max_features=1.0, max_bins=255,
            categorical_features="from_dtype", early_stopping=False,
            warm_start=False, random_state=42,
        )
        params.update(hgb_params)
        if params["early_stopping"] is not False or params["warm_start"] is not False:
            raise ValueError("HGB temporal experiments require early_stopping=False and warm_start=False")
        if params["categorical_features"] != "from_dtype":
            raise ValueError("HGB encoding requires categorical_features='from_dtype'")
        self.track = track
        self.encoding = encoding
        self.target_mode = target_mode
        self.estimator = HistGradientBoostingRegressor(**params)
        self.numeric_features = features.TREE_NUMERIC_FEATURES + (["list_price"] if track == "list_aware" else [])
        self.categorical_features = list(features.TREE_CATEGORICAL_FEATURES)
        self.preprocessor = None
        if encoding == "onehot":
            self.preprocessor = ColumnTransformer([
                ("numeric", "passthrough", self.numeric_features),
                ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                 self.categorical_features),
            ], sparse_threshold=0.0, remainder="drop")

    @staticmethod
    def _json_safe(value):
        def numeric_value(item):
            if isinstance(item, np.generic):
                return item.item()
            if isinstance(item, np.ndarray):
                return item.tolist()
            raise TypeError(f"HGB configuration must be JSON-safe, got {type(item).__name__}")
        return json.loads(json.dumps(value, default=numeric_value, allow_nan=False))

    @property
    def options(self):
        """Flat constructor kwargs; no learned fold state is included."""
        return {
            "track": self.track, "encoding": self.encoding, "target_mode": self.target_mode,
            **self._json_safe(self.estimator.get_params()),
        }

    def configuration(self):
        preprocessing = {
            "encoding": self.encoding, "imputation": None, "scaling": None,
            "category_levels_source": "observed training values only",
        }
        if self.encoding == "onehot":
            encoder_params = self.preprocessor.transformers[1][1].get_params()
            encoder_params["dtype"] = np.dtype(encoder_params["dtype"]).name
            preprocessing.update(
                output="dense_numeric_matrix", one_hot_encoder_params=encoder_params,
                sparse_threshold=0.0, remainder="drop", unknown_category="all-zero encoded category block",
            )
        else:
            preprocessing.update(
                output="pandas_dataframe_with_categorical_dtype", unknown_category="missing",
                categorical_features="from_dtype", max_bins=self.estimator.max_bins,
            )
        return self._json_safe({
            "options": self.options,
            "estimator_class": type(self.estimator).__name__,
            "estimator_params": self.estimator.get_params(),
            "schema": {
                "feature_builder": "features.tree_features",
                "raw_features": features.TREE_RAW_FEATURES + (["list_price"] if self.track == "list_aware" else []),
                "numeric_features": self.numeric_features,
                "categorical_features": self.categorical_features,
                "input_feature_order": self.numeric_features + self.categorical_features,
            },
            "preprocessing": preprocessing,
            "target": "natural_log_sold_price" if self.target_mode == "log_price" else "natural_log_sold_to_list_ratio",
            "prediction_log_offset": "log(list_price)" if self.target_mode == "log_ratio" else None,
            "restore": "exp(final_log_price)",
        })

    def prepare_features(self, homes, fit=False):
        """Fit train-only preprocessing, or transform using its unchanged schema."""
        if not fit and not hasattr(self, "feature_order_"):
            raise NotFittedError("Fit HGB preprocessing before transforming homes")
        frame = features.tree_features(homes, self.track)
        if self.encoding == "onehot":
            if fit:
                matrix = self.preprocessor.fit_transform(frame)
                self.feature_order_ = self.preprocessor.get_feature_names_out().tolist()
                encoder = self.preprocessor.named_transformers_["categorical"]
                self.category_levels_ = {
                    column: pd.Index(levels).dropna().tolist()
                    for column, levels in zip(self.categorical_features, encoder.categories_)
                }
                return matrix
            return self.preprocessor.transform(frame)
        if fit:
            levels = {column: pd.unique(frame[column].dropna()).tolist()
                      for column in self.categorical_features}
            for column, values in levels.items():
                if len(values) > self.estimator.max_bins:
                    raise ValueError(f"Native categorical {column} has {len(values)} training levels; "
                                     f"max_bins={self.estimator.max_bins}")
            self.category_levels_ = levels
            self.feature_order_ = frame.columns.tolist()
        for column in self.categorical_features:
            frame[column] = pd.Categorical(frame[column], categories=self.category_levels_[column])
        return frame.loc[:, self.feature_order_]

    @staticmethod
    def _positive_prices(homes, column):
        values = homes[column].to_numpy(dtype=float)
        valid = np.isfinite(values) & (values > 0)
        if not valid.all():
            raise ValueError(f"{column} must be positive and finite; invalid row indices: "
                             f"{homes.index[~valid].tolist()[:10]}")
        return values

    def fit(self, train):
        if train.empty:
            raise ValueError("HGB training table must not be empty")
        target = np.log(self._positive_prices(train, "sold_price"))
        if self.target_mode == "log_ratio":
            target = target - np.log(self._positive_prices(train, "list_price"))
        started = perf_counter()
        matrix = self.prepare_features(train, fit=True)
        self.fit_preprocessing_seconds_ = perf_counter() - started
        started = perf_counter()
        self.estimator.fit(matrix, target)
        self.fit_estimator_seconds_ = perf_counter() - started
        return self

    def _log_offset(self, homes):
        return np.log(self._positive_prices(homes, "list_price")) if self.target_mode == "log_ratio" else 0.0

    @staticmethod
    def _final_log_price(predicted_target, n_homes, offset):
        values = np.asarray(predicted_target, dtype=float) + offset
        if values.shape != (n_homes,) or not np.isfinite(values).all():
            raise ValueError("Log-price predictions must be a finite one-dimensional array")
        return values

    def predict_log(self, homes):
        matrix = self.prepare_features(homes)
        offset = self._log_offset(homes)
        return self._final_log_price(self.estimator.predict(matrix), len(homes), offset)

    def staged_predict_log(self, homes):
        """Yield final log(price) after every boosting iteration, from one transform."""
        matrix = self.prepare_features(homes)
        offset = self._log_offset(homes)
        for predicted_target in self.estimator.staged_predict(matrix):
            yield self._final_log_price(predicted_target, len(homes), offset)

    def predict(self, homes):
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            prices = np.exp(self.predict_log(homes))
        if prices.shape != (len(homes),) or not (np.isfinite(prices) & (prices > 0)).all():
            raise ValueError("Predictions must be a positive, finite, one-dimensional price array")
        return prices


def make_hgb_model(track, config_path):
    """Read an independent frozen HGB configuration without changing the RF factory."""
    config = json.loads(Path(config_path).read_text())
    if track not in config["tracks"]:
        raise ValueError(f"Unknown track in HGB configuration: {track}")
    selected = config["tracks"][track]["model_config"]
    if selected["options"]["track"] != track:
        raise ValueError("Frozen HGB configuration track differs from the requested track")
    model = HistPriceModel(**selected["options"])
    if model.configuration() != selected:
        raise ValueError("Frozen HGB configuration differs from the current feature / estimator schema")
    return model
