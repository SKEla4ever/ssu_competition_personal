"""HGB prediction, preprocessing, target restoration and frozen-config contracts."""
import copy
import json

import joblib
import numpy as np
import pandas as pd
import pytest

import data
import features
import models
from test_starter import make_listings


CASES = [("off_market", "log_price"), ("list_aware", "log_price"), ("list_aware", "log_ratio")]


def listings():
    train = make_listings(240)
    train.loc[train.index[::2], ["lot_front", "lot_depth"]] = [30.0, 100.0]
    train.loc[train.index[::3], "community"] = "Second community"
    return train


def small_model(track, encoding, target_mode="log_price", **params):
    options = dict(max_iter=8, max_leaf_nodes=7, min_samples_leaf=5)
    options.update(params)
    return models.HistPriceModel(track, encoding, target_mode, **options)


@pytest.mark.parametrize("encoding", ["onehot", "native"])
@pytest.mark.parametrize("track,target_mode", CASES)
def test_hgb_prediction_contract_unknown_nan_order_seed_and_reload(encoding, track, target_mode, tmp_path):
    train = listings()
    homes = data.hide_answers(make_listings(12, seed=7), track)
    for column in features.TREE_CATEGORICAL_FEATURES:
        homes[column] = "Previously unseen value"
    model = small_model(track, encoding, target_mode)
    before_fit = model.configuration()
    assert model.fit(train) is model
    levels_before = copy.deepcopy(model.category_levels_)
    predicted = model.predict(homes)
    assert predicted.shape == (len(homes),)
    assert np.isfinite(predicted).all() and (predicted > 0).all()
    np.testing.assert_allclose(np.exp(model.predict_log(homes)), predicted, rtol=1e-12)
    np.testing.assert_allclose(model.predict(homes.iloc[::-1]), predicted[::-1], rtol=1e-12)
    assert model.category_levels_ == levels_before
    assert model.configuration() == before_fit
    assert json.loads(json.dumps(before_fit, allow_nan=False)) == before_fit
    assert model.fit_preprocessing_seconds_ >= 0 and model.fit_estimator_seconds_ >= 0
    matrix = model.prepare_features(homes)
    assert matrix.shape[1] == len(model.feature_order_)
    if encoding == "native":
        assert isinstance(matrix, pd.DataFrame)
        assert matrix.columns.tolist() == model.feature_order_
        assert matrix[["lot_front", "lot_depth", "lot_area"]].isna().all().all()
        for column in features.TREE_CATEGORICAL_FEATURES:
            assert isinstance(matrix[column].dtype, pd.CategoricalDtype)
            assert matrix[column].isna().all()
    else:
        assert isinstance(matrix, np.ndarray) and np.isnan(matrix).any()
    repeated = models.HistPriceModel(**before_fit["options"]).fit(train)
    np.testing.assert_allclose(repeated.predict(homes), predicted, rtol=1e-12)
    path = tmp_path / "hgb.joblib"
    joblib.dump(model, path)
    restored = joblib.load(path)
    np.testing.assert_allclose(restored.predict(homes), predicted, rtol=1e-12)
    assert restored.category_levels_ == levels_before


@pytest.mark.parametrize("encoding", ["onehot", "native"])
def test_off_market_ignores_list_price_and_non_whitelisted_fields(encoding):
    train = listings()
    homes = data.hide_answers(make_listings(10, seed=8), "off_market")
    reference = small_model("off_market", encoding).fit(train).predict(homes)
    changed_train = train.assign(list_price=-1e30, remarks="ignored", external_avm_estimate=1e30,
                                 ml_num="ignored", date_sold=pd.Timestamp("2099-01-01"))
    changed_homes = homes.assign(list_price=np.nan, sold_price=1e30, remarks="ignored",
                                 days_on_market=1e9, external_avm_estimate=1e30)
    actual = small_model("off_market", encoding).fit(changed_train).predict(changed_homes)
    np.testing.assert_allclose(actual, reference, rtol=1e-12)
    with pytest.raises(ValueError, match="Off-market"):
        small_model("off_market", encoding, "log_ratio")


@pytest.mark.parametrize("encoding", ["onehot", "native"])
@pytest.mark.parametrize("track,target_mode", CASES)
def test_staged_predictions_match_standalone_checkpoint_and_final(encoding, track, target_mode):
    train = listings()
    homes = data.hide_answers(make_listings(10, seed=2), track)
    full = small_model(track, encoding, target_mode, max_iter=12).fit(train)
    short = small_model(track, encoding, target_mode, max_iter=5).fit(train)
    stages = list(full.staged_predict_log(homes))
    assert len(stages) == full.estimator.n_iter_ == 12
    np.testing.assert_allclose(stages[4], short.predict_log(homes), rtol=1e-12)
    np.testing.assert_allclose(stages[-1], full.predict_log(homes), rtol=1e-12)


@pytest.mark.parametrize("encoding", ["onehot", "native"])
def test_log_ratio_zero_residual_restores_exact_list_price_at_every_stage(encoding):
    train = listings()
    train["sold_price"] = train["list_price"]
    homes = data.hide_answers(make_listings(10, seed=9), "list_aware")
    model = small_model("list_aware", encoding, "log_ratio", max_iter=4).fit(train)
    np.testing.assert_allclose(model.predict(homes), homes["list_price"], rtol=1e-12)
    for prediction in model.staged_predict_log(homes):
        np.testing.assert_allclose(prediction, np.log(homes["list_price"]), rtol=1e-12)


@pytest.mark.parametrize("bad_target", [0, -10, np.nan, np.inf])
def test_invalid_sold_target_rejected(bad_target):
    train = listings()
    train.loc[0, "sold_price"] = bad_target
    with pytest.raises(ValueError, match="sold_price must be positive and finite"):
        small_model("off_market", "onehot").fit(train)


@pytest.mark.parametrize("bad_price", [0, -10, np.nan, np.inf])
def test_ratio_requires_positive_finite_list_prices_for_fit_and_predict(bad_price):
    train = listings()
    invalid_train = train.copy()
    invalid_train.loc[0, "list_price"] = bad_price
    with pytest.raises(ValueError, match="list_price must be positive and finite"):
        small_model("list_aware", "native", "log_ratio").fit(invalid_train)
    model = small_model("list_aware", "native", "log_ratio").fit(train)
    homes = data.hide_answers(make_listings(5, seed=1), "list_aware")
    homes.loc[0, "list_price"] = bad_price
    with pytest.raises(ValueError, match="list_price must be positive and finite"):
        model.predict(homes)


def test_native_category_cardinality_checked_on_training_only():
    train = listings()
    train["community"] = ["Alpha", "Beta", "Gamma"] * 80
    with pytest.raises(ValueError, match="community.*max_bins=2"):
        small_model("off_market", "native", max_bins=2).fit(train)


def test_onehot_preprocessing_matches_unchanged_tree_pipeline():
    train = listings()
    legacy = models.TreePriceModel("off_market", "dt")
    expected = legacy.preprocessor.fit_transform(features.tree_features(train, "off_market"))
    model = small_model("off_market", "onehot")
    actual = model.prepare_features(train, fit=True)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=0, equal_nan=True)
    assert model.feature_order_ == legacy.preprocessor.get_feature_names_out().tolist()


def test_independent_hgb_factory_reconstructs_and_checks_frozen_config(tmp_path):
    original = small_model("list_aware", "native", "log_ratio")
    config = {"tracks": {"list_aware": {"model_config": original.configuration()}}}
    path = tmp_path / "frozen_hgb.json"
    path.write_text(json.dumps(config))
    restored = models.make_hgb_model("list_aware", path)
    assert restored.configuration() == original.configuration()
    assert not hasattr(restored, "category_levels_")
    config["tracks"]["list_aware"]["model_config"]["schema"]["numeric_features"].append("forbidden")
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="current feature / estimator schema"):
        models.make_hgb_model("list_aware", path)


@pytest.mark.parametrize("params", [
    {"track": "invalid"}, {"track": "off_market", "encoding": "invalid"},
    {"track": "off_market", "target_mode": "invalid"},
    {"track": "off_market", "early_stopping": "auto"},
    {"track": "off_market", "early_stopping": True},
    {"track": "off_market", "warm_start": True},
    {"track": "off_market", "categorical_features": None},
])
def test_invalid_or_temporally_unsafe_configuration_rejected(params):
    with pytest.raises(ValueError):
        models.HistPriceModel(**params)
