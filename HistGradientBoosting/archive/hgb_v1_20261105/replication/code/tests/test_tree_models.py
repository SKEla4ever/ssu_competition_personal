"""Prediction contracts, leakage boundaries, log averaging and temporal evaluation."""
import json

import numpy as np
import pandas as pd
import pytest

import data
import features
import models
from analysis.run_tree_baselines import point_metrics, temporal_fold
from test_starter import make_listings


@pytest.mark.parametrize("family", ["dt", "bagging", "rf"])
@pytest.mark.parametrize("track", ["off_market", "list_aware"])
def test_prediction_contract_nan_unknown_order_and_reproducibility(family, track):
    train = make_listings(300)
    train.loc[train.index[::2], ["lot_front", "lot_depth"]] = [30.0, 100.0]
    homes = data.hide_answers(make_listings(10, seed=3), track)
    homes["community"] = "Previously unseen community"
    model = models.TreePriceModel(track, family, max_depth=4, min_samples_leaf=3, n_estimators=9, n_jobs=1)
    model.fit(train)
    predictions = model.predict(homes)
    assert predictions.shape == (len(homes),)
    assert np.isfinite(predictions).all() and (predictions > 0).all()
    np.testing.assert_allclose(model.predict(homes.iloc[::-1]), predictions[::-1], rtol=1e-12)
    repeated = models.TreePriceModel(**model.options).fit(train).predict(homes)
    np.testing.assert_allclose(repeated, predictions, rtol=1e-12)
    matrix = model.preprocessor.transform(features.tree_features(homes, track))
    assert isinstance(matrix, np.ndarray) and np.isnan(matrix).any()
    assert model.feature_order_ == model.preprocessor.get_feature_names_out().tolist()
    if family != "dt":
        per_tree = np.array([tree.predict(matrix) for tree in model.estimator.estimators_])
        np.testing.assert_allclose(predictions, np.exp(per_tree.mean(axis=0)), rtol=1e-12)
    json.dumps(model.configuration(), allow_nan=False)


@pytest.mark.parametrize("family", ["dt", "bagging", "rf"])
def test_off_market_ignores_list_price_and_all_non_whitelisted_fields(family):
    train = make_listings(250)
    homes = data.hide_answers(make_listings(10, seed=8), "off_market")
    options = dict(track="off_market", family=family, max_depth=4, min_samples_leaf=3,
                   n_estimators=9, n_jobs=1)
    reference = models.TreePriceModel(**options).fit(train).predict(homes)
    changed = train.assign(list_price=1e30, remarks="changed", external_avm_estimate=1e30,
                           date_sold=pd.Timestamp("2099-01-01"), ml_num="ignored")
    changed_homes = homes.assign(list_price=-1e30, sold_price=1e30, days_on_market=1e9,
                                remarks="ignored", external_avm_estimate=1e30)
    actual = models.TreePriceModel(**options).fit(changed).predict(changed_homes)
    np.testing.assert_allclose(actual, reference, rtol=1e-12)
    assert "list_price" not in features.tree_features(changed_homes, "off_market")


@pytest.mark.parametrize("bad_target", [0, -10, np.nan, np.inf])
def test_invalid_target_rejected(bad_target):
    train = make_listings(20)
    train.loc[0, "sold_price"] = bad_target
    with pytest.raises(ValueError, match="positive and finite"):
        models.TreePriceModel("off_market").fit(train)


def test_leaf_price_is_geometric_mean():
    train = make_listings(2).assign(sold_price=[100.0, 400.0])
    model = models.TreePriceModel("off_market", "dt", min_samples_leaf=20).fit(train)
    np.testing.assert_allclose(model.predict(data.hide_answers(train, "off_market")), [200, 200])


def test_temporal_cutoff_is_strict_and_horizon_does_not_add_labels():
    listings = make_listings(5).assign(date_sold=pd.to_datetime([
        "2026-01-31", "2026-02-01", "2026-02-28", "2026-03-01", "2026-05-01"]))
    train, valid, cutoff = temporal_fold(listings, pd.Period("2026-02", "M"))
    assert train["ml_num"].tolist() == listings.iloc[[0]]["ml_num"].tolist()
    assert (train["date_sold"] < cutoff).all() and len(valid) == 2
    later_train, _, _ = temporal_fold(listings, pd.Period("2026-05", "M"), "2026-02-01")
    pd.testing.assert_frame_equal(later_train, train)


def test_pooled_mdape_uses_individual_rows():
    # Monthly medians would average to 50%; the pooled median is 0%.
    scores = point_metrics([100] * 5, [100, 100, 100, 100, 200])
    assert scores == {"n": 5, "mdape": 0.0, "mape": 20.0, "median_signed_error": 0.0}
