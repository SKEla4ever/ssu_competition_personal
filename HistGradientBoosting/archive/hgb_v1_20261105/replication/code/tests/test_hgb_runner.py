"""Runner integration checks for checkpoint accounting, cache/resume and temporal folds."""
import json

import numpy as np
import pandas as pd
import pytest

import models
from analysis import run_hgb_baselines as runner
from test_starter import make_listings


@pytest.fixture
def small_runner(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "DEVELOPMENT_MONTHS", pd.period_range("2025-10", "2025-11", freq="M"))
    listings = make_listings(700)
    return runner.HGBRunner(tmp_path / "hgb_v1", listings=listings, guard=False)


def test_checkpoint_cache_and_same_iteration_diagnostics(small_runner, monkeypatch):
    option = dict(track="off_market", encoding="native", max_iter=4, max_leaf_nodes=7, min_samples_leaf=5)
    results = small_runner.evaluate(option, checkpoints=[2, 4], stage="integration")
    assert small_runner.fit_count() == 2
    assert [r["iterations_evaluated"] for r in results] == [2, 4]
    for result in results:
        directory = small_runner.root / "experiments" / result["experiment_id"]
        diagnostics = json.loads((directory / "model_diagnostics.json").read_text())
        assert all(d["iterations_trained"] == 4 for d in diagnostics)
        assert all(d["iterations_evaluated"] == result["iterations_evaluated"] for d in diagnostics)
        assert all(pd.Timestamp(d["max_training_date_sold"]) < pd.Timestamp(d["training_cutoff"]) for d in diagnostics)
        assert result["model_config"]["options"]["max_iter"] == result["iterations_evaluated"]
    runner.assert_same_rows(small_runner.frame(results[0]), small_runner.frame(results[1]))
    monkeypatch.setattr(models.HistPriceModel, "fit", lambda *a: pytest.fail("A completed cache must not refit"))
    repeated = small_runner.evaluate(option, checkpoints=[2, 4])
    assert repeated == results


def test_failed_fold_resume_reuses_completed_folds(small_runner, monkeypatch):
    original = models.HistPriceModel.fit
    calls = []
    def fail_second(model, train):
        calls.append(len(train))
        if len(calls) == 2:
            raise RuntimeError("simulated interruption")
        return original(model, train)
    option = dict(track="off_market", max_iter=3, max_leaf_nodes=7, min_samples_leaf=5)
    monkeypatch.setattr(models.HistPriceModel, "fit", fail_second)
    with pytest.raises(RuntimeError, match="simulated"):
        small_runner.evaluate(option)
    assert small_runner.fit_count() == 1
    monkeypatch.setattr(models.HistPriceModel, "fit", original)
    resumed = runner.HGBRunner(small_runner.root, listings=small_runner.listings, guard=False)
    result = resumed.evaluate(option)[0]
    assert resumed.fit_count() == 2
    assert len(list(resumed.root.glob("physical_runs/*/*/failure.json"))) == 1
    assert result["metrics"]["n"] > 0


def test_horizon_is_one_fit_and_retains_fixed_cutoff(small_runner):
    (small_runner.root / "frozen_config.json").write_text('{"test_fixture": true}\n')
    result = small_runner.evaluate(dict(track="list_aware", encoding="native", target_mode="log_ratio",
        max_iter=4, max_leaf_nodes=7, min_samples_leaf=5), phase="horizon")[0]
    assert small_runner.fit_count("horizon") == 1
    frame = small_runner.frame(result)
    assert frame["training_cutoff"].unique().tolist() == ["2026-02-01T00:00:00"]
    assert sorted(frame["horizon_months"].unique()) == [1, 2, 3, 4]
    assert (frame["predicted_price_raw"] > 0).all()


def test_staged_predictions_match_saved_standalone(small_runner):
    base = dict(track="list_aware", encoding="onehot", target_mode="log_ratio", max_iter=200,
                max_leaf_nodes=3, min_samples_leaf=50)
    reference = small_runner.evaluate(base)[0]
    # The 200 checkpoint must be checked against a previously independently fitted model.
    staged = small_runner.evaluate({**base, "max_iter": 201}, checkpoints=[200, 201], reference_200=reference)
    np.testing.assert_allclose(small_runner.frame(reference)["predicted_price_raw"],
                               small_runner.frame(staged[0])["predicted_price_raw"], rtol=1e-10)
    assert small_runner.fit_count() == 4


def test_selection_ties_prefer_simple_configs_and_price_target():
    def result(encoding="native", target="log_price", leaves=31, iteration=200, samples=20):
        return {"metrics": {"mdape": 5., "mape": 6.}, "model_config": {"options": {
            "encoding": encoding, "target_mode": target, "max_leaf_nodes": leaves,
            "max_iter": iteration, "min_samples_leaf": samples}}}
    assert runner.selection_key(result(leaves=15)) < runner.selection_key(result(leaves=31))
    assert runner.selection_key(result(samples=50)) < runner.selection_key(result(samples=20))
    assert runner.selection_key(result(encoding="onehot")) < runner.selection_key(result(encoding="native"))
    assert runner.selection_key(result(target="log_price")) < runner.selection_key(result(target="log_ratio"))


def test_identity_rejects_changed_truth_or_cutoff():
    frame = pd.DataFrame({"ml_num": ["A"], "month": ["2025-11"],
                          "training_cutoff": ["2025-11-01"], "sold_price": [100.]})
    runner.assert_same_rows(frame, frame.copy())
    with pytest.raises(AssertionError):
        runner.assert_same_rows(frame, frame.assign(sold_price=200.))
    with pytest.raises(AssertionError):
        runner.assert_same_rows(frame, frame.assign(training_cutoff="2025-12-01"))


def fixture_frozen(small_runner):
    results = [small_runner.evaluate(dict(track=track, encoding="native", max_iter=3,
        max_leaf_nodes=7, min_samples_leaf=5))[0] for track in runner.TRACKS]
    frozen = {"version": small_runner.version, "execution_contract_sha256": small_runner.contract_hash,
        "tracks": {result["track"]: {"model_config": result["model_config"],
            "development_metrics": result["metrics"], "selected_experiment_id": result["experiment_id"]}
                   for result in results}}
    runner.write_json(small_runner.root / "frozen_config.json", frozen)
    return results


def test_freeze_summary_interruption_recovers_without_refit(small_runner, monkeypatch):
    fixture_frozen(small_runner)
    before = runner.sha256(small_runner.root / "frozen_config.json")
    count = small_runner.fit_count()
    monkeypatch.setattr(models.HistPriceModel, "fit", lambda *a: pytest.fail("Freeze recovery must not refit"))
    summary = small_runner.development()
    assert small_runner.fit_count() == count
    assert summary["frozen_config_sha256"] == before
    assert runner.sha256(small_runner.root / "frozen_config.json") == before
    small_runner.read_frozen()


def test_frozen_mutation_is_rejected(small_runner):
    results = fixture_frozen(small_runner)
    small_runner.save_summary("development", results)
    small_runner.read_frozen()
    path = small_runner.root / "frozen_config.json"
    changed = runner.read_json(path)
    changed["tracks"]["off_market"]["model_config"]["options"]["max_iter"] = 2
    runner.atomic_json(path, changed)
    with pytest.raises(ValueError, match="differs"):
        small_runner.read_frozen()
