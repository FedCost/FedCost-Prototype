"""Engine tests: aggregation rules and contract-conformant output."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import contract
import engine
from contract import FEAT_DIM, N_CLASSES, N_CLIENTS


def rand_updates(n=6, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.normal(0, 1, (FEAT_DIM + 1, N_CLASSES)) for _ in range(n)]


def test_fedavg_size_weighted_average():
    a, b = np.zeros((3, 2)), np.ones((3, 2))
    W, info = engine.agg_fedavg([a, b], [1, 3])
    assert np.allclose(W, 0.75)
    assert info == {0: (False, None), 1: (False, None)}


def test_trimmed_does_not_drop_client_entirely_and_trims_extremes():
    ups = [np.full((4, 2), v) for v in (0.0, 1.0, 2.0, 3.0, 4.0, 100.0)]
    W, info = engine.agg_trimmed(ups, [1] * 6, beta=0.2)   # m=1: smallest and largest are dropped
    assert np.allclose(W, np.mean([1.0, 2.0, 3.0, 4.0]))
    # The two extreme clients are trimmed at every coordinate; the middle ones are never trimmed
    assert info[5][0] and info[0][0]
    assert not any(info[k][0] for k in (1, 2, 3, 4))
    assert info[5][1].startswith("trimmed coordinate fraction 1.00")


def test_trimmed_info_contract_format():
    _, info = engine.agg_trimmed(rand_updates(), [1] * 6)
    assert set(info) == set(range(6))
    for ex, reason in info.values():
        assert isinstance(ex, bool) and isinstance(reason, str)


def test_krum_selects_single_client_others_excluded():
    ups = rand_updates()
    ups[3] = np.zeros_like(ups[3]) + 0.01 * ups[3]          # not far from the others, central
    W, info = engine.agg_krum(ups, [1] * 6, f=1)
    selected = [k for k, (ex, _) in info.items() if not ex]
    assert len(selected) == 1
    assert np.array_equal(W, ups[selected[0]])
    assert all(info[k][1] for k in info if k != selected[0])


def test_krum_does_not_select_outlier_client():
    rng = np.random.default_rng(1)
    base = rng.normal(0, 1, (5, 2))
    ups = [base + rng.normal(0, 0.01, base.shape) for _ in range(5)] + [base + 50.0]
    _, info = engine.agg_krum(ups, [1] * 6, f=1)
    assert info[5][0] is True
    assert sum(not ex for ex, _ in info.values()) == 1


def test_local_train_does_not_mutate_input_and_is_seed_reproducible():
    rng0 = np.random.default_rng(0)
    X = rng0.normal(size=(50, FEAT_DIM)).astype(np.float32)
    y = rng0.integers(0, N_CLASSES, 50)
    W0 = np.zeros((FEAT_DIM + 1, N_CLASSES))
    W1 = engine.local_train(W0, X, y, rng=np.random.default_rng(5))
    W2 = engine.local_train(W0, X, y, rng=np.random.default_rng(5))
    assert np.all(W0 == 0) and np.array_equal(W1, W2) and not np.all(W1 == 0)


def test_class_recall_missing_class_is_nan():
    W = np.zeros((FEAT_DIM + 1, N_CLASSES))
    X = np.zeros((4, FEAT_DIM)); y = np.array([0, 0, 1, 1])
    r = engine.class_recall(W, X, y)
    assert len(r) == N_CLASSES and np.isnan(r[5])


@pytest.fixture
def mini_features(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(contract, "FEATURES_DIR", tmp_path / "features")
    monkeypatch.setattr(contract, "RUNS_DIR", tmp_path / "runs")
    rng = np.random.default_rng(0)
    protos = rng.normal(0, 1, (N_CLASSES, FEAT_DIM))
    for k in range(N_CLIENTS):
        for split, n in (("train", 80), ("test", 40)):
            y = rng.integers(0, N_CLASSES, n)
            contract.save_features(k, split, protos[y] + rng.normal(0, 1, (n, FEAT_DIM)), y)
    return tmp_path


@pytest.mark.parametrize("agg", ["fedavg", "trimmed", "krum"])
def test_run_writes_contract_conformant_output(mini_features, agg):
    engine.run(agg, rounds=2, attack=0, seed=3, run_id="t")
    rd = mini_features / "runs" / "t"
    rec = json.loads((rd / "round_1.json").read_text())
    assert set(rec["clients"]) == {str(k) for k in range(N_CLIENTS)}
    for c in rec["clients"].values():
        assert len(c["class_recall"]) == N_CLASSES
        assert {"update_norm", "excluded", "reason"} <= set(c)
    assert len(rec["global_class_recall"]) == N_CLASSES
    assert rec["meta"] == {"agg": agg, "attack": 0, "seed": 3}
    W = contract.load_update("t", 1, 0)
    assert W.shape == (FEAT_DIM + 1, N_CLASSES) and W.dtype == np.float32


def test_run_is_reproducible_with_seed(mini_features):
    W1 = engine.run("trimmed", rounds=2, seed=7, run_id="a")
    W2 = engine.run("trimmed", rounds=2, seed=7, run_id="b")
    assert np.array_equal(W1, W2)
