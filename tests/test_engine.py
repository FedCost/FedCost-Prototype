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
    assert rec["meta"] == {"agg": agg, "attack": 0, "seed": 3, "aggregation": agg, "attack_client": 0,
                           "client_sizes": [80] * N_CLIENTS}
    W = contract.load_update("t", 1, 0)
    assert W.shape == (FEAT_DIM + 1, N_CLASSES) and W.dtype == np.float32


def test_run_is_reproducible_with_seed(mini_features):
    W1 = engine.run("trimmed", rounds=2, seed=7, run_id="a")
    W2 = engine.run("trimmed", rounds=2, seed=7, run_id="b")
    assert np.array_equal(W1, W2)


def test_run_leaves_excluded_clients_out(mini_features):
    engine.run("fedavg", rounds=1, attack=2, seed=3, run_id="loo", excluded_clients=[2, 4])
    rd = mini_features / "runs" / "loo"
    rec = json.loads((rd / "round_0.json").read_text())
    assert rec["meta"]["excluded_clients"] == [2, 4]
    assert sorted(p.name for p in (rd / "updates").iterdir()) == [f"r0_c{k}.npy" for k in (0, 1, 3, 5)]
    assert [rec["clients"][str(k)]["excluded"] for k in range(N_CLIENTS)] == [k in (2, 4) for k in range(N_CLIENTS)]
    with pytest.raises(ValueError):
        engine.run("fedavg", rounds=1, run_id="empty", excluded_clients=range(N_CLIENTS))


def test_synthetic_never_writes_real_features(tmp_path, monkeypatch):
    import synthetic
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(synthetic, "SIZES", [10] * N_CLIENTS)
    synthetic.make(out_dir="syn")
    assert len(list((tmp_path / "syn").glob("*.npz"))) == 2 * N_CLIENTS
    assert not (tmp_path / "features").exists() and contract.FEATURES_DIR == Path("features")
    with pytest.raises(ValueError):
        synthetic.make(out_dir="features")


def test_dp_vectorized_clipping_matches_naive_per_sample_loop():
    rng = np.random.default_rng(42)
    X = rng.normal(0, 2.0, (16, 20)).astype(np.float32)
    y = rng.integers(0, N_CLASSES, 16)
    W0 = rng.normal(0, 0.2, (21, N_CLASSES))
    clip, lr, l2 = 3.0, 0.05, 1e-4

    # Naive per-sample loop (single batch, zero noise)
    W_ref = W0.copy()
    Y = np.eye(N_CLASSES)[y]
    z = X @ W_ref[:-1] + W_ref[-1]
    z -= z.max(1, keepdims=True)
    P = np.exp(z); P /= P.sum(1, keepdims=True)
    E = P - Y
    gW_sum = np.zeros_like(W_ref[:-1])
    gb_sum = np.zeros_like(W_ref[-1])
    clipped_cnt = 0
    for i in range(len(X)):
        gW_i = np.outer(X[i], E[i])
        gb_i = E[i]
        norm_i = np.sqrt(np.sum(gW_i ** 2) + np.sum(gb_i ** 2))
        if norm_i > clip:
            clipped_cnt += 1
            s = clip / norm_i
            gW_i *= s
            gb_i = gb_i * s
        gW_sum += gW_i
        gb_sum += gb_i
    W_ref[:-1] -= lr * (gW_sum / len(X) + l2 * W_ref[:-1])
    W_ref[-1] -= lr * (gb_sum / len(X))

    # Vectorized engine.local_train (bs=16, epochs=1, dp_sigma=0)
    W_vec, stats = engine.local_train(
        W0, X, y, epochs=1, lr=lr, bs=16, l2=l2,
        rng=np.random.default_rng(0), dp_clip=clip, dp_sigma=0.0, return_stats=True,
    )
    assert np.allclose(W_vec, W_ref, atol=1e-12)
    assert np.isclose(stats["clipped_frac"], clipped_cnt / len(X))


def test_dp_accountant_monotonic_and_accurate():
    s_eps8 = engine.solve_noise_multiplier(3000, rounds=15, epochs=2, bs=128, target_eps=8.0)
    s_eps3 = engine.solve_noise_multiplier(3000, rounds=15, epochs=2, bs=128, target_eps=3.0)
    assert s_eps3 > s_eps8 > 0.0
    q = 128 / 3000
    steps = 15 * 2 * int(np.ceil(3000 / 128))
    assert np.isclose(engine.eps_from_sigma(q, s_eps8, steps), 8.0, atol=1e-2)
    assert np.isclose(engine.eps_from_sigma(q, s_eps3, steps), 3.0, atol=1e-2)
    assert engine.solve_noise_multiplier(3000, 15, 2, 128, None) == 0.0
    assert engine.solve_noise_multiplier(3000, 15, 2, 128, float("inf")) == 0.0


def test_dp_run_reproducible_and_logs_telemetry(mini_features):
    W1 = engine.run("fedavg", rounds=2, seed=11, run_id="dp_a", dp_eps=8.0, dp_clip=5.0)
    W2 = engine.run("fedavg", rounds=2, seed=11, run_id="dp_b", dp_eps=8.0, dp_clip=5.0)
    W_clean = engine.run("fedavg", rounds=2, seed=11, run_id="clean", dp_eps=None)
    assert np.array_equal(W1, W2)
    assert not np.array_equal(W1, W_clean)
    rec = json.loads((mini_features / "runs" / "dp_a" / "round_1.json").read_text())
    assert rec["meta"]["dp_eps"] == 8.0
    assert rec["meta"]["dp_clip"] == 5.0
    assert rec["meta"]["dp_delta"] == engine.DEFAULT_DP_DELTA
    for c in rec["clients"].values():
        assert 0.0 <= c["clipped_frac"] <= 1.0
        assert len(c["class_clipped_frac"]) == N_CLASSES


def test_dp_inf_matches_none_exact(mini_features):
    W_none = engine.run("trimmed", rounds=2, seed=4, run_id="dp_none", dp_eps=None)
    W_inf = engine.run("trimmed", rounds=2, seed=4, run_id="dp_inf", dp_eps=float("inf"))
    assert np.array_equal(W_none, W_inf)
    rec = json.loads((mini_features / "runs" / "dp_inf" / "round_1.json").read_text())
    assert "dp_eps" not in rec["meta"]


def test_shadow_clients_write_updates_without_affecting_global_model(mini_features):
    W_ex = engine.run("fedavg", rounds=2, seed=5, run_id="ex", excluded_clients=[2])
    W_sh = engine.run("fedavg", rounds=2, seed=5, run_id="sh", shadow_clients=[2])
    assert np.array_equal(W_ex, W_sh)
    rd = mini_features / "runs" / "sh"
    rec = json.loads((rd / "round_1.json").read_text())
    assert rec["meta"]["shadow_clients"] == [2]
    assert rec["clients"]["2"]["excluded"] is True
    assert rec["clients"]["2"]["shadow"] is True
    assert rec["clients"]["2"]["reason"] == "quarantine (shadow)"
    assert rec["clients"]["2"]["update_norm"] > 0.0
    # All 6 update files exist (including shadow client 2) for reverse re-admission preview
    assert sorted(p.name for p in (rd / "updates").iterdir() if p.name.startswith("r1_")) == [
        f"r1_c{k}.npy" for k in range(N_CLIENTS)
    ]


def test_data_fraction_subsampling_is_deterministic_and_recorded(mini_features):
    W1 = engine.run("fedavg", rounds=2, seed=3, run_id="p1a", data_fraction=0.3, split_seed=42)
    W2 = engine.run("fedavg", rounds=2, seed=3, run_id="p1b", data_fraction=0.3, split_seed=42)
    assert np.array_equal(W1, W2)
    rec = json.loads((mini_features / "runs" / "p1a" / "round_1.json").read_text())
    assert rec["meta"]["data_fraction"] == 0.3
    assert rec["meta"]["split_seed"] == 42
    assert rec["meta"]["client_sizes"] == [24] * N_CLIENTS    # 30% of 80 = 24


def test_shadow_and_excluded_overlap_raises(mini_features):
    with pytest.raises(ValueError):
        engine.run("fedavg", rounds=1, run_id="bad", excluded_clients=[1], shadow_clients=[1])
