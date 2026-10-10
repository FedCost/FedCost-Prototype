"""Flower layer tests: one full round run in-process (no simulation runtime)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, RecordDict
from flwr.supercore.task_identity import TaskIdentity

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import contract
import engine
from contract import FEAT_DIM, N_CLASSES, N_CLIENTS
from fedcost_flwr import client_app
from fedcost_flwr.strategy import FedCostStrategy


@pytest.fixture
def project(tmp_path, monkeypatch):
    for attr in ("_task_id", "_run_id", "_node_id"):           # normally set by the Flower runtime
        monkeypatch.setattr(TaskIdentity, attr, 1)
    feats, runs = contract.FEATURES_DIR, contract.RUNS_DIR
    contract.FEATURES_DIR, contract.RUNS_DIR = tmp_path / "features", tmp_path / "runs"
    rng = np.random.default_rng(0)
    protos = rng.normal(0, 1, (N_CLASSES, FEAT_DIM))
    for k in range(N_CLIENTS):
        for split, n in (("train", 80), ("test", 40)):
            y = rng.integers(0, N_CLASSES, n)
            contract.save_features(k, split, protos[y] + rng.normal(0, 1, (n, FEAT_DIM)), y)
    client_app._CACHE.clear()
    yield tmp_path
    contract.FEATURES_DIR, contract.RUNS_DIR = feats, runs      # the app redirects these globals at runtime


def ask(fn, kind, cid, W, cfg, root):
    """Sends one message to a center's ClientApp function and returns its reply."""
    content = RecordDict({"arrays": ArrayRecord.from_numpy_ndarrays([W]), "config": ConfigRecord(cfg)})
    msg = Message(content=content, dst_node_id=100 + cid, message_type=kind)
    ctx = Context(run_id=1, node_id=100 + cid, node_config={"partition-id": cid},
                  state=RecordDict(), run_config={"project-dir": str(root)})
    return fn(msg, ctx)


def one_round(agg, root, attack=-1, seed=3, order=range(N_CLIENTS)):
    meta = {"agg": agg, "attack": None if attack < 0 else attack, "seed": seed}
    strategy = FedCostStrategy(agg, "t", meta)
    W0 = np.zeros((FEAT_DIM + 1, N_CLASSES))
    strategy._W = W0                                           # set by configure_train in a real run
    cfg = {"server-round": 1, "seed": seed, "attack": attack}
    replies = [ask(client_app.train, "train", k, W0, cfg, root) for k in order]
    arrays, _ = strategy.aggregate_train(1, replies)
    W1 = arrays.to_numpy_ndarrays()[0]
    replies = [ask(client_app.evaluate, "evaluate", k, W1, {"server-round": 1}, root) for k in order]
    metrics = strategy.aggregate_evaluate(1, replies)
    return W1, metrics, json.loads((root / "runs" / "t" / "round_0.json").read_text())


@pytest.mark.parametrize("agg", ["fedavg", "trimmed", "krum"])
def test_round_writes_contract_conformant_output(project, agg):
    W1, metrics, rec = one_round(agg, project, attack=0)
    assert set(rec["clients"]) == {str(k) for k in range(N_CLIENTS)}
    for c in rec["clients"].values():
        assert len(c["class_recall"]) == N_CLASSES and c["update_norm"] > 0
        assert isinstance(c["excluded"], bool)
    assert rec["meta"] == {"agg": agg, "attack": 0, "seed": 3}
    assert metrics["balanced-acc"] == pytest.approx(np.nanmean(rec["global_class_recall"]))
    for k in range(N_CLIENTS):
        assert contract.load_update("t", 0, k).shape == (FEAT_DIM + 1, N_CLASSES)
    if agg == "krum":
        assert sum(not c["excluded"] for c in rec["clients"].values()) == 1


def test_strategy_matches_engine_aggregation(project):
    """The wrapper must add nothing: its global model equals engine.AGG on the same client updates."""
    W1, _, rec = one_round("trimmed", project)
    ups = [contract.load_update("t", 0, k) for k in range(N_CLIENTS)]
    sizes = [len(contract.load_features(k, "train")[1]) for k in range(N_CLIENTS)]
    expected, info = engine.AGG["trimmed"](ups, sizes)
    assert np.allclose(W1, expected, atol=1e-6)                # saved updates are float32
    assert [rec["clients"][str(k)]["excluded"] for k in range(N_CLIENTS)] == [info[k][0] for k in range(N_CLIENTS)]


def test_global_recall_equals_pooled_test_recall(project):
    W1, _, rec = one_round("fedavg", project)
    X = np.concatenate([contract.load_features(k, "test")[0] for k in range(N_CLIENTS)])
    y = np.concatenate([contract.load_features(k, "test")[1] for k in range(N_CLIENTS)])
    assert np.allclose(rec["global_class_recall"], engine.class_recall(W1, X, y))


def test_reply_order_does_not_change_result(project):
    W_a, _, _ = one_round("krum", project)
    W_b, _, _ = one_round("krum", project, order=[4, 1, 5, 0, 3, 2])
    assert np.array_equal(W_a, W_b)


def test_attack_only_changes_attacked_center(project):
    one_round("fedavg", project, attack=-1)
    clean = [contract.load_update("t", 0, k) for k in range(N_CLIENTS)]
    one_round("fedavg", project, attack=2)
    attacked = [contract.load_update("t", 0, k) for k in range(N_CLIENTS)]
    changed = [not np.array_equal(a, b) for a, b in zip(clean, attacked)]
    assert changed == [k == 2 for k in range(N_CLIENTS)]
