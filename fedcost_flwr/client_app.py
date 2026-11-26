"""ClientApp: each virtual node runs as one center (partition-id = center number).
The center reads its own features locally; only the update and summary metrics go to the server."""
import numpy as np
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

import contract
from contract import N_CLASSES, load_features
from engine import (DEFAULT_DP_CLIP, DEFAULT_DP_DELTA, class_recall,
                    local_train, predict, solve_noise_multiplier)
from .paths import apply_project_dir

app = ClientApp()

# Center data is loaded once per process (to avoid re-reading from disk every round)
_CACHE: dict = {}


def _data(cid: int, split: str):
    key = (str(contract.FEATURES_DIR), cid, split)
    if key not in _CACHE:
        _CACHE[key] = load_features(cid, split)
    return _CACHE[key]


@app.train()
def train(message: Message, context: Context) -> Message:
    apply_project_dir(context.run_config)
    cid = int(context.node_config["partition-id"])
    cfg = message.content["config"]
    rnd, seed, attack = int(cfg["server-round"]), int(cfg["seed"]), int(cfg["attack"])
    dp_eps = float(cfg.get("dp-eps", -1.0))
    dp_clip = float(cfg.get("dp-clip", DEFAULT_DP_CLIP))
    dp_delta = float(cfg.get("dp-delta", DEFAULT_DP_DELTA))
    total_rounds = int(cfg.get("rounds", 15))

    W = message.content["arrays"].to_numpy_ndarrays()[0]
    X, y = _data(cid, "train")
    if attack == cid:                                   # label-flipping attack
        y = N_CLASSES - 1 - y
    # Each (seed, round, center) triple gets its own random stream: runs are order-independent and reproducible
    rng = np.random.default_rng([seed, rnd, cid])
    dp_stats = None
    if dp_eps > 0:
        sigma = solve_noise_multiplier(len(y), total_rounds, epochs=2, bs=128, target_eps=dp_eps, delta=dp_delta)
        U, dp_stats = local_train(W, X, y, rng=rng, dp_clip=dp_clip, dp_sigma=sigma, return_stats=True)
    else:
        U = local_train(W, X, y, rng=rng)

    Xt, yt = _data(cid, "test")                         # telemetry: class recall on its own validation data
    metric_dict = {
        "client-id": cid,
        "num-examples": len(y),
        "class-recall": class_recall(U, Xt, yt),
    }
    if dp_stats is not None:
        metric_dict["clipped-frac"] = dp_stats["clipped_frac"]
        metric_dict["class-clipped-frac"] = dp_stats["class_clipped_frac"]
    metrics = MetricRecord(metric_dict)
    content = RecordDict({"arrays": ArrayRecord.from_numpy_ndarrays([U]), "metrics": metrics})
    return Message(content=content, reply_to=message)


@app.evaluate()
def evaluate(message: Message, context: Context) -> Message:
    """Evaluates the global model on its own test data; returns per-class correct/total counts.
    The server sums them to obtain class recall on the pooled test set."""
    apply_project_dir(context.run_config)
    cid = int(context.node_config["partition-id"])
    W = message.content["arrays"].to_numpy_ndarrays()[0]
    Xt, yt = _data(cid, "test")
    yp = predict(W, Xt)
    correct = [int((yp[yt == c] == c).sum()) for c in range(N_CLASSES)]
    count = [int((yt == c).sum()) for c in range(N_CLASSES)]
    metrics = MetricRecord({"client-id": cid, "num-examples": len(yt),
                            "correct": correct, "count": count})
    return Message(content=RecordDict({"metrics": metrics}), reply_to=message)
