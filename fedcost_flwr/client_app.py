"""ClientApp: each virtual node runs as one center (partition-id = center number).
The center reads its own features locally; only the update and summary metrics go to the server."""
import numpy as np
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

import contract
from contract import N_CLASSES, load_features
from engine import class_recall, local_train, predict
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

    W = message.content["arrays"].to_numpy_ndarrays()[0]
    X, y = _data(cid, "train")
    if attack == cid:                                   # label-flipping attack
        y = N_CLASSES - 1 - y
    # Each (seed, round, center) triple gets its own random stream: runs are order-independent and reproducible
    rng = np.random.default_rng([seed, rnd, cid])
    U = local_train(W, X, y, rng=rng)

    Xt, yt = _data(cid, "test")                         # telemetry: class recall on its own validation data
    metrics = MetricRecord({
        "client-id": cid,
        "num-examples": len(y),
        "class-recall": class_recall(U, Xt, yt),
    })
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
