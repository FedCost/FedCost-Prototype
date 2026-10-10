"""Core of the federation engine: pure numpy, no Flower.
This must work correctly first; the Flower wrapper sits on top of it from day 3.
Usage: python engine.py --agg trimmed --attack 0 --rounds 30"""
import argparse
import numpy as np
from contract import (N_CLIENTS, N_CLASSES, FEAT_DIM, load_features,
                      save_round, save_update)

# ---------- Model: linear softmax head ----------
def predict(W, X):
    return (X @ W[:-1] + W[-1]).argmax(1)

# Defaults match reference/coalition_meta.json (Shapley coalitions); change both together.
def local_train(W, X, y, epochs=2, lr=0.05, bs=128, l2=1e-4, rng=None):
    W = W.copy()
    Y = np.eye(N_CLASSES)[y]
    for _ in range(epochs):
        idx = rng.permutation(len(X))
        for i in range(0, len(X), bs):
            b = idx[i:i + bs]
            z = X[b] @ W[:-1] + W[-1]
            z -= z.max(1, keepdims=True)
            P = np.exp(z); P /= P.sum(1, keepdims=True)
            G = (P - Y[b]) / len(b)
            W[:-1] -= lr * (X[b].T @ G + l2 * W[:-1])   # L2 on weights only, not the bias
            W[-1] -= lr * G.sum(0)
    return W

def class_recall(W, X, y):
    yp = predict(W, X)
    return [float((yp[y == c] == c).mean()) if (y == c).any() else float("nan")
            for c in range(N_CLASSES)]

# ---------- Aggregation rules: return (new_global, per_client_info) ----------
def agg_fedavg(updates, sizes):
    w = np.array(sizes, float) / sum(sizes)
    return sum(wi * u for wi, u in zip(w, updates)), {k: (False, None) for k in range(len(updates))}

def agg_trimmed(updates, sizes, beta=0.2):
    """Coordinate-wise: does NOT drop a client entirely, trims the extreme values at every coordinate.
    As 'exclusion' we report each client's fraction of trimmed coordinates."""
    U = np.stack(updates)                       # (n, d1, d2)
    n = len(U); m = int(beta * n)
    order = np.argsort(U, axis=0)
    trimmed = np.zeros(n)
    for j in list(range(m)) + list(range(n - m, n)):
        trimmed += np.bincount(order[j].ravel(), minlength=n)
    frac = trimmed / U[0].size
    agg = np.sort(U, axis=0)[m:n - m].mean(0)
    info = {k: (bool(frac[k] > 0.5), f"trimmed coordinate fraction {frac[k]:.2f}") for k in range(n)}
    return agg, info

def agg_krum(updates, sizes, f=1):
    U = np.stack([u.ravel() for u in updates]); n = len(U)
    D = ((U[:, None] - U[None]) ** 2).sum(-1)
    scores = [np.sort(D[i])[1:n - f - 1].sum() for i in range(n)]
    best = int(np.argmin(scores))
    info = {k: (k != best, None if k == best else "not selected by Krum") for k in range(n)}
    return updates[best].copy(), info

AGG = {"fedavg": agg_fedavg, "trimmed": agg_trimmed, "krum": agg_krum}

# ---------- Run ----------
def run(agg="fedavg", rounds=30, attack=None, seed=0, run_id=None):
    rng = np.random.default_rng(seed)
    run_id = run_id or f"{agg}_atk{attack}_s{seed}"
    data = [load_features(k, "train") for k in range(N_CLIENTS)]
    test = [load_features(k, "test") for k in range(N_CLIENTS)]
    if attack is not None:                                   # label flipping
        X, y = data[attack]
        data[attack] = (X, (N_CLASSES - 1 - y))
    Xg = np.concatenate([t[0] for t in test]); yg = np.concatenate([t[1] for t in test])
    W = np.zeros((FEAT_DIM + 1, N_CLASSES))
    for t in range(rounds):
        updates = [local_train(W, X, y, rng=rng) for X, y in data]
        for k, u in enumerate(updates):
            save_update(run_id, t, k, u)
        new_W, info = AGG[agg](updates, [len(d[1]) for d in data])
        clients = {k: {"class_recall": class_recall(updates[k], *test[k]),   # telemetry: on its own validation data
                       "update_norm": float(np.linalg.norm(updates[k] - W)),
                       "excluded": info[k][0], "reason": info[k][1]} for k in range(N_CLIENTS)}
        W = new_W
        gr = class_recall(W, Xg, yg)
        save_round(run_id, t, clients, gr, {"agg": agg, "attack": attack, "seed": seed})
    bal = np.nanmean(gr)
    print(f"{run_id}: balanced acc={bal:.3f} | class recall={np.round(gr, 2).tolist()}")
    return W

if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--agg", default="fedavg", choices=AGG)
    a.add_argument("--rounds", type=int, default=30)
    a.add_argument("--attack", type=int, default=None)
    a.add_argument("--seed", type=int, default=0)
    args = a.parse_args()
    run(args.agg, args.rounds, args.attack, args.seed)
