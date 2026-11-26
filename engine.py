"""Core of the federation engine: pure numpy, no Flower.
This must work correctly first; the Flower wrapper sits on top of it from day 3.
Usage: python engine.py --agg trimmed --attack 0 --rounds 30 [--dp-eps 8]"""
import argparse
from functools import lru_cache
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from scipy.special import gammaln, logsumexp
import contract
from contract import (N_CLIENTS, N_CLASSES, FEAT_DIM, load_features,
                      save_round, save_update)

# ---------- Differential Privacy (record-level DP-SGD, Mironov 2019 RDP) ----------
_RDP_ALPHAS = np.arange(2, 129, dtype=float)
DEFAULT_DP_CLIP = 10.0     # ~median per-sample grad norm on Fed-ISIC2019 EfficientNet-B0 features
DEFAULT_DP_DELTA = 1e-5    # satisfies delta < 1/N_k for all 6 Fed-ISIC2019 centers (N_k in [351, 9930])


def rdp_subsampled_gaussian(q, sigma, alpha):
    """Exact Renyi DP of one subsampled Gaussian step at integer order alpha >= 2 (Mironov 2019)."""
    if q >= 1.0:
        return float(alpha / (2.0 * sigma * sigma))
    k = np.arange(0, int(alpha) + 1, dtype=float)
    log_binom = gammaln(alpha + 1.0) - gammaln(k + 1.0) - gammaln(alpha - k + 1.0)
    log_terms = (log_binom
                 + (alpha - k) * np.log(1.0 - q)
                 + k * np.log(q)
                 + k * (k - 1.0) / (2.0 * sigma * sigma))
    return float(logsumexp(log_terms) / (alpha - 1.0))


def eps_from_sigma(q, sigma, steps, delta=DEFAULT_DP_DELTA):
    """Converts T steps of subsampled Gaussian RDP to (epsilon, delta)-DP (Balle et al. 2020)."""
    rdps = np.array([steps * rdp_subsampled_gaussian(q, sigma, a) for a in _RDP_ALPHAS])
    eps = rdps + np.log((_RDP_ALPHAS - 1.0) / _RDP_ALPHAS) - (np.log(delta) + np.log(_RDP_ALPHAS)) / (_RDP_ALPHAS - 1.0)
    return float(np.min(eps))


@lru_cache(maxsize=256)
def solve_noise_multiplier(n_samples, rounds, epochs, bs, target_eps, delta=DEFAULT_DP_DELTA):
    """Solves for per-step Gaussian noise multiplier sigma achieving (target_eps, delta)-DP."""
    if target_eps is None or not np.isfinite(target_eps):
        return 0.0
    if target_eps <= 0:
        raise ValueError(f"dp_eps must be positive, got {target_eps}")
    q = min(1.0, bs / max(int(n_samples), 1))
    steps = max(1, int(rounds) * int(epochs) * int(np.ceil(n_samples / bs)))
    lo, hi = 0.05, 100.0
    while eps_from_sigma(q, hi, steps, delta) > target_eps:
        hi *= 2.0
    return float(brentq(lambda s: eps_from_sigma(q, s, steps, delta) - target_eps, lo, hi, xtol=1e-4))


# ---------- Model: linear softmax head ----------
def predict(W, X):
    return (X @ W[:-1] + W[-1]).argmax(1)

# Defaults match reference/coalition_meta.json (Shapley coalitions); change both together.
def local_train(W, X, y, epochs=2, lr=0.05, bs=128, l2=1e-4, rng=None,
                dp_clip=None, dp_sigma=0.0, return_stats=False):
    W = W.copy()
    Y = np.eye(N_CLASSES)[y]
    if dp_clip is None:
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
        return (W, None) if return_stats else W

    # Record-level DP-SGD: exact per-sample L2 gradient clipping via rank-1 identity
    # ||[x_i e_i^T; e_i^T]||_F = sqrt((||x_i||_2^2 + 1) * ||e_i||_2^2)
    x_sq = np.sum(X * X, axis=1)
    clipped_by_cls = np.zeros(N_CLASSES, dtype=np.int64)
    total_by_cls = np.zeros(N_CLASSES, dtype=np.int64)
    for _ in range(epochs):
        idx = rng.permutation(len(X))
        for i in range(0, len(X), bs):
            b = idx[i:i + bs]
            Xb = X[b]
            yb = y[b]
            z = Xb @ W[:-1] + W[-1]
            z -= z.max(1, keepdims=True)
            P = np.exp(z); P /= P.sum(1, keepdims=True)
            E = P - Y[b]
            norms = np.sqrt((x_sq[b] + 1.0) * np.sum(E * E, axis=1))
            clipped_mask = norms > dp_clip
            scale = np.minimum(1.0, dp_clip / np.maximum(norms, 1e-12))
            total_by_cls += np.bincount(yb, minlength=N_CLASSES)
            if np.any(clipped_mask):
                clipped_by_cls += np.bincount(yb[clipped_mask], minlength=N_CLASSES)
            Ec = E * scale[:, None]
            gW = (Xb.T @ Ec) / len(b)
            gb = Ec.sum(0) / len(b)
            if dp_sigma > 0.0:
                std = dp_sigma * dp_clip / len(b)
                gW = gW + rng.normal(0.0, std, size=gW.shape)
                gb = gb + rng.normal(0.0, std, size=gb.shape)
            W[:-1] -= lr * (gW + l2 * W[:-1])
            W[-1] -= lr * gb
    if not return_stats:
        return W
    total_all = int(total_by_cls.sum())
    stats = {
        "clipped_frac": float(clipped_by_cls.sum() / total_all) if total_all > 0 else 0.0,
        "class_clipped_frac": [
            float(clipped_by_cls[c] / total_by_cls[c]) if total_by_cls[c] > 0 else float("nan")
            for c in range(N_CLASSES)
        ],
    }
    return W, stats

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
def _active_dp(dp_eps):
    return dp_eps is not None and np.isfinite(dp_eps)


def subsample_client(X, y, frac=1.0, split_seed=42, client_id=0):
    """Deterministic local subsampling for Period 1 (e.g. 30%) -> Period 2 (100%) lifecycle runs.
    Shared helper so the reference track (Sevval) uses the exact same subset per (split_seed, client_id)."""
    if frac >= 1.0:
        return X, y
    if not (0.0 < frac <= 1.0):
        raise ValueError(f"data_fraction must be in (0, 1], got {frac}")
    rng = np.random.default_rng([int(split_seed), int(client_id)])
    n_keep = max(1, int(round(frac * len(y))))
    idx = np.sort(rng.choice(len(y), size=n_keep, replace=False))
    return X[idx], y[idx]


def run_meta(agg, attack, seed, sizes=None, excluded_clients=(),
             dp_eps=None, dp_clip=DEFAULT_DP_CLIP, dp_delta=DEFAULT_DP_DELTA,
             shadow_clients=(), data_fraction=1.0, split_seed=42):
    """Round-log meta. 'aggregation'/'attack_client' are the keys the console reads; 'client_sizes' are the
    train set sizes in center order, which the exclusion preview needs to reproduce size-weighted FedAvg."""
    meta = {"agg": agg, "attack": attack, "seed": seed, "aggregation": agg, "attack_client": attack}
    if sizes is not None:
        meta["client_sizes"] = [int(s) for s in sizes]
    if excluded_clients:
        meta["excluded_clients"] = sorted(int(k) for k in excluded_clients)
    if shadow_clients:
        meta["shadow_clients"] = sorted(int(k) for k in shadow_clients)
    if data_fraction < 1.0:
        meta["data_fraction"] = float(data_fraction)
        meta["split_seed"] = int(split_seed)
    if _active_dp(dp_eps):
        meta["dp_eps"] = float(dp_eps)
        meta["dp_clip"] = float(dp_clip)
        meta["dp_delta"] = float(dp_delta)
    return meta

def run(agg="fedavg", rounds=30, attack=None, seed=0, run_id=None, excluded_clients=None,
        dp_eps=None, dp_clip=DEFAULT_DP_CLIP, dp_delta=DEFAULT_DP_DELTA,
        shadow_clients=None, data_fraction=1.0, split_seed=42):
    """excluded_clients: centers left out of training completely (no update file written).
    shadow_clients: quarantined centers excluded from global aggregation, yet still training locally and
    writing updates/r{t}_c{k}.npy so the console can run reverse re-admission previews without retraining.
    data_fraction: fraction of each center's training data used (e.g. 0.30 for Period 1, 1.0 for Period 2)."""
    rng = np.random.default_rng(seed)
    shadow_rng = np.random.default_rng([int(seed), 99991])    # isolated RNG so active global W is unchanged
    dp_on = _active_dp(dp_eps)
    left_out = set(excluded_clients or [])
    shadow = set(shadow_clients or [])
    if left_out & shadow:
        raise ValueError(f"clients cannot be both excluded and shadow: {sorted(left_out & shadow)}")
    active = [k for k in range(N_CLIENTS) if k not in left_out and k not in shadow]
    if not active:
        raise ValueError("at least one center must stay in the coalition")
    if run_id is None:
        dp_tag = f"_dp{dp_eps:g}" if dp_on else ""
        frac_tag = f"_f{int(round(data_fraction * 100))}" if data_fraction < 1.0 else ""
        sh_tag = f"_sh{''.join(str(k) for k in sorted(shadow))}" if shadow else ""
        run_id = f"{agg}_atk{attack}{dp_tag}{frac_tag}{sh_tag}_s{seed}"
    raw_data = [load_features(k, "train") for k in range(N_CLIENTS)]
    data = [subsample_client(X, y, frac=data_fraction, split_seed=split_seed, client_id=k)
            for k, (X, y) in enumerate(raw_data)]
    test = [load_features(k, "test") for k in range(N_CLIENTS)]
    if attack in active or attack in shadow:                 # label flipping
        X, y = data[attack]
        data[attack] = (X, (N_CLASSES - 1 - y))
    Xg = np.concatenate([t[0] for t in test]); yg = np.concatenate([t[1] for t in test])
    sizes = [len(d[1]) for d in data]
    sigmas = {
        k: solve_noise_multiplier(sizes[k], rounds, epochs=2, bs=128, target_eps=float(dp_eps), delta=dp_delta)
        if dp_on else 0.0
        for k in range(N_CLIENTS) if k not in left_out
    }
    meta = run_meta(agg, attack, seed, sizes, left_out, dp_eps=dp_eps, dp_clip=dp_clip, dp_delta=dp_delta,
                    shadow_clients=shadow, data_fraction=data_fraction, split_seed=split_seed)
    W = np.zeros((FEAT_DIM + 1, N_CLASSES))
    for t in range(rounds):
        updates, dp_stats = [], {}
        for k in active:
            if dp_on:
                u, st = local_train(W, *data[k], rng=rng, dp_clip=dp_clip, dp_sigma=sigmas[k], return_stats=True)
                dp_stats[k] = st
            else:
                u = local_train(W, *data[k], rng=rng)
            updates.append(u)
        for k, u in zip(active, updates):
            save_update(run_id, t, k, u)
        new_W, info = AGG[agg](updates, [sizes[k] for k in active])   # info indices follow list order
        clients = {}
        for i, (k, u) in enumerate(zip(active, updates)):
            entry = {"class_recall": class_recall(u, *test[k]),       # telemetry: on its own validation data
                     "update_norm": float(np.linalg.norm(u - W)),
                     "excluded": info[i][0], "reason": info[i][1]}
            if dp_on:
                entry["clipped_frac"] = dp_stats[k]["clipped_frac"]
                entry["class_clipped_frac"] = dp_stats[k]["class_clipped_frac"]
            clients[k] = entry
        for k in sorted(shadow):
            if dp_on:
                u_sh, st_sh = local_train(W, *data[k], rng=shadow_rng,
                                          dp_clip=dp_clip, dp_sigma=sigmas[k], return_stats=True)
            else:
                u_sh, st_sh = local_train(W, *data[k], rng=shadow_rng), None
            save_update(run_id, t, k, u_sh)                  # saved for reverse re-admission preview
            entry = {"class_recall": class_recall(u_sh, *test[k]),
                     "update_norm": float(np.linalg.norm(u_sh - W)),
                     "excluded": True, "shadow": True, "reason": "quarantine (shadow)"}
            if dp_on and st_sh is not None:
                entry["clipped_frac"] = st_sh["clipped_frac"]
                entry["class_clipped_frac"] = st_sh["class_clipped_frac"]
            clients[k] = entry
        for k in left_out:
            clients[k] = {"class_recall": [float("nan")] * N_CLASSES, "update_norm": 0.0,
                          "excluded": True, "reason": "left out of the coalition"}
        W = new_W
        gr = class_recall(W, Xg, yg)
        save_round(run_id, t, clients, gr, meta)
    bal = np.nanmean(gr)
    print(f"{run_id}: balanced acc={bal:.3f} | class recall={np.round(gr, 2).tolist()}")
    return W

if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--agg", default="fedavg", choices=AGG)
    a.add_argument("--rounds", type=int, default=30)
    a.add_argument("--attack", type=int, default=None)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--dp-eps", type=float, default=None, help="record-level DP-SGD target epsilon (e.g. 8 or 3)")
    a.add_argument("--dp-clip", type=float, default=DEFAULT_DP_CLIP, help="per-sample L2 gradient clipping norm")
    a.add_argument("--dp-delta", type=float, default=DEFAULT_DP_DELTA, help="DP delta (default 1e-5)")
    a.add_argument("--shadow", type=int, nargs="*", default=None, help="quarantined center IDs in shadow mode")
    a.add_argument("--data-fraction", type=float, default=1.0, help="training data fraction in (0, 1], e.g. 0.3")
    a.add_argument("--split-seed", type=int, default=42, help="seed for deterministic data_fraction subsampling")
    a.add_argument("--features", default="features", help="feature directory, e.g. features_synthetic")
    args = a.parse_args()
    contract.FEATURES_DIR = Path(args.features)
    run(args.agg, args.rounds, args.attack, args.seed,
        dp_eps=args.dp_eps, dp_clip=args.dp_clip, dp_delta=args.dp_delta,
        shadow_clients=args.shadow, data_fraction=args.data_fraction, split_seed=args.split_seed)
