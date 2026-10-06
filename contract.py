"""Shared data contract. All three tracks import this file; whoever changes the format must tell the others."""
import json
from pathlib import Path
import numpy as np

N_CLIENTS = 6
N_CLASSES = 8
FEAT_DIM = 1280  # EfficientNet-B0 global pooling output
CLASS_NAMES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]

FEATURES_DIR = Path("features")
RUNS_DIR = Path("runs")


# --- Features: features/center_{k}_{train|test}.npz -> X (N, 1280) float32, y (N,) int
def save_features(k, split, X, y):
    FEATURES_DIR.mkdir(exist_ok=True)
    np.savez_compressed(FEATURES_DIR / f"center_{k}_{split}.npz",
                        X=X.astype(np.float32), y=y.astype(np.int64))

def load_features(k, split):
    d = np.load(FEATURES_DIR / f"center_{k}_{split}.npz")
    return d["X"], d["y"]


# --- Round log: runs/{run_id}/round_{t}.json
def save_round(run_id, t, clients, global_class_recall, meta=None):
    """clients: {k: {"class_recall": [8], "update_norm": float, "excluded": bool, "reason": str|None}}"""
    p = RUNS_DIR / run_id
    p.mkdir(parents=True, exist_ok=True)
    rec = {"round": t, "clients": {str(k): v for k, v in clients.items()},
           "global_class_recall": list(map(float, global_class_recall)), "meta": meta or {}}
    (p / f"round_{t}.json").write_text(json.dumps(rec, indent=2))


# --- Client update: runs/{run_id}/updates/r{t}_c{k}.npy -> W (FEAT_DIM+1, N_CLASSES), last row is the bias
def save_update(run_id, t, k, W):
    p = RUNS_DIR / run_id / "updates"
    p.mkdir(parents=True, exist_ok=True)
    np.save(p / f"r{t}_c{k}.npy", W.astype(np.float32))

def load_update(run_id, t, k):
    return np.load(RUNS_DIR / run_id / "updates" / f"r{t}_c{k}.npy")
