"""Synthetic data that mimics the structure of Fed-ISIC2019 until the real features arrive.
It deliberately includes two important properties of the real data:
  - Center 2: carries markedly more of the rare classes (DF, VASC)  -> "different but valuable"
  - Center 4: device difference (constant shift in the features)    -> natural deviation
CLASS_P and SIZES are updated once Şevval's real distribution table arrives.
Writes to features_synthetic/, never to features/ (that directory holds the real features).
Run the engine on it with: python engine.py --features features_synthetic"""
from pathlib import Path
import numpy as np
import contract
from contract import N_CLIENTS, N_CLASSES, FEAT_DIM, save_features

OUT_DIR = Path("features_synthetic")

SIZES = [9000, 3500, 2500, 1800, 900, 350]  # imbalanced center sizes (training)
CLASS_P = np.array([0.18, 0.49, 0.13, 0.03, 0.10, 0.01, 0.01, 0.05])  # between 49% and 1%

def make(seed=0, test_frac=0.2, out_dir=OUT_DIR):
    out_dir = Path(out_dir)
    if out_dir.resolve() == Path("features").resolve():
        raise ValueError("features/ holds the real features; write synthetic data somewhere else")
    real_dir, contract.FEATURES_DIR = contract.FEATURES_DIR, out_dir   # save_features writes to FEATURES_DIR
    try:
        _write(np.random.default_rng(seed), test_frac)
    finally:
        contract.FEATURES_DIR = real_dir
    print(f"Synthetic features written under {out_dir}/.")

def _write(rng, test_frac):
    protos = rng.normal(0, 0.06, (N_CLASSES, FEAT_DIM))  # class prototypes
    for k in range(N_CLIENTS):
        p = CLASS_P.copy()
        if k == 2:                      # rare-class-rich center
            p[5] *= 8; p[6] *= 8
        else:                           # rarer still at the others
            p[5] *= 0.3; p[6] *= 0.3
        p /= p.sum()
        n = int(SIZES[k] * (1 + test_frac))
        y = rng.choice(N_CLASSES, size=n, p=p)
        X = protos[y] + rng.normal(0, 1.0, (n, FEAT_DIM))
        if k == 4:                      # device shift
            X += rng.normal(0, 0.8, FEAT_DIM)
        cut = SIZES[k]
        save_features(k, "train", X[:cut], y[:cut])
        save_features(k, "test", X[cut:], y[cut:])

if __name__ == "__main__":
    make()
