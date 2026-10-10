"""Synthetic data that mimics the structure of Fed-ISIC2019 until the real features arrive.
It deliberately includes two important properties of the real data:
  - Center 2: carries markedly more of the rare classes (DF, VASC)  -> "different but valuable"
  - Center 4: device difference (constant shift in the features)    -> natural deviation
CLASS_P and SIZES are updated once Şevval's real distribution table arrives."""
import numpy as np
from contract import N_CLIENTS, N_CLASSES, FEAT_DIM, save_features

SIZES = [9000, 3500, 2500, 1800, 900, 350]  # imbalanced center sizes (training)
CLASS_P = np.array([0.18, 0.49, 0.13, 0.03, 0.10, 0.01, 0.01, 0.05])  # between 49% and 1%

def make(seed=0, test_frac=0.2):
    rng = np.random.default_rng(seed)
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
    print("Synthetic features written under features/.")

if __name__ == "__main__":
    make()
