# reference/ — data track outputs (Şevval)

Reference measurements for the console. Nothing here changes `contract.py`: the Shapley files are a
convention documented in this README, not part of the shared contract.

## Conventions

- Centers are indexed 0-5, classes follow `contract.CLASS_NAMES` = `MEL, NV, BCC, AK, BKL, DF, VASC, SCC`.
- Value function `v(S)`: per-class recall of the FedAvg linear head trained only on the centers in coalition `S`,
  evaluated on the **pooled test set of all 6 centers** (4650 images).
- `v(empty set) = 0` for every class. With this choice, for each class the Shapley values of the 6 centers
  add up to the recall of the full federation.
- Training setup (pure numpy, zero init, FedAvg weighted by client size, minibatch SGD): 15 rounds, 2 local
  epochs, lr 0.05, batch 128, L2 1e-4. Exact values are in `coalition_meta.json`.
  These must stay aligned with `engine.py`; if they change, the coalitions are re-run (about 14 minutes).
- 20 seeds (0-19). Per-class recall varies a lot between seeds (std up to about 0.07 for BCC/BKL on the full
  federation), so everything is reported as a mean over seeds with the std next to it.

## Files

| File | Shape / format | Content |
|---|---|---|
| `shapley_class.npy` | (6, 8) float32 | Exact per-class Shapley value, mean over seeds. Rows = centers, columns = classes. |
| `shapley_class_std.npy` | (6, 8) float32 | Std over seeds of the same values (`np.std`, ddof=0). |
| `shapley_class_seeds.npy` | (20, 6, 8) float32 | Per-seed Shapley values. |
| `shapley_class.csv` | text | Mean table; the last row is the column sum (= full-federation recall). |
| `shapley_heatmap.png` | image | Heatmap of the mean values with std in each cell. |
| `coalition_recall_seeds.npy` | (64, 20, 8) float64 | `v(S)` for every coalition and seed. Index = bitmask: bit k set means center k is in S. Index 0 (empty) is all zeros, index 63 is the full federation. |
| `coalition_meta.json` | json | Value function, index convention and hyperparameters. |
| `class_distribution.csv` | text, 12 rows | Center x class counts. Columns: `split, center, MEL..SCC, total`. |
| `features_meta.json` | json | Backbone, preprocessing, dataset revision. |

## Loading

```python
import numpy as np
from contract import CLASS_NAMES

phi = np.load("reference/shapley_class.npy")          # (6, 8)
std = np.load("reference/shapley_class_std.npy")      # (6, 8)
V   = np.load("reference/coalition_recall_seeds.npy") # (64, 20, 8)

# Exact cost of excluding center k from the full federation, per class (retrained from scratch).
# Positive = recall lost when center k is excluded.
FULL = 63
exclusion_cost = V[FULL].mean(0) - np.stack([V[FULL ^ (1 << k)].mean(0) for k in range(6)])  # (6, 8)

# Cells where the value is smaller than its seed noise
low_confidence = np.abs(phi) < std                    # (6, 8) bool
```

`exclusion_cost` is a ground truth for the exclusion preview: it retrains the federation without center k,
while the preview re-aggregates saved weights. The gap between the two is the RQ3 question.

## Caveats the UI should respect

1. **DF, SCC and AK are barely learned** by plain FedAvg (full-federation recall about 0.03, 0.06 and 0.17).
   Their Shapley values are near zero and mostly inside the seed noise. Do not present them as precise.
2. **Small test classes.** DF has 55 and VASC 52 test images, so one image is about 2 recall points.
3. **Seed noise.** Show the std, and consider greying out cells where `low_confidence` is True.
4. **Class order.** The Hugging Face dataset only has integer labels 0-7. The order was checked against
   class frequencies (NV about 49%, DF and VASC about 1%), but DF vs VASC should still be confirmed against
   the original label file.

## Provenance

- Data: `flwrlabs/fed-isic2019` at revision `d1457caea44f57dfe855bcf67e2ec09bb06781e9` (Flower's Hugging Face
  copy derived from FLamby; same 6-center split; images already resized to about 224 px). License CC BY-NC 4.0.
- Features: torchvision `efficientnet_b0`, `IMAGENET1K_V1` weights, `Resize(224)` + `CenterCrop(224)` + ImageNet
  normalization, no augmentation, 1280-d global-pooled output.
- `features/` and `runs/` are not in git. Ask for the 12 `.npz` feature files if you need the real features.
