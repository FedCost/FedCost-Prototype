# One-Week Prototype Plan

The advisor asked for a working prototype of the project within a week. The prototype is not the whole thesis but a
single end-to-end slice: the federation runs, the console shows per-class recall round by round, an
aggregation rule excludes a hospital, and before the operator says "exclude" they see which class would lose how much.

## Core technical decision

No training on images. Features of the 23,247 images are extracted once with a pretrained EfficientNet-B0
(1280 dimensions) and saved as one `.npz` per center. Federated training trains only a
linear classifier head. This means:

- a federation run takes seconds; a laptop is enough;
- the exclusion preview can run in real time;
- exact per-class Shapley values over all 64 coalitions of the 6 clients can be computed in the first week.

## Three independent tracks

The work is split evenly and independently among three people. Independence is provided by the shared contract in `contract.py`:
everyone starts with synthetic data that conforms to the contract, and the real pieces are merged on day 6.

| Track | Owner | Produces |
|---|---|---|
| Federation engine | Umut | Aggregation rules (FedAvg, Trimmed Mean, Krum), label-flipping attack, client telemetry, contract-conformant log and weight files |
| Data and reference measurement | Şevval | ISIC 2019 download, FLamby center split, real features, center × class distribution table, per-class Shapley over 64 coalitions |
| Console and exclusion preview | Bülent | Streamlit console, exclusion preview module, defense comparison view, audit log |

## Shared contract (summary; the source of truth is `contract.py`)

```
features/center_{k}_{train|test}.npz   → X: (N, 1280) float32, y: (N,) int 0–7
runs/{run_id}/round_{t}.json           → for each client: class_recall[8], update_norm,
                                          excluded, reason; plus global_class_recall[8]
runs/{run_id}/updates/r{t}_c{k}.npy    → the client's classifier-head weights for that round
                                          (FEAT_DIM+1, N_CLASSES), last row is the bias
```

## Day-by-day flow

| Day | Umut | Şevval | Bülent |
|---|---|---|---|
| 1 | Contract, environment, synthetic data, pure-numpy engine | Data download, FLamby setup | Fake log generator, Streamlit skeleton |
| 2 | Engine tests (pytest) | Feature extraction | Round-by-round class recall chart |
| 3–4 | Port to Flower: custom Strategy + ClientApp, telemetry | Distribution table, pure-numpy FedAvg, training of the 64 coalitions | Exclusion preview module (remove a client from saved weights and re-aggregate) |
| 5 | Multi-seed comparison runs (3 rules × attacked/unattacked) | Per-class Shapley and heatmap | "Exclude" button, defense comparison view, audit log |
| 6 | **Merge:** real features into the engine, real logs into the console, Shapley next to the preview | | |
| 7 | Demo rehearsal; everyone presents their own part in 3 minutes | | |

## Definitions of "done"

- **Engine:** a single command runs with the chosen defense and attack and writes contract-conformant logs and weights.
- **Data and reference:** real feature files and the 6 center × 8 class Shapley table are ready.
- **Console:** opens a real run, previews the exclusion of a hospital and records it.

## Out of scope (this week)

Differential privacy, deviation explanation, real distributed deployment. If time remains, the first thing to add is
DP-SGD with Opacus on the linear head (about half a day).

## Known risks

- **Data preparation:** FLamby scripts may break with new Python/PyTorch versions. Fallback:
  take only the center split from FLamby's metadata CSV and load the images directly.
  If delayed, the other tracks continue with synthetic data and only day 6 gets tight.
- **Flower API:** it changed a lot between versions; `flwr==1.39.0` is pinned.
- **Meaning of the aggregation rules:** Trimmed Mean is coordinate-wise and does not drop a client entirely;
  for it, the "excluded" label is derived from the fraction of trimmed coordinates. Krum is the one that excludes whole clients.
  So the meaning of the "excluded" label in the console depends on the rule.
- **Demo scenario depends on the data:** it is not guaranteed that a rule will exclude the center carrying the rare classes
  on its own. If it is not excluded, the operator starts the exclusion manually in the demo; that is also a finding.

## Expected behavior on synthetic data

Center 2 is rich in rare classes (DF, VASC) and center 4 has a device shift. Balanced accuracy measured over 15 rounds:
FedAvg about 0.52, Trimmed Mean 0.49, Krum 0.34. The loss is largest on DF and VASC. Under the
label-flipping attack on center 0, FedAvg collapses to 0.18; Krum blocks the attack but the cost falls on the rare classes.
If a change breaks this ordering, the cause must be investigated.
