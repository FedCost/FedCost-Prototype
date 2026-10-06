# FedCost — Federation Engine

## Project
Senior thesis: a system that measures **which disease class pays how much** when robust aggregation
(Trimmed Mean, Krum) is used in federated medical imaging, or when a hospital is excluded. Data: Fed-ISIC2019
(6 centers, 8 classes, class frequency between 49% and 1%). Details: `docs/thesis_proposal.md` (proposal), `docs/prototype_plan.md` (this week's plan).

We are in the 1-week prototype phase. The team is split into three independent tracks:
- **Umut (this repo directory):** federation engine, aggregation rules, attack, telemetry logs
- Şevval: real feature extraction (EfficientNet-B0), per-class Shapley over the 64 coalitions
- Bülent: Streamlit console, exclusion preview, audit log

For this week the work split in `docs/prototype_plan.md` is authoritative (the table in the proposal is outdated).
Do NOT do the other tracks' work. Only produce the outputs they will read, in conformance with the contract.

## Invariants
- `contract.py` is the shared contract of all three tracks. **Do not change** file formats, constants
  or function signatures. If a change is needed, stop and ask me.
- No training on images. The model is a linear softmax head over features:
  W has shape (FEAT_DIM+1, N_CLASSES), the last row is the bias.
- Every aggregation function returns `(updates, sizes) -> (new_global, info)`.
  `info[k] = (excluded: bool, reason: str|None)`. The console relies on this; do not remove it.
- **Trimmed Mean is coordinate-wise** and never drops a client entirely. For it, the "excluded" field is derived
  from the fraction of trimmed coordinates. Krum selects a single client and marks the others excluded=True.
- Keep the core logic pure numpy (`engine.py`). Flower is only a wrapper layer.
  Do not use Flower's built-in FedTrimmedAvg/Krum strategies directly; they do not expose exclusion
  information. Call our own aggregation functions inside a custom Strategy.
- Flower version is pinned to `flwr==1.39.0`. Use the new Message API (ServerApp/ClientApp),
  not the deprecated `start_simulation`. For APIs you are unsure about, inspect the installed package instead of guessing.
- Out of scope this week: differential privacy, deviation explanation, real distributed deployment.

## Commands
```
pip install -r requirements.txt
python synthetic.py                       # synthetic features -> features/
python engine.py --agg trimmed --rounds 15 [--attack 0] [--seed 1]
pytest -q
```
In the synthetic data, center 2 is rich in rare classes (DF, VASC) and center 4 has a device shift.
Expected behavior: FedAvg > Trimmed Mean > Krum (balanced acc), with the largest loss on DF/VASC.
If a change breaks this ordering, explain why.

## Code style
- English comments and log messages, English variable names.
- Runs must be reproducible via `seed`.
- `features/` and `runs/` are not committed to git.
