"""Two-period decision lifecycle & shadow-mode simulation (RQ3-Ek / Advisor Q1).

Simulates Period 1 (30% initial training data) -> Period 2 (100% full training data)
with a quarantined center operating in shadow mode:
  - The shadow center trains locally and writes updates/r{t}_c{k}.npy,
    but its update is excluded from global aggregation.
  - Reverse re-admission preview (0.2s, no retraining) compares the current global model
    (without shadow center k) against re-aggregating WITH center k's shadow update.
  - Ground-truth LOO re-admission gain compares retraining all 6 centers vs retraining without k.

Usage:
    python scripts/lifecycle_sim.py --centers 0 1 2 --seeds 20 --rounds 15
"""
import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import contract
import engine
import preview
from contract import CLASS_NAMES, N_CLIENTS


def readmission_preview(run_id, round_t, shadow_k, rule="fedavg"):
    """Reverse exclusion preview for a quarantined shadow center k.
    Returns (current_recalls_without_k, readmitted_recalls_with_k, delta_gain)."""
    res = preview.preview_exclusion(run_id, round_t, excluded_clients=[shadow_k], rule=rule)
    # In preview_exclusion:
    #   full_recalls    = model WITH all saved updates (i.e. WITH shadow center k re-admitted)
    #   preview_recalls = model WITHOUT shadow center k (i.e. current quarantined global model)
    with_k = np.array(res["full_recalls"], dtype=float)
    without_k = np.array(res["preview_recalls"], dtype=float)
    gain = with_k - without_k   # positive = recall gained by re-admitting center k
    return without_k, with_k, gain


def run_period_comparison(center_k, frac, seeds, rounds, agg, split_seed, keep_seed):
    """For a given (center_k, data_fraction), runs:
      1. Shadow run (center_k quarantined in shadow mode) -> reverse preview gain (0.2s)
      2. Full coalition run (all 6 centers active) -> exact retrain re-admission gain
    Returns dict of (seeds, 8) arrays."""
    prev_gains, true_gains = [], []
    shadow_recalls, full_recalls = [], []
    for s in range(seeds):
        p_tag = f"p1_f{int(round(frac * 100))}" if frac < 1.0 else "p2_f100"
        sh_id = f"lifecycle_{p_tag}_shadow_c{center_k}_s{s}"
        full_id = f"lifecycle_{p_tag}_full_s{s}"

        engine.run(agg, rounds=rounds, seed=s, run_id=sh_id,
                   shadow_clients=[center_k], data_fraction=frac, split_seed=split_seed)
        _, _, pgain = readmission_preview(sh_id, rounds - 1, center_k, rule=agg)

        # Reuse full run across centers for the same (period, seed)
        full_last = contract.RUNS_DIR / full_id / f"round_{rounds - 1}.json"
        if not full_last.exists():
            engine.run(agg, rounds=rounds, seed=s, run_id=full_id,
                       data_fraction=frac, split_seed=split_seed)
            if s != keep_seed:
                shutil.rmtree(contract.RUNS_DIR / full_id / "updates", ignore_errors=True)

        sh_log = json.loads((contract.RUNS_DIR / sh_id / f"round_{rounds - 1}.json").read_text())
        fl_log = json.loads(full_last.read_text())
        sh_rec = np.array(sh_log["global_class_recall"], dtype=float)
        fl_rec = np.array(fl_log["global_class_recall"], dtype=float)

        shadow_recalls.append(sh_rec)
        full_recalls.append(fl_rec)
        prev_gains.append(pgain)
        true_gains.append(fl_rec - sh_rec)

        if s != keep_seed:
            shutil.rmtree(contract.RUNS_DIR / sh_id / "updates", ignore_errors=True)

    return {
        "shadow_recall": np.stack(shadow_recalls),
        "full_recall": np.stack(full_recalls),
        "preview_gain": np.stack(prev_gains),
        "true_gain": np.stack(true_gains),
    }


def fmt(m, s):
    return f"{m:+.2f}±{s:.2f}"


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--centers", type=int, nargs="+", default=[0, 1, 2],
                   help="quarantined shadow centers to evaluate across Period 1 -> Period 2")
    a.add_argument("--p1-frac", type=float, default=0.30, help="Period 1 data fraction (default 0.30)")
    a.add_argument("--p2-frac", type=float, default=1.00, help="Period 2 data fraction (default 1.00)")
    a.add_argument("--seeds", type=int, default=20)
    a.add_argument("--rounds", type=int, default=15)
    a.add_argument("--agg", default="fedavg", choices=engine.AGG)
    a.add_argument("--split-seed", type=int, default=42)
    a.add_argument("--keep-demo-seed", type=int, default=0,
                   help="keep updates/ for this seed so the Streamlit console can open and preview it")
    a.add_argument("--out", default="results/lifecycle")
    a.add_argument("--features", default="features")
    args = a.parse_args()
    contract.FEATURES_DIR = Path(args.features)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Clear cached full runs from previous invocations so split_seed/rounds match
    for p_tag in (f"p1_f{int(round(args.p1_frac * 100))}", "p2_f100"):
        for s in range(args.seeds):
            d = contract.RUNS_DIR / f"lifecycle_{p_tag}_full_s{s}"
            if d.exists():
                shutil.rmtree(d)

    rows_csv = []
    npz_payload = {}
    periods = [("P1_30pct", args.p1_frac), ("P2_100pct", args.p2_frac)]

    for k in args.centers:
        print(f"\n=== Shadow Center c{k} (agg={args.agg}, seeds={args.seeds}, split_seed={args.split_seed}) ===")
        print("| Period | Metric | bal.acc | " + " | ".join(CLASS_NAMES) + " |")
        print("|---|---|---|" + "---|" * len(CLASS_NAMES))
        for pname, frac in periods:
            res = run_period_comparison(k, frac, args.seeds, args.rounds, args.agg, args.split_seed, args.keep_demo_seed)
            for mkey, mlabel in (("preview_gain", "Re-admit Preview (0.2s)"), ("true_gain", "Exact Retrain Gain")):
                G = res[mkey]
                bal = np.nanmean(G, axis=1)
                vals = [fmt(bal.mean(), bal.std())] + [fmt(m, s) for m, s in zip(np.nanmean(G, 0), np.nanstd(G, 0))]
                print(f"| {pname} | {mlabel} | " + " | ".join(vals) + " |")
                for c, cname in enumerate(CLASS_NAMES):
                    rows_csv.append([k, pname, frac, mkey, cname, float(np.nanmean(G[:, c])), float(np.nanstd(G[:, c]))])
            # Pearson correlation between 20-seed mean preview gain and 20-seed mean true gain across 8 classes
            pm, tm = np.nanmean(res["preview_gain"], 0), np.nanmean(res["true_gain"], 0)
            r = float(np.corrcoef(pm, tm)[0, 1]) if np.std(pm) > 0 and np.std(tm) > 0 else float("nan")
            print(f"  -> {pname} c{k}: 8-class mean preview vs exact retrain Pearson r = {r:.3f}")
            npz_payload[f"c{k}_{pname}_preview_gain"] = res["preview_gain"]
            npz_payload[f"c{k}_{pname}_true_gain"] = res["true_gain"]
            npz_payload[f"c{k}_{pname}_shadow_recall"] = res["shadow_recall"]
            npz_payload[f"c{k}_{pname}_full_recall"] = res["full_recall"]

    with open(out / "lifecycle_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["center", "period", "data_fraction", "metric", "class", "gain_mean", "gain_std"])
        w.writerows(rows_csv)
    np.savez(out / "lifecycle_raw.npz", **npz_payload)
    print(f"\nWrote {out / 'lifecycle_summary.csv'} and {out / 'lifecycle_raw.npz'}")


if __name__ == "__main__":
    main()
