"""Multi-seed comparison: 3 aggregation rules x attacked/unattacked x N seeds.
Answers RQ1 on whatever is in features/ (synthetic now, real features later):
per-class recall as mean +- std over seeds, and the per-class cost of each rule against unattacked FedAvg.
Usage: python scripts/sweep.py --seeds 20 [--rounds 15] [--attack 0] [--keep-updates]"""
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
from contract import CLASS_NAMES, N_CLIENTS

RULES = ["fedavg", "trimmed", "krum"]


def one_run(agg, attack, seed, rounds, keep_updates):
    """Runs the engine once; returns final global class recall and per-round exclusion flags (rounds, clients)."""
    run_id = f"sweep_{agg}_atk{attack}_s{seed}"
    engine.run(agg, rounds, attack, seed, run_id=run_id)
    run_dir = contract.RUNS_DIR / run_id
    logs = [json.loads((run_dir / f"round_{t}.json").read_text()) for t in range(rounds)]
    if not keep_updates:                                  # weights are ~0.5 GB for a full sweep; logs are kept
        shutil.rmtree(run_dir / "updates", ignore_errors=True)
    excluded = np.array([[r["clients"][str(k)]["excluded"] for k in range(N_CLIENTS)] for r in logs])
    return np.array(logs[-1]["global_class_recall"], float), excluded


def fmt(mean, std):
    return f"{mean:.2f}±{std:.2f}"


def table(title, header, rows):
    print(f"\n{title}\n| " + " | ".join(header) + " |\n|" + "---|" * len(header))
    for r in rows:
        print("| " + " | ".join(r) + " |")


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--seeds", type=int, default=20)
    a.add_argument("--rounds", type=int, default=15)
    a.add_argument("--attack", type=int, default=0, help="center that flips labels in the attacked condition")
    a.add_argument("--keep-updates", action="store_true")
    a.add_argument("--out", default="results")
    args = a.parse_args()

    conds = [None, args.attack]
    recall = {}                                           # (agg, attack) -> (seeds, classes)
    excl = {}                                             # (agg, attack) -> (seeds, rounds, clients)
    for atk in conds:
        for agg in RULES:
            res = [one_run(agg, atk, s, args.rounds, args.keep_updates) for s in range(args.seeds)]
            recall[agg, atk] = np.stack([r[0] for r in res])
            excl[agg, atk] = np.stack([r[1] for r in res])

    base = recall["fedavg", None]                         # undefended, unattacked baseline; paired by seed
    out = Path(args.out); out.mkdir(exist_ok=True)
    rows_csv = []
    for atk in conds:
        name = "no attack" if atk is None else f"label flip at center {atk}"
        rec_rows, cost_rows = [], []
        for agg in RULES:
            R = recall[agg, atk]
            bal = np.nanmean(R, axis=1)
            C = base - R                                  # positive = recall lost against the baseline
            rec_rows.append([agg, fmt(bal.mean(), bal.std())] +
                            [fmt(m, s) for m, s in zip(np.nanmean(R, 0), np.nanstd(R, 0))])
            cost_rows.append([agg, fmt((np.nanmean(base, 1) - bal).mean(), (np.nanmean(base, 1) - bal).std())] +
                             [fmt(m, s) for m, s in zip(np.nanmean(C, 0), np.nanstd(C, 0))])
            for c, cname in enumerate(CLASS_NAMES):
                rows_csv.append([agg, "none" if atk is None else atk, cname,
                                 np.nanmean(R[:, c]), np.nanstd(R[:, c]), np.nanmean(C[:, c]), np.nanstd(C[:, c])])
        table(f"Class recall, {name} (mean±std over {args.seeds} seeds)", ["rule", "bal.acc"] + CLASS_NAMES, rec_rows)
        table(f"Cost vs unattacked FedAvg, {name} (positive = recall lost)", ["rule", "bal.acc"] + CLASS_NAMES, cost_rows)

    # Who gets excluded: fraction of (seed, round) pairs in which each center was flagged
    ex_rows = [[agg, "none" if atk is None else str(atk)] + [f"{v:.2f}" for v in excl[agg, atk].mean((0, 1))]
               for atk in conds for agg in ("trimmed", "krum")]
    table("Exclusion rate per center (fraction of seed x round)", ["rule", "attack"] + [f"c{k}" for k in range(N_CLIENTS)], ex_rows)
    # Krum keeps exactly one center per round; which one it keeps in the last round, per seed
    for atk in conds:
        kept = (~excl["krum", atk][:, -1]).argmax(1)
        print(f"\nKrum, attack={atk}: center kept in the last round, count over seeds = "
              f"{dict(zip(*map(lambda x: x.tolist(), np.unique(kept, return_counts=True))))}")

    with open(out / "sweep_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rule", "attack", "class", "recall_mean", "recall_std", "cost_mean", "cost_std"])
        w.writerows(rows_csv)
    np.savez(out / "sweep_raw.npz", **{f"{agg}_atk{atk}": recall[agg, atk] for agg, atk in recall})
    print(f"\nWrote {out / 'sweep_summary.csv'} and {out / 'sweep_raw.npz'}")


if __name__ == "__main__":
    main()
