#!/usr/bin/env bash
# Usage: scripts/flwr_run.sh <fedavg|trimmed|krum> [rounds=15] [attack=-1] [seed=0]
# flwr run starts the run and returns immediately; this script waits until the last round log is written.
set -euo pipefail
cd "$(dirname "$0")/.."
AGG=${1:?agg is required}; ROUNDS=${2:-15}; ATTACK=${3:--1}; SEED=${4:-0}
ATK=$([ "$ATTACK" -lt 0 ] && echo None || echo "$ATTACK")
RUN_ID="flwr_${AGG}_atk${ATK}_s${SEED}"
rm -rf "runs/$RUN_ID"
export PATH="$PWD/.venv/bin:$PATH"
OUT=$(flwr run . --run-config "agg='$AGG' rounds=$ROUNDS attack=$ATTACK seed=$SEED project-dir='$PWD'" 2>&1)
FLWR_ID=$(echo "$OUT" | sed -n 's/.*Successfully started run \([0-9]*\).*/\1/p')
[ -n "$FLWR_ID" ] || { echo "$OUT"; exit 1; }
for _ in $(seq 1 300); do
  [ -f "runs/$RUN_ID/round_$((ROUNDS-1)).json" ] && break
  if flwr log "$FLWR_ID" --show 2>&1 | grep -q "Exit Code"; then
    flwr log "$FLWR_ID" --show 2>&1 | tail -30; exit 1
  fi
  sleep 2
done
python3 - "$RUN_ID" "$ROUNDS" <<'PY'
import json, sys, numpy as np
rid, n = sys.argv[1], int(sys.argv[2])
gr = json.load(open(f"runs/{rid}/round_{n-1}.json"))["global_class_recall"]
print(f"{rid}: balanced acc={np.nanmean(gr):.3f} | class recall={np.round(gr, 2).tolist()}")
PY
