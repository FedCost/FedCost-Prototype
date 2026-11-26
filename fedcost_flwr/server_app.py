"""ServerApp: reads the run configuration and starts the custom strategy."""
import numpy as np
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp

from contract import FEAT_DIM, N_CLASSES
from engine import DEFAULT_DP_CLIP, DEFAULT_DP_DELTA
from .paths import apply_project_dir
from .strategy import FedCostStrategy

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    rc = context.run_config
    agg, rounds, seed = str(rc["agg"]), int(rc["rounds"]), int(rc["seed"])
    attack = int(rc["attack"])                         # -1: no attack
    dp_eps_raw = float(rc.get("dp-eps", -1.0))         # <= 0: no DP
    dp_clip = float(rc.get("dp-clip", DEFAULT_DP_CLIP))
    dp_delta = float(rc.get("dp-delta", DEFAULT_DP_DELTA))
    apply_project_dir(rc)
    atk = None if attack < 0 else attack
    dp_eps = dp_eps_raw if dp_eps_raw > 0 else None
    dp_tag = f"_dp{dp_eps:g}" if dp_eps is not None else ""
    run_id = f"flwr_{agg}_atk{atk}{dp_tag}_s{seed}"

    strategy = FedCostStrategy(agg, run_id, atk, seed, dp_eps=dp_eps, dp_clip=dp_clip, dp_delta=dp_delta)
    W0 = np.zeros((FEAT_DIM + 1, N_CLASSES))
    train_cfg = {"seed": seed, "attack": attack, "rounds": rounds}
    if dp_eps is not None:
        train_cfg.update({"dp-eps": dp_eps, "dp-clip": dp_clip, "dp-delta": dp_delta})
    strategy.start(grid=grid, initial_arrays=ArrayRecord.from_numpy_ndarrays([W0]),
                   num_rounds=rounds, train_config=ConfigRecord(train_cfg))
    gr = strategy.last_global_recall
    print(f"{run_id}: balanced acc={np.nanmean(gr):.3f} | class recall={np.round(gr, 2).tolist()}")
