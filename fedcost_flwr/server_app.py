"""ServerApp: reads the run configuration and starts the custom strategy."""
import numpy as np
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp

from contract import FEAT_DIM, N_CLASSES
from .paths import apply_project_dir
from .strategy import FedCostStrategy

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    rc = context.run_config
    agg, rounds, seed = str(rc["agg"]), int(rc["rounds"]), int(rc["seed"])
    attack = int(rc["attack"])                         # -1: no attack
    apply_project_dir(rc)
    atk = None if attack < 0 else attack
    run_id = f"flwr_{agg}_atk{atk}_s{seed}"

    strategy = FedCostStrategy(agg, run_id, {"agg": agg, "attack": atk, "seed": seed})
    W0 = np.zeros((FEAT_DIM + 1, N_CLASSES))
    strategy.start(grid=grid, initial_arrays=ArrayRecord.from_numpy_ndarrays([W0]),
                   num_rounds=rounds, train_config=ConfigRecord({"seed": seed, "attack": attack}))
    gr = strategy.last_global_recall
    print(f"{run_id}: balanced acc={np.nanmean(gr):.3f} | class recall={np.round(gr, 2).tolist()}")
