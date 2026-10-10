"""Custom Strategy: calls the engine.AGG aggregation functions and writes contract-conformant logs/weights.
Flower's built-in FedTrimmedAvg/Krum strategies are not used because they give no exclusion information."""
from collections.abc import Iterable
from logging import INFO

import numpy as np
from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg
from flwr.supercore import log

from contract import N_CLASSES, N_CLIENTS, save_round, save_update
from engine import AGG


class FedCostStrategy(FedAvg):
    def __init__(self, agg: str, run_id: str, meta: dict, n_clients: int = N_CLIENTS):
        super().__init__(fraction_train=1.0, fraction_evaluate=1.0,
                         min_train_nodes=n_clients, min_evaluate_nodes=n_clients,
                         min_available_nodes=n_clients)
        self.agg, self.run_id, self.meta = agg, run_id, meta
        self._W = None                 # global model at the start of the round (for update_norm)
        self._pending = None           # (t, client telemetry): completed in the evaluation round
        self.last_global_recall = None

    def configure_train(self, server_round: int, arrays: ArrayRecord,
                        config: ConfigRecord, grid: Grid) -> Iterable[Message]:
        self._W = arrays.to_numpy_ndarrays()[0]
        return super().configure_train(server_round, arrays, config, grid)

    def aggregate_train(self, server_round, replies):
        ok = [m for m in replies if not m.has_error()]
        for m in replies:
            if m.has_error():
                log(INFO, "Node %d returned an error: %s", m.metadata.src_node_id, m.error.reason)
        if not ok:
            return None, None

        t = server_round - 1                               # same numbering as the engine: rounds start at 0
        ok.sort(key=lambda m: int(m.content["metrics"]["client-id"]))   # aggregation order follows center number
        cids = [int(m.content["metrics"]["client-id"]) for m in ok]
        ups = [m.content["arrays"].to_numpy_ndarrays()[0] for m in ok]
        sizes = [int(m.content["metrics"]["num-examples"]) for m in ok]

        for cid, u in zip(cids, ups):
            save_update(self.run_id, t, cid, u)
        new_W, info = AGG[self.agg](ups, sizes)            # info indices follow list order

        clients = {}
        for i, (cid, m) in enumerate(zip(cids, ok)):
            clients[cid] = {"class_recall": list(m.content["metrics"]["class-recall"]),
                            "update_norm": float(np.linalg.norm(ups[i] - self._W)),
                            "excluded": info[i][0], "reason": info[i][1]}
        self._pending = (t, clients)
        n_ex = sum(c["excluded"] for c in clients.values())
        log(INFO, "Round %d: aggregated with %s, number of excluded clients: %d", t, self.agg, n_ex)
        return ArrayRecord.from_numpy_ndarrays([new_W]), MetricRecord({"num-excluded": n_ex})

    def aggregate_evaluate(self, server_round, replies):
        ok = [m for m in replies if not m.has_error()]
        if not ok or self._pending is None:
            return None
        correct = np.sum([m.content["metrics"]["correct"] for m in ok], axis=0)
        count = np.sum([m.content["metrics"]["count"] for m in ok], axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            gr = np.where(count > 0, correct / count, np.nan)
        t, clients = self._pending
        save_round(self.run_id, t, clients, gr, self.meta)
        self.last_global_recall = gr
        self._pending = None
        return MetricRecord({"balanced-acc": float(np.nanmean(gr))})
