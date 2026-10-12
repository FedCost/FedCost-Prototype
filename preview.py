"""Exclusion preview module (preview.py).
Re-aggregates updates without a target client and predicts class-level recall impact
at decision time without retraining, conforming to contract.py.
"""

from pathlib import Path
import numpy as np
from contract import (
    N_CLIENTS, N_CLASSES, FEAT_DIM, CLASS_NAMES,
    load_update, load_features, FEATURES_DIR, RUNS_DIR
)

def evaluate_linear_model(W, X, y):
    """
    Evaluate a linear head W on feature matrix X and labels y.
    W shape: (FEAT_DIM + 1, N_CLASSES) where last row is bias.
    """
    weights = W[:-1, :]  # (1280, 8)
    bias = W[-1, :]      # (8,)
    logits = X @ weights + bias  # (N, 8)
    preds = np.argmax(logits, axis=1)

    recalls = np.zeros(N_CLASSES, dtype=float)
    for c in range(N_CLASSES):
        mask = (y == c)
        total_c = np.sum(mask)
        if total_c > 0:
            recalls[c] = float(np.sum((preds == c) & mask) / total_c)
        else:
            recalls[c] = 0.0

    return recalls, float(np.mean(recalls))

def aggregate_updates(updates, sizes=None, rule="fedavg"):
    """
    Aggregate a list of weight matrices W_k.
    updates: list of (1281, 8) numpy arrays.
    sizes: optional list of dataset sample counts.
    rule: 'fedavg' | 'trimmed' | 'krum'
    """
    if not updates:
        raise ValueError("Updates list cannot be empty")
    
    n = len(updates)
    if n == 1:
        return updates[0]
        
    stack = np.stack(updates, axis=0)  # (n, 1281, 8)

    if rule == "fedavg":
        if sizes is not None and sum(sizes) > 0:
            if len(sizes) != n:
                sizes = list(sizes)[:n] if len(sizes) > n else list(sizes) + [1] * (n - len(sizes))
            weights = np.array(sizes, dtype=float) / sum(sizes)
            # Weighted average across axis 0
            return np.tensordot(weights, stack, axes=(0, 0))
        return np.mean(stack, axis=0)

    elif rule == "trimmed":
        # Coordinate-wise trimmed mean: drop min and max if n >= 3
        if n >= 3:
            # Sort along client axis
            sorted_stack = np.sort(stack, axis=0)
            # Trim top 1 and bottom 1
            return np.mean(sorted_stack[1:-1, :, :], axis=0)
        return np.mean(stack, axis=0)

    elif rule == "krum":
        # Select single client minimizing sum of squared distances to closest (n-2) peers
        flat = stack.reshape(n, -1)  # (n, 1281*8)
        # Compute pairwise distance matrix
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = np.sum((flat[i] - flat[j]) ** 2)
                dists[i, j] = dists[j, i] = d
        
        m = max(1, n - 2)
        scores = []
        for i in range(n):
            sorted_d = np.sort(dists[i])
            scores.append(np.sum(sorted_d[1:m+1]))
        
        best_idx = int(np.argmin(scores))
        return updates[best_idx]

    else:
        return np.mean(stack, axis=0)

# Real Fed-ISIC2019 center training dataset sample counts (Center 0 to 5)
DEFAULT_CLIENT_SIZES = [9930, 3163, 2691, 1807, 655, 351]

def preview_exclusion(run_id, round_t, excluded_clients, rule="fedavg", client_sizes=None):
    """
    Calculates the counterfactual preview of excluding one or more clients from the federation.
    excluded_clients: int or list/set of ints
    
    Returns:
      dict with:
        - full_recalls: per-class recall with all clients
        - preview_recalls: predicted per-class recall without excluded_clients
        - delta_recalls: preview_recalls - full_recalls (negative means loss)
        - class_costs: full_recalls - preview_recalls (positive means positive cost/loss)
        - explanation: neutral factual summary of balanced accuracy and class changes
        - worst_class, worst_delta: class with the greatest drop
        - best_class, best_delta: class with the greatest gain
    """
    if isinstance(excluded_clients, (int, np.integer)):
        excluded_list = [int(excluded_clients)]
    else:
        excluded_list = [int(k) for k in excluded_clients]

    # Check round telemetry to discover client_sizes if not explicitly provided
    round_file = RUNS_DIR / run_id / f"round_{round_t}.json"
    round_meta = {}
    round_json_data = None
    if round_file.exists():
        try:
            import json
            round_json_data = json.loads(round_file.read_text(encoding="utf-8"))
            round_meta = round_json_data.get("meta", {})
        except Exception:
            pass

    if client_sizes is None:
        client_sizes = round_meta.get("client_sizes")

    all_sizes = list(client_sizes) if client_sizes is not None else list(DEFAULT_CLIENT_SIZES)

    # 1. Load all client updates
    all_updates = []
    active_sizes = []
    included_updates = []
    included_sizes = []

    for k in range(N_CLIENTS):
        try:
            w_k = load_update(run_id, round_t, k)
            all_updates.append(w_k)
            active_sizes.append(all_sizes[k])
            if k not in excluded_list:
                included_updates.append(w_k)
                included_sizes.append(all_sizes[k])
        except FileNotFoundError:
            # Update not found for this client (e.g. excluded in training)
            pass

    if len(all_updates) == 0:
        raise FileNotFoundError(f"No updates found for run '{run_id}', round {round_t}")

    # 2. Re-aggregate full model and counterfactual (preview) model
    W_full = aggregate_updates(all_updates, sizes=active_sizes, rule=rule)
    if included_updates:
        W_preview = aggregate_updates(included_updates, sizes=included_sizes, rule=rule)
    else:
        # If all clients excluded, return zero weights
        W_preview = np.zeros_like(W_full)

    # 3. Evaluate models on validation features if available
    has_features = False
    all_X = []
    all_y = []
    for k in range(N_CLIENTS):
        feat_path = FEATURES_DIR / f"center_{k}_test.npz"
        if feat_path.exists():
            X_k, y_k = load_features(k, "test")
            all_X.append(X_k)
            all_y.append(y_k)

    if len(all_X) > 0:
        has_features = True
        X_eval = np.concatenate(all_X, axis=0)
        y_eval = np.concatenate(all_y, axis=0)
        full_recalls, full_bal_acc = evaluate_linear_model(W_full, X_eval, y_eval)
        prev_recalls, prev_bal_acc = evaluate_linear_model(W_preview, X_eval, y_eval)
    else:
        # Fallback: estimate impact analytically from saved round telemetry
        if round_json_data is not None:
            data = round_json_data
            full_recalls = np.array(data.get("global_class_recall", [0.5]*N_CLASSES))
            
            # Aggregate client penalties for all excluded clients
            total_penalty = np.zeros(N_CLASSES, dtype=float)
            for ex_k in excluded_list:
                client_data = data.get("clients", {}).get(str(ex_k), {})
                c_recalls = np.array(client_data.get("class_recall", full_recalls))
                # Client carrying higher recall than global loses signal when dropped
                total_penalty += np.maximum(0, (c_recalls - full_recalls)) * 0.35
            
            # Additional penalty if a large fraction of the federation is excluded
            if len(excluded_list) > 1:
                frac_excluded = len(excluded_list) / N_CLIENTS
                total_penalty += frac_excluded * 0.15

            prev_recalls = np.clip(full_recalls - total_penalty, 0.01, 1.0)
            full_bal_acc = float(np.mean(full_recalls))
            prev_bal_acc = float(np.mean(prev_recalls))
        else:
            full_recalls = np.full(N_CLASSES, 0.5)
            prev_recalls = np.full(N_CLASSES, 0.5)
            full_bal_acc = 0.5
            prev_bal_acc = 0.5

    delta_recalls = prev_recalls - full_recalls
    class_costs = -delta_recalls  # Positive cost = performance dropped when excluded
    bal_acc_delta = prev_bal_acc - full_bal_acc

    # 4. Neutral, objective factual summary (no diagnostic labels)
    worst_idx = int(np.argmin(delta_recalls))
    worst_delta = float(delta_recalls[worst_idx])
    worst_cls = CLASS_NAMES[worst_idx]

    best_idx = int(np.argmax(delta_recalls))
    best_delta = float(delta_recalls[best_idx])
    best_cls = CLASS_NAMES[best_idx]

    bal_pct = f"{bal_acc_delta:+.1%}"
    if worst_delta < -0.005:
        explanation = f"Dışlanırsa dengeli doğruluk {bal_pct}, en büyük kayıp {worst_cls} {worst_delta:+.1%}"
    else:
        explanation = f"Dışlanırsa dengeli doğruluk {bal_pct}, belirgin bir sınıf kaybı yok"

    if best_delta > 0.01 and best_idx != worst_idx:
        explanation += f", en yüksek kazanç {best_cls} {best_delta:+.1%}"
    explanation += "."

    return {
        "run_id": run_id,
        "round": round_t,
        "excluded_clients": excluded_list,
        "rule": rule,
        "client_sizes": all_sizes,
        "has_real_features": has_features,
        "full_recalls": full_recalls.tolist(),
        "preview_recalls": prev_recalls.tolist(),
        "delta_recalls": delta_recalls.tolist(),
        "class_costs": class_costs.tolist(),
        "full_balanced_acc": full_bal_acc,
        "preview_balanced_acc": prev_bal_acc,
        "bal_acc_delta": bal_acc_delta,
        "explanation": explanation,
        "worst_class": worst_cls,
        "worst_delta": worst_delta,
        "best_class": best_cls,
        "best_delta": best_delta,
        "class_names": CLASS_NAMES,
        "low_confidence_classes": ["DF", "SCC", "AK"]
    }
