"""Fake run generator for console development.
Produces realistic run logs and weight matrices strictly conforming to contract.py.
This allows Bülent to build and test the Streamlit console and exclusion preview
without waiting for the federation engine or real ISIC dataset.
"""

import numpy as np
from pathlib import Path
from contract import (
    N_CLIENTS, N_CLASSES, FEAT_DIM, CLASS_NAMES,
    save_round, save_update
)

def generate_mock_weights(seed=42):
    """Generate a realistic classifier head W of shape (1281, 8)."""
    rng = np.random.default_rng(seed)
    # 1280 features + 1 bias row
    return (rng.standard_normal((FEAT_DIM + 1, N_CLASSES)) * 0.05).astype(np.float32)

def generate_run(run_id, aggregation="trimmed", attack_client=None, num_rounds=15, seed=42, excluded_clients=None):
    """
    Generate a full synthetic run with rounds and weights.
    excluded_clients: list or set of client IDs to exclude entirely from training (LOO / custom coalition).
    
    Center characteristics:
      Center 0: Normal / Target for attack if attack_client == 0
      Center 1: Large clinic, dominated by NV / MEL
      Center 2: Maverick clinic, holds rare classes (DF, VASC)
      Center 3: Moderate general hospital
      Center 4: Device shift (different lighting/camera)
      Center 5: Small clinic
    """
    rng = np.random.default_rng(seed)
    excluded_set = set(excluded_clients or [])
    print(f"Generating run: {run_id} (agg: {aggregation}, attack: {attack_client}, excluded: {list(excluded_set)})...")

    active_set = set(range(N_CLIENTS)) - excluded_set
    n_active = len(active_set)
    volume_factor = 0.35 + 0.65 * (n_active / N_CLIENTS)

    # Base recall curves per class over rounds
    for t in range(num_rounds):
        progress = (t + 1) / num_rounds
        
        # Center-specific contributions to classes
        # Center 1 (Barcelona) is crucial for MEL and NV
        c0_contrib = (1.0 if 1 in active_set else 0.60) * (1.0 if 0 in active_set else 0.85)
        c1_contrib = 1.0 if 1 in active_set else 0.50
        # Center 3 & 0 are crucial for BCC & SCC
        c2_contrib = (1.0 if 3 in active_set else 0.70) * (1.0 if 0 in active_set else 0.85)
        c3_contrib = (1.0 if 0 in active_set else 0.75) * (1.0 if 4 in active_set else 0.85)
        c4_contrib = 1.0 if 3 in active_set else 0.80
        # Center 2 (Maverick) is crucial for DF & VASC
        c5_contrib = 1.0 if 2 in active_set else 0.18
        c6_contrib = 1.0 if 2 in active_set else 0.14
        c7_contrib = (1.0 if 3 in active_set else 0.75) * (1.0 if 5 in active_set else 0.85)

        base_recall = np.zeros(N_CLASSES, dtype=float)
        base_recall[0] = min(0.85, (0.30 + 0.55 * np.sqrt(progress)) * volume_factor * c0_contrib)  # MEL
        base_recall[1] = min(0.92, (0.40 + 0.52 * np.sqrt(progress)) * volume_factor * c1_contrib)  # NV
        base_recall[2] = min(0.78, (0.25 + 0.53 * progress) * volume_factor * c2_contrib)          # BCC
        base_recall[3] = min(0.65, (0.15 + 0.50 * progress) * volume_factor * c3_contrib)          # AK
        base_recall[4] = min(0.70, (0.20 + 0.50 * progress) * volume_factor * c4_contrib)          # BKL
        base_recall[5] = min(0.55, (0.08 + 0.47 * progress) * volume_factor * c5_contrib)          # DF (rare)
        base_recall[6] = min(0.50, (0.05 + 0.45 * progress) * volume_factor * c6_contrib)          # VASC (rare)
        base_recall[7] = min(0.62, (0.18 + 0.44 * progress) * volume_factor * c7_contrib)          # SCC

        # Adjust for aggregation method (when Center 2 is participating)
        if 2 in active_set:
            if aggregation == "trimmed":
                # Center 2's rare classes get trimmed in coordinate sorting
                base_recall[5] *= 0.65  # DF penalized
                base_recall[6] *= 0.60  # VASC penalized
            elif aggregation == "krum":
                # Krum often excludes Center 2 and Center 4 (outliers)
                base_recall[5] *= 0.40
                base_recall[6] *= 0.35
                base_recall[4] *= 0.80

        # If attack active and fedavg used without defense:
        is_attack_effective = (attack_client is not None) and (attack_client in active_set)
        if is_attack_effective and aggregation == "fedavg":
            base_recall = base_recall * 0.35  # Severe model poisoning

        # Client telemetry
        clients_data = {}
        for k in range(N_CLIENTS):
            # Client recall varies around base recall
            c_recall = np.clip(base_recall + rng.normal(0, 0.03, size=N_CLASSES), 0.0, 1.0)
            
            # Client-specific strengths
            if k == 1:
                # Barcelona has tons of NV and MEL
                c_recall[0] = min(0.95, c_recall[0] + 0.15)
                c_recall[1] = min(0.98, c_recall[1] + 0.12)
            elif k == 2:
                # Maverick clinic has ground truth signal on rare classes
                c_recall[5] = min(0.90, c_recall[5] + 0.35)
                c_recall[6] = min(0.85, c_recall[6] + 0.40)
            elif k == 3:
                c_recall[2] = min(0.88, c_recall[2] + 0.15)
                c_recall[7] = min(0.85, c_recall[7] + 0.18)

            # Attack client behavior
            if attack_client == k:
                # Flips rare labels to NV, recall on rare is 0
                c_recall[5] = 0.0
                c_recall[6] = 0.0
                c_norm = float(rng.uniform(1.8, 2.5))
            elif k == 2:
                c_norm = float(rng.uniform(1.2, 1.6))  # Distinct gradient norm
            elif k == 4:
                c_norm = float(rng.uniform(1.1, 1.4))  # Device shift norm
            else:
                c_norm = float(rng.uniform(0.7, 1.0))  # Standard norm

            # Exclusion / Trimmed status
            if k in excluded_set:
                excluded = True
                reason = "Manuel dışlama (Koalisyon deneyi / LOO)"
            elif aggregation == "krum":
                # Krum selects 1 client, excludes the remaining 5
                # Usually selects Center 1 or 3 (central)
                selected = 1 if (t % 2 == 0) else 3
                if k != selected:
                    excluded = True
                    reason = "Krum score too high (distance from cluster)"
            elif aggregation == "trimmed":
                # Trimmed mean: coordinate-wise trimming
                # Mark as excluded if trimmed coordinate fraction > threshold (e.g. Center 2 or attacker)
                if attack_client == k:
                    excluded = True
                    reason = "High coordinate trimming rate (>45% coordinates trimmed)"
                elif k == 2 and t >= 3:
                    # Honest outlier (Maverick) flagged!
                    excluded = (t % 3 != 0)
                    reason = "Trimmed in >30% coordinates (primarily DF/VASC)" if excluded else None
                elif k == 4 and t % 4 == 0:
                    excluded = True
                    reason = "Trimmed in feature coordinates (device shift)"
                else:
                    excluded = False
                    reason = None
            else:
                excluded = False
                reason = None

            clients_data[k] = {
                "class_recall": [round(float(x), 4) for x in c_recall],
                "update_norm": round(c_norm, 4),
                "excluded": excluded,
                "reason": reason
            }

            # Generate and save client update matrix W (1281, 8)
            # Center 2 has stronger gradients in rows 5 and 6 (DF, VASC)
            W_k = generate_mock_weights(seed=seed + t * 10 + k)
            if k == 2:
                W_k[:, 5] *= 2.5
                W_k[:, 6] *= 2.8
            elif attack_client == k:
                W_k[:, 1] += 0.5  # Pushing towards NV
            
            save_update(run_id, t, k, W_k)

        # Global class recall
        global_recall = [round(float(x), 4) for x in base_recall]
        meta = {
            "aggregation": aggregation,
            "attack_client": attack_client,
            "excluded_clients": list(excluded_set),
            "round": t,
            "balanced_accuracy": round(float(np.mean(global_recall)), 4)
        }
        
        save_round(run_id, t, clients_data, global_recall, meta=meta)

    print(f"Run {run_id} generated successfully ({num_rounds} rounds).")

def main():
    print("=== Generating Synthetic Runs for Console Development ===")
    generate_run("demo_fedavg_clean", aggregation="fedavg", attack_client=None, num_rounds=15, seed=101)
    generate_run("demo_trimmed_clean", aggregation="trimmed", attack_client=None, num_rounds=15, seed=102)
    generate_run("demo_krum_clean", aggregation="krum", attack_client=None, num_rounds=15, seed=103)
    generate_run("demo_trimmed_attack_c0", aggregation="trimmed", attack_client=0, num_rounds=15, seed=104)
    print("All mock runs created in 'runs/' directory!")

if __name__ == "__main__":
    main()
