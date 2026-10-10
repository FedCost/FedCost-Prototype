"""Audit log manager for the Federation Health Console.
Records operator decisions, reasons, and class-level costs for medical AI compliance.
"""

import json
from pathlib import Path
from datetime import datetime
import pandas as pd

AUDIT_FILE = Path("runs/audit_log.json")

def load_audit_log():
    if not AUDIT_FILE.exists():
        return []
    try:
        return json.loads(AUDIT_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []

def log_decision(run_id, round_t, client_id, decision, operator_note, preview_summary=None):
    """
    Log an exclusion/retention decision.
    decision: 'EXCLUDED' | 'RETAINED' | 'QUARANTINED'
    """
    AUDIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    records = load_audit_log()
    
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run_id": run_id,
        "round": round_t,
        "client_id": int(client_id),
        "decision": decision,
        "operator_note": operator_note,
        "preview_summary": preview_summary or {}
    }
    
    records.append(entry)
    AUDIT_FILE.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    return entry

def get_audit_df():
    records = load_audit_log()
    if not records:
        return pd.DataFrame(columns=["timestamp", "run_id", "round", "client_id", "decision", "operator_note", "rare_class_cost"])
    
    rows = []
    for r in records:
        prev = r.get("preview_summary", {})
        rows.append({
            "Zaman": r["timestamp"],
            "Koşum ID": r["run_id"],
            "Tur": r["round"],
            "Merkez": f"Center {r['client_id']}",
            "Karar": r["decision"],
            "Gerekçe": r["operator_note"],
            "Nadir Sınıf Maliyeti": prev.get("rare_cost_str", "—")
        })
    return pd.DataFrame(rows)
