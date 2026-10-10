"""Unit tests for contract conformance and preview module."""

import numpy as np
import pytest
from contract import (
    N_CLIENTS, N_CLASSES, FEAT_DIM, CLASS_NAMES,
    save_features, load_features, save_round, save_update, load_update,
    RUNS_DIR, FEATURES_DIR
)
import preview
import audit_log

def test_contract_constants():
    assert N_CLIENTS == 6
    assert N_CLASSES == 8
    assert FEAT_DIM == 1280
    assert len(CLASS_NAMES) == 8

def test_contract_round_and_update_io(tmp_path):
    run_id = "test_run"
    t = 0
    k = 1
    W = np.ones((FEAT_DIM + 1, N_CLASSES), dtype=np.float32)
    save_update(run_id, t, k, W)
    loaded_W = load_update(run_id, t, k)
    assert loaded_W.shape == (1281, 8)
    assert np.allclose(loaded_W, W)

def test_preview_module():
    res = preview.preview_exclusion("demo_trimmed_clean", 14, 2, "trimmed")
    assert "class_costs" in res
    assert len(res["class_costs"]) == 8
    assert res["diagnosis"] in ["VALUABLE_MAVERICK", "HARMFUL_ATTACKER", "STANDARD_PEER"]

def test_audit_log():
    entry = audit_log.log_decision("test_run", 0, 2, "RETAINED", "Test note", {"rare_cost_str": "DF: +10%"})
    assert entry["decision"] == "RETAINED"
    df = audit_log.get_audit_df()
    assert not df.empty
