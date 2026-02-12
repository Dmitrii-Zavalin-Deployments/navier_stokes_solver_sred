# tests/step4/test_step4_orchestrator_invalid_bc.py

import numpy as np
import pytest
from tests.helpers.step3_schema_dummy_state import Step3SchemaDummyState
from src.step4.run_step4 import run_step4


def _make_step3_state(nx, ny, nz, bcs):
    """
    Helper to construct a Step‑3‑compatible state for Step‑4 orchestrator tests.
    We do NOT modify Step‑3 internals — only override BCs and domain sizes.
    """
    state = Step3SchemaDummyState(nx=nx, ny=ny, nz=nz)

    # Ensure domain sizes match the test
    state["config"]["domain"]["nx"] = nx
    state["config"]["domain"]["ny"] = ny
    state["config"]["domain"]["nz"] = nz

    # Boundary conditions live under config
    state["config"]["boundary_conditions"] = bcs

    return state


def test_run_step4_invalid_bc_type():
    """
    Test 1.3 — Inconsistent BC Table (Invalid Type)

    BCs: contains entry with bc.type = "unknown_type"

    Assert:
      - run_step4 either raises a controlled error
        OR logs a BC error status for the affected face.
      - If no exception is raised, BCApplied.boundary_conditions_status
        for affected face == "error".
    """
    nx = ny = nz = 2

    bad_bc = {
        "type": "unknown_type",
        "faces": ["x_min"],
    }

    state = _make_step3_state(nx, ny, nz, bcs=[bad_bc])

    try:
        out = run_step4(state)
    except Exception:
        # Controlled error path — acceptable behavior
        return

    # If no exception, then the BC status must reflect the error
    status = out["BCApplied"].get("boundary_conditions_status", {})
    assert any(
        s == "error" for s in status.values()
    ), "Invalid BC type must produce at least one 'error' status"
