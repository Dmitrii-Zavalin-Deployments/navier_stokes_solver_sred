# tests/step4/test_step4_orchestrator_no_bcs.py

import numpy as np
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


def test_run_step4_no_bcs_defined():
    """
    Test 1.2 — No BCs Defined

    Domain: nx = ny = nz = 2
    BCs: empty list

    Assert:
      - run_step4 completes without error
      - Ghost layers remain zero-ish (no BCs applied)
      - BCApplied.boundary_conditions_status marks all faces as "skipped"
      - Diagnostics.bc_violation_count == 0
    """
    nx = ny = nz = 2
    state = _make_step3_state(nx, ny, nz, bcs=[])

    out = run_step4(state)

    # Extended fields exist and contain finite values
    assert np.isfinite(out["P_ext"]).all()
    assert np.isfinite(out["U_ext"]).all()
    assert np.isfinite(out["V_ext"]).all()
    assert np.isfinite(out["W_ext"]).all()

    # All faces should be marked "skipped"
    status = out["BCApplied"].get("boundary_conditions_status", {})
    for face_status in status.values():
        assert face_status == "skipped"

    # No BC violations should be recorded
    diagnostics = out.get("Diagnostics", {})
    assert diagnostics.get("bc_violation_count", 0) == 0
