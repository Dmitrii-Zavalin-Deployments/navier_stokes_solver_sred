# tests/step4/test_step4_orchestrator_minimal_grid.py

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


def test_run_step4_minimal_grid_nx1_ny1_nz1():
    """
    Test 1.4 — Minimal Grid Orchestration (nx=1, ny=1, nz=1)

    Domain: nx = ny = nz = 1
    BCs: no-slip on all faces

    Assert:
      - run_step4 completes without index errors
      - Extended shapes and ghost layers are correct
      - BCs applied without out-of-bounds access
    """
    nx = ny = nz = 1

    bcs = [
        {
            "type": "no-slip",
            "faces": ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"],
        }
    ]

    state = _make_step3_state(nx, ny, nz, bcs)
    out = run_step4(state)

    # Extended field shapes must be correct
    assert np.asarray(out["P_ext"]).shape == (nx + 2, ny + 2, nz + 2)
    assert np.asarray(out["U_ext"]).shape == (nx + 3, ny + 2, nz + 2)
    assert np.asarray(out["V_ext"]).shape == (nx, ny + 3, nz + 2)
    assert np.asarray(out["W_ext"]).shape == (nx, ny, nz + 3)

    # No NaNs or infs introduced
    assert np.isfinite(out["P_ext"]).all()
    assert np.isfinite(out["U_ext"]).all()
    assert np.isfinite(out["V_ext"]).all()
    assert np.isfinite(out["W_ext"]).all()
