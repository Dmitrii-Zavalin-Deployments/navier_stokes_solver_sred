# tests/step4/test_step4_orchestrator_happy_path.py

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


def test_run_step4_happy_path_minimal_domain():
    """
    Test 1.1 — Happy Path (Minimal Domain)

    Domain: nx = ny = nz = 2
    BCs: simple no-slip on all walls, no external forces

    Assert:
      - run_step4 completes without error
      - state.P_ext, U_ext, V_ext, W_ext exist with correct shapes
      - state.Domain, RHS_Source, BCApplied, Diagnostics exist
      - state.Initialized == True
      - state.ReadyForTimeLoop == True
    """
    nx = ny = nz = 2

    bcs = [
        {
            "type": "no-slip",
            "faces": ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"],
        }
    ]

    state = _make_step3_state(nx, ny, nz, bcs)
    out = run_step4(state)

    # Extended fields must exist
    assert "P_ext" in out
    assert "U_ext" in out
    assert "V_ext" in out
    assert "W_ext" in out

    # Shapes must match the extended-grid specification
    assert np.asarray(out["P_ext"]).shape == (nx + 2, ny + 2, nz + 2)
    assert np.asarray(out["U_ext"]).shape == (nx + 3, ny + 2, nz + 2)
    assert np.asarray(out["V_ext"]).shape == (nx, ny + 3, nz + 2)
    assert np.asarray(out["W_ext"]).shape == (nx, ny, nz + 3)

    # Domain structures and diagnostics must exist
    assert "Domain" in out
    assert "RHS_Source" in out
    assert "BCApplied" in out
    assert "Diagnostics" in out

    # Flags for readiness
    assert out.get("Initialized") is True
    assert out.get("ReadyForTimeLoop") is True
