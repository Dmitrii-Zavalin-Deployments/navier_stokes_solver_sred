# tests/step3/test_step3_optional.py

import numpy as np
import pytest

from src.step3.orchestrate_step3 import orchestrate_step3_state
from src.solver_state import SolverState
from src.step3.update_health import update_health


def _make_minimal_state(nx: int = 3, ny: int = 3, nz: int = 3) -> SolverState:
    """
    Minimal valid SolverState for Step‑3 tests.
    """
    state = SolverState()

    state.config = {}
    state.grid = {"nx": nx, "ny": ny, "nz": nz}

    state.fields = {
        "P": np.zeros((nx, ny, nz)),
        "U": np.zeros((nx + 1, ny, nz)),
        "V": np.zeros((nx, ny + 1, nz)),
        "W": np.zeros((nx, ny, nz + 1)),
    }

    state.mask = np.ones((nx, ny, nz), dtype=int)
    state.constants = {"rho": 1.0}
    state.boundary_conditions = {}
    state.health = {}
    state.ppe = {"solver": lambda rhs: (np.zeros_like(rhs), {"converged": True})}
    state.operators = {}
    return state


def test_missing_mask_and_fields_raises_runtimeerror():
    """
    Step‑3 must fail cleanly when both mask and fields are missing.
    """
    bad_state = SolverState()
    bad_state.grid = {"nx": 3, "ny": 3, "nz": 3}
    bad_state.fields = None
    bad_state.mask = None

    with pytest.raises(RuntimeError):
        orchestrate_step3_state(bad_state, current_time=0.0, step_index=0)


def test_divergence_operator_raising_exception_is_handled():
    """
    update_health must not crash if divergence op raises.
    """
    # update_health still accepts dict-like inputs
    s2 = {
        "fields": {
            "P": np.zeros((3, 3, 3)),
            "U": np.zeros((4, 3, 3)),
            "V": np.zeros((3, 4, 3)),
            "W": np.zeros((3, 3, 4)),
        },
        "divergence": {"op": None},
    }

    def bad_div(U, V, W):
        raise RuntimeError("boom")

    s2["divergence"]["op"] = bad_div

    fields = s2["fields"]
    P = fields["P"]

    health = update_health(s2, fields, P)
    assert health["post_correction_divergence_norm"] == 0.0


def test_pressure_solver_returning_nans_is_handled():
    """
    Step‑3 must not crash if pressure solver returns NaNs.
    """
    state = _make_minimal_state()

    def fake_solver(rhs):
        P = np.full_like(rhs, np.nan)
        return P, {"converged": False}

    state.ppe["solver"] = fake_solver

    out = orchestrate_step3_state(state, current_time=0.0, step_index=0)
    assert np.isnan(out.fields["P"]).all()
