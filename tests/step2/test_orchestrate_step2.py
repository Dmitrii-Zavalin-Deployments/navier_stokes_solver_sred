import numpy as np
import pytest

from src.solver_state import SolverState
from src.step2.orchestrate_step2 import orchestrate_step2


def make_minimal_state(nx=2, ny=2, nz=2, dx=1.0, dt=0.1, rho=1.0):
    """
    Construct a minimal valid SolverState for orchestrator tests.
    This mimics the output of Step 1.
    """
    state = SolverState()

    # Grid
    state.grid = type(
        "Grid", (), {"nx": nx, "ny": ny, "nz": nz, "dx": dx, "dy": dx, "dz": dx}
    )()

    # Config
    state.config = type("Config", (), {"dt": dt})()

    # Constants (Step 1 provides rho; Step 2 fills inv_dx etc.)
    state.constants = {"rho": rho}

    # Mask (all fluid)
    state.mask = np.ones((nx, ny, nz), dtype=int)

    # Fields (staggered)
    state.fields = {
        "P": np.zeros((nx, ny, nz)),
        "U": np.zeros((nx + 1, ny, nz)),
        "V": np.zeros((nx, ny + 1, nz)),
        "W": np.zeros((nx, ny, nz + 1)),
    }

    # Boundary conditions (empty but valid)
    state.boundary_conditions = {}

    # Health block
    state.health = {}

    return state


def test_orchestrate_step2_minimal_state():
    """
    Minimal structural test for the final Step 2 orchestrator.
    Ensures that Step 2 populates:
      - mask semantics
      - constants
      - operators
      - PPE structure
      - health diagnostics
    """

    state = make_minimal_state()

    # Run Step 2
    orchestrate_step2(state)

    # ------------------------------------------------------------
    # Mask semantics
    # ------------------------------------------------------------
    assert state.is_fluid is not None
    assert state.is_boundary_cell is not None
    assert state.is_fluid.shape == state.mask.shape

    # ------------------------------------------------------------
    # Constants
    # ------------------------------------------------------------
    assert "inv_dx" in state.constants
    assert "inv_dy" in state.constants
    assert "inv_dz" in state.constants
    assert state.constants["inv_dx"] == pytest.approx(1.0)

    # ------------------------------------------------------------
    # Operators
    # ------------------------------------------------------------
    assert state.operators is not None
    assert "divergence" in state.operators
    assert callable(state.operators["divergence"])

    assert "grad_x" in state.operators
    assert "lap_u" in state.operators

    # ------------------------------------------------------------
    # PPE structure
    # ------------------------------------------------------------
    assert state.ppe is not None
    assert isinstance(state.ppe, dict)
    assert "rhs_builder" in state.ppe
    assert callable(state.ppe["rhs_builder"])
    assert "ppe_is_singular" in state.ppe

    # ------------------------------------------------------------
    # Health diagnostics
    # ------------------------------------------------------------
    assert state.health is not None
    assert isinstance(state.health, dict)
    assert "divergence_norm" in state.health
    assert "max_velocity" in state.health
    assert "cfl" in state.health

    # Ensure values are finite
    assert np.isfinite(state.health["divergence_norm"])
    assert np.isfinite(state.health["max_velocity"])
    assert np.isfinite(state.health["cfl"])
