# tests/step2/test_orchestrate_step2.py

import numpy as np
import pytest

from src.step2.orchestrate_step2 import orchestrate_step2
from tests.helpers.solver_step1_output_dummy import make_step1_dummy_state


def make_minimal_state(nx=2, ny=2, nz=2, dx=1.0, dt=0.1, rho=1.0):
    """
    Create a canonical Step‑1 dummy state and override only the fields
    relevant for the Step 2 orchestrator tests.
    """
    # Create Step‑1 dummy
    state = make_step1_dummy_state(nx=nx, ny=ny, nz=nz, dx=dx, dt=dt, rho=rho)

    # Override grid spacing (dummy uses dx for all unless overridden)
    state.grid.dx = dx
    state.grid.dy = dx
    state.grid.dz = dx

    # Override dt and rho
    state.config.dt = dt
    state.constants["rho"] = rho

    # Ensure mask is all fluid
    state.mask = np.ones((nx, ny, nz), dtype=int)

    # Override fields (dummy already has correct shapes)
    state.fields["P"] = np.zeros((nx, ny, nz))
    state.fields["U"] = np.zeros((nx + 1, ny, nz))
    state.fields["V"] = np.zeros((nx, ny + 1, nz))
    state.fields["W"] = np.zeros((nx, ny, nz + 1))

    # Boundary conditions (empty but valid)
    state.boundary_conditions = {}

    # Health block (Step‑1 sets empty dict, but we ensure it)
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
