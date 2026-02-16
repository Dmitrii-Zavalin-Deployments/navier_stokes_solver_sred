import numpy as np
import pytest

from src.solver_state import SolverState
from src.step2.prepare_ppe_structure import prepare_ppe_structure
from src.step2.build_divergence_operator import build_divergence_operator
from src.step2.create_fluid_mask import create_fluid_mask


def make_state(nx=4, ny=4, nz=4, dx=1.0, dt=0.1, rho=1.0):
    """
    Construct a minimal valid SolverState for PPE structure tests.
    """
    state = SolverState()

    # Grid
    state.grid = type(
        "Grid", (), {"nx": nx, "ny": ny, "nz": nz, "dx": dx, "dy": dx, "dz": dx}
    )()

    # Config
    state.config = type("Config", (), {"dt": dt})()

    # Constants
    state.constants = {"rho": rho}

    # Mask (all fluid)
    state.mask = np.ones((nx, ny, nz), dtype=int)
    state.is_fluid, state.is_boundary_cell = create_fluid_mask(state)

    # Fields
    state.fields = {
        "P": np.zeros((nx, ny, nz)),
        "U": np.zeros((nx + 1, ny, nz)),
        "V": np.zeros((nx, ny + 1, nz)),
        "W": np.zeros((nx, ny, nz + 1)),
    }

    # Boundary conditions (empty by default)
    state.boundary_conditions = {}

    return state


# ------------------------------------------------------------
# 1. Enclosed box → PPE is singular
# ------------------------------------------------------------
def test_ppe_singular_enclosed_box():
    state = make_state()

    prepare_ppe_structure(state)

    assert state.ppe["ppe_is_singular"] is True


# ------------------------------------------------------------
# 2. One pressure outlet → PPE is non-singular
# ------------------------------------------------------------
def test_ppe_non_singular_with_outlet():
    state = make_state()

    # Add a Dirichlet pressure outlet
    state.boundary_conditions = {
        "pressure_outlet": [{"location": "x_max", "value": 0.0}]
    }

    prepare_ppe_structure(state)

    assert state.ppe["ppe_is_singular"] is False


# ------------------------------------------------------------
# 3. Empty BC table → PPE is singular
# ------------------------------------------------------------
def test_ppe_singular_empty_bc_table():
    state = make_state()

    # Explicitly empty BCs
    state.boundary_conditions = {}

    prepare_ppe_structure(state)

    assert state.ppe["ppe_is_singular"] is True


# ------------------------------------------------------------
# 4. RHS builder: rhs = -rho/dt * divergence
# ------------------------------------------------------------
def test_ppe_rhs_builder_correct_formula():
    nx, ny, nz = 4, 4, 4
    state = make_state(nx, ny, nz, dx=1.0, dt=0.1, rho=2.0)

    # Simple divergence field
    divergence = np.ones((nx, ny, nz))

    prepare_ppe_structure(state)

    rhs_builder = state.ppe["rhs_builder"]
    rhs = rhs_builder(divergence)

    expected = -state.constants["rho"] / state.config.dt * divergence
    assert np.allclose(rhs, expected)


# ------------------------------------------------------------
# 5. Masked divergence → RHS zero in solid cells
# ------------------------------------------------------------
def test_ppe_rhs_zero_in_solid_cells():
    state = make_state()

    # Mark a solid cell
    state.is_fluid[1, 1, 1] = False

    # Divergence field
    divergence = np.ones_like(state.fields["P"])

    prepare_ppe_structure(state)
    rhs_builder = state.ppe["rhs_builder"]
    rhs = rhs_builder(divergence)

    assert rhs[1, 1, 1] == 0.0


# ------------------------------------------------------------
# 6. Units consistency: RHS has units of pressure Laplacian
# ------------------------------------------------------------
def test_ppe_rhs_units_consistency():
    state = make_state(dx=0.5, dt=0.2, rho=3.0)

    divergence = np.random.randn(*state.fields["P"].shape)

    prepare_ppe_structure(state)
    rhs_builder = state.ppe["rhs_builder"]
    rhs = rhs_builder(divergence)

    # Check finite values (no NaN/inf)
    assert np.all(np.isfinite(rhs))

    # Check scaling: RHS ∝ rho/dt
    scaled = rhs / (-state.constants["rho"] / state.config.dt)
    assert np.allclose(scaled, divergence, atol=1e-6)
