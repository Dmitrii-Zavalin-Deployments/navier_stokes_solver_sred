# tests/step2/test_step2_orchestrator_state.py

from src.solver_state import SolverState
from src.step2.orchestrate_step2 import orchestrate_step2_state


def test_orchestrate_step2_state_minimal():
    """
    Minimal structural test for the new state-based Step 2 orchestrator.
    Ensures that Step 2 populates operators, PPE structure, masks, and health.
    """

    dummy = {
        "config": {"domain": {"nx": 2, "ny": 2, "nz": 2}, "dt": 0.1},
        "grid": {"dx": 1.0, "dy": 1.0, "dz": 1.0},
        "fields": {"P": None, "U": None, "V": None, "W": None},
        "mask_3d": None,
        "constants": {"rho": 1.0},
        "boundary_conditions": {},
        "health": {},
    }

    state = SolverState()
    state.config = dummy["config"]
    state.grid = dummy["grid"]
    state.fields = dummy["fields"]
    state.mask = dummy["mask_3d"]
    state.constants = dummy["constants"]
    state.boundary_conditions = dummy["boundary_conditions"]
    state.health = dummy["health"]

    state = orchestrate_step2_state(state)

    assert state.operators is not None
    assert "divergence" in state.operators

    assert state.ppe is not None
    assert isinstance(state.ppe, dict)

    assert state.is_fluid is not None
    assert state.is_boundary_cell is not None

    assert state.health is not None
    assert isinstance(state.health, dict)
