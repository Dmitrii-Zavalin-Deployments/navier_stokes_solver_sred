# tests/step1/test_step1_orchestrator_state.py

from src.step1.orchestrate_step1 import orchestrate_step1_state
from src.solver_state import SolverState


def test_orchestrate_step1_state_minimal():
    """
    Minimal structural test for the new state-based Step 1 orchestrator.
    Ensures that Step 1 populates config, grid, fields, mask, constants, and BCs.
    """

    json_input = {
        "domain": {"nx": 2, "ny": 2, "nz": 2},
        "geometry_definition": {
            "geometry_mask_flat": [1] * 8,
            "geometry_mask_shape": [2, 2, 2],
            "flattening_order": "C",
        },
        "initial_conditions": {
            "initial_pressure": 0.0,
            "initial_velocity": [0.0, 0.0, 0.0],
        },
        "boundary_conditions": [],
        "fluid": {"density": 1.0, "viscosity": 0.001},
        "simulation": {"dt": 0.1},
    }

    state = orchestrate_step1_state(json_input)

    assert isinstance(state, SolverState)
    assert state.config is not None
    assert state.grid is not None
    assert state.fields is not None
    assert state.mask is not None
    assert state.constants is not None
    assert state.boundary_conditions is not None
    assert state.health == {}
