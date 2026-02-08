# tests/step3/test_update_health.py

import numpy as np
from src.step3.update_health import update_health
from tests.helpers.step2_schema_dummy_state import Step2SchemaDummyState


# ----------------------------------------------------------------------
# Helper: convert Step‑2 dummy → Step‑3 input shape
# ----------------------------------------------------------------------
def adapt_step2_to_step3(state):
    return {
        "Config": state["config"],
        "Mask": state["fields"]["Mask"],
        "is_fluid": state["fields"]["Mask"] == 1,
        "is_boundary_cell": np.zeros_like(state["fields"]["Mask"], bool),

        "P": state["fields"]["P"],
        "U": state["fields"]["U"],
        "V": state["fields"]["V"],
        "W": state["fields"]["W"],

        "BCs": state["boundary_table_list"],

        "Constants": {
            "rho": state["config"]["fluid"]["density"],
            "mu": state["config"]["fluid"]["viscosity"],
            "dt": state["config"]["simulation"]["dt"],
            "dx": state["grid"]["dx"],
            "dy": state["grid"]["dy"],
            "dz": state["grid"]["dz"],
        },

        "Operators": state["operators"],

        "PPE": {
            "solver": None,
            "tolerance": 1e-6,
            "max_iterations": 100,
            "ppe_is_singular": False,
        },

        "Health": {},
        "History": {},
    }


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------

def test_zero_velocity():
    s2 = Step2SchemaDummyState(nx=3, ny=3, nz=3)
    state = adapt_step2_to_step3(s2)

    update_health(state)

    assert state["Health"]["post_correction_divergence_norm"] == 0.0
    assert state["Health"]["max_velocity_magnitude"] == 0.0


def test_uniform_velocity():
    s2 = Step2SchemaDummyState(nx=3, ny=3, nz=3)
    state = adapt_step2_to_step3(s2)

    state["U"].fill(2.0)

    update_health(state)

    assert state["Health"]["max_velocity_magnitude"] == 2.0


def test_divergent_field():
    s2 = Step2SchemaDummyState(nx=3, ny=3, nz=3)
    state = adapt_step2_to_step3(s2)

    pattern = np.ones_like(state["P"])
    state["_divergence_pattern"] = pattern

    update_health(state)

    assert state["Health"]["post_correction_divergence_norm"] > 0.0


def test_minimal_grid():
    """
    This test intentionally bypasses Step‑2 schema validation.
    It only checks that the function does not crash on a 1×1×1 grid.
    """
    state = {
        "Mask": np.ones((1,1,1), int),
        "is_fluid": np.ones((1,1,1), bool),
        "U": np.zeros((2,1,1)),
        "V": np.zeros((1,2,1)),
        "W": np.zeros((1,1,2)),
        "P": np.zeros((1,1,1)),
        "Constants": {"dt": 0.1, "dx": 1, "dy": 1, "dz": 1},
        "Operators": {"divergence": lambda U, V, W, s: np.zeros((1,1,1))},
    }

    update_health(state)
