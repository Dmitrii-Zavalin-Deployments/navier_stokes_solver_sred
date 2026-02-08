# tests/step3/test_correct_velocity.py

import numpy as np
from src.step3.correct_velocity import correct_velocity
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

def test_zero_gradient():
    s2 = Step2SchemaDummyState(nx=3, ny=3, nz=3)
    state = adapt_step2_to_step3(s2)

    U_star = np.ones_like(state["U"])
    V_star = np.ones_like(state["V"])
    W_star = np.ones_like(state["W"])
    P_new = np.zeros_like(state["P"])

    U_new, V_new, W_new = correct_velocity(state, U_star, V_star, W_star, P_new)

    assert np.allclose(U_new, U_star)
    assert np.allclose(V_new, V_star)
    assert np.allclose(W_new, W_star)


def test_solid_mask():
    s2 = Step2SchemaDummyState(nx=3, ny=3, nz=3)
    state = adapt_step2_to_step3(s2)

    state["Mask"][1, 1, 1] = 0

    U_star = np.ones_like(state["U"])
    V_star = np.ones_like(state["V"])
    W_star = np.ones_like(state["W"])
    P_new = np.zeros_like(state["P"])

    U_new, _, _ = correct_velocity(state, U_star, V_star, W_star, P_new)

    assert np.any(U_new == 0.0)


def test_fluid_adjacent_faces():
    s2 = Step2SchemaDummyState(nx=3, ny=3, nz=3)
    state = adapt_step2_to_step3(s2)

    state["is_fluid"][1, 1, 1] = False

    U_star = np.ones_like(state["U"])
    P_new = np.zeros_like(state["P"])

    U_new, _, _ = correct_velocity(state, U_star, state["V"], state["W"], P_new)

    assert np.any(U_new == 0.0)


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
        "Constants": {"rho": 1, "dt": 0.1},
        "Operators": {
            "gradient_p_x": lambda P, s: np.zeros((2,1,1)),
            "gradient_p_y": lambda P, s: np.zeros((1,2,1)),
            "gradient_p_z": lambda P, s: np.zeros((1,1,2)),
        },
    }

    correct_velocity(state, state["U"], state["V"], state["W"], state["P"])
