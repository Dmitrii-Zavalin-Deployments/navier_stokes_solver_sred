# tests/step3/test_predict_velocity.py

import numpy as np
from src.step3.predict_velocity import predict_velocity
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

def test_zero_ops_zero_forces():
    s2 = Step2SchemaDummyState(nx=3, ny=3, nz=3)
    state = adapt_step2_to_step3(s2)

    U_star, V_star, W_star = predict_velocity(state)

    assert np.allclose(U_star, state["U"])
    assert np.allclose(V_star, state["V"])
    assert np.allclose(W_star, state["W"])


def test_constant_force():
    s2 = Step2SchemaDummyState(nx=3, ny=3, nz=3)
    state = adapt_step2_to_step3(s2)

    state["Config"]["external_forces"] = {"fx": 1.0}

    U_star, V_star, W_star = predict_velocity(state)

    assert np.any(U_star != 0.0)
    assert np.allclose(V_star, 0.0)
    assert np.allclose(W_star, 0.0)


def test_solid_mask_respected():
    s2 = Step2SchemaDummyState(nx=3, ny=3, nz=3)
    state = adapt_step2_to_step3(s2)

    state["Mask"][1, 1, 1] = 0
    state["Config"]["external_forces"] = {"fx": 1.0}

    U_star, _, _ = predict_velocity(state)

    assert np.any(U_star == 0.0)


def test_temp_buffers():
    """
    predict_velocity must not mutate the input state arrays.
    """
    s2 = Step2SchemaDummyState(nx=3, ny=3, nz=3)
    state = adapt_step2_to_step3(s2)

    U_before = state["U"].copy()
    predict_velocity(state)

    assert np.allclose(state["U"], U_before)


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
        "Config": {"external_forces": {}},
        "Constants": {"rho": 1, "mu": 0.1, "dt": 0.01},
        "Operators": {
            "advection_u": lambda U, V, W, s: np.zeros_like(U),
            "advection_v": lambda U, V, W, s: np.zeros_like(V),
            "advection_w": lambda U, V, W, s: np.zeros_like(W),
            "laplacian_u": lambda U, s: np.zeros_like(U),
            "laplacian_v": lambda V, s: np.zeros_like(V),
            "laplacian_w": lambda W, s: np.zeros_like(W),
        },
    }

    predict_velocity(state)
