# tests/step3/test_apply_boundary_conditions_post.py

import numpy as np
from src.step3.apply_boundary_conditions_post import apply_boundary_conditions_post
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

def test_state_update():
    s2 = Step2SchemaDummyState(nx=3, ny=3, nz=3)
    state = adapt_step2_to_step3(s2)

    U_new = np.ones_like(state["U"])
    V_new = np.ones_like(state["V"])
    W_new = np.ones_like(state["W"])
    P_new = np.ones_like(state["P"])

    apply_boundary_conditions_post(state, U_new, V_new, W_new, P_new)

    assert np.allclose(state["U"], U_new)
    assert np.allclose(state["P"], P_new)


def test_bc_handler():
    s2 = Step2SchemaDummyState(nx=3, ny=3, nz=3)
    state = adapt_step2_to_step3(s2)

    class Handler:
        def __init__(self):
            self.called = False
        def apply_post(self, state):
            self.called = True

    handler = Handler()
    state["BC_handler"] = handler

    apply_boundary_conditions_post(state, state["U"], state["V"], state["W"], state["P"])
    assert handler.called


def test_minimal_grid():
    """
    This test checks only that the function does not crash on a 1×1×1 grid.
    It does NOT require Step‑2 schema validity.
    """
    state = {
        "Mask": np.ones((1,1,1), int),
        "U": np.zeros((2,1,1)),
        "V": np.zeros((1,2,1)),
        "W": np.zeros((1,1,2)),
        "P": np.zeros((1,1,1)),
    }

    apply_boundary_conditions_post(state, state["U"], state["V"], state["W"], state["P"])
