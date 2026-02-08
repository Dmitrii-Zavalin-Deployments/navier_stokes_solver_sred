# tests/step3/test_apply_boundary_conditions_pre.py

import numpy as np
from src.step3.apply_boundary_conditions_pre import apply_boundary_conditions_pre
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

def test_solid_zeroing():
    s2 = Step2SchemaDummyState(nx=3, ny=3, nz=3)
    state = adapt_step2_to_step3(s2)

    state["Mask"][1, 1, 1] = 0
    state["U"].fill(1.0)

    apply_boundary_conditions_pre(state)

    assert np.any(state["U"] == 0.0)


def test_bc_handler_invocation():
    s2 = Step2SchemaDummyState(nx=3, ny=3, nz=3)
    state = adapt_step2_to_step3(s2)

    class Handler:
        def __init__(self):
            self.called = False
        def apply_pre(self, state):
            self.called = True

    handler = Handler()
    state["BC_handler"] = handler

    apply_boundary_conditions_pre(state)
    assert handler.called


def test_pressure_shape_preserved():
    s2 = Step2SchemaDummyState(nx=3, ny=3, nz=3)
    state = adapt_step2_to_step3(s2)

    P_before = state["P"].copy()
    apply_boundary_conditions_pre(state)

    assert state["P"].shape == P_before.shape


def test_no_bc_handler():
    s2 = Step2SchemaDummyState(nx=3, ny=3, nz=3)
    state = adapt_step2_to_step3(s2)

    apply_boundary_conditions_pre(state)  # should not crash


def test_minimal_grid():
    """
    This test intentionally bypasses Step‑2 schema validation.
    It only checks that the function does not crash on a 1×1×1 grid.
    """
    state = {
        "Mask": np.ones((1,1,1), int),
        "U": np.zeros((2,1,1)),
        "V": np.zeros((1,2,1)),
        "W": np.zeros((1,1,2)),
        "P": np.zeros((1,1,1)),
        "is_fluid": np.ones((1,1,1), bool),
        "is_boundary_cell": np.zeros((1,1,1), bool),
        "BCs": [],
    }

    apply_boundary_conditions_pre(state)
