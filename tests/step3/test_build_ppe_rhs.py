# tests/step3/test_build_ppe_rhs.py

import numpy as np
from src.step3.build_ppe_rhs import build_ppe_rhs
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

def test_zero_divergence():
    s2 = Step2SchemaDummyState(nx=3, ny=3, nz=3)
    state = adapt_step2_to_step3(s2)

    rhs = build_ppe_rhs(state, state["U"], state["V"], state["W"])
    assert np.allclose(rhs, 0.0)


def test_uniform_divergence():
    s2 = Step2SchemaDummyState(nx=3, ny=3, nz=3)
    state = adapt_step2_to_step3(s2)

    pattern = np.ones_like(state["P"])
    state["_divergence_pattern"] = pattern

    rhs = build_ppe_rhs(state, state["U"], state["V"], state["W"])

    rho = state["Constants"]["rho"]
    dt = state["Constants"]["dt"]

    assert np.allclose(rhs, rho / dt)


def test_solid_zeroing():
    s2 = Step2SchemaDummyState(nx=3, ny=3, nz=3)
    state = adapt_step2_to_step3(s2)

    state["_divergence_pattern"] = np.ones_like(state["P"])
    state["Mask"][1, 1, 1] = 0

    rhs = build_ppe_rhs(state, state["U"], state["V"], state["W"])

    assert rhs[1, 1, 1] == 0.0


def test_minimal_grid():
    """
    This test intentionally bypasses Step‑2 schema validation.
    It only checks that the function does not crash on a 1×1×1 grid.
    """
    state = {
        "Mask": np.ones((1,1,1), int),
        "is_fluid": np.ones((1,1,1), bool),
        "P": np.zeros((1,1,1)),
        "Constants": {"rho": 1, "dt": 0.1},
        "Operators": {"divergence": lambda U, V, W, s: np.zeros((1,1,1))},
    }

    rhs = build_ppe_rhs(
        state,
        np.zeros((2,1,1)),
        np.zeros((1,2,1)),
        np.zeros((1,1,2)),
    )

    assert rhs.shape == (1,1,1)
