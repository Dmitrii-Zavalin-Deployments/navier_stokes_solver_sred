# tests/step3/test_solve_pressure.py

import numpy as np
from src.step3.solve_pressure import solve_pressure
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

def test_shape_consistency():
    s2 = Step2SchemaDummyState(nx=3, ny=3, nz=3)
    state = adapt_step2_to_step3(s2)

    rhs = np.random.randn(*state["P"].shape)
    P_new = solve_pressure(state, rhs)

    assert P_new.shape == rhs.shape


def test_singular_mean_subtraction():
    s2 = Step2SchemaDummyState(nx=3, ny=3, nz=3)
    state = adapt_step2_to_step3(s2)

    state["PPE"]["ppe_is_singular"] = True

    rhs = np.ones_like(state["P"])
    P_new = solve_pressure(state, rhs)

    assert abs(P_new[state["is_fluid"]].mean()) < 1e-12


def test_non_singular():
    s2 = Step2SchemaDummyState(nx=3, ny=3, nz=3)
    state = adapt_step2_to_step3(s2)

    state["PPE"]["ppe_is_singular"] = False

    rhs = np.ones_like(state["P"])
    P_new = solve_pressure(state, rhs)

    assert np.allclose(P_new, 0.0)


def test_solve_pressure_non_singular():
    s2 = Step2SchemaDummyState(nx=3, ny=3, nz=3)
    state = adapt_step2_to_step3(s2)

    state["PPE"]["ppe_is_singular"] = False

    rhs = np.ones_like(state["P"])
    P_new = solve_pressure(state, rhs)

    assert np.allclose(P_new, 0.0)


def test_solve_pressure_singular():
    s2 = Step2SchemaDummyState(nx=3, ny=3, nz=3)
    state = adapt_step2_to_step3(s2)

    state["PPE"]["ppe_is_singular"] = True

    rhs = np.ones_like(state["P"])
    P_new = solve_pressure(state, rhs)

    fluid = state["is_fluid"]
    assert abs(P_new[fluid].mean()) < 1e-12


def test_convergence_flag():
    s2 = Step2SchemaDummyState(nx=3, ny=3, nz=3)
    state = adapt_step2_to_step3(s2)

    rhs = np.zeros_like(state["P"])
    solve_pressure(state, rhs)

    assert "ppe_converged" in state["PPE"]


def test_minimal_grid():
    """
    This test intentionally bypasses Step‑2 schema validation.
    It only checks that the function does not crash on a 1×1×1 grid.
    """
    state = {
        "Mask": np.ones((1, 1, 1), int),
        "is_fluid": np.ones((1, 1, 1), bool),
        "P": np.zeros((1, 1, 1)),
        "PPE": {"solver": None, "ppe_is_singular": False},
    }

    rhs = np.zeros((1, 1, 1))
    P_new = solve_pressure(state, rhs)

    assert P_new.shape == (1, 1, 1)


def test_solve_pressure_with_custom_solver():
    s2 = Step2SchemaDummyState(nx=3, ny=3, nz=3)
    state = adapt_step2_to_step3(s2)

    def fake_solver(rhs):
        P = rhs + 5.0
        info = {"converged": False, "iterations": 7}
        return P, info

    state["PPE"]["solver"] = fake_solver
    state["PPE"]["ppe_is_singular"] = False

    rhs = np.ones_like(state["P"])
    P_new = solve_pressure(state, rhs)

    assert np.allclose(P_new, rhs + 5.0)
    assert state["PPE"]["ppe_converged"] is False
    assert state["PPE"]["last_iterations"] == 7
