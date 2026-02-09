# tests/step3/test_solve_pressure.py

import numpy as np
from src.step3.solve_pressure import solve_pressure
from tests.helpers.step2_schema_dummy_state import Step2SchemaDummyState


def test_shape_consistency():
    """
    Output pressure must match RHS shape.
    """
    s2 = Step2SchemaDummyState(nx=3, ny=3, nz=3)

    rhs = np.random.randn(*s2["fields"]["P"].shape)
    P_new, meta = solve_pressure(s2, rhs)

    assert P_new.shape == rhs.shape
    assert "converged" in meta
    assert "last_iterations" in meta


def test_singular_mean_subtraction():
    """
    For singular PPE, mean over fluid cells must be zero.
    """
    s2 = Step2SchemaDummyState(nx=3, ny=3, nz=3)

    s2["ppe_structure"]["ppe_is_singular"] = True

    rhs = np.ones_like(s2["fields"]["P"])
    P_new, meta = solve_pressure(s2, rhs)

    fluid = s2["mask_semantics"]["is_fluid"]
    assert abs(P_new[fluid].mean()) < 1e-12


def test_non_singular_zero_solver():
    """
    With no solver and non‑singular PPE, pressure must be zero.
    """
    s2 = Step2SchemaDummyState(nx=3, ny=3, nz=3)

    s2["ppe_structure"]["ppe_is_singular"] = False

    rhs = np.ones_like(s2["fields"]["P"])
    P_new, meta = solve_pressure(s2, rhs)

    assert np.allclose(P_new, 0.0)


def test_custom_solver():
    """
    Custom solver must be invoked and metadata returned.
    """
    s2 = Step2SchemaDummyState(nx=3, ny=3, nz=3)

    def fake_solver(rhs):
        P = rhs + 5.0
        info = {"converged": False, "iterations": 7}
        return P, info

    s2["ppe_structure"]["solver"] = fake_solver
    s2["ppe_structure"]["ppe_is_singular"] = False

    rhs = np.ones_like(s2["fields"]["P"])
    P_new, meta = solve_pressure(s2, rhs)

    assert np.allclose(P_new, rhs + 5.0)
    assert meta["converged"] is False
    assert meta["last_iterations"] == 7


def test_minimal_grid_no_crash():
    """
    Minimal 1×1×1 grid: only checks that the function does not crash.
    """
    state = {
        "constants": {"rho": 1.0, "dt": 0.1},
        "ppe_structure": {"solver": None, "ppe_is_singular": False},
        "mask_semantics": {"is_fluid": np.ones((1, 1, 1), bool)},
    }

    rhs = np.zeros((1, 1, 1))
    P_new, meta = solve_pressure(state, rhs)

    assert P_new.shape == (1, 1, 1)
    assert "converged" in meta
    assert "last_iterations" in meta
