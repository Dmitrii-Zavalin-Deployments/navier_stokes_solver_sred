# tests/step3/test_solve_pressure.py

import numpy as np
from src.step3.solve_pressure import solve_pressure
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy


def test_shape_consistency():
    """
    Output pressure must match RHS shape.
    """
    state = make_step2_output_dummy(nx=3, ny=3, nz=3)

    rhs = np.random.randn(*state.fields["P"].shape)
    P_new, meta = solve_pressure(state, rhs)

    assert P_new.shape == rhs.shape
    assert "converged" in meta
    assert "last_iterations" in meta


def test_singular_mean_subtraction():
    """
    For singular PPE, mean over fluid cells must be zero.
    """
    state = make_step2_output_dummy(nx=3, ny=3, nz=3)

    # Mark PPE as singular
    state.ppe["ppe_is_singular"] = True

    rhs = np.ones_like(state.fields["P"])
    P_new, meta = solve_pressure(state, rhs)

    fluid = state.is_fluid
    assert abs(P_new[fluid].mean()) < 1e-12


def test_non_singular_zero_solver():
    """
    With no solver and non‑singular PPE, pressure must be zero.
    """
    state = make_step2_output_dummy(nx=3, ny=3, nz=3)

    state.ppe["ppe_is_singular"] = False
    state.ppe["solver"] = None  # no custom solver

    rhs = np.ones_like(state.fields["P"])
    P_new, meta = solve_pressure(state, rhs)

    assert np.allclose(P_new, 0.0)


def test_custom_solver():
    """
    Custom solver must be invoked and metadata returned.
    """
    state = make_step2_output_dummy(nx=3, ny=3, nz=3)

    def fake_solver(rhs):
        P = rhs + 5.0
        info = {"converged": False, "iterations": 7}
        return P, info

    state.ppe["solver"] = fake_solver
    state.ppe["ppe_is_singular"] = False

    rhs = np.ones_like(state.fields["P"])
    P_new, meta = solve_pressure(state, rhs)

    assert np.allclose(P_new, rhs + 5.0)
    assert meta["converged"] is False
    assert meta["last_iterations"] == 7


def test_minimal_grid_no_crash():
    """
    Minimal 1×1×1 grid: only checks that the function does not crash.
    """
    state = make_step2_output_dummy(nx=1, ny=1, nz=1)

    # Minimal PPE config
    state.ppe["solver"] = None
    state.ppe["ppe_is_singular"] = False

    rhs = np.zeros((1, 1, 1))
    P_new, meta = solve_pressure(state, rhs)

    assert P_new.shape == (1, 1, 1)
    assert "converged" in meta
    assert "last_iterations" in meta
