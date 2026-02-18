# tests/step3/test_build_ppe_rhs.py

import numpy as np
from src.step3.build_ppe_rhs import build_ppe_rhs
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy


def test_zero_divergence():
    """
    If divergence operator returns zero everywhere, RHS must be zero.
    """
    state = make_step2_output_dummy(nx=3, ny=3, nz=3)

    def div_zero(U, V, W):
        return np.zeros_like(state.fields["P"])

    state.operators["divergence"] = div_zero

    U = state.fields["U"]
    V = state.fields["V"]
    W = state.fields["W"]

    rhs = build_ppe_rhs(state, U, V, W)
    assert np.allclose(rhs, 0.0)


def test_uniform_divergence():
    """
    If divergence is uniformly 1, RHS must be rho/dt everywhere.
    """
    state = make_step2_output_dummy(nx=3, ny=3, nz=3)

    pattern = np.ones_like(state.fields["P"])

    def div_one(U, V, W):
        return pattern

    state.operators["divergence"] = div_one

    U = state.fields["U"]
    V = state.fields["V"]
    W = state.fields["W"]

    rhs = build_ppe_rhs(state, U, V, W)

    rho = state.constants["rho"]
    dt = state.constants["dt"]

    assert np.allclose(rhs, rho / dt)


def test_solid_zeroing():
    """
    RHS must be zeroed inside solid cells.
    """
    state = make_step2_output_dummy(nx=3, ny=3, nz=3)

    def div_one(U, V, W):
        return np.ones_like(state.fields["P"])

    state.operators["divergence"] = div_one

    # Mark a solid cell
    state.is_fluid[1, 1, 1] = False

    U = state.fields["U"]
    V = state.fields["V"]
    W = state.fields["W"]

    rhs = build_ppe_rhs(state, U, V, W)

    assert rhs[1, 1, 1] == 0.0


def test_minimal_grid_no_crash():
    """
    Minimal 1×1×1 grid: only checks that the function does not crash.
    """
    state = make_step2_output_dummy(nx=1, ny=1, nz=1)

    def div_zero(U, V, W):
        return np.zeros((1, 1, 1))

    state.operators["divergence"] = div_zero

    U = state.fields["U"]
    V = state.fields["V"]
    W = state.fields["W"]

    rhs = build_ppe_rhs(state, U, V, W)

    assert rhs.shape == (1, 1, 1)
