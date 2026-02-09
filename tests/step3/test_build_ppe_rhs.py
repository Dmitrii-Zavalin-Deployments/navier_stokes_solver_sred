# tests/step3/test_build_ppe_rhs.py

import numpy as np
from src.step3.build_ppe_rhs import build_ppe_rhs
from tests.helpers.step2_schema_dummy_state import Step2SchemaDummyState


def test_zero_divergence():
    """
    If divergence operator returns zero everywhere, RHS must be zero.
    """
    s2 = Step2SchemaDummyState(nx=3, ny=3, nz=3)

    def div_zero(U, V, W):
        return np.zeros_like(s2["fields"]["P"])

    s2["divergence"]["op"] = div_zero

    U = np.asarray(s2["fields"]["U"])
    V = np.asarray(s2["fields"]["V"])
    W = np.asarray(s2["fields"]["W"])

    rhs = build_ppe_rhs(s2, U, V, W)
    assert np.allclose(rhs, 0.0)


def test_uniform_divergence():
    """
    If divergence is uniformly 1, RHS must be rho/dt everywhere.
    """
    s2 = Step2SchemaDummyState(nx=3, ny=3, nz=3)

    pattern = np.ones_like(s2["fields"]["P"])

    def div_one(U, V, W):
        return pattern

    s2["divergence"]["op"] = div_one

    U = np.asarray(s2["fields"]["U"])
    V = np.asarray(s2["fields"]["V"])
    W = np.asarray(s2["fields"]["W"])

    rhs = build_ppe_rhs(s2, U, V, W)

    rho = s2["constants"]["rho"]
    dt = s2["constants"]["dt"]

    assert np.allclose(rhs, rho / dt)


def test_solid_zeroing():
    """
    RHS must be zeroed inside solid cells.
    """
    s2 = Step2SchemaDummyState(nx=3, ny=3, nz=3)

    # Uniform divergence
    def div_one(U, V, W):
        return np.ones_like(s2["fields"]["P"])

    s2["divergence"]["op"] = div_one

    # Mark a solid cell
    mask = np.array(s2["mask_semantics"]["mask"], copy=True)
    mask[1, 1, 1] = 0
    s2["mask_semantics"]["mask"] = mask
    s2["mask_semantics"]["is_solid"] = (mask == 0)

    U = np.asarray(s2["fields"]["U"])
    V = np.asarray(s2["fields"]["V"])
    W = np.asarray(s2["fields"]["W"])

    rhs = build_ppe_rhs(s2, U, V, W)

    assert rhs[1, 1, 1] == 0.0


def test_minimal_grid_no_crash():
    """
    Minimal 1×1×1 grid: only checks that the function does not crash.
    """
    def div_zero(U, V, W):
        return np.zeros((1, 1, 1))

    state = {
        "constants": {"rho": 1.0, "dt": 0.1},
        "divergence": {"op": div_zero},
        "mask_semantics": {
            "is_solid": np.zeros((1, 1, 1), bool),
        },
    }

    U = np.zeros((2, 1, 1))
    V = np.zeros((1, 2, 1))
    W = np.zeros((1, 1, 2))

    rhs = build_ppe_rhs(state, U, V, W)

    assert rhs.shape == (1, 1, 1)
