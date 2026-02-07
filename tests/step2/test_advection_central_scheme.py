# tests/step2/test_advection_central_scheme.py

import numpy as np
import pytest

from tests.helpers.step2_schema_dummy_state import Step2SchemaDummyState
from src.step2.build_advection_structure import build_advection_structure


def test_central_advection_linear_field_u():
    """
    Central-difference advection on a linear U field.
    This hits:
    - interior central stencil
    - boundary forward/backward differences
    - central branch of advection_u
    """
    state = Step2SchemaDummyState(3, 3, 3, scheme="central")

    # U(i) = i, shape (nx+1, ny, nz) = (4, 3, 3)
    U = np.zeros((4, 3, 3))
    for i in range(4):
        U[i] = i

    V = np.zeros((3, 4, 3))
    W = np.zeros((3, 3, 4))

    ops = build_advection_structure(state)
    adv_u = ops["advection_u"](U, V, W)

    # Interior derivative of linear field is constant = 1
    assert np.allclose(adv_u[1:-1], 1.0, atol=1e-6)

    # Boundary forward/backward differences
    assert adv_u[0] == pytest.approx(1.0)
    assert adv_u[-1] == pytest.approx(1.0)


def test_central_advection_linear_field_v():
    """
    Same test for V to hit central_y and central_z paths.
    """
    state = Step2SchemaDummyState(3, 3, 3, scheme="central")

    # V(j) = j, shape (nx, ny+1, nz) = (3, 4, 3)
    V = np.zeros((3, 4, 3))
    for j in range(4):
        V[:, j] = j

    U = np.zeros((4, 3, 3))
    W = np.zeros((3, 3, 4))

    ops = build_advection_structure(state)
    adv_v = ops["advection_v"](U, V, W)

    # Interior derivative = 1
    assert np.allclose(adv_v[:, 1:-1], 1.0, atol=1e-6)

    # Boundaries
    assert np.allclose(adv_v[:, 0], 1.0, atol=1e-6)
    assert np.allclose(adv_v[:, -1], 1.0, atol=1e-6)


def test_central_advection_linear_field_w():
    """
    Same test for W to hit central_z and boundary logic.
    """
    state = Step2SchemaDummyState(3, 3, 3, scheme="central")

    # W(k) = k, shape (nx, ny, nz+1) = (3, 3, 4)
    W = np.zeros((3, 3, 4))
    for k in range(4):
        W[:, :, k] = k

    U = np.zeros((4, 3, 3))
    V = np.zeros((3, 4, 3))

    ops = build_advection_structure(state)
    adv_w = ops["advection_w"](U, V, W)

    # Interior derivative = 1
    assert np.allclose(adv_w[:, :, 1:-1], 1.0, atol=1e-6)

    # Boundaries
    assert np.allclose(adv_w[:, :, 0], 1.0, atol=1e-6)
    assert np.allclose(adv_w[:, :, -1], 1.0, atol=1e-6)
