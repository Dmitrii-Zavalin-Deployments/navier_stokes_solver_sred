# tests/step2/test_advection_central_scheme.py

import numpy as np
import pytest

from tests.helpers.step2_schema_dummy_state import Step2SchemaDummyState
from src.step2.build_advection_structure import build_advection_structure
from src.step2.precompute_constants import precompute_constants


def test_central_advection_linear_field_u():
    state = Step2SchemaDummyState(3, 3, 3, scheme="central")
    precompute_constants(state)

    U = np.zeros((4, 3, 3))
    for i in range(4):
        U[i] = i

    V = np.zeros((3, 4, 3))
    W = np.zeros((3, 3, 4))

    ops = build_advection_structure(state)
    adv_u = ops["advection_u"](U, V, W)

    # Expected: adv_u[i] = i
    expected = np.zeros_like(U)
    for i in range(4):
        expected[i] = i

    assert np.allclose(adv_u, expected, atol=1e-6)


def test_central_advection_linear_field_v():
    state = Step2SchemaDummyState(3, 3, 3, scheme="central")
    precompute_constants(state)

    V = np.zeros((3, 4, 3))
    for j in range(4):
        V[:, j] = j

    U = np.zeros((4, 3, 3))
    W = np.zeros((3, 3, 4))

    ops = build_advection_structure(state)
    adv_v = ops["advection_v"](U, V, W)

    expected = np.zeros_like(V)
    for j in range(4):
        expected[:, j] = j

    assert np.allclose(adv_v, expected, atol=1e-6)


def test_central_advection_linear_field_w():
    state = Step2SchemaDummyState(3, 3, 3, scheme="central")
    precompute_constants(state)

    W = np.zeros((3, 3, 4))
    for k in range(4):
        W[:, :, k] = k

    U = np.zeros((4, 3, 3))
    V = np.zeros((3, 4, 3))

    ops = build_advection_structure(state)
    adv_w = ops["advection_w"](U, V, W)

    expected = np.zeros_like(W)
    for k in range(4):
        expected[:, :, k] = k

    assert np.allclose(adv_w, expected, atol=1e-6)
