# tests/step2/test_advection_central_scheme.py

import numpy as np
import pytest

from src.step2.build_advection_structure import build_advection_structure
from src.step2.precompute_constants import precompute_constants


def make_state(nx, ny, nz, scheme="central"):
    return {
        "grid": {
            "nx": nx, "ny": ny, "nz": nz,
            "dx": 1.0, "dy": 1.0, "dz": 1.0,
        },
        "config": {
            "fluid": {"density": 1.0, "viscosity": 0.1},
            "simulation": {"dt": 0.1, "advection_scheme": scheme},
        },
        "constants": precompute_constants({
            "grid": {
                "nx": nx, "ny": ny, "nz": nz,
                "dx": 1.0, "dy": 1.0, "dz": 1.0,
            },
            "config": {
                "fluid": {"density": 1.0, "viscosity": 0.1},
                "simulation": {"dt": 0.1},
            },
        }),
        "fields": {
            "U": None,
            "V": None,
            "W": None,
            "P": np.zeros((nx, ny, nz)).tolist(),
        },
        "mask_3d": np.ones((nx, ny, nz), dtype=int).tolist(),
        "boundary_table_list": [],
    }


def test_central_advection_linear_field_u():
    state = make_state(3, 3, 3, scheme="central")

    U = np.zeros((4, 3, 3))
    for i in range(4):
        U[i] = i

    V = np.zeros((3, 4, 3))
    W = np.zeros((3, 3, 4))

    state["fields"]["U"] = U.tolist()
    state["fields"]["V"] = V.tolist()
    state["fields"]["W"] = W.tolist()

    result = build_advection_structure(state)
    adv_u = np.array(result["advection"]["u"])

    expected = np.zeros_like(U)
    for i in range(4):
        expected[i] = i

    assert np.allclose(adv_u, expected, atol=1e-6)


def test_central_advection_linear_field_v():
    state = make_state(3, 3, 3, scheme="central")

    V = np.zeros((3, 4, 3))
    for j in range(4):
        V[:, j] = j

    U = np.zeros((4, 3, 3))
    W = np.zeros((3, 3, 4))

    state["fields"]["U"] = U.tolist()
    state["fields"]["V"] = V.tolist()
    state["fields"]["W"] = W.tolist()

    result = build_advection_structure(state)
    adv_v = np.array(result["advection"]["v"])

    expected = np.zeros_like(V)
    for j in range(4):
        expected[:, j] = j

    assert np.allclose(adv_v, expected, atol=1e-6)


def test_central_advection_linear_field_w():
    state = make_state(3, 3, 3, scheme="central")

    W = np.zeros((3, 3, 4))
    for k in range(4):
        W[:, :, k] = k

    U = np.zeros((4, 3, 3))
    V = np.zeros((3, 4, 3))

    state["fields"]["U"] = U.tolist()
    state["fields"]["V"] = V.tolist()
    state["fields"]["W"] = W.tolist()

    result = build_advection_structure(state)
    adv_w = np.array(result["advection"]["w"])

    expected = np.zeros_like(W)
    for k in range(4):
        expected[:, :, k] = k

    assert np.allclose(adv_w, expected, atol=1e-6)
