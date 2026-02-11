# tests/step2/test_step2_optional.py

import numpy as np
import pytest

from tests.helpers.step2_schema_dummy_state import Step2SchemaDummyState
from src.step2.build_gradient_operators import build_gradient_operators
from src.step2.build_laplacian_operators import build_laplacian_operators
from src.step2.build_ppe_rhs import build_ppe_rhs


def test_gradient_of_constant_pressure_is_zero():
    s2 = Step2SchemaDummyState(nx=4, ny=4, nz=4)
    s2["mask_3d"] = np.asarray(s2["mask"])

    s2["fields"]["P"].fill(5.0)

    grad_x, grad_y, grad_z = build_gradient_operators(s2)

    assert np.allclose(grad_x(s2["fields"]["P"]), 0.0)
    assert np.allclose(grad_y(s2["fields"]["P"]), 0.0)
    assert np.allclose(grad_z(s2["fields"]["P"]), 0.0)


def test_laplacian_of_linear_field_is_zero():
    nx, ny, nz = 4, 4, 4
    s2 = Step2SchemaDummyState(nx=nx, ny=ny, nz=nz)
    s2["mask_3d"] = np.asarray(s2["mask"])

    X = np.linspace(0, 1, nx)
    Y = np.linspace(0, 1, ny)
    Z = np.linspace(0, 1, nz)
    xx, yy, zz = np.meshgrid(X, Y, Z, indexing="ij")

    s2["fields"]["P"] = xx + 2 * yy + 3 * zz

    laps = build_laplacian_operators(s2)
    lap_p = laps["laplacian_p"]["op"]

    lap = lap_p(s2["fields"]["P"])
    assert np.allclose(lap, 0.0, atol=1e-6)


def test_ppe_rhs_integrates_to_zero_for_singular_case():
    s2 = Step2SchemaDummyState(nx=4, ny=4, nz=4)
    s2["ppe"]["ppe_is_singular"] = True
    s2["mask_3d"] = np.asarray(s2["mask"])

    div = np.zeros_like(s2["fields"]["P"])
    mask = np.asarray(s2["mask"])
    rho = s2["constants"]["rho"]
    dt = s2["constants"]["dt"]
    dx = s2["constants"]["dx"]
    dy = s2["constants"]["dy"]
    dz = s2["constants"]["dz"]

    rhs = build_ppe_rhs(div, mask, rho, dt, dx, dy, dz)
    assert abs(float(np.sum(rhs))) < 1e-6
