# tests/step2/test_step2_optional.py

import numpy as np
import pytest

from tests.helpers.step2_schema_dummy_state import Step2SchemaDummyState
from src.step2.build_gradient_operators import build_gradient_operators
from src.step2.build_laplacian_operators import build_laplacian_operators
from src.step2.build_ppe_rhs import build_ppe_rhs


def test_gradient_of_constant_pressure_is_zero():
    """
    ∇P = 0 for constant pressure field.
    """
    s2 = Step2SchemaDummyState(nx=4, ny=4, nz=4)
    s2["fields"]["P"].fill(5.0)

    grads = build_gradient_operators(s2)
    gx = grads["gradient_p_x"]["op"](s2["fields"]["P"])
    gy = grads["gradient_p_y"]["op"](s2["fields"]["P"])
    gz = grads["gradient_p_z"]["op"](s2["fields"]["P"])

    assert np.allclose(gx, 0.0)
    assert np.allclose(gy, 0.0)
    assert np.allclose(gz, 0.0)


def test_laplacian_of_linear_field_is_zero():
    """
    ∇²(ax + by + cz) = 0 for linear fields.
    """
    nx, ny, nz = 4, 4, 4
    s2 = Step2SchemaDummyState(nx=nx, ny=ny, nz=nz)

    X = np.linspace(0, 1, nx)
    Y = np.linspace(0, 1, ny)
    Z = np.linspace(0, 1, nz)
    xx, yy, zz = np.meshgrid(X, Y, Z, indexing="ij")

    s2["fields"]["P"] = xx + 2 * yy + 3 * zz

    laps = build_laplacian_operators(s2)
    lap = laps["laplacian_p"]["op"](s2["fields"]["P"])

    assert np.allclose(lap, 0.0, atol=1e-6)


def test_ppe_rhs_integrates_to_zero_for_singular_case():
    """
    For singular PPE, RHS must integrate to zero.
    """
    s2 = Step2SchemaDummyState(nx=4, ny=4, nz=4)
    s2["ppe"]["ppe_is_singular"] = True

    U = s2["fields"]["U"]
    V = s2["fields"]["V"]
    W = s2["fields"]["W"]

    rhs = build_ppe_rhs(s2, U, V, W)

    total = float(np.sum(rhs))
    assert abs(total) < 1e-6
