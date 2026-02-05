# tests/step2/test_divergence_gradient_laplacian.py
import numpy as np
import pytest

from step2.build_divergence_operator import build_divergence_operator
from step2.build_gradient_operators import build_gradient_operators
from step2.build_laplacian_operators import build_laplacian_operators
from step2.create_fluid_mask import create_fluid_mask


class DummyState:
    def __init__(self, nx, ny, nz, dx=1.0, dy=1.0, dz=1.0, mask=None):
        self.grid = type("Grid", (), dict(nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz))()
        if mask is None:
            mask = np.ones((nx, ny, nz), dtype=int)
        self.Mask = mask
        self.is_fluid, self.is_boundary_cell = create_fluid_mask(self)


def make_uniform_velocity(state, u_val=0.0, v_val=0.0, w_val=0.0):
    nx, ny, nz = state.grid.nx, state.grid.ny, state.grid.nz
    U = np.full((nx + 1, ny, nz), u_val, dtype=float)
    V = np.full((nx, ny + 1, nz), v_val, dtype=float)
    W = np.full((nx, ny, nz + 1), w_val, dtype=float)
    return U, V, W


def test_divergence_zero_velocity():
    state = DummyState(4, 4, 4)
    div_op = build_divergence_operator(state)
    U, V, W = make_uniform_velocity(state, 0.0, 0.0, 0.0)
    div = div_op(U, V, W)
    assert div.shape == (4, 4, 4)
    assert np.allclose(div, 0.0)


def test_divergence_uniform_velocity_zero():
    state = DummyState(4, 4, 4, dx=0.5)
    div_op = build_divergence_operator(state)
    U, V, W = make_uniform_velocity(state, 1.0, 0.0, 0.0)
    div = div_op(U, V, W)
    assert np.allclose(div, 0.0)


def test_divergence_linear_u_field():
    nx, ny, nz = 8, 1, 1
    state = DummyState(nx, ny, nz, dx=1.0)
    div_op = build_divergence_operator(state)
    U = np.zeros((nx + 1, ny, nz), dtype=float)
    for i in range(nx + 1):
        U[i, 0, 0] = float(i)
    V = np.zeros((nx, ny + 1, nz), dtype=float)
    W = np.zeros((nx, ny, nz + 1), dtype=float)
    div = div_op(U, V, W)
    interior = div[1:-1, 0, 0]
    assert np.allclose(interior, 1.0)


def test_divergence_solid_region_zeroed():
    nx, ny, nz = 4, 4, 4
    mask = np.ones((nx, ny, nz), dtype=int)
    mask[1:3, 1:3, 1:3] = 0
    state = DummyState(nx, ny, nz, mask=mask)
    div_op = build_divergence_operator(state)
    U, V, W = make_uniform_velocity(state, 1.0, 1.0, 1.0)
    div = div_op(U, V, W)
    assert np.all(div[1:3, 1:3, 1:3] == 0.0)


def test_divergence_no_through_mask_single_fluid():
    mask = np.zeros((3, 3, 3), dtype=int)
    mask[1, 1, 1] = 1
    state = DummyState(3, 3, 3, mask=mask)
    div_op = build_divergence_operator(state)
    U, V, W = make_uniform_velocity(state, 10.0, -5.0, 3.0)
    div = div_op(U, V, W)
    assert div[1, 1, 1] == pytest.approx(0.0)


def test_divergence_minimal_grid():
    state = DummyState(1, 1, 1)
    div_op = build_divergence_operator(state)
    U, V, W = make_uniform_velocity(state, 1.0, 2.0, 3.0)
    div = div_op(U, V, W)
    assert div.shape == (1, 1, 1)


def test_gradient_constant_pressure_zero():
    nx, ny, nz = 4, 4, 4
    state = DummyState(nx, ny, nz, dx=0.5, dy=0.5, dz=0.5)
    grad_x, grad_y, grad_z = build_gradient_operators(state)
    P = np.full((nx, ny, nz), 5.0, dtype=float)
    gx = grad_x(P)
    gy = grad_y(P)
    gz = grad_z(P)
    assert np.allclose(gx, 0.0)
    assert np.allclose(gy, 0.0)
    assert np.allclose(gz, 0.0)


def test_gradient_linear_pressure_x():
    nx, ny, nz = 8, 1, 1
    state = DummyState(nx, ny, nz, dx=0.5)
    grad_x, _, _ = build_gradient_operators(state)
    P = np.zeros((nx, ny, nz), dtype=float)
    for i in range(nx):
        P[i, 0, 0] = float(i)
    gx = grad_x(P)
    interior = gx[1:-1, 0, 0]
    assert np.allclose(interior, 1.0 / 0.5)


def test_gradient_solid_pressure_spike_zeroed():
    nx, ny, nz = 4, 4, 4
    mask = np.ones((nx, ny, nz), dtype=int)
    mask[1, 1, 1] = 0
    state = DummyState(nx, ny, nz, mask=mask)
    grad_x, grad_y, grad_z = build_gradient_operators(state)
    P = np.zeros((nx, ny, nz), dtype=float)
    P[1, 1, 1] = 100.0
    gx = grad_x(P)
    gy = grad_y(P)
    gz = grad_z(P)
    assert gx[1, 1, 1] == 0.0
    assert gy[1, 1, 1] == 0.0
    assert gz[1, 1, 1] == 0.0


def test_gradient_minimal_grid():
    state = DummyState(1, 1, 1)
    grad_x, grad_y, grad_z = build_gradient_operators(state)
    P = np.zeros((1, 1, 1), dtype=float)
    gx = grad_x(P)
    gy = grad_y(P)
    gz = grad_z(P)
    assert gx.shape[0] == 2
    assert gy.shape[1] == 2
    assert gz.shape[2] == 2


def test_laplacian_constant_zero():
    nx, ny, nz = 4, 4, 4
    state = DummyState(nx, ny, nz, dx=1.0)
    lap_u, lap_v, lap_w = build_laplacian_operators(state)
    U = np.full((nx + 1, ny, nz), 3.0, dtype=float)
    V = np.full((nx, ny + 1, nz), -2.0, dtype=float)
    W = np.full((nx, ny, nz + 1), 1.0, dtype=float)
    assert np.allclose(lap_u(U), 0.0)
    assert np.allclose(lap_v(V), 0.0)
    assert np.allclose(lap_w(W), 0.0)


def test_laplacian_linear_zero():
    nx, ny, nz = 8, 1, 1
    state = DummyState(nx, ny, nz, dx=1.0)
    lap_u, _, _ = build_laplacian_operators(state)
    U = np.zeros((nx + 1, ny, nz), dtype=float)
    for i in range(nx + 1):
        U[i, 0, 0] = float(i)
    lap = lap_u(U)
    interior = lap[1:-1, 0, 0]
    assert np.allclose(interior, 0.0)


def test_laplacian_quadratic_constant():
    nx, ny, nz = 16, 1, 1
    state = DummyState(nx, ny, nz, dx=1.0)
    lap_u, _, _ = build_laplacian_operators(state)
    U = np.zeros((nx + 1, ny, nz), dtype=float)
    for i in range(nx + 1):
        U[i, 0, 0] = float(i ** 2)
    lap = lap_u(U)
    interior = lap[2:-2, 0, 0]
    assert np.allclose(interior, 2.0)


def test_laplacian_solid_neighbors_truncated():
    nx, ny, nz = 4, 4, 4
    mask = np.ones((nx, ny, nz), dtype=int)
    mask[1, 1, 1] = 0
    state = DummyState(nx, ny, nz, mask=mask)
    lap_u, _, _ = build_laplacian_operators(state)
    U = np.zeros((nx + 1, ny, nz), dtype=float)
    U[:, :, :] = 1.0
    lap = lap_u(U)
    assert lap[1, 1, 1] == 0.0


def test_laplacian_boundary_fluid_treated_as_fluid():
    nx, ny, nz = 4, 1, 1
    mask = np.ones((nx, ny, nz), dtype=int)
    mask[1, 0, 0] = -1
    state = DummyState(nx, ny, nz, mask=mask)
    lap_u, _, _ = build_laplacian_operators(state)
    U = np.zeros((nx + 1, ny, nz), dtype=float)
    for i in range(nx + 1):
        U[i, 0, 0] = float(i ** 2)
    lap = lap_u(U)
    assert np.isfinite(lap[1, 0, 0])


def test_laplacian_minimal_grid():
    state = DummyState(1, 1, 1)
    lap_u, lap_v, lap_w = build_laplacian_operators(state)
    U = np.zeros((2, 1, 1), dtype=float)
    V = np.zeros((1, 2, 1), dtype=float)
    W = np.zeros((1, 1, 2), dtype=float)
    assert lap_u(U).shape == (2, 1, 1)
    assert lap_v(V).shape == (1, 2, 1)
    assert lap_w(W).shape == (1, 1, 2)
