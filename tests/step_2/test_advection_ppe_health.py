# tests/step2/test_advection_ppe_health.py
import numpy as np
import pytest

from step2.build_advection_structure import build_advection_structure
from step2.prepare_ppe_structure import prepare_ppe_structure
from step2.compute_initial_health import compute_initial_health
from step2.create_fluid_mask import create_fluid_mask
from step2.precompute_constants import precompute_constants


class DummyState:
    def __init__(
        self,
        nx,
        ny,
        nz,
        dx=1.0,
        dy=1.0,
        dz=1.0,
        dt=0.1,
        rho=1.0,
        mu=0.1,
        mask=None,
        boundary_table=None,
        scheme="upwind",
    ):
        self.grid = type("Grid", (), dict(nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz))()
        if mask is None:
            mask = np.ones((nx, ny, nz), dtype=int)
        self.Mask = mask
        self.config = type(
            "Config",
            (),
            dict(
                fluid_properties={"density": rho, "viscosity": mu},
                simulation_parameters={"dt": dt, "advection_scheme": scheme},
            ),
        )()
        self.boundary_table = boundary_table or []
        self.constants = None
        self.is_fluid, self.is_boundary_cell = create_fluid_mask(self)
        self.constants = precompute_constants(self)
        self.P = np.zeros((nx, ny, nz), dtype=float)
        self.U = np.zeros((nx + 1, ny, nz), dtype=float)
        self.V = np.zeros((nx, ny + 1, nz), dtype=float)
        self.W = np.zeros((nx, ny, nz + 1), dtype=float)


def test_advection_zero_velocity():
    state = DummyState(4, 4, 4)
    adv = build_advection_structure(state)
    U = np.zeros_like(state.U)
    V = np.zeros_like(state.V)
    W = np.zeros_like(state.W)
    a_u = adv["advection_u"](U, V, W)
    a_v = adv["advection_v"](U, V, W)
    a_w = adv["advection_w"](U, V, W)
    assert np.allclose(a_u, 0.0)
    assert np.allclose(a_v, 0.0)
    assert np.allclose(a_w, 0.0)


def test_advection_uniform_velocity_zero():
    state = DummyState(4, 4, 4)
    adv = build_advection_structure(state)
    U = np.full_like(state.U, 1.0)
    V = np.full_like(state.V, 1.0)
    W = np.full_like(state.W, 1.0)
    a_u = adv["advection_u"](U, V, W)
    assert np.allclose(a_u, 0.0)


def test_advection_linear_field_nonzero():
    state = DummyState(8, 1, 1)
    adv = build_advection_structure(state)
    U = np.zeros_like(state.U)
    for i in range(U.shape[0]):
        U[i, 0, 0] = float(i)
    V = np.zeros_like(state.V)
    W = np.zeros_like(state.W)
    a_u = adv["advection_u"](U, V, W)
    assert np.any(np.abs(a_u) > 0.0)


def test_advection_scheme_switch_upwind_vs_central():
    state_upwind = DummyState(8, 1, 1, scheme="upwind")
    state_central = DummyState(8, 1, 1, scheme="central")
    adv_upwind = build_advection_structure(state_upwind)
    adv_central = build_advection_structure(state_central)
    U = np.zeros_like(state_upwind.U)
    for i in range(U.shape[0]):
        U[i, 0, 0] = float(i)
    V = np.zeros_like(state_upwind.V)
    W = np.zeros_like(state_upwind.W)
    a_u_upwind = adv_upwind["advection_u"](U, V, W)
    a_u_central = adv_central["advection_u"](U, V, W)
    assert not np.allclose(a_u_upwind, a_u_central)


def test_advection_upwind_discontinuous_no_oscillations():
    state = DummyState(16, 1, 1, scheme="upwind")
    adv = build_advection_structure(state)
    U = np.zeros_like(state.U)
    mid = U.shape[0] // 2
    U[:mid, 0, 0] = 1.0
    U[mid:, 0, 0] = 0.0
    V = np.zeros_like(state.V)
    W = np.zeros_like(state.W)
    a_u = adv["advection_u"](U, V, W)
    assert np.all(np.isfinite(a_u))


def test_advection_minimal_grid():
    state = DummyState(1, 1, 1)
    adv = build_advection_structure(state)
    U = np.zeros_like(state.U)
    V = np.zeros_like(state.V)
    W = np.zeros_like(state.W)
    a_u = adv["advection_u"](U, V, W)
    assert a_u.shape == state.U.shape


def test_ppe_enclosed_box_singular():
    nx, ny, nz = 4, 4, 4
    state = DummyState(nx, ny, nz, boundary_table=[{"face": f, "type": "wall"} for f in ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"]])
    ppe = prepare_ppe_structure(state)
    assert ppe["ppe_is_singular"] is True


def test_ppe_with_pressure_outlet_nonsingular():
    nx, ny, nz = 4, 4, 4
    bcs = [
        {"face": "x_min", "type": "wall"},
        {"face": "x_max", "type": "pressure_outlet"},
    ]
    state = DummyState(nx, ny, nz, boundary_table=bcs)
    ppe = prepare_ppe_structure(state)
    assert ppe["ppe_is_singular"] is False


def test_ppe_empty_bc_table_singular():
    state = DummyState(4, 4, 4, boundary_table=[])
    ppe = prepare_ppe_structure(state)
    assert ppe["ppe_is_singular"] is True


def test_ppe_rhs_builder_units_and_masking():
    nx, ny, nz = 4, 4, 4
    state = DummyState(nx, ny, nz)
    ppe = prepare_ppe_structure(state)
    rhs_builder = ppe["rhs_builder"]
    divergence = np.ones((nx, ny, nz), dtype=float)
    divergence[state.is_fluid == False] = 5.0  # solid cells
    rhs = rhs_builder(divergence)
    rho = state.config.fluid_properties["density"]
    dt = state.config.simulation_parameters["dt"]
    assert np.allclose(rhs[state.is_fluid], -rho / dt * divergence[state.is_fluid])
    assert np.all(rhs[state.is_fluid == False] == 0.0)


def test_compute_initial_health_zero_velocity():
    state = DummyState(4, 4, 4, dt=0.1, dx=1.0, dy=1.0, dz=1.0)
    health = compute_initial_health(state)
    assert health["initial_divergence_norm"] == pytest.approx(0.0)
    assert health["max_velocity_magnitude"] == pytest.approx(0.0)
    assert health["cfl_advection_estimate"] == pytest.approx(0.0)


def test_compute_initial_health_uniform_velocity():
    state = DummyState(4, 4, 4, dt=0.1, dx=0.5, dy=0.25, dz=0.2)
    state.U[:] = 1.0
    state.V[:] = 2.0
    state.W[:] = 3.0
    health = compute_initial_health(state)
    max_vel = health["max_velocity_magnitude"]
    assert max_vel > 0.0
    cfl = health["cfl_advection_estimate"]
    assert cfl == pytest.approx(
        state.config.simulation_parameters["dt"]
        * (abs(1.0) / state.grid.dx + abs(2.0) / state.grid.dy + abs(3.0) / state.grid.dz),
        rel=1e-2,
    )


def test_compute_initial_health_divergent_field_norm():
    state = DummyState(4, 4, 4)
    state.U[:] = 1.0
    health = compute_initial_health(state)
    assert health["initial_divergence_norm"] >= 0.0


def test_compute_initial_health_cfl_greater_than_one():
    state = DummyState(4, 4, 4, dt=10.0, dx=0.1, dy=0.1, dz=0.1)
    state.U[:] = 1.0
    state.V[:] = 1.0
    state.W[:] = 1.0
    health = compute_initial_health(state)
    assert health["cfl_advection_estimate"] > 1.0
    assert np.isfinite(health["cfl_advection_estimate"])


def test_compute_initial_health_extremely_small_dx_no_nan():
    state = DummyState(4, 4, 4, dt=0.1, dx=1e-8, dy=1e-8, dz=1e-8)
    state.U[:] = 1.0
    state.V[:] = 1.0
    state.W[:] = 1.0
    health = compute_initial_health(state)
    cfl = health["cfl_advection_estimate"]
    assert np.isfinite(cfl)
