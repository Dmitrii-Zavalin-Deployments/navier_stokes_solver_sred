# tests/step2/test_advection_ppe_health.py

import numpy as np
import pytest

from tests.helpers.schema_dummy_state import SchemaDummyState

from src.step2.build_advection_structure import build_advection_structure
from src.step2.prepare_ppe_structure import prepare_ppe_structure
from src.step2.compute_initial_health import compute_initial_health
from src.step2.precompute_constants import precompute_constants


def test_advection_zero_velocity():
    state = SchemaDummyState(4, 4, 4)
    precompute_constants(state)
    adv = build_advection_structure(state)

    U = np.zeros_like(state["fields"]["U"])
    V = np.zeros_like(state["fields"]["V"])
    W = np.zeros_like(state["fields"]["W"])

    a_u = adv["advection_u"](U, V, W)
    a_v = adv["advection_v"](U, V, W)
    a_w = adv["advection_w"](U, V, W)

    assert np.allclose(a_u, 0.0)
    assert np.allclose(a_v, 0.0)
    assert np.allclose(a_w, 0.0)


def test_advection_uniform_velocity_zero():
    state = SchemaDummyState(4, 4, 4)
    precompute_constants(state)
    adv = build_advection_structure(state)

    U = np.full_like(state["fields"]["U"], 1.0)
    V = np.full_like(state["fields"]["V"], 1.0)
    W = np.full_like(state["fields"]["W"], 1.0)

    a_u = adv["advection_u"](U, V, W)
    assert np.allclose(a_u, 0.0)


def test_advection_linear_field_nonzero():
    state = SchemaDummyState(8, 1, 1)
    precompute_constants(state)
    adv = build_advection_structure(state)

    U = np.zeros_like(state["fields"]["U"])
    for i in range(U.shape[0]):
        U[i, 0, 0] = float(i)

    V = np.zeros_like(state["fields"]["V"])
    W = np.zeros_like(state["fields"]["W"])

    a_u = adv["advection_u"](U, V, W)
    assert np.any(np.abs(a_u) > 0.0)


def test_advection_scheme_switch_upwind_vs_central():
    state_upwind = SchemaDummyState(8, 1, 1, scheme="upwind")
    precompute_constants(state_upwind)
    adv_upwind = build_advection_structure(state_upwind)

    state_central = SchemaDummyState(8, 1, 1, scheme="central")
    precompute_constants(state_central)
    adv_central = build_advection_structure(state_central)

    U = np.zeros_like(state_upwind["fields"]["U"])
    for i in range(U.shape[0]):
        U[i, 0, 0] = float(i)

    V = np.zeros_like(state_upwind["fields"]["V"])
    W = np.zeros_like(state_upwind["fields"]["W"])

    a_u_upwind = adv_upwind["advection_u"](U, V, W)
    a_u_central = adv_central["advection_u"](U, V, W)

    assert not np.allclose(a_u_upwind, a_u_central)


def test_advection_upwind_discontinuous_no_oscillations():
    state = SchemaDummyState(16, 1, 1, scheme="upwind")
    precompute_constants(state)
    adv = build_advection_structure(state)

    U = np.zeros_like(state["fields"]["U"])
    mid = U.shape[0] // 2
    U[:mid, 0, 0] = 1.0
    U[mid:, 0, 0] = 0.0

    V = np.zeros_like(state["fields"]["V"])
    W = np.zeros_like(state["fields"]["W"])

    a_u = adv["advection_u"](U, V, W)
    assert np.all(np.isfinite(a_u))


def test_advection_minimal_grid():
    state = SchemaDummyState(1, 1, 1)
    precompute_constants(state)
    adv = build_advection_structure(state)

    U = np.zeros_like(state["fields"]["U"])
    V = np.zeros_like(state["fields"]["V"])
    W = np.zeros_like(state["fields"]["W"])

    a_u = adv["advection_u"](U, V, W)
    assert a_u.shape == state["fields"]["U"].shape


def test_ppe_enclosed_box_singular():
    nx, ny, nz = 4, 4, 4
    bcs = [{"face": f, "type": "wall"} for f in ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"]]

    state = SchemaDummyState(nx, ny, nz, boundary_table=bcs)
    precompute_constants(state)
    ppe = prepare_ppe_structure(state)

    assert ppe["ppe_is_singular"] is True


def test_ppe_with_pressure_outlet_nonsingular():
    nx, ny, nz = 4, 4, 4
    bcs = [
        {"face": "x_min", "type": "wall"},
        {"face": "x_max", "type": "pressure_outlet"},
    ]

    state = SchemaDummyState(nx, ny, nz, boundary_table=bcs)
    precompute_constants(state)
    ppe = prepare_ppe_structure(state)

    assert ppe["ppe_is_singular"] is False


def test_ppe_empty_bc_table_singular():
    state = SchemaDummyState(4, 4, 4, boundary_table=[])
    precompute_constants(state)
    ppe = prepare_ppe_structure(state)

    assert ppe["ppe_is_singular"] is True


def test_ppe_rhs_builder_units_and_masking():
    nx, ny, nz = 4, 4, 4
    state = SchemaDummyState(nx, ny, nz)
    precompute_constants(state)

    ppe = prepare_ppe_structure(state)
    rhs_builder = ppe["rhs_builder"]

    divergence = np.ones((nx, ny, nz), float)
    divergence[state["fields"]["Mask"] == 0] = 5.0

    rhs = rhs_builder(divergence)

    rho = state["config"]["fluid"]["density"]
    dt = state["config"]["simulation"]["dt"]

    assert np.allclose(
        rhs[state["fields"]["Mask"] == 1],
        -rho / dt * divergence[state["fields"]["Mask"] == 1],
    )
    assert np.all(rhs[state["fields"]["Mask"] == 0] == 0.0)


def test_compute_initial_health_zero_velocity():
    state = SchemaDummyState(4, 4, 4, dt=0.1)
    precompute_constants(state)
    health = compute_initial_health(state)

    assert health["initial_divergence_norm"] == pytest.approx(0.0)
    assert health["max_velocity_magnitude"] == pytest.approx(0.0)
    assert health["cfl_advection_estimate"] == pytest.approx(0.0)


def test_compute_initial_health_uniform_velocity():
    state = SchemaDummyState(4, 4, 4, dt=0.1, dx=0.5, dy=0.25, dz=0.2)
    precompute_constants(state)

    state["fields"]["U"][:] = 1.0
    state["fields"]["V"][:] = 2.0
    state["fields"]["W"][:] = 3.0

    health = compute_initial_health(state)

    dt = state["config"]["simulation"]["dt"]
    dx = state["grid"]["dx"]
    dy = state["grid"]["dy"]
    dz = state["grid"]["dz"]

    expected_cfl = dt * (abs(1.0) / dx + abs(2.0) / dy + abs(3.0) / dz)

    assert health["cfl_advection_estimate"] == pytest.approx(expected_cfl, rel=1e-2)


def test_compute_initial_health_divergent_field_norm():
    state = SchemaDummyState(4, 4, 4)
    precompute_constants(state)

    state["fields"]["U"][:] = 1.0
    health = compute_initial_health(state)

    assert health["initial_divergence_norm"] >= 0.0


def test_compute_initial_health_cfl_greater_than_one():
    state = SchemaDummyState(4, 4, 4, dt=10.0, dx=0.1, dy=0.1, dz=0.1)
    precompute_constants(state)

    state["fields"]["U"][:] = 1.0
    state["fields"]["V"][:] = 1.0
    state["fields"]["W"][:] = 1.0

    health = compute_initial_health(state)

    assert health["cfl_advection_estimate"] > 1.0
    assert np.isfinite(health["cfl_advection_estimate"])


def test_compute_initial_health_extremely_small_dx_no_nan():
    state = SchemaDummyState(4, 4, 4, dt=0.1, dx=1e-8, dy=1e-8, dz=1e-8)
    precompute_constants(state)

    state["fields"]["U"][:] = 1.0
    state["fields"]["V"][:] = 1.0
    state["fields"]["W"][:] = 1.0

    health = compute_initial_health(state)

    assert np.isfinite(health["cfl_advection_estimate"])
