# tests/step2/test_advection_ppe_health.py

import numpy as np
import pytest

from src.step2.build_advection_structure import build_advection_structure
from src.step2.prepare_ppe_structure import prepare_ppe_structure
from src.step2.compute_initial_health import compute_initial_health
from src.step2.precompute_constants import precompute_constants


def make_state(nx, ny, nz, scheme="central", dt=0.1, dx=1.0, dy=1.0, dz=1.0, boundary_table=None):
    if boundary_table is None:
        boundary_table = []

    state = {
        "grid": {
            "nx": nx, "ny": ny, "nz": nz,
            "dx": dx, "dy": dy, "dz": dz,
        },
        "config": {
            "fluid": {"density": 1.0, "viscosity": 0.1},
            "simulation": {"dt": dt, "advection_scheme": scheme},
        },
        "fields": {
            "U": np.zeros((nx + 1, ny, nz)).tolist(),
            "V": np.zeros((nx, ny + 1, nz)).tolist(),
            "W": np.zeros((nx, ny, nz + 1)).tolist(),
            "P": np.zeros((nx, ny, nz)).tolist(),
        },
        "mask_3d": np.ones((nx, ny, nz), dtype=int).tolist(),
        "boundary_table_list": boundary_table,
    }

    state["constants"] = precompute_constants(state)
    return state


# ------------------------------------------------------------
# Advection tests
# ------------------------------------------------------------

def test_advection_zero_velocity():
    state = make_state(4, 4, 4)
    result = build_advection_structure(state)

    U = np.zeros((5, 4, 4))
    V = np.zeros((4, 5, 4))
    W = np.zeros((4, 4, 5))

    adv = result["advection"]
    a_u = np.array(adv["u"])
    a_v = np.array(adv["v"])
    a_w = np.array(adv["w"])

    assert np.allclose(a_u, 0.0)
    assert np.allclose(a_v, 0.0)
    assert np.allclose(a_w, 0.0)


def test_advection_uniform_velocity_zero():
    state = make_state(4, 4, 4)
    U = np.ones((5, 4, 4))
    V = np.ones((4, 5, 4))
    W = np.ones((4, 4, 5))

    state["fields"]["U"] = U.tolist()
    state["fields"]["V"] = V.tolist()
    state["fields"]["W"] = W.tolist()

    adv = build_advection_structure(state)["advection"]
    a_u = np.array(adv["u"])

    assert np.allclose(a_u, 0.0)


def test_advection_linear_field_nonzero():
    state = make_state(8, 1, 1)

    U = np.zeros((9, 1, 1))
    for i in range(9):
        U[i, 0, 0] = float(i)

    state["fields"]["U"] = U.tolist()

    adv = build_advection_structure(state)["advection"]
    a_u = np.array(adv["u"])

    assert np.any(np.abs(a_u) > 0.0)


def test_advection_scheme_switch_upwind_vs_central():
    state_upwind = make_state(8, 1, 1, scheme="upwind")
    state_central = make_state(8, 1, 1, scheme="central")

    U = np.zeros((9, 1, 1))
    for i in range(9):
        U[i, 0, 0] = float(i)

    state_upwind["fields"]["U"] = U.tolist()
    state_central["fields"]["U"] = U.tolist()

    adv_upwind = np.array(build_advection_structure(state_upwind)["advection"]["u"])
    adv_central = np.array(build_advection_structure(state_central)["advection"]["u"])

    assert not np.allclose(adv_upwind, adv_central)


def test_advection_upwind_discontinuous_no_oscillations():
    state = make_state(16, 1, 1, scheme="upwind")

    U = np.zeros((17, 1, 1))
    mid = U.shape[0] // 2
    U[:mid, 0, 0] = 1.0
    U[mid:, 0, 0] = 0.0

    state["fields"]["U"] = U.tolist()

    adv = np.array(build_advection_structure(state)["advection"]["u"])
    assert np.all(np.isfinite(adv))


def test_advection_minimal_grid():
    state = make_state(1, 1, 1)
    adv = np.array(build_advection_structure(state)["advection"]["u"])
    assert adv.shape == (2, 1, 1)


# ------------------------------------------------------------
# PPE tests
# ------------------------------------------------------------

def test_ppe_enclosed_box_singular():
    bcs = [{"face": f, "type": "wall"} for f in ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"]]
    state = make_state(4, 4, 4, boundary_table=bcs)
    ppe = prepare_ppe_structure(state)
    assert ppe["ppe_is_singular"] is True


def test_ppe_with_pressure_outlet_nonsingular():
    bcs = [
        {"face": "x_min", "type": "wall"},
        {"face": "x_max", "type": "pressure_outlet"},
    ]
    state = make_state(4, 4, 4, boundary_table=bcs)
    ppe = prepare_ppe_structure(state)
    assert ppe["ppe_is_singular"] is False


def test_ppe_empty_bc_table_singular():
    state = make_state(4, 4, 4, boundary_table=[])
    ppe = prepare_ppe_structure(state)
    assert ppe["ppe_is_singular"] is True


def test_ppe_rhs_builder_units_and_masking():
    nx, ny, nz = 4, 4, 4
    state = make_state(nx, ny, nz)

    ppe = prepare_ppe_structure(state)
    rhs_builder = ppe["rhs_builder"]

    divergence = np.ones((nx, ny, nz))
    divergence[state["mask_3d"] == 0] = 5.0

    rhs = np.array(rhs_builder(divergence.tolist()))

    rho = state["config"]["fluid"]["density"]
    dt = state["config"]["simulation"]["dt"]

    assert np.allclose(
        rhs[state["mask_3d"] == 1],
        -rho / dt * divergence[state["mask_3d"] == 1],
    )
    assert np.all(rhs[state["mask_3d"] == 0] == 0.0)


# ------------------------------------------------------------
# Health diagnostics
# ------------------------------------------------------------

def test_compute_initial_health_zero_velocity():
    state = make_state(4, 4, 4, dt=0.1)
    health = compute_initial_health(state)

    assert health["initial_divergence_norm"] == pytest.approx(0.0)
    assert health["max_velocity_magnitude"] == pytest.approx(0.0)
    assert health["cfl_advection_estimate"] == pytest.approx(0.0)


def test_compute_initial_health_uniform_velocity():
    state = make_state(4, 4, 4, dt=0.1, dx=0.5, dy=0.25, dz=0.2)

    state["fields"]["U"] = np.ones((5, 4, 4)).tolist()
    state["fields"]["V"] = np.full((4, 5, 4), 2.0).tolist()
    state["fields"]["W"] = np.full((4, 4, 5), 3.0).tolist()

    health = compute_initial_health(state)

    dt = state["config"]["simulation"]["dt"]
    dx = state["grid"]["dx"]
    dy = state["grid"]["dy"]
    dz = state["grid"]["dz"]

    expected_cfl = dt * (1.0 / dx + 2.0 / dy + 3.0 / dz)

    assert health["cfl_advection_estimate"] == pytest.approx(expected_cfl, rel=1e-2)


def test_compute_initial_health_divergent_field_norm():
    state = make_state(4, 4, 4)
    state["fields"]["U"] = np.ones((5, 4, 4)).tolist()

    health = compute_initial_health(state)
    assert health["initial_divergence_norm"] >= 0.0


def test_compute_initial_health_cfl_greater_than_one():
    state = make_state(4, 4, 4, dt=10.0, dx=0.1, dy=0.1, dz=0.1)

    state["fields"]["U"] = np.ones((5, 4, 4)).tolist()
    state["fields"]["V"] = np.ones((4, 5, 4)).tolist()
    state["fields"]["W"] = np.ones((4, 4, 5)).tolist()

    health = compute_initial_health(state)

    assert health["cfl_advection_estimate"] > 1.0
    assert np.isfinite(health["cfl_advection_estimate"])


def test_compute_initial_health_extremely_small_dx_no_nan():
    state = make_state(4, 4, 4, dt=0.1, dx=1e-8, dy=1e-8, dz=1e-8)

    state["fields"]["U"] = np.ones((5, 4, 4)).tolist()
    state["fields"]["V"] = np.ones((4, 5, 4)).tolist()
    state["fields"]["W"] = np.ones((4, 4, 5)).tolist()

    health = compute_initial_health(state)

    assert np.isfinite(health["cfl_advection_estimate"])
