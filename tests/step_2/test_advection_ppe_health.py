# tests/step_2/test_advection_ppe_health.py

import numpy as np
import pytest

from tests.helpers.dummy_state_step2 import DummyState

from src.step2.build_advection_structure import build_advection_structure
from src.step2.prepare_ppe_structure import prepare_ppe_structure
from src.step2.compute_initial_health import compute_initial_health
from src.step2.precompute_constants import precompute_constants


def _make_state_from_dummy(dummy: DummyState) -> dict:
    """
    Adapt DummyState (old-style) into a dict that matches the
    Step 1 output schema expected by Step 2.
    """
    # Old-style keys on DummyState
    cfg = dummy["Config"]
    grid = dummy["Grid"]
    mask = dummy["Mask"]
    const = dummy["Constants"]
    boundary_table = dummy.get("BoundaryTable", [])

    nx = int(grid["nx"])
    ny = int(grid["ny"])
    nz = int(grid["nz"])

    # Build full Step 1-style config block
    config = {
        "boundary_conditions": cfg.get("boundary_conditions", {}),
        "domain": cfg.get("domain", {}),
        "fluid": cfg.get("fluid_properties", {}),
        "forces": cfg.get("forces", {}),
        "geometry_definition": cfg.get("geometry_definition", {}),
        "simulation": cfg.get("simulation_parameters", {}),
    }

    # Build full Step 1-style grid block
    dx = float(grid["dx"])
    dy = float(grid["dy"])
    dz = float(grid["dz"])

    grid_new = {
        "x_min": 0.0,
        "x_max": dx * nx,
        "y_min": 0.0,
        "y_max": dy * ny,
        "z_min": 0.0,
        "z_max": dz * nz,
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "dx": dx,
        "dy": dy,
        "dz": dz,
    }

    # Fields block
    fields = {
        "P": np.zeros((nx, ny, nz), dtype=float),
        "U": np.asarray(dummy.U),
        "V": np.asarray(dummy.V),
        "W": np.asarray(dummy.W),
        "Mask": np.asarray(mask),
    }

    state = {
        "config": config,
        "grid": grid_new,
        "fields": fields,
        "mask_3d": np.asarray(mask),
        "boundary_table": boundary_table,
        "constants": const,
    }

    return state


def test_advection_zero_velocity():
    dummy = DummyState(4, 4, 4)
    state = _make_state_from_dummy(dummy)
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
    dummy = DummyState(4, 4, 4)
    state = _make_state_from_dummy(dummy)
    precompute_constants(state)
    adv = build_advection_structure(state)

    U = np.full_like(state["fields"]["U"], 1.0)
    V = np.full_like(state["fields"]["V"], 1.0)
    W = np.full_like(state["fields"]["W"], 1.0)

    a_u = adv["advection_u"](U, V, W)
    assert np.allclose(a_u, 0.0)


def test_advection_linear_field_nonzero():
    dummy = DummyState(8, 1, 1)
    state = _make_state_from_dummy(dummy)
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
    dummy_upwind = DummyState(8, 1, 1, scheme="upwind")
    state_upwind = _make_state_from_dummy(dummy_upwind)
    precompute_constants(state_upwind)
    adv_upwind = build_advection_structure(state_upwind)

    dummy_central = DummyState(8, 1, 1, scheme="central")
    state_central = _make_state_from_dummy(dummy_central)
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
    dummy = DummyState(16, 1, 1, scheme="upwind")
    state = _make_state_from_dummy(dummy)
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
    dummy = DummyState(1, 1, 1)
    state = _make_state_from_dummy(dummy)
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
    dummy = DummyState(nx, ny, nz, boundary_table=bcs)
    state = _make_state_from_dummy(dummy)
    precompute_constants(state)
    ppe = prepare_ppe_structure(state)
    assert ppe["ppe_is_singular"] is True


def test_ppe_with_pressure_outlet_nonsingular():
    nx, ny, nz = 4, 4, 4
    bcs = [
        {"face": "x_min", "type": "wall"},
        {"face": "x_max", "type": "pressure_outlet"},
    ]
    dummy = DummyState(nx, ny, nz, boundary_table=bcs)
    state = _make_state_from_dummy(dummy)
    precompute_constants(state)
    ppe = prepare_ppe_structure(state)
    assert ppe["ppe_is_singular"] is False


def test_ppe_empty_bc_table_singular():
    dummy = DummyState(4, 4, 4, boundary_table=[])
    state = _make_state_from_dummy(dummy)
    precompute_constants(state)
    ppe = prepare_ppe_structure(state)
    assert ppe["ppe_is_singular"] is True


def test_ppe_rhs_builder_units_and_masking():
    nx, ny, nz = 4, 4, 4
    dummy = DummyState(nx, ny, nz)
    state = _make_state_from_dummy(dummy)
    precompute_constants(state)
    ppe = prepare_ppe_structure(state)
    rhs_builder = ppe["rhs_builder"]

    divergence = np.ones((nx, ny, nz), dtype=float)
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
    dummy = DummyState(4, 4, 4, dt=0.1)
    state = _make_state_from_dummy(dummy)
    precompute_constants(state)
    health = compute_initial_health(state)

    assert health["initial_divergence_norm"] == pytest.approx(0.0)
    assert health["max_velocity_magnitude"] == pytest.approx(0.0)
    assert health["cfl_advection_estimate"] == pytest.approx(0.0)


def test_compute_initial_health_uniform_velocity():
    dummy = DummyState(4, 4, 4, dt=0.1, dx=0.5, dy=0.25, dz=0.2)
    state = _make_state_from_dummy(dummy)
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
    dummy = DummyState(4, 4, 4)
    state = _make_state_from_dummy(dummy)
    precompute_constants(state)

    state["fields"]["U"][:] = 1.0
    health = compute_initial_health(state)

    assert health["initial_divergence_norm"] >= 0.0


def test_compute_initial_health_cfl_greater_than_one():
    dummy = DummyState(4, 4, 4, dt=10.0, dx=0.1, dy=0.1, dz=0.1)
    state = _make_state_from_dummy(dummy)
    precompute_constants(state)

    state["fields"]["U"][:] = 1.0
    state["fields"]["V"][:] = 1.0
    state["fields"]["W"][:] = 1.0

    health = compute_initial_health(state)

    assert health["cfl_advection_estimate"] > 1.0
    assert np.isfinite(health["cfl_advection_estimate"])


def test_compute_initial_health_extremely_small_dx_no_nan():
    dummy = DummyState(4, 4, 4, dt=0.1, dx=1e-8, dy=1e-8, dz=1e-8)
    state = _make_state_from_dummy(dummy)
    precompute_constants(state)

    state["fields"]["U"][:] = 1.0
    state["fields"]["V"][:] = 1.0
    state["fields"]["W"][:] = 1.0

    health = compute_initial_health(state)

    assert np.isfinite(health["cfl_advection_estimate"])