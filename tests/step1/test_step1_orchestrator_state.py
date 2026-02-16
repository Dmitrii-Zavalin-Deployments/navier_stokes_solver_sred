# file: tests/step1/test_step1_orchestrator.py
import numpy as np
import pytest

from src.step1.orchestrate_step1 import orchestrate_step1_state


# ---------------------------------------------------------------------
# Helper: minimal valid JSON input for Step 1
# ---------------------------------------------------------------------
def make_minimal_input(**overrides):
    base = {
        "domain_definition": {
            "nx": 4, "ny": 4, "nz": 4,
            "x_min": 0.0, "x_max": 1.0,
            "y_min": 0.0, "y_max": 1.0,
            "z_min": 0.0, "z_max": 1.0,
        },
        "fluid_properties": {
            "density": 1.0,
            "viscosity": 0.01,
        },
        "simulation_parameters": {
            "time_step": 0.1,
        },
        "initial_conditions": {
            "initial_pressure": 0.0,
            "initial_velocity": [0.0, 0.0, 0.0],
        },
        "geometry_definition": {
            "geometry_mask_flat": [0] * 64,
            "geometry_mask_shape": [4, 4, 4],
            "mask_encoding": {"fluid": 0, "solid": 1},
            "flattening_order": "C",
        },
        "boundary_conditions": [],
        "external_forces": {
            "force_vector": [0.0, 0.0, 0.0],
        },
    }
    base.update(overrides)
    return base


# =====================================================================
# 1. GEOMETRY MASK STRUCTURAL TESTS
# =====================================================================

def test_geometry_mask_perfect_reshape():
    json_input = make_minimal_input()
    state = orchestrate_step1_state(json_input)

    assert state.mask.shape == (4, 4, 4)
    assert state.mask.dtype == np.int_


def test_geometry_mask_length_mismatch_raises():
    bad_geom = {
        "geometry_definition": {
            "geometry_mask_flat": [0] * 63,  # should be 64
            "geometry_mask_shape": [4, 4, 4],
            "mask_encoding": {"fluid": 0, "solid": 1},
            "flattening_order": "C",
        }
    }
    json_input = make_minimal_input(**bad_geom)

    with pytest.raises(ValueError):
        orchestrate_step1_state(json_input)


def test_geometry_mask_type_pollution_raises():
    polluted = [0] * 63 + ["1"]  # non-int
    bad_geom = {
        "geometry_definition": {
            "geometry_mask_flat": polluted,
            "geometry_mask_shape": [4, 4, 4],
            "mask_encoding": {"fluid": 0, "solid": 1},
            "flattening_order": "C",
        }
    }
    json_input = make_minimal_input(**bad_geom)

    with pytest.raises(ValueError):
        orchestrate_step1_state(json_input)


def test_geometry_mask_opaque_labels_are_accepted():
    opaque = list(range(64))  # arbitrary integers
    geom = {
        "geometry_definition": {
            "geometry_mask_flat": opaque,
            "geometry_mask_shape": [4, 4, 4],
            "mask_encoding": {"fluid": 0, "solid": 1},
            "flattening_order": "C",
        }
    }
    json_input = make_minimal_input(**geom)
    state = orchestrate_step1_state(json_input)

    assert np.array_equal(state.mask.flatten(), np.array(opaque, dtype=int))


def test_geometry_mask_flattening_order_round_trip_C():
    nx = ny = nz = 4
    flat = list(range(nx * ny * nz))
    geom = {
        "geometry_definition": {
            "geometry_mask_flat": flat,
            "geometry_mask_shape": [nx, ny, nz],
            "mask_encoding": {"fluid": 0, "solid": 1},
            "flattening_order": "C",
        }
    }
    json_input = make_minimal_input(**geom)
    state = orchestrate_step1_state(json_input)

    mask = state.mask
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                idx = i + nx * (j + ny * k)
                assert mask[i, j, k] == flat[idx]


# =====================================================================
# 2. FIELD ALLOCATION TESTS (CELL-CENTERED)
# =====================================================================

def test_cell_centered_field_shapes():
    json_input = make_minimal_input()
    json_input["domain_definition"]["nx"] = 10
    json_input["domain_definition"]["ny"] = 20
    json_input["domain_definition"]["nz"] = 30
    json_input["geometry_definition"]["geometry_mask_flat"] = [0] * (10 * 20 * 30)
    json_input["geometry_definition"]["geometry_mask_shape"] = [10, 20, 30]

    state = orchestrate_step1_state(json_input)

    expected = (10, 20, 30)
    for name in ["P", "U", "V", "W", "Mask"]:
        arr = state.fields[name]
        assert isinstance(arr, np.ndarray)
        assert arr.shape == expected


def test_zero_or_negative_grid_counts_raise():
    json_input = make_minimal_input()
    json_input["domain_definition"]["nx"] = 0

    with pytest.raises(ValueError):
        orchestrate_step1_state(json_input)

    json_input["domain_definition"]["nx"] = 4
    json_input["domain_definition"]["ny"] = -5

    with pytest.raises(ValueError):
        orchestrate_step1_state(json_input)


# =====================================================================
# 3. BOUNDARY CONDITION STRUCTURAL TESTS
# =====================================================================

def test_bc_invalid_role_raises():
    json_input = make_minimal_input()
    json_input["boundary_conditions"] = [
        {"role": "rocket_engine", "faces": ["x_min"], "apply_to": []}
    ]

    with pytest.raises(ValueError):
        orchestrate_step1_state(json_input)


def test_bc_invalid_face_raises():
    json_input = make_minimal_input()
    json_input["boundary_conditions"] = [
        {"role": "inlet", "faces": ["diagonal_top"], "apply_to": []}
    ]

    with pytest.raises(ValueError):
        orchestrate_step1_state(json_input)


def test_bc_duplicate_face_raises():
    json_input = make_minimal_input()
    json_input["boundary_conditions"] = [
        {"role": "inlet", "faces": ["x_min"], "apply_to": []},
        {"role": "outlet", "faces": ["x_min"], "apply_to": []},
    ]

    with pytest.raises(ValueError):
        orchestrate_step1_state(json_input)


def test_bc_velocity_apply_to_consistency():
    json_input = make_minimal_input()
    json_input["boundary_conditions"] = [
        {"role": "inlet", "faces": ["x_min"], "apply_to": ["velocity"]}
    ]

    with pytest.raises(ValueError):
        orchestrate_step1_state(json_input)


def test_bc_pressure_apply_to_consistency():
    json_input = make_minimal_input()
    json_input["boundary_conditions"] = [
        {"role": "outlet", "faces": ["x_max"], "apply_to": ["pressure"]}
    ]

    with pytest.raises(ValueError):
        orchestrate_step1_state(json_input)


# =====================================================================
# 4. INITIAL CONDITION TESTS
# =====================================================================

def test_uniform_velocity_broadcast():
    json_input = make_minimal_input()
    json_input["initial_conditions"]["initial_velocity"] = [1.0, 0.0, -0.5]

    state = orchestrate_step1_state(json_input)

    assert np.all(state.fields["U"] == 1.0)
    assert np.all(state.fields["V"] == 0.0)
    assert np.all(state.fields["W"] == -0.5)


def test_non_finite_velocity_raises():
    json_input = make_minimal_input()
    json_input["initial_conditions"]["initial_velocity"] = [float("nan"), 1.0, 0.0]

    with pytest.raises(ValueError):
        orchestrate_step1_state(json_input)

    json_input["initial_conditions"]["initial_velocity"] = [float("inf"), 0.0, 0.0]

    with pytest.raises(ValueError):
        orchestrate_step1_state(json_input)


# =====================================================================
# 5. GRID & SIMULATION PARAMETER VALIDATION
# =====================================================================

def test_invalid_domain_extents_raise():
    json_input = make_minimal_input()
    json_input["domain_definition"]["x_max"] = json_input["domain_definition"]["x_min"]

    with pytest.raises(ValueError):
        orchestrate_step1_state(json_input)


def test_invalid_time_step_raises():
    json_input = make_minimal_input()
    json_input["simulation_parameters"]["time_step"] = 0.0

    with pytest.raises(ValueError):
        orchestrate_step1_state(json_input)