import pytest
import numpy as np
from src.step1.construct_simulation_state import construct_simulation_state


@pytest.fixture
def valid_json():
    """Baseline valid JSON input for Step 1 (schema-compliant)."""
    return {
        "domain_definition": {
            "x_min": 0.0, "x_max": 1.0,
            "y_min": 0.0, "y_max": 1.0,
            "z_min": 0.0, "z_max": 1.0,
            "nx": 4, "ny": 4, "nz": 4
        },
        "fluid_properties": {
            "density": 1.0,
            "viscosity": 0.1
        },
        "initial_conditions": {
            "initial_velocity": [1.0, 0.0, 0.0],
            "initial_pressure": 0.0
        },
        "simulation_parameters": {
            "time_step": 0.01,
            "total_time": 1.0,
            "output_interval": 10
        },
        "boundary_conditions": [
            {
                "role": "wall",
                "type": "dirichlet",
                "faces": ["x_min"],
                "apply_to": ["velocity"],
                "velocity": [0.0, 0.0, 0.0],
                "pressure": 0.0,
                "pressure_gradient": 0.0,
                "no_slip": True,
                "comment": "test"
            }
        ],
        "geometry_definition": {
            "geometry_mask_flat": [1] * (4 * 4 * 4),
            "geometry_mask_shape": [4, 4, 4],
            "mask_encoding": {"fluid": 1, "solid": 0},
            "flattening_order": "i + nx*(j + ny*k)"
        },
        "external_forces": {
            "force_vector": [0.0, 0.0, 0.0],
            "units": "N",
            "comment": "none"
        }
    }


# ---------------------------------------------------------------------------
# 1. Mask reshaping (flattening order)
# ---------------------------------------------------------------------------

def test_mask_reshape_default_order(valid_json):
    nx, ny, nz = 4, 4, 4
    flat = list(range(nx * ny * nz))
    valid_json["geometry_definition"]["geometry_mask_flat"] = flat

    state = construct_simulation_state(valid_json)
    assert state.mask.shape == (nx, ny, nz)


def test_mask_reshape_custom_order(valid_json):
    valid_json["geometry_definition"]["flattening_order"] = "j + ny*(i + nx*k)"

    nx, ny, nz = 4, 4, 4
    flat = list(range(nx * ny * nz))
    valid_json["geometry_definition"]["geometry_mask_flat"] = flat

    state = construct_simulation_state(valid_json)
    assert state.mask.shape == (nx, ny, nz)


# ---------------------------------------------------------------------------
# 2. Mask values must be valid
# ---------------------------------------------------------------------------

def test_invalid_mask_value_negative(valid_json):
    valid_json["geometry_definition"]["geometry_mask_flat"][0] = -1
    with pytest.raises(Exception):
        construct_simulation_state(valid_json)


def test_invalid_mask_value_large(valid_json):
    valid_json["geometry_definition"]["geometry_mask_flat"][0] = 5
    with pytest.raises(Exception):
        construct_simulation_state(valid_json)


# ---------------------------------------------------------------------------
# 3. Fully solid or fully fluid masks must be accepted
# ---------------------------------------------------------------------------

def test_mask_all_fluid(valid_json):
    valid_json["geometry_definition"]["geometry_mask_flat"] = [1] * (4 * 4 * 4)
    state = construct_simulation_state(valid_json)
    assert state.mask.shape == (4, 4, 4)


def test_mask_all_solid(valid_json):
    valid_json["geometry_definition"]["geometry_mask_flat"] = [0] * (4 * 4 * 4)
    state = construct_simulation_state(valid_json)
    assert state.mask.shape == (4, 4, 4)


# ---------------------------------------------------------------------------
# 4. Minimal grid mask
# ---------------------------------------------------------------------------

def test_minimal_grid_mask():
    json_input = {
        "domain_definition": {
            "x_min": 0.0, "x_max": 1.0,
            "y_min": 0.0, "y_max": 1.0,
            "z_min": 0.0, "z_max": 1.0,
            "nx": 1, "ny": 1, "nz": 1
        },
        "fluid_properties": {"density": 1.0, "viscosity": 0.1},
        "initial_conditions": {
            "initial_velocity": [0.0, 0.0, 0.0],
            "initial_pressure": 0.0
        },
        "simulation_parameters": {
            "time_step": 0.01,
            "total_time": 1.0,
            "output_interval": 10
        },
        "boundary_conditions": [],
        "geometry_definition": {
            "geometry_mask_flat": [1],
            "geometry_mask_shape": [1, 1, 1],
            "mask_encoding": {"fluid": 1, "solid": 0},
            "flattening_order": "i + nx*(j + ny*k)"
        },
        "external_forces": {
            "force_vector": [0.0, 0.0, 0.0],
            "units": "N",
            "comment": "none"
        }
    }

    state = construct_simulation_state(json_input)
    assert state.mask.shape == (1, 1, 1)
