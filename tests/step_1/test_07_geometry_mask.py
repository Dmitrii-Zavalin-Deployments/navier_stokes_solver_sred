import pytest
import numpy as np
from src.step1.construct_simulation_state import construct_simulation_state


@pytest.fixture
def valid_json():
    """Baseline valid JSON input for Step 1."""
    return {
        "domain": {
            "x_min": 0.0, "x_max": 1.0,
            "y_min": 0.0, "y_max": 1.0,
            "z_min": 0.0, "z_max": 1.0,
            "nx": 4, "ny": 4, "nz": 4
        },
        "fluid": {
            "density": 1.0,
            "viscosity": 0.1
        },
        "simulation": {
            "dt": 0.01,
            "total_time": 1.0,
            "flattening_order": "i + nx*(j + ny*k)",
            "initial_pressure": 0.0,
            "initial_velocity": [1.0, 0.0, 0.0],
            "force_vector": [0.0, 0.0, 0.0]
        },
        "geometry_mask_flat": [1] * (4 * 4 * 4),
        "boundary_conditions": [
            {"face": "x_min", "role": "wall"},
            {"face": "x_max", "role": "outlet"},
            {"face": "y_min", "role": "wall"},
            {"face": "y_max", "role": "wall"},
            {"face": "z_min", "role": "wall"},
            {"face": "z_max", "role": "wall"}
        ]
    }


# ---------------------------------------------------------------------------
# 1. Mask reshaping (flattening order)
# ---------------------------------------------------------------------------

def test_mask_reshape_default_order(valid_json):
    # Create a mask with a known pattern
    nx, ny, nz = 4, 4, 4
    flat = list(range(nx * ny * nz))
    valid_json["geometry_mask_flat"] = flat

    state = construct_simulation_state(valid_json)

    # Stub: mask is zero-initialized, so this will fail later
    assert state.mask.shape == (nx, ny, nz)


def test_mask_reshape_custom_order(valid_json):
    valid_json["simulation"]["flattening_order"] = "j + ny*(i + nx*k)"

    nx, ny, nz = 4, 4, 4
    flat = list(range(nx * ny * nz))
    valid_json["geometry_mask_flat"] = flat

    state = construct_simulation_state(valid_json)

    # Stub: mask is zero-initialized
    assert state.mask.shape == (nx, ny, nz)


# ---------------------------------------------------------------------------
# 2. Mask values must be 0 or 1
# ---------------------------------------------------------------------------

def test_invalid_mask_value_negative(valid_json):
    valid_json["geometry_mask_flat"][0] = -1
    with pytest.raises(Exception):
        construct_simulation_state(valid_json)


def test_invalid_mask_value_large(valid_json):
    valid_json["geometry_mask_flat"][0] = 5
    with pytest.raises(Exception):
        construct_simulation_state(valid_json)


# ---------------------------------------------------------------------------
# 3. Fully solid or fully fluid masks must be accepted
# ---------------------------------------------------------------------------

def test_mask_all_fluid(valid_json):
    valid_json["geometry_mask_flat"] = [1] * (4 * 4 * 4)
    state = construct_simulation_state(valid_json)
    assert state.mask.shape == (4, 4, 4)


def test_mask_all_solid(valid_json):
    valid_json["geometry_mask_flat"] = [0] * (4 * 4 * 4)
    state = construct_simulation_state(valid_json)
    assert state.mask.shape == (4, 4, 4)


# ---------------------------------------------------------------------------
# 4. Minimal grid mask
# ---------------------------------------------------------------------------

def test_minimal_grid_mask():
    json_input = {
        "domain": {
            "x_min": 0.0, "x_max": 1.0,
            "y_min": 0.0, "y_max": 1.0,
            "z_min": 0.0, "z_max": 1.0,
            "nx": 1, "ny": 1, "nz": 1
        },
        "fluid": {"density": 1.0, "viscosity": 0.1},
        "simulation": {
            "dt": 0.01,
            "total_time": 1.0,
            "flattening_order": "i + nx*(j + ny*k)",
            "initial_pressure": 0.0,
            "initial_velocity": [0.0, 0.0, 0.0],
            "force_vector": [0.0, 0.0, 0.0]
        },
        "geometry_mask_flat": [1],
        "boundary_conditions": []
    }

    state = construct_simulation_state(json_input)
    assert state.mask.shape == (1, 1, 1)
