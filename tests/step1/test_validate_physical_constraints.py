# tests/step1/test_validate_physical_constraints.py

import pytest
from src.step1.validate_physical_constraints import validate_physical_constraints


def base_json():
    return {
        "domain_definition": {
            "nx": 2, "ny": 2, "nz": 2,
            "x_min": 0, "x_max": 1,
            "y_min": 0, "y_max": 1,
            "z_min": 0, "z_max": 1,
        },
        "fluid_properties": {
            "density": 1.0,
            "viscosity": 0.1,
        },
        "simulation_parameters": {
            "time_step": 0.01,
        },
        "initial_conditions": {
            "initial_pressure": 1.0,
            "initial_velocity": [0, 0, 0],
        },
        "geometry_definition": {
            "geometry_mask_flat": [1] * 8,   # valid mask values ∈ {-1, 0, 1}
            "geometry_mask_shape": [2, 2, 2],
            "mask_encoding": {"fluid": 1, "solid": -1},
            "flattening_order": "i + nx*(j + ny*k)",
        },
        "boundary_conditions": [],
    }


def test_density_must_be_positive():
    data = base_json()
    data["fluid_properties"]["density"] = 0
    with pytest.raises(ValueError):
        validate_physical_constraints(data)


def test_mask_length_mismatch():
    data = base_json()
    data["geometry_definition"]["geometry_mask_flat"] = [1] * 7
    with pytest.raises(ValueError):
        validate_physical_constraints(data)
