# tests/step1/test_validate_physical_constraints_full.py

import pytest
from src.step1.validate_physical_constraints import validate_physical_constraints


def base():
    return {
        "domain_definition": {
            "nx": 2, "ny": 2, "nz": 2,
            "x_min": 0, "x_max": 1,
            "y_min": 0, "y_max": 1,
            "z_min": 0, "z_max": 1,
        },
        "fluid_properties": {"density": 1.0, "viscosity": 0.1},
        "simulation_parameters": {"time_step": 0.01},
        "initial_conditions": {"initial_pressure": 1.0, "initial_velocity": [0, 0, 0]},
        "geometry_definition": {
            "geometry_mask_flat": [0]*8,
            "geometry_mask_shape": [2,2,2],
            "flattening_order": "i + nx*(j + ny*k)",
        },
        "boundary_conditions": [],
    }


def test_negative_viscosity():
    d = base()
    d["fluid_properties"]["viscosity"] = -1
    with pytest.raises(ValueError):
        validate_physical_constraints(d)


def test_zero_density():
    d = base()
    d["fluid_properties"]["density"] = 0
    with pytest.raises(ValueError):
        validate_physical_constraints(d)


def test_invalid_time_step():
    d = base()
    d["simulation_parameters"]["time_step"] = -0.1
    with pytest.raises(ValueError):
        validate_physical_constraints(d)


def test_invalid_mask_shape():
    d = base()
    d["geometry_definition"]["geometry_mask_shape"] = [2, 2, 3]
    with pytest.raises(ValueError):
        validate_physical_constraints(d)


def test_invalid_mask_length():
    d = base()
    d["geometry_definition"]["geometry_mask_flat"] = [0]*7
    with pytest.raises(ValueError):
        validate_physical_constraints(d)
