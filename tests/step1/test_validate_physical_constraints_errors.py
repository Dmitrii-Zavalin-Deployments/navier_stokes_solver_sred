# tests/step1/test_validate_physical_constraints_errors.py

import pytest
from src.step1.validate_physical_constraints import validate_physical_constraints


def base_valid_state():
    return {
        "domain_definition": {
            "nx": 2, "ny": 2, "nz": 2,
            "x_min": 0.0, "x_max": 1.0,
            "y_min": 0.0, "y_max": 1.0,
            "z_min": 0.0, "z_max": 1.0,
        },
        "fluid_properties": {
            "density": 1.0,
            "viscosity": 0.1,
        },
        "initial_conditions": {
            "initial_velocity": [0.0, 0.0, 0.0],
            "initial_pressure": 0.0,
        },
        "geometry_definition": {
            "geometry_mask_shape": (2, 2, 2),
            "geometry_mask_flat": [0] * 8,
        },
        "simulation_parameters": {
            "time_step": 0.1,
        },
    }


def test_density_negative_rejected():
    data = base_valid_state()
    data["fluid_properties"]["density"] = 0.0
    with pytest.raises(ValueError):
        validate_physical_constraints(data)


def test_viscosity_negative_rejected():
    data = base_valid_state()
    data["fluid_properties"]["viscosity"] = -1.0
    with pytest.raises(ValueError):
        validate_physical_constraints(data)


def test_geometry_shape_wrong_length():
    data = base_valid_state()
    data["geometry_definition"]["geometry_mask_shape"] = (2, 2)
    with pytest.raises(ValueError):
        validate_physical_constraints(data)


def test_geometry_shape_mismatch():
    data = base_valid_state()
    data["geometry_definition"]["geometry_mask_shape"] = (3, 3, 3)
    with pytest.raises(ValueError):
        validate_physical_constraints(data)


def test_geometry_flat_length_mismatch():
    data = base_valid_state()
    data["geometry_definition"]["geometry_mask_flat"] = [0] * 7
    with pytest.raises(ValueError):
        validate_physical_constraints(data)


def test_initial_velocity_wrong_length():
    data = base_valid_state()
    data["initial_conditions"]["initial_velocity"] = [0.0, 0.0]
    with pytest.raises(ValueError):
        validate_physical_constraints(data)


def test_initial_pressure_non_finite():
    data = base_valid_state()
    data["initial_conditions"]["initial_pressure"] = float("inf")
    with pytest.raises(ValueError):
        validate_physical_constraints(data)


def test_cfl_precheck_failure():
    data = base_valid_state()
    data["initial_conditions"]["initial_velocity"] = [10.0, 0.0, 0.0]  # large velocity
    data["simulation_parameters"]["time_step"] = 1.0  # large dt
    with pytest.raises(ValueError):
        validate_physical_constraints(data)
