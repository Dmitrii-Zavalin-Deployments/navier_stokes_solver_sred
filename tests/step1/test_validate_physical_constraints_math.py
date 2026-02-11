# tests/step1/test_validate_physical_constraints_math.py

import pytest
from src.step1.validate_physical_constraints import validate_physical_constraints


# ---------------------------------------------------------
# Helper: minimal valid config
# ---------------------------------------------------------

def make_valid():
    return {
        "domain_definition": {
            "nx": 2, "ny": 2, "nz": 2,
            "x_min": 0.0, "x_max": 2.0,
            "y_min": 0.0, "y_max": 2.0,
            "z_min": 0.0, "z_max": 2.0,
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
            "geometry_mask_flat": [0] * 8,
            "geometry_mask_shape": [2, 2, 2],
            "mask_encoding": {"fluid": 1, "solid": -1},
            "flattening_order": "C",
        },
        "simulation_parameters": {
            "time_step": 0.1,
        },
        "external_forces": {
            "force_vector": [0.0, 0.0, 0.0],
        },
    }


# ---------------------------------------------------------
# Fluid properties
# ---------------------------------------------------------

def test_density_must_be_positive():
    cfg = make_valid()
    cfg["fluid_properties"]["density"] = 0.0
    with pytest.raises(ValueError):
        validate_physical_constraints(cfg)


def test_viscosity_must_be_non_negative():
    cfg = make_valid()
    cfg["fluid_properties"]["viscosity"] = -1.0
    with pytest.raises(ValueError):
        validate_physical_constraints(cfg)


# ---------------------------------------------------------
# Grid counts
# ---------------------------------------------------------

def test_grid_counts_must_be_positive_ints():
    cfg = make_valid()
    cfg["domain_definition"]["nx"] = 0
    with pytest.raises(ValueError):
        validate_physical_constraints(cfg)


# ---------------------------------------------------------
# Domain extents
# ---------------------------------------------------------

def test_domain_extents_must_be_ordered():
    cfg = make_valid()
    cfg["domain_definition"]["x_max"] = 0.0
    with pytest.raises(ValueError):
        validate_physical_constraints(cfg)


def test_domain_extents_must_be_finite():
    cfg = make_valid()
    cfg["domain_definition"]["x_min"] = float("inf")
    with pytest.raises(ValueError):
        validate_physical_constraints(cfg)


# ---------------------------------------------------------
# Geometry mask shape consistency
# ---------------------------------------------------------

def test_geometry_mask_shape_must_have_length_3():
    cfg = make_valid()
    cfg["geometry_definition"]["geometry_mask_shape"] = [2, 2]
    with pytest.raises(ValueError):
        validate_physical_constraints(cfg)


def test_geometry_mask_shape_must_match_grid():
    cfg = make_valid()
    cfg["geometry_definition"]["geometry_mask_shape"] = [3, 2, 2]  # 12 != 8
    with pytest.raises(ValueError):
        validate_physical_constraints(cfg)


def test_geometry_mask_flat_length_must_match():
    cfg = make_valid()
    cfg["geometry_definition"]["geometry_mask_flat"] = [0] * 7  # should be 8
    with pytest.raises(ValueError):
        validate_physical_constraints(cfg)


def test_geometry_mask_entries_must_be_valid():
    cfg = make_valid()
    cfg["geometry_definition"]["geometry_mask_flat"][0] = 9  # invalid
    with pytest.raises(ValueError):
        validate_physical_constraints(cfg)


# ---------------------------------------------------------
# mask_encoding
# ---------------------------------------------------------

def test_mask_encoding_must_be_dict():
    cfg = make_valid()
    cfg["geometry_definition"]["mask_encoding"] = "not_a_dict"
    with pytest.raises(ValueError):
        validate_physical_constraints(cfg)


def test_mask_encoding_requires_fluid_and_solid():
    cfg = make_valid()
    del cfg["geometry_definition"]["mask_encoding"]["fluid"]
    with pytest.raises(ValueError):
        validate_physical_constraints(cfg)


def test_mask_encoding_values_must_be_ints():
    cfg = make_valid()
    cfg["geometry_definition"]["mask_encoding"]["fluid"] = 1.5
    with pytest.raises(ValueError):
        validate_physical_constraints(cfg)


# ---------------------------------------------------------
# flattening_order
# ---------------------------------------------------------

def test_flattening_order_must_be_string():
    cfg = make_valid()
    cfg["geometry_definition"]["flattening_order"] = 123
    with pytest.raises(ValueError):
        validate_physical_constraints(cfg)


# ---------------------------------------------------------
# Initial velocity
# ---------------------------------------------------------

def test_initial_velocity_must_have_three_components():
    cfg = make_valid()
    cfg["initial_conditions"]["initial_velocity"] = [0.0, 0.0]
    with pytest.raises(ValueError):
        validate_physical_constraints(cfg)


def test_initial_velocity_components_must_be_finite():
    cfg = make_valid()
    cfg["initial_conditions"]["initial_velocity"][1] = float("inf")
    with pytest.raises(ValueError):
        validate_physical_constraints(cfg)


# ---------------------------------------------------------
# Initial pressure
# ---------------------------------------------------------

def test_initial_pressure_must_be_finite():
    cfg = make_valid()
    cfg["initial_conditions"]["initial_pressure"] = float("nan")
    with pytest.raises(ValueError):
        validate_physical_constraints(cfg)


# ---------------------------------------------------------
# Time step
# ---------------------------------------------------------

def test_time_step_must_be_positive():
    cfg = make_valid()
    cfg["simulation_parameters"]["time_step"] = 0.0
    with pytest.raises(ValueError):
        validate_physical_constraints(cfg)


# ---------------------------------------------------------
# External forces
# ---------------------------------------------------------

def test_external_forces_must_be_dict():
    cfg = make_valid()
    cfg["external_forces"] = "not_a_dict"
    with pytest.raises(ValueError):
        validate_physical_constraints(cfg)


def test_force_vector_must_exist():
    cfg = make_valid()
    del cfg["external_forces"]["force_vector"]
    with pytest.raises(ValueError):
        validate_physical_constraints(cfg)


def test_force_vector_must_be_length_3():
    cfg = make_valid()
    cfg["external_forces"]["force_vector"] = [1, 2]
    with pytest.raises(ValueError):
        validate_physical_constraints(cfg)


def test_force_vector_components_must_be_finite():
    cfg = make_valid()
    cfg["external_forces"]["force_vector"][2] = float("inf")
    with pytest.raises(ValueError):
        validate_physical_constraints(cfg)
