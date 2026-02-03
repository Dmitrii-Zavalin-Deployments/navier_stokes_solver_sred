import pytest
import json
from src.step1.schema_validator import validate_input_schema


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_valid_json():
    """Return a minimal valid JSON input that matches the schema."""
    return {
        "domain_definition": {
            "x_min": 0.0,
            "x_max": 1.0,
            "y_min": 0.0,
            "y_max": 1.0,
            "z_min": 0.0,
            "z_max": 1.0,
            "nx": 4,
            "ny": 4,
            "nz": 4
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
            "mask_encoding": {
                "fluid": 1,
                "solid": 0
            },
            "flattening_order": "i + nx*(j + ny*k)"
        },
        "external_forces": {
            "force_vector": [0.0, 0.0, 0.0],
            "units": "N",
            "comment": "none"
        }
    }


# ---------------------------------------------------------------------------
# 1. Valid input should pass schema validation
# ---------------------------------------------------------------------------

def test_schema_accepts_valid_input():
    data = load_valid_json()
    validate_input_schema(data)  # should not raise


# ---------------------------------------------------------------------------
# 2. Missing top-level sections should fail
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("missing_key", [
    "domain_definition",
    "fluid_properties",
    "initial_conditions",
    "simulation_parameters",
    "boundary_conditions",
    "geometry_definition",
    "external_forces",
])
def test_missing_top_level_key_fails(missing_key):
    data = load_valid_json()
    del data[missing_key]
    with pytest.raises(Exception):
        validate_input_schema(data)


# ---------------------------------------------------------------------------
# 3. Wrong types should fail
# ---------------------------------------------------------------------------

def test_wrong_type_in_domain_definition():
    data = load_valid_json()
    data["domain_definition"]["nx"] = "not_an_int"
    with pytest.raises(Exception):
        validate_input_schema(data)


def test_wrong_type_in_initial_velocity():
    data = load_valid_json()
    data["initial_conditions"]["initial_velocity"] = ["a", "b", "c"]
    with pytest.raises(Exception):
        validate_input_schema(data)


def test_wrong_type_in_boundary_conditions():
    data = load_valid_json()
    data["boundary_conditions"][0]["no_slip"] = "not_bool"
    with pytest.raises(Exception):
        validate_input_schema(data)


# ---------------------------------------------------------------------------
# 4. Wrong geometry mask shape should fail
# ---------------------------------------------------------------------------

def test_geometry_mask_shape_mismatch():
    data = load_valid_json()
    data["geometry_definition"]["geometry_mask_shape"] = [2, 2, 2]  # wrong
    with pytest.raises(Exception):
        validate_input_schema(data)


# ---------------------------------------------------------------------------
# 5. Wrong flattening order type should fail
# ---------------------------------------------------------------------------

def test_invalid_flattening_order_type():
    data = load_valid_json()
    data["geometry_definition"]["flattening_order"] = 123  # must be string
    with pytest.raises(Exception):
        validate_input_schema(data)
