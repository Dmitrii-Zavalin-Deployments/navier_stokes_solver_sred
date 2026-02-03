import pytest
from jsonschema import ValidationError
from src.step1.schema_validator import validate_input_schema


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
# 1. Valid input must pass schema validation
# ---------------------------------------------------------------------------

def test_schema_accepts_valid_input(valid_json):
    validate_input_schema(valid_json)  # should not raise


# ---------------------------------------------------------------------------
# 2. Missing required top-level keys must fail
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
def test_missing_top_level_key_fails(valid_json, missing_key):
    data = valid_json.copy()
    del data[missing_key]
    with pytest.raises(ValidationError):
        validate_input_schema(data)


# ---------------------------------------------------------------------------
# 3. Wrong types must fail
# ---------------------------------------------------------------------------

def test_invalid_nx_type(valid_json):
    valid_json["domain_definition"]["nx"] = "four"
    with pytest.raises(ValidationError):
        validate_input_schema(valid_json)


def test_invalid_velocity_length(valid_json):
    valid_json["initial_conditions"]["initial_velocity"] = [1.0, 2.0]
    with pytest.raises(ValidationError):
        validate_input_schema(valid_json)


def test_invalid_force_vector_length(valid_json):
    valid_json["external_forces"]["force_vector"] = [0.0, 0.0]
    with pytest.raises(ValidationError):
        validate_input_schema(valid_json)


# ---------------------------------------------------------------------------
# 4. Missing flattening_order must fail
# ---------------------------------------------------------------------------

def test_missing_flattening_order(valid_json):
    del valid_json["geometry_definition"]["flattening_order"]
    with pytest.raises(ValidationError):
        validate_input_schema(valid_json)


# ---------------------------------------------------------------------------
# 5. Geometry mask shape mismatch must fail
# ---------------------------------------------------------------------------

def test_geometry_mask_shape_mismatch(valid_json):
    valid_json["geometry_definition"]["geometry_mask_shape"] = [2, 2, 2]
    with pytest.raises(ValidationError):
        validate_input_schema(valid_json)
