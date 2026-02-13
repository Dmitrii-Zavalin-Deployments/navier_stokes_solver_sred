# tests/step1/test_step1_schema_output.py

import json
from pathlib import Path

from src.step1.orchestrate_step1 import orchestrate_step1


def load_schema(path: str):
    with open(path, "r") as f:
        return json.load(f)


def validate_json_schema(instance, schema):
    from jsonschema import validate
    validate(instance=instance, schema=schema)


def make_minimal_step1_input():
    """
    Minimal valid input for Step 1.
    Matches schema/input_schema.json exactly.
    """
    return {
        "domain_definition": {
            "x_min": 0.0, "x_max": 1.0,
            "y_min": 0.0, "y_max": 1.0,
            "z_min": 0.0, "z_max": 1.0,
            "nx": 2, "ny": 2, "nz": 2
        },

        "fluid_properties": {
            "density": 1.0,
            "viscosity": 0.1
        },

        "initial_conditions": {
            "initial_velocity": [0.0, 0.0, 0.0],
            "initial_pressure": 0.0
        },

        "simulation_parameters": {
            "time_step": 0.1,
            "total_time": 1.0,
            "output_interval": 1
        },

        "boundary_conditions": [
            {
                "role": "wall",
                "type": "dirichlet",
                "faces": ["x_min"],
                "apply_to": ["velocity", "pressure"],
                "velocity": [0.0, 0.0, 0.0],
                "pressure": 0.0,
                "pressure_gradient": 0.0,
                "no_slip": True,
                "comment": "minimal BC"
            }
        ],

        "geometry_definition": {
            "geometry_mask_flat": [1, 1, 1, 1, 1, 1, 1, 1],
            "geometry_mask_shape": [2, 2, 2],
            "mask_encoding": {"fluid": 1, "solid": -1},
            "flattening_order": "C"
        },

        "external_forces": {
            "force_vector": [0.0, 0.0, 0.0],
            "units": "N",
            "comment": "no external forces"
        }
    }


def test_step1_output_matches_schema():
    """
    Contract test:
    Step‑1 output MUST validate against step1_output_schema.json.
    """

    schema_path = (
        Path(__file__).resolve().parents[2] / "schema" / "step1_output_schema.json"
    )
    schema = load_schema(str(schema_path))

    state_in = make_minimal_step1_input()

    state_out = orchestrate_step1(
        state_in,
        validate_json_schema=validate_json_schema,
        load_schema=load_schema,
    )

    validate_json_schema(state_out, schema)
