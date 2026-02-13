# tests/step2/test_step2_schema_output.py

import json
from pathlib import Path

from src.step2.orchestrate_step2 import orchestrate_step2


def load_schema(path: str):
    with open(path, "r") as f:
        return json.load(f)


def validate_json_schema(instance, schema):
    from jsonschema import validate
    validate(instance=instance, schema=schema)


def make_minimal_step2_input():
    """
    Minimal valid Step‑1 output that Step‑2 can accept.
    Must match step1_output_schema.json exactly.
    """

    return {
        "grid": {
            "x_min": 0.0, "x_max": 1.0,
            "y_min": 0.0, "y_max": 1.0,
            "z_min": 0.0, "z_max": 1.0,
            "nx": 1, "ny": 1, "nz": 1,
            "dx": 1.0, "dy": 1.0, "dz": 1.0,
        },

        "fields": {
            "P": [[[0.0]]],          # (1,1,1)
            "U": [[[0.0]], [[0.0]]], # (2,1,1)
            "V": [[[0.0], [0.0]]],   # (1,2,1)
            "W": [[[0.0, 0.0]]],     # (1,1,2)
            "Mask": [[[1]]],         # (1,1,1)
        },

        "mask_3d": [[[1]]],

        "boundary_table": {
            "x_min": [],
            "x_max": [],
            "y_min": [],
            "y_max": [],
            "z_min": [],
            "z_max": [],
        },

        "constants": {
            "rho": 1.0,
            "mu": 1.0,
            "dt": 0.1,
            "dx": 1.0,
            "dy": 1.0,
            "dz": 1.0,
            "inv_dx": 1.0,
            "inv_dy": 1.0,
            "inv_dz": 1.0,
            "inv_dx2": 1.0,
            "inv_dy2": 1.0,
            "inv_dz2": 1.0,
        },

        "config": {
            "domain": {"nx": 1, "ny": 1, "nz": 1},
            "fluid": {"density": 1.0, "viscosity": 0.1},
            "simulation": {"time_step": 0.1},
            "forces": {"force_vector": [0.0, 0.0, 0.0]},
            "boundary_conditions": [],
            "geometry_definition": {
                "geometry_mask_flat": [1],
                "geometry_mask_shape": [1, 1, 1],
                "mask_encoding": {"fluid": 1, "solid": -1},
                "flattening_order": "C",
            },
        },

        "state_as_dict": {}
    }


def test_step2_output_matches_schema():
    """
    Contract test:
    Step‑2 output MUST validate against step2_output_schema.json.
    """

    schema_path = (
        Path(__file__).resolve().parents[2] / "schema" / "step2_output_schema.json"
    )
    schema = load_schema(str(schema_path))

    state_in = make_minimal_step2_input()

    state_out = orchestrate_step2(
        state_in,
        validate_json_schema=validate_json_schema,
        load_schema=load_schema,
    )

    validate_json_schema(state_out, schema)
