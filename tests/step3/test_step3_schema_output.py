# tests/step3/test_step3_schema_output.py

import json
from pathlib import Path

from src.step3.orchestrate_step3 import orchestrate_step3


def load_schema(path: str):
    with open(path, "r") as f:
        return json.load(f)


def validate_json_schema(instance, schema):
    from jsonschema import validate
    validate(instance=instance, schema=schema)


def make_minimal_step3_input():
    """
    Minimal valid Step‑2 output that Step‑3 can accept.
    Must match step2_output_schema.json exactly.
    """

    # Grid: 1×1×1 domain
    grid = {
        "x_min": 0.0, "x_max": 1.0,
        "y_min": 0.0, "y_max": 1.0,
        "z_min": 0.0, "z_max": 1.0,
        "nx": 1, "ny": 1, "nz": 1,
        "dx": 1.0, "dy": 1.0, "dz": 1.0,
    }

    # Staggered fields for nx=1, ny=1, nz=1
    fields = {
        "P": [[[0.0]]],            # (1,1,1)
        "U": [[[0.0]], [[0.0]]],   # (2,1,1)
        "V": [[[0.0], [0.0]]],     # (1,2,1)
        "W": [[[0.0, 0.0]]],       # (1,1,2)
        "Mask": [[[1]]],           # (1,1,1)
    }

    # Mask
    mask = [[[1]]]

    # Fluid mask
    is_fluid = [[[True]]]

    # Boundary mask
    is_boundary_cell = [[[False]]]

    # Constants (from Step‑2)
    constants = {
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
    }

    # Operators (placeholders)
    operators = {
        "divergence": "div",
        "gradient_p_x": "gpx",
        "gradient_p_y": "gpy",
        "gradient_p_z": "gpz",
        "laplacian_u": "lu",
        "laplacian_v": "lv",
        "laplacian_w": "lw",
        "advection_u": "au",
        "advection_v": "av",
        "advection_w": "aw",
    }

    # PPE metadata
    ppe = {
        "rhs_builder": "rhs",
        "solver_type": "cg",
        "tolerance": 1e-6,
        "max_iterations": 50,
        "ppe_is_singular": False,
        "ppe_converged": True,
    }

    # Config (from Step‑1)
    config = {
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
    }

    # History (Step‑2 always provides this)
    history = {
        "times": [],
        "divergence_norms": [],
        "max_velocity_history": [],
        "ppe_iterations_history": [],
        "energy_history": [],
    }

    return {
        "grid": grid,
        "fields": fields,
        "mask": mask,
        "is_fluid": is_fluid,
        "is_boundary_cell": is_boundary_cell,
        "constants": constants,
        "operators": operators,
        "ppe": ppe,
        "config": config,
        "history": history,
    }


def test_step3_output_matches_schema():
    """
    Contract test:
    Step‑3 output MUST validate against step3_output_schema.json.
    """

    schema_path = (
        Path(__file__).resolve().parents[2] / "schema" / "step3_output_schema.json"
    )
    schema = load_schema(str(schema_path))

    state_in = make_minimal_step3_input()

    # Step‑3 signature: (state, current_time, step_index)
    state_out = orchestrate_step3(
        state_in,
        current_time=0.0,
        step_index=0,
        validate_json_schema=validate_json_schema,
        load_schema=load_schema,
    )

    # Step‑3 output is JSON‑compatible after _to_json_compatible()
    validate_json_schema(state_out, schema)
