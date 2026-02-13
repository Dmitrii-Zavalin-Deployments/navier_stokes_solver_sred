# tests/step3/test_step3_schema_output.py

import json
from pathlib import Path
import numpy as np

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
        "P": np.zeros((1, 1, 1)),
        "U": np.zeros((2, 1, 1)),
        "V": np.zeros((1, 2, 1)),
        "W": np.zeros((1, 1, 2)),
        "Mask": np.ones((1, 1, 1), dtype=int),
    }

    # Mask
    mask = np.ones((1, 1, 1), dtype=int)

    # Fluid mask
    is_fluid = np.ones((1, 1, 1), dtype=bool)

    # Boundary mask
    is_boundary_cell = np.zeros((1, 1, 1), dtype=bool)

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
        "divergence": {"op": lambda U, V, W: np.zeros((1, 1, 1))},
        "gradient_p_x": {"op": None},
        "gradient_p_y": {"op": None},
        "gradient_p_z": {"op": None},
        "laplacian_u": {"op": None},
        "laplacian_v": {"op": None},
        "laplacian_w": {"op": None},
        "advection_u": {"op": None},
        "advection_v": {"op": None},
        "advection_w": {"op": None},
    }

    # PPE metadata
    ppe = {
        "rhs_builder": {"op": None},
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

    # Step‑2 health block (REQUIRED)
    health = {
        "initial_divergence_norm": 0.0,
        "max_velocity_magnitude": 0.0,
        "cfl_advection_estimate": 0.0,
    }

    # Step‑2 history block (REQUIRED)
    history = {
        "times": [],
        "divergence_norms": [],
        "max_velocity_history": [],
        "ppe_iterations_history": [],
        "energy_history": [],
    }

    # Mask semantics (used by integration tests)
    mask_semantics = {
        "mask": mask,
        "is_fluid": is_fluid,
        "is_solid": (mask == 0),
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
        "health": health,
        "history": history,
        "mask_semantics": mask_semantics,
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

    state_out = orchestrate_step3(
        state_in,
        current_time=0.0,
        step_index=0,
        validate_json_schema=validate_json_schema,
        load_schema=load_schema,
    )

    validate_json_schema(state_out, schema)
