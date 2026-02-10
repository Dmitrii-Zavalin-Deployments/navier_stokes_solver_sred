# tests/step2/test_orchestrate_step2_schema_fields.py

import numpy as np
from src.step2.orchestrate_step2 import orchestrate_step2


def make_minimal_step1_state():
    return {
        "grid": {
            "nx": 4, "ny": 4, "nz": 4,
            "dx": 1.0, "dy": 1.0, "dz": 1.0,
        },
        "config": {
            "fluid": {"density": 1.0, "viscosity": 0.1},
            "simulation": {"dt": 0.1, "advection_scheme": "central"},
        },
        "fields": {
            "U": np.zeros((5, 4, 4)).tolist(),
            "V": np.zeros((4, 5, 4)).tolist(),
            "W": np.zeros((4, 4, 5)).tolist(),
            "P": np.zeros((4, 4, 4)).tolist(),
        },
        "mask_3d": np.ones((4, 4, 4), int).tolist(),
        "boundary_table_list": [],
    }


def test_orchestrate_step2_schema_fields():
    # Create a fully Step‑1‑schema‑compliant dummy state
    state = make_minimal_step1_state()

    # Run orchestrator
    result = orchestrate_step2(state)

    required = [
        "constants",
        "mask_semantics",
        "fluid_mask",
        "divergence",
        "pressure_gradients",
        "laplacians",
        "advection",
        "ppe_structure",
        "health",
        "meta",
    ]

    for key in required:
        assert key in result, f"Missing required field: {key}"
