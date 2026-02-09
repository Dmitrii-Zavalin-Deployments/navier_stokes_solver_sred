# tests/step2/test_orchestrate_step2_operators_present.py

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


def test_orchestrate_step2_operators_present():
    # Create a fully Step‑1‑schema‑compliant dummy state
    state = make_minimal_step1_state()

    # Run orchestrator
    result = orchestrate_step2(state)

    # Operators block (schema requires STRINGS)
    ops = result["operators"]

    expected = [
        "divergence",
        "gradient_p_x",
        "gradient_p_y",
        "gradient_p_z",
        "laplacian_u",
        "laplacian_v",
        "laplacian_w",
        "advection_u",
        "advection_v",
        "advection_w",
    ]

    for name in expected:
        assert name in ops, f"Missing operator: {name}"
        assert isinstance(ops[name], str), f"Operator {name} must be a string"
