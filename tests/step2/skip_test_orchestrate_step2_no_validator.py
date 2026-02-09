# tests/step2/test_orchestrate_step2_no_validator.py

import numpy as np
import src.step2.orchestrate_step2 as o


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


def test_orchestrate_step2_no_validator():
    # Simulate missing validator
    o.validate_json_schema = None
    o.load_schema = None

    # Create a fully Step‑1‑schema‑compliant dummy state
    state = make_minimal_step1_state()

    # Run orchestrator
    result = o.orchestrate_step2(state)

    # NEW: Step 2 stores constants under lowercase "constants"
    assert "constants" in result
