# tests/step4/test_step4_schema_output.py

import numpy as np
from src.step4.orchestrate_step4 import orchestrate_step4


def test_step4_schema_output(load_schema, validate_json_schema):
    nx = ny = nz = 2

    state = {
        "config": {
            "domain": {"nx": nx, "ny": ny, "nz": nz},
            "initial_conditions": {
                "initial_pressure": 0.0,
                "initial_velocity": [0.0, 0.0, 0.0],
            },
            "boundary_conditions": [],
        },
        "mask": np.ones((nx, ny, nz), dtype=int),
        "fields": {
            "P": np.zeros((nx, ny, nz)),
            "U": np.zeros((nx+1, ny, nz)),
            "V": np.zeros((nx, ny+1, nz)),
            "W": np.zeros((nx, ny, nz+1)),
        },
        "health": {"post_correction_divergence_norm": 0.0},
    }

    out = orchestrate_step4(state, validate_json_schema, load_schema)
    assert "diagnostics" in out
