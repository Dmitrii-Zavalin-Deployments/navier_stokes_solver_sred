# tests/step4/test_step4_orchestrator_happy_path.py

import numpy as np
from src.step4.orchestrate_step4 import orchestrate_step4


def make_minimal_step3_state(nx=2, ny=2, nz=2):
    return {
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


def test_step4_orchestrator_happy_path():
    state_in = make_minimal_step3_state()
    state_out = orchestrate_step4(state_in)

    # Extended fields exist
    assert "U_ext" in state_out
    assert "V_ext" in state_out
    assert "W_ext" in state_out
    assert "P_ext" in state_out

    # Diagnostics exist
    assert "diagnostics" in state_out

    # Ready for time loop
    assert state_out["ready_for_time_loop"] is True
