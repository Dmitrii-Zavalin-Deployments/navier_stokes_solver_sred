# tests/step2/test_orchestrate_step2_ppe_structure.py

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


def test_orchestrate_step2_ppe_structure():
    # Create a fully Step‑1‑schema‑compliant dummy state
    state = make_minimal_step1_state()

    # Run orchestrator
    result = orchestrate_step2(state)

    # PPE block (schema requires STRINGS, not callables)
    ppe = result["ppe"]

    assert "rhs_builder" in ppe
    assert isinstance(ppe["rhs_builder"], str)

    assert "solver_type" in ppe
    assert isinstance(ppe["solver_type"], str)

    assert "tolerance" in ppe
    assert isinstance(ppe["tolerance"], (int, float))

    assert "max_iterations" in ppe
    assert isinstance(ppe["max_iterations"], int)

    assert "ppe_is_singular" in ppe
    assert isinstance(ppe["ppe_is_singular"], bool)
