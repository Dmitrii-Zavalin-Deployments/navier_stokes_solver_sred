# tests/step4/test_step4_diagnostics.py

import json
from pathlib import Path

from src.step4.orchestrate_step4 import orchestrate_step4


def load_schema(path: str):
    with open(path, "r") as f:
        return json.load(f)


def validate_json_schema(instance, schema):
    from jsonschema import validate
    validate(instance=instance, schema=schema)


def make_minimal_step3_state():
    """
    Minimal valid Step‑3 output that Step‑4 can accept.
    """
    return {
        "config": {
            "domain": {"nx": 2, "ny": 2, "nz": 2},
            "forces": {"gravity": [0.0, 0.0, 0.0]},
            "initial_conditions": {"initial_velocity": [0.0, 0.0, 0.0]},
            "boundary_conditions": [],
        },
        "mask": [[[1, 1], [1, 1]]],
        "is_fluid": [[[True, True], [True, True]]],
        "is_boundary_cell": [[[False, False], [False, False]]],
        "fields": {
            "P": [[[0.0, 0.0], [0.0, 0.0]]],
            "U": [[[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]],  # include a non-zero velocity
            "V": [[[0.0, 0.0], [0.0, 0.0]]],
            "W": [[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]],
        },
        "bcs": [],
        "constants": {},
        "operators": {},
        "ppe": {"ppe_is_singular": False},
        "health": {
            "post_correction_divergence_norm": 0.123,
            "max_velocity_magnitude": 1.0,
            "cfl_advection_estimate": 0.0,
        },
        "advection_meta": None,
        "history": {},
    }


def test_step4_diagnostics_values():
    """
    Contract test:
    Step‑4 MUST compute diagnostics correctly.

    Requirements:
    - total_fluid_cells correct
    - max velocity computed
    - divergence norm propagated
    - bc_violation_count computed
    """

    state_in = make_minimal_step3_state()

    state_out = orchestrate_step4(
        state_in,
        validate_json_schema=validate_json_schema,
        load_schema=load_schema,
    )

    assert "diagnostics" in state_out, "Step‑4 output must contain 'diagnostics'"
    diag = state_out["diagnostics"]

    # ---------------------------------------------------------
    # total_fluid_cells
    # ---------------------------------------------------------
    expected_fluid = sum(
        1 for k in state_in["mask"]
          for j in k
          for v in j
          if v == 1
    )
    assert diag["total_fluid_cells"] == expected_fluid, \
        "total_fluid_cells must match mask"

    # ---------------------------------------------------------
    # post_bc_max_velocity
    # ---------------------------------------------------------
    # U field contains a 1.0, so max velocity must be >= 1.0
    assert diag["post_bc_max_velocity"] >= 1.0, \
        "post_bc_max_velocity must reflect max velocity in extended fields"

    # ---------------------------------------------------------
    # post_bc_divergence_norm
    # ---------------------------------------------------------
    expected_div = state_in["health"]["post_correction_divergence_norm"]
    assert abs(diag["post_bc_divergence_norm"] - expected_div) < 1e-12, \
        "post_bc_divergence_norm must match health block"

    # ---------------------------------------------------------
    # bc_violation_count
    # ---------------------------------------------------------
    # With no BCs applied, all 6 faces default to applied=False → count = 6
    assert isinstance(diag["bc_violation_count"], int), \
        "bc_violation_count must be an integer"
    assert diag["bc_violation_count"] == 6, \
        "bc_violation_count must count all 6 faces when no BCs are applied"
