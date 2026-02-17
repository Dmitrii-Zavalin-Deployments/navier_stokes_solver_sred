# tests/helpers/solver_step1_output_schema.py

"""
Internal schema describing the expected structure of a SolverState
immediately after Step 1 completes.

This is NOT an external JSON schema.
It is used only for internal validation of:
  • the Step 1 dummy
  • the relationship between Step 1 output and the final output schema
"""

STEP1_OUTPUT_SCHEMA = {
    "grid": {
        "nx": int,
        "ny": int,
        "nz": int,
        "dx": float,
        "dy": float,
        "dz": float,
    },

    "config": {
        "dt": float,
    },

    "constants": {
        "rho": float,
    },

    "mask": "ndarray",

    "fields": {
        "P": "ndarray",
        "U": "ndarray",
        "V": "ndarray",
        "W": "ndarray",
    },

    "boundary_conditions": dict,

    "health": dict,
}
