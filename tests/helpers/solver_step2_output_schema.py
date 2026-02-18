# tests/helpers/solver_step2_output_schema.py

EXPECTED_STEP2_SCHEMA = {
    "config": dict,
    "grid": dict,

    "fields": {
        "P": "ndarray",
        "U": "ndarray",
        "V": "ndarray",
        "W": "ndarray",
    },

    "mask": "ndarray",
    "is_fluid": "ndarray",
    "is_boundary_cell": "ndarray",

    "constants": dict,

    # Step 2 does not modify BCs structurally, but they remain present
    "boundary_conditions": (type(None), object),

    # Step 2 additions / updates
    "operators": dict,   # gradient/divergence/laplacian operators
    "ppe": dict,         # PPE structure (matrices, RHS placeholders)
    "health": dict,      # updated health metrics after Step 2

    # ❌ Removed: Step 2 does NOT output history
    # "history": dict,
}
