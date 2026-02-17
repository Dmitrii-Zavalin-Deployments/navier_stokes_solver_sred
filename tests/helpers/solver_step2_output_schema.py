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
    "boundary_conditions": (type(None), object),

    # Step 2 additions
    "operators": dict,
    "ppe": dict,
    "health": dict,

    # Step 2 still has empty history
    "history": dict,
}
