# tests/helpers/solver_step1_output_schema.py

EXPECTED_STEP1_SCHEMA = {
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
    "operators": dict,   # empty at Step 1
    "ppe": dict,         # empty at Step 1
    "health": dict,      # empty at Step 1
    "history": dict,     # empty at Step 1
}
