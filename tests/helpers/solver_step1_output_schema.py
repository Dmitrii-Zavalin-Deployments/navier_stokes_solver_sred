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

    # Step 1 parses BCs but does not expand them into extended fields yet
    "boundary_conditions": (type(None), object),

    # Step 1 initializes these as empty containers
    "operators": dict,   # empty at Step 1
    "ppe": dict,         # empty at Step 1
    "health": dict,      # empty at Step 1

    # ❌ Removed: Step 1 no longer initializes history
    # "history": dict,
}
