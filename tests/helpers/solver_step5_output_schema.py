# tests/helpers/solver_step5_output_schema.py

# Step 5 expected top-level fields in SolverState.
# These must be a subset of the final solver_output_schema.json keys.

EXPECTED_STEP5_SCHEMA = [
    # Optional metadata included in final schema
    "step_index",

    # Core solver state
    "config",
    "constants",
    "grid",
    "mask",
    "fields",
    "boundary_conditions",

    # Extended fields (from Step 4)
    "P_ext",
    "U_ext",
    "V_ext",
    "W_ext",

    # PPE and health
    "ppe",
    "health",

    # Step 5 structured outputs
    "step5_outputs",
    "final_health",

    # Step 5 sets this to True
    "ready_for_time_loop",
]
