# tests/helpers/solver_step4_output_schema.py

# Step 4 expected top-level fields in SolverState.
# These must be a subset of the final solver_output_schema.json keys.

EXPECTED_STEP4_SCHEMA = [
    "config",
    "grid",
    "fields",
    "mask",
    "constants",
    "boundary_conditions",

    # Extended fields created in Step 4
    "P_ext",
    "U_ext",
    "V_ext",
    "W_ext",

    # Diagnostics accumulated up to Step 4
    "step3_diagnostics",
    "step4_diagnostics",

    # PPE persists through all steps
    "ppe",

    # Step 4 sets this to False (Step 5 sets it to True)
    "ready_for_time_loop",
]
