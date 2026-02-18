# tests/helpers/solver_step4_output_schema.py

step4_output_schema = {
    "type": "object",
    "required": [
        "config",
        "grid",
        "fields",

        "is_fluid",
        "is_boundary_cell",
        "mask",
        "constants",
        "boundary_conditions",

        # Step 2 carry‑overs
        "operators",
        "ppe",
        "health",

        # Step 4 additions: extended fields
        "P_ext",
        "U_ext",
        "V_ext",
        "W_ext",

        # Step 4 diagnostics block
        "step4_diagnostics",

        # Step 4 must set this to False (Step 5 will set it to True)
        "ready_for_time_loop",
    ],
}
