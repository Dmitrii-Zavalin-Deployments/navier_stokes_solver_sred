# tests/helpers/solver_step4_output_schema.py

EXPECTED_STEP4_SCHEMA = {
    "type": "object",
    "required": [
        "config",
        "grid",
        "fields",

        # Final schema requires mask, not is_fluid/is_boundary_cell
        "mask",
        "constants",
        "boundary_conditions",

        # Step 4 additions: extended fields
        "P_ext",
        "U_ext",
        "V_ext",
        "W_ext",

        # Step 4 must set this to False (Step 5 will set it to True)
        "ready_for_time_loop",
    ],
}
