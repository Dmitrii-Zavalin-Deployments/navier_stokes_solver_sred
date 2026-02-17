# tests/helpers/solver_step4_output_schema.py

step4_output_schema = {
    "type": "object",
    "required": [
        "config",
        "grid",
        "fields",
        "is_fluid",
        "is_boundary_cell",
        "constants",
        "operators",
        "ppe",
        "health",
        "history",
        "P_ext",
        "U_ext",
        "V_ext",
        "W_ext",
        "step4_diagnostics",
        "ready_for_time_loop",
    ],
}
