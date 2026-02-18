# tests/helpers/solver_step5_output_schema.py

step5_output_schema = {
    "type": "object",

    # Step 5 required fields must be a subset of final schema required fields
    "required": [
        "config",
        "constants",

        "fields",          # P, U, V, W
        "P_ext",
        "U_ext",
        "V_ext",
        "W_ext",

        "ppe",
        "health",

        # Step 5 structured outputs (replaces old 'history')
        "step5_outputs",

        # Step 5 must set this to True
        "ready_for_time_loop",
    ],

    "properties": {
        # Optional metadata (not required by final schema)
        "time": {"type": "number"},
        "step_index": {"type": "integer"},

        "config": {"type": "object"},
        "constants": {"type": "object"},

        "fields": {
            "type": "object",
            "required": ["P", "U", "V", "W"],
        },

        "P_ext": {"type": ["array", "null"]},
        "U_ext": {"type": ["array", "null"]},
        "V_ext": {"type": ["array", "null"]},
        "W_ext": {"type": ["array", "null"]},

        "ppe": {"type": "object"},
        "health": {"type": "object"},

        # Step 5 structured outputs
        "step5_outputs": {"type": "object"},

        # Optional (not required by final schema)
        "final_health": {"type": "object"},

        "ready_for_time_loop": {"type": "boolean"},
    },
}
