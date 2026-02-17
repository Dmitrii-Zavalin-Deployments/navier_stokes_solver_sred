# tests/helpers/solver_step5_output_schema.py

step5_output_schema = {
    "type": "object",
    "required": [
        "time",
        "step_index",
        "config",
        "constants",
        "fields",
        "P_ext",
        "U_ext",
        "V_ext",
        "W_ext",
        "ppe",
        "health",
        "history",
        "final_health",
    ],
    "properties": {
        "time": {"type": "number"},
        "step_index": {"type": "integer"},

        "config": {"type": "object"},
        "constants": {"type": "object"},

        "fields": {
            "type": "object",
            "required": ["P", "U", "V", "W"],
        },

        "P_ext": {"type": "array"},
        "U_ext": {"type": "array"},
        "V_ext": {"type": "array"},
        "W_ext": {"type": "array"},

        "ppe": {"type": "object"},
        "health": {"type": "object"},
        "history": {"type": "object"},
        "final_health": {"type": "object"},
    },
}
