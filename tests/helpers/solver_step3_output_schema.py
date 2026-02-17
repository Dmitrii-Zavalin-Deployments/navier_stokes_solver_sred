# tests/helpers/solver_step3_output_schema.py

EXPECTED_STEP3_SCHEMA = {
    "config": dict,
    "grid": dict,
    "fields": {
        "U": "ndarray",
        "V": "ndarray",
        "W": "ndarray",
        "P": "ndarray",
    },
    "mask": "ndarray",
    "is_fluid": "ndarray",
    "is_boundary_cell": "ndarray",
    "constants": dict,
    "boundary_conditions": (type(None), object),
    "operators": dict,
    "ppe": dict,
    "health": {
        "post_correction_divergence_norm": float,
        "max_velocity_magnitude": float,
        "cfl_advection_estimate": float,
    },
    "history": {
        "times": list,
        "divergence_norms": list,
        "max_velocity_history": list,
        "ppe_iterations_history": list,
        "energy_history": list,
    },
}
