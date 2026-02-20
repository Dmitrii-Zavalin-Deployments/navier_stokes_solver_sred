# tests/helpers/solver_step3_output_schema.py

"""
EXPECTED_STEP3_SCHEMA defines the structural contract for the SolverState 
after Step 3 (Projection & Correction).

This schema is explicit about the health metrics and PPE metadata 
required for post-simulation scientific analysis.
"""

EXPECTED_STEP3_SCHEMA = {
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
    "is_solid": (type(None), "ndarray"), 

    "constants": dict,
    "boundary_conditions": (type(None), dict, list, object),

    # Step 2 & 3: Operators and Pressure Solver Metadata
    "operators": dict,
    "ppe": {
        "solver_type": str,
        "A": (dict, object),
        "tolerance": float,
        "max_iterations": int,
        "ppe_is_singular": bool,
        "rhs_norm": float,
        "iterations": int,    # Added/Updated in Step 3 solve
        "converged": bool,    # Added/Updated in Step 3 solve
    },

    # Step 3 Health: Accumulated metrics (Step 2 Initial + Step 3 Post-Correction)
    "health": {
        "divergence_norm": float,
        "max_velocity": float,
        "cfl": float,
        "post_correction_divergence_norm": float,
        "max_velocity_magnitude": float,
        "cfl_advection_estimate": float,
    },

    # Simulation Progress (Official keys for analysis and restarts)
    "ready_for_time_loop": bool,
    "iteration": int,
    "time": float,
}