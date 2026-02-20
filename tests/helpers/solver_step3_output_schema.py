# tests/helpers/solver_step3_output_schema.py

"""
EXPECTED_STEP3_SCHEMA defines the structural contract for the SolverState 
after Step 3 (Projection & Correction).

Note: While 'history' exists in the SolverState object for internal tracking,
it is excluded from this schema to keep the step-to-step validation focused 
on physical fields and solver convergence metadata.
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

    # Step 2 additions (Matrices and PPE settings)
    "operators": dict,
    "ppe": {
        "solver_type": str,
        "A": (dict, object),
        "tolerance": float,
        "max_iterations": int,
        "ppe_is_singular": bool,
        "rhs_norm": float,
        "iterations": int,    # Added in Step 3 solve
        "converged": bool,    # Added in Step 3 solve
    },

    # Step 3 Health: Contains both initial (Step 2) and post-solve metrics
    "health": {
        "divergence_norm": float,
        "max_velocity": float,
        "cfl": float,
        "post_correction_divergence_norm": float,
        "max_velocity_magnitude": float,
        "cfl_advection_estimate": float,
    },

    "ready_for_time_loop": bool,
    "iteration": int,
    "time": float,
}