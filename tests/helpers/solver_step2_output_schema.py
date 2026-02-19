# tests/helpers/solver_step2_output_schema.py

"""
EXPECTED_STEP2_SCHEMA defines the structural contract for the SolverState 
after Step 2 (Operators & PPE Setup).

Note: Because validation occurs on the result of SolverState.to_json_safe(),
sparse matrices appear as metadata dictionaries containing 'type', 'shape', and 'nnz'.
"""

EXPECTED_STEP2_SCHEMA = {
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

    # Step 2 maintains BCs but doesn't transform them into fields yet
    "boundary_conditions": (type(None), dict, list),

    # Step 2 additions / updates
    # These contain metadata dicts for Laplacian, Divergence, and Gradient matrices
    "operators": {
        "laplacian": dict,
        "divergence": dict,
        "gradient": dict,
    },
    
    # PPE contains solver settings and the system matrix A metadata
    "ppe": {
        "solver_type": str,
        "A": dict,
        "tolerance": float,
        "max_iterations": int,
        "ppe_is_singular": bool,
        "rhs_norm": float,
    },
    
    # Updated health metrics initialized in Step 2
    "health": {
        "divergence_norm": float,
        "max_velocity": float,
        "cfl": float,
    },

    "ready_for_time_loop": bool,
}