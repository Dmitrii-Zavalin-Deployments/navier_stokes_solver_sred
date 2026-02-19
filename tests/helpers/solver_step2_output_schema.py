# tests/helpers/solver_step2_output_schema.py

"""
EXPECTED_STEP2_SCHEMA defines the structural contract for the SolverState 
after Step 2 (Operators & PPE Setup).

This schema is 'Live-Aware': it accepts both SciPy sparse objects (for unit tests)
and metadata dictionaries (for JSON-safe validation).
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
    # Scale Guard: Accepts dict (serialized) or object (live SciPy CSR)
    "operators": {
        "laplacian": (dict, object),
        "divergence": (dict, object),
        "gradient": (dict, object),
    },
    
    # PPE contains solver settings and the system matrix A metadata/object
    "ppe": {
        "solver_type": str,
        "A": (dict, object),
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