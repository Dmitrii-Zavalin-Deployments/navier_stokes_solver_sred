# tests/helpers/solver_step2_output_schema.py

"""
Schema describing the full Step 2 SolverState structure.

This schema is used to validate:
- the Step 2 dummy
- the Step 2 output of the real orchestrator
- compatibility with the final solver_output_schema.json
"""

solver_step2_output_schema = {
    "grid": {
        "nx": int,
        "ny": int,
        "nz": int,
        "dx": float,
        "dy": float,
        "dz": float,
    },

    "config": {
        "dt": float,
        "advection_scheme": str,
    },

    "constants": {
        "rho": float,
        "inv_dx": float,
        "inv_dy": float,
        "inv_dz": float,
        "inv_dx2": float,
        "inv_dy2": float,
        "inv_dz2": float,
    },

    "mask": "ndarray",
    "is_fluid": "ndarray",
    "is_boundary_cell": "ndarray",

    "fields": {
        "P": "ndarray",
        "U": "ndarray",
        "V": "ndarray",
        "W": "ndarray",
    },

    "operators": {
        "divergence": "callable",
        "grad_x": "callable",
        "grad_y": "callable",
        "grad_z": "callable",
        "lap_u": "callable",
        "lap_v": "callable",
        "lap_w": "callable",
        "adv_u": "callable",
        "adv_v": "callable",
        "adv_w": "callable",
    },

    "ppe": {
        "rhs_builder": "callable",
        "ppe_is_singular": bool,
    },

    "health": {
        "divergence_norm": float,
        "max_velocity": float,
        "cfl": float,
    },

    "boundary_conditions": dict,
}
