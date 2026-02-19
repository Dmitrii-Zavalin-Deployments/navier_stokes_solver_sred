# tests/helpers/solver_output_schema_dummy.py

"""
solver_output_schema_dummy.py

Canonical JSON‑safe dummy final output state that fully satisfies
solver_output_schema.json. Used for schema validation tests and
end‑to‑end contract tests.

Updated to comply with Phase C, Rule 7 (Scale Guard):
- Sparse matrices are represented as metadata dictionaries.
"""

def solver_output_schema_dummy():
    nx, ny, nz = 2, 2, 2

    # Helper to build nested lists (JSON-safe equivalent of ndarrays)
    def zeros(shape, value=0.0):
        if len(shape) == 1:
            return [value for _ in range(shape[0])]
        return [zeros(shape[1:], value) for _ in range(shape[0])]

    def ints(shape, value=0):
        if len(shape) == 1:
            return [value for _ in range(shape[0])]
        return [ints(shape[1:], value) for _ in range(shape[0])]

    # Dimensions for metadata
    dof_p = nx * ny * nz
    total_vel_dof = ((nx+1)*ny*nz) + (nx*(ny+1)*nz) + (nx*ny*(nz+1))

    return {
        "config": {
            "grid": {"nx": nx, "ny": ny, "nz": nz},
            "simulation_parameters": {
                "time_step": 0.1,
                "total_time": 0.2,
                "output_interval": 1,
            },
            "fluid_properties": {"density": 1.0, "viscosity": 1.0},
            "boundary_conditions": {},
            "initial_conditions": {"velocity": [0, 0, 0], "pressure": 0.0},
        },

        "grid": {
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "dx": 1.0,
            "dy": 1.0,
            "dz": 1.0,
        },

        "fields": {
            "P": zeros((nx, ny, nz)),
            "U": zeros((nx + 1, ny, nz)), # Staggered shape
            "V": zeros((nx, ny + 1, nz)),
            "W": zeros((nx, ny, nz + 1)),
        },

        "mask": ints((nx, ny, nz), value=1),
        "is_fluid": ints((nx, ny, nz), value=1),
        "is_boundary_cell": ints((nx, ny, nz), value=0),
        "is_solid": ints((nx, ny, nz), value=0),

        "constants": {
            "rho": 1.0,
            "mu": 1.0,
            "dt": 0.1,
            "dx": 1.0,
            "dy": 1.0,
            "dz": 1.0,
        },

        "boundary_conditions": {},

        "health": {
            "divergence_norm": 0.0,
            "max_velocity": 0.0,
            "cfl": 0.0,
        },

        # Scale Guard: Sparse matrices as metadata dicts
        "operators": {
            "laplacian": {"type": "csr", "shape": [dof_p, dof_p], "nnz": 0},
            "divergence": {"type": "csr", "shape": [dof_p, total_vel_dof], "nnz": 0},
            "gradient": {"type": "csr", "shape": [total_vel_dof, dof_p], "nnz": 0},
        },

        "ppe": {
            "solver_type": "sparse_cg",
            "A": {"type": "csr", "shape": [dof_p, dof_p], "nnz": 0},
            "tolerance": 1e-6,
            "max_iterations": 1000,
            "ppe_is_singular": True,
            "rhs_norm": 0.0,
        },

        "step3_diagnostics": {
            "divergence_norm": 0.0,
            "max_velocity": 0.0,
        },

        # Extended fields (padded for BCs)
        "P_ext": zeros((nx+2, ny+2, nz+2)),
        "U_ext": zeros((nx+2, ny+2, nz+2)),
        "V_ext": zeros((nx+2, ny+2, nz+2)),
        "W_ext": zeros((nx+2, ny+2, nz+2)),

        "step4_diagnostics": {
            "total_fluid_cells": nx * ny * nz,
            "post_bc_max_velocity": 0.0,
            "post_bc_divergence_norm": 0.0,
            "bc_violation_count": 0,
        },

        "step5_outputs": {
            "final_time": 0.2,
            "total_steps_taken": 2,
        },

        "ready_for_time_loop": True,
    }