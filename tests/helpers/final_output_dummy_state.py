"""
final_output_dummy_state.py

Provides a canonical JSON‑safe dummy final output state that fully satisfies
final_output_schema.json. Used for schema validation tests and end‑to‑end
contract tests.
"""

def final_output_dummy():
    # Small test grid
    nx, ny, nz = 2, 2, 2

    # Helper to build 3D arrays as nested lists
    def zeros(shape):
        if len(shape) == 1:
            return [0.0] * shape[0]
        return [zeros(shape[1:]) for _ in range(shape[0])]

    return {
        "config": {
            "domain": {"nx": nx, "ny": ny, "nz": nz},
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
            "U": zeros((nx+1, ny, nz)),
            "V": zeros((nx, ny+1, nz)),
            "W": zeros((nx, ny, nz+1)),
        },

        "mask": zeros((nx, ny, nz)),
        "is_fluid": [[ [True]*nz for _ in range(ny) ] for _ in range(nx)],
        "is_boundary_cell": [[ [False]*nz for _ in range(ny) ] for _ in range(nx)],

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
            "post_correction_divergence_norm": 0.0,
            "max_velocity_magnitude": 0.0,
            "cfl_advection_estimate": 0.0,
        },

        "operators": {},  # runtime-only, empty in JSON-safe output

        "ppe": {
            "singularity_detected": False,
            "rhs_norm": 0.0,
        },

        "step3_diagnostics": {
            "divergence_norm": 0.0,
            "max_velocity": 0.0,
        },

        # Extended fields with ghost layers
        "P_ext": zeros((nx+2, ny+2, nz+2)),
        "U_ext": zeros((nx+3, ny+2, nz+2)),
        "V_ext": zeros((nx,   ny+3, nz+2)),
        "W_ext": zeros((nx,   ny,   nz+3)),

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
