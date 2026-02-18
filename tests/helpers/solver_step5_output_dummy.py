# tests/helpers/solver_step5_output_dummy.py

import numpy as np

def make_step5_output_dummy():
    """
    A full, production-style Step 5 output dummy.
    Represents a fully evolved SolverState after the entire simulation pipeline.
    """

    nx, ny, nz = 4, 3, 2

    dummy = {
        "time": 0.12,
        "step_index": 6,

        "config": {
            "domain": {"nx": nx, "ny": ny, "nz": nz},
            "total_time": 0.12,
            "max_steps": 100,
            "output_interval": 2,
        },

        "constants": {
            "dt": 0.02,
            "rho": 1.0,
            "nu": 0.1,
        },

        "fields": {
            "P": np.zeros((nx, ny, nz)).tolist(),
            "U": np.zeros((nx + 1, ny, nz)).tolist(),
            "V": np.zeros((nx, ny + 1, nz)).tolist(),
            "W": np.zeros((nx, ny, nz + 1)).tolist(),
        },

        "P_ext": np.zeros((nx + 2, ny + 2, nz + 2)).tolist(),
        "U_ext": np.zeros((nx + 3, ny + 2, nz + 2)).tolist(),
        "V_ext": np.zeros((nx + 2, ny + 3, nz + 2)).tolist(),
        "W_ext": np.zeros((nx + 2, ny + 2, nz + 3)).tolist(),

        "ppe": {
            "iterations": 7,
            "residual_norm": 0.0004,
        },

        "health": {
            "post_correction_divergence_norm": 0.0003,
            "max_velocity_magnitude": 0.8,
            "cfl_advection_estimate": 0.4,
        },

        # ---------------------------------------------------------
        # Step 5 structured outputs (replaces old 'history')
        # ---------------------------------------------------------
        "step5_outputs": {
            "times": [0.0, 0.02, 0.04, 0.06, 0.08, 0.10],
            "steps": [0, 1, 2, 3, 4, 5],
            "divergence_norms": [0.01, 0.005, 0.002, 0.001, 0.0006, 0.0003],
            "max_velocity_history": [0.2, 0.3, 0.4, 0.55, 0.7, 0.8],
            "cfl_values": [0.1, 0.15, 0.2, 0.25, 0.32, 0.4],
            "ppe_iterations": [5, 5, 6, 6, 7, 7],
            "output_file_pairs": [
                {"time": 0.0, "step": 0, "json": None, "vti": None},
                {"time": 0.04, "step": 2, "json": None, "vti": None},
                {"time": 0.08, "step": 4, "json": None, "vti": None},
            ],
        },

        "final_health": {
            "final_time": 0.12,
            "total_steps_taken": 6,
            "final_divergence_norm": 0.0003,
            "final_max_velocity": 0.8,
            "average_ppe_iterations": 6.0,
            "max_cfl_encountered": 0.4,
            "simulation_success": True,
            "termination_reason": "normal",
        },

        # Step 5 must set this to True
        "ready_for_time_loop": True,
    }

    return dummy
