# tests/helpers/solver_step1_output_dummy.py

import numpy as np
from src.solver_state import SolverState

def make_step1_output_dummy(nx=4, ny=4, nz=4):
    """
    High-Fidelity Step 1 Dummy using production SolverState.
    Dynamically scales geometry and populates all mandatory history keys.
    """
    state = SolverState()

    # --- DYNAMIC GEOMETRY DERIVATION ---
    # We fix the domain at [0, 1] and derive spacing to satisfy: 
    # dx = (x_max - x_min) / nx. This fixes the 0.25 vs 0.02 mismatch.
    x_min, x_max = 0.0, 1.0
    y_min, y_max = 0.0, 1.0
    z_min, z_max = 0.0, 1.0
    
    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny
    dz = (z_max - z_min) / nz

    state.grid = {
        "nx": nx, "ny": ny, "nz": nz,
        "dx": dx, "dy": dy, "dz": dz,
        "x_min": x_min, "x_max": x_max,
        "y_min": y_min, "y_max": y_max,
        "z_min": z_min, "z_max": z_max,
        "total_cells": nx * ny * nz
    }

    # --- MANDATORY DEPARTMENTS ---
    state.constants = {"nu": 0.001, "dt": 0.01}
    state.fluid_properties = {"density": 1000.0, "viscosity": 0.001}
    state.config = {"solver_type": "projection", "precision": "float64"}
    state.boundary_conditions = {"type": "lid_driven_cavity"}

    # --- FIELD ALLOCATION ---
    state.fields = {
        "P": np.zeros((nx, ny, nz)),
        "U": np.zeros((nx + 1, ny, nz)),
        "V": np.zeros((nx, ny + 1, nz)),
        "W": np.zeros((nx, ny, nz + 1)),
    }

    # Extended Fields (Ghost cells N+2)
    ext_shape = (nx + 2, ny + 2, nz + 2)
    state.P_ext = np.zeros(ext_shape)
    state.U_ext = np.zeros(ext_shape)
    state.V_ext = np.zeros(ext_shape)
    state.W_ext = np.zeros(ext_shape)

    # --- MASKING & HISTORY (Fixes KeyErrors) ---
    state.mask = np.ones((nx, ny, nz), dtype=int).tolist()
    state.ppe = {"dimension": nx * ny * nz}
    
    # These specific keys are required by the production to_json_safe()
    state.history = {
        "times": [],
        "divergence_norms": [],
        "max_velocity_history": [],
        "ppe_iterations_history": [],
        "energy_history": []
    }
    
    state.health = {"status": "initialized"}

    return state