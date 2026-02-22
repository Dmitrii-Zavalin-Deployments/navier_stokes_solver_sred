# tests/helpers/solver_step1_output_dummy.py

import numpy as np
# Import the actual production class
from src.solver_state import SolverState

def make_step1_output_dummy(nx=4, ny=4, nz=4):
    """
    High-Fidelity Step 1 Dummy using the production SolverState class.
    Populates mandatory fields to satisfy both Physics Theory and Schema Contracts.
    """
    # Initialize the real production object
    state = SolverState()

    # --- DYNAMIC GEOMETRY ---
    # Derived to satisfy: dx = (x_max - x_min) / nx
    dx, dy, dz = 0.25, 0.25, 0.25
    
    state.grid = {
        "nx": nx, "ny": ny, "nz": nz,
        "dx": dx, "dy": dy, "dz": dz,
        "x_min": 0.0, "x_max": nx * dx,
        "y_min": 0.0, "y_max": ny * dy,
        "z_min": 0.0, "z_max": nz * dz,
        "total_cells": nx * ny * nz
    }

    # --- PHYSICS & CONFIG ---
    state.constants = {"nu": 0.001, "dt": 0.01}
    state.fluid_properties = {"density": 1000.0, "viscosity": 0.001}
    state.config = {"solver_type": "projection", "precision": "float64"}
    state.boundary_conditions = {"type": "lid_driven_cavity"}

    # --- FIELD ALLOCATION ---
    # Primary staggered fields
    state.fields = {
        "P": np.zeros((nx, ny, nz)),
        "U": np.zeros((nx + 1, ny, nz)),
        "V": np.zeros((nx, ny + 1, nz)),
        "W": np.zeros((nx, ny, nz + 1)),
    }

    # Extended fields (Ghost cells N+2) - required by Step 5 schema
    ext_shape = (nx + 2, ny + 2, nz + 2)
    state.P_ext = np.zeros(ext_shape)
    state.U_ext = np.zeros(ext_shape)
    state.V_ext = np.zeros(ext_shape)
    state.W_ext = np.zeros(ext_shape)

    # --- MASKING & DIAGNOSTICS ---
    state.mask = np.ones((nx, ny, nz), dtype=int).tolist()
    state.ppe = {"dimension": nx * ny * nz}
    state.history = {
        "times": [], 
        "divergence_norms": [],
        "energy_history": []
    }
    state.health = {"status": "initialized"}

    return state