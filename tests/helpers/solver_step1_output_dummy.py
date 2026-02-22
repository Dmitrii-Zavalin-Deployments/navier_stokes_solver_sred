# tests/helpers/solver_step1_output_dummy.py

import numpy as np
from src.solver_state import SolverState

def make_step1_output_dummy(nx=4, ny=4, nz=4):
    """
    Step 1 Dummy: The Physical Foundation.
    Follows the 'No Aliasing' rule:
    - Geometry metrics (dx, nx, bounds) live ONLY in .grid
    - Physics metrics (rho, mu, dt) live ONLY in .constants
    """
    # Initialize the 'Empty Slate'
    state = SolverState()

    # --- Step 1 Responsibility: Geometry Department ---
    # Define physical bounds
    x_min, x_max = 0.0, 1.0
    y_min, y_max = 0.0, 1.0
    z_min, z_max = 0.0, 1.0

    # Derive spacing: (max - min) / N
    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny
    dz = (z_max - z_min) / nz

    state.grid = {
        "nx": nx, "ny": ny, "nz": nz,
        "dx": dx, "dy": dy, "dz": dz,
        "x_min": x_min, "x_max": x_max,
        "y_min": y_min, "y_max": y_max,
        "z_min": z_min, "z_max": z_max,
        "total_cells": nx * ny * nz  # FIX: Satisfies Property Integrity tests
    }

    # --- Step 1 Responsibility: Constants (Physics Only) ---
    state.constants = {
        "rho": 1.0,
        "mu": 0.001,
        "dt": 0.01
    }

    # --- Step 1 Responsibility: Basic Staggered Fields ---
    # Staggered Grid Logic: One extra face node in the primary direction
    state.fields = {
        "P": np.zeros((nx, ny, nz)),
        "U": np.zeros((nx + 1, ny, nz)),
        "V": np.zeros((nx, ny + 1, nz)),
        "W": np.zeros((nx, ny, nz + 1)),
    }

    # --- Step 1 Responsibility: PPE Department Plan ---
    state.ppe = {
        "dimension": nx * ny * nz  # FIX: Required for Step 2/3 matrix scaling
    }

    # --- Step 1 Responsibility: Basic Masking ---
    state.mask = np.ones((nx, ny, nz), dtype=int)
    state.is_fluid = (state.mask == 1)
    state.is_solid = ~state.is_fluid
    state.is_boundary_cell = np.zeros((nx, ny, nz), dtype=bool)

    # Initialize history to avoid Step 3 attribute errors later
    state.history = {
        "times": [],
        "divergence_norms": [],
        "max_velocity_history": [],
        "ppe_iterations_history": [],
        "energy_history": [],
    }

    return state