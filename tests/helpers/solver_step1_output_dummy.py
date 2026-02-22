# tests/helpers/solver_step1_output_dummy.py

import numpy as np
from src.solver_state import SolverState

def make_step1_output_dummy(nx=4, ny=4, nz=4):
    """
    Step 1 Implementation: Initialization & Allocation.
    Ensures all masks are arrays (lists) to satisfy the contract.
    """
    state = SolverState()

    # 1. Grid Definition (Essential for Step 1)
    x_min, x_max = 0.0, 1.0
    y_min, y_max = 0.0, 1.0
    z_min, z_max = 0.0, 1.0
    
    state.grid = {
        "nx": nx, "ny": ny, "nz": nz,
        "dx": (x_max - x_min) / nx,
        "dy": (y_max - y_min) / ny,
        "dz": (z_max - z_min) / nz,
        "x_min": x_min, "x_max": x_max,
        "y_min": y_min, "y_max": y_max,
        "z_min": z_min, "z_max": z_max,
        "total_cells": nx * ny * nz
    }

    # 2. Field Allocation (Step 1 Output)
    state.fields = {
        "P": np.zeros((nx, ny, nz)),
        "U": np.zeros((nx + 1, ny, nz)),
        "V": np.zeros((nx, ny + 1, nz)),
        "W": np.zeros((nx, ny, nz + 1)),
    }

    # 3. MASKING (THE FIX FOR THE ERROR)
    # The error says "False is not of type 'array'". 
    # We must overwrite the default Boolean with a List.
    mask_shape = (nx, ny, nz)
    fluid_mask = np.ones(mask_shape, dtype=int).tolist() # All fluid for Step 1
    solid_mask = np.zeros(mask_shape, dtype=int).tolist()
    
    state.mask = fluid_mask
    state.is_fluid = fluid_mask
    state.is_solid = solid_mask
    state.is_boundary_cell = solid_mask

    # 4. Department Initialization
    # PPE dimension must be present to pass property integrity tests
    state.ppe = {"dimension": nx * ny * nz}
    
    state.config = {"solver_type": "projection", "precision": "float64"}
    state.constants = {"nu": 0.01, "dt": 0.001}
    state.history = {
        "times": [], "divergence_norms": [], "energy_history": [],
        "max_velocity_history": [], "ppe_iterations_history": []
    }
    state.health = {"status": "initialized"}

    return state